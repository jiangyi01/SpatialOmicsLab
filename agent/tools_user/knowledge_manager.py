"""Knowledge storage, backup management, and modification safety for user MCP tools.

Provides three integrated capabilities:

Knowledge Storage:
  - save_knowledge()       : Save creation-time knowledge (README, API, signatures, tests)
  - read_knowledge()       : Read all stored knowledge for a tool
  - update_knowledge()     : Update specific knowledge files after modification
  - bootstrap_knowledge()  : Reconstruct partial knowledge from existing files
  - knowledge_exists()     : Check if knowledge directory exists
  - knowledge_info()       : Summary of knowledge state across all tools

Backup Management:
  - create_backup()        : Versioned pre-modification backup with SHA-256 checksums
  - rollback_to_backup()   : Atomic two-phase rollback from backup
  - list_backups()         : List all available backups for a tool

Modification Orchestration (three-phase API for STCoscientist):
  - begin_modification()   : Phase 1 — pre-flight checks + create backup
  - complete_modification(): Phase 2 — run 6-test suite on modified files
  - finalize_modification(): Phase 3a — record success in modification log
  (rollback_to_backup serves as Phase 3b on failure)

Safety model:
  1. TOOL_ID_RE regex (path traversal prevention)
  2. install_log lookup (only known tools)
  3. Status gate (only active tools can be modified)
  4. PROTECTED_FILES / PROTECTED_DIRS frozensets
  5. SHA-256 verification on rollback
  6. Atomic two-phase rollback (stage .rollback_tmp → os.replace)
  7. Advisory file lock (_knowledge_lock)
  8. Crash recovery (_repair_modification_state)

All JSON writes use atomic temp-file-then-rename pattern.
All operations serialized via fcntl advisory lock (.knowledge.lock).

NOTE: .knowledge/ is runtime state and is ignored by the root .gitignore rule
`tools_user/.knowledge/`. It is NOT covered by any blanket dotfile pattern -- that .gitignore has
none, it lists dot-paths individually -- so the rule has to name this directory. Pinned by
test/test_repo_hygiene.py::TestToolsUserRuntimeStateStaysOutOfTheHistory.
"""

from __future__ import annotations

import ast
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
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
KNOWLEDGE_DIR = TOOLS_USER_DIR / ".knowledge"
INSTALL_LOG = TOOLS_USER_DIR / "install_log.json"
MCP_CONFIG_USER = TOOLS_USER_DIR.parent / "MCP_server" / "mcp_config_user.yaml"
#: What ``MCP_CONFIG_USER`` was at import, so an assignment to it (a test, an embedding caller) is told
#: apart from the default by ``mcp_config_user_path``.
_DEFAULT_MCP_CONFIG_USER = MCP_CONFIG_USER


def mcp_config_user_path() -> Path:
    """The ``mcp_config_user.yaml`` this module reads and writes, resolved per call.

    ``SOG_MCP_USER_CONFIG`` moved every READER of the file (the merger, the resync pruner, the mtime
    watcher, the portal's settings view) and none of its writers: this module, ``trash_manager`` and
    ``broker.register_tool`` wrote ``MCP_CONFIG_USER`` regardless, so with the variable set a created
    tool was reported registered and never wired (hunt 2026-09-30, u14-mcp-wiring-7). Now the writers
    read the same override, and nothing changes when it is unset.

    An assignment to ``MCP_CONFIG_USER`` still wins over the variable: it is the more specific
    instruction, and it is the seam every test of these writers already uses.
    """
    if MCP_CONFIG_USER != _DEFAULT_MCP_CONFIG_USER:
        return Path(MCP_CONFIG_USER)
    from spatialomicsgym.mcp_user_config import user_config_override

    override = user_config_override()
    return override if override is not None else MCP_CONFIG_USER


# The live conda root, not the /opt/conda of the machine this was written on -- see the same
# constant in trash_manager.py. Stays a module-level Path so tests can monkeypatch the attribute.
CONDA_ENVS_DIR = Path(conda_envs_root())

# tool_id must be: lowercase letter, then up to 62 lowercase letters/digits/underscores
TOOL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

# Known file suffixes for user tool files
KNOWN_SUFFIXES = ("_worker.py", "_mcp_server.py", "_env.yaml", "_worker.R")

# Files that must NEVER be modified by knowledge/modification operations
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
        TOOLS_USER_DIR / ".trash",
    }
)

# Conda environments that must NEVER be modified. Single-sourced from sog_install.constants -- a local
# copy here once drifted (it lacked the pre-rename alias env, ``constants.LEGACY_ENV_ALIASES``'s
# value, which the setup side protects), so a mis-recorded ``env_name`` could route a
# modification's pip install into that shared env. Kept as a module attribute so tests can
# monkeypatch it.
PROTECTED_ENVS = _SETUP_PROTECTED_ENVS

# Maximum backup versions per tool (oldest pruned when exceeded)
MAX_BACKUP_VERSIONS = 10

# Schema versions for forward compatibility
KNOWLEDGE_SCHEMA_VERSION = 1
BACKUP_META_VERSION = 2

# Valid modification types
MODIFICATION_TYPES = frozenset(
    {
        "parameter_change",
        "pipeline_addition",
        "pipeline_removal",
        "function_swap",
        "output_change",
        "error_handling",
        "signature_change",
        "environment_change",
        "multi_file_change",
    }
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KnowledgeError(Exception):
    """Base exception for all knowledge/modification operations."""


class KnowledgeSafetyError(KnowledgeError):
    """Raised when a safety check fails (path/permission violation)."""


class KnowledgeNotFoundError(KnowledgeError):
    """Raised when the requested tool or knowledge does not exist."""


class KnowledgeStateError(KnowledgeError):
    """Raised when tool is in wrong state for the requested operation."""


class ModificationError(KnowledgeError):
    """Raised when a modification operation fails."""


class ModificationTestError(ModificationError):
    """Raised when post-modification tests fail."""


class RollbackError(ModificationError):
    """Raised when rollback itself fails (critical)."""


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


@contextmanager
def _knowledge_lock():
    """Advisory file lock to serialize all knowledge/modification operations.

    Uses LOCK_NB (non-blocking) so concurrent attempts get a clear error.
    Separate from trash_manager's .trash.lock — they protect different resources.
    """
    lock_path = TOOLS_USER_DIR / ".knowledge.lock"
    lock_fd = None
    try:
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            lock_fd.close()
            raise KnowledgeError("Another knowledge operation is in progress. Please wait and retry.") from exc
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


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_state_fingerprint(files_info: dict) -> str:
    """SHA-256 of sorted (filename, file_sha256) pairs — identifies identical file states."""
    parts = sorted((fname, info["sha256"]) for fname, info in files_info.items())
    combined = "|".join(f"{name}:{sha}" for name, sha in parts)
    return hashlib.sha256(combined.encode()).hexdigest()


def _get_latest_backup_fingerprint(tool_id: str) -> tuple:
    """Return (version, fingerprint) of the latest backup, or (None, None).

    Fast path: reads ``state_fingerprint`` from metadata (BACKUP_META_VERSION >= 2).
    Fallback: computes fingerprint from per-file SHA-256 hashes (backward compat).
    """
    bdir = _backups_dir(tool_id)
    if not bdir.exists():
        return (None, None)

    # Find latest version directory
    versions = []
    for d in bdir.iterdir():
        if d.is_dir():
            m = re.match(r"^v(\d{3})_", d.name)
            if m:
                versions.append((int(m.group(1)), d))
    if not versions:
        return (None, None)
    versions.sort(key=lambda x: x[0])
    latest_ver, latest_dir = versions[-1]

    meta_path = latest_dir / ".backup_meta.json"
    if not meta_path.exists():
        return (None, None)

    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return (None, None)

    # Fast path: state_fingerprint stored in metadata
    fp = meta.get("state_fingerprint")
    if fp:
        return (latest_ver, fp)

    # Fallback: compute from per-file SHA-256 hashes
    files_backed_up = meta.get("files_backed_up", {})
    if files_backed_up:
        fp = _compute_state_fingerprint(files_backed_up)
        return (latest_ver, fp)

    return (latest_ver, None)


# ---------------------------------------------------------------------------
# Utility helpers (disk size, symlink checks — mirrors trash_manager for decoupling)
# ---------------------------------------------------------------------------


def _dir_size_bytes(path: Path) -> int:
    """Total size of directory tree in bytes. Returns 0 if path doesn't exist."""
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _human_size(nbytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _check_symlink(path: Path) -> tuple[bool, str]:
    """Verify a symlink resolves to an existing, readable file.

    Returns (ok, detail).
    """
    if not path.exists() and not path.is_symlink():
        return False, f"not found: {path.name}"
    if path.is_symlink():
        target = path.resolve()
        if not target.exists():
            return False, f"broken symlink: {path.name} -> {target}"
        if not os.access(target, os.R_OK):
            return False, f"symlink target not readable: {target}"
        return True, f"OK ({path.name} -> {target.name})"
    if not os.access(path, os.R_OK):
        return False, f"not readable: {path.name}"
    return True, f"OK ({path.name})"


# ---------------------------------------------------------------------------
# Install log helpers (mirror of trash_manager, independent for decoupling)
# ---------------------------------------------------------------------------


def _log_key(entry: dict) -> tuple[str, str]:
    """What makes a registry row unique: its owner and its tool id.

    The id alone was the key, and with one operator that was right. With accounts it is not: two
    people who both create ``liana`` collapse into one row and the second silently replaces the
    first -- the registry then describes one tool and there are two on disk, which is a worse
    state than either having failed.

    Backward compatible by construction. No record written before owners existed carries one, so
    they all key on ``("", tool_id)`` and the dedup is byte-identical to what it was; the collapse
    stops the moment an owner appears, which is exactly when it starts to matter.
    """
    return (str(entry.get("owner") or ""), str(entry.get("tool_id") or ""))


def _read_install_log() -> list[dict]:
    """Read install_log.json with dedup by (owner, tool_id) -- last entry wins."""
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
    because the ids match.
    """
    for entry in _read_install_log():
        if entry.get("tool_id") != tool_id:
            continue
        if owner is not None and str(entry.get("owner") or "") != str(owner):
            continue
        return entry
    return None


def _install_log_unreadable() -> str | None:
    """Why install_log.json must not be rewritten from what we just read, or None if it may be.

    ``_read_install_log`` fails open -- a truncated, empty or non-list file reads as "no tools are
    installed" -- and that is the right answer for a reader. For a writer it is not: saving that
    answer back replaces the registry with ``[]`` and every created tool stops existing, from a
    file whose remaining bytes were the only record of them.
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

    Returns None when the entry was updated, or why it was not -- the ``_clear_modification_lock``
    shape. Callers that cannot act on the answer may ignore it; what they must not do is report
    the update as done.

    Matched on the SAME key ``_read_install_log`` dedups by. It used to match on ``tool_id``
    alone, which meant the file could hold two accounts' ``liana`` -- the whole point of the key
    -- and a write addressed at one of them would land on whichever came first. ``owner=None``
    keeps every existing caller's behaviour while there is one row (there is, on every box today)
    and refuses rather than guessing when there are two.
    """
    unreadable = _install_log_unreadable()
    if unreadable:
        logger.warning("Refusing to rewrite the tool registry: %s", unreadable)
        return (
            f"{unreadable}, so '{tool_id}' was left as it is rather than rewriting the registry "
            f"from a file that did not parse. Repair or remove that file to record the change."
        )
    entries = _read_install_log()
    rows = [
        e for e in entries if e.get("tool_id") == tool_id and (owner is None or str(e.get("owner") or "") == str(owner))
    ]
    if not rows:
        # Nothing matched, so the old code wrote the list back unchanged and returned as if the
        # update had landed. Say it did not: the fields are lost either way.
        return f"No install_log entry for '{tool_id}', so {sorted(updates)} was not recorded."
    if len(rows) > 1:
        owners = sorted(str(r.get("owner") or "") or "(no owner)" for r in rows)
        reason = (
            f"{len(rows)} accounts have a tool called '{tool_id}' ({', '.join(owners)}), so "
            f"recording {sorted(updates)} without naming an owner would have picked one of them "
            f"at random. Pass owner=."
        )
        logger.warning("Refusing an ambiguous registry update: %s", reason)
        return reason
    rows[0].update(updates)
    _write_install_log(entries)
    return None


def current_owner() -> str:
    """The account whose turn is running, or ``""`` when nobody is signed in.

    The env-backed creation playbook stamps this onto the registry row it writes, which is what
    makes the ``(owner, tool_id)`` key above do anything at all: until something records an owner,
    every row this tier produces keys on ``("", tool_id)`` and two accounts' tools still collapse
    into one.

    Read off the agent rather than passed in, for the reason ``execution.make_prompt_tool``
    refuses to take it as an argument: the front door stamps ``_turn_owner`` before the turn
    starts, so a value the model could type is a value the model could type someone else's name
    into. The playbook is model-written code in a shared REPL, so this is filing, not a wall --
    ``tools_user/declarative.py`` says the same thing about the other tier.

    ``""`` is the honest answer for a library or CLI run, and it is also the key every record
    written before accounts existed already has, so an unowned row keeps behaving as it did.
    Never raises: a registry row must not fail to be written because the agent module is absent.
    """
    try:
        from spatialomicsgym.agent.execution import _CURRENT_AGENT

        if not _CURRENT_AGENT:
            return ""
        return str(getattr(_CURRENT_AGENT[0], "_turn_owner", "") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_tool_id(tool_id: str) -> None:
    """Validate tool_id format. Raises KnowledgeSafetyError on invalid."""
    if not tool_id or not isinstance(tool_id, str):
        raise KnowledgeSafetyError("tool_id must be a non-empty string.")
    if not TOOL_ID_RE.match(tool_id):
        raise KnowledgeSafetyError(
            f"Invalid tool_id '{tool_id}'. Must match: lowercase letter followed by "
            f"up to 62 lowercase letters, digits, or underscores."
        )


def _require_active(tool_id: str) -> dict:
    """Validate tool exists and is active. Returns the log entry."""
    entry = _find_log_entry(tool_id)
    if entry is None:
        available = [e.get("tool_id") for e in _read_install_log() if e.get("status") == "active"]
        raise KnowledgeStateError(f"Tool '{tool_id}' not found. Available tools: {available}")
    status = entry.get("status", "")
    if status == "trashed":
        raise KnowledgeStateError(
            f"Tool '{tool_id}' is in trash. Restore it first with restore_tool('{tool_id}'), then modify."
        )
    if status != "active":
        raise KnowledgeStateError(f"Tool '{tool_id}' has unexpected status: '{status}'")
    return entry


def _discover_tool_files(tool_id: str) -> list[Path]:
    """Find ALL files belonging to a tool in tools_user/.

    Combines install_log files field + known suffix patterns.
    Skips symlinks and protected files.
    """
    found: set[Path] = set()
    entry = _find_log_entry(tool_id)
    if entry:
        for f in entry.get("files", []):
            p = TOOLS_USER_DIR / f
            if p.exists() and p.is_file() and not p.is_symlink():
                found.add(p)
    for suffix in KNOWN_SUFFIXES:
        p = TOOLS_USER_DIR / f"{tool_id}{suffix}"
        if p.exists() and p.is_file() and not p.is_symlink() and p not in PROTECTED_FILES:
            found.add(p)
    return sorted(found)


# ---------------------------------------------------------------------------
# MCP config helpers
# ---------------------------------------------------------------------------


def _read_mcp_config_user() -> dict:
    """Read mcp_config_user.yaml safely."""
    path = mcp_config_user_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _looks_like_wiring(value: object) -> bool:
    """Whether a top-level mapping is a server block the merger would serve.

    ``mcp_config_merger._looks_like_a_server_block``'s test, kept here so the reader below and the
    stray-removal above it cannot disagree about what they are looking at: whatever a lookup is
    willing to treat as this tool's wiring is exactly what a write of that wiring may supersede.
    A bookkeeping key, a note, or a half-written block with a command and no tools is neither.
    """
    return isinstance(value, dict) and "command" in value and isinstance(value.get("tools"), list)


def _strip_top_level_stray(data: dict, server_key: str) -> bool:
    """Drop a column-0 copy of ``server_key`` from an already-loaded config. True if one went.

    Called by the writers below once they have published the canonical nested block. Leaving the
    stray costs nothing while the tool is active -- the merger prefers nested
    (``{**stray, **user_servers}``) -- and everything at teardown: ``trash_manager`` removes the one
    location a lookup resolves to, so the survivor keeps a tool whose server file has just moved to
    ``.trash/`` advertised to the model.

    Only what ``_looks_like_wiring`` accepts is removed, and only under this tool's own key.
    """
    if _looks_like_wiring(data.get(server_key)):
        del data[server_key]
        return True
    return False


def _server_block(config: dict, server_key: str) -> dict:
    """Return the block that wires ``server_key``, wherever the agent actually wrote it.

    mcp_config_user.yaml is agent-authored, and a model that emits ``mcp_servers: {}`` and then
    appends ``user_<id>:`` at column 0 produces a file the agent still runs from:
    ``mcp_config_merger._recover_top_level_servers`` rescues the misplaced sibling precisely so the
    created tool stays callable. Indexing ``mcp_servers`` straight makes every check here disagree
    with the wiring the agent uses -- a callable tool's spatialomicsgym_name check reports the name
    absent when it is there one indent up, and the tool reads BROKEN. A correctly nested block
    always wins, so this can only ever find more.

    ``health_check_tools`` section 4b keeps reading the raw map on purpose: the file is still wrong
    and ``unwired_tools`` is what tells the agent to move the block.
    """
    servers = config.get("mcp_servers")
    if isinstance(servers, dict) and isinstance(servers.get(server_key), dict):
        return servers[server_key]
    # Same strictness as the merger's _looks_like_a_server_block: a bookkeeping key, or a
    # half-written block with a command and no tools, is not something to wire.
    block = config.get(server_key)
    return block if _looks_like_wiring(block) else {}


def _restore_mcp_config_entry(tool_id: str, config_block: dict) -> None:
    """Overwrite THIS tool's ``user_<tid>`` entry in mcp_config_user.yaml with ``config_block``,
    preserving sibling user servers.

    Used by rollback to undo the agent's yaml edit alongside the file restore: a rename/signature
    modification edits the yaml's ``spatialomicsgym_name``/param-schema next to the server file (mandated
    by the modify playbook), so reverting only the files would leave the config describing the NEW
    signature -> server-vs-config mismatch (call_tool fails). Prefers ruamel round-trip (sibling
    formatting preserved), PyYAML fallback when ruamel is absent; atomic tmp+rename either way.

    The entry goes back NESTED -- canonical, and where the merger and every lookup here look first --
    and a column-0 copy of the same key goes with it. ``create_backup`` snapshots the block through
    ``_server_block``, which finds a stray, so writing nested without ``_strip_top_level_stray``
    leaves the tool wired twice; ``trash_manager`` then removes the one location a lookup resolves
    to and the survivor keeps a trashed tool advertised. Both writes below strip in the same atomic
    dump as the nested write, so there is no window where the file has neither.
    """
    server_key = f"user_{tool_id}"
    config_path = mcp_config_user_path()
    try:
        from ruamel.yaml import YAML

        rt = YAML()
        rt.preserve_quotes = True
        data = rt.load(config_path.read_text()) if config_path.exists() else {}
        if not isinstance(data, dict):
            data = {}
        servers = data.get("mcp_servers")
        if not isinstance(servers, dict):
            servers = {}
            data["mcp_servers"] = servers
        servers[server_key] = config_block
        _strip_top_level_stray(data, server_key)
        tmp = config_path.with_suffix(".yaml.tmp")
        try:
            with open(tmp, "w") as f:
                rt.dump(data, f)
            os.replace(str(tmp), str(config_path))
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    except ImportError:
        config = _read_mcp_config_user()
        servers = config.get("mcp_servers")
        if not isinstance(servers, dict):
            servers = {}
        servers[server_key] = config_block
        config["mcp_servers"] = servers
        _strip_top_level_stray(config, server_key)
        tmp = config_path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
        os.replace(str(tmp), str(config_path))


# ---------------------------------------------------------------------------
# Knowledge storage
# ---------------------------------------------------------------------------


def _knowledge_tool_dir(tool_id: str) -> Path:
    """Path to .knowledge/{tool_id}/."""
    return KNOWLEDGE_DIR / tool_id


def knowledge_exists(tool_id: str) -> bool:
    """Check if knowledge directory exists for a tool."""
    _validate_tool_id(tool_id)
    return _knowledge_tool_dir(tool_id).is_dir()


def save_knowledge(
    tool_id: str,
    *,
    readme: str = "",
    api_discovery: dict | None = None,
    function_signatures: dict | None = None,
    micro_test_code: str = "",
    creation_log: dict | None = None,
) -> dict:
    """Save creation-time knowledge for a tool.

    Called by STCoscientist during the creation pipeline. Creates .knowledge/{tool_id}/
    and writes provided files atomically. Safe to call multiple times (updates).

    Args:
        tool_id: The tool to save knowledge for.
        readme: README text from the source repo.
        api_discovery: API discovery results dict.
        function_signatures: Function signature mappings dict.
        micro_test_code: Micro-test source code string.
        creation_log: Creation process record dict.

    Returns:
        {"success": True, "tool_id": str, "files_written": list[str]}
    """
    _validate_tool_id(tool_id)
    kdir = _knowledge_tool_dir(tool_id)
    kdir.mkdir(parents=True, exist_ok=True)

    files_written = []

    if readme:
        (kdir / "README.md").write_text(readme)
        files_written.append("README.md")

    if api_discovery is not None:
        _atomic_write_json(kdir / "api_discovery.json", api_discovery)
        files_written.append("api_discovery.json")

    if function_signatures is not None:
        _atomic_write_json(kdir / "function_signatures.json", function_signatures)
        files_written.append("function_signatures.json")

    if micro_test_code:
        (kdir / "micro_tests.py").write_text(micro_test_code)
        files_written.append("micro_tests.py")

    if creation_log is not None:
        _atomic_write_json(kdir / "creation_log.json", creation_log)
        files_written.append("creation_log.json")

    # Update install_log with knowledge_dir pointer
    try:
        _update_log_entry(tool_id, {"knowledge_dir": f".knowledge/{tool_id}"})
    except Exception:
        pass  # Non-fatal

    return {"success": True, "tool_id": tool_id, "files_written": files_written}


def read_knowledge(tool_id: str) -> dict:
    """Read all stored knowledge for a tool.

    Returns a merged dict with all available knowledge files.
    Returns {"has_knowledge": False} if no knowledge directory.
    Never raises for missing or corrupt data — returns partial results with warnings.
    """
    _validate_tool_id(tool_id)
    kdir = _knowledge_tool_dir(tool_id)
    result: dict = {"tool_id": tool_id, "has_knowledge": False, "warnings": []}

    if not kdir.is_dir():
        return result

    result["has_knowledge"] = True

    # README
    readme_path = kdir / "README.md"
    if readme_path.exists():
        try:
            result["readme"] = readme_path.read_text()
        except OSError as e:
            result["warnings"].append(f"README.md unreadable: {e}")

    # api_discovery.json
    api_path = kdir / "api_discovery.json"
    if api_path.exists():
        try:
            result["api_discovery"] = json.loads(api_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            result["warnings"].append(f"api_discovery.json corrupt: {e}")

    # function_signatures.json
    sigs_path = kdir / "function_signatures.json"
    if sigs_path.exists():
        try:
            result["function_signatures"] = json.loads(sigs_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            result["warnings"].append(f"function_signatures.json corrupt: {e}")

    # micro_tests.py
    tests_path = kdir / "micro_tests.py"
    if tests_path.exists():
        try:
            result["micro_test_code"] = tests_path.read_text()
        except OSError as e:
            result["warnings"].append(f"micro_tests.py unreadable: {e}")

    # creation_log.json
    clog_path = kdir / "creation_log.json"
    if clog_path.exists():
        try:
            result["creation_log"] = json.loads(clog_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            result["warnings"].append(f"creation_log.json corrupt: {e}")

    # modification_log.json
    mlog_path = kdir / "modification_log.json"
    if mlog_path.exists():
        try:
            result["modification_log"] = json.loads(mlog_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            result["warnings"].append(f"modification_log.json corrupt: {e}")

    return result


def update_knowledge(
    tool_id: str,
    *,
    function_signatures: dict | None = None,
    creation_log: dict | None = None,
) -> dict:
    """Update specific knowledge files after a modification.

    For function_signatures: replaces the entire file.
    For creation_log: replaces the entire file (caller provides full dict).

    Returns:
        {"success": True, "files_updated": list}
    """
    _validate_tool_id(tool_id)
    kdir = _knowledge_tool_dir(tool_id)
    kdir.mkdir(parents=True, exist_ok=True)

    files_updated = []

    if function_signatures is not None:
        _atomic_write_json(kdir / "function_signatures.json", function_signatures)
        files_updated.append("function_signatures.json")

    if creation_log is not None:
        _atomic_write_json(kdir / "creation_log.json", creation_log)
        files_updated.append("creation_log.json")

    return {"success": True, "files_updated": files_updated}


def bootstrap_knowledge(tool_id: str) -> dict:
    """Reconstruct partial knowledge from existing files for pre-knowledge tools.

    Parses worker.py (AST) for argparse params + pipeline steps,
    mcp_server.py for MCP function signature, env.yaml, and install_log.json.

    Marks all data with {"bootstrapped": True, "bootstrapped_at": timestamp}.

    Returns:
        {"success": True, "tool_id": str, "completeness": float}
    """
    _validate_tool_id(tool_id)
    entry = _require_active(tool_id)
    now = datetime.now().isoformat()
    completeness_parts = 0
    completeness_total = 6  # README, api_disc, func_sigs, micro_test, creation_log, env

    kdir = _knowledge_tool_dir(tool_id)
    kdir.mkdir(parents=True, exist_ok=True)

    # --- Parse worker.py ---
    worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
    worker_cli_params = {}
    pipeline_steps = []
    data_checks = []

    if worker_path.exists():
        try:
            worker_source = worker_path.read_text()
            tree = ast.parse(worker_source)
            worker_cli_params = _extract_argparse_params(tree)
            pipeline_steps = _extract_pipeline_steps(worker_source)
            data_checks = _extract_data_checks(tree)
        except Exception:
            pass

    # --- Parse mcp_server.py ---
    server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
    mcp_params = {}
    mcp_func_name = ""

    if server_path.exists():
        try:
            server_source = server_path.read_text()
            tree = ast.parse(server_source)
            mcp_func_name, mcp_params = _extract_mcp_signature(tree)
        except Exception:
            pass

    # --- Build function_signatures.json ---
    func_sigs = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "tool_id": tool_id,
        "bootstrapped": True,
        "bootstrapped_at": now,
        "core_function": {
            "import_path": "unknown (bootstrapped)",
            "parameters": {},
        },
        "worker_cli": {"parameters": worker_cli_params},
        "mcp_function": {"name": mcp_func_name or entry.get("function_name", ""), "parameters": mcp_params},
        "pipeline_steps": pipeline_steps,
    }
    # Validate spatialomicsgym_name matches actual @mcp.tool() function
    if mcp_func_name:
        bn_check = _validate_spatialomicsgym_name(tool_id)
        if not bn_check["ok"] and bn_check["actual_name"]:
            logger.warning(
                "bootstrap_knowledge: spatialomicsgym_name mismatch for %s — YAML='%s', actual='%s'. Auto-fixing.",
                tool_id,
                bn_check["yaml_name"],
                bn_check["actual_name"],
            )
            _fix_spatialomicsgym_name(tool_id, bn_check["actual_name"])

    _atomic_write_json(kdir / "function_signatures.json", func_sigs)
    if worker_cli_params or mcp_params:
        completeness_parts += 1

    # --- Build api_discovery.json (partial) ---
    api_disc = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "tool_id": tool_id,
        "bootstrapped": True,
        "bootstrapped_at": now,
        "source_url": entry.get("source_url", ""),
        "package": entry.get("package", ""),
        "module": entry.get("module") or entry.get("package", ""),
        "language": entry.get("language", "python"),
        "all_submodules": [],
        "all_functions": {},
        "selected_function": {
            "module_path": "unknown (bootstrapped)",
            "name": entry.get("function_name", ""),
            "reason": "bootstrapped — original selection reason not available",
            "signature": "",
        },
        "rejected_alternatives": [],
        "install_method": "unknown (bootstrapped)",
    }
    _atomic_write_json(kdir / "api_discovery.json", api_disc)
    completeness_parts += 0.5  # Partial — missing detailed discovery

    # --- Build creation_log.json (partial) ---
    creation_log = {
        "schema_version": KNOWLEDGE_SCHEMA_VERSION,
        "tool_id": tool_id,
        "bootstrapped": True,
        "bootstrapped_at": now,
        "created_at": entry.get("created_at", ""),
        "source_url": entry.get("source_url", ""),
        "creation_phases": {},
        "data_validation_checks": data_checks,
        "strengths": [],
        "limitations": [],
        "files": {
            "worker": f"{tool_id}_worker.py",
            "mcp_server": f"{tool_id}_mcp_server.py",
            "env_yaml": f"{tool_id}_env.yaml",
        },
    }
    _atomic_write_json(kdir / "creation_log.json", creation_log)
    completeness_parts += 0.5

    # --- Parse env.yaml ---
    env_path = TOOLS_USER_DIR / f"{tool_id}_env.yaml"
    if env_path.exists():
        completeness_parts += 0.5

    # No README available for bootstrap
    # No micro_test available for bootstrap
    # completeness_parts stays as-is for those

    completeness = completeness_parts / completeness_total

    # Update install_log
    try:
        _update_log_entry(tool_id, {"knowledge_dir": f".knowledge/{tool_id}"})
    except Exception:
        pass

    return {"success": True, "tool_id": tool_id, "completeness": completeness}


def knowledge_info() -> dict:
    """Summary of knowledge state for all tools.

    Returns:
        {"tools_with_knowledge": int, "tools_without_knowledge": int,
         "total_backup_count": int, "per_tool": list[dict]}
    """
    entries = _read_install_log()
    per_tool = []
    total_backups = 0

    for entry in entries:
        tid = entry.get("tool_id", "")
        if not tid:
            continue
        kdir = KNOWLEDGE_DIR / tid
        has_knowledge = kdir.is_dir()
        backup_count = 0
        if has_knowledge:
            backups_dir = kdir / "backups"
            if backups_dir.exists():
                backup_count = sum(1 for d in backups_dir.iterdir() if d.is_dir())
        total_backups += backup_count
        per_tool.append(
            {
                "tool_id": tid,
                "status": entry.get("status", ""),
                "has_knowledge": has_knowledge,
                "backup_count": backup_count,
            }
        )

    with_k = sum(1 for t in per_tool if t["has_knowledge"])
    without_k = sum(1 for t in per_tool if not t["has_knowledge"])

    return {
        "tools_with_knowledge": with_k,
        "tools_without_knowledge": without_k,
        "total_backup_count": total_backups,
        "per_tool": per_tool,
    }


# ---------------------------------------------------------------------------
# Health check system
# ---------------------------------------------------------------------------


def _extract_tool_info(tool_id: str, entry: dict) -> dict:
    """Extract tool details for health report display.

    Two paths:
      - With knowledge: reads function_signatures.json.
      - Without knowledge (old tools): falls back to AST parsing + MCP config.

    Every extraction is wrapped in try/except — never crashes, returns partial data.
    """
    info = {
        "package": entry.get("module") or entry.get("package", tool_id),
        "task_type": entry.get("task_type", ""),
        "source_url": entry.get("source_url", ""),
        "language": entry.get("language", "python"),
        "created_at": entry.get("created_at", ""),
        "env_name": entry.get("env_name", f"user_{tool_id}"),
        "modification_count": entry.get("modification_count", 0),
        "pipeline_steps": [],
        "worker_parameters": {},
        "mcp_signature": {},
    }

    # Try knowledge-based extraction first
    kdir = KNOWLEDGE_DIR / tool_id
    func_sig_path = kdir / "function_signatures.json"
    if func_sig_path.exists():
        try:
            sigs = json.loads(func_sig_path.read_text())
            # Pipeline steps
            if "pipeline_steps" in sigs:
                steps = sigs["pipeline_steps"]
                if isinstance(steps, list):
                    info["pipeline_steps"] = [
                        s.get("call", s.get("step", str(s))) if isinstance(s, dict) else str(s) for s in steps
                    ]
            # Worker CLI parameters (may be nested under "parameters" key)
            if "worker_cli" in sigs and isinstance(sigs["worker_cli"], dict):
                wc = sigs["worker_cli"]
                info["worker_parameters"] = wc.get("parameters", wc) if "parameters" in wc else wc
            # MCP function signature
            if "mcp_function" in sigs and isinstance(sigs["mcp_function"], dict):
                info["mcp_signature"] = sigs["mcp_function"].get("parameters", sigs["mcp_function"])
            if info["pipeline_steps"] or info["worker_parameters"]:
                return info
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    # Fallback: AST parsing for Python workers
    is_r = info["language"].lower() == "r"
    if not is_r:
        worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
        if worker_path.exists():
            try:
                source = worker_path.read_text()
                tree = ast.parse(source)
                params = _extract_argparse_params(tree)
                if params:
                    info["worker_parameters"] = params
                steps = _extract_pipeline_steps(source)
                if steps:
                    info["pipeline_steps"] = [s.get("call", "") for s in steps]
            except Exception:
                pass

        server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
        if server_path.exists():
            try:
                tree = ast.parse(server_path.read_text())
                _fname, mcp_params = _extract_mcp_signature(tree)
                if mcp_params:
                    info["mcp_signature"] = mcp_params
            except Exception:
                pass

    # Also try MCP config as fallback for signature
    if not info["mcp_signature"]:
        try:
            config = _read_mcp_config_user()
            server_entry = _server_block(config, f"user_{tool_id}")
            tools_list = server_entry.get("tools", [])
            if tools_list:
                params_from_config = tools_list[0].get("parameters", {})
                info["mcp_signature"] = params_from_config
        except Exception:
            pass

    return info


def _check_param_consistency(tool_id: str, entry: dict) -> dict:
    """Compare parameters across MCP config, mcp_server.py, and worker.py.

    Returns {"ok": bool, "detail": str, "mismatches": [...]}.
    """
    mismatches = []
    is_r = entry.get("language", "python").lower() == "r"

    # Source 1: MCP config params
    config_params = {}
    config_spatialomicsgym_name = ""
    try:
        config = _read_mcp_config_user()
        server_entry = _server_block(config, f"user_{tool_id}")
        tools_list = server_entry.get("tools", [])
        if tools_list:
            config_params = tools_list[0].get("parameters", {})
            config_spatialomicsgym_name = tools_list[0].get("spatialomicsgym_name", "")
    except Exception:
        pass

    # Source 2: MCP server function signature (AST)
    server_params = {}
    extracted_func_name = ""
    server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
    if server_path.exists():
        try:
            tree = ast.parse(server_path.read_text())
            extracted_func_name, server_params = _extract_mcp_signature(tree)
        except Exception:
            pass

    # Check spatialomicsgym_name matches @mcp.tool() function name
    if extracted_func_name and config_spatialomicsgym_name:
        if config_spatialomicsgym_name != extracted_func_name:
            mismatches.append(
                f"spatialomicsgym_name mismatch: YAML='{config_spatialomicsgym_name}', @mcp.tool()='{extracted_func_name}'"
            )

    # Source 3: Worker argparse params
    worker_params = {}
    if not is_r:
        worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
        if worker_path.exists():
            try:
                tree = ast.parse(worker_path.read_text())
                worker_params = _extract_argparse_params(tree)
            except Exception:
                pass

    # Compare server vs worker parameter counts
    if server_params and worker_params:
        server_names = set(server_params.keys())
        # Normalize worker param names (strip -- prefix, replace - with _)
        worker_names = set()
        worker_name_map = {}
        for k in worker_params:
            normalized = k.lstrip("-").replace("-", "_")
            worker_names.add(normalized)
            worker_name_map[normalized] = k

        # Known MCP-to-worker name aliases (MCP servers often use descriptive names)
        known_aliases = {
            "input_h5ad": "input",
            "input_file": "input",
            "input_path": "input",
            "output_directory": "output_dir",
            "out_dir": "output_dir",
        }
        # Expand server names with aliases for matching
        server_normalized = set()
        for sn in server_names:
            server_normalized.add(sn)
            if sn in known_aliases:
                server_normalized.add(known_aliases[sn])

        only_server = server_names - worker_names - set(known_aliases.keys())
        only_worker = worker_names - server_normalized - {"input", "output_dir"}
        if only_server:
            mismatches.append(f"In MCP server but not worker: {only_server}")
        if only_worker:
            mismatches.append(f"In worker but not MCP server: {only_worker}")

        # Compare defaults where both exist
        for pname in server_names & worker_names:
            s_default = server_params.get(pname, {}).get("default")
            w_key = worker_name_map.get(pname, f"--{pname}")
            w_default = worker_params.get(w_key, {}).get("default")
            if s_default is not None and w_default is not None and s_default != w_default:
                mismatches.append(f"{pname}: server default={s_default}, worker default={w_default}")

    # Compare MCP config vs server
    if config_params and server_params:
        config_names = set(config_params.keys())
        server_names = set(server_params.keys())
        diff = config_names.symmetric_difference(server_names)
        if diff:
            mismatches.append(f"Config/server param mismatch: {diff}")

    if mismatches:
        return {"ok": False, "detail": "; ".join(mismatches[:3]), "mismatches": mismatches}
    if not server_params and not worker_params:
        return {"ok": True, "detail": "No params extracted (unable to compare)"}
    return {"ok": True, "detail": "Parameters consistent across layers"}


def _memory_context(source_url: str) -> dict:
    """Read-only memory annotation for one tool, or {} when there is nothing to say.

    Its own try/except is the point. The per-tool body of ``health_check_tools`` runs inside an
    ``except BaseException`` that downgrades whatever it catches to CHECK_FAILED, so a memory
    backend that raised in here would turn an optional annotation into a health verdict -- the
    F11 violation the flag exists to avoid. Anything that goes wrong reads as "no memory".
    """
    if not source_url:
        return {}
    try:
        from tools_user.memory_manager import MemoryManager

        mm = MemoryManager.get()
        # Gated inside: returns {} when memory is disabled, so no config check is needed here.
        mem = mm.read_short_term(source_url)
        if not mem or not mem.get("attempts"):
            return {}
        # best_attempt RANKS rather than filters, so this outcome may well be a failure -- which is
        # what a health report should show. The key says outcome, so it carries the outcome string;
        # best_attempt_id is an int index into the attempt list and published "outcome: 3".
        best = mm.best_attempt(source_url) or {}
        return {
            "prior_attempts": len(mem["attempts"]),
            "best_attempt_outcome": best.get("outcome"),
            "last_verified": mem.get("last_verified"),
        }
    except Exception:
        return {}


def health_check_tools(
    tool_ids: list[str] | None = None,
    *,
    include_real_data: bool = False,
    test_timeout: int = 120,
    include_memory: bool = False,
) -> dict:
    """Comprehensive health check on all (or specified) user-created MCP tools.

    Read-only operation — does NOT modify any files, configs, or state.
    Fully autonomous — STCoscientist calls this from a single user prompt, no interaction needed.
    Works for both new tools (with knowledge) and old tools (without).

    Args:
        tool_ids: Specific tool_ids to check. None = all tools.
        include_real_data: If True, run the expensive real_data test (~2min/tool).
        test_timeout: Timeout in seconds for real data test (per tool).
        include_memory: If True, attach a read-only ``memory`` key to each ACTIVE tool's entry.
            Off by default so an existing caller's report keeps exactly the shape it had. The
            annotation is additive: it can never change the HEALTHY/DEGRADED/BROKEN verdict, and
            a memory subsystem that raises leaves ``memory`` empty rather than failing the tool
            (F4 zero writes, F11 verdict independence — ``know_how/memory_user_mcp_tools.md``).

    Returns:
        Structured health report with infrastructure status, per-tool checks,
        recommendations, tool details, and disk usage.

        ``disk_usage["total_reclaimable"]`` is what ``permanent_delete`` would actually free
        across every trashed tool -- conda env plus ``.trash/{tool_id}/`` plus
        ``vendor_{tool_id}/`` -- the same quantity, under the same name, that
        ``trash_manager.trash_info`` and ``preview_permanent_delete`` report. The three
        components are published beside it so the total can be checked against its parts.
    """
    import time as _time

    start = _time.time()
    report: dict = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": 0.0,
        "registry_status": "OK",
        "infrastructure": {
            "base_mcp_symlink": {"ok": True, "detail": ""},
            "worker_utils_symlink": {"ok": True, "detail": ""},
            "install_log_valid": True,
            "install_log_warnings": [],
        },
        "orphaned_files": [],
        "stale_mcp_entries": [],
        "unwired_tools": [],
        "summary": {"total": 0, "healthy": 0, "degraded": 0, "broken": 0, "trashed": 0, "check_failed": 0},
        "tools": [],
        "disk_usage": {
            "active_conda_size": "0 B",
            "trashed_conda_size": "0 B",
            "trash_files_size": "0 B",
            "trashed_vendor_size": "0 B",
            "total_reclaimable": "0 B",
        },
    }

    # === Global infrastructure checks ===

    # 1. Check install_log.json
    if not INSTALL_LOG.exists():
        report["registry_status"] = "EMPTY"
        report["infrastructure"]["install_log_valid"] = False
        report["infrastructure"]["install_log_warnings"].append("install_log.json not found")
    else:
        try:
            raw = json.loads(INSTALL_LOG.read_text())
            if not isinstance(raw, list):
                report["registry_status"] = "CORRUPT"
                report["infrastructure"]["install_log_valid"] = False
                report["infrastructure"]["install_log_warnings"].append("install_log.json is not a JSON array")
            else:
                # Check for duplicates and missing fields
                seen_ids = set()
                for i, entry in enumerate(raw):
                    tid = entry.get("tool_id", "")
                    if not tid:
                        report["infrastructure"]["install_log_warnings"].append(f"Entry {i}: missing tool_id")
                    elif tid in seen_ids:
                        report["infrastructure"]["install_log_warnings"].append(f"Duplicate tool_id: {tid}")
                    else:
                        seen_ids.add(tid)
                    for field in ("status", "files"):
                        if field not in entry:
                            report["infrastructure"]["install_log_warnings"].append(f"Entry '{tid}': missing '{field}'")
        except (json.JSONDecodeError, OSError) as e:
            report["registry_status"] = "CORRUPT"
            report["infrastructure"]["install_log_valid"] = False
            report["infrastructure"]["install_log_warnings"].append(f"Parse error: {str(e)[:100]}")

    # 2. Check shared symlinks
    base_mcp_path = TOOLS_USER_DIR / "base_mcp.py"
    worker_utils_path = TOOLS_USER_DIR / "worker_utils.py"
    ok, detail = _check_symlink(base_mcp_path)
    report["infrastructure"]["base_mcp_symlink"] = {"ok": ok, "detail": detail}
    ok, detail = _check_symlink(worker_utils_path)
    report["infrastructure"]["worker_utils_symlink"] = {"ok": ok, "detail": detail}
    infra_broken = (
        not report["infrastructure"]["base_mcp_symlink"]["ok"]
        or not report["infrastructure"]["worker_utils_symlink"]["ok"]
    )

    # 3. Scan for orphaned files
    entries = _read_install_log()
    known_ids = {e.get("tool_id", "") for e in entries}
    try:
        for f in TOOLS_USER_DIR.iterdir():
            if f.is_symlink() or not f.is_file():
                continue
            for suffix in ("_worker.py", "_mcp_server.py"):
                if f.name.endswith(suffix):
                    inferred_id = f.name[: -len(suffix)]
                    if inferred_id and inferred_id not in known_ids:
                        report["orphaned_files"].append(f.name)
    except OSError:
        pass

    # 4. Scan for stale MCP config entries
    try:
        config = _read_mcp_config_user()
        servers = config.get("mcp_servers")
        # An LLM-authored config can leave `mcp_servers:` null; iterating None would raise and skip
        # BOTH scans below (the bare except would swallow it silently).
        servers = servers if isinstance(servers, dict) else {}
        active_ids = {e.get("tool_id", "") for e in entries if e.get("status") == "active"}
        for server_key in servers:
            if server_key.startswith("user_"):
                inferred_id = server_key[5:]
                if inferred_id and inferred_id not in active_ids:
                    report["stale_mcp_entries"].append(server_key)

        # 4b. The INVERSE scan: an active tool that no mcp_servers entry points at. Sections 3 and 4
        # both look from the config/filesystem toward the tool, so the state where a tool is fully
        # installed and simply never wired -- files present, install log happy, uncallable -- had
        # nothing looking for it. That is the state a real session left this repo in. The two shapes
        # need different fixes, so the reason distinguishes them: a block written as a SIBLING of
        # mcp_servers must be moved, a genuinely missing one must be added.
        for tool_id in sorted(tid for tid in active_ids if tid):
            server_key = f"user_{tool_id}"
            if server_key in servers:
                continue
            misplaced = isinstance(config.get(server_key), dict)
            report["unwired_tools"].append(
                {
                    "tool_id": tool_id,
                    "server_key": server_key,
                    "reason": (
                        f"'{server_key}' sits at the top level of mcp_config_user.yaml instead of "
                        "under 'mcp_servers' -- move the block"
                        if misplaced
                        else f"'{server_key}' is absent from mcp_config_user.yaml -- the tool is "
                        "installed but was never wired, so it cannot be called"
                    ),
                }
            )
    except Exception:
        pass

    # === Per-tool checks ===

    if tool_ids is not None:
        entries = [e for e in entries if e.get("tool_id") in tool_ids]

    active_conda_bytes = 0
    # permanent_delete removes three things per trashed tool, so "reclaimable" has to count all
    # three or it undersells the free space by however large the vendor checkout is (measured on
    # this host: vendor_stereopy is 371 MB against a 632 MB env). trash_manager.trash_info and
    # preview_permanent_delete already report the sum; this is the same number under the same name.
    trashed_conda_bytes = 0
    trash_files_bytes = 0
    trashed_vendor_bytes = 0

    for entry in entries:
        tid = entry.get("tool_id", "")
        status = entry.get("status", "unknown")
        language = entry.get("language", "python")

        # Trashed tools — skip checks
        if status == "trashed":
            report["summary"]["trashed"] += 1
            report["summary"]["total"] += 1
            report["tools"].append(
                {
                    "tool_id": tid,
                    "health": "TRASHED",
                    "info": _extract_tool_info(tid, entry),
                    "checks": [],
                    "recommendations": [f"Restore with restore_tool('{tid}') or permanently delete."],
                }
            )
            trashed_conda_bytes += _dir_size_bytes(CONDA_ENVS_DIR / f"user_{tid}")
            # Derived from the live TOOLS_USER_DIR rather than a module constant, so a relocated
            # store (and a monkeypatched one) is measured where it actually is.
            trash_files_bytes += _dir_size_bytes(TOOLS_USER_DIR / ".trash" / tid)
            trashed_vendor_bytes += _dir_size_bytes(TOOLS_USER_DIR / f"vendor_{tid}")
            continue

        report["summary"]["total"] += 1

        # Active tool — full check, wrapped in error isolation
        try:
            checks = []
            recommendations = []

            # Check 1: Files exist
            expected_files = []
            is_r = language.lower() == "r"
            if is_r:
                expected_files = [f"{tid}_worker.R", f"{tid}_mcp_server.py", f"{tid}_env.yaml"]
            else:
                expected_files = [f"{tid}_worker.py", f"{tid}_mcp_server.py", f"{tid}_env.yaml"]
            missing = [f for f in expected_files if not (TOOLS_USER_DIR / f).exists()]
            if missing:
                checks.append(
                    {
                        "name": "files",
                        "critical": True,
                        "result": "FAIL",
                        "detail": f"Missing: {', '.join(missing)}",
                    }
                )
                recommendations.append(f"CRITICAL: Missing files {missing}. Re-create or restore from backup.")
            else:
                checks.append(
                    {
                        "name": "files",
                        "critical": True,
                        "result": "PASS",
                        "detail": f"{len(expected_files)}/{len(expected_files)} files present",
                    }
                )

            # Check 2: Conda env exists
            env_path = CONDA_ENVS_DIR / f"user_{tid}"
            if env_path.exists():
                checks.append(
                    {
                        "name": "conda_env",
                        "critical": True,
                        "result": "PASS",
                        "detail": f"user_{tid} exists",
                    }
                )
                active_conda_bytes += _dir_size_bytes(env_path)
            else:
                checks.append(
                    {
                        "name": "conda_env",
                        "critical": True,
                        "result": "FAIL",
                        "detail": f"Conda env not found: user_{tid}",
                    }
                )
                recommendations.append(f"CRITICAL: Conda env missing. Recreate from {tid}_env.yaml.")

            # Check 3: Infrastructure (broken symlinks affect all tools)
            if infra_broken:
                checks.append(
                    {
                        "name": "infrastructure",
                        "critical": True,
                        "result": "FAIL",
                        "detail": "Shared symlinks (base_mcp.py / worker_utils.py) broken",
                    }
                )
                recommendations.append("CRITICAL: Fix broken symlinks in tools_user/.")

            # Checks 4-8: Run test suite (import, syntax, dry_run, config, real_data)
            skip = frozenset() if include_real_data else frozenset({"real_data"})
            module = entry.get("module") or entry.get("package", tid)
            task_type = entry.get("task_type", "spatial_clustering")
            test_results = _run_test_suite(
                tid,
                module=module,
                task_type=task_type,
                test_timeout=test_timeout,
                skip_tests=skip,
                language=language,
                module_language=(entry or {}).get("wrapped_language"),
            )

            critical_tests = {"import", "syntax", "imports_scan", "dry_run", "config"}
            for tr in test_results:
                is_critical = tr["test"] in critical_tests
                # Both encodings of "did not run" -- see _test_did_not_run. Reading only the
                # skip_tests one filed a fixture-less host's real_data as a failure, so every tool on
                # a host without benchmark data (a pip install) went DEGRADED under
                # include_real_data=True with a "real_data failed" note for a test that never ran.
                if _test_did_not_run(tr):
                    result_str = "SKIP"
                elif tr["passed"]:
                    result_str = "PASS"
                else:
                    result_str = "FAIL"
                checks.append(
                    {
                        "name": tr["test"],
                        "critical": is_critical,
                        "result": result_str,
                        "detail": tr["detail"],
                    }
                )
                if tr.get("skipped"):
                    # Not a failure, but not a pass either -- say which one it was and that it is unproven.
                    recommendations.append(f"Note: {tr['test']} did not run — {tr['detail'][:100]}")
                elif not tr["passed"]:
                    severity = "CRITICAL" if is_critical else "Warning"
                    recommendations.append(f"{severity}: {tr['test']} failed — {tr['detail'][:100]}")

            # Check 9: Parameter consistency
            param_check = _check_param_consistency(tid, entry)
            checks.append(
                {
                    "name": "param_consistency",
                    "critical": False,
                    "result": "PASS" if param_check["ok"] else "WARN",
                    "detail": param_check["detail"],
                }
            )
            if not param_check["ok"]:
                recommendations.append("Parameter drift detected. Run bootstrap_knowledge or reconcile params.")

            # Check 10: spatialomicsgym_name validity
            bn_check = _validate_spatialomicsgym_name(tid)
            if bn_check["ok"]:
                checks.append(
                    {
                        "name": "spatialomicsgym_name",
                        "critical": True,
                        "result": "PASS",
                        "detail": bn_check["detail"],
                    }
                )
            else:
                checks.append(
                    {
                        "name": "spatialomicsgym_name",
                        "critical": True,
                        "result": "FAIL",
                        "detail": bn_check["detail"],
                    }
                )
                if bn_check["actual_name"]:
                    recommendations.append(
                        f"CRITICAL: {bn_check['detail']}. "
                        f"Fix: update mcp_config_user.yaml spatialomicsgym_name to '{bn_check['actual_name']}'."
                    )
                else:
                    recommendations.append(f"CRITICAL: {bn_check['detail']}.")

            # Check 11: Knowledge state
            has_knowledge = knowledge_exists(tid)
            mod_count = 0
            # read_knowledge never raises: it returns what it could parse and lists what it could
            # not in `warnings`. Initialise before the try so a read that dies anyway leaves this
            # defined rather than falling through to a green PASS.
            k_warnings: list = []
            if has_knowledge:
                try:
                    k = read_knowledge(tid)
                    k_warnings = list(k.get("warnings") or [])
                    mod_log = k.get("modification_log", {})
                    if isinstance(mod_log, dict):
                        mod_count = len(mod_log.get("modifications", []))
                except Exception:
                    pass
                if k_warnings:
                    # A file that failed to parse arrives here as an absent key, so `mod_count`
                    # falls back to 0 -- and reporting "0 modifications recorded" states as fact
                    # the one thing that could not be determined. Every warning is "<file> corrupt:"
                    # or "<file> unreadable:", so the first token names the file.
                    damaged = sorted({w.split(" ", 1)[0] for w in k_warnings})
                    names = ", ".join(damaged)
                    detail = (
                        f"Knowledge present, unreadable: {names}"
                        if "modification_log.json" in damaged
                        else f"Knowledge present, {mod_count} modifications recorded; unreadable: {names}"
                    )
                    checks.append({"name": "knowledge", "critical": False, "result": "WARN", "detail": detail})
                    recommendations.append(
                        f"Knowledge files unreadable for '{tid}' ({names}). The tool still runs; "
                        f"its recorded history does not. Re-run bootstrap_knowledge('{tid}') to "
                        f"rebuild what can be recovered."
                    )
                else:
                    checks.append(
                        {
                            "name": "knowledge",
                            "critical": False,
                            "result": "PASS",
                            "detail": f"Knowledge present, {mod_count} modifications recorded",
                        }
                    )
            else:
                checks.append(
                    {
                        "name": "knowledge",
                        "critical": False,
                        "result": "WARN",
                        "detail": "No knowledge directory",
                    }
                )
                recommendations.append(f"Run bootstrap_knowledge('{tid}') to create knowledge base.")

            # Check 12: Backup state
            backups = list_backups(tid)
            # Count only what rollback would accept. create_backup writes .backup_meta.json last,
            # so an interrupted backup leaves a version directory that list_backups still reports;
            # counting it here overstates how much of this tool's history can actually be undone,
            # and "v1 preserved" would vouch for a v1 that cannot be restored.
            restorable = [b for b in backups if b.get("restorable")]
            backup_count = len(restorable)
            unrestorable = len(backups) - backup_count
            v1_preserved = any(b.get("version") == 1 for b in restorable)
            if backup_count > 0:
                detail_parts = [f"{backup_count} backup(s)"]
                if v1_preserved:
                    detail_parts.append("v1 preserved")
                else:
                    detail_parts.append("v1 MISSING")
                if unrestorable:
                    detail_parts.append(f"{unrestorable} unrestorable (rollback would refuse them)")
                checks.append(
                    {
                        "name": "backups",
                        "critical": False,
                        "result": "PASS" if v1_preserved and not unrestorable else "WARN",
                        "detail": ", ".join(detail_parts),
                    }
                )
                if not v1_preserved:
                    recommendations.append(f"Genesis backup (v1) missing. Create with create_backup('{tid}').")
                if unrestorable:
                    # Quote the first refusal rather than name one cause: a version can be
                    # unrestorable because its metadata is gone, because its files are, or because
                    # they no longer hash to what was recorded, and the reader's next move differs.
                    reasons = [b["refusal"] for b in backups if b.get("refusal")]
                    why = reasons[0] if reasons else "rollback refuses them."
                    recommendations.append(
                        f"{unrestorable} backup(s) of '{tid}' cannot be rolled back to. {why} "
                        f"Roll back to a version list_backups() marks restorable, and create a "
                        f"fresh backup."
                    )
            else:
                if unrestorable:
                    no_backup_detail = f"No restorable backups ({unrestorable} rollback would refuse)"
                else:
                    no_backup_detail = "No backups"
                checks.append(
                    {
                        "name": "backups",
                        "critical": False,
                        "result": "WARN",
                        "detail": no_backup_detail,
                    }
                )
                recommendations.append(f"No backups exist. Create one with create_backup('{tid}').")

            # Classify health
            critical_failed = any(c["result"] == "FAIL" and c["critical"] for c in checks)
            non_critical_warned = any(c["result"] in ("FAIL", "WARN") and not c["critical"] for c in checks)

            if critical_failed:
                health = "BROKEN"
                report["summary"]["broken"] += 1
            elif non_critical_warned:
                health = "DEGRADED"
                report["summary"]["degraded"] += 1
            else:
                health = "HEALTHY"
                report["summary"]["healthy"] += 1

            # Extract tool info for display
            tool_info = _extract_tool_info(tid, entry)

            tool_entry = {
                "tool_id": tid,
                "health": health,
                "info": tool_info,
                "checks": checks,
                "recommendations": recommendations,
            }
            if include_memory:
                # Attached after the verdict is already computed, and only ever as its own key.
                tool_entry["memory"] = _memory_context(tool_info.get("source_url", ""))
            report["tools"].append(tool_entry)

        except BaseException as exc:
            # Error isolation — one tool's crash doesn't affect others
            report["summary"]["check_failed"] += 1
            report["tools"].append(
                {
                    "tool_id": tid,
                    "health": "CHECK_FAILED",
                    "info": {"package": entry.get("package", tid), "error": str(exc)[:200]},
                    "checks": [],
                    "recommendations": [f"Health check crashed: {str(exc)[:100]}. Investigate manually."],
                }
            )

    # === Disk usage ===
    report["disk_usage"]["active_conda_size"] = _human_size(active_conda_bytes)
    report["disk_usage"]["trashed_conda_size"] = _human_size(trashed_conda_bytes)
    report["disk_usage"]["trash_files_size"] = _human_size(trash_files_bytes)
    report["disk_usage"]["trashed_vendor_size"] = _human_size(trashed_vendor_bytes)
    reclaimable = trashed_conda_bytes + trash_files_bytes + trashed_vendor_bytes
    report["disk_usage"]["total_reclaimable"] = _human_size(reclaimable)

    report["duration_seconds"] = round(_time.time() - start, 1)
    return report


# ---------------------------------------------------------------------------
# AST parsing helpers (for bootstrap and health check)
# ---------------------------------------------------------------------------


def _extract_argparse_params(tree: ast.AST) -> dict:
    """Extract argparse add_argument calls from AST.

    Returns dict of param_name -> {"type": str, "default": value, "required": bool}.
    """
    params = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match: parser.add_argument("--name", ...)
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
            continue
        param_name = first_arg.value
        info: dict = {}
        for kw in node.keywords:
            if kw.arg == "type" and isinstance(kw.value, ast.Name):
                info["type"] = kw.value.id
            elif kw.arg == "default" and isinstance(kw.value, ast.Constant):
                info["default"] = kw.value.value
            elif kw.arg == "required" and isinstance(kw.value, ast.Constant):
                info["required"] = kw.value.value
            elif kw.arg == "help" and isinstance(kw.value, ast.Constant):
                info["help"] = kw.value.value
        params[param_name] = info
    return params


def _extract_pipeline_steps(source: str) -> list[dict]:
    """Extract pipeline step calls from worker source code.

    Looks for lines matching common scanpy/analysis patterns.
    """
    patterns = [
        (r"sc\.pp\.\w+\(.*?\)", "preprocessing"),
        (r"sc\.tl\.\w+\(.*?\)", "analysis"),
        (r"sc\.pl\.\w+\(.*?\)", "plotting"),
        (r"ad\.read_h5ad\(.*?\)", "data_loading"),
        (r"adata\.write_h5ad\(.*?\)", "save_output"),
    ]
    steps = []
    step_num = 0
    for line in source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        for pattern, category in patterns:
            match = re.search(pattern, stripped)
            if match:
                step_num += 1
                steps.append({"step": step_num, "call": match.group(0), "purpose": category})
                break
    return steps


def _extract_data_checks(tree: ast.AST) -> list[str]:
    """Extract data readiness check strings from AST."""
    checks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and "check" in node.name.lower():
            # Extract string comparisons inside the function
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    val = child.value.strip()
                    if len(val) > 5 and ("Missing" in val or "spatial" in val.lower()):
                        checks.append(val)
    return checks


def _extract_mcp_signature(tree: ast.AST) -> tuple[str, dict]:
    """Extract MCP tool function name and parameters from mcp_server.py AST.

    Returns (function_name, params_dict).
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Look for functions decorated with @mcp.tool()
        for dec in node.decorator_list:
            is_mcp = False
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                if dec.func.attr == "tool":
                    is_mcp = True
            if not is_mcp:
                continue
            func_name = node.name
            params = {}
            for arg in node.args.args:
                arg_name = arg.arg
                if arg_name == "self":
                    continue
                type_str = ""
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        type_str = arg.annotation.id
                    elif isinstance(arg.annotation, ast.Constant):
                        type_str = str(arg.annotation.value)
                params[arg_name] = {"type": type_str}
            # Extract defaults (aligned from the end)
            defaults = node.args.defaults
            non_default_count = len(node.args.args) - len(defaults)
            param_names = [a.arg for a in node.args.args if a.arg != "self"]
            for i, default in enumerate(defaults):
                idx = non_default_count + i
                if idx < len(param_names):
                    pname = param_names[idx]
                    if isinstance(default, ast.Constant) and pname in params:
                        params[pname]["default"] = default.value
            # Mark required params
            for i, pname in enumerate(param_names):
                if pname in params:
                    params[pname]["required"] = i < non_default_count
            return func_name, params
    return "", {}


def _validate_spatialomicsgym_name(tool_id: str) -> dict:
    """Check that mcp_config_user.yaml spatialomicsgym_name matches the actual @mcp.tool() function.

    Returns {"ok": bool, "yaml_name": str, "actual_name": str, "detail": str}.
    """
    server_key = f"user_{tool_id}"
    try:
        config = _read_mcp_config_user()
    except Exception:  # pragma: no cover - _read_mcp_config_user swallows its own errors
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": "Cannot read mcp_config_user.yaml"}

    # Each branch below used to raise inside one broad try and be reported as "Cannot read
    # mcp_config_user.yaml". The file parsed in every one of them -- the read cannot fail, it
    # returns {} for a missing file, a parse error and a non-dict document -- so that message named
    # the one thing that had not gone wrong and sent the repair looking for a YAML syntax error.
    # mcp_config_user.yaml is agent-authored and `tools:` as a list of bare names is a common shape
    # for a model to write, so each failure has to name the field that is actually wrong.
    server_entry = _server_block(config, server_key)
    if not server_entry:
        detail = f"{server_key} not in mcp_config_user.yaml"
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": detail}

    tools_list = server_entry.get("tools")
    if tools_list is None:
        detail = f"no tools: array on {server_key} -- needs one entry per @mcp.tool() function"
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": detail}
    if not isinstance(tools_list, list):
        detail = f"tools: on {server_key} is {type(tools_list).__name__}, not a list of entries"
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": detail}
    if not tools_list:
        detail = f"tools: on {server_key} is empty -- needs one entry per @mcp.tool() function"
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": detail}

    first = tools_list[0]
    if not isinstance(first, dict):
        detail = f"tools[0] on {server_key} is {type(first).__name__}, not a mapping with spatialomicsgym_name"
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": detail}

    yaml_name = first.get("spatialomicsgym_name", "")
    if not yaml_name:
        return {"ok": False, "yaml_name": "", "actual_name": "", "detail": "No spatialomicsgym_name in YAML config"}

    server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"
    if not server_path.exists():
        return {"ok": False, "yaml_name": yaml_name, "actual_name": "", "detail": "MCP server file not found"}

    try:
        tree = ast.parse(server_path.read_text())
        actual_name, _ = _extract_mcp_signature(tree)
    except Exception:
        return {"ok": False, "yaml_name": yaml_name, "actual_name": "", "detail": "Cannot parse MCP server file"}

    if not actual_name:
        return {
            "ok": False,
            "yaml_name": yaml_name,
            "actual_name": "",
            "detail": "No @mcp.tool() function found in server",
        }

    if yaml_name == actual_name:
        return {
            "ok": True,
            "yaml_name": yaml_name,
            "actual_name": actual_name,
            "detail": "spatialomicsgym_name matches @mcp.tool() function",
        }

    return {
        "ok": False,
        "yaml_name": yaml_name,
        "actual_name": actual_name,
        "detail": f"MISMATCH: YAML spatialomicsgym_name='{yaml_name}' but @mcp.tool() function is '{actual_name}'",
    }


def _fix_spatialomicsgym_name(tool_id: str, actual_name: str) -> bool:
    """Update mcp_config_user.yaml spatialomicsgym_name to match the actual @mcp.tool() function name.

    Returns True if successfully fixed.
    """
    try:
        config = _read_mcp_config_user()
        server_key = f"user_{tool_id}"
        # The block is mutated in place and the whole config written back, so this repairs the
        # name whichever level the agent wrote the block at.
        server_entry = _server_block(config, server_key)
        if not server_entry:
            return False
        tools_list = server_entry.get("tools", [])
        if not tools_list:
            return False
        tools_list[0]["spatialomicsgym_name"] = actual_name
        _atomic_write_yaml(mcp_config_user_path(), config)
        logger.info("Fixed spatialomicsgym_name for %s: set to '%s'", tool_id, actual_name)
        return True
    except Exception as e:
        logger.error("Failed to fix spatialomicsgym_name for %s: %s", tool_id, e)
        return False


# ---------------------------------------------------------------------------
# Backup management
# ---------------------------------------------------------------------------


def _backups_dir(tool_id: str) -> Path:
    """Path to .knowledge/{tool_id}/backups/."""
    return KNOWLEDGE_DIR / tool_id / "backups"


def _next_backup_version(tool_id: str) -> int:
    """Determine next backup version number (1-based, monotonically increasing)."""
    bdir = _backups_dir(tool_id)
    if not bdir.exists():
        return 1
    existing = []
    for d in bdir.iterdir():
        if d.is_dir():
            m = re.match(r"^v(\d{3})_", d.name)
            if m:
                existing.append(int(m.group(1)))
    return max(existing, default=0) + 1


def _backup_refusal(version_dir: Path, version: int | None = None) -> tuple[str | None, dict]:
    """Why ``rollback_to_backup`` would refuse this version, or None -- plus its parsed metadata.

    One statement of the five grounds rollback enforces, so that the flag ``list_backups``
    publishes, the anchor ``_prune_old_backups`` protects, and the refusal itself cannot answer
    differently. They did: rollback checks the metadata AND the files it records, while both
    readers stopped at the metadata. A version whose backed-up files were deleted or altered after
    the backup was taken therefore read ``restorable: True`` right up to the refusal -- and was
    offered, by ``_older_restorable``, as the remedy for another version's refusal.

    ``version`` appears only in the message and is read off the directory name when not given. The
    metadata comes back with the answer so callers need not parse the file a second time; it is
    empty on the two grounds that are about the metadata itself.
    """
    if version is None:
        m = re.match(r"^v(\d{3})_", version_dir.name)
        version = int(m.group(1)) if m else 0

    meta_path = version_dir / ".backup_meta.json"
    if not meta_path.exists():
        # create_backup writes this file last, so this is an interrupted backup: the files are
        # here but the checksums they must be verified against never were. Restoring them
        # unverified would break the guarantee rollback exists to make.
        return (
            f"Backup metadata missing for version {version} -- the backup was interrupted "
            f"before it was recorded, so its contents cannot be verified.",
            {},
        )
    try:
        meta = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return (f"Corrupted backup metadata for version {version} at {meta_path}: {e}.", {})

    files_backed_up = meta.get("files_backed_up", {})
    if not files_backed_up:
        return (f"No files recorded in backup metadata for version {version}.", meta)

    for filename, info in files_backed_up.items():
        backup_file = version_dir / filename
        if not backup_file.exists():
            return (f"Backup file missing: {filename} in version {version}.", meta)
        expected_hash = info.get("sha256", "") if isinstance(info, dict) else ""
        # An entry recorded without a checksum is accepted, exactly as the restore loop accepts it.
        if expected_hash and _sha256(backup_file) != expected_hash:
            return (f"Backup file corrupted: {filename} (SHA-256 mismatch).", meta)

    return (None, meta)


def _backup_state(version_dir: Path) -> tuple[bool, str | None]:
    """(restorable, state fingerprint) for one backup version directory.

    ``restorable`` is the whole question ``list_backups`` publishes and ``rollback_to_backup``
    enforces, via ``_backup_refusal`` -- not just the metadata half of it.
    """
    refusal, meta = _backup_refusal(version_dir)
    if refusal is not None:
        return (False, None)
    fp = meta.get("state_fingerprint")
    if not fp:
        files_info = meta.get("files_backed_up", {})
        if files_info:
            fp = _compute_state_fingerprint(files_info)
    return (True, fp or None)


def _prune_old_backups(tool_id: str) -> None:
    """Smart pruning: keep the genesis and latest restorable versions, remove duplicates first.

    Strategy:
      1. Remove versions rollback cannot restore (any of _backup_refusal's grounds) first.
      2. Always keep the oldest and the latest restorable version.
      3. Remove duplicate-state intermediates (same fingerprint as a kept version).
      4. Remove oldest-first among remaining unique intermediates.
    """
    bdir = _backups_dir(tool_id)
    if not bdir.exists():
        return
    versions = []
    for d in bdir.iterdir():
        if d.is_dir():
            m = re.match(r"^v(\d{3})_", d.name)
            if m:
                versions.append((int(m.group(1)), d))
    if len(versions) <= MAX_BACKUP_VERSIONS:
        return
    versions.sort(key=lambda x: x[0])
    states = {ver: _backup_state(vdir) for ver, vdir in versions}

    # Anchor on versions that can actually be rolled back to. A directory whose metadata was never
    # written is not a backup -- rollback_to_backup refuses it and _repair_modification_state calls
    # removing it "removed_incomplete_backup". Protecting one as genesis spends a slot of the
    # MAX_BACKUP_VERSIONS budget on history nothing can restore, for the life of the tool.
    restorable = [(ver, vdir) for ver, vdir in versions if states[ver][0]]
    anchors = restorable or versions
    keep = {anchors[0][0], anchors[-1][0]}

    # Build fingerprint map for kept versions
    kept_fps = {states[ver][1] for ver in keep if states[ver][1]}

    # Classify intermediates as unrestorable, duplicate or unique
    dead = []
    duplicates = []
    unique_intermediates = []
    for ver, vdir in versions:
        if ver in keep:
            continue
        ok, fp = states[ver]
        if not ok:
            dead.append((ver, vdir))
        elif fp and fp in kept_fps:
            duplicates.append((ver, vdir))
        else:
            unique_intermediates.append((ver, vdir))
            if fp:
                kept_fps.add(fp)

    # Remove what cannot be restored first, then duplicates, then oldest unique intermediates
    to_remove = dead + duplicates + unique_intermediates
    removed = 0
    for _ver, vdir in to_remove:
        if len(versions) - removed <= MAX_BACKUP_VERSIONS:
            break
        try:
            shutil.rmtree(str(vdir))
        except OSError:
            # A directory we cannot remove (root-owned from a sudo run, read-only mount, a file
            # another process holds open) is still on disk and still fills a slot. Counting it as
            # freed satisfied the cap on paper and ended the prune early, so the deletable
            # candidates behind it were never attempted and the store settled permanently at
            # MAX_BACKUP_VERSIONS + (however many refuse) -- one full copy of the tool per slot.
            continue
        removed += 1


def create_backup(
    tool_id: str,
    *,
    user_request: str = "",
    modification_type: str = "multi_file_change",
) -> dict:
    """Create a versioned backup of all tool files before modification.

    Copies all tool files to .knowledge/{tool_id}/backups/v{NNN}_{timestamp}/.
    Stores SHA-256 checksums, MCP config snapshot, and install_log snapshot.

    Args:
        tool_id: The tool to back up (must be active).
        user_request: Human-readable description of what the user asked.
        modification_type: One of MODIFICATION_TYPES.

    Returns:
        {"success": True, "tool_id": str, "version": int, "timestamp": str,
         "backup_dir": str, "files_backed_up": list[str]}
    """
    _validate_tool_id(tool_id)
    entry = _require_active(tool_id)

    with _knowledge_lock():
        # --- Dedup check: skip if current files match the latest backup ---
        tool_files = _discover_tool_files(tool_id)
        current_files_info = {src.name: {"sha256": _sha256(src)} for src in tool_files}
        current_fp = _compute_state_fingerprint(current_files_info)

        latest_ver, latest_fp = _get_latest_backup_fingerprint(tool_id)
        if latest_ver is not None and latest_fp == current_fp:
            return {
                "success": True,
                "tool_id": tool_id,
                "version": latest_ver,
                "timestamp": datetime.now().strftime("%Y%m%dT%H%M%S"),
                "backup_dir": str(_backups_dir(tool_id)),
                "files_backed_up": list(current_files_info.keys()),
                "deduplicated": True,
            }

        # --- Create new backup version ---
        version = _next_backup_version(tool_id)
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        version_dir = _backups_dir(tool_id) / f"v{version:03d}_{timestamp}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy all tool files
        files_info = {}
        try:
            for src in tool_files:
                dst = version_dir / src.name
                shutil.copy2(str(src), str(dst))
                files_info[src.name] = {
                    "original_path": str(src.relative_to(TOOLS_USER_DIR.parent)),
                    "sha256": _sha256(dst),
                    "size_bytes": dst.stat().st_size,
                }
        except Exception:
            shutil.rmtree(str(version_dir), ignore_errors=True)
            raise

        # Snapshot MCP config entry
        server_key = f"user_{tool_id}"
        mcp_snapshot = {}
        raw_mcp_yaml = ""
        config = _read_mcp_config_user()
        block = _server_block(config, server_key)
        if block:
            mcp_snapshot = {server_key: block}
        config_path = mcp_config_user_path()
        if config_path.exists():
            raw_mcp_yaml = config_path.read_text()

        # Write backup metadata
        state_fp = _compute_state_fingerprint(files_info)
        meta = {
            "tool_id": tool_id,
            "backup_version": version,
            "meta_version": BACKUP_META_VERSION,
            "state_fingerprint": state_fp,
            "created_at": datetime.now().isoformat(),
            "user_request": user_request,
            "modification_type": modification_type,
            "files_backed_up": files_info,
            "mcp_config_snapshot": mcp_snapshot,
            "raw_mcp_yaml": raw_mcp_yaml,
            "install_log_snapshot": dict(entry),
        }
        _atomic_write_json(version_dir / ".backup_meta.json", meta)

        # Prune old versions
        _prune_old_backups(tool_id)

        return {
            "success": True,
            "tool_id": tool_id,
            "version": version,
            "timestamp": timestamp,
            "backup_dir": str(version_dir),
            "files_backed_up": list(files_info.keys()),
        }


def rollback_to_backup(tool_id: str, version: int | None = None, *, reason: str = "") -> dict:
    """Atomically restore tool files from a backup version.

    Two-phase commit:
      Phase 1 (Stage): Copy backup files to .rollback_tmp alongside originals
      Phase 2 (Commit): os.replace() each temp file to final location (atomic per-file)

    If version is None, uses the most recent backup.

    Args:
        tool_id: The tool to roll back.
        version: Specific version (None = latest).
        reason: Why the rollback is happening.

    Returns:
        {"success": True, "tool_id": str, "version_restored": int,
         "files_restored": list[str], "reason": str}
        plus "warning" when the restore succeeded but the modification lock could not be released,
        which is what will refuse the next begin_modification.
    """
    _validate_tool_id(tool_id)

    with _knowledge_lock():
        bdir = _backups_dir(tool_id)
        if not bdir.exists():
            raise RollbackError(f"No backups exist for tool '{tool_id}'.")

        # Find the target version directory
        if version is None:
            # Latest version
            versions = []
            for d in bdir.iterdir():
                if d.is_dir():
                    m = re.match(r"^v(\d{3})_", d.name)
                    if m:
                        versions.append((int(m.group(1)), d))
            if not versions:
                raise RollbackError(f"No backup versions found for tool '{tool_id}'.")
            versions.sort(key=lambda x: x[0])
            version, version_dir = versions[-1]
        else:
            # Specific version
            candidates = [d for d in bdir.iterdir() if d.is_dir() and d.name.startswith(f"v{version:03d}_")]
            if not candidates:
                raise RollbackError(f"Backup version {version} not found for tool '{tool_id}'.")
            version_dir = candidates[0]

        def _older_restorable() -> str:
            """Name the newest version below this one that rollback would actually accept.

            The SHA-256 branch below already says "Try an older version with
            rollback_to_backup(..., version=N)". The two metadata branches were dead ends, which
            is how the documented Manual Rollback flow -- list, then roll back to the latest --
            ends in an uncaught RollbackError with an intact backup sitting one version down.
            """
            older = [b["version"] for b in list_backups(tool_id) if b.get("restorable") and b["version"] < version]
            if not older:
                return "No older restorable backup exists; restore the tool by hand or recreate it."
            return f"Try rollback_to_backup('{tool_id}', version={max(older)}) -- that version is restorable."

        # Read and verify the backup. _backup_refusal is the one statement of what makes a version
        # unrestorable, shared with list_backups and the pruner so the three cannot disagree.
        refusal, meta = _backup_refusal(version_dir, version)
        if refusal:
            # Every ground gets the same remedy. The checksum branch used to name version-1
            # unconditionally, and R80/AZ's pruning removes intermediate versions, so that hint
            # regularly pointed at a directory that is not there -- "Backup version N not found".
            raise RollbackError(f"{refusal} {_older_restorable()}")

        files_backed_up = meta["files_backed_up"]

        # Phase 1: Stage — copy backup files to .rollback_tmp
        staged_files = []
        try:
            for filename in files_backed_up:
                backup_file = version_dir / filename
                target = TOOLS_USER_DIR / filename
                tmp_target = target.with_suffix(target.suffix + ".rollback_tmp")
                shutil.copy2(str(backup_file), str(tmp_target))
                staged_files.append((tmp_target, target))
        except Exception as e:
            # Cleanup staged files
            for tmp, _ in staged_files:
                tmp.unlink(missing_ok=True)
            raise RollbackError(f"Failed to stage rollback files: {e}") from e

        # Phase 2: Commit — atomic replace per file
        restored_files = []
        try:
            for tmp_target, final_target in staged_files:
                os.replace(str(tmp_target), str(final_target))
                restored_files.append(final_target.name)
        except Exception as e:
            raise RollbackError(
                f"Rollback partially completed ({len(restored_files)}/{len(staged_files)} files). "
                f"Error: {e}. Run _repair_modification_state('{tool_id}') to fix."
            ) from e

        # Restore the tool's mcp_config_user.yaml entry too (create_backup snapshots it as
        # mcp_config_snapshot). A rename/signature modification edits the yaml alongside the server
        # file, so reverting only files would leave a server-vs-config mismatch. Restore JUST this
        # tool's entry (siblings preserved). Best-effort — never fail an otherwise-good file rollback.
        try:
            mcp_snap = meta.get("mcp_config_snapshot") or {}
            if isinstance(mcp_snap, dict) and f"user_{tool_id}" in mcp_snap:
                _restore_mcp_config_entry(tool_id, mcp_snap[f"user_{tool_id}"])
        except Exception as e:
            print(f"[knowledge_manager] WARNING: could not restore mcp_config entry for {tool_id}: {e}")

        # Post-rollback cleanup: remove duplicate backups (best-effort)
        try:
            restored_fp = meta.get("state_fingerprint")
            if not restored_fp:
                restored_fp = _compute_state_fingerprint(files_backed_up)
            if restored_fp:
                bdir = _backups_dir(tool_id)
                for d in sorted(bdir.iterdir(), reverse=True):
                    if not d.is_dir():
                        continue
                    m = re.match(r"^v(\d{3})_", d.name)
                    if not m:
                        continue
                    other_ver = int(m.group(1))
                    if other_ver == version or other_ver == 1:
                        continue  # Never remove the restored version or v1
                    other_meta_path = d / ".backup_meta.json"
                    if not other_meta_path.exists():
                        continue
                    try:
                        other_meta = json.loads(other_meta_path.read_text())
                    except (json.JSONDecodeError, OSError) as e:
                        print(f"[knowledge_manager] WARNING: Corrupted metadata at {other_meta_path}: {e}")
                        continue
                    other_fp = other_meta.get("state_fingerprint")
                    if not other_fp:
                        other_files = other_meta.get("files_backed_up", {})
                        if other_files:
                            other_fp = _compute_state_fingerprint(other_files)
                    if other_fp == restored_fp and other_ver != version:
                        shutil.rmtree(str(d), ignore_errors=True)
        except Exception:
            pass  # Never fail the rollback due to cleanup

        lock_error = _clear_modification_lock(tool_id)  # the open modification has been undone -> a fresh begin
        out = {
            "success": True,
            "tool_id": tool_id,
            "version_restored": version,
            "files_restored": restored_files,
            "reason": reason,
        }
        if lock_error:  # the rollback DID happen; what failed is the release the next begin needs
            out["warning"] = lock_error
        return out


def list_backups(tool_id: str) -> list[dict]:
    """List all available backups for a tool.

    Returns:
        [{"version": int, "timestamp": str, "user_request": str,
          "modification_type": str, "files": list, "restorable": bool}]

        ``restorable`` is False when ``rollback_to_backup`` would refuse this version, on any of
        the grounds it refuses on -- ``.backup_meta.json`` missing or unparseable, no files
        recorded in it, a recorded file absent from the version directory, or a file that no
        longer matches its recorded SHA-256. ``refusal`` is the reason, or None. create_backup
        writes the metadata LAST, after copying, so an interrupted backup leaves the first of
        those -- a directory that looks like a backup from the outside; the later grounds are what
        a disk error or a hand-edit under ``.knowledge/`` leaves. Callers pick a version off this
        list (know_how/modify_user_mcp_tool.md, safety rules "use the EXACT version number from
        list_backups() output" and "use list_backups() to find an earlier clean version"), so the
        list has to say which ones those are. Every entry carries every key above, restorable or
        not.

        Verifying the files means hashing them: the set is the tool's own sources (worker, server,
        env yaml), a few KB each, at most MAX_BACKUP_VERSIONS versions per tool.
    """
    _validate_tool_id(tool_id)
    bdir = _backups_dir(tool_id)
    if not bdir.exists():
        return []

    backups = []
    for d in sorted(bdir.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^v(\d{3})_(.+)$", d.name)
        if not m:
            continue
        ver = int(m.group(1))
        ts = m.group(2)
        refusal, meta = _backup_refusal(d, ver)
        info = {
            "version": ver,
            "timestamp": ts,
            "dir_name": d.name,
            "user_request": meta.get("user_request", ""),
            "modification_type": meta.get("modification_type", ""),
            "files": list(meta.get("files_backed_up", {}).keys()),
            "restorable": refusal is None,
            "refusal": refusal,
        }
        if refusal and not meta:
            # The two grounds about the metadata itself: there is nothing to show, so rather than
            # let a dead version print like a backup taken without a request, say which it is.
            info["user_request"] = (
                "(metadata corrupt)"
                if (d / ".backup_meta.json").exists()
                else "(incomplete backup -- metadata never written)"
            )
        backups.append(info)

    return backups


# ---------------------------------------------------------------------------
# Modification orchestration (three-phase API for STCoscientist)
# ---------------------------------------------------------------------------


def _modification_lock_path(tool_id: str) -> Path:
    return _knowledge_tool_dir(tool_id) / ".modification.lock"


def _modification_in_progress(tool_id: str) -> bool:
    """Whether a modification of ``tool_id`` is already open (a begin with no matching finalize/rollback)."""
    return _modification_lock_path(tool_id).exists()


def _set_modification_lock(tool_id: str, user_request: str) -> None:
    p = _modification_lock_path(tool_id)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"started_at": datetime.now().isoformat(), "user_request": user_request}))
    except OSError as e:  # a lock we couldn't write just means no double-begin guard -- never block the op
        logger.warning("Could not set modification lock for %s: %s", tool_id, e)


def _read_modification_lock(tool_id: str) -> dict:
    """What the open modification recorded about itself, or {} if the lock says nothing readable.

    The reader half of _set_modification_lock, which has always written started_at and user_request
    with nothing on the other end. Never raises: a lock whose contents we cannot parse is still a
    lock, and begin_modification must refuse either way.
    """
    try:
        record = json.loads(_modification_lock_path(tool_id).read_text())
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _format_lock_age(seconds: float) -> str:
    """Coarse age of an open modification. The question it answers is stale-or-live, not how long."""
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    if seconds < 90 * 60:
        return f"{seconds // 60}m"
    if seconds < 48 * 3600:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _describe_open_modification(tool_id: str) -> str:
    """A parenthetical for the refusal: when the modification holding the lock began, and what it asked.

    modify_user_mcp_tool.md's remedy for a lock left behind by a dead session is rollback_to_backup,
    which undoes whatever that session had written -- the right move against a dead session and a
    destructive one against a session still running. Age and request are what separate the two, and
    both were already on disk. Returns "" when the lock records nothing usable; the refusal does not
    depend on this, and never invents an age from a stamp datetime could not read.
    """
    record = _read_modification_lock(tool_id)
    parts = []

    started = record.get("started_at")
    if isinstance(started, str) and started.strip():
        started = started.strip()
        try:
            stamp = datetime.fromisoformat(started)
        except ValueError:
            parts.append(f"started {started}")
        else:
            # This file writes naive stamps, but the lock is a file anyone can have written: an
            # aware stamp (ISO with an offset) is readable, and naive-minus-aware is a TypeError
            # the ValueError arm never catches -- crashing the refusal this string decorates. So
            # "now" is taken in the stamp's own frame; a negative age is clock skew, not an age.
            now = datetime.now(stamp.tzinfo) if stamp.tzinfo is not None else datetime.now()
            age = (now - stamp).total_seconds()
            parts.append(f"started {started}" if age < 0 else f"started {started}, {_format_lock_age(age)} ago")

    request = record.get("user_request")
    if isinstance(request, str) and request.strip():
        request = " ".join(request.split())  # free text from the user; a newline must not split the message
        parts.append('for: "{}"'.format(request if len(request) <= 120 else request[:117] + "..."))

    return f" ({'; '.join(parts)})" if parts else ""


def _clear_modification_lock(tool_id: str) -> str | None:
    """Release the modification lock. Returns None on success, or why the lock is still there.

    The asymmetry with _set_modification_lock above is deliberate and worth keeping: a lock we
    could not WRITE costs us only the double-begin guard, while a lock we cannot DELETE costs the
    tool -- _modification_in_progress stays true, so every later begin_modification is refused.

    What was wrong was the silence. Both callers (finalize_modification, rollback_to_backup) return
    success either way, and _repair_modification_state handles .rollback_tmp files and unrecorded
    backup dirs, not this file -- so a failed unlink left an unmodifiable tool with no trace, and
    the refusal named two remedies that both come back through here. The reason now travels back
    for the caller to hand on, and names the file, because deleting it is the way out.
    """
    path = _modification_lock_path(tool_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not clear modification lock for %s (%s): %s", tool_id, path, e)
        return (
            f"The modification lock {path} could not be removed ({e}), so begin_modification will "
            f"refuse to modify '{tool_id}' until that file is deleted."
        )
    return None


def begin_modification(
    tool_id: str,
    *,
    user_request: str,
    modification_type: str = "multi_file_change",
) -> dict:
    """Phase 1: Pre-flight checks + create backup.

    Verifies tool exists, is active, not benchmarking, tool_creation_enabled.
    If no knowledge exists, auto-calls bootstrap_knowledge().
    Creates a versioned backup.

    Args:
        tool_id: The tool to modify.
        user_request: Description of what the user asked.
        modification_type: One of MODIFICATION_TYPES.

    Returns:
        {"success": True, "tool_id": str, "backup_version": int, "knowledge": dict}
    """
    _validate_tool_id(tool_id)

    # Check tool_creation_enabled
    try:
        from spatialomicsgym.config import default_config

        if not default_config.tool_creation_enabled:
            raise KnowledgeStateError("tool_creation_enabled is False. Cannot modify tools.")
        if default_config.benchmarking_enabled:
            raise KnowledgeStateError("Cannot modify tools while benchmarking is active.")
    except ImportError:
        pass  # Config not available, allow

    # Verify active
    _require_active(tool_id)

    # Repair any interrupted state -- and read what it answers. A staged .rollback_tmp the repair
    # could not commit leaves the tool part one version and part another; create_backup below would
    # record that mixture as v{N}, and v{N} is exactly the version the caller is told to roll back
    # to when the modification fails. Refusing costs nothing recoverable: the repair deliberately
    # keeps the staged copy on disk, so the next call finishes the rollback and this begin proceeds.
    repair = _repair_modification_state(tool_id) or {}
    if repair.get("failed"):
        raise ModificationError(
            f"Cannot modify '{tool_id}': an interrupted rollback could not be completed for "
            f"{repair['failed']}, so its files are part one version and part another. Backing up "
            f"now would record that mixture as the version a failed modification rolls back to. "
            f"The staged copies are still in {TOOLS_USER_DIR} as *.rollback_tmp -- clear whatever "
            f"blocks replacing the originals (permissions, a lock, a full disk), then run "
            f"_repair_modification_state('{tool_id}') or just begin again."
        )

    # Refuse a second begin while a modification is already open (documented in
    # modify_user_mcp_tool.md, previously unenforced). Without this, a re-entry -- a common LLM fix-loop
    # retry after a failed test -- would snapshot the already-EDITED, untested state as a NEW backup
    # version, and a later rollback to that returned version restores the broken state. The open
    # modification must be finalized or rolled back first (both clear the lock).
    if _modification_in_progress(tool_id):
        raise ModificationError(
            f"A modification of '{tool_id}' is already in progress{_describe_open_modification(tool_id)}. "
            f"Finalize it (finalize_modification) or undo it (rollback_to_backup) before starting "
            f"another. Both release the lock at "
            f"{_modification_lock_path(tool_id)}; if neither can (each warns and reports why), delete "
            f"that file to clear this."
        )

    # Bootstrap knowledge if needed
    if not knowledge_exists(tool_id):
        try:
            bootstrap_knowledge(tool_id)
        except Exception as e:
            print(f"[knowledge_manager] WARNING: bootstrap_knowledge failed for {tool_id}: {e}")

    knowledge = read_knowledge(tool_id)

    # Create backup, THEN mark the modification open (after the backup so a backup failure doesn't
    # strand a lock that blocks all future begins).
    backup = create_backup(tool_id, user_request=user_request, modification_type=modification_type)
    _set_modification_lock(tool_id, user_request)

    return {
        "success": True,
        "tool_id": tool_id,
        "backup_version": backup["version"],
        "knowledge": knowledge,
    }


# ---------------------------------------------------------------------------
# Import extraction helpers for imports_scan test
# ---------------------------------------------------------------------------

# R base packages shipped with r-base (should not be flagged as missing)
_R_BASE_PACKAGES = frozenset(
    {
        "base",
        "compiler",
        "datasets",
        "grDevices",
        "graphics",
        "grid",
        "methods",
        "parallel",
        "splines",
        "stats",
        "stats4",
        "tcltk",
        "tools",
        "utils",
    }
)

# Regex to validate that a package name is a valid Python identifier (used by imports_scan)
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _extract_third_party_imports(source_code: str) -> set[str]:
    """AST-parse Python source and return top-level package names of third-party imports.

    Walks ALL nodes (including function-scoped imports). Filters out:
    - stdlib modules (via sys.stdlib_module_names, Python 3.10+)
    - relative imports (from .foo import bar)
    - the tool's own files (handled by caller)

    Returns set of top-level package names, e.g. {"sklearn", "umap", "matplotlib"}.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()  # syntax test will catch this separately

    stdlib = getattr(sys, "stdlib_module_names", set()) | {
        "typing_extensions",
        "pkg_resources",
        "setuptools",
        "pip",
    }
    third_party: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in stdlib:
                    third_party.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # skip relative imports
                top = node.module.split(".")[0]
                if top not in stdlib:
                    third_party.add(top)

    return third_party


def _extract_r_library_calls(source_code: str) -> set[str]:
    """Extract R package names from library() and require() calls.

    Handles patterns:
      library(PackageName)
      library("PackageName")
      library('PackageName')
      require(PackageName)
    inside or outside suppressPackageStartupMessages({...}).

    Filters out R base packages (methods, stats, utils, etc.).

    Returns set of package names, e.g. {"Seurat", "Matrix", "jsonlite"}.
    """
    pattern = r"""(?:library|require)\s*\(\s*["']?([A-Za-z][A-Za-z0-9._]*)["']?\s*\)"""
    packages = set(re.findall(pattern, source_code))
    return packages - _R_BASE_PACKAGES


# Known mismatches between Python import names and pip package names
_IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "Bio": "biopython",
    "attr": "attrs",
}


def _auto_install_missing_imports(
    tool_id: str,
    env_name: str,
    language: str = "python",
) -> dict:
    """Scan worker+server files for imports, install any missing in conda env.

    Called by complete_modification() before running the test suite so that
    imports_scan doesn't fail on fixable dependency issues.

    Returns {"installed": [...], "still_missing": [...], "detail": str}.
    """
    is_r = language.lower() == "r"
    worker_ext = ".R" if is_r else ".py"
    worker_path = TOOLS_USER_DIR / f"{tool_id}_worker{worker_ext}"
    server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"

    all_third_party: set[str] = set()
    for fpath in [worker_path, server_path]:
        if not fpath.exists():
            continue
        src = fpath.read_text()
        if fpath.suffix == ".py":
            all_third_party |= _extract_third_party_imports(src)
        elif fpath.suffix == ".R":
            all_third_party |= _extract_r_library_calls(src)

    entry = _find_log_entry(tool_id)
    module = (entry.get("module") or entry.get("package", tool_id)) if entry else tool_id
    exclude = {"worker_utils", "base_mcp", module, tool_id}
    third_party = sorted(all_third_party - exclude)

    if not third_party:
        return {"installed": [], "still_missing": [], "detail": "No third-party imports found"}

    missing = []
    for mod in third_party:
        r = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            missing.append(mod)

    if not missing:
        return {"installed": [], "still_missing": [], "detail": "All imports available"}

    pip_names = [_IMPORT_TO_PIP.get(m, m) for m in missing]
    subprocess.run(
        ["conda", "run", "-n", env_name, "pip", "install"] + pip_names,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    still_missing = []
    installed = []
    for mod in missing:
        r = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", f"import {mod}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            still_missing.append(mod)
        else:
            installed.append(mod)

    detail = f"Installed: {installed}" if installed else ""
    if still_missing:
        detail += f" Still missing: {still_missing}"
    return {"installed": installed, "still_missing": still_missing, "detail": detail.strip()}


def _test_did_not_run(result: dict) -> bool:
    """Whether a ``_run_test_suite`` entry describes a test that never executed.

    The suite says this two ways. A test named in ``skip_tests`` comes back ``passed=True`` with
    detail ``"SKIPPED"`` -- the documented contract of that parameter. ``real_data`` with no
    benchmark fixture on the host comes back ``passed=False`` + ``skipped=True``, so that a
    modification cannot commit on a real-data check that never happened.

    Both mean "no evidence", and every scorer has to read both. Reading only the second counted a
    ``skip_tests`` entry as a pass: it inflates ``tests_passed``/``tests_total``, it can carry
    ``success=True`` on a suite where nothing executed, and on ``real_data`` it sets
    ``real_data_validated=True`` -- the "5/5 passed, worker never ran" claim, arriving through the
    other encoding. Reading only the first filed a fixture-less host as a failure and called every
    tool DEGRADED. One predicate, so the two readers cannot drift apart again.
    """
    if result.get("skipped"):
        return True
    return bool(result.get("passed")) and "SKIP" in result.get("detail", "")


def _resolve_real_test_dataset() -> str | None:
    """A real dataset this host can hand a worker, or ``None`` when it has none.

    Two rungs. The repo's mini benchmark fixtures first: small, deterministic, and the pair this
    suite has always preferred. Then the spatial-library catalogue.

    Rung two used to name ``/workspace/data/spatial_library/registry.json`` and read ``h5ad_path``
    off the first record. Neither half reaches a deployment. That path is the one
    ``transcriptomics_skills`` calls "missing on most installs"; the documented pointer is
    ``SOG_SPATIAL_LIBRARY_REGISTRY``, and the catalogue snapshot that ships inside the package is
    found with no pointer at all. ``h5ad_path`` is carried by none of the 63 catalogued samples --
    they record ``spatial_dir``, the directory the ``.h5ad`` sits inside.

    Rung one does not survive a clone either: ``benchmarks/benchmark_data/mini/*.h5ad`` is untracked
    and ``benchmarks/`` is pruned from the wheel. So rung two carries every host but the one that
    built those fixtures, and while it named only the legacy path the real-data test -- the only one
    of the six that runs the worker, and the one ``real_data_validated`` is computed from -- could
    not fire anywhere else, even with a catalogued dataset on disk and the pointer set.

    Resolution is delegated rather than reimplemented. ``_load_spatial_registry`` already reads the
    pointer, treats a blank value as unset, accepts both catalogue shapes and survives a malformed
    file; ``_pick_dataset_h5ad`` already prefers a canonical basename over ``glob`` order, which is
    filesystem order and so answered differently on different machines. A second copy of that search
    here would be a worse copy.

    Only a path present on disk is returned. The catalogue is a record of another machine's
    filesystem, so a catalogued-but-absent dataset keeps degrading to "no test data on this host":
    handing the worker a path that is not there would trade a clear skip for a confusing crash.
    """
    for candidate in (
        "benchmarks/benchmark_data/mini/mini_visium_clustering.h5ad",
        "benchmarks/benchmark_data/mini/mini_merfish_clustering.h5ad",
    ):
        full_path = TOOLS_USER_DIR.parent / candidate
        if full_path.exists():
            return str(full_path)

    try:
        from spatialomicsgym.tool.transcriptomics_skills import _load_spatial_registry, _pick_dataset_h5ad
    except Exception as exc:
        logger.warning("spatial dataset catalogue unavailable, real-data test stays inconclusive: %s", exc)
        return None

    for entry in _load_spatial_registry():
        if not isinstance(entry, dict):
            continue
        h5ad = entry.get("h5ad_path") or ""
        spatial_dir = entry.get("spatial_dir") or ""
        # Guarded: _pick_dataset_h5ad("") globs "*.h5ad" against the process CWD, which would make
        # the answer depend on where the agent happened to be started.
        if not h5ad and spatial_dir:
            h5ad = _pick_dataset_h5ad(spatial_dir)[0] or ""
        if h5ad and os.path.exists(h5ad):
            return h5ad
    return None


def _run_test_suite(
    tool_id: str,
    *,
    module: str,
    task_type: str = "spatial_clustering",
    test_timeout: int = 300,
    skip_tests: frozenset[str] = frozenset(),
    language: str = "python",
    module_language: str | None = None,
) -> list[dict]:
    """Run the 6-test validation suite on a tool (standalone, no backup context).

    Tests:
      1. import       — verify base package imports in conda env
      2. syntax       — ast.parse() (Python) or Rscript parse() (R) on tool files
      3. imports_scan — verify ALL third-party imports available in conda env
      4. dry_run      — run worker with nonexistent input, expect error JSON
      5. config       — verify MCP config entry exists
      6. real_data    — run with actual test data, verify output

    Args:
        tool_id: The tool to test (must match TOOL_ID_RE).
        module: Python module or R package name for import testing.
        task_type: Task type for output validation heuristics.
        test_timeout: Timeout in seconds for real data test.
        skip_tests: Test names to skip (return passed=True, detail="SKIPPED").
        language: "python" or "R" — adapts import, syntax, and run commands.
        module_language: the language ``module`` itself is in, when it differs from the worker's --
            a Python worker wrapping an R package records ``wrapped_language="R"``.

    Returns:
        [{"test": str, "passed": bool, "detail": str}, ...]
    """
    _validate_tool_id(tool_id)
    env_name = f"user_{tool_id}"
    is_r = language.lower() == "r"
    results = []

    # Determine file paths
    if is_r:
        worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.R"
        server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"  # MCP server is always Python
    else:
        worker_path = TOOLS_USER_DIR / f"{tool_id}_worker.py"
        server_path = TOOLS_USER_DIR / f"{tool_id}_mcp_server.py"

    # Test 1: Import test
    if not module or not module.strip():
        results.append({"test": "import", "passed": False, "detail": "module parameter is empty"})
    elif "import" in skip_tests:
        results.append({"test": "import", "passed": True, "detail": "SKIPPED"})
    else:
        try:
            # A hybrid tool -- a Python worker driving an R package (``wrapped_language="R"``) --
            # names an R package as its module, and ``python -c "import <R package>"`` failed for
            # every one of them, so a working tool read BROKEN (hunt 2026-09-30, u13k1-knowhow-5).
            if is_r or str(module_language or "").lower() == "r":
                cmd = ["conda", "run", "-n", env_name, "Rscript", "-e", f"library({module}); cat('OK\\n')"]
            else:
                cmd = ["conda", "run", "-n", env_name, "python", "-c", f"import {module}; print('OK')"]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            passed = "OK" in r.stdout
            results.append({"test": "import", "passed": passed, "detail": "OK" if passed else r.stderr[:200]})
        except Exception as e:
            results.append({"test": "import", "passed": False, "detail": str(e)[:200]})

    # Test 2: Syntax check
    if "syntax" in skip_tests:
        results.append({"test": "syntax", "passed": True, "detail": "SKIPPED"})
    else:
        try:
            if worker_path.exists():
                if is_r:
                    r = subprocess.run(
                        ["conda", "run", "-n", env_name, "Rscript", "-e", f"parse(file='{worker_path}')"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if r.returncode != 0:
                        raise SyntaxError(r.stderr[:200])
                else:
                    ast.parse(worker_path.read_text())
            if server_path.exists():
                ast.parse(server_path.read_text())  # MCP server is always Python
            results.append({"test": "syntax", "passed": True, "detail": "OK"})
        except (SyntaxError, OSError, UnicodeDecodeError) as e:
            results.append({"test": "syntax", "passed": False, "detail": str(e)[:200]})

    # Test 3: Import scan — verify all third-party imports are available in conda env
    if "imports_scan" in skip_tests:
        results.append({"test": "imports_scan", "passed": True, "detail": "SKIPPED"})
    else:
        try:
            if is_r:
                # R tools: extract library()/require() calls from worker .R file
                all_imports: set[str] = set()
                if worker_path.exists():
                    all_imports |= _extract_r_library_calls(worker_path.read_text())
                # MCP server is Python — scan it too
                if server_path.exists():
                    py_imports = _extract_third_party_imports(server_path.read_text())
                    py_imports -= {module, tool_id, "base_mcp", "worker_utils"}
                    all_imports |= py_imports

                # Filter out the base R module already tested in Test 1
                all_imports.discard(module)

                if not all_imports:
                    results.append({"test": "imports_scan", "passed": True, "detail": "No extra imports to check"})
                else:
                    # Split R vs Python packages for separate checks
                    r_pkgs = all_imports & _extract_r_library_calls(
                        worker_path.read_text() if worker_path.exists() else ""
                    )
                    py_pkgs = all_imports - r_pkgs

                    missing: list[str] = []

                    # Check R packages via requireNamespace()
                    if r_pkgs:
                        r_check = "; ".join(
                            f'if (!requireNamespace("{pkg}", quietly=TRUE)) cat("{pkg},")' for pkg in sorted(r_pkgs)
                        )
                        r_cmd = f'{r_check} cat("\\n")'
                        r_result = subprocess.run(
                            ["conda", "run", "-n", env_name, "Rscript", "-e", r_cmd],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        r_out = r_result.stdout.strip().rstrip(",")
                        if r_out:
                            missing.extend(f"R:{m}" for m in r_out.split(",") if m.strip())

                    # Check Python packages via importlib.util.find_spec()
                    if py_pkgs:
                        sanitized = sorted(pkg for pkg in py_pkgs if _IDENT_RE.match(pkg))
                        if sanitized:
                            check_lines = ["import importlib.util", "missing = []"]
                            for pkg in sanitized:
                                check_lines.append(
                                    f"missing.append('{pkg}') if importlib.util.find_spec('{pkg}') is None else None"
                                )
                            check_lines.append("print(','.join(missing) if missing else 'OK')")
                            py_result = subprocess.run(
                                ["conda", "run", "-n", env_name, "python", "-c", "; ".join(check_lines)],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            py_out = py_result.stdout.strip().split("\n")[-1] if py_result.stdout.strip() else ""
                            if py_out and py_out != "OK":
                                missing.extend(f"py:{m}" for m in py_out.split(",") if m.strip())

                    if missing:
                        results.append(
                            {
                                "test": "imports_scan",
                                "passed": False,
                                "detail": f"Missing in {env_name}: {', '.join(missing)}"[:200],
                            }
                        )
                    else:
                        results.append(
                            {
                                "test": "imports_scan",
                                "passed": True,
                                "detail": f"All {len(all_imports)} imports available",
                            }
                        )
            else:
                # Python tools: extract import/from-import via AST
                all_imports = set()
                for fpath in [worker_path, server_path]:
                    if fpath.exists():
                        all_imports |= _extract_third_party_imports(fpath.read_text())

                own_modules = {module, tool_id, "base_mcp", "worker_utils"}
                all_imports -= own_modules

                if not all_imports:
                    results.append(
                        {
                            "test": "imports_scan",
                            "passed": True,
                            "detail": "No third-party imports to check",
                        }
                    )
                else:
                    sanitized = sorted(pkg for pkg in all_imports if _IDENT_RE.match(pkg))

                    if not sanitized:
                        results.append(
                            {
                                "test": "imports_scan",
                                "passed": True,
                                "detail": f"No valid identifiers to check (skipped: {all_imports - set(sanitized)})",
                            }
                        )
                    else:
                        check_lines = ["import importlib.util", "missing = []"]
                        for pkg in sanitized:
                            check_lines.append(
                                f"missing.append('{pkg}') if importlib.util.find_spec('{pkg}') is None else None"
                            )
                        check_lines.append("print(','.join(missing) if missing else 'OK')")
                        check_cmd = "; ".join(check_lines)
                        r = subprocess.run(
                            ["conda", "run", "-n", env_name, "python", "-c", check_cmd],
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        output = r.stdout.strip().split("\n")[-1] if r.stdout.strip() else ""
                        if output == "OK":
                            results.append(
                                {
                                    "test": "imports_scan",
                                    "passed": True,
                                    "detail": f"All {len(sanitized)} third-party imports available",
                                }
                            )
                        elif output:
                            missing = [m for m in output.split(",") if m.strip()]
                            results.append(
                                {
                                    "test": "imports_scan",
                                    "passed": False,
                                    "detail": f"Missing in {env_name}: {', '.join(missing)}. "
                                    f"Install: conda run -n {env_name} pip install {' '.join(missing)}"[:200],
                                }
                            )
                        else:
                            results.append(
                                {
                                    "test": "imports_scan",
                                    "passed": False,
                                    "detail": f"Import scan failed: {r.stderr[:200]}",
                                }
                            )
        except Exception as e:
            results.append({"test": "imports_scan", "passed": False, "detail": str(e)[:200]})

    # Test 4: Worker dry-run
    if "dry_run" in skip_tests:
        results.append({"test": "dry_run", "passed": True, "detail": "SKIPPED"})
    else:
        try:
            if is_r:
                run_cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "Rscript",
                    str(worker_path),
                    "--input",
                    "/nonexistent/file.h5ad",
                    "--output-dir",
                    f"/tmp/test_{tool_id}_mod",
                ]
            else:
                run_cmd = [
                    "conda",
                    "run",
                    "-n",
                    env_name,
                    "python",
                    str(worker_path),
                    "--input",
                    "/nonexistent/file.h5ad",
                    "--output-dir",
                    f"/tmp/test_{tool_id}_mod",
                ]
            r = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60)
            try:
                out = json.loads(r.stdout.strip().split("\n")[-1])
                if out.get("status") == "error":
                    results.append(
                        {"test": "dry_run", "passed": True, "detail": f"Expected error: {out['error'][:80]}"}
                    )
                else:
                    results.append(
                        {"test": "dry_run", "passed": False, "detail": f"Expected error, got: {out.get('status')}"}
                    )
            except (json.JSONDecodeError, IndexError):
                results.append({"test": "dry_run", "passed": False, "detail": f"Invalid JSON: {r.stdout[:100]}"})
        except Exception as e:
            results.append({"test": "dry_run", "passed": False, "detail": str(e)[:200]})

    # Test 5: MCP config validation
    if "config" in skip_tests:
        results.append({"test": "config", "passed": True, "detail": "SKIPPED"})
    else:
        try:
            config = _read_mcp_config_user()
            server_key = f"user_{tool_id}"
            block = _server_block(config, server_key)
            if not block:
                results.append({"test": "config", "passed": False, "detail": f"{server_key} not in config"})
            elif not block.get("enabled", True):
                # The `enabled` flag is what mcp_config_merger line 214 reads to decide whether the
                # agent is told this server exists; a disabled block is skipped, so calling the tool
                # resolves to nothing -- health_check_user_mcp_tools.md's own definition of BROKEN
                # (line 34), and the `config` row is documented (line 98) to cover the flag. Absent
                # key means served, so default to True and only fail on an explicit false.
                # Front-load the flag name: the recommendation truncates this detail at 100 chars.
                detail = f"enabled: false on {server_key} -- the merger skips it, so the tool cannot be called"
                results.append({"test": "config", "passed": False, "detail": detail})
            else:
                # Also validate spatialomicsgym_name matches actual @mcp.tool() function
                bn_check = _validate_spatialomicsgym_name(tool_id)
                if bn_check["ok"]:
                    results.append(
                        {
                            "test": "config",
                            "passed": True,
                            "detail": f"{server_key} present, spatialomicsgym_name valid",
                        }
                    )
                else:
                    results.append({"test": "config", "passed": False, "detail": bn_check["detail"]})
        except Exception as e:
            results.append({"test": "config", "passed": False, "detail": str(e)[:200]})

    # Test 6: Real data test
    if "real_data" in skip_tests:
        results.append({"test": "real_data", "passed": True, "detail": "SKIPPED"})
    else:
        test_data = _resolve_real_test_dataset()

        if test_data and os.path.exists(test_data):
            test_out = f"/tmp/test_{tool_id}_hc_real"
            os.makedirs(test_out, exist_ok=True)
            try:
                if is_r:
                    run_cmd = [
                        "conda",
                        "run",
                        "-n",
                        env_name,
                        "Rscript",
                        str(worker_path),
                        "--input",
                        test_data,
                        "--output-dir",
                        test_out,
                    ]
                else:
                    run_cmd = [
                        "conda",
                        "run",
                        "-n",
                        env_name,
                        "python",
                        str(worker_path),
                        "--input",
                        test_data,
                        "--output-dir",
                        test_out,
                    ]
                r = subprocess.run(run_cmd, capture_output=True, text=True, timeout=test_timeout)
                try:
                    out = json.loads(r.stdout.strip().split("\n")[-1])
                    if out.get("status") == "ok":
                        output_files = out.get("output_files", {})
                        has_results = any(os.path.exists(p) and os.path.getsize(p) > 100 for p in output_files.values())
                        if has_results:
                            results.append({"test": "real_data", "passed": True, "detail": "Output files produced"})
                        else:
                            results.append({"test": "real_data", "passed": False, "detail": "No valid output files"})
                    else:
                        results.append(
                            {"test": "real_data", "passed": False, "detail": f"Error: {out.get('error', '')[:200]}"}
                        )
                except (json.JSONDecodeError, IndexError):
                    results.append({"test": "real_data", "passed": False, "detail": f"Invalid JSON: {r.stdout[:100]}"})
            except subprocess.TimeoutExpired:
                results.append({"test": "real_data", "passed": False, "detail": f"Timeout ({test_timeout}s)"})
            except Exception as e:
                results.append({"test": "real_data", "passed": False, "detail": str(e)[:200]})
            finally:
                shutil.rmtree(test_out, ignore_errors=True)
        else:
            # No test data on this host (e.g. a pip-installed deployment -- benchmark_data is pruned from
            # the wheel). Mark INCONCLUSIVE, not a silent pass: counting it as passed turned the strongest
            # gate into a rubber stamp (a modification could commit without the worker ever running on real
            # data). `skipped` excludes it from complete_modification's pass count while surfacing it.
            results.append(
                {
                    "test": "real_data",
                    "passed": False,
                    "skipped": True,
                    "detail": "INCONCLUSIVE — no test data on this host; the worker was NOT run on real data",
                }
            )

    return results


def complete_modification(
    tool_id: str,
    *,
    backup_version: int,
    module: str,
    task_type: str = "spatial_clustering",
    test_timeout: int = 300,
) -> dict:
    """Phase 2: Run 6-test suite on current file state (after STCoscientist made changes).

    Delegates to _run_test_suite() for the actual testing.

    Args:
        tool_id: The tool to test.
        backup_version: The backup version (for tracking).
        module: Python module name for import testing (e.g., "sopa").
        task_type: Task type for output validation heuristics.
        test_timeout: Timeout in seconds for real data test.

    Returns:
        {"success": bool, "tests_passed": int, "tests_total": int,
         "results": [{"test": str, "passed": bool, "detail": str}],
         "backup_version": int}
    """
    _validate_tool_id(tool_id)
    entry = _find_log_entry(tool_id)
    language = entry.get("language", "python") if entry else "python"
    env_name = entry.get("env_name", f"user_{tool_id}") if entry else f"user_{tool_id}"

    # Never pip-install into / execute a test suite against a PROTECTED env. A user tool's env must be
    # a per-tool ``user_*`` env; a stray/mis-recorded ``env_name`` of base/spatialomicsgym_env/e1/
    # sog_reproduce would otherwise let a modification mutate a shared env (trash_manager already guards
    # this; this path did not).
    if env_name in PROTECTED_ENVS:
        raise ModificationError(
            f"Refusing to modify/test tool '{tool_id}' in PROTECTED env '{env_name}'. "
            f"A user tool must live in a per-tool 'user_*' env, never a shared/base env."
        )

    # Auto-install missing dependencies before testing (code-level enforcement)
    try:
        install_result = _auto_install_missing_imports(tool_id, env_name, language)
        if install_result["installed"]:
            logger.info("Auto-installed missing deps for %s: %s", tool_id, install_result["installed"])
    except Exception as e:
        logger.warning("Auto-install check failed for %s (non-blocking): %s", tool_id, e)

    results = _run_test_suite(
        tool_id,
        module=module,
        task_type=task_type,
        test_timeout=test_timeout,
        language=language,
        module_language=(entry or {}).get("wrapped_language"),
    )
    # Score only the tests that actually ran: a test that did not run (real_data with no fixtures on
    # this host, or anything named in skip_tests) is neither a pass nor a fail. Both encodings, via
    # the one predicate health_check_tools reads -- the skip_tests shape carries passed=True, so
    # scoring it counted a test that never executed as evidence that the modification survived.
    # Require at least one real test to have run, so an all-skipped suite can't rubber-stamp
    # `success=True`.
    scored = [r for r in results if not _test_did_not_run(r)]
    passed = sum(1 for r in scored if r["passed"])
    real_data_validated = not any(r.get("test") == "real_data" and _test_did_not_run(r) for r in results)
    return {
        "success": len(scored) > 0 and passed == len(scored),
        "tests_passed": passed,
        "tests_total": len(scored),
        "real_data_validated": real_data_validated,
        "results": results,
        "backup_version": backup_version,
    }


def finalize_modification(
    tool_id: str,
    *,
    backup_version: int,
    user_request: str,
    modification_type: str,
    files_changed: list[str],
    test_results: dict,
) -> dict:
    """Phase 3a (success path): Record modification in modification log.

    Updates install_log with modification_count and last_modified_at.
    Appends to modification_log.json in the knowledge directory.

    Args:
        tool_id: The tool that was modified.
        backup_version: The backup version created before modification.
        user_request: What the user asked for.
        modification_type: Classification of the change.
        files_changed: List of filenames that were modified.
        test_results: Results from complete_modification().

    Returns:
        {"success": True, "mod_id": str}
        plus "warning" when the modification was recorded but the lock could not be released,
        which is what will refuse the next begin_modification.
    """
    _validate_tool_id(tool_id)

    if not test_results.get("success"):
        return {"success": False, "error": "Cannot finalize: tests did not pass", "tool_id": tool_id}

    kdir = _knowledge_tool_dir(tool_id)
    kdir.mkdir(parents=True, exist_ok=True)

    # Read or create modification log
    mlog_path = kdir / "modification_log.json"
    if mlog_path.exists():
        try:
            mlog = json.loads(mlog_path.read_text())
        except (json.JSONDecodeError, OSError):
            mlog = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "tool_id": tool_id, "modifications": []}
    else:
        mlog = {"schema_version": KNOWLEDGE_SCHEMA_VERSION, "tool_id": tool_id, "modifications": []}

    mod_count = len(mlog.get("modifications", [])) + 1
    mod_id = f"mod_{mod_count:03d}"
    now = datetime.now().isoformat()

    mlog.setdefault("modifications", []).append(
        {
            "mod_id": mod_id,
            "timestamp": now,
            "type": modification_type,
            "user_request": user_request,
            "files_changed": files_changed,
            "backup_version": backup_version,
            # `total` is the number of tests that RAN: complete_modification excludes a skipped
            # test from both sides of the fraction, so a host with no benchmark fixture stores the
            # skipped real_data case as 5/5. Carry the qualifier next to the numbers -- without it
            # the durable record re-makes the "5/5 passed" claim that the live layer no longer
            # makes, and this record is what "what changed in this tool" is answered from.
            "test_results": {
                "passed": test_results.get("tests_passed", 0),
                "total": test_results.get("tests_total", 0),
                "real_data_validated": test_results.get("real_data_validated"),
            },
            "rollback_available": True,
        }
    )

    _atomic_write_json(mlog_path, mlog)

    # Update install_log. Non-fatal -- the modification really was made and modification_log.json
    # above is the durable record of it -- but not silent: modification_count and last_modified_at
    # are what "how often has this tool changed" is answered from (health_check's tool info, and
    # health_check_user_mcp_tools.md's per-tool line), and a swallowed failure leaves them frozen
    # at their old values with the caller told the finalize succeeded.
    try:
        log_error = _update_log_entry(
            tool_id,
            {
                "modification_count": mod_count,
                "last_modified_at": now,
            },
        )
    except Exception as e:
        log_error = f"install_log could not be updated for '{tool_id}' ({e})"
    if log_error:
        logger.warning("finalize_modification: %s", log_error)

    # Re-validate spatialomicsgym_name after modification (function name may have changed)
    bn_check = _validate_spatialomicsgym_name(tool_id)
    if not bn_check["ok"] and bn_check["actual_name"]:
        logger.warning(
            "finalize_modification: spatialomicsgym_name mismatch detected for %s. Auto-fixing YAML to '%s'.",
            tool_id,
            bn_check["actual_name"],
        )
        _fix_spatialomicsgym_name(tool_id, bn_check["actual_name"])

    lock_error = _clear_modification_lock(tool_id)  # modification complete -> a fresh begin is allowed again
    result = {"success": True, "mod_id": mod_id}
    warnings = [w for w in (log_error, lock_error) if w]
    if warnings:  # the modification IS recorded; say what did not happen around it
        result["warning"] = " ".join(warnings)
    return result


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------


def _repair_modification_state(tool_id: str) -> dict:
    """Repair inconsistent state after a crash during modification.

    Detects and handles:
    1. .rollback_tmp files present → complete the rollback (Phase 2 interrupted)
    2. Backup dir without .backup_meta.json → remove incomplete backup

    Both are cleared in one call, and every incomplete backup is attempted rather than the first.
    begin_modification is the only caller and it runs this once, so what this reports is the only
    account of the state it found.

    A staged file that cannot be committed is left staged instead of deleted: it is a verified
    copy of the backup, so keeping it means the next call can finish the job. Its name goes in
    ``failed``, which begin_modification refuses over, and a call that restored nothing does not
    claim the tool was clean. An incomplete backup directory that cannot be *removed* is a weaker
    problem -- create_backup mints a fresh version, so it blocks no modification -- and is reported
    as ``incomplete_backup_not_removed`` rather than as a removal that did not happen.

    Returns:
        {"repaired": bool, "action": str, "details": str, "failed": list[str]}
    """
    actions: list[str] = []
    details: list[str] = []
    failed: list[str] = []
    stuck_backups: list[str] = []

    # Complete an interrupted rollback (Phase 2 of rollback_to_backup)
    restored = []
    for tmp in sorted(TOOLS_USER_DIR.glob(f"{tool_id}_*.rollback_tmp")):
        # Derive original filename: sopa_worker.py.rollback_tmp → sopa_worker.py
        original_name = tmp.name.replace(".rollback_tmp", "")
        try:
            os.replace(str(tmp), str(TOOLS_USER_DIR / original_name))
            restored.append(original_name)
        except Exception:
            failed.append(original_name)
    if restored:
        actions.append("completed_rollback")
        details.append(f"Completed interrupted rollback for {tool_id}: {restored}")

    # Remove every backup directory create_backup never finished recording
    bdir = _backups_dir(tool_id)
    if bdir.exists():
        for d in sorted(bdir.iterdir()):
            if d.is_dir() and not (d / ".backup_meta.json").exists():
                try:
                    shutil.rmtree(str(d))
                except OSError as e:
                    # Still on disk: say so rather than reporting a removal, and keep sweeping the
                    # rest. begin_modification reads this answer, and a claim it cannot check is
                    # re-made on every later call.
                    stuck_backups.append(d.name)
                    details.append(f"Could not remove incomplete backup {d.name}: {e}")
                    continue
                details.append(f"Removed incomplete backup {d.name}")
                if "removed_incomplete_backup" not in actions:
                    actions.append("removed_incomplete_backup")

    if failed:
        details.append(f"Could not restore {failed} for {tool_id} -- still staged, run the repair again")

    if actions:
        return {"repaired": True, "action": "+".join(actions), "details": "; ".join(details), "failed": failed}
    if failed:
        return {"repaired": False, "action": "rollback_incomplete", "details": "; ".join(details), "failed": failed}
    if stuck_backups:
        # Not `failed`: begin_modification refuses over that, and it must not -- create_backup
        # mints a fresh version, so a metadata-less directory nothing can restore from blocks
        # no modification. What it does do is keep the store above its cap, silently.
        return {
            "repaired": False,
            "action": "incomplete_backup_not_removed",
            "details": "; ".join(details),
            "failed": [],
        }
    return {"repaired": False, "action": "none", "details": "No inconsistent state found.", "failed": []}
