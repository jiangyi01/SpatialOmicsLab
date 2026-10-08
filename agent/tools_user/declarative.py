"""
Tools a user can make in a second, that cannot break the install.

Lives beside the other managers in ``tools_user/`` -- the same top-level namespace package
``stcoscientist.py`` already imports ``user_skill`` and ``memory_manager`` from.

The env-backed tier -- write a worker, solve a conda environment, register an MCP server -- is
~7,000 lines of manager code and 7,394 lines of playbook, and on this box it was **100% broken**
for a different reason every time: a ``20 < 20`` env cap refused every creation, a 600-second step
budget against conda steps the playbook itself budgets 1800-3600 seconds for, and a timeout
implemented as a daemon-thread join that cannot kill a blocked ``subprocess.run``. That tier is
worth keeping and is not what most tools need.

A **declarative** tool is a JSON record. No environment, no import-path mutation, no subprocess, no
restart. Creating one is a validated write; deleting one is deleting a file; it cannot leave a
half-installed environment behind because it never had one. Modelled on upstream ToolUniverse's
``AgenticTool``, but through this project's own seams -- and failing in OUR shape,
``{"status": "error", ...}``, which ``execution._EXEC_ERROR_RE`` recognises. Upstream's
``{"success": false}`` is invisible to it, which is the same class of bug that let a failed tool
creation here score as a good action by printing ``FAILED:`` instead of raising.

Scoping, honestly
-----------------
Records are stored, listed and run per owner. That is **scoping, not a security boundary**, and
saying so is the only honest option: one process-global agent serves every account
(``server.py``'s own comment), and a turn executes model-written Python as the server's user in a
shared REPL, so any owner check is bypassable with ``open()``. The repo already says this in
``users_cli.py`` (and said it in ``users.py`` until that module was removed on 2026-10-07), and this module
does not contradict it.

What is deliberately NOT here
-----------------------------
**REST tools.** A tool that fetches a user-supplied URL is a server-side request forgery surface
pointed at this box's network, and doing it properly needs an egress allowlist, private-range
refusal on every hop of every redirect, and a decision about credentials that belongs to the
operator and not to this module. It is the next tier and it lands with those, not before -- and
:data:`RESERVED_KINDS` refuses it by name in the meantime rather than leaving the word free for a
half-built version.
"""

from __future__ import annotations

import json
import keyword
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The kinds this tier can run today.
KINDS: tuple[str, ...] = ("prompt",)

#: Refused by name, with the reason, so the word is not quietly available for a half-built one.
RESERVED_KINDS: dict[str, str] = {
    "rest": (
        "A tool that calls a URL needs an egress allowlist and a redirect policy this install "
        "does not have yet, so it is refused rather than half-built."
    ),
}

#: A tool name has to be a Python identifier: the agent calls it as a function in a REPL.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,47}$")

#: Placeholders in a prompt template: ``{gene}``. Anything else in braces is a literal.
SLOT_RE = re.compile(r"\{([a-z][a-z0-9_]{0,31})\}")

MAX_TEMPLATE_CHARS = 8000
MAX_TOOLS_PER_OWNER = 200


class DeclarativeError(Exception):
    """A refusal with a sentence the person who wrote the tool can act on."""


def reserved_names() -> set[str]:
    """Every callable the shipped and user MCP configs publish.

    A declarative tool may not shadow one: the merged config resolves a conflict in favour of the
    shipped server, so a shadowing record would be a tool the user made, sees listed, and can
    never call.

    Empty on any failure. A resolver that cannot answer must not silently permit a collision *or*
    block every name, and of the two, permitting is recoverable -- the merge refuses it later --
    while blocking every name is not.
    """
    try:
        from sog_portal.server import (
            _load_mcp_servers,
            _tool_function_names,
            _user_mcp_servers,
        )
        from spatialomicsgym.mcp_config_path import find_mcp_config
    except Exception:
        return set()
    names: set[str] = set()
    try:
        # `_load_mcp_servers(None)` returns `{}` -- `server.py` short-circuits on a falsy path --
        # so passing None reserved NOTHING from the shipped config and this guard was inert against
        # exactly the collision the docstring above describes. Measured before the fix: 2 names
        # reserved (both from the *user* config), 0 of 88 shipped servers and 0 of their functions,
        # and `create(..., name="run_card")`, `"seurat_find_markers"` and `"seurat"` each returned
        # state `ready`. Resolving the shipped path first is the whole fix; it stays inside this
        # `try` so a missing config still degrades to the documented empty set.
        shipped = find_mcp_config()
        for source in (_load_mcp_servers(str(shipped) if shipped else None), _user_mcp_servers()):
            for server, meta in (source or {}).items():
                names.add(str(server))
                for fn in _tool_function_names(meta if isinstance(meta, dict) else {}):
                    names.add(str(fn))
    except Exception:
        return set()
    return names


def root() -> Path:
    """Where the records live: ``<checkout>/agent/tools_user/declarative``, or, outside a checkout,
    ``<instance root>/tools_user/declarative``.

    An anchored path and not the CWD. ``resolve_user_config_path`` exists because a reader spelled
    that path CWD-relative while the writer used a repo-anchored absolute one, so every created
    tool vanished when the portal was started from a data directory. The same mistake is available
    here and this is how it is not made.

    A checkout keeps ``tools_user/`` under ``agent/`` while the instance root is the repository
    root, so the checkout's own ``tools_user`` directory is asked first; a pip install (no
    checkout) keeps the flat seeded layout under the instance root.
    """
    override = (os.environ.get("SOG_DECLARATIVE_TOOLS_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    from spatialomicsgym.layout import tools_user_dir

    checkout = tools_user_dir()
    if checkout is not None:
        return checkout / "declarative"
    from spatialomicsgym.platform_root import instance_root

    return Path(instance_root()) / "tools_user" / "declarative"


def owner_dir(owner: str) -> Path:
    """This owner's directory, with the owner re-validated as a path component.

    Validated here and not trusted from the caller: the owner reaches this module from a session,
    but a module that only works when its caller is careful is a module that breaks the first time
    it gains a second caller.
    """
    canon = _slug(owner)
    if not canon:
        raise DeclarativeError("No account was given for this tool.")
    return root() / canon


def create(
    owner: str,
    *,
    name: str,
    description: str,
    kind: str = "prompt",
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and store one tool. Returns the record.

    Every check runs BEFORE anything is written, so a refused tool leaves no file -- which is the
    whole reason this tier exists beside one that can fail with a half-built environment on disk.
    """
    body = dict(spec or {})
    kind = str(kind or "prompt").strip().lower()
    if kind in RESERVED_KINDS:
        raise DeclarativeError(RESERVED_KINDS[kind])
    if kind not in KINDS:
        raise DeclarativeError(f"This install can make {', '.join(KINDS)} tools; it cannot make a {kind!r} one.")

    clean_name = str(name or "").strip().lower()
    if not NAME_RE.match(clean_name):
        raise DeclarativeError(
            "A tool name has to look like a Python function: lower case, letters, digits and "
            "underscores, starting with a letter, 3 to 48 characters. The agent calls it by name."
        )
    if keyword.iskeyword(clean_name) or keyword.issoftkeyword(clean_name):
        # The agent calls this as a function in a REPL, so `def`, `class`, `import`, `match` and
        # friends produce a SyntaxError at the call site rather than a missing-tool message. Found
        # by a test that listed `def` among the names a tool cannot have.
        raise DeclarativeError(
            f"{clean_name} is a Python keyword, and the agent calls a tool by writing its name as "
            "a function call. Pick another."
        )
    if clean_name in reserved_names():
        raise DeclarativeError(
            f"Something this install already provides is called {clean_name}. A tool with that "
            "name would be listed and never called, because the shipped one wins."
        )

    clean_desc = " ".join(str(description or "").split())[:600]
    if len(clean_desc) < 10:
        raise DeclarativeError(
            "Describe what the tool does in a sentence. The retriever picks tools by their "
            "description, so a tool without one is a tool the agent never chooses."
        )

    validated = _validate_prompt(body)

    existing = list_tools(owner)
    if any(r.get("name") == clean_name for r in existing):
        raise DeclarativeError(f"You already have a tool called {clean_name}.")
    if len(existing) >= MAX_TOOLS_PER_OWNER:
        raise DeclarativeError(f"That is more than {MAX_TOOLS_PER_OWNER} tools on this account.")

    record = {
        "id": uuid.uuid4().hex[:16],
        "owner": _slug(owner),
        "name": clean_name,
        "description": clean_desc,
        "kind": kind,
        "spec": validated,
        "created": time.time(),
        # No environment, nothing to build, nothing to fail later. The state a caller reads is the
        # state it is in, which is the difference from the env-backed tier.
        "state": "ready",
    }
    _write(owner_dir(owner) / f"{record['id']}.json", record)
    return record


def list_tools(owner: str) -> list[dict[str, Any]]:
    """This owner's tools, newest first. Never raises: an unreadable record is skipped."""
    try:
        directory = owner_dir(owner)
    except DeclarativeError:
        return []
    out: list[dict[str, Any]] = []
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for path in entries:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("skipping an unreadable declarative tool record")
            continue
        if isinstance(payload, dict) and payload.get("name"):
            out.append(payload)
    out.sort(key=lambda r: float(r.get("created") or 0), reverse=True)
    return out


def all_tools() -> list[dict[str, Any]]:
    """Every declarative tool on this box, with its owner. For the operator surface only.

    Walks the store rather than the account list, deliberately: a tool whose account was deleted
    still has bytes on disk, and an operator asking "what is here" needs to be shown exactly
    that. Listing by account would hide the orphans, which are the rows worth acting on.

    Never raises. An unreadable owner directory is skipped; a listing that cannot be produced at
    all is worse than one that is short.
    """
    out: list[dict[str, Any]] = []
    base = root()
    try:
        owners = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return out
    for owner_dir_path in owners:
        for record in list_tools(owner_dir_path.name):
            out.append({**record, "owner": record.get("owner") or owner_dir_path.name})
    out.sort(key=lambda r: (str(r.get("owner") or ""), -float(r.get("created") or 0)))
    return out


def get(owner: str, tool_id: str) -> dict[str, Any] | None:
    wanted = str(tool_id or "").strip()
    for record in list_tools(owner):
        if record.get("id") == wanted or record.get("name") == wanted:
            return record
    return None


def delete(owner: str, tool_id: str) -> bool:
    """Deleting a tool is deleting a file. No environment to reclaim, nothing to unregister."""
    record = get(owner, tool_id)
    if not record:
        return False
    try:
        (owner_dir(owner) / f"{record['id']}.json").unlink()
    except OSError:
        return False
    return True


def render(record: dict[str, Any], arguments: dict[str, Any]) -> str:
    """Fill a prompt tool's template.

    Substitution is by exact slot name and nothing else -- no ``eval``, no format spec, no
    attribute access. ``str.format`` would accept ``{x.__class__}`` and ``{0}``, which on a
    template the model may have written is an attribute walk into this process.
    """
    spec = record.get("spec") if isinstance(record, dict) else None
    template = str((spec or {}).get("template") or "")
    supplied = dict(arguments) if isinstance(arguments, dict) else {}

    def swap(match: re.Match[str]) -> str:
        key = match.group(1)
        value = supplied.get(key)
        if value is None:
            raise DeclarativeError(f"This tool needs a value for {key}.")
        return str(value)[:4000]

    return SLOT_RE.sub(swap, template)


def as_error(message: str) -> dict[str, Any]:
    """A failure in the shape this project's own error detector recognises.

    ``execution._EXEC_ERROR_RE`` matches ``Error: ``, ``{"status": "error"}`` and tracebacks. It
    does NOT match ``{"success": false}``, which is upstream's shape -- and it did not match
    ``FAILED:``, which is how a failed tool creation came to score as a good action here.
    """
    return {"status": "error", "message": str(message)[:2000]}


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _validate_prompt(spec: dict[str, Any]) -> dict[str, Any]:
    template = str(spec.get("template") or "").strip()
    if not template:
        raise DeclarativeError("A prompt tool needs a template -- the text sent to the model.")
    if len(template) > MAX_TEMPLATE_CHARS:
        raise DeclarativeError(f"That template is longer than {MAX_TEMPLATE_CHARS} characters.")
    if any(ord(ch) > 127 for ch in template):
        raise DeclarativeError(
            "Keep the template to plain ASCII. A curly quote or a bullet reaches the agent's "
            "Python REPL inside a string and has produced a SyntaxError loop here before."
        )
    slots = list(dict.fromkeys(SLOT_RE.findall(template)))
    declared = [str(i).strip().lower() for i in (spec.get("inputs") or []) if str(i).strip()]
    unknown = [d for d in declared if d not in slots]
    if unknown:
        raise DeclarativeError(f"These inputs are declared and never used in the template: {', '.join(unknown)}.")
    # The slots ARE the inputs. Keeping a separate list is a second thing to hold in step, so the
    # template wins and `inputs` records what was found in it.
    return {"template": template, "inputs": slots}


def _slug(value: object) -> str:
    text = "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum() or ch in "._-")
    return "" if text in ("", ".", "..") else text[:64]


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Atomic, 0600, directory created on the way. Never a partial record on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tool_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
