"""Brokered server-side actions for a REPL worker that may not do them itself.

Under the privilege boundary (``sog_portal/boundary.py``) model-written code runs as ``sog-agent`` and
cannot create a conda environment, install into one, write ``tools_user/`` or register a server
in ``MCP_server/mcp_config_user.yaml`` -- which is exactly what the tool-creation playbook
(``know_how/add_new_mcp_tool*.md``) tells the model to do with raw ``subprocess`` and file I/O.
The user's decision (Program 4) was to broker those actions rather than disable the feature: the
worker asks, the server does it as itself under a FIXED policy, and every request -- granted or
refused -- is one line in the existing audit log (``services.audit``; no new audit infrastructure,
D-044).

The policy is written for a hostile caller, because that is what a sandbox boundary means. The
first version of this file trusted the shape of its inputs and an audit found four ways through it
(2026-09-20): a worker could overwrite the platform's own modules under ``tools_user/`` (which the
root server imports), hand pip a local path or a ``file:`` URL (pip runs its build backend as the
server), smuggle ``--index-url`` or an arbitrary channel through a recipe, and reach any other
account's environments, files and config entries because ``owner`` was only an audit label. So:

* **paths are an allowlist of tool-shaped names**, never a denylist: ``<tool_id>_worker.py``,
  ``<tool_id>_mcp_server.py``, ``<tool_id>_env.yaml``, ``<tool_id>_worker.R``,
  ``vendor_<tool_id>/...`` and ``.knowledge/<tool_id>/...`` (never its ``backups/``), resolved and
  contained; nothing else under ``tools_user/`` can be written, which covers every shipped module,
  ``__init__.py`` and ``install_log.json`` by construction;
* **package specs are parsed, not pattern-matched**: a PEP 508 name with an optional extras/version
  clause, or an ``https://`` / ``git+https://`` URL with a host and no userinfo; a path, a
  ``file:``/``ssh:``/``http:`` URL or anything starting with ``-`` is refused;
* **a recipe is rebuilt from validated parts**: ``channels`` (names), conda specs, and one ``pip:``
  list through the same parser; every other key is dropped;
* **everything is owner-scoped through a claims ledger** the server keeps (``broker_claims.json``
  in the state dir): an environment or a tool id belongs to the account that first created it, and
  another account's request for it is refused; caps are per owner;
* **a registered server may only run a claimed environment's interpreter on the tool's own
  files**, with an ``env:`` block that names no placeholder.

Executing an environment needs no privilege: the worker runs ``conda run -n user_x python ...``
itself. Nothing here is reachable in-process or under benchmarking: ``upcalls_for`` answers ``{}``
unless ``tool_creation_enabled`` is on AND ``process_isolation()`` is true, and the CLI/benchmark
run in-process, where the playbook's direct writes work exactly as before.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ENV_NAME_RE = re.compile(r"^user_[a-z0-9][a-z0-9_]{0,40}$")
TOOL_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
#: A PEP 508 requirement without URL or marker: name, optional extras, optional version clause.
_NAME_SPEC_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?:\[[A-Za-z0-9._,\-]+\])?"
    r"(?:\s*(?:===|[<>=!~]=?)\s*[A-Za-z0-9.*+!_-]+(?:\s*,\s*(?:===|[<>=!~]=?)\s*[A-Za-z0-9.*+!_-]+)*)?$"
)
#: A conda match spec as a recipe lists it: ``name``, ``name=ver``, ``name=ver=build``, ``name>=ver``.
_CONDA_SPEC_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?(?:\s*(?:[<>=!~]=?|=)\s*[A-Za-z0-9.*+_-]+)?(?:=[A-Za-z0-9._*+]+)?$"
)
_CHANNEL_RE = re.compile(r"^[A-Za-z0-9._\-]+$")
_ENV_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FLAG_RE = re.compile(r"^--?[A-Za-z0-9][A-Za-z0-9_.-]*(?:=[A-Za-z0-9._/-]*)?$")
_TOOL_FILE_RE = re.compile(r"^(?P<tid>[a-z][a-z0-9_]{0,62})_(?:worker\.py|mcp_server\.py|env\.yaml|worker\.R)$")
_VENDOR_RE = re.compile(r"^vendor_(?P<tid>[a-z][a-z0-9_]{0,62})(?:/.*)?$")
_KNOWLEDGE_RE = re.compile(r"^\.knowledge/(?P<tid>[a-z][a-z0-9_]{0,62})/(?P<rest>.+)$")

MAX_RECIPE_BYTES = 64 * 1024
MAX_TOOL_FILE_BYTES = 512 * 1024
MAX_FILES_PER_CALL = 32
#: Per-owner ceilings; the box-wide ``max_user_envs`` still applies on top.
MAX_ENVS_PER_OWNER_VAR = "SOG_BROKER_MAX_ENVS_PER_OWNER"
DEFAULT_MAX_ENVS_PER_OWNER = 5
MAX_BYTES_PER_OWNER_VAR = "SOG_BROKER_MAX_BYTES_PER_OWNER"
DEFAULT_MAX_BYTES_PER_OWNER = 64 * 1024 * 1024
#: Wall clock for one conda/pip call; the playbook's envs take minutes, not hours.
BROKER_TIMEOUT_VAR = "SOG_BROKER_TIMEOUT"
DEFAULT_BROKER_TIMEOUT = 1800.0
CLAIMS_NAME = "broker_claims.json"

NOTICE_TAG = "[boundary notice]"

_CLAIMS_LOCK = threading.Lock()

#: Env names with a conda verb running right now, and its guard. See :func:`_one_conda_op_per_env`.
_IN_FLIGHT: set[str] = set()
_IN_FLIGHT_LOCK = threading.Lock()


def active(agent: Any = None) -> bool:
    """The verbs exist only when tool creation is on AND cells run in the worker."""
    try:
        from spatialomicsgym.config import default_config
        from spatialomicsgym.tool.support_tools import process_isolation

        if not process_isolation():
            return False
        return bool(getattr(default_config, "tool_creation_enabled", False))
    except Exception:
        return False


def upcalls_for(agent: Any) -> dict[str, Any]:
    """The verbs, bound to this turn's owner. ``{}`` when the broker is not active or the turn has
    no account -- a tool made by nobody would belong to nobody."""
    if not active(agent):
        return {}
    owner = str(getattr(agent, "_turn_owner", "") or "")
    if not owner:
        return {}
    # A space of the public demo builds nothing (2026-10-04): what it built would be wired for every
    # account once an operator switched it on, and the whole internet signs in to the demo. The name's
    # shape is the test (``sog_portal/guest.py``), read here without importing the portal.
    if re.fullmatch(r"guest-[0-9a-f]{12}", owner):
        return {}
    return {
        "create_tool_env": _bound(create_tool_env, owner),
        "install_in_tool_env": _bound(install_in_tool_env, owner),
        "remove_tool_env": _bound(remove_tool_env, owner),
        "write_tool_files": _bound(write_tool_files, owner),
        "register_tool": _bound(register_tool, owner),
    }


def protected_roots() -> list[str]:
    """Where a cell's ``Errno 13`` means "ask the server": the repository, the conda root, the
    state directory."""
    roots: list[str] = []
    try:
        from sog_install import constants

        roots.append(str(constants.repo_root()))
        roots.append(str(constants.state_dir()))
        try:
            roots.append(str(constants.live_conda_root()))
        except Exception:
            pass
        try:
            roots.append(str(constants.conda_envs_root()))
        except Exception:
            pass
    except Exception:
        pass
    return sorted({r for r in roots if r})


def notice(path: str) -> str:
    """What the worker prints, once per protected root, when a cell is denied there."""
    return (
        f"{NOTICE_TAG} This portal runs your code as an unprivileged user, so '{path}' cannot be written "
        "from a cell. The server does these on your behalf under a fixed policy, and records each: "
        "create_tool_env(name, recipe=<yaml text>) or create_tool_env(name, packages=[...]), "
        "install_in_tool_env(name, packages=[...]), remove_tool_env(name), "
        "write_tool_files({'<tool>_worker.py': <text>, ...}) for this tool's own files under tools_user/, and "
        "register_tool(tool_id, config_block, install_log_entry) for mcp_config_user.yaml and the install log. "
        "Environments are named user_<id> and belong to the account that created them. "
        "Executing an environment needs no privilege: run the worker and its tests yourself with "
        "conda run -n <env> python <script>. Say in your <solution> which of these you used."
    )


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def _bound(fn: Any, owner: str) -> Any:
    def _call(**kwargs: Any) -> Any:
        return fn(_owner=owner, **kwargs)

    _call.__name__ = fn.__name__
    _call.__qualname__ = fn.__name__
    _call.__doc__ = fn.__doc__
    try:
        import inspect

        params = [p for p in inspect.signature(fn).parameters.values() if p.name != "_owner"]
        _call.__signature__ = inspect.Signature(params)
    except Exception:
        pass
    return _call


def _refuse(action: str, owner: str, target: str, message: str, **detail: Any) -> dict[str, Any]:
    _audit(action, owner, target, {"status": "refused", "message": message, **detail})
    return {"status": "error", "message": message}


def _audit(action: str, owner: str, target: str, detail: dict[str, Any]) -> None:
    try:
        from sog_portal.services import audit as _audit_log

        _audit_log.default_service().record(f"broker.{action}", actor=owner, target=target, detail=detail)
    except Exception:
        pass


def _timeout() -> float:
    try:
        return float(os.environ.get(BROKER_TIMEOUT_VAR) or DEFAULT_BROKER_TIMEOUT)
    except ValueError:
        return DEFAULT_BROKER_TIMEOUT


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name) or default))
    except ValueError:
        return default


def _conda() -> str | None:
    found = shutil.which("conda")
    if found:
        return found
    try:
        from sog_install import constants

        candidate = Path(constants.live_conda_root()) / "bin" / "conda"
        if candidate.is_file():
            return str(candidate)
    except Exception:
        pass
    return None


def _run(argv: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """One subprocess, as the server. Patched by tests; never a shell.

    Its own process group, killed whole on a timeout: subprocess.run killed only conda itself, and
    the pip and solver children it had started kept running as root (hunt 2026-09-30,
    u14-mcp-wiring-3). The TimeoutExpired is re-raised, as before.
    """
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout or _timeout())
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def _release_env_unless_left_behind(name: str) -> str:
    """Release the claim on ``name`` only if no env directory was left on disk; a note for the caller.

    A failed or timed-out create released the claim whether or not conda had already made the prefix,
    so the half-built env stayed with no owner: create said "already exists", remove and install said
    it "belongs to another account" -- nobody could clear it, and it counted against max_user_envs.
    """
    if _env_prefix(name).exists():
        return f" A partial environment was left at {name!r} and is still yours: remove_tool_env({name!r}) clears it."
    _release("envs", name)
    return ""


def _tail(text: str, n: int = 3000) -> str:
    return (text or "")[-n:]


def _envs_root() -> Path:
    from sog_install import constants

    return Path(constants.conda_envs_root())


def _env_prefix(name: str) -> Path:
    return _envs_root() / name


def _protected(name: str) -> bool:
    try:
        from sog_install import constants

        return name in constants.PROTECTED_ENVS
    except Exception:
        return False


def _user_env_count() -> int:
    try:
        return sum(1 for p in _envs_root().iterdir() if p.is_dir() and p.name.startswith("user_"))
    except Exception:
        return 0


def _max_user_envs() -> int:
    try:
        from spatialomicsgym.config import default_config

        return max(0, int(getattr(default_config, "max_user_envs", 20)))
    except Exception:
        return 20


def _tools_user_dir() -> Path:
    from tools_user import knowledge_manager as _km

    return Path(_km.TOOLS_USER_DIR)


# --------------------------------------------------------------------------- #
# the claims ledger: who owns which environment and which tool id
# --------------------------------------------------------------------------- #
def _claims_path() -> Path:
    from sog_install import constants

    return Path(constants.state_dir()) / CLAIMS_NAME


def _read_claims() -> dict[str, dict[str, str]]:
    try:
        loaded = json.loads(_claims_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    return {"envs": dict(loaded.get("envs") or {}), "tools": dict(loaded.get("tools") or {})}


def _write_claims(claims: dict[str, dict[str, str]]) -> None:
    path = _claims_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(claims, indent=1, sort_keys=True), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _exists_unclaimed(kind: str, key: str) -> bool:
    """Whether ``key`` already exists on disk although nobody has claimed it.

    Tools created before the claims ledger (or by the in-process path, which records no owner) have
    files under ``tools_user/`` and are wired into ``mcp_config_user.yaml`` -- which is merged into
    EVERY account's agent. Granting such an id to the first account that asks let it overwrite the
    shared tool's portal and worker, and every other account's later call then ran that code in its
    own worker, with its own data (hunt 2026-09-30, u14-mcp-wiring-4). A ``conda create -y`` over an
    existing unclaimed env would replace it the same way. Such ids belong to the operator.
    """
    try:
        if kind == "tools":
            root = _tools_user_dir()
            return any(root.glob(f"{key}_mcp_server.py")) or any(root.glob(f"{key}_worker.*"))
        if kind == "envs":
            return (_envs_root() / key).exists()
    except OSError:
        return False
    return False


def _claim(kind: str, key: str, owner: str, *, dry_run: bool = False) -> str | None:
    """Claim ``key`` for ``owner``; ``None`` on success, else the message saying whose it is.

    A tool id is also checked against the install log, so a tool created before this ledger
    existed (or by the in-process path) keeps its owner. One that exists on disk with no recorded
    owner anywhere is the operator's, not the first caller's (:func:`_exists_unclaimed`).

    ``dry_run`` answers the same question and records nothing -- for a caller that has more to
    check before it may take the key.
    """
    with _CLAIMS_LOCK:
        claims = _read_claims()
        held = claims[kind].get(key)
        if held is None and kind == "tools":
            held = _install_log_owner(key)
        if held is None and _exists_unclaimed(kind, key):
            return f"{key!r} already exists and has no recorded owner; an operator must assign it"
        if held is not None and held != owner:
            return f"{key!r} belongs to another account"
        if not dry_run and claims[kind].get(key) != owner:
            claims[kind][key] = owner
            _write_claims(claims)
    return None


def _owns(kind: str, key: str, owner: str) -> bool:
    with _CLAIMS_LOCK:
        held = _read_claims()[kind].get(key)
    if held is None and kind == "tools":
        held = _install_log_owner(key)
    return held == owner


def _release(kind: str, key: str) -> None:
    with _CLAIMS_LOCK:
        claims = _read_claims()
        if key in claims[kind]:
            del claims[kind][key]
            _write_claims(claims)


def _owned(kind: str, owner: str) -> list[str]:
    with _CLAIMS_LOCK:
        return sorted(k for k, o in _read_claims()[kind].items() if o == owner)


def _install_log_owner(tool_id: str) -> str | None:
    try:
        from tools_user import knowledge_manager as _km

        for entry in _km._read_install_log():
            if str(entry.get("tool_id") or "") == tool_id and entry.get("owner"):
                return str(entry["owner"])
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def spec_kind(text: str) -> str | None:
    """``"name"`` for a PEP 508 name spec, ``"url"`` for an https/git+https URL with a host and no
    userinfo, ``None`` for anything else (a path, ``file:``, ``ssh:``, ``http:``, an option)."""
    text = str(text or "").strip()
    if not text or len(text) > 400 or any(c.isspace() for c in text) or text.startswith("-"):
        return None
    if _NAME_SPEC_RE.match(text):
        return "name"
    try:
        parts = urlsplit(text)
    except ValueError:
        return None
    try:
        parts.port  # noqa: B018 -- raises on 'pypi.org:abc', which the hostname check never looks at
    except ValueError:
        return None  # u16-llm-config-8: the host was approved and the malformed port rode along
    if parts.scheme in ("https", "git+https") and parts.netloc and "@" not in parts.netloc and parts.netloc.strip():
        return "url"
    return None


def _valid_packages(packages: Any) -> tuple[list[str], str | None]:
    if isinstance(packages, str):
        packages = [packages]
    if not isinstance(packages, list) or not packages:
        return [], "packages must be a non-empty list of package specs"
    out: list[str] = []
    for spec in packages:
        text = str(spec).strip()
        kind = spec_kind(text)
        if kind is None:
            return [], (
                f"package spec {text[:60]!r} is not a package name (with an optional version) or an https:// / "
                "git+https:// URL; paths, file: and options are refused"
            )
        if kind == "url":
            # WHICH host and WHICH repository, not merely "a URL with a host and no userinfo".
            # Until 2026-09-23 ``spec_kind`` was the whole gate, so any https:// origin was
            # installable -- and "github.com is reachable" meant every repository on it. The
            # policy is path-granular for exactly that reason.
            from spatialomicsgym.policy import egress as _egress

            checker = _egress.repo_allowed if text.startswith("git+") else _egress.index_allowed
            try:
                ok, why = checker(text)
            except _egress.EgressPolicyError as exc:
                return [], f"the egress policy could not be read, so no install can be approved: {exc}"
            if not ok:
                return [], f"package spec {_egress.safe_url(text)!r} is refused: {why}"
        out.append(text)
    return out, None


def _valid_channels(channels: Any) -> tuple[list[str], str | None]:
    from spatialomicsgym.policy import egress as _egress

    out: list[str] = []
    for ch in channels or []:
        if isinstance(ch, str) and _CHANNEL_RE.match(ch):
            # Judged on the leading segment, matching ``remediation_planner``'s existing split, so
            # ``conda-forge/label/broken`` is judged on ``conda-forge``.
            try:
                ok, why = _egress.channel_allowed(ch)
            except _egress.EgressPolicyError as exc:
                return [], f"the egress policy could not be read, so no channel can be approved: {exc}"
            if not ok:
                return [], why
        if not isinstance(ch, str) or not _CHANNEL_RE.match(ch):
            return [], f"channel {str(ch)[:60]!r} is not a channel name"
        out.append(ch)
    return out, None


def _sanitize_recipe(text: str, name: str) -> tuple[str | None, str | None]:
    """Rebuild a conda environment YAML from its validated parts, or say why not."""
    try:
        import yaml
    except Exception:
        return None, "PyYAML is not available to the server"
    try:
        loaded = yaml.safe_load(text)
    except Exception as exc:
        return None, f"the recipe is not valid YAML: {exc}"
    if not isinstance(loaded, dict) or not isinstance(loaded.get("dependencies"), list) or not loaded["dependencies"]:
        return None, "the recipe must be a conda environment YAML with a dependencies list"
    channels, why = _valid_channels(loaded.get("channels") or [])
    if why:
        return None, why
    deps: list[Any] = []
    pip_seen = False
    for dep in loaded["dependencies"]:
        if isinstance(dep, str):
            spec = dep.strip()
            if not spec or spec.startswith("-") or not _CONDA_SPEC_RE.match(spec):
                return None, f"dependency {spec[:60]!r} is not a conda package spec"
            deps.append(spec)
        elif isinstance(dep, dict) and set(dep) == {"pip"}:
            if pip_seen:
                return None, "the recipe lists pip twice"
            pip_seen = True
            pkgs, why = _valid_packages(dep.get("pip") or [])
            if why:
                return None, why
            deps.append({"pip": pkgs})
        else:
            return None, "a dependency entry must be a package spec or a single pip: list"
    rebuilt: dict[str, Any] = {"name": name, "dependencies": deps}
    if channels:
        rebuilt["channels"] = channels
    return yaml.safe_dump(rebuilt, sort_keys=False), None


def _tool_target(rel: Any, root: Path) -> tuple[str | None, Path | None, str | None]:
    """``(tool_id, resolved path, None)`` for a writable tool-shaped path, else ``(None, None, why)``.

    The containment check and the shape check, FUSED, because separately they are a pairing a
    caller has to remember. ``_tool_path_kind`` is lexical -- it reads a string and knows nothing
    about symlinks -- so it is only a containment answer when it sits beside a ``.resolve()`` and a
    parent test. Both callers did pair them, and in OPPOSITE orders: ``write_tool_files``
    classified the raw string and then resolved, ``_valid_command`` resolved and then classified
    the result. Two orders for one invariant is how the third caller gets it wrong.

    Resolve first, contain, then classify what came back -- the order ``report.manifest.safe_subpath``
    and ``datastore.resolve_derived`` both use, and the one that makes a symlink planted inside
    ``tools_user/`` resolve to its target and FAIL containment rather than pass it.
    """
    text = str(rel).strip()
    if not text:
        return None, None, "a path is required"
    clean = text[len("tools_user/") :] if text.startswith("tools_user/") else text
    if not clean or os.path.isabs(clean) or "\\" in clean:
        return None, None, f"{text!r} is not a relative path under tools_user/"
    if any(part in ("..", ".", "") for part in Path(clean).parts):
        return None, None, f"{text!r} is not a relative path under tools_user/"
    try:
        target = (root / clean).resolve()
    except OSError:
        return None, None, f"{text!r} could not be resolved"
    if root not in target.parents:
        return None, None, f"{text!r} escapes tools_user/"
    tid, why = _tool_path_kind(str(target.relative_to(root)))
    if why:
        return None, None, why
    return tid, target, None


def _tool_path_kind(rel: str) -> tuple[str | None, str | None]:
    """``(tool_id, None)`` for a writable tool-shaped path, else ``(None, reason)``."""
    if rel.startswith("tools_user/"):
        rel = rel[len("tools_user/") :]
    if not rel or os.path.isabs(rel) or "\\" in rel:
        return None, f"{rel!r} is not a relative path under tools_user/"
    parts = Path(rel).parts
    if any(p in ("..", ".", "") for p in parts):
        return None, f"{rel!r} is not a relative path under tools_user/"
    m = _TOOL_FILE_RE.match(rel)
    if m:
        return m.group("tid"), None
    m = _VENDOR_RE.match(rel)
    if m and len(parts) > 1:
        return m.group("tid"), None
    m = _KNOWLEDGE_RE.match(rel)
    if m:
        if Path(m.group("rest")).parts[0] == "backups":
            return None, f"{rel!r}: the rollback snapshots are the server's, not a tool's to write"
        return m.group("tid"), None
    return None, (
        f"{rel!r} is not one of a tool's own files: <tool_id>_worker.py, <tool_id>_mcp_server.py, "
        "<tool_id>_env.yaml, <tool_id>_worker.R, vendor_<tool_id>/..., .knowledge/<tool_id>/..."
    )


def _owner_bytes(owner: str, also: tuple[str, ...] = ()) -> int:
    """What this owner's tool files already cost under tools_user/, counting ids in ``also`` as theirs.

    ``also`` is for the ids a caller has checked but not yet claimed, so the cap counts them as it did
    when the claim came first.
    """
    total = 0
    try:
        root = _tools_user_dir()
        for tid in sorted(set(_owned("tools", owner)) | set(also)):
            for candidate in (
                root / f"{tid}_worker.py",
                root / f"{tid}_mcp_server.py",
                root / f"{tid}_env.yaml",
                root / f"{tid}_worker.R",
            ):
                if candidate.is_file():
                    total += candidate.stat().st_size
            for sub in (root / f"vendor_{tid}", root / ".knowledge" / tid):
                if sub.is_dir():
                    for dirpath, _d, files in os.walk(sub):
                        for f in files:
                            try:
                                total += (Path(dirpath) / f).stat().st_size
                            except OSError:
                                pass
    except Exception:
        pass
    return total


def _one_conda_op_per_env(fn: Any) -> Any:
    """One conda verb at a time per env name; a second one, meanwhile, is refused and says why.

    A Stop ends the request waiting on a brokered upcall at once, and the abandoned call finishes on
    its own thread (``repl_client.ReplProcess._run_upcall``). So the next turn's retry of the same
    build could start while the first conda still ran: the prefix-exists check narrows that, since
    conda makes the prefix early, but in the window before it two ``conda create -n user_x`` ran into
    one directory (hunt 2026-09-30, uL2-concurrency-2). Another env is not held up.
    """
    import functools

    @functools.wraps(fn)
    def _guarded(name: str, *args: Any, _owner: str = "", **kwargs: Any) -> dict[str, Any]:
        key = str(name or "").strip()
        with _IN_FLIGHT_LOCK:
            busy = key in _IN_FLIGHT
            if not busy:
                _IN_FLIGHT.add(key)
        if busy:
            return _refuse(
                fn.__name__,
                _owner,
                key,
                f"a conda operation on {key!r} from an earlier request is still running on the server; "
                "wait for it to finish, then check the environment before trying again",
            )
        try:
            return fn(name, *args, _owner=_owner, **kwargs)
        finally:
            with _IN_FLIGHT_LOCK:
                _IN_FLIGHT.discard(key)

    return _guarded


# --------------------------------------------------------------------------- #
# the verbs
# --------------------------------------------------------------------------- #
@_one_conda_op_per_env
def create_tool_env(
    name: str,
    recipe: str | None = None,
    packages: list[str] | None = None,
    python: str = "3.11",
    channels: list[str] | None = None,
    *,
    _owner: str = "",
) -> dict[str, Any]:
    """Create the conda environment ``name`` (``user_<id>``) as the server, under policy."""
    name = str(name or "").strip()
    if not ENV_NAME_RE.match(name):
        return _refuse("create_tool_env", _owner, name, "the environment name must be user_<id> (lowercase, digits, _)")
    if _protected(name):
        return _refuse("create_tool_env", _owner, name, f"{name!r} is a protected environment")
    if _env_prefix(name).exists():
        return _refuse("create_tool_env", _owner, name, f"{name!r} already exists; remove_tool_env(name) first")
    if _user_env_count() >= _max_user_envs():
        return _refuse(
            "create_tool_env", _owner, name, f"the box's cap of {_max_user_envs()} user environments is reached"
        )
    per_owner = _int_env(MAX_ENVS_PER_OWNER_VAR, DEFAULT_MAX_ENVS_PER_OWNER)
    if len(_owned("envs", _owner)) >= per_owner:
        return _refuse(
            "create_tool_env",
            _owner,
            name,
            f"your cap of {per_owner} environments is reached; remove_tool_env one first",
        )
    conda = _conda()
    if not conda:
        return _refuse("create_tool_env", _owner, name, "conda is not available to the server")
    path = None
    if recipe is not None:
        text = str(recipe)
        if len(text.encode("utf-8")) > MAX_RECIPE_BYTES:
            return _refuse("create_tool_env", _owner, name, f"the recipe exceeds {MAX_RECIPE_BYTES} bytes")
        rebuilt, why = _sanitize_recipe(text, name)
        if why:
            return _refuse("create_tool_env", _owner, name, why)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".yml", prefix="broker_env_", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(rebuilt or "")
            path = fh.name
        argv = [conda, "env", "create", "-y", "-n", name, "-f", path]
        detail_argv = ["env", "create", "-y", "-n", name, "-f", os.path.basename(path)]
    else:
        pkgs: list[str] = []
        if packages:
            pkgs, why = _valid_packages(packages)
            if why:
                return _refuse("create_tool_env", _owner, name, why)
        py = str(python or "3.11").strip()
        if not re.match(r"^3\.\d{1,2}(\.\d{1,2})?$", py):
            return _refuse("create_tool_env", _owner, name, f"python={py!r} is not a version")
        chans, why = _valid_channels(channels or [])
        if why:
            return _refuse("create_tool_env", _owner, name, why)
        for spec in pkgs:
            if spec_kind(spec) != "name" or not _CONDA_SPEC_RE.match(spec):
                return _refuse("create_tool_env", _owner, name, f"conda create takes package names, not {spec[:60]!r}")
        argv = [conda, "create", "-y", "-n", name, f"python={py}"]
        for ch in chans:
            argv += ["-c", ch]
        argv += pkgs
        detail_argv = argv[1:]
    taken = _claim("envs", name, _owner)
    if taken:
        return _refuse("create_tool_env", _owner, name, taken)
    try:
        proc = _run(argv)
    except subprocess.TimeoutExpired:
        left = _release_env_unless_left_behind(name)
        return _refuse("create_tool_env", _owner, name, f"conda did not finish within {int(_timeout())}s.{left}")
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    ok = proc.returncode == 0 and _env_prefix(name).exists()
    _audit(
        "create_tool_env",
        _owner,
        name,
        {"status": "ok" if ok else "failed", "rc": proc.returncode, "argv": detail_argv},
    )
    if not ok:
        left = _release_env_unless_left_behind(name)
        return {
            "status": "error",
            "message": f"conda exited {proc.returncode}{left}\n{_tail(proc.stderr)}\n{_tail(proc.stdout, 1500)}",
        }
    from sog_install import constants

    return {
        "status": "ok",
        "env": name,
        "python": constants.interp_path(str(_env_prefix(name))),
        "stdout_tail": _tail(proc.stdout, 1500),
    }


@_one_conda_op_per_env
def install_in_tool_env(
    name: str,
    packages: list[str],
    manager: str = "pip",
    channels: list[str] | None = None,
    *,
    _owner: str = "",
) -> dict[str, Any]:
    """Install ``packages`` into YOUR user environment ``name`` with pip or conda, as the server."""
    name = str(name or "").strip()
    if not ENV_NAME_RE.match(name) or _protected(name):
        return _refuse("install_in_tool_env", _owner, name, "only a user_<id> environment can be installed into")
    if not _env_prefix(name).exists():
        return _refuse(
            "install_in_tool_env", _owner, name, f"{name!r} does not exist; create_tool_env(name, ...) first"
        )
    if not _owns("envs", name, _owner):
        return _refuse("install_in_tool_env", _owner, name, f"{name!r} belongs to another account")
    pkgs, why = _valid_packages(packages)
    if why:
        return _refuse("install_in_tool_env", _owner, name, why)
    conda = _conda()
    if not conda:
        return _refuse("install_in_tool_env", _owner, name, "conda is not available to the server")
    manager = str(manager or "pip").strip().lower()
    if manager == "pip":
        argv = [conda, "run", "-n", name, "python", "-m", "pip", "install", "--no-input", *pkgs]
    elif manager == "conda":
        chans, why = _valid_channels(channels or [])
        if why:
            return _refuse("install_in_tool_env", _owner, name, why)
        for spec in pkgs:
            if spec_kind(spec) != "name" or not _CONDA_SPEC_RE.match(spec):
                return _refuse(
                    "install_in_tool_env", _owner, name, f"conda install takes package names, not {spec[:60]!r}"
                )
        argv = [conda, "install", "-y", "-n", name]
        for ch in chans:
            argv += ["-c", ch]
        argv += pkgs
    else:
        return _refuse("install_in_tool_env", _owner, name, "manager must be 'pip' or 'conda'")
    try:
        proc = _run(argv)
    except subprocess.TimeoutExpired:
        return _refuse("install_in_tool_env", _owner, name, f"the install did not finish within {int(_timeout())}s")
    ok = proc.returncode == 0
    _audit(
        "install_in_tool_env",
        _owner,
        name,
        {"status": "ok" if ok else "failed", "rc": proc.returncode, "packages": pkgs},
    )
    if not ok:
        return {
            "status": "error",
            "message": f"{manager} exited {proc.returncode}\n{_tail(proc.stderr)}\n{_tail(proc.stdout, 1500)}",
        }
    return {"status": "ok", "env": name, "installed": pkgs, "stdout_tail": _tail(proc.stdout, 1500)}


@_one_conda_op_per_env
def remove_tool_env(name: str, *, _owner: str = "") -> dict[str, Any]:
    """``conda remove --all`` one of YOUR user environments -- the playbook's rollback."""
    name = str(name or "").strip()
    if not ENV_NAME_RE.match(name) or _protected(name):
        return _refuse("remove_tool_env", _owner, name, "only a user_<id> environment can be removed")
    if not _env_prefix(name).exists():
        return _refuse("remove_tool_env", _owner, name, f"{name!r} does not exist")
    if not _owns("envs", name, _owner):
        return _refuse("remove_tool_env", _owner, name, f"{name!r} belongs to another account")
    conda = _conda()
    if not conda:
        return _refuse("remove_tool_env", _owner, name, "conda is not available to the server")
    try:
        proc = _run([conda, "remove", "-y", "-n", name, "--all"])
    except subprocess.TimeoutExpired:
        return _refuse("remove_tool_env", _owner, name, f"conda did not finish within {int(_timeout())}s")
    ok = proc.returncode == 0
    _audit("remove_tool_env", _owner, name, {"status": "ok" if ok else "failed", "rc": proc.returncode})
    if not ok:
        return {"status": "error", "message": f"conda exited {proc.returncode}\n{_tail(proc.stderr)}"}
    _release("envs", name)
    return {"status": "ok", "env": name}


def write_tool_files(files: dict[str, str], *, _owner: str = "") -> dict[str, Any]:
    """Write ``{relpath: text}`` -- only a tool's own files, only YOUR tool's -- as the server."""
    if not isinstance(files, dict) or not files:
        return _refuse("write_tool_files", _owner, "", "files must be a non-empty mapping of relative path to text")
    if len(files) > MAX_FILES_PER_CALL:
        return _refuse("write_tool_files", _owner, "", f"at most {MAX_FILES_PER_CALL} files per call")
    root = _tools_user_dir().resolve()
    planned: list[tuple[Path, str, str]] = []
    planned_tids: list[str] = []
    incoming = 0
    for rel, content in files.items():
        rel = str(rel).strip()
        # One call that resolves, contains AND classifies -- see `_tool_target`. This used to
        # classify the raw string here and resolve thirty lines further down, which worked and was
        # a pairing the next caller had to know about.
        tid, target, why = _tool_target(rel, root)
        if why:
            return _refuse("write_tool_files", _owner, rel, why)
        assert tid is not None and target is not None
        if not isinstance(content, str):
            return _refuse("write_tool_files", _owner, rel, f"{rel!r}: content must be text")
        size = len(content.encode("utf-8"))
        if size > MAX_TOOL_FILE_BYTES:
            return _refuse("write_tool_files", _owner, rel, f"{rel!r} exceeds {MAX_TOOL_FILE_BYTES} bytes")
        incoming += size
        clean = str(target.relative_to(root))
        if target.suffix == ".py":
            try:
                ast.parse(content)
            except SyntaxError as exc:
                return _refuse("write_tool_files", _owner, rel, f"{rel!r} does not parse: {exc}")
        # Asked, not taken: a call refused further on -- a later file that does not parse, the byte
        # cap below -- claimed every id it had planned so far and wrote nothing, so an account could
        # squat any future tool id for free (hunt 2026-09-30, u14-mcp-wiring-extra-23).
        taken = _claim("tools", tid, _owner, dry_run=True)
        if taken:
            return _refuse("write_tool_files", _owner, rel, taken)
        planned.append((target, content, clean))
        planned_tids.append(tid)
    cap = _int_env(MAX_BYTES_PER_OWNER_VAR, DEFAULT_MAX_BYTES_PER_OWNER)
    if _owner_bytes(_owner, tuple(planned_tids)) + incoming > cap:
        return _refuse("write_tool_files", _owner, "", f"your tool files would exceed {cap} bytes under tools_user/")
    # Every check passed: now take the ids. One another account took since the dry run refuses the
    # call, and the ids this call newly took are given back, so a refusal still leaves no claim.
    tids = sorted(set(planned_tids))
    held_before = {tid for tid in tids if _read_claims()["tools"].get(tid) == _owner}
    for tid in tids:
        taken = _claim("tools", tid, _owner)
        if taken:
            for newly in set(tids[: tids.index(tid)]) - held_before:
                _release("tools", newly)
            return _refuse("write_tool_files", _owner, tid, taken)
    written: list[str] = []
    for target, content, clean in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
        written.append(clean)
    _audit("write_tool_files", _owner, ",".join(written)[:200], {"status": "ok", "files": written})
    return {"status": "ok", "written": written}


_INTERPRETER_NAME_RE = re.compile(r"^(?:python(?:3(?:\.\d+)?)?|Rscript)$")


def _valid_command(command: Any, tool_id: str, owner: str) -> tuple[list[str], str | None]:
    """A claimed environment's interpreter, running this tool's own files."""
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        return [], "config_block.command must be a list of strings"
    try:
        head = Path(command[0]).resolve()
        envs = _envs_root().resolve()
    except OSError:
        return [], "config_block.command[0] could not be resolved"
    # The NAME is checked on the path as given and on what it resolves to. In every conda env bin/python
    # is a symlink to python3.X, so checking only the resolved name refused every real Python
    # interpreter, create_tool_env's own answer included (hunt 2026-09-30, u14-mcp-wiring-2); checking
    # only the given name would accept an env-internal link to anything. Containment stays on the
    # resolved path.
    given = Path(command[0]).name
    if envs not in head.parents or not _INTERPRETER_NAME_RE.match(given) or not _INTERPRETER_NAME_RE.match(head.name):
        return [], "config_block.command[0] must be the python or Rscript of one of your user_<id> environments"
    env_name = head.relative_to(envs).parts[0]
    if not ENV_NAME_RE.match(env_name) or not _owns("envs", env_name, owner):
        return [], f"environment {env_name!r} is not one of yours"
    root = _tools_user_dir().resolve()
    out = [str(head)]
    for arg in command[1:]:
        if _FLAG_RE.match(arg):
            out.append(arg)
            continue
        tid, target, why = _tool_target(arg, root)
        if why or tid != tool_id or target is None:
            return [], f"argument {arg[:60]!r} is not one of {tool_id}'s own files"
        out.append(str(target))
    return out, None


def _merge_would_refuse(tool_id: str, command: list[str], tools: list[Any], km: Any) -> str | None:
    """Why the next merge would not wire this server, in the merger's own words; None when it will.

    register_tool answered ok for a block the merge then skipped -- a shipped or peer function-name
    clash, or a script that was never written -- so the model was told the tool was registered and
    then met a NameError (u14-mcp-wiring-17). The same predicate the merge and the resync pruner use.
    """
    for arg in command[1:]:
        if not _FLAG_RE.match(arg) and not os.path.isfile(arg):
            return f"{arg} does not exist yet -- write the tool's files (write_tool_files) before registering it"
    try:
        import yaml

        from spatialomicsgym.mcp_config_path import find_mcp_config
        from spatialomicsgym.mcp_user_config import shipped_identity, user_function_names, user_server_skip_reason

        base = find_mcp_config()
        shipped = (yaml.safe_load(base.read_text(encoding="utf-8")) or {}) if base else {}
        shipped_servers = shipped.get("mcp_servers") if isinstance(shipped, dict) else None
        names, functions = shipped_identity(shipped_servers if isinstance(shipped_servers, dict) else {})
        peers: set[str] = set()
        # The file register_tool is about to write -- knowledge_manager's per-call answer, which
        # honours SOG_MCP_USER_CONFIG like every reader of it; the bare constant did not (hunt
        # 2026-09-30, u14-mcp-wiring-7). A stand-in module with only the constant still works.
        resolve = getattr(km, "mcp_config_user_path", None)
        user_file = Path(resolve()) if callable(resolve) else Path(km.MCP_CONFIG_USER)
        if user_file.is_file():
            user = yaml.safe_load(user_file.read_text(encoding="utf-8")) or {}
            servers = user.get("mcp_servers") if isinstance(user, dict) else None
            for other, meta in (servers if isinstance(servers, dict) else {}).items():
                if other != f"user_{tool_id}":
                    peers |= user_function_names(meta)
    except Exception:
        return None  # the check could not be made; the merge still decides, and says so in its log
    skip = user_server_skip_reason(f"user_{tool_id}", {"tools": tools}, names, functions, peers)
    return skip[1] if skip is not None and skip[0] != "disabled" else None


def register_tool(
    tool_id: str, config_block: dict[str, Any], install_log_entry: dict[str, Any], *, _owner: str = ""
) -> dict[str, Any]:
    """Put ``user_<tool_id>`` into ``mcp_config_user.yaml`` and its entry into the install log."""
    from tools_user import knowledge_manager as _km

    tool_id = str(tool_id or "").strip()
    if not TOOL_ID_RE.match(tool_id):
        return _refuse("register_tool", _owner, tool_id, "tool_id must be a lowercase identifier (letters, digits, _)")
    if not isinstance(config_block, dict):
        return _refuse("register_tool", _owner, tool_id, "config_block must be a mapping (the server block)")
    command, why = _valid_command(config_block.get("command"), tool_id, _owner)
    if why:
        return _refuse("register_tool", _owner, tool_id, why)
    tools = config_block.get("tools")
    if not isinstance(tools, list) or not tools:
        return _refuse(
            "register_tool",
            _owner,
            tool_id,
            "config_block.tools must list the tools -- a server without tools: is never run to discover them",
        )
    from spatialomicsgym.agent.mcp_integration import is_bindable_tool_name

    for entry in tools:
        if not isinstance(entry, dict) or not str(entry.get("spatialomicsgym_name") or "").strip():
            return _refuse("register_tool", _owner, tool_id, "every tools[] entry needs a spatialomicsgym_name")
        if not is_bindable_tool_name(entry.get("spatialomicsgym_name")):
            return _refuse(
                "register_tool",
                _owner,
                tool_id,
                f"spatialomicsgym_name {entry.get('spatialomicsgym_name')!r} must be a Python identifier that is not "
                "a keyword, a builtin or a common REPL alias (it is bound into every account's REPL)",
            )
    env_block = config_block.get("env") or {}
    if not isinstance(env_block, dict):
        return _refuse("register_tool", _owner, tool_id, "config_block.env must be a mapping")
    try:
        from spatialomicsgym.redaction import looks_like_credential
    except Exception:  # pragma: no cover

        def looks_like_credential(_n: str, _v: Any) -> bool:
            return False

    clean_env: dict[str, str] = {}
    for key, value in env_block.items():
        if not isinstance(key, str) or not _ENV_KEY_RE.match(key) or not isinstance(value, str):
            return _refuse(
                "register_tool", _owner, tool_id, "config_block.env must map UPPER_CASE names to plain strings"
            )
        if "${" in value or "$(" in value or looks_like_credential(key, value):
            return _refuse(
                "register_tool",
                _owner,
                tool_id,
                f"config_block.env[{key!r}] may not name a placeholder or carry a credential",
            )
        clean_env[key] = value
    if not isinstance(install_log_entry, dict) or str(install_log_entry.get("tool_id") or "") != tool_id:
        return _refuse("register_tool", _owner, tool_id, "install_log_entry must be a mapping whose tool_id matches")
    unwired = _merge_would_refuse(tool_id, command, tools, _km)
    if unwired:
        return _refuse("register_tool", _owner, tool_id, unwired)
    taken = _claim("tools", tool_id, _owner)
    if taken:
        return _refuse("register_tool", _owner, tool_id, taken)
    block = {
        "enabled": bool(config_block.get("enabled", True)),
        "command": command,
        "env": clean_env,
        "tools": [dict(t) for t in tools],
    }
    if isinstance(config_block.get("description"), str):
        block["description"] = config_block["description"][:500]
    try:
        _km._restore_mcp_config_entry(tool_id, block)
        entries = _km._read_install_log()
        entry = {k: v for k, v in dict(install_log_entry).items() if k != "owner"}
        entry["owner"] = _owner
        entry.setdefault("status", "active")
        kept = [e for e in entries if str(e.get("tool_id")) != tool_id]
        kept.append(entry)
        _km._write_install_log(kept)
    except Exception as exc:
        return _refuse("register_tool", _owner, tool_id, f"could not register: {type(exc).__name__}: {exc}")
    _audit(
        "register_tool",
        _owner,
        tool_id,
        {"status": "ok", "server": f"user_{tool_id}", "tools": [t.get("spatialomicsgym_name") for t in tools]},
    )
    return {"status": "ok", "server": f"user_{tool_id}", "tool_id": tool_id}
