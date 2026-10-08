"""One-file transplant bundles: ``sog-setup pack`` / ``sog-setup unpack``.

The git repo carries the platform; it does NOT carry the machine-grown layer a working
deployment accumulates: user-created tools under ``agent/tools_user/`` (workers, portals, env
recipes), their ``agent/MCP_server/mcp_config_user.yaml`` wiring, the ``.sog_setup`` run state
and logs, the curated ``.knowledge`` corpus, and the ``user_*`` conda envs themselves.
``pack`` folds all of that (plus the exact working-tree code it ran against) into ONE
deterministic ``tar.gz``; ``unpack`` replays it onto another checkout: place the user
layer, rebuild the ``user_*`` envs from their recipes, rebase the user MCP config for the
new box (:func:`mcp_resolver.rebase_user_config`), and report. It composes with the
wizard rather than replacing it — built-in tool envs and LLM keys stay ``sog-setup``'s
resumable job on the destination machine.

Layout inside the archive (single top-level directory, ``MANIFEST.json`` always the
first member)::

    <name>/MANIFEST.json                     provenance + per-file sha256 + gate summary
    <name>/code/<repo-relative path>         the working tree of `git ls-files`
    <name>/userlayer/tools_user/...          user tools: workers, portals, recipes
    <name>/userlayer/MCP_server/mcp_config_user.yaml
    <name>/state/setup_state.json            provenance only — never applied on unpack
    <name>/logs/...                          .sog_setup/logs (the wizard's run journals)
    <name>/repos/...                         --with-repos: cloned source trees, .git pruned
    <name>/envs/<env>.tar.gz                 --with-conda-pack: binary env archives

``userlayer/`` paths (``tools_user/…``, ``MCP_server/mcp_config_user.yaml``, and ``repos/``'s
``tools_user/repos/…``) are relative to the checkout's AGENT part (``agent/``), not its root, so a
bundle packed before the ``agent/`` re-layout and one packed after it read the same, and unpack lands
them in the destination's agent part (:func:`_placed_at`) whichever layout that checkout has.

Determinism: members are sorted, tar metadata is normalized (uid/gid 0, mtime 0, mode
0644/0755), and the gzip wrapper is built by hand with ``mtime=0`` and an empty embedded
filename — ``tarfile``'s own ``w:gz`` stamps the wall clock into the stream. The ONLY
timestamp in a bundle is ``created_at`` inside MANIFEST.json, taken from :func:`_now` so
tests can pin the clock and assert byte-identical output.

Secrets policy (two layers, no bypass flag by design):

* **Layer A — paths.** A deny list (``.env*`` except the committed ``.env.example``
  template, ``llm_keys.json``, ``agent/tools_user/.memory/`` and any ``.secret``, key material
  like ``id_rsa*``/``*.pem``, every ``.git`` tree, setup backups/tmp/bundles, the huge
  local-only ``_live_out``/``_archive``/``_safety`` trees, config ``.bak_*`` copies,
  locks, the usage log) plus a positive assertion that every member sits under a known
  component prefix. A hit REFUSES the pack and names each path.
* **Layer B — contents.** Text files in the user layer / state / logs / knowledge (and
  config-ish files in ``code/`` outside the fixture-heavy ``test/``, ``agent/benchmarks/``
  and ``huggingface_data/`` trees, where planted fake tokens are test material) are
  line-scanned with :func:`capture._scrub_export_secrets` (registered secrets, anaconda
  ``/t/<token>/`` channels, URL userinfo, credential query params) and
  :func:`spatialomicsgym.redaction.looks_like_credential`. A hit refuses the pack and
  prints ``file:line`` and the matched NAME — never the value. ``.env.example`` is a
  key TEMPLATE, so only value-shape detection applies there (every name in it is
  credential-ish by design, and placeholders are not leaks). The ``envs/`` tier
  (``--with-conda-pack``) is binary env archives and is NOT content-scanned: pack an env
  only if nothing in it (``conda env config vars``, an ``activate.d`` script) holds a secret.

Stdlib + PyYAML + sibling setup modules only; conda/provision imports stay lazy so
``pack --dry-run`` works on a box with no conda at all.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from . import constants
from .session_log import redact

SCHEMA_VERSION = 1

#: Every member (bar MANIFEST.json) must live under one of these — the positive half of
#: the Layer-A gate: a path that reaches the writer without a component claiming it is a
#: selection bug, and refusing is cheaper than shipping a mystery file.
COMPONENT_PREFIXES = ("code/", "userlayer/", "state/", "logs/", "repos/", "envs/")

#: Prefixes that never enter a bundle, whatever selected them. Matched against the path and against
#: its agent-part spelling (``agent/`` stripped), so ``tools_user/.memory/`` denies both coordinates.
_DENY_PREFIXES = (
    ".git/",
    ".sog_setup/backups/",
    ".sog_setup/tmp/",
    ".sog_setup/bundles/",
    "tools_user/.memory/",
)

#: Directory NAMES denied anywhere in a path (all are local-only working trees; none is
#: tracked — verified against `git ls-files` before these went segment-wide).
_DENY_SEGMENTS = frozenset({".git", "__pycache__", "_live_out", "_archive", "_safety"})

#: ``code/`` suffixes worth a Layer-B content scan (config-ish text; source trees are
#: covered by review + the path rules, and fixture trees deliberately contain fakes).
_CONFIGISH_SUFFIXES = frozenset({".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".conf"})

#: ``code/`` prefixes excluded from the content scan: their yaml/json fixtures PIN the
#: scrubbers with planted fake tokens, so scanning them can only produce false refusals
#: of tracked, already-public test material.
_SCAN_EXCLUDED_CODE_PREFIXES = ("test/", "agent/benchmarks/", "huggingface_data/")

#: The agent part's prefix in a repo-relative path (``agent/tools_user/x``).
_AGENT_PREFIX = constants.AGENT_DIRNAME + "/"


def _agent_area(rel: str) -> str:
    """``rel`` in agent-part coordinates: ``agent/tools_user/x`` -> ``tools_user/x``; others unchanged."""
    return rel[len(_AGENT_PREFIX) :] if rel.startswith(_AGENT_PREFIX) else rel


def _in_agent_area(rel: str) -> bool:
    """Whether a ``userlayer/``/``repos/`` coordinate lives in the agent part (vs the repo root)."""
    return rel.startswith("tools_user/") or rel == constants.USER_MCP_CONFIG_REL


def _placed_at(dest: Path, rel: str) -> Path:
    """Where a ``userlayer/``/``repos/`` member with coordinate ``rel`` lands under ``dest``.

    Agent-area coordinates (``tools_user/…``, the user MCP config) go under ``dest``'s agent part --
    ``dest/agent`` on a re-laid-out checkout, ``dest`` itself on one still laid out flat
    (:func:`constants.agent_root`); root-level ``repos/`` and ``vendor_*`` stay under ``dest``.
    """
    return (constants.agent_root(dest) / rel) if _in_agent_area(rel) else (dest / rel)


_SCAN_MAX_BYTES = 5 * 1024 * 1024

#: Values under a credential-ish NAME that are templates, not leaks: ``${VAR}`` refs,
#: ``<fill-me-in>`` brackets, and the usual placeholder words. Layer B skips these so a
#: committed example config can be packed; a real token matches none of them.
_PLACEHOLDER_RE = re.compile(
    r"^(\$\{[^}]*\}|<[^>]*>|your[-_].*|change_?me.*|placeholder.*|example.*|xxx+|\.{3}|todo|null|none|unset)$",
    re.IGNORECASE,
)

#: ``NAME=value`` / ``name: value`` / ``"name": "value"`` pairs in scanned text (the
#: shapes .env files, yaml scalars and json lines all reduce to — the optional closing
#: quote before the separator is what lets a JSON key match at all).
_KV_RE = re.compile(r"""([A-Za-z_][A-Za-z0-9_.\-]{1,63})["']?\s*[:=]\s*["']?([^\s"',]+)""")

#: Digests and name lists that Layer B used to read as credentials (hunt 2026-09-30, u36-setup-install-2).
#: ``SECRET_RE`` takes any 32+ hex run for a key, and ``looks_like_credential`` any long value under an
#: ``*_api_key*`` name — so this repository's own tracked ``know_how/packs/MANIFEST.yaml`` (``commit:``,
#: ``sha256:``, ``licence_sha256:`` and ``may_name_api_keys: [NCBI_API_KEY]``) refused every pack, with no
#: bypass by design. A value under a digest NAME that has exactly a digest's SHAPE (hex of an md5 / sha1 /
#: sha256 / sha512 length, or a Subresource-Integrity ``sha512-<base64>``) is a hash; a bracketed list of
#: ENV_VAR names is a list of names. Both stay subject to the value-shape check, so a real key placed under
#: either kind of name — which has neither shape — is still caught.
_DIGEST_NAME_RE = re.compile(
    r"(?:^|[_.\-])(?:commit|sha|sha1|sha224|sha256|sha384|sha512|md5|digest|hash|checksum|integrity)$", re.I
)
_HEX_DIGEST_RE = re.compile(
    r"^(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{40}|[0-9a-fA-F]{56}|[0-9a-fA-F]{64}|[0-9a-fA-F]{96}|[0-9a-fA-F]{128})$"
)
#: A Subresource-Integrity hash (``sha512-<base64>``, npm's ``integrity`` field). It is never a credential,
#: and its base64 body can contain ``/t/`` — the anaconda token-channel shape the scrubber masks — by
#: chance, as ``frontend/package-lock.json`` does; it is blanked before the scrubber leg reads a line.
_SRI_RE = re.compile(r"\bsha(?:256|384|512)-[A-Za-z0-9+/]{40,}={0,2}")
_NAME_LIST_RE = re.compile(r"^\[[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\]?$")

#: User-tool artifacts the default tier carries (beside the tracked helpers, which ship
#: in ``code/`` already).
_USERLAYER_PATTERNS = ("*_worker.py", "*_worker.R", "*_mcp_server.py", "*_env.yaml", "*_requirements.txt")

_ENV_YAML_SUFFIX = "_env.yaml"


def _now() -> datetime:
    """The bundle's single clock — a seam so tests can pin ``created_at`` and prove the
    rest of the byte stream carries no timestamp at all."""
    return datetime.now(UTC)


# --------------------------------------------------------------------------- selection


@dataclass(frozen=True)
class Member:
    """One file headed into the archive.

    ``rel`` is the posix coordinate the deny rules and unpack placement both reason in:
    REPO-relative for ``code/``, state and logs, AGENT-part-relative for the user layer
    (``tools_user/…``; see the module docstring and :func:`_placed_at`); ``arc`` is where it
    lives inside the bundle. The two differ only by the component prefix (and for state/logs,
    whose canonical home is ``.sog_setup/`` however the state-dir seam is pointed)."""

    arc: str
    src: Path
    rel: str
    component: str
    link: str | None = None


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip() or 'failed'}")
    return proc.stdout


def _git_facts(root: Path) -> dict:
    commit = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    porcelain = [ln for ln in _git(root, "status", "--porcelain").splitlines() if ln.strip()]
    untracked = sum(1 for ln in porcelain if ln.startswith("??"))
    return {
        "commit": commit,
        "short": commit[:8],
        "branch": branch,
        "dirty": bool(porcelain),
        "changed": len(porcelain) - untracked,
        "untracked": untracked,
    }


def _member_for(root: Path, rel: str, arc: str, component: str) -> Member | None:
    src = root / rel
    try:
        st = src.lstat()
    except OSError:
        return None  # listed by git but vanished from the tree — counted by the caller
    link = os.readlink(src) if stat.S_ISLNK(st.st_mode) else None
    if link is None and not stat.S_ISREG(st.st_mode):
        return None  # sockets/dirs never enter a bundle
    return Member(arc=arc, src=src, rel=rel, component=component, link=link)


def _select_code(root: Path) -> tuple[list[Member], dict, set[str]]:
    """The WORKING TREE of every tracked path (not ``git archive`` — a dirty tree packs
    what actually runs, and the manifest's dirty flag keeps that honest). Untracked-
    unignored files are skipped + counted, except the spec corpus
    (``install/recipes/tool_specs/**``), which a capture run may have refreshed without a commit
    and which unpack's env rebuild story depends on."""
    members: list[Member] = []
    vanished = 0
    tracked: set[str] = set()
    for rel in sorted(p for p in _git(root, "ls-files", "-z").split("\0") if p):
        tracked.add(rel)
        m = _member_for(root, rel, f"code/{rel}", "code")
        if m is None:
            vanished += 1
        else:
            members.append(m)
    untracked_skipped = 0
    forced: list[str] = []
    spec_prefix = constants.SPEC_DIRNAME.rstrip("/") + "/"
    for rel in sorted(p for p in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0") if p):
        if rel.startswith(spec_prefix):
            m = _member_for(root, rel, f"code/{rel}", "code")
            if m is not None:
                members.append(m)
                forced.append(rel)
        else:
            untracked_skipped += 1
    notes = {
        "tracked": len(tracked),
        "vanished": vanished,
        "untracked_skipped": untracked_skipped,
        "spec_files_forced": len(forced),
    }
    return members, notes, tracked


def _repo_rel(root: Path, agent: Path, rel: str) -> str:
    """The repo-relative spelling of agent-part coordinate ``rel`` (``agent/tools_user/x``)."""
    return (agent / rel).relative_to(root).as_posix()


def _select_userlayer(root: Path, tracked: set[str]) -> list[Member]:
    members: list[Member] = []
    agent = constants.agent_root(root)
    tools_user = agent / "tools_user"
    names: set[str] = set()
    if tools_user.is_dir():
        for pattern in _USERLAYER_PATTERNS:
            names.update(p.name for p in tools_user.glob(pattern))
        if (tools_user / "install_log.json").is_file():
            names.add("install_log.json")
    for name in sorted(names):
        rel = f"tools_user/{name}"  # agent-part coordinate (module docstring)
        if _repo_rel(root, agent, rel) in tracked:
            continue  # helper twins (base_mcp & co) ship in code/ — never duplicated
        m = _member_for(agent, rel, f"userlayer/{rel}", "userlayer")
        if m is not None:
            members.append(m)
    cfg_rel = constants.USER_MCP_CONFIG_REL
    if _repo_rel(root, agent, cfg_rel) not in tracked:
        m = _member_for(agent, cfg_rel, f"userlayer/{cfg_rel}", "userlayer")
        if m is not None:
            members.append(m)
    return members


def _walk_files(base: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _DENY_SEGMENTS)
        for name in sorted(filenames):
            out.append(Path(dirpath) / name)
    return out


def _select_knowledge(root: Path) -> list[Member]:
    """``--with-knowledge``: the curated ``.knowledge`` corpus under ``agent/tools_user/`` —
    archived inside ``userlayer/`` so unpack's one placement rule covers it, counted as
    its own component so the manifest tells the tiers apart."""
    agent = constants.agent_root(root)
    base = agent / "tools_user" / ".knowledge"
    members: list[Member] = []
    if not base.is_dir():
        return members
    for path in _walk_files(base):
        rel = path.relative_to(agent).as_posix()
        m = _member_for(agent, rel, f"userlayer/{rel}", "knowledge")
        if m is not None:
            members.append(m)
    return members


def _select_state() -> list[Member]:
    state_file = Path(constants.state_dir()) / "setup_state.json"
    if not state_file.is_file():
        return []
    return [Member(arc="state/setup_state.json", src=state_file, rel=".sog_setup/setup_state.json", component="state")]


def _select_logs() -> list[Member]:
    logs = Path(constants.logs_dir())
    members: list[Member] = []
    if not logs.is_dir():
        return members
    for path in _walk_files(logs):
        tail = path.relative_to(logs).as_posix()
        members.append(
            Member(
                arc=f"logs/{tail}",
                src=path,
                rel=f".sog_setup/{constants.LOGS_SUBDIR}/{tail}",
                component="logs",
            )
        )
    return members


def _select_repos(root: Path) -> tuple[list[Member], dict]:
    """``--with-repos``: the cloned source trees (``agent/tools_user/repos/**`` plus any
    root-level ``repos/``/``vendor_*``), with every nested ``.git`` pruned — remotes can
    embed credentials. The manifest records each clone's scrubbed origin URL + HEAD sha
    so machine B can re-clone instead of trusting the snapshot. ``tools_user/repos`` paths are
    agent-part coordinates, the root-level trees repo-root ones (:func:`_placed_at`)."""
    agent = constants.agent_root(root)
    bases: list[tuple[Path, Path]] = []  # (tree, the root its coordinates are relative to)
    for cand, anchor in ((agent / "tools_user" / "repos", agent), (root / "repos", root)):
        if cand.is_dir():
            bases.append((cand, anchor))
    bases.extend((p, root) for p in sorted(root.glob("vendor_*")) if p.is_dir())
    members: list[Member] = []
    origins: dict[str, dict] = {}
    for base, anchor in bases:
        for child in sorted(p for p in base.iterdir() if p.is_dir()):
            git_dir = child / ".git"
            if git_dir.exists():
                origins[child.relative_to(anchor).as_posix()] = _repo_origin(child)
        for path in _walk_files(base):
            rel = path.relative_to(anchor).as_posix()
            m = _member_for(anchor, rel, f"repos/{rel}", "repos")
            if m is not None:
                members.append(m)
    return members, origins


def _repo_origin(repo: Path) -> dict:
    from . import capture  # lazy: reuse the committed scrubbers, nothing heavier

    out = {"origin": "unknown", "head": "unknown"}
    try:
        url = _git(repo, "config", "--get", "remote.origin.url").strip()
        out["origin"] = capture._scrub_export_secrets(url)
    except Exception:
        pass
    try:
        out["head"] = _git(repo, "rev-parse", "HEAD").strip()
    except Exception:
        pass
    return out


# ------------------------------------------------------------------------ secrets gate


def _deny_reason(rel: str) -> str | None:
    """Layer A: why ``rel`` (repo-relative or agent-part posix) may never enter a bundle — or None."""
    area = _agent_area(rel)
    parts = rel.split("/")
    base = parts[-1]
    if base == ".env" or (base.startswith(".env") and base != ".env.example"):
        return "environment file (holds live keys)"
    if base == "llm_keys.json":
        return "saved-key vault"
    if base == ".secret":
        return "HMAC/secret key material"
    if base.startswith("id_rsa"):
        return "SSH private key material"
    if base.endswith(".pem"):
        return "PEM key material"
    for segment in parts[:-1]:
        if segment in _DENY_SEGMENTS:
            return f"local-only tree ({segment}/)"
    for prefix in _DENY_PREFIXES:
        if rel.startswith(prefix) or area.startswith(prefix):
            return f"local-only tree ({prefix})"
    if area.startswith("MCP_server/") and ".bak_" in base:
        return "config backup copy"
    if area.startswith("tools_user/") and base.endswith(".lock"):
        return "lock file"
    if area == "tools_user/usage_log.jsonl":
        return "usage log"
    return None


#: A packed env archive's member name: ``envs/<env>.tar.gz``, the env name restore turns into a prefix.
_ENV_ARC_RE = re.compile(r"^envs/[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz$")


def _placement_problem(component: str, rel: str, link: str | None = None) -> str | None:
    """Why unpack must not place ``rel`` (repo-relative) for ``component`` — or None.

    unpack places every ``userlayer/`` and ``repos/`` member at ``<dest>/<rel>``, and the manifest a
    bundle carries is written by whoever made the bundle, so every sha256 in a crafted one matches
    (hunt 2026-09-30, u36-setup-install-1). Without these checks ``userlayer/../outside/x``, an absolute
    ``userlayer//home/u/.bashrc`` (``Path``'s ``/`` lets an absolute right side replace ``dest``) and a
    symlink member followed by a member written through it all landed outside the destination — even
    under ``--no-envs``, where the user expects only file placement. A placed path must be a plain
    relative path inside the area that component owns (the areas ``_select_userlayer`` /
    ``_select_knowledge`` / ``_select_repos`` read from), and a symlink must stay inside the
    destination.

    That symlink rule used to be ``posixpath.normpath`` of the link's target, which reads ``..`` AFTER a
    directory name lexically -- but that name can itself be a symlink, so ``a -> b/../../outside`` with
    ``b -> ../../..`` passed as "inside" and resolved outside (hunt 2026-09-30, u36-setup-install-1,
    reviewer's chained-link repro). A link target may now climb only at its START (``../../x``): then it
    climbs from the link's own directory and every later segment descends, so no link placed after it
    can move where it lands. Whatever this check cannot see (the destination's own symlinks, a
    case-insensitive filesystem) the write-time check in :func:`_place` refuses."""
    parts = rel.split("/")
    if rel.startswith("/") or any(part in ("", ".", "..") for part in parts):
        return "absolute or non-normalized path ('..', '.', or an empty segment)"
    if component == "userlayer":
        if not (rel.startswith("tools_user/") or rel == constants.USER_MCP_CONFIG_REL):
            return "user-layer member outside tools_user/ and the user MCP config"
    elif component == "repos":
        in_area = rel.startswith(("tools_user/repos/", "repos/")) or (len(parts) > 1 and parts[0].startswith("vendor_"))
        if not in_area:
            return "repos member outside tools_user/repos/, repos/ and vendor_*/"
    if link is not None:
        if link.startswith("/"):
            return f"symlink to an absolute path ({link})"
        if not link:
            return "symlink with an empty target (symlink(2) refuses it mid-unpack)"
        hops = [seg for seg in link.split("/") if seg not in ("", ".")]
        climbs = 0
        while climbs < len(hops) and hops[climbs] == "..":
            climbs += 1
        if ".." in hops[climbs:]:
            return f"symlink whose target climbs ('..') after naming a directory ({link})"
        if climbs > len(parts) - 1:
            return f"symlink pointing outside the destination ({link})"
    return None


def _links_passed_through(entries: list[tuple[str, bool]]) -> list[str]:
    """The placed paths in ``entries`` (``(rel, is_symlink)`` pairs) that lie UNDER a symlink placed by the
    same bundle -- a write through that link would land wherever it points.

    Keyed on the placed path, not the archive path (hunt 2026-09-30, u36-setup-install-1): ``userlayer/``
    and ``repos/`` both place at ``<dest>/<rel>``, so a link from one component and a member under it from
    the other land on one tree, and an archive-path comparison never matched them. Case-folded, because a
    case-insensitive destination (macOS) resolves ``A/x`` through the link ``a``."""
    links = {rel.casefold() for rel, is_link in entries if is_link}
    through: list[str] = []
    for rel, _is_link in entries:
        if any(anc in links for anc in _ancestors(rel.casefold())):
            through.append(rel)
    return through


def _members_under_files(entries: list[tuple[str, bool]]) -> list[str]:
    """The placed paths in ``entries`` that lie under a FILE member of the same bundle. No tree pack walks
    can hold both, and unpack crashed partway (NotADirectoryError) on a crafted one, after earlier members
    were placed (hunt 2026-09-30, u36-setup-install-1 follow-on, found by fuzzing)."""
    files = {rel for rel, is_link in entries if not is_link}
    return [rel for rel, _ in entries if any(anc in files for anc in _ancestors(rel))]


def _ancestors(rel: str) -> list[str]:
    parts = rel.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def _rel_of_arc(arc: str) -> str | None:
    """Map an archive path back to its repo-relative coordinate (None for envs/)."""
    if arc == "MANIFEST.json":
        return None
    component, _, tail = arc.partition("/")
    if component in ("code", "userlayer", "repos"):
        return tail
    if component == "state":
        return f".sog_setup/{tail}"
    if component == "logs":
        return f".sog_setup/{constants.LOGS_SUBDIR}/{tail}"
    return None  # envs/ tarballs have no repo coordinate


def _gate_paths(members: list[Member]) -> list[tuple[str, str]]:
    violations: list[tuple[str, str]] = []
    for m in members:
        reason = _deny_reason(m.rel)
        if reason:
            violations.append((m.rel, reason))
        elif not m.arc.startswith(COMPONENT_PREFIXES):
            violations.append((m.arc, "outside every known bundle component"))
        else:
            # The rule unpack's verify applies (u36-setup-install-1), checked here so a user-layer
            # symlink that points off this checkout is refused by name before anything is written,
            # rather than failing the bundle's own self-verify afterwards.
            arc_component = m.arc.split("/", 1)[0]
            if arc_component in ("userlayer", "repos"):
                problem = _placement_problem(arc_component, m.rel, m.link)
                if problem:
                    violations.append((m.rel, f"{problem} -- replace the link with the file it points at"))
    placed = [(m.rel, m.link is not None) for m in members if m.arc.startswith(("userlayer/", "repos/"))]
    for rel in _links_passed_through(placed):
        violations.append((rel, "lies under a symlink this bundle also places -- replace the link with its files"))
    return violations


def _scan_scope(m: Member) -> bool:
    if m.link is not None:
        return False
    if m.component in ("userlayer", "knowledge", "state", "logs"):
        return True
    if m.component != "code":
        return False
    if m.rel.startswith(_SCAN_EXCLUDED_CODE_PREFIXES):
        return False
    base = m.rel.rsplit("/", 1)[-1]
    if base == ".env.example":
        return True
    return "." in base and ("." + base.rsplit(".", 1)[-1]).lower() in _CONFIGISH_SUFFIXES


def _gate_contents(members: list[Member]) -> tuple[list[tuple[str, int, str]], int]:
    """Layer B: line-scan the in-scope text members. Findings carry the file, the line
    number and WHAT tripped — never the flagged value itself."""
    from spatialomicsgym import redaction

    from . import capture  # lazy import: brings the scrubber regex set + session_log

    findings: list[tuple[str, int, str]] = []
    scanned = 0
    for m in members:
        if not _scan_scope(m):
            continue
        try:
            if m.src.stat().st_size > _SCAN_MAX_BYTES:
                continue
            data = m.src.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue  # binary — the path rules own these
        scanned += 1
        template_only = m.rel.rsplit("/", 1)[-1] == ".env.example"
        for lineno, raw_line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            line = _SRI_RE.sub("<sri-hash>", raw_line)
            if capture._scrub_export_secrets(line) != line:
                findings.append((m.rel, lineno, "credential-shaped URL/token (scrubber would mask it)"))
                continue
            for match in _KV_RE.finditer(line):
                name, value = match.group(1), match.group(2)
                if _DIGEST_NAME_RE.search(name) and _HEX_DIGEST_RE.match(value):
                    continue  # a commit / file digest, not a key (u36-setup-install-2)
                if value.startswith("//"):
                    # `scheme://…` parses as name=scheme, value=`//host/…` — a URL fragment, not an
                    # assignment. Every captured env recipe carries pip VCS pins
                    # (`git+https://host/repo.git@<40-hex-sha>`) whose commit sha is SECRET_RE-shaped
                    # by construction. URL-EMBEDDED credentials stay covered: userinfo (`://user:tok@`),
                    # query tokens and registered index secrets are the scrubber leg's job above.
                    continue
                if _PLACEHOLDER_RE.match(value):
                    continue
                if template_only or _NAME_LIST_RE.match(value):
                    # A template, or a list of variable NAMES (``may_name_api_keys: [NCBI_API_KEY]``):
                    # only the value's own shape can make it a secret (u36-setup-install-2).
                    hit = bool(redaction.SECRET_RE.search(value))
                else:
                    hit = redaction.looks_like_credential(name, value)
                if hit:
                    findings.append((m.rel, lineno, f"value of {name!r} looks like a credential"))
                    break
    return findings, scanned


# ---------------------------------------------------------------- manifest + tar writer


def _hash_stream(fh) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return _hash_stream(fh)


def _configured_key_names(root: Path) -> list[str]:
    """The variable NAMES machine A's ``.env`` configures — names only, so unpack can
    tell machine B which keys to re-enter. Values never leave this function."""
    env_path = root / ".env"
    names: list[str] = []
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return names
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() and value.strip():
            names.append(name.strip())
    return sorted(set(names))


def _file_entry(m: Member) -> dict:
    if m.link is not None:
        return {"path": m.arc, "link": m.link}
    st = m.src.stat()
    return {
        "path": m.arc,
        "sha256": _hash_file(m.src),
        "size": st.st_size,
        "mode": "0755" if st.st_mode & 0o100 else "0644",
    }


def _build_manifest(
    *,
    name: str,
    root: Path,
    git_facts: dict,
    members: list[Member],
    notes: dict,
    tiers: dict,
    repo_origins: dict,
    scanned: int,
) -> dict:
    components: dict[str, dict] = {}
    for m in members:
        slot = components.setdefault(m.component, {"files": 0, "bytes": 0})
        slot["files"] += 1
        if m.link is None:
            slot["bytes"] += m.src.stat().st_size
    user_recipes = sorted(m.rel.rsplit("/", 1)[-1] for m in members if m.rel.endswith(_ENV_YAML_SUFFIX))
    builtin_recipes = sum(
        1 for m in members if m.component == "code" and m.rel.startswith(constants.SPEC_DIRNAME + "/env/")
    )
    return {
        "schema": SCHEMA_VERSION,
        "bundle": name,
        "created_at": _now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": git_facts,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "tiers": tiers,
        "components": dict(sorted(components.items())),
        "selection_notes": notes,
        "env_recipes": {"builtin": builtin_recipes, "user": user_recipes},
        "repo_origins": repo_origins,
        "configured_key_names": _configured_key_names(root),
        "secrets_gate": {
            "layer_a": "clean",
            "layer_b": "clean",
            "scanned": scanned,
            "code_scan_excludes": list(_SCAN_EXCLUDED_CODE_PREFIXES),
        },
        "files": sorted((_file_entry(m) for m in members), key=lambda e: e["path"]),
    }


def _tarinfo(name: str, *, size: int = 0, mode: int = 0o644, link: str | None = None) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if link is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = link
        info.mode = 0o777
    else:
        info.size = size
        info.mode = mode
    return info


def _write_bundle(out: Path, topdir: str, members: list[Member], manifest: dict) -> None:
    """Deterministic writer: gzip built by hand (``mtime=0`` + empty embedded filename —
    the two fields ``w:gz`` would stamp per-run), GNU format for long paths, sorted
    members, normalized metadata, `.partial` + ``os.replace``."""
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    tmp = out.with_name(out.name + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
                    tar.addfile(
                        _tarinfo(f"{topdir}/MANIFEST.json", size=len(manifest_bytes)),
                        io.BytesIO(manifest_bytes),
                    )
                    for m in sorted(members, key=lambda m: m.arc):
                        name = f"{topdir}/{m.arc}"
                        if m.link is not None:
                            tar.addfile(_tarinfo(name, link=m.link))
                            continue
                        st = m.src.stat()
                        mode = 0o755 if st.st_mode & 0o100 else 0o644
                        with open(m.src, "rb") as fh:
                            tar.addfile(_tarinfo(name, size=st.st_size, mode=mode), fh)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)


def verify_bundle(path: Path) -> dict:
    """Re-open a written bundle and prove it: MANIFEST.json first, every member present
    in the manifest with a matching sha256, and the Layer-A rules re-run on the archive's
    own paths. Run by ``pack`` on its output and by ``unpack`` before any placement."""
    errors: list[str] = []
    with tarfile.open(path, mode="r:gz") as tar:
        names = tar.getnames()
        if not names:
            return {"ok": False, "errors": ["empty archive"], "members": 0}
        topdir = names[0].split("/", 1)[0]
        if names[0] != f"{topdir}/MANIFEST.json":
            errors.append("MANIFEST.json is not the first member")
        manifest = _read_manifest(tar)
        by_path = {e["path"]: e for e in manifest.get("files", [])}
        seen = 0
        members = tar.getmembers()
        # Placeable members (placed path, is a symlink), so a member written THROUGH a link of the same
        # bundle is refused -- matched on the placed path across both placing components (u36-setup-install-1).
        placeable: list[tuple[str, bool]] = []
        for m in members:
            m_top, _, m_arc = m.name.partition("/")
            if m_top == topdir and m_arc.startswith(("userlayer/", "repos/")):
                placeable.append((_rel_of_arc(m_arc) or "", m.issym()))
        through_link = set(_links_passed_through(placeable))
        under_file = set(_members_under_files(placeable))
        names_seen: set[str] = set()
        rels_seen: set[str] = set()
        for info in members:
            top, _, arc = info.name.partition("/")
            if not arc or arc == "MANIFEST.json":
                continue
            seen += 1
            if top != topdir:
                errors.append(f"member outside the bundle's top directory: {info.name}")
                continue
            # One member per name and one per placed path: a repeated name passed the count check whenever a
            # manifest entry was left out to make room for it, and the later copy silently replaced the earlier
            # one -- a link where a file was verified, or the reverse (hunt 2026-09-30, u36-setup-install-1).
            if info.name in names_seen:
                errors.append(f"duplicate member: {arc}")
                continue
            names_seen.add(info.name)
            entry = by_path.get(arc)
            if entry is None:
                errors.append(f"member not in manifest: {arc}")
                continue
            # Only regular files and symlinks are ever written by pack; a directory, FIFO, device or
            # hard-link member has no sha256 to check and no placement rule (u36-setup-install-1).
            if not (info.isfile() or info.issym()):
                errors.append(f"unsupported member type: {arc}")
                continue
            if not arc.startswith(COMPONENT_PREFIXES):
                errors.append(f"outside every known bundle component: {arc}")
                continue
            component = arc.split("/", 1)[0]
            rel = _rel_of_arc(arc)
            if rel:
                reason = _deny_reason(rel)
                if reason:
                    errors.append(f"denied path inside archive: {rel} ({reason})")
            if component in ("userlayer", "repos"):
                problem = _placement_problem(component, rel or "", info.linkname if info.issym() else None)
                if problem:
                    errors.append(f"unsafe placement: {arc} ({problem})")
                if rel in through_link:
                    errors.append(f"member placed through a symlink in the same bundle: {arc}")
                if rel in under_file:
                    errors.append(f"member placed under a file of the same bundle: {arc}")
                if rel in rels_seen:
                    errors.append(f"two members place the same path: {rel}")
                rels_seen.add(rel or "")
            elif component == "envs" and not _ENV_ARC_RE.match(arc):
                errors.append(f"env archive name is not envs/<env>.tar.gz: {arc}")
            if info.issym():
                if entry.get("link") != info.linkname:
                    errors.append(f"symlink target mismatch: {arc}")
            elif "sha256" not in entry:
                errors.append(f"manifest carries no sha256 for: {arc}")
            else:
                with tar.extractfile(info) as fh:
                    if _hash_stream(fh) != entry["sha256"]:
                        errors.append(f"sha256 mismatch: {arc}")
        if seen != len(by_path):
            errors.append(f"manifest lists {len(by_path)} files, archive holds {seen}")
    return {"ok": not errors, "errors": errors, "members": seen}


def _read_manifest(tar: tarfile.TarFile) -> dict:
    for info in tar.getmembers():
        if info.name.partition("/")[2] == "MANIFEST.json":
            fh = tar.extractfile(info)
            doc = json.loads(fh.read().decode("utf-8"))
            if not isinstance(doc, dict):
                raise RuntimeError("MANIFEST.json is not a mapping")
            return doc
    raise RuntimeError("bundle has no MANIFEST.json")


# ---------------------------------------------------------------------- conda-pack tier


def _conda_pack_probe() -> list[str] | None:
    """The command that runs conda-pack here, or None. Checked up front so a
    ``--with-conda-pack`` request fails BEFORE any selection work, with the install
    hint — this box simply may not have it."""
    exe = shutil.which("conda-pack")
    if exe:
        return [exe]
    probe = subprocess.run(
        [sys.executable, "-m", "conda_pack", "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "conda_pack"]
    return None


def _pack_envs(env_names: list[str], pack_cmd: list[str], workdir: Path) -> tuple[list[Member], list[str]]:
    members: list[Member] = []
    failures: list[str] = []
    workdir.mkdir(parents=True, exist_ok=True)
    for env in env_names:
        out = workdir / f"{env}.tar.gz"
        proc = subprocess.run(
            [*pack_cmd, "-n", env, "-o", str(out), "--force"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            failures.append(f"{env}: {tail[-1] if tail else 'conda-pack failed'}")
            continue
        members.append(Member(arc=f"envs/{env}.tar.gz", src=out, rel=f"envs/{env}.tar.gz", component="envs"))
    return members, failures


# ------------------------------------------------------------------------------- pack


def _human(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}GB"


def pack(
    out: str | None = None,
    *,
    with_knowledge: bool = False,
    with_repos: bool = False,
    conda_pack_envs: list[str] | None = None,
    allow_dirty: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    root = Path(constants.repo_root())
    try:
        git_facts = _git_facts(root)
    except Exception as exc:
        print(f" -- pack needs a git checkout to select code from: {exc}", file=sys.stderr)
        return 1

    if git_facts["dirty"] and not allow_dirty:
        print(
            f" -- the working tree is DIRTY ({git_facts['changed']} changed, "
            f"{git_facts['untracked']} untracked). A bundle should be reproducible from a\n"
            "    commit; commit first, or pass --allow-dirty to pack the tree as it stands\n"
            "    (the manifest will record the dirty state).",
            file=sys.stderr,
        )
        return 1

    conda_pack_envs = list(conda_pack_envs or [])
    pack_cmd: list[str] | None = None
    if conda_pack_envs:
        pack_cmd = _conda_pack_probe()
        if pack_cmd is None:
            print(
                " -- --with-conda-pack needs the conda-pack tool, which is not installed here.\n"
                "    Install it first (e.g. `pip install conda-pack` in your base env) and re-run,\n"
                "    or drop the flag: unpack rebuilds user envs from their bundled recipes anyway.",
                file=sys.stderr,
            )
            return 1
        from .envtools import Conda

        conda = Conda()
        unknown = [e for e in conda_pack_envs if not conda.env_exists(e)]
        if unknown:
            print(f" -- --with-conda-pack names env(s) that do not exist: {', '.join(unknown)}", file=sys.stderr)
            return 1

    try:  # live index-URL credentials become registered secrets → Layer B catches them
        from . import session_log

        session_log.register_index_url_secrets()
    except Exception:
        pass

    code_members, notes, tracked = _select_code(root)
    members = list(code_members)
    members.extend(_select_userlayer(root, tracked))
    if with_knowledge:
        members.extend(_select_knowledge(root))
    members.extend(_select_state())
    members.extend(_select_logs())
    repo_origins: dict[str, dict] = {}
    if with_repos:
        repo_members, repo_origins = _select_repos(root)
        members.extend(repo_members)

    violations = _gate_paths(members)
    if violations:
        print(" -- REFUSED: the selection contains paths a bundle must never carry:", file=sys.stderr)
        for rel, reason in violations[:20]:
            print(f"      {rel}  ({reason})", file=sys.stderr)
        if len(violations) > 20:
            print(f"      ... and {len(violations) - 20} more", file=sys.stderr)
        return 1

    findings, scanned = _gate_contents(members)
    if findings:
        print(
            " -- REFUSED: content scan found credential-shaped values (file:line -- what tripped):",
            file=sys.stderr,
        )
        for rel, lineno, why in findings[:20]:
            print(f"      {rel}:{lineno} -- {why}", file=sys.stderr)
        if len(findings) > 20:
            print(f"      ... and {len(findings) - 20} more", file=sys.stderr)
        print("    Move the secret out of the file (or into .env, which never packs) and re-run.", file=sys.stderr)
        return 1

    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    default_name = f"sog-bundle-{stamp}-{git_facts['short']}{'-dirty' if git_facts['dirty'] else ''}.tar.gz"
    out_path = Path(out) if out else Path(constants.state_dir()) / "bundles" / default_name
    topdir = out_path.name[: -len(".tar.gz")] if out_path.name.endswith(".tar.gz") else out_path.stem

    total = sum(m.src.stat().st_size for m in members if m.link is None)
    by_component: dict[str, int] = {}
    for m in members:
        if m.link is None:
            by_component[m.component] = by_component.get(m.component, 0) + m.src.stat().st_size
    print(f" -- bundle: {len(members)} files, {_human(total)} raw, from {git_facts['branch']}@{git_facts['short']}")
    for component in sorted(by_component):
        count = sum(1 for m in members if m.component == component)
        print(f"      {component:<10} {count:>6} files  {_human(by_component[component]):>10}")
    if notes["untracked_skipped"]:
        print(f"      (skipped {notes['untracked_skipped']} untracked file(s); commit or ignore them to include)")
    if git_facts["dirty"]:
        print("      WARNING: packing a DIRTY tree -- the manifest records it, but prefer a commit.")

    if dry_run:
        print(f" -- dry run: secrets gate clean ({scanned} file(s) content-scanned); nothing written.")
        print(f"    would write {out_path}")
        return 0

    if out_path.exists() and not force:
        print(f" -- {out_path} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1

    env_failures: list[str] = []
    if conda_pack_envs and pack_cmd:
        constants.ensure_state_dirs()
        env_members, env_failures = _pack_envs(
            conda_pack_envs, pack_cmd, Path(constants.state_dir()) / "tmp" / "conda_pack"
        )
        members.extend(env_members)
        for line in env_failures:
            print(f" -- conda-pack failed for {line}", file=sys.stderr)

    tiers = {
        "knowledge": with_knowledge,
        "repos": with_repos,
        "conda_pack": [m.arc.split("/")[-1][: -len(".tar.gz")] for m in members if m.component == "envs"],
    }
    manifest = _build_manifest(
        name=topdir,
        root=root,
        git_facts=git_facts,
        members=members,
        notes=notes,
        tiers=tiers,
        repo_origins=repo_origins,
        scanned=scanned,
    )
    try:
        _write_bundle(out_path, topdir, members, manifest)
    finally:
        # The conda-pack outputs are copies of whole envs (often GBs each) and the bundle now holds
        # them; nothing else ever deleted them, so every pack left one extra copy of every packed env
        # under .sog_setup/tmp (hunt 2026-09-30, u36-setup-install-12).
        for m in members:
            if m.component == "envs":
                m.src.unlink(missing_ok=True)

    report = verify_bundle(out_path)
    if not report["ok"]:
        print(" -- self-verify FAILED on the written bundle:", file=sys.stderr)
        for err in report["errors"][:10]:
            print(f"      {err}", file=sys.stderr)
        return 1
    print(f" -- wrote {out_path} ({_human(out_path.stat().st_size)}); self-verify ok ({report['members']} members).")
    print("    restore elsewhere with: sog-setup unpack " + out_path.name)
    return 1 if env_failures else 0


# ------------------------------------------------------------------------------ unpack


def _classify(tar: tarfile.TarFile, manifest: dict, dest: Path) -> dict:
    """One read-only pass over the archive: what would land where, what already matches
    (sha-equal → skip), and what CONFLICTS with different bytes on disk. Nothing is
    written here — the conflict refusal must come before the first byte moves."""
    by_path = {e["path"]: e for e in manifest.get("files", [])}
    plan = {
        "place": [],
        "skip": [],
        "conflict": [],
        "blocked": [],
        "unsafe": [],
        "provenance": 0,
        "envs": [],
        "user_config": None,
    }
    dest_real = os.path.realpath(dest)
    for info in tar.getmembers():
        arc = info.name.partition("/")[2]
        if not arc or arc == "MANIFEST.json":
            continue
        component = arc.split("/", 1)[0]
        if component in ("code", "state", "logs"):
            plan["provenance"] += 1
            continue
        if component == "envs":
            plan["envs"].append(info.name)
            continue
        rel = _rel_of_arc(arc)
        if rel is None:
            continue
        target = _placed_at(dest, rel)
        # Where the write would REALLY land, through the destination's own symlinks as they are now
        # (hunt 2026-09-30, u36-setup-install-1): verify_bundle reads only the archive, so a link already
        # on this disk -- one an earlier bundle planted, or a case-insensitive alias of one -- is invisible
        # to it. Refused here, before the first byte moves; _place re-checks at each write.
        escape = _escape_reason(target, dest_real, info.linkname if info.issym() else None)
        if escape:
            plan["unsafe"].append((rel, escape))
            continue
        obstruction = _obstruction(target, dest_real)
        if obstruction:
            plan["blocked"].append((rel, obstruction))
            continue
        if rel == constants.USER_MCP_CONFIG_REL:
            plan["user_config"] = info.name  # merged, never blind-placed
            continue
        entry = by_path.get(arc, {})
        if info.issym():
            if target.is_symlink() and os.readlink(target) == info.linkname:
                plan["skip"].append(rel)
            elif target.is_dir() and not target.is_symlink():
                # A link where the destination has a real directory: --force's unlink() cannot remove a
                # directory, so it crashed mid-unpack with earlier members already placed. Refused up front
                # like a file member's (hunt 2026-09-30, u36-setup-install-1 follow-on, symlink half).
                plan["blocked"].append((rel, "the destination has a DIRECTORY here"))
            elif target.exists() or target.is_symlink():
                plan["conflict"].append(rel)
            else:
                plan["place"].append((info.name, rel))
        elif target.is_file():
            if entry.get("sha256") and _hash_file(target) == entry["sha256"]:
                plan["skip"].append(rel)
            else:
                plan["conflict"].append(rel)
        elif target.is_dir() and not target.is_symlink():
            # A directory where the bundle has a file: os.replace cannot put a file over it, so placing
            # it crashed mid-unpack with earlier members already written. Refused up front, with or
            # without --force (hunt 2026-09-30, u36-setup-install-1 follow-on).
            plan["blocked"].append((rel, "the destination has a DIRECTORY here"))
        else:
            plan["place"].append((info.name, rel))
    return plan


def _obstruction(target: Path, root_real: str) -> str | None:
    """A non-directory on the way to ``target``'s directory -- or None.

    ``mkdir(parents=True)`` raised NotADirectoryError / FileExistsError / ELOOP partway through unpack,
    with earlier members already placed, when a FILE (or a looping link) sat where a member needs a
    directory (hunt 2026-09-30, u36-setup-install-1 follow-on, found by fuzzing crafted bundles).
    Walked along the RESOLVED path, so a link to a directory is a directory."""
    parent_real = os.path.realpath(target.parent)
    probe = root_real
    for part in os.path.relpath(parent_real, root_real).split(os.sep):
        if part in ("", "."):
            break
        probe = os.path.join(probe, part)
        if not os.path.lexists(probe):
            return None  # from here down, mkdir creates it
        if not os.path.isdir(probe):
            return f"{probe} is a file (or a symlink loop) where a directory is needed"
    return None


class UnsafePlacementError(RuntimeError):
    """A bundle member whose write would land outside the directory unpack is filling."""


def _within(path_real: str, root_real: str) -> bool:
    return os.path.commonpath([path_real, root_real]) == root_real


def _escape_reason(target: Path, root_real: str, link: str | None) -> str | None:
    """Why writing ``target`` (and, for a symlink member, the link it would hold) leaves ``root_real`` -- or
    None. Resolved through the symlinks on disk NOW, not read lexically (hunt 2026-09-30, u36-setup-install-1):
    the directory the member goes into, and where the link itself would point from there."""
    parent_real = os.path.realpath(target.parent)
    if not _within(parent_real, root_real):
        return f"its directory resolves to {parent_real}, outside {root_real}"
    if link is not None:
        link_real = os.path.realpath(os.path.join(parent_real, link))
        if not _within(link_real, root_real):
            return f"the link would point at {link_real}, outside {root_real}"
    return None


def _place(tar: tarfile.TarFile, member_name: str, target: Path, *, root: Path) -> None:
    """Write one member at ``target``, which must resolve inside ``root`` AT THIS MOMENT.

    The check runs at every write, after the members placed before it, and the write goes to the path it
    checked (hunt 2026-09-30, u36-setup-install-1): the archive-only rules in :func:`verify_bundle` cannot
    see the destination's own symlinks, and a lexical check passed a chained link whose next member then
    landed in ``<dest>/../outside``. The ``.partial`` sibling is a fresh ``mkstemp`` name, so a symlink
    already sitting at ``<name>.partial`` is never followed."""
    info = tar.getmember(member_name)
    parent = Path(os.path.realpath(target.parent))
    real_target = parent / target.name
    escape = _escape_reason(real_target, os.path.realpath(root), info.linkname if info.issym() else None)
    if escape:
        raise UnsafePlacementError(f"{target}: {escape}")
    parent.mkdir(parents=True, exist_ok=True)
    if info.issym():
        if real_target.exists() or real_target.is_symlink():
            real_target.unlink()
        real_target.symlink_to(info.linkname)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target.name}.", suffix=".partial", dir=parent)
    try:
        with os.fdopen(fd, "wb") as dst, tar.extractfile(info) as src:
            shutil.copyfileobj(src, dst)
        os.chmod(tmp_name, 0o755 if info.mode & 0o100 else 0o644)
        os.replace(tmp_name, real_target)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _normalize_user_doc(doc: dict) -> tuple[dict, dict]:
    """(servers, top-level extras) with the runtime merger's stray-block recovery applied
    — the tool-creation agent has written server blocks BESIDE ``mcp_servers``."""
    from . import wiring

    merger = wiring.merger_module()
    servers = dict(doc.get("mcp_servers") or {}) if isinstance(doc.get("mcp_servers"), dict) else {}
    try:
        recovered = merger._recover_top_level_servers(doc, servers)
    except Exception:
        recovered = {}
    servers.update(recovered)
    extras = {k: v for k, v in doc.items() if k != "mcp_servers" and k not in recovered}
    return servers, extras


def _merge_user_configs(existing: dict, incoming: dict) -> tuple[dict, list[str]]:
    """Union the two overlays, machine B's entries always surviving. A same-name entry
    with different content is a CONFLICT for the caller to refuse or force."""
    b_servers, b_extras = _normalize_user_doc(existing)
    a_servers, a_extras = _normalize_user_doc(incoming)
    conflicts = sorted(name for name in a_servers if name in b_servers and a_servers[name] != b_servers[name])
    merged_servers = dict(b_servers)
    for name, block in a_servers.items():
        if name not in merged_servers:
            merged_servers[name] = block
    merged = {"mcp_servers": merged_servers, **b_extras}
    for key, value in a_extras.items():
        merged.setdefault(key, value)
    return merged, conflicts


#: How an ``unmergeable`` outcome names the side that cannot be read (see :func:`_apply_user_config`).
_CONFIG_HERE = "this machine's copy"
_CONFIG_BUNDLE = "the bundle's copy"


def _load_user_config(text: str) -> tuple[dict | None, str]:
    """``(mapping, "")``, or ``(None, why it is unusable)``."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError:
        return None, "does not parse as YAML"
    if not isinstance(doc, dict):
        return None, "is not a YAML mapping"
    return doc, ""


def _apply_user_config(
    tar: tarfile.TarFile, member_name: str, dest: Path, *, force: bool, backups_dir: Path
) -> tuple[str, list[str]]:
    """Merge the bundle's ``mcp_config_user.yaml`` into machine B's — never blind-place.

    Returns ``(outcome, kept_existing)``: outcome is one of placed / merged / skipped /
    unmergeable. A same-name server with different content is NOT fatal: machine B's
    entry wins and the name is reported in ``kept_existing`` — the post-place rebase
    rewrites this very file for machine B, so a strict conflict refusal would turn every
    legitimate re-run (idempotence check, env-build retry) into an error. ``--force``
    flips the rule: the bundle's block wins, with machine B's file backed up first.
    Unparseable yaml on either side leaves B's file untouched (outcome unmergeable)."""
    from . import llm_setup, mcp_resolver

    target = _placed_at(dest, constants.USER_MCP_CONFIG_REL)
    incoming_text = tar.extractfile(tar.getmember(member_name)).read().decode("utf-8")
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        llm_setup._atomic_write_text(target, incoming_text)
        return "placed", []
    existing_text = target.read_text(encoding="utf-8")
    if existing_text == incoming_text:
        return "skipped", []
    # Which side is unusable decides the remedy, so each is named (hunt 2026-09-30, reviewer minor on
    # u36-setup-install-17): the message used to tell the user to fix THIS machine's file even when it was
    # the bundle's copy that would not parse, and editing the destination cannot repair that.
    existing, existing_bad = _load_user_config(existing_text)
    incoming, incoming_bad = _load_user_config(incoming_text)
    unusable = [f"{_CONFIG_HERE} {existing_bad}"] if existing_bad else []
    unusable += [f"{_CONFIG_BUNDLE} {incoming_bad}"] if incoming_bad else []
    if unusable:
        return "unmergeable", unusable
    merged, conflicts = _merge_user_configs(existing, incoming)
    kept_existing = conflicts
    if conflicts and force:  # the bundle's block wins; machine B's original is backed up
        incoming_servers, _ = _normalize_user_doc(incoming)
        for name in conflicts:
            merged["mcp_servers"][name] = incoming_servers[name]
        kept_existing = []
    if merged == existing:  # nothing new from the bundle (the idempotent re-run shape)
        return "skipped", kept_existing
    backups_dir.mkdir(parents=True, exist_ok=True)
    mcp_resolver._atomic_backup(target, backups_dir, "mcp_config_user.yaml")
    llm_setup._atomic_write_text(target, yaml.safe_dump(merged, sort_keys=False))
    return "merged", kept_existing


def _rebuild_user_envs(dest: Path, *, only_yamls: list[Path] | None = None) -> dict:
    """Build every ``user_<id>`` env whose recipe landed under ``agent/tools_user/`` and does
    not already exist (existing envs are REUSED, never touched — unpack deletes no env it
    did not create in this same call). One strip-builds retry on a solver failure; failures
    are collected, not fatal — the report and exit code carry them.

    A build that fails leaves no env behind (hunt 2026-09-30, u36-setup-install-10). conda does
    not roll back an ``env create`` whose pip layer died, so the half-built prefix stayed, and the
    re-run the user is told to do after fixing the network counted it as "reused" and exited 0
    with the tool's pip layer still missing. The env did not exist when this call began, so the
    prefix being removed is the one this call just made."""
    from . import provision
    from .envtools import Conda, CondaError

    result = {"reused": [], "built": [], "failed": {}}
    yamls = (
        only_yamls if only_yamls is not None else sorted(constants.tools_user_dir(dest).glob("*" + _ENV_YAML_SUFFIX))
    )
    if not yamls:
        return result
    try:
        conda = Conda()
        conda.env_exists("base")  # the cheapest "does a conda frontend answer at all" probe
    except Exception as exc:
        result["failed"]["<conda>"] = f"no working conda frontend here ({exc}); re-run unpack later or use --no-envs"
        return result
    for recipe in yamls:
        env = "user_" + recipe.name[: -len(_ENV_YAML_SUFFIX)]
        if conda.env_exists(env):
            result["reused"].append(env)
            continue
        try:
            staged = provision.materialize_recipe(str(recipe), env)
            run = conda.create_from_yaml(str(staged), name=env, check=False)
            if not run.ok and provision._SOLVE_FAILURE_RE.search(run.stderr or ""):
                staged = provision.materialize_recipe(str(recipe), env, strip_builds=True)
                run = conda.create_from_yaml(str(staged), name=env, check=False)
            if run.ok:
                result["built"].append(env)
            else:
                tail = "\n".join((run.stderr or "").strip().splitlines()[-3:])
                result["failed"][env] = tail or "conda create failed"
        except (CondaError, OSError) as exc:
            result["failed"][env] = str(exc)
        if env in result["failed"]:
            leftover = _remove_failed_user_env(conda, env)
            if leftover:
                result["failed"][env] += f"\n{leftover}"
    return result


def _remove_failed_user_env(conda, env: str) -> str:
    """Remove the half-built ``env`` a failed create left (see :func:`_rebuild_user_envs`). Returns a
    line for the report when a prefix is still there afterwards, else ``""``. Never raises."""
    try:
        if not conda.env_exists(env, refresh=True):
            return ""
        conda.remove_env(env)
        if not conda.env_exists(env, refresh=True):
            return ""
    except Exception as exc:  # a slow/locked env: say so, never crash the unpack report
        return f"the half-built env {env} could not be removed ({type(exc).__name__}); remove it before re-running"
    return f"the half-built env {env} could not be removed; remove it (conda env remove -n {env}) before re-running"


def _restore_packed_envs(tar: tarfile.TarFile, env_member_names: list[str], manifest: dict) -> dict:
    """The conda-pack lane: extract each ``envs/<env>.tar.gz`` into the local envs root
    (only when that env does not exist), run its ``bin/conda-unpack``, and verify the
    interpreter answers. Any failure falls back to the recipe lane."""
    result = {"restored": [], "skipped": [], "failed": {}}
    if not env_member_names:
        return result
    envs_root = Path(constants.conda_envs_root())
    here = platform.platform()
    there = (manifest.get("host") or {}).get("platform", "")
    if there and there != here:
        print(f" -- note: packed envs were built on '{there}', this box is '{here}' -- they may not run.")
    for name in env_member_names:
        env = name.rsplit("/", 1)[-1][: -len(".tar.gz")]
        prefix = envs_root / env
        if prefix.exists():
            result["skipped"].append(env)
            continue
        try:
            prefix.mkdir(parents=True)
            # r|gz streams straight out of the outer archive -- a packed env can be GBs.
            with tar.extractfile(tar.getmember(name)) as raw:
                with tarfile.open(fileobj=raw, mode="r|gz") as env_tar:
                    env_tar.extractall(prefix, filter="data")
            unpack_tool = prefix / "bin" / "conda-unpack"
            if unpack_tool.is_file():
                # Its exit status decides (hunt 2026-09-30, u36-setup-install-11 / uL4-honesty-13): the
                # `python --version` probe below passes whether or not the machine-A prefixes were
                # rewritten, so a failed relocation was reported "restored" with every script still
                # pointing at machine A. A failure raises into the except below, which removes the new
                # prefix and records the env under failed.
                fixed = subprocess.run([str(unpack_tool)], capture_output=True, text=True, check=False, timeout=600)
                if fixed.returncode != 0:
                    tail = (fixed.stderr or fixed.stdout or "").strip().splitlines()[-1:] or ["no output"]
                    raise RuntimeError(f"conda-unpack exited {fixed.returncode}: {redact(tail[0])[:200]}")
            probe = subprocess.run(
                [constants.interp_path(str(prefix), "python"), "--version"],
                capture_output=True,
                check=False,
                timeout=60,
            )
            if probe.returncode != 0:
                raise RuntimeError("restored env's python does not run")
            result["restored"].append(env)
        except Exception as exc:
            result["failed"][env] = str(exc)
            shutil.rmtree(prefix, ignore_errors=True)  # a half-restored NEW dir only; never a pre-existing env
    return result


def _dest_base_python(dest: Path) -> str:
    """Machine B's base interpreter for user-config commands: its own setup state if it
    has one, else the interpreter running unpack (base_mcp's launch-time chain still
    self-heals either way)."""
    try:
        state = json.loads((dest / ".sog_setup" / "setup_state.json").read_text(encoding="utf-8"))
        basic = state.get("basic_env_name") or ""
        if basic:
            from . import wiring
            from .envtools import Conda

            return wiring.base_python_path(basic, conda=Conda())
    except Exception:
        pass
    return sys.executable


def unpack(
    bundle: str,
    *,
    dest: str | None = None,
    force: bool = False,
    no_envs: bool = False,
    envs_only: bool = False,
    assume_yes: bool = False,
) -> int:
    bundle_path = Path(bundle).expanduser()
    if not bundle_path.is_file():
        print(f" -- no such bundle: {bundle_path}", file=sys.stderr)
        return 1
    dest_path = Path(dest).expanduser().resolve() if dest else Path(constants.repo_root())
    if not dest_path.is_dir():
        print(f" -- destination is not a directory: {dest_path}", file=sys.stderr)
        return 1

    report = verify_bundle(bundle_path)
    if not report["ok"]:
        print(" -- REFUSED: the bundle fails verification:", file=sys.stderr)
        for err in report["errors"][:10]:
            print(f"      {err}", file=sys.stderr)
        return 1

    with tarfile.open(bundle_path, mode="r:gz") as tar:
        manifest = _read_manifest(tar)
        git_info = manifest.get("git") or {}
        print(
            f" -- bundle {manifest.get('bundle', bundle_path.name)}: "
            f"{git_info.get('branch', '?')}@{git_info.get('short', '?')}"
            f"{' (packed from a DIRTY tree)' if git_info.get('dirty') else ''}, "
            f"created {manifest.get('created_at', '?')}"
        )
        try:
            here = _git(dest_path, "rev-parse", "HEAD").strip()
            if git_info.get("commit") and here != git_info["commit"]:
                print(
                    f"    note: destination checkout is at {here[:8]}, bundle came from "
                    f"{git_info.get('short', '?')} -- the user layer usually transplants fine, "
                    "but keep the difference in mind."
                )
        except Exception:
            print("    note: destination is not a git checkout; placing the user layer onto it as-is.")

        plan = _classify(tar, manifest, dest_path)
        if envs_only:  # no file placement at all -- placement conflicts are moot too
            plan["place"], plan["skip"], plan["conflict"], plan["user_config"] = [], [], [], None
            plan["blocked"], plan["unsafe"] = [], []

        if plan["unsafe"]:
            print(
                f" -- REFUSED: these bundle members would be written OUTSIDE {dest_path}, through symlinks already "
                "on this disk:",
                file=sys.stderr,
            )
            for rel, why in plan["unsafe"][:20]:
                print(f"      {rel}: {why}", file=sys.stderr)
            if len(plan["unsafe"]) > 20:
                print(f"      ... and {len(plan['unsafe']) - 20} more", file=sys.stderr)
            print(
                "    Nothing was written. Remove or repoint those links (or unpack into a fresh --dest) and re-run.",
                file=sys.stderr,
            )
            return 1

        if plan["blocked"]:
            print(" -- REFUSED: these bundle members cannot be placed over what the destination has:", file=sys.stderr)
            for rel, why in plan["blocked"][:20]:
                print(f"      {rel}: {why}", file=sys.stderr)
            if len(plan["blocked"]) > 20:
                print(f"      ... and {len(plan['blocked']) - 20} more", file=sys.stderr)
            print(
                "    Nothing was written. Move those paths aside and re-run (--force replaces files, never a "
                "directory).",
                file=sys.stderr,
            )
            return 1

        if plan["conflict"] and not force:
            print(" -- REFUSED: these files exist at the destination with DIFFERENT content:", file=sys.stderr)
            for rel in plan["conflict"][:20]:
                print(f"      {rel}", file=sys.stderr)
            if len(plan["conflict"]) > 20:
                print(f"      ... and {len(plan['conflict']) - 20} more", file=sys.stderr)
            print("    Re-run with --force to replace them (originals are backed up first).", file=sys.stderr)
            return 1

        if not assume_yes and sys.stdin.isatty() and (plan["place"] or plan["conflict"] or plan["user_config"]):
            answer = input(
                f" -- place {len(plan['place'])} file(s)"
                + (f", replace {len(plan['conflict'])}" if plan["conflict"] else "")
                + f" into {dest_path}? [y/N] "
            )
            if answer.strip().lower() not in ("y", "yes"):
                print("    stopped -- nothing was written.")
                return 1

        backups_dir = dest_path / ".sog_setup" / "backups"
        backups: list[str] = []
        placed: list[str] = []
        from . import mcp_resolver

        try:
            # ``placed`` records where each member landed, relative to the destination root
            # (``agent/tools_user/x`` on a re-laid-out checkout), not its archive coordinate.
            for member_name, rel in plan["place"]:
                target = _placed_at(dest_path, rel)
                _place(tar, member_name, target, root=dest_path)
                placed.append(target.relative_to(dest_path).as_posix())
            for rel in plan["conflict"]:  # only reachable under --force
                target = _placed_at(dest_path, rel)
                backups_dir.mkdir(parents=True, exist_ok=True)
                backups.append(str(mcp_resolver._atomic_backup(target, backups_dir, target.name)))
                arc = next(n for n, r in _iter_arcs(tar) if r == rel)
                _place(tar, arc, target, root=dest_path)
                placed.append(target.relative_to(dest_path).as_posix())
        except (UnsafePlacementError, OSError) as exc:
            # The write-time half of u36-setup-install-1: _classify saw the disk before any member was placed;
            # this is a write that only became unsafe (or impossible) after earlier members landed. It is never
            # made, and the user is told what WAS placed rather than handed a traceback.
            if isinstance(exc, UnsafePlacementError):
                print(f" -- STOPPED: refused to write outside the destination -- {exc}", file=sys.stderr)
            else:
                print(f" -- STOPPED: a member could not be placed -- {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"    {len(placed)} file(s) were placed before it and are left as they are:", file=sys.stderr)
            for rel in placed[:20]:
                print(f"      {rel}", file=sys.stderr)
            return 1

        config_outcome, config_kept = "absent", []
        if plan["user_config"]:
            config_outcome, config_kept = _apply_user_config(
                tar, plan["user_config"], dest_path, force=force, backups_dir=backups_dir
            )
            if config_outcome == "unmergeable":
                # Its own message, and a non-zero exit (hunt 2026-09-30, u36-setup-install-17): the user
                # tools' workers were placed but none of their server entries were, so they are
                # unreachable — and --force cannot help, it is decided before the force rule runs.
                remedies = []
                if any(side.startswith(_CONFIG_HERE) for side in config_kept):
                    remedies.append("    Fix this machine's file by hand (it must parse as a YAML mapping).")
                if any(side.startswith(_CONFIG_BUNDLE) for side in config_kept):
                    remedies.append(
                        f"    The bundle's copy is the broken one: fix {constants.USER_MCP_CONFIG_REL} on the machine "
                        "the bundle came from and re-pack, or add its user servers to this machine's file by hand."
                    )
                print(
                    f" -- {_placed_at(dest_path, constants.USER_MCP_CONFIG_REL)} could not be merged: "
                    f"{', '.join(config_kept)}.\n"
                    "    None of the bundle's user servers were added, so its user tools are not wired.\n"
                    + "\n".join(remedies)
                    + "\n    Then re-run unpack.",
                    file=sys.stderr,
                )
            elif config_kept:
                print(
                    " -- mcp_config_user.yaml: kept this machine's version of "
                    f"{len(config_kept)} same-name server(s): {', '.join(config_kept)}\n"
                    "    (nothing was overwritten; pass --force to take the bundle's versions instead)"
                )

        env_summary = {"reused": [], "built": [], "failed": {}}
        packed_summary = {"restored": [], "skipped": [], "failed": {}}
        if not no_envs:
            packed_summary = _restore_packed_envs(tar, plan["envs"], manifest)
            if envs_only:
                staged_dir = Path(constants.state_dir()) / "tmp" / "bundle_recipes"
                staged_dir.mkdir(parents=True, exist_ok=True)
                staged: list[Path] = []
                try:
                    for name, rel in _iter_arcs(tar):
                        if rel.startswith("tools_user/") and rel.endswith(_ENV_YAML_SUFFIX) and "/" not in rel[11:]:
                            target = staged_dir / rel.rsplit("/", 1)[-1]
                            _place(tar, name, target, root=staged_dir)
                            staged.append(target)
                except UnsafePlacementError as exc:
                    print(f" -- REFUSED: a recipe would be staged outside {staged_dir} -- {exc}", file=sys.stderr)
                    return 1
                env_summary = _rebuild_user_envs(dest_path, only_yamls=staged)
            else:
                env_summary = _rebuild_user_envs(dest_path)

    rebase_summary = None
    if not envs_only:
        try:
            rebase_summary = mcp_resolver.rebase_user_config(
                base_python=_dest_base_python(dest_path), repo_root=dest_path
            )
        except Exception as exc:
            print(f" -- couldn't rebase the user-tool config ({type(exc).__name__}: {exc})")

    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    journal_path = dest_path / ".sog_setup" / f"bundle_restore_{stamp}.json"
    journal = {
        "bundle": str(bundle_path),
        "manifest_git": git_info,
        "dest": str(dest_path),
        "placed": placed,
        "skipped_identical": plan["skip"],
        "forced_backups": backups,
        "user_config": {"outcome": config_outcome, "kept_existing": config_kept},
        "envs": env_summary,
        "packed_envs": packed_summary,
        "rebase": {k: str(v) if isinstance(v, Path) else v for k, v in (rebase_summary or {}).items()}
        if isinstance(rebase_summary, dict)
        else None,
        "finished_at": stamp,
    }
    try:
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = journal_path.with_name(journal_path.name + ".partial")
        tmp.write_text(json.dumps(journal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, journal_path)
    except OSError:
        journal_path = None

    print(f" -- placed {len(placed)} file(s), skipped {len(plan['skip'])} identical; user config: {config_outcome}.")
    if env_summary["reused"] or env_summary["built"] or env_summary["failed"]:
        print(
            f"    user envs: {len(env_summary['reused'])} reused, {len(env_summary['built'])} built, "
            f"{len(env_summary['failed'])} failed."
        )
        for env, why in env_summary["failed"].items():
            print(f"      FAILED {env}: {why.splitlines()[-1] if why else 'unknown'}")
    if packed_summary["restored"] or packed_summary["failed"]:
        print(
            f"    packed envs: {len(packed_summary['restored'])} restored, "
            f"{len(packed_summary['skipped'])} already present, {len(packed_summary['failed'])} failed."
        )
    disabled = list((rebase_summary or {}).get("disabled") or []) if isinstance(rebase_summary, dict) else []
    if disabled:
        print(f"    user server(s) disabled (script missing here): {', '.join(disabled)}")
    if journal_path:
        print(f"    journal: {journal_path}")
    key_names = manifest.get("configured_key_names") or []
    if key_names:
        print(f"    machine A had these key NAMES configured (re-enter their values here): {', '.join(key_names)}")
    print(
        " -- next: run `sog-setup` to configure LLM keys and provision built-in tool envs, then\n"
        "    `sog-setup conncheck` to verify the wiring end to end."
    )
    unmergeable = config_outcome == "unmergeable"
    return 1 if (env_summary["failed"] or packed_summary["failed"] or disabled or unmergeable) else 0


def _iter_arcs(tar: tarfile.TarFile):
    for info in tar.getmembers():
        arc = info.name.partition("/")[2]
        rel = _rel_of_arc(arc) if arc else None
        if rel:
            yield info.name, rel


# --------------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """Entry point for both verbs: ``main(["pack", ...])`` / ``main(["unpack", ...])``.
    ``cli.py`` hands the verb through unchanged so each keeps its flags here."""
    import argparse

    argv = list(sys.argv[1:] if argv is None else argv)
    verb = argv[0] if argv else ""
    if verb == "pack":
        ap = argparse.ArgumentParser(
            prog="sog-setup pack",
            description="Pack this deployment (code + user tools + state) into one portable bundle.",
        )
        ap.add_argument("out", nargs="?", default=None, help="output path (default: .sog_setup/bundles/...)")
        ap.add_argument("--with-knowledge", action="store_true", help="include the agent/tools_user/.knowledge corpus")
        ap.add_argument("--with-repos", action="store_true", help="include cloned source repos (large; .git pruned)")
        ap.add_argument(
            "--with-conda-pack",
            nargs="+",
            metavar="ENV",
            default=[],
            help="also archive these conda envs as binaries (needs the conda-pack tool)",
        )
        ap.add_argument("--allow-dirty", action="store_true", help="pack an uncommitted working tree anyway")
        ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
        ap.add_argument("--dry-run", action="store_true", help="select + gate + report, write nothing")
        args = ap.parse_args(argv[1:])
        return pack(
            args.out,
            with_knowledge=args.with_knowledge,
            with_repos=args.with_repos,
            conda_pack_envs=args.with_conda_pack,
            allow_dirty=args.allow_dirty,
            force=args.force,
            dry_run=args.dry_run,
        )
    if verb == "unpack":
        ap = argparse.ArgumentParser(
            prog="sog-setup unpack",
            description="Restore a sog-setup bundle onto this checkout: place the user layer, rebuild user envs.",
        )
        ap.add_argument("bundle", help="path to a sog-bundle-*.tar.gz")
        ap.add_argument("--dest", default=None, help="destination checkout (default: this repo root)")
        ap.add_argument("--force", action="store_true", help="replace differing files (originals backed up)")
        ap.add_argument("--no-envs", action="store_true", help="place files only; build no conda envs")
        ap.add_argument("--envs-only", action="store_true", help="build/restore envs only; place no files")
        ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
        args = ap.parse_args(argv[1:])
        if args.no_envs and args.envs_only:
            print(" -- --no-envs and --envs-only exclude each other.", file=sys.stderr)
            return 2
        return unpack(
            args.bundle,
            dest=args.dest,
            force=args.force,
            no_envs=args.no_envs,
            envs_only=args.envs_only,
            assume_yes=args.yes,
        )
    print(" -- bundle: expected `pack` or `unpack`.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
