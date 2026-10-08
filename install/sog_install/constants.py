"""
Static constants and path helpers for the setup wizard.

Pure data + trivial path math only — no third-party imports, and no side effects
at import time. The ``ensure_*`` helpers are the only side-effecting calls
(``mkdir`` for the dir helpers; a ``sys.path`` append in
:func:`ensure_repo_importable`). Safe to import before any conda env exists.

Directory contract (all resolved relative to the repo root unless an env
override is given):

  * ``.sog_setup/``            durable state, logs, backups, archive — GIT-IGNORED,
                               lives OUTSIDE ``test/installation/`` so end-of-run
                               "delete results" can never destroy resume state.
  * ``test/installation/``     per-run test artifacts (the only thing cleanup deletes).
  * ``install/recipes/tool_specs/``  committed per-tool env recipes (produced by ``capture``).
  * ``install/recipes/mcp_config.setup.yaml``  generated wiring — the wizard's OWN MCP config.
                               The agent's ``agent/MCP_server/mcp_config.yaml`` is regenerated
                               for this box by ``finalize`` (timestamped backup first)
                               unless ``--keep-agent-config`` is passed.

Two roots: the **repository root** (:func:`repo_root`) holds the state, ``.env``, ``install/recipes``,
``test/`` and the run trees; the **agent part** (:func:`agent_root`, ``<root>/agent``) holds ``tools/``,
``tools_user/``, ``MCP_server/`` and ``skills/``. A seeded SOG_HOME (pip-only install) is laid out like
the wheel's packaged copy instead -- the agent trees at its top and the recipes under ``setup/`` --
and :func:`agent_root` / :func:`recipes_root` answer for whichever layout a root has
(``platform_root.platform_dir`` / ``platform_root.recipes_dir``, the one statement of that rule).
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from spatialomicsgym import layout as _layout
from spatialomicsgym import platform_root as _platform_root

# --------------------------------------------------------------------------- #
# Repository / package roots
# --------------------------------------------------------------------------- #
# This file is install/sog_install/constants.py. PACKAGE_ROOT is the ``spatialomicsgym`` package directory
# (agent/spatialomicsgym in a checkout; site-packages/spatialomicsgym in a wheel) -- where the packaged
# ``spatialomicsgym_env`` tree lives. REPO_ROOT is the checkout root (``layout.repo_root()``), or the
# package's parent in a non-editable install, where there is no checkout.
PACKAGE_ROOT: Path = Path(_platform_root.__file__).resolve().parent  # .../spatialomicsgym
REPO_ROOT: Path = _layout.repo_root() or PACKAGE_ROOT.parent  # repo checkout root

#: The checkout part that holds ``tools/``, ``tools_user/``, ``MCP_server/`` and ``skills/``.
AGENT_DIRNAME = _platform_root.AGENT_PART
#: The development tree's test directory (untracked: the suite, the smoke harness, the mini data).
TEST_DIRNAME = "test"
#: The one sentence every caller gives when a test-tree resource is wanted and ``test/`` is absent.
TEST_TREE_MISSING = "test/ is not present in this checkout — the smoke harness ships only with the development tree"


def repo_root() -> Path:
    """The wizard's working root: ``SOG_SETUP_REPO_ROOT`` first, else the platform instance root.

    Rung 1 is this module's own test/operator seam, unchanged. The rest is delegated to
    ``platform_root.instance_root()`` -- on a checkout it answers the repository root (the
    directory holding ``agent/``, ``install/`` and ``.sog_setup``), so in-repo behavior matches
    ``REPO_ROOT``; on a pip-only install it degrades to a marked cwd and finally to SOG_HOME
    (default ``~/.spatialomicsgym``), the writable instance the wizard seeds -- instead of an
    unwritable ``site-packages`` parent that nothing had ever seeded.
    """
    # ``… or``-style emptiness check (not ``get(key, default)``): a present-but-EMPTY
    # ``SOG_SETUP_REPO_ROOT=`` must fall through, not become ``Path("")`` == cwd (which would
    # relocate the durable state/artifact tree under the caller's cwd and lose resume state).
    # This matches the empty-is-unset idiom the three sibling overrides already use
    # (``state_dir``/``artifact_dir``/``dotenv_path``) and platform_root's own rungs.
    override = os.environ.get("SOG_SETUP_REPO_ROOT")
    if override:
        return Path(override)
    return _platform_root.instance_root()


def repo_path(*parts: str) -> Path:
    """Join ``parts`` under the repo root."""
    return repo_root().joinpath(*parts)


def agent_root(root: Path | str | None = None) -> Path:
    """The agent part of ``root`` (default :func:`repo_root`) -- where ``tools/``, ``tools_user/`` and
    ``MCP_server/`` live.

    ``<root>/agent`` for a root laid out like the repository; the root itself for one laid out like
    the packaged copy (a seeded SOG_HOME, or a scratch root that has neither ``agent/`` nor
    ``install/recipes/``). Delegated to ``platform_root.platform_dir`` so the wizard and the agent
    runtime can never disagree about where a root keeps its trees.
    """
    return _platform_root.platform_dir(Path(root) if root is not None else repo_root())


def recipes_root(root: Path | str | None = None) -> Path:
    """The recipe tree of ``root`` (default :func:`repo_root`): ``tool_specs/`` and the generated wiring.

    ``<root>/install/recipes`` for a root laid out like the repository, ``<root>/setup`` for one laid
    out like the packaged copy (``platform_root.recipes_dir``).
    """
    return _platform_root.recipes_dir(Path(root) if root is not None else repo_root())


def agent_path(*parts: str, root: Path | str | None = None) -> Path:
    """Join ``parts`` under :func:`agent_root` (``tools``, ``tools_user``, ``MCP_server``, ``skills``)."""
    return agent_root(root).joinpath(*parts)


def tools_dir(root: Path | str | None = None) -> Path:
    """:func:`agent_root`'s ``tools/`` (``<root>/agent/tools`` in a checkout) -- the built-in portals and workers."""
    return agent_path("tools", root=root)


def tools_user_dir(root: Path | str | None = None) -> Path:
    """:func:`agent_root`'s ``tools_user/`` -- user-created tools, their recipes and the helper twins."""
    return agent_path("tools_user", root=root)


def dev_test_dir() -> Path | None:
    """The development tree's ``test/`` directory, or ``None`` when this checkout has none.

    ``<repo_root>/test`` first (so the ``SOG_SETUP_REPO_ROOT`` seam carries it), then the checkout's
    own (``layout.test_dir()``). A candidate counts only when it carries the development tree
    (``smoke/`` or ``test_data/``): ``test/installation/`` -- the wizard's own artifact dir -- creates
    a bare ``test/`` on every box, a seeded SOG_HOME included. ``test/`` is untracked, so a fresh
    clone and every wheel install answer ``None``; callers then say :data:`TEST_TREE_MISSING`
    instead of failing on an import.
    """
    candidates = [repo_path(TEST_DIRNAME), _layout.test_dir()]
    for cand in candidates:
        try:
            if cand is not None and ((cand / "smoke").is_dir() or (cand / "test_data").is_dir()):
                return cand
        except OSError:
            continue
    return None


def load_test_module(name: str):
    """Import ``name`` (``smoke.registry``, ``test_core_env`` …) from the ``test/`` tree.

    ``test`` is a stdlib package, so the tree has no ``__init__.py`` and its modules are imported
    top-level; this appends :func:`dev_test_dir` to ``sys.path`` (append, not insert: an installed module
    of the same name keeps winning) and imports ``name``. Raises :class:`ModuleNotFoundError` carrying
    :data:`TEST_TREE_MISSING` when ``test/`` is absent, so the caller's message names the cause.
    """
    import importlib

    tree = dev_test_dir()
    if tree is None:
        raise ModuleNotFoundError(TEST_TREE_MISSING, name=name)
    if str(tree) not in sys.path:
        sys.path.append(str(tree))
    return importlib.import_module(name)


def ensure_repo_importable() -> Path:
    """Make the agent part's top-level packages (``tools_user``, ``tools``, ``skills`` …) importable.

    ``pip install -e .`` registers a setuptools *editable finder* that only exposes the
    distribution's declared packages (``spatialomicsgym``, ``sog_install``,
    ``skills``, ``benchmarks``) — **not** ``tools_user``. So when ``sog-setup`` runs as an installed
    console script from any cwd, the Stage-B engines' lazy ``tools_user.self_review`` import would
    fail. The ``test/`` tree (the Tier-1 smoke harness) is loaded separately, by path, through
    :func:`load_test_module`, which says plainly when ``test/`` is absent.

    Idempotently *appends* the agent root to ``sys.path`` (append, not insert, so a truly installed
    package always wins over the checkout — no shadowing). Returns the repository root.
    """
    root = repo_root()
    agent = str(agent_root(root))
    if agent not in sys.path:
        sys.path.append(agent)
    return Path(root)


# --------------------------------------------------------------------------- #
# Conda envs root (portable — derived from the running interpreter, not hardcoded)
# --------------------------------------------------------------------------- #
def conda_envs_root() -> str:
    """The live conda ``envs/`` dir on THIS machine, derived from the running interpreter's
    prefix — never the hardcoded ``/opt/conda/envs``.

    Correct whether conda lives at ``/opt/conda``, ``~/miniconda3``, a relocated Miniforge, or a
    standalone/symlinked ``micromamba``: this process runs inside a conda env, so a *named* env
    sits at ``<root>/envs/<name>`` (its parent IS ``envs/``) while the *base* env sits at
    ``<root>`` (with ``envs`` at ``<root>/envs``). Honors ``CONDA_PREFIX`` first, then
    ``sys.prefix``. Pure string math — the dir need not exist yet (a fresh box has no per-tool
    envs); take ``os.path.dirname`` of the result for the base *mount* (the disk gate).
    """
    prefix = (os.environ.get("CONDA_PREFIX") or sys.prefix).rstrip("/")
    parent = os.path.dirname(prefix)
    if os.path.basename(parent) == "envs":
        return parent
    return os.path.join(prefix, "envs")


def interp_path(prefix: str, exe: str = "python", *, posix: bool | None = None) -> str:
    """Per-OS path to console executable ``exe`` (``python`` / ``Rscript``) inside conda env dir ``prefix``.

    a1c#2 — the single source of truth for the conda per-env interpreter layout, so every path
    builder (``wiring``, ``specs``, ``mcp_resolver``, ``testing``) agrees across OSes instead of each
    hardcoding POSIX ``/bin/``. POSIX conda puts binaries under ``<env>/bin/``; **Windows** conda has
    no per-env ``bin/`` — ``python.exe`` sits at the env root and other console scripts (Rscript) live
    under ``<env>\\Scripts\\``. A POSIX-hardcoded ``<env>/bin/python`` simply does not resolve there, so
    a Windows deploy box would silently synthesize a dead interpreter path. Forward-slash separators
    are returned on every OS (conda + Python both accept them on Windows), matching ``testing._interp``.

    ``posix`` defaults to the live ``os.name`` (the production source of truth); callers that own their
    own POSIX seam (``testing._interp`` threads ``testing._POSIX``) may pass it explicitly so a test can
    force either layout on either host. On POSIX the result is byte-identical to the old
    ``f"{prefix}/bin/{exe}"``, so nothing changes on the shipped POSIX-first deployment.
    """
    base = str(prefix).rstrip("/\\")
    exe = exe or "python"
    is_posix = (os.name == "posix") if posix is None else posix
    if not is_posix:
        return f"{base}/python.exe" if exe == "python" else f"{base}/Scripts/{exe}.exe"
    return f"{base}/bin/{exe}"


# --------------------------------------------------------------------------- #
# Conda env-name validation (N15 / #100 — canonical home; guide.py + answers.py re-import)
# --------------------------------------------------------------------------- #
# A name that flows into ``conda create -n <name>`` (typed at the "new env" prompt, invented by
# the LLM guide, OR supplied in an ``--answers`` scenario's ``base_env.name``) must be a legal
# conda env name. An invalid one — a space (``my env``), a slash (``sog/2``), a leading dot/dash —
# is otherwise accepted verbatim and only surfaces MUCH later as a generic ``CondaError`` exit-3
# with no hint. Catch it early and re-ask / friendly-fail with a plain reason. Conda env names are
# a single path-safe token: start with a letter/digit, then letters/digits/``.``/``_``/``-``.
# Lives here (a stdlib leaf) so every entry point shares ONE rule without importing ``guide`` (which
# pulls the heavier ``llm_chat``).
_ENV_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ENV_NAME_MAXLEN = 64  # leaves headroom for the ``_<server>`` per-tool suffix; well under path limits


def _validate_env_name(name: str) -> str | None:
    """Return a one-line reason ``name`` is unusable as a conda env name, or ``None`` if it's fine."""
    n = (name or "").strip()
    if not n:
        return "the env name can't be empty"
    if len(n) > _ENV_NAME_MAXLEN:
        return f"the env name is too long (>{_ENV_NAME_MAXLEN} characters)"
    if not _ENV_NAME_RE.match(n):
        return (
            "conda env names must be a single token of letters, digits, '.', '_' or '-' — "
            "no spaces or slashes, and it must start with a letter or digit"
        )
    return None


# --------------------------------------------------------------------------- #
# Directory names (relative) and resolvers (absolute, override-aware)
# --------------------------------------------------------------------------- #
STATE_DIRNAME = ".sog_setup"  # durable, git-ignored
ARTIFACT_DIRNAME = "test/installation"  # per-run test results (cleanup target)
# The repository-layout spellings; resolve through recipes_root()/spec_dir(), which also answer for a
# root laid out like the packaged copy (``setup/``).
RECIPES_DIRNAME = _platform_root.RECIPES_PART  # "install/recipes": tool_specs/ + the generated wiring
SPEC_DIRNAME = f"{RECIPES_DIRNAME}/tool_specs"  # committed recipes
#: Where the recipe tree lived before the re-layout (and still lives in the packaged copy and a seeded
#: home). Spec ``recipe:`` strings recorded by an older run carry this prefix; :func:`resolve_recipe_path`
#: maps it.
LEGACY_RECIPES_PREFIX = _platform_root.PACKAGED_RECIPES_PART + "/"

STATE_FILENAME = "setup_state.json"
KEYLIB_FILENAME = "llm_keys.json"  # saved-key vault (SECRET-bearing → git-ignored + chmod 0600; see key_library)
LOGS_SUBDIR = "logs"
BACKUPS_SUBDIR = "backups"
ARCHIVE_SUBDIR = "archive"
TRANSCRIPTS_SUBDIR = "transcripts"  # durable demo/real-run HTML + text (few, small; not a prune target)

# Generated wiring (git-ignored) — the wizard's own config, never the agent's.
GENERATED_MCP_CONFIG_REL = f"{RECIPES_DIRNAME}/mcp_config.setup.yaml"
GENERATED_ENV_OVERRIDES_REL = f"{RECIPES_DIRNAME}/env_overrides.env"
# The agent's real MCP config, relative to the AGENT part (:func:`agent_root`), not the repository root.
# Finalize regenerates it for this box (``mcp_resolver.apply_to_canonical``, after a timestamped backup)
# unless --keep-agent-config is passed; every other phase only reads it. An earlier note here called it
# read-only, which contradicted that (hunt 2026-09-30, u35-setup-ux-18).
ORIGINAL_MCP_CONFIG_REL = "MCP_server/mcp_config.yaml"
# The machine-local user-tool config the tool-creation playbook writes (git-ignored;
# absent on a fresh clone), relative to the agent part. The agent runtime merges it in at launch via
# ``spatialomicsgym/agent/mcp_config_merger.py`` — original wins name conflicts.
USER_MCP_CONFIG_REL = "MCP_server/mcp_config_user.yaml"

# The one file the wizard shares with the agent (it *is* the agent's config).
DOTENV_REL = ".env"


def state_dir() -> Path:
    """Durable state root (``SOG_SETUP_STATE_DIR`` override → repo/.sog_setup)."""
    override = os.environ.get("SOG_SETUP_STATE_DIR")
    return Path(override) if override else repo_path(STATE_DIRNAME)


def state_file() -> Path:
    return state_dir() / STATE_FILENAME


def key_library_file() -> Path:
    """Path to the saved-key vault (``.sog_setup/llm_keys.json``).

    Lives under the durable, git-ignored state dir and so honors the ``SOG_SETUP_STATE_DIR``
    test seam exactly like :func:`state_file` — a test run's vault lands in tmp, never the real
    one. This is the wizard's only SECRET-bearing state file; :mod:`key_library` hardens it
    (``chmod 0600`` + a belt-and-braces ``.sog_setup/.gitignore``)."""
    return state_dir() / KEYLIB_FILENAME


def logs_dir() -> Path:
    return state_dir() / LOGS_SUBDIR


def transcripts_dir() -> Path:
    """Durable per-run transcript root (``.sog_setup/transcripts``) for the demo/real-run HTML + text.

    Under the same ``SOG_SETUP_STATE_DIR`` test seam as ``state_dir``/``logs_dir`` and, like the
    demo run-data root, never a cleanup target — a deferred "add a key and re-run" still finds it.
    Not added to the log-prune globs (transcripts are few and small)."""
    return state_dir() / TRANSCRIPTS_SUBDIR


def backups_dir() -> Path:
    return state_dir() / BACKUPS_SUBDIR


def archive_dir() -> Path:
    return state_dir() / ARCHIVE_SUBDIR


def artifact_dir() -> Path:
    """Per-run test-artifact root (``SOG_SETUP_ARTIFACT_DIR`` override → repo/test/installation)."""
    override = os.environ.get("SOG_SETUP_ARTIFACT_DIR")
    return Path(override) if override else repo_path(ARTIFACT_DIRNAME)


def resolve_recipe_path(rel: str | os.PathLike[str]) -> Path:
    """The file a spec's ``recipe:`` string names on this box.

    Absolute paths are taken as they are. A repo-relative one is resolved by the tree it names, so
    it lands right on either layout and whichever spelling recorded it: ``install/recipes/...`` and
    the pre-re-layout ``setup/...`` (still in specs and state written by an older run) under
    :func:`recipes_root`; ``agent/...`` and the older bare ``tools_user/...`` under :func:`agent_root`.
    """
    raw = os.fspath(rel)
    if os.path.isabs(raw):
        return Path(raw)
    for prefix in (RECIPES_DIRNAME + "/", LEGACY_RECIPES_PREFIX):
        if raw.startswith(prefix):
            return recipes_root().joinpath(*raw[len(prefix) :].split("/"))
    if raw.startswith(AGENT_DIRNAME + "/"):
        return agent_path(*raw[len(AGENT_DIRNAME) + 1 :].split("/"))
    if raw.startswith("tools_user/"):
        return agent_path(*raw.split("/"))
    return repo_path(raw)


def spec_dir() -> Path:
    return recipes_root() / "tool_specs"


def generated_mcp_config() -> Path:
    return recipes_root() / "mcp_config.setup.yaml"


def generated_env_overrides() -> Path:
    return recipes_root() / "env_overrides.env"


def original_mcp_config() -> Path:
    """The agent's canonical MCP config (``agent/MCP_server/mcp_config.yaml``)."""
    return agent_path(ORIGINAL_MCP_CONFIG_REL)


def user_mcp_config() -> Path:
    """The user-tool config's agent-anchored home; the file may not exist (fresh clone)."""
    return agent_path(USER_MCP_CONFIG_REL)


def stale_mcp_pointer_note(resolved: str | os.PathLike[str] | None) -> str | None:
    """The one line owed to the user when ``SOG_MCP_CONFIG`` is set and cannot be used.

    ``resolved`` is the config the caller wired instead (``None`` if nothing resolved at all).
    Returns ``None`` whenever there is nothing to disclose: the pointer is unset or blank, or it
    names a real file.

    Both readers of the pointer — ``chat_cli._resolve_mcp_config`` and
    ``conncheck._discover_config`` — deliberately *skip* a pointer that no longer resolves and
    fall through to the next candidate, because a session with partly-right tools beats no
    session at all. What they did not do is say so. The last candidate is the canonical repo
    config, in which every tool names an interpreter built on somebody else's box, so falling
    through in silence can deliver exactly the outcome the pointer was recorded to prevent — and
    the ways a pointer dies are all ordinary ones (the repo was copied to a second machine,
    ``setup/`` was moved to ``install/recipes/``, a ``.env`` was carried over from another checkout).

    The sentence lives here, once, rather than in each reader: the two are a documented mirrored
    pair, and a second hand-written copy of a rule in this repo has a long history of drifting
    from the first.
    """
    pointer = (os.environ.get("SOG_MCP_CONFIG") or "").strip()
    if not pointer:
        return None
    expanded = os.path.expanduser(pointer)
    if os.path.isfile(expanded):
        return None
    # "is a directory" and "is not there" have different fixes, so never collapse them into one
    # message. os.path.isfile is also False for "cannot look" (an unreadable parent), which reads
    # as absent — the path is still named, which is what sends the user to the right place.
    why = "is a directory, not a config file" if os.path.isdir(expanded) else "no longer exists"
    if resolved is None:
        return f"SOG_MCP_CONFIG ({expanded}) {why}, and no other analysis-tool config was found either."
    return (
        f"SOG_MCP_CONFIG ({expanded}) {why} - using {resolved} instead; "
        "re-run sog-setup if this box's tools look wrong."
    )


def dotenv_path() -> Path:
    """Path to the shared ``.env`` (``SOG_SETUP_DOTENV`` override → repo/.env).

    Mirrors the ``SOG_SETUP_STATE_DIR`` / ``SOG_SETUP_ARTIFACT_DIR`` seams: the override
    lets the standardized test loop point every ``.env`` read/backup/merge-write at a scratch
    file, so a full-portal run never mutates the developer's real ``.env``. Default unchanged."""
    override = os.environ.get("SOG_SETUP_DOTENV")
    return Path(override) if override else repo_path(DOTENV_REL)


def _prune_logs() -> None:
    """Best-effort size/age pruning of accumulated run logs (C7).

    Deletes the OLDEST ``*.build.log`` / ``run-*.jsonl`` (by mtime) under ``.sog_setup/logs`` once a
    glob exceeds :data:`LOG_KEEP_MAX_FILES` files or :data:`LOG_KEEP_MAX_BYTES` total bytes, so a host
    that runs the wizard many times cannot grow the log dir without bound. **Never raises** — it is
    called from the hot ``ensure_state_dirs`` init path, so a stat/unlink error just leaves that file
    in place, still counted against both caps, and the prune moves on to the next candidate rather
    than stopping. Only ever touches the two run-log globs; the active (newest) log is pruned last, never
    first, so a build streaming into a fresh ``<target>.build.log`` is safe. The env-var caps
    (:data:`LOG_KEEP_MAX_FILES`/``_BYTES``, both clamped) let a CI box keep less or a debugger keep more.
    """
    try:
        logs = logs_dir()
        for pattern in ("*.build.log", "run-*.jsonl"):
            files: list[tuple[Path, float, int]] = []
            for p in logs.glob(pattern):
                try:
                    stt = p.stat()
                except OSError:
                    continue
                files.append((p, stt.st_mtime, stt.st_size))
            files.sort(key=lambda t: t[1])  # oldest first
            total = sum(sz for _, _, sz in files)
            # Neither cap may delete the LAST (newest/active) log — a single build streaming into a
            # >cap `<target>.build.log` would otherwise be unlinked mid-write, contradicting the
            # docstring's "active log pruned last, never first". (S2, and `stuck` below can now walk
            # the count cap down to the last file too, so the guard fronts the whole loop.)
            stuck = 0  # failed to unlink: still on disk, so it frees nothing and still fills a slot
            while len(files) > 1 and (len(files) + stuck > LOG_KEEP_MAX_FILES or total > LOG_KEEP_MAX_BYTES):
                p, _, sz = files.pop(0)
                try:
                    p.unlink()
                except OSError:
                    # A log we cannot remove (root-owned, read-only mount, held open) must not be
                    # counted as freed — doing so satisfied both caps on paper and ended the prune
                    # early, leaving the deletable logs behind it and the dir over budget.
                    stuck += 1
                    continue
                total -= sz
    except Exception:  # pruning is best-effort; never break state-dir init
        pass


def ensure_state_dirs(*, prune: bool = True) -> None:
    """Create the durable-state directory tree (idempotent), then prune overgrown run logs (C7).

    ``prune=False`` creates the dirs but skips the log pruning. ``reset --dry-run`` passes this: a
    dry-run must change NOTHING, but the default prune ``unlink``s the oldest ``run-*.jsonl`` once a
    box has accumulated many, so a preview would silently delete real logs (F2) — the exact opposite
    of the "logs are always preserved" reset guarantee. Every other caller keeps the default so the
    wizard's per-run size-capping is unchanged."""
    for d in (state_dir(), logs_dir(), backups_dir(), archive_dir()):
        d.mkdir(parents=True, exist_ok=True)
    if prune:
        _prune_logs()


def ensure_artifact_dir() -> Path:
    """Create and return the per-run artifact directory (idempotent)."""
    d = artifact_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# Conda-env namespace + isolation guards
# --------------------------------------------------------------------------- #
# The wizard ONLY ever creates/repairs/deletes envs named "<basic>" or
# "<basic>_<server>". Everything else on the machine is at most read-only
# inspected. These helpers are the single choke-point that enforces that.

# One-time migration alias for the SpatialOmicsLab rebrand. A box provisioned
# before the rename carries the heavy shared tool env (svca / R conversion / CNVkit)
# under its legacy name; the resolver adopts it IN PLACE — no rebuild — by probing
# the legacy name as a FINAL fallback after the branded name (see
# ``mcp_resolver.candidate_target_envs``). Fresh deployments build the branded env
# and never touch this map. This is the single deliberate mention of a legacy env
# name in the tree; delete it once no legacy-named envs remain in the fleet.
LEGACY_ENV_ALIASES: dict[str, str] = {"spatialomicsgym_e1": "biomni_e1"}

# Never delete these even if a prefix match somehow occurs (defense-in-depth).
# ``sog_reproduce`` is a real developer env that falls INSIDE the ``sog_*`` namespace,
# so a run with ``basic="sog"`` would otherwise treat it as a deletable tool env — pin
# it here so no provision/repair path can ever remove it. Legacy-aliased env names (a
# pre-rename box's adopted env) are pinned too, so in-place adoption never risks the
# live env.
#
# THE single source of this list: ``tools_user/{trash_manager,knowledge_manager}.py`` import it
# rather than keeping copies (their old local copies drifted — both lacked the legacy-alias
# names this union adds). Extend it HERE only.
#: The general-purpose fallback env the agent may redo a failed tool step in
#: (``spatialomicsgym/tool/general_env.py``). Inside the ``<base>_*`` namespace on purpose, so
#: ``is_managed_env`` tracks it -- and on :data:`PROTECTED_ENVS` on purpose, so ``reset``,
#: ``doctor``, ``remove_env`` and the self-review keep it, exactly as ``sog_reproduce`` is kept
#: under base ``sog``. It is not a tool env: no portal points at it and no recipe under
#: ``install/recipes/tool_specs`` names it, so nothing in the wizard ever repairs it either.
GENERAL_ENV_NAME = "spatialomicsgym_env_general"
#: The env var naming its interpreter explicitly. An explicit path that does not exist is
#: REPORTED by :func:`general_python`, never silently skipped over for a discovered one.
GENERAL_PYTHON_ENV = "SOG_GENERAL_PYTHON"

PROTECTED_ENVS: frozenset[str] = frozenset(
    {"base", "spatialomicsgym_e1", "spatialomicsgym_env", "sog_reproduce", GENERAL_ENV_NAME}
) | frozenset(LEGACY_ENV_ALIASES.values())


def env_aliases() -> dict[str, str]:
    """The branded/legacy conda env names, both directions, as ``base_mcp._env_aliases`` builds them."""
    both: dict[str, str] = {}
    for branded, legacy in LEGACY_ENV_ALIASES.items():
        both[branded] = legacy
        both[legacy] = branded
    return both


def live_conda_root() -> str:
    """The ``envs/`` parent of the interpreter running *this* process.

    Deliberately not ``CONDA_PREFIX`` (which reflects whatever was last *activated* in the shell
    rather than the env this process is in), and deliberately ``sys.prefix`` itself when this
    process is not inside an ``envs/`` tree, because a base conda install still has ``envs/`` under
    it. ``conda_envs_root`` above answers a different question -- where this run should *build* --
    and honours ``CONDA_PREFIX`` for exactly that reason.
    """
    parts = Path(sys.prefix).parts
    return str(Path(*parts[: parts.index("envs")])) if "envs" in parts else sys.prefix


def general_python() -> tuple[str | None, str]:
    """The interpreter of :data:`GENERAL_ENV_NAME`, or ``None``, and how that was decided.

    Resolution, first that EXISTS wins:

    1. ``$SOG_GENERAL_PYTHON`` -- an explicit path. If it names a file that is not there the answer
       is ``(None, why)``: the operator said where it is, and quietly using a different interpreter
       would run their code somewhere they did not choose.
    2. ``interp_path(conda_envs_root()/spatialomicsgym_env_general)`` -- where this run builds.
    3. ``interp_path(live_conda_root()/envs/spatialomicsgym_env_general)`` -- where this process lives.

    Never raises and never creates anything; the second element is a sentence a notice can quote
    verbatim, including the build command when the env simply is not there.
    """
    explicit = (os.environ.get(GENERAL_PYTHON_ENV) or "").strip()
    if explicit:
        if os.path.isfile(explicit):
            return explicit, f"{GENERAL_PYTHON_ENV}={explicit}"
        if os.path.isdir(explicit):
            return None, (
                f"{GENERAL_PYTHON_ENV}={explicit!r} names a directory, not an interpreter; point it at the "
                f"env's python (for example {interp_path(explicit)})"
            )
        return None, f"{GENERAL_PYTHON_ENV}={explicit!r} names a file that does not exist on this machine"
    tried: list[str] = []
    for root in (conda_envs_root(), os.path.join(live_conda_root(), "envs")):
        candidate = interp_path(os.path.join(root, GENERAL_ENV_NAME))
        if os.path.isfile(candidate):
            return candidate, f"found at {candidate}"
        if candidate not in tried:
            tried.append(candidate)
    return None, (
        f"the general env '{GENERAL_ENV_NAME}' is not built on this machine (looked for "
        + ", ".join(tried)
        + "); build it with: conda env create -n "
        + GENERAL_ENV_NAME
        + " -f agent/spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env_general.yml"
    )


def interpreter_on_this_box(interpreter: str) -> str:
    """Adopt this box's copy of a conda interpreter whose configured path is absent.

    A shipped config, spec or candidate list names an interpreter by the path it had on the box that
    produced it. Deployed anywhere else, that literal is wrong in one of two ways the fleet actually
    hits: the env was **renamed** (the SpatialOmicsLab rebrand -- see :data:`LEGACY_ENV_ALIASES`), or
    the whole conda **root** moved off ``/opt/conda``. Either way an interpreter that is present is
    reported missing, and the launcher either dies in ``execvp`` or silently degrades to a bare name
    on ``PATH`` -- a *different* interpreter, which for R means one that need not have the packages.

    Three descending-authority steps, byte-identical to ``tools/base_mcp.py::_resolve_worker_python``
    so both launchers build the same command from the same config: the env **alias** under the pinned
    root; the same env name under the root **this process** runs from; then both at once. The
    interpreter tail (``bin/python`` vs ``bin/Rscript``) is carried through untouched -- it names the
    language, so a python is never substituted for an Rscript.

    Returned unchanged, so the operator is still told what they configured: a falsy or non-string
    value, a relative path (including a bare ``python``/``Rscript``), one that already exists, one
    with no ``envs/`` segment, and one that resolves nowhere.

    ``base_mcp`` is not imported here and never will be: nothing under ``spatialomicsgym/`` imports
    ``tools/``, because a worker must run in a per-tool env that has none of this package installed.
    The two copies are pinned against each other by
    ``test_cross_device_stability.py::TestTheTunerLaunchesTheInterpreterThisBoxHas``.
    """
    if not interpreter or not isinstance(interpreter, str):
        return interpreter
    path = Path(interpreter)
    if not path.is_absolute() or path.exists():
        return interpreter
    parts = path.parts
    if "envs" not in parts:
        return interpreter
    i = parts.index("envs")
    if i + 1 >= len(parts):
        return interpreter
    name = parts[i + 1]
    alias = env_aliases().get(name)
    tail = parts[i + 2 :]
    pinned_root = Path(*parts[:i])
    live_root = Path(live_conda_root())
    for root, env_name in ((pinned_root, alias), (live_root, name), (live_root, alias)):
        if not env_name:
            continue
        candidate = str(root.joinpath("envs", env_name, *tail))
        if candidate != interpreter and Path(candidate).exists():
            return candidate
    return interpreter


def tool_env_name(basic_env: str, server_key: str) -> str:
    """Canonical per-tool env name: ``<basic>_<server>``."""
    return f"{basic_env}_{server_key}"


def is_managed_env(basic_env: str, env_name: str) -> bool:
    """True iff ``env_name`` is owned by this run (the base env or a ``<basic>_*`` tool env)."""
    if not basic_env or not env_name:
        return False
    return env_name == basic_env or env_name.startswith(f"{basic_env}_")


def assert_deletable_env(basic_env: str, env_name: str) -> None:
    """Raise unless ``env_name`` may be destroyed by this run.

    Guards every rmtree / RECREATE / orphan-cleanup: rejects anything outside
    the ``<basic>_*`` namespace and anything on the protected list. The base env
    itself is never *deleted* by the guard (only tool envs are), so reusing an
    existing base env can never trigger its removal.
    """
    if env_name in PROTECTED_ENVS:
        raise PermissionError(f"refusing to delete protected env {env_name!r}")
    if not is_managed_env(basic_env, env_name):
        raise PermissionError(f"refusing to delete env {env_name!r}: not in the '{basic_env}_*' namespace")
    if env_name == basic_env:
        raise PermissionError(f"refusing to delete the base env {env_name!r}")


def assert_deletable_base_env(basic_env: str, env_name: str) -> None:
    """Raise unless ``env_name`` is exactly this run's base env and safe to remove.

    Base removal is the ONLY path allowed to delete ``<basic>`` itself. It is never
    automatic — it is reachable solely through ``sog-setup reset --with-base`` behind a
    separate confirmation — so it has its own guard instead of the general
    :func:`assert_deletable_env` (which deliberately refuses the base env). Still refuses
    anything on the protected list (so the live ``spatialomicsgym_env`` can never go).
    """
    if env_name in PROTECTED_ENVS:
        raise PermissionError(f"refusing to delete protected env {env_name!r}")
    if not basic_env or env_name != basic_env:
        raise PermissionError(f"refusing to delete {env_name!r}: not this run's base env {basic_env!r}")


def base_env_protection_reason(name: str) -> str | None:
    """Return a friendly, actionable reason ``name`` may NOT be used as the setup base env, or
    ``None`` if it's fine.

    ``_validate_env_name`` only checks a name is a *syntactically legal* conda token — a reserved
    env like ``spatialomicsgym_env`` or ``base`` passes that check, so nothing else stops the wizard
    from selecting it as the base env and then **mutating** it: NEW-mode installs ``pip install -e .``
    into a pre-existing env, and REUSE installs the core requirements into the active env. Either
    silently modifies a :data:`PROTECTED_ENVS` member — which the whole design treats as hands-off
    (a stray ``--answers`` name, an LLM-invented suggestion, or simply running ``sog-setup`` from
    inside ``base`` are all realistic triggers). Refuse it here at the naming boundary so setup never
    creates-into or writes-into a protected env; the caller re-asks / fails the phase legibly.
    """
    if (name or "").strip() in PROTECTED_ENVS:
        return (
            f"{name!r} is a reserved environment SpatialOmicsLab relies on — setup must never modify "
            "it. Pick a different name (e.g. 'sog') or activate the env you actually want to reuse."
        )
    return None


# Artifact-cleanup guard: "delete results" may only touch the artifact dir.
# It must never reach a conda env or the durable state.
def assert_deletable_artifact(path: Path) -> None:
    """Raise unless ``path`` lives strictly inside the per-run artifact dir."""
    p = Path(path).resolve()
    root = artifact_dir().resolve()
    # N13: derive the conda envs root from the running interpreter — never a hardcoded
    # ``/opt/conda/envs`` (wrong/inert on a micromamba or ``~/miniconda3`` box). Defense-in-depth
    # under the primary artifact-dir containment guard below.
    forbidden = (Path(conda_envs_root()).resolve(), state_dir().resolve())
    for bad in forbidden:
        try:
            p.relative_to(bad)
            raise PermissionError(f"refusing to delete {p}: inside protected {bad}")
        except ValueError:
            pass
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"refusing to delete {p}: outside artifact dir {root}") from exc


# --------------------------------------------------------------------------- #
# Thresholds / timeouts (seconds unless noted)
# --------------------------------------------------------------------------- #
def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
    """Read an int from ``os.environ[name]``, clamped to ``[lo, hi]``; fall back to ``default`` on an
    unset/blank/unparseable value. Lets a power user widen (or a CI run tighten) a self-heal loop
    budget without touching code, while the clamp keeps a fat-fingered value (``0``, ``-1``, a giant
    number) from wedging or run-away-looping the harness. Read ONCE at import — the loop bound is set
    for the process, not re-read mid-run."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, min(hi, int(raw.strip())))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, *, lo: float, hi: float) -> float:
    """:func:`_int_env` for a float budget (wall-clock seconds) — same clamp + fall-back discipline."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, min(hi, float(raw.strip())))
    except (TypeError, ValueError):
        return default


MIN_DISK_GB_BASE = 15  # hard floor for a base env
MIN_DISK_GB_PER_TOOL = 5  # advisory floor per tool env
WARN_DISK_GB = 30  # warn if free space below this before a big run

NETWORK_TIMEOUT_SEC = 10  # preflight connectivity probe
LLM_PING_TIMEOUT_SEC = 30  # Tier-1 REST validation
CONDA_ENVLIST_TIMEOUT_SEC = 60
CONDA_CREATE_TIMEOUT_SEC = 3600  # a heavy solve can take an hour
CONDA_CLONE_TIMEOUT_SEC = 1800  # cloning a present env is faster than a solve
CONDA_REMOVE_TIMEOUT_SEC = 600  # tearing down an env before RECREATE
PIP_INSTALL_TIMEOUT_SEC = 1800  # pip layer on top of a conda env
CONDA_EXPORT_TIMEOUT_SEC = 300  # `conda env export` / `pip freeze` for capture
CONDA_RUN_TIMEOUT_SEC = 120  # quick in-env probes
WORKER_TEST_TIMEOUT_SEC = 900  # Tier-1 worker on mini data
AGENT_TEST_TIMEOUT_SEC = 1800  # Tier-2 agent.go on a category
# Self-heal loop budgets — "Aggressive" defaults, each overridable by the matching SOG_* env var
# (clamped, parsed ONCE at import). The env var is read a single time here; the loop then reads the
# *module attribute* (``constants.MAX_ENV_REPAIRS`` …) on every turn, so raising a default widens the
# loop with no loop-code change and a test can monkeypatch the attribute — but the SOG_* env var itself
# is NOT re-read mid-run. The no-progress signature guard still stops early when an error is unchanged,
# so a bigger cap is only ever spent on turns that make real progress (never futile identical retries).
SELF_REVIEW_BUDGET_SEC = _float_env("SOG_SELF_REVIEW_BUDGET_SEC", 3600.0, lo=60, hi=14400)  # was 1800; wall-clock/tool
MAX_ENV_REPAIRS = _int_env(
    "SOG_MAX_ENV_REPAIRS", 6, lo=1, hi=20
)  # was 2; deterministic env auto-repairs/tool (envdoctor)
MAX_REACT_STEPS = _int_env("SOG_MAX_REACT_STEPS", 8, lo=1, hi=20)  # was 3; LLM-planner turns/tool (Lane-2 hand-off)
# Round-2 (Part C) self-heal bounds — same read-once / clamp / SOG_* override discipline.
MAX_TRANSIENT_RETRIES = _int_env("SOG_MAX_TRANSIENT_RETRIES", 3, lo=1, hi=10)  # Lane-4 network retries/tool (C5)
MAX_ENV_RECREATES = _int_env("SOG_MAX_ENV_RECREATES", 1, lo=0, hi=10)  # destructive full-env RECREATEs/run (C2)
# Absolute per-tool turn ceiling — a cheap final backstop *under* the per-lane caps so the ``while True``
# self-heal loop can never outrun the sum of budgets, even under maxed overrides or an unforeseen lane
# interaction (C4). Default ≈ the sum of the lane budgets + a small margin.
MAX_TURNS = _int_env("SOG_MAX_TURNS", MAX_ENV_REPAIRS + MAX_REACT_STEPS + MAX_TRANSIENT_RETRIES + 6, lo=1, hi=200)
# Wall-clock floor: never LAUNCH another repair turn when fewer than this many seconds of the budget
# remain — a guarded build/pip can run for many minutes, and the between-turns wall-clock check alone
# would let a turn started right at the deadline overrun by a whole subprocess timeout (C4).
SELF_HEAL_MIN_TURN_SEC = _float_env("SOG_SELF_HEAL_MIN_TURN_SEC", 60.0, lo=1, hi=3600)
# Transient-retry backoff: sleep ``base * 2**(retry-1)`` seconds before re-running the same build,
# clamped to the remaining budget AND this ceiling so a retry storm can never itself blow the
# wall-clock (C5). Jitter is derived from the monotonic clock, not ``random`` (the setup package
# avoids ``random`` for reproducibility).
TRANSIENT_RETRY_BACKOFF_SEC = _float_env("SOG_TRANSIENT_RETRY_BACKOFF_SEC", 2.0, lo=0, hi=120)
TRANSIENT_RETRY_BACKOFF_MAX_SEC = _float_env("SOG_TRANSIENT_RETRY_BACKOFF_MAX_SEC", 30.0, lo=0, hi=600)
STREAM_LOG_MAX_LINES = 400  # cap on per-line `build_line` events teed to the SessionLog JSONL when a build is
# live-streamed (opt-in SOG_STREAM_BUILD_LOGS). The full-fidelity, `tail -f`-able sink is the per-tool
# `<target>.build.log`; the JSONL keeps only a bounded head so a chatty solve can't explode the transcript.
# Log-hygiene caps (C7): keep at most this many NEWEST `*.build.log` / `run-*.jsonl` per glob under
# `.sog_setup/logs`, and at most this many total bytes — the oldest are pruned on state-dir init so a
# machine that runs the wizard many times cannot accumulate logs without bound.
LOG_KEEP_MAX_FILES = _int_env("SOG_LOG_KEEP_MAX_FILES", 400, lo=10, hi=100_000)
LOG_KEEP_MAX_BYTES = _int_env("SOG_LOG_KEEP_MAX_BYTES", 200 * 1024 * 1024, lo=1024 * 1024, hi=8 * 1024 * 1024 * 1024)
# Per-file build-log size clip (C7): a single failed build's persisted stderr is clipped to this many
# bytes (keeping the diagnostic TAIL) in both the `<target>.build.log` file and the JSONL row, so one
# pathological multi-hundred-MB solve log cannot bloat the state dir or a transcript event.
BUILD_LOG_CLIP_BYTES = _int_env("SOG_BUILD_LOG_CLIP_BYTES", 256 * 1024, lo=4 * 1024, hi=64 * 1024 * 1024)

# Recognized opt-in env flags for the assisted (LLM) remediation tiers — all default OFF, all no-ops
# without an LLM handle, all env-only. Read inline at each call site (see testing.py / provision.py):
#   SOG_PROVISION_LLM_REMEDIATION  — PROVISION phase may consult the planner
#   SOG_TEST_LLM_REMEDIATION       — Tier-1 test phase may consult the planner (closes the Tier-1 gap)
#   SOG_SELF_REVIEW_ENABLED        — the deep RemediationContext self-review path
# The deterministic repairs (envdoctor.repair_env / repair_diagnosis) are governed separately by
# SOG_PROVISION_AUTOREPAIR / SOG_ENV_AUTOREPAIR (default ON) — no LLM, no key needed.

# --------------------------------------------------------------------------- #
# Mini datasets (local, reliable) + optional public data mirror
# --------------------------------------------------------------------------- #
MINI_DATA_DIRNAME = "test/test_data"  # local mini_*.h5ad live here (the untracked test/ tree)
MINI_SPATIAL = "mini_spatial.h5ad"
MINI_SC_REF = "mini_sc_ref.h5ad"
MINI_VISIUM_HE = "mini_visium_he.h5ad"  # optional local Visium+H&E slide (gitignored, full checkout only)

# The mini_*.h5ad above are gitignored even inside test/test_data/. The small AnnData files below
# sit beside them in test/test_data/creation_demo/ and are what the real-data probe / preflight /
# Tier-1 fall back to when the mini files are absent. test/ itself is the untracked development tree,
# so a fresh clone or a wheel install has neither; the callers say so (TEST_TREE_MISSING) rather
# than reporting a bare missing file. Verified: demo_spatial has obsm['spatial'] + total_counts, 300
# genes, cell_type TypeA/TypeB/TypeC; demo_sc_ref is a COHERENT deconvolution partner built
# from the slide's own per-type signal (build_demo_sc_ref.py) — the SAME 300 genes, the same
# TypeA/B/C vocabulary, obs CellType/cell_type/louvain + two Sample batches, raw counts with
# separable per-type signatures. Together they are a drop-in for the gitignored
# mini_spatial/mini_sc_ref in the Tier-1 smoke cases AND a runnable deconvolution demo pair.
FALLBACK_SPATIAL_REL = "creation_demo/demo_spatial.h5ad"
FALLBACK_SC_REF_REL = "creation_demo/demo_sc_ref.h5ad"
# A Visium + H&E slide for the image tools (cell/nucleus segmentation). Derived from demo_spatial
# (same 300 genes, same 150 spots) with a canonical scanpy layout added: obsm['spatial'] full-res
# coords + uns['spatial'][lib]['images']['hires'/'lowres'] RGB + the four Visium scalefactors. Small
# (< 2 MB), so a development tree can exercise a segmentation demo. build_demo_visium_he.py.
FALLBACK_IMAGE_REL = "creation_demo/demo_visium_he.h5ad"

# Public release mirror — OPTIONAL and warn-only (may 404 post-rename). The
# wizard prefers the local mini datasets above and never hard-fails on S3.
DEFAULT_PUBLIC_DATA_URL = "https://spatialomicsgym-release.s3.amazonaws.com"


def mini_data_dir() -> Path:
    """``test/test_data`` of the development tree (:func:`dev_test_dir`).

    When ``test/`` is absent the repo-root spelling is still returned, so a caller's "not found"
    names the place the data would live; :func:`mini_data_note` gives the sentence to add to it.
    """
    tree = dev_test_dir()
    return tree / "test_data" if tree is not None else repo_path(MINI_DATA_DIRNAME)


def mini_data_note() -> str:
    """Why the mini data may be missing: ``""`` when ``test/`` is present, else :data:`TEST_TREE_MISSING`."""
    return "" if dev_test_dir() is not None else TEST_TREE_MISSING


def fallback_spatial_dataset() -> Path:
    """The small spatial dataset (``test/test_data/creation_demo``) used when the gitignored
    mini_*.h5ad are absent. Resolves under :func:`mini_data_dir` so test redirection of
    the mini-data dir carries the fallback with it."""
    return mini_data_dir() / FALLBACK_SPATIAL_REL


def fallback_sc_ref_dataset() -> Path:
    """The small single-cell reference (``test/test_data/creation_demo``) used when the gitignored
    ``mini_sc_ref.h5ad`` is absent. The deconvolution/mapping Tier-1 cases
    pass this as ``--sc-h5ad`` / ``--scrna-h5ad``; it carries ``obs['CellType']`` /
    ``obs['louvain']`` (cell types matched to the slide's TypeA/B/C) + ``obs['Sample']`` and
    the SAME 300 genes as the slide, so cell2location/tangram deconvolve it to completion.
    Resolves under :func:`mini_data_dir` for the same test-redirection reason as the spatial
    fallback."""
    return mini_data_dir() / FALLBACK_SC_REF_REL


def fallback_image_dataset() -> Path:
    """The small Visium + H&E slide (``test/test_data/creation_demo``) used when the gitignored
    ``mini_visium_he.h5ad`` is absent. The image-segmentation Tier-1/demo cases (cellpose, deepcell, bidcell,
    clustermap) need a histology image: this slide carries ``uns['spatial'][lib]['images']`` (hires +
    lowres RGB) + the four Visium scalefactors + ``obsm['spatial']``, on the SAME 300 genes as the
    slide. Resolves under :func:`mini_data_dir` for the same test-redirection reason as the others."""
    return mini_data_dir() / FALLBACK_IMAGE_REL


# --------------------------------------------------------------------------- #
# UI voice — mirrors the agent's config banner (stcoscientist.py:138) and the
# plan-first checklist (prompt_builder.py:212). Reused by prompts.py.
# --------------------------------------------------------------------------- #
BANNER_RULE = "=" * 50
PRODUCT_NAME = "SPATIALOMICSLAB"
CHECK_TODO = "[ ]"
CHECK_DONE = "[✓]"
CHECK_FAIL = "[✗]"

# The roadmap shown at the top of a run and re-stated at the Stage-A→B handoff.
ROADMAP_STEPS: tuple[str, ...] = (
    "Check your machine (conda, disk, GPU, network)",
    "Connect your LLM",
    "Pick the analysis tools you want",
    "Set up an environment per tool",
    "Test each on a tiny dataset",
    "Wrap up",
)
