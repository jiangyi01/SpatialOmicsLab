"""Where the *platform trees* live: ``tools/``, ``MCP_server/``, ``tools_user/`` (the agent part),
and the install recipes (``tool_specs/``) -- the parts of the system that are directories beside
the package, not modules inside it.

Two layouts carry them, and every helper here answers for both:

* **the repository** (a checkout, or an editable install of one): the instance root is the
  repository root -- runtime state (``.sog_setup/``, ``.env``, ``work/``, ``data/``) lives there --
  while the agent trees sit one level down under ``agent/`` (``agent/tools``, ``agent/MCP_server``,
  ``agent/tools_user``) and the recipes under ``install/recipes/`` (``tool_specs/``, the generated
  ``mcp_config.setup.yaml``, ``env_overrides.env``);
* **the packaged copy** (the wheel's read-only ``spatialomicsgym/_platform/``) and the **seeded
  home** built from it: flat -- ``tools/``, ``MCP_server/``, ``tools_user/`` and ``setup/tool_specs/``
  directly under the root, the layout every frozen resolver inside the copied portal files was
  written against.

:func:`is_repository_layout` tells the two apart (a root with ``agent/`` or ``install/recipes/`` is
laid out like the repository); :func:`platform_dir` and :func:`recipes_dir` join through it, so a
caller holding any root -- the instance root, a seeded home, ``_platform`` -- asks them instead of
joining ``"tools"`` or ``"setup"`` by hand.

The module earns its keep on a **pip-only install**, where there is no checkout: the wheel
carries the packaged copy (injected at wheel-build time by ``setup.py``; absent in checkouts and
editable installs, which serve the real trees), and the writable instance state lives under
**SOG_HOME** (default ``~/.spatialomicsgym``), seeded from that copy by the setup wizard.

Layering: stdlib-only, filesystem-read-only, and importable before any conda env exists --
``sog_install/constants.py`` delegates its root resolution here, and that module holds the same
promises. Nothing here creates a directory; seeding is the wizard's explicit job
(``sog_install/home_seed.py``), never a side effect of asking where things are.

Resolution order for :func:`instance_root` (first hit wins):

  W1. ``SOG_PLATFORM_ROOT``  -- operator override, the same empty-is-unset idiom as the
                                sibling ``SOG_SETUP_REPO_ROOT`` seam.
  W2. the checkout above the package -- ``<repo>/agent/spatialomicsgym`` makes ``<repo>`` the
                                answer when it carries the trees; ``site-packages`` never does,
                                so this rung cannot misfire there.
  W3. the current directory   -- an operator standing in a checkout (or a seeded SOG_HOME)
                                while running a console script installed elsewhere.
  W4. :func:`sog_home`        -- the pip-only default; may not exist yet (the wizard seeds it).

A W1/W3 answer that is a checkout's own ``agent/`` directory is lifted to the checkout root: it
carries the trees too, but state written there would be a second, invisible instance.

Zip-imports are unsupported (already true platform-wide: portal scripts are executed by path).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = [
    "PLATFORM_ROOT_ENV",
    "SOG_HOME_ENV",
    "AGENT_PART",
    "RECIPES_PART",
    "PACKAGED_RECIPES_PART",
    "PLATFORM_PAYLOAD",
    "PAYLOAD_PACKAGED_DIRS",
    "PLATFORM_MANIFEST_NAME",
    "sog_home",
    "is_repository_layout",
    "platform_dir",
    "recipes_dir",
    "is_platform_root",
    "running_from_checkout",
    "instance_root",
    "packaged_platform_dir",
    "packaged_canonical_config",
    "resource_root",
    "tools_dir",
    "payload_files",
    "packaged_relpath",
    "describe_search",
]

#: Operator override for the instance root (rung W1). Empty string counts as unset.
PLATFORM_ROOT_ENV = "SOG_PLATFORM_ROOT"

#: Override for the writable pip-only instance root (rung W4). Empty string counts as unset.
SOG_HOME_ENV = "SOG_HOME"

#: Where the agent trees (``tools/``, ``tools_user/``, ``MCP_server/``) sit under a root laid out like the
#: repository. Under a packaged or seeded root they sit at the top.
AGENT_PART = "agent"

#: Where the install recipes (``tool_specs/``, ``mcp_config.setup.yaml``, ``env_overrides.env``) sit under a
#: root laid out like the repository ...
RECIPES_PART = "install/recipes"

#: ... and under a packaged or seeded root -- the directory name they had before the re-layout.
PACKAGED_RECIPES_PART = "setup"

#: Name of the build-generated inventory inside ``_platform/`` (version + relative paths).
#: Its presence is what makes a ``_platform`` directory trustworthy: a half-copied tree
#: without it is treated as absent rather than served.
PLATFORM_MANIFEST_NAME = "PLATFORM_MANIFEST.json"

#: The one statement of what the wheel's ``_platform/`` payload contains, resolved by
#: :func:`payload_files`. ``setup.py`` loads this module **by path** at build time (never by
#: importing the package -- build isolation may lack the package's dependencies), the seeder
#: copies from it, and the payload pin test counts it; three consumers, one spec.
#:
#: Shape: ``(directory-relative-to-the-repository-root, (non-recursive fnmatch patterns...))``.
#: Each directory lands in the packaged copy under its :data:`PAYLOAD_PACKAGED_DIRS` name.
#: Directories are never walked recursively and only regular files match -- ``agent/tools/`` on a
#: dev box carries gitignored multi-GB trees (``tools/data``, ``tools/third_party``) that a
#: ``copytree`` would ship; explicit per-directory patterns cannot.
#:
#: ``tools_user`` names its files exactly: the directory admits machine-local extras (user-created
#: tools) that must never ride along. The creation-demo fixtures no longer ship: they live under the
#: untracked ``test/`` tree, which a release does not carry.
PLATFORM_PAYLOAD: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("agent/tools", ("*.py", "*.R", ".ruff.toml")),
    ("agent/MCP_server", ("mcp_config.yaml",)),
    ("install/recipes/tool_specs", ("*.yaml",)),
    ("install/recipes/tool_specs/env", ("*.yaml",)),
    (
        "agent/tools_user",
        (
            ".ruff.toml",
            "base_mcp.py",
            "create_all_21_tools.py",
            # The declarative tool tier the portal's Created-tools routes and the REPL import, and the
            # env reclaimer the creation story documents. Unshipped, a wheel install answered
            # /api/tools/mine with a 500 (hunt 2026-09-30, uL5-drift-4).
            "declarative.py",
            "reclaim_envs.py",
            "knowledge_manager.py",
            "memory_manager.py",
            "seed_tool_creation_memory.py",
            "self_review.py",
            "trash_manager.py",
            "user_skill.py",
            "worker_utils.py",
        ),
    ),
)

#: Where each :data:`PLATFORM_PAYLOAD` directory lands in the packaged copy (and so in a seeded home):
#: the flat, pre-re-layout names. ``tools/``, ``tools_user/`` and ``MCP_server/`` stay siblings in
#: both layouts -- the frozen portal files find each other as ``dirname(dirname(__file__))/<name>`` --
#: and only the recipes change their name (``install/recipes`` -> ``setup``).
PAYLOAD_PACKAGED_DIRS: dict[str, str] = {
    "agent/tools": "tools",
    "agent/MCP_server": "MCP_server",
    "install/recipes/tool_specs": "setup/tool_specs",
    "install/recipes/tool_specs/env": "setup/tool_specs/env",
    "agent/tools_user": "tools_user",
}

_CANONICAL_CONFIG_REL = ("MCP_server", "mcp_config.yaml")


def sog_home() -> Path:
    """The writable pip-only instance root. Never created here; the wizard seeds it.

    Always absolute: a relative answer would mean every process cwd names a different
    "home", and the seeder would happily create it there.
    """
    override = (os.environ.get(SOG_HOME_ENV) or "").strip()
    if override:
        return Path(os.path.abspath(os.path.expanduser(override)))
    home = os.path.expanduser("~")
    if not os.path.isabs(home):
        # HOME unset and no passwd entry for this uid (arbitrary-UID containers):
        # expanduser returns "~" verbatim, and joining it would make the seeder create a
        # literal "./~" directory under whatever the cwd happens to be. Fall back to a
        # stable absolute per-user location; operators who care set SOG_HOME.
        uid = getattr(os, "getuid", lambda: "any")()
        return Path(tempfile.gettempdir()) / f"spatialomicsgym-home-{uid}"
    return Path(home) / ".spatialomicsgym"


def is_repository_layout(root: Path | str) -> bool:
    """Whether ``root`` is laid out like the repository (``agent/``, ``install/recipes/``) rather than
    like the packaged copy and a seeded home (``tools/``, ``MCP_server/``, ``setup/`` at the top).

    Decided by the directories alone, so it also answers for a root that is still being built (a
    test's scratch tree, a half-provisioned box): any root without them reads as the flat layout.
    """
    base = Path(root)
    try:
        return (base / AGENT_PART).is_dir() or base.joinpath(*RECIPES_PART.split("/")).is_dir()
    except OSError:  # an unreadable candidate is just a candidate that does not apply
        return False


def platform_dir(root: Path | str | None = None) -> Path:
    """The directory holding ``tools/``, ``tools_user/`` and ``MCP_server/`` under ``root``.

    ``root`` defaults to :func:`instance_root`. ``<root>/agent`` when the root is laid out like the
    repository, else ``root`` itself (the packaged copy, a seeded home). A location, not a promise:
    nothing need exist there yet.
    """
    base = instance_root() if root is None else Path(root)
    return base / AGENT_PART if is_repository_layout(base) else base


def recipes_dir(root: Path | str | None = None) -> Path:
    """The directory holding ``tool_specs/``, ``mcp_config.setup.yaml`` and ``env_overrides.env``.

    ``root`` defaults to :func:`instance_root`. ``<root>/install/recipes`` when the root is laid out
    like the repository, else ``<root>/setup`` (the packaged copy, a seeded home).
    """
    base = instance_root() if root is None else Path(root)
    if is_repository_layout(base):
        return base.joinpath(*RECIPES_PART.split("/"))
    return base / PACKAGED_RECIPES_PART


def _carries_the_trees(directory: Path) -> bool:
    """The canonical config file **and** the ``tools/`` directory, directly under ``directory``."""
    try:
        return directory.joinpath(*_CANONICAL_CONFIG_REL).is_file() and (directory / "tools").is_dir()
    except OSError:
        return False


def is_platform_root(path: Path | str) -> bool:
    """Whether ``path`` carries the platform trees -- a checkout, or a seeded SOG_HOME.

    The marker is the canonical config file **and** the ``tools/`` directory together, looked for
    where :func:`platform_dir` says they sit (``agent/`` in a checkout, the top of a seeded home):
    no single stray file can make an unrelated directory answer as the platform, and both exist
    in every checkout since the portal layer landed and in every wizard-seeded home.
    """
    return _carries_the_trees(platform_dir(path))


def _lift_agent_dir(path: Path) -> Path:
    """A checkout's own ``agent/`` directory carries the trees too; the instance is the checkout."""
    if path.name == AGENT_PART and is_repository_layout(path.parent) and is_platform_root(path.parent):
        return path.parent
    return path


def _checkout_root() -> Path:
    # <repo>/agent/spatialomicsgym/platform_root.py -> <repo>. Derived from ``__file__`` at call time, so
    # a test that relocates the module (or a real site-packages install) is answered for where it is.
    return Path(__file__).resolve().parents[2]


def running_from_checkout() -> bool:
    """True when the imported package sits inside a checkout (or an editable install of one).

    The gate the serving layer uses to leave checkout behavior alone: pip-only fallbacks
    (SOG_HOME, the packaged ``_platform`` copy) engage only when this is False. The checkout is
    the directory two levels above the package (``<repo>/agent/spatialomicsgym``), laid out like
    the repository and carrying the trees under ``agent/``.
    """
    root = _checkout_root()
    return is_repository_layout(root) and is_platform_root(root)


def instance_root() -> Path:
    """The directory this process should treat as its instance root (module docstring order).

    In a checkout this is the **repository root** -- where ``.sog_setup/``, ``.env`` and the
    recipes live -- not ``agent/``; join the agent trees through :func:`platform_dir`.

    Always answers -- the last rung is unconditional -- but the answer is a *location*, not a
    promise of contents: rung W4 may name a SOG_HOME that has never been seeded. Callers that
    need the trees to actually exist ask :func:`resource_root` instead.
    """
    override = (os.environ.get(PLATFORM_ROOT_ENV) or "").strip()
    if override:
        return _lift_agent_dir(Path(os.path.abspath(os.path.expanduser(override))))

    # Rung W2 routes through running_from_checkout() -- the same check on the same directory --
    # so this resolver and the serving layer's off-checkout gates can never disagree about
    # whether the checkout rung applies.
    if running_from_checkout():
        return _checkout_root()

    try:
        cwd = Path(os.getcwd())
    except OSError:
        cwd = None  # a deleted working directory is a rung that does not apply, not a crash
    if cwd is not None and is_platform_root(cwd):
        return _lift_agent_dir(cwd)

    return sog_home()


def packaged_platform_dir() -> Path | None:
    """The wheel's read-only ``_platform/`` copy, or ``None`` in a checkout/editable install.

    Trusted only when its build-generated manifest is present -- see
    :data:`PLATFORM_MANIFEST_NAME`. Laid out flat (see :data:`PAYLOAD_PACKAGED_DIRS`).
    """
    candidate = Path(__file__).resolve().parent / "_platform"
    try:
        if (candidate / PLATFORM_MANIFEST_NAME).is_file():
            return candidate
    except OSError:
        pass
    return None


def packaged_canonical_config() -> Path | None:
    """The canonical MCP config inside the wheel payload, or ``None`` when not packaged.

    Deliberately distinct from :func:`resource_root`: ``find_mcp_config`` appends exactly
    this as its LAST rung -- never a SOG_HOME rung -- so its search stays machine-independent.
    """
    packaged = packaged_platform_dir()
    if packaged is None:
        return None
    # Flat (``_platform/MCP_server/...``) as ``setup.py`` writes it; joined through platform_dir so a
    # copy laid out like the repository would be read too.
    candidate = platform_dir(packaged).joinpath(*_CANONICAL_CONFIG_REL)
    return candidate if candidate.is_file() else None


def resource_root() -> Path | None:
    """Where the platform trees can actually be *read* right now.

    The instance root when it carries them (checkout, seeded SOG_HOME), else the wheel's
    ``_platform`` copy, else ``None``. Either layout may come back -- join through
    :func:`platform_dir` / :func:`recipes_dir`, never by hand. ``None`` rather than a raise for
    the same reason ``find_mcp_config`` returns ``None``: import-time consumers must degrade, and
    callers that need a path say so themselves -- :func:`describe_search` gives them the
    actionable sentence to raise with.
    """
    root = instance_root()
    if is_platform_root(root):
        return root
    return packaged_platform_dir()


def tools_dir() -> Path | None:
    """The readable ``tools/`` directory (portal servers + workers), or ``None``."""
    root = resource_root()
    if root is None:
        return None
    candidate = platform_dir(root) / "tools"
    return candidate if candidate.is_dir() else None


def payload_files(root: Path | str) -> list[str]:
    """Resolve :data:`PLATFORM_PAYLOAD` against ``root`` -- sorted, slash-separated relpaths.

    ``root`` laid out like the repository (the wheel build's source, the pin tests' checkout): the
    :data:`PLATFORM_PAYLOAD` directories, and repository-relative paths (``agent/tools/x.py``).
    Any other root (the packaged ``_platform`` copy, a seeded home): the
    :data:`PAYLOAD_PACKAGED_DIRS` names, and packaged paths (``tools/x.py``) -- the same strings the
    build's manifest records. :func:`packaged_relpath` maps the first form to the second.

    Only regular files that exist are returned; patterns are non-recursive within their own
    directory by construction. Shared by the wheel build, the SOG_HOME seeder, and the payload
    pin test so the three can never disagree about what "the payload" means.
    """
    base = Path(root)
    repository = is_repository_layout(base)
    selected: set[str] = set()
    for rel_dir, patterns in PLATFORM_PAYLOAD:
        if not repository:
            rel_dir = PAYLOAD_PACKAGED_DIRS[rel_dir]
        directory = base / rel_dir
        if not directory.is_dir():
            continue
        for pattern in patterns:
            for hit in directory.glob(pattern):
                # is_file() follows symlinks on purpose: the tools_user twins are checked-in
                # links to tools/, and the payload ships their real bytes (copies are blessed
                # by the byte-equality pin in test_repo_hygiene).
                if hit.is_file():
                    selected.add(f"{rel_dir}/{hit.name}")
    return sorted(selected)


def packaged_relpath(rel: str) -> str:
    """Where a repository-relative payload path lands in the packaged copy.

    ``agent/tools/x.py`` -> ``tools/x.py``; ``install/recipes/tool_specs/env/y.yaml`` ->
    ``setup/tool_specs/env/y.yaml``. Raises ``KeyError`` for a path outside every payload directory
    -- the build must never invent a destination for a file the spec did not select.
    """
    directory, _, name = rel.rpartition("/")
    return f"{PAYLOAD_PACKAGED_DIRS[directory]}/{name}"


def describe_search() -> str:
    """A one-line account of where the platform trees were looked for, for error messages."""
    packaged = packaged_platform_dir()
    return (
        f"looked for the platform trees at {instance_root()} (rungs: ${PLATFORM_ROOT_ENV}, "
        f"the checkout above the package, the current directory, then {sog_home()}) and at the "
        f"packaged copy ({packaged if packaged else 'not present in this install'}); "
        f"set ${PLATFORM_ROOT_ENV} to a checkout, or run sog-setup to seed the home instance"
    )
