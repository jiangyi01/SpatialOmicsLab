"""Find ``mcp_config.yaml`` -- the file that says which portals exist and what they accept.

Three modules used to answer this by walking up from their own ``__file__`` and stopping there::

    Path(__file__).resolve().parent.parent.parent / "MCP_server" / "mcp_config.yaml"

(in today's checkout that file is ``agent/MCP_server/mcp_config.yaml``, beside the package).

That is the right first guess and it is correct on a clone, but ``MCP_server/`` is not in the
wheel: ``pyproject.toml`` includes ``spatialomicsgym*``, ``skills*`` and ``benchmarks*``, and
ships package data only for its own modules. ``MCP_server`` matches none of
them, and ``MANIFEST.in`` governs the sdist rather than the wheel. So on a ``pip install`` the
walk lands in ``site-packages`` and the file is not there -- with no way to say where it really
is. Measured on a copy of that layout, ``transcriptomics_skills`` went from 114 registered tools
and 114 parameter contracts to 0 and 0, and ``_build_tool_params`` fell back to naming
``st_h5ad`` for every tool -- true for 22 of the 114, and wrong for 31 of the 44 the planner can
reach.

**Precedence, and why it differs from the setup layer's.** ``chat_cli._resolve_mcp_config`` and
``sog_install.conncheck._discover_config`` answer "which config should this deployment *serve*", so
they put the wizard's recorded ``SOG_MCP_CONFIG`` pointer first. This function answers a
different question -- "which portals does *this package* ship, and what do they accept" -- and so
it puts the package-relative file first. A stale pointer must not be able to silently rewrite
the parameter names in every plan the agent emits on a working checkout, nor swap the canonical
88-server catalogue for a generated one describing a subset of it.

The remaining rungs only matter once the package-relative file is gone, i.e. on exactly the
installs this module exists for.
"""

from __future__ import annotations

import os
from pathlib import Path

from spatialomicsgym import layout
from spatialomicsgym.platform_root import packaged_canonical_config, platform_dir, recipes_dir

#: Where the pointer the setup wizard records in ``.env`` is read from.
POINTER_ENV_VAR = "SOG_MCP_CONFIG"

#: The canonical config, relative to the directory holding the agent trees (``agent/`` in a checkout --
#: :func:`spatialomicsgym.platform_root.platform_dir`).
CANONICAL_REL = ("MCP_server", "mcp_config.yaml")
#: The wizard's generated config, relative to a checkout root. A root laid out like the packaged copy
#: (a seeded home) keeps it under ``setup/``; the cwd rung below asks ``recipes_dir`` which one applies.
GENERATED_REL = ("install", "recipes", "mcp_config.setup.yaml")


def _canonical_config_default() -> str:
    mcp_server = layout.mcp_server_dir()
    if mcp_server is not None:
        return str(mcp_server / CANONICAL_REL[-1])
    return "/".join(CANONICAL_REL)


#: The canonical config as a *default argument* (``add_mcp``, the user-tool resync, the merger): the
#: checkout's own ``<repo>/agent/MCP_server/mcp_config.yaml`` (``layout.mcp_server_dir()``), absolute so
#: no working directory can change what it names. Off a checkout there is no such directory and it
#: keeps the bare relative spelling ``MCP_server/mcp_config.yaml``, which a seeded home answers when it
#: is the working directory. A default, not a search: :func:`find_mcp_config` is the search.
CANONICAL_CONFIG_DEFAULT = _canonical_config_default()


def candidate_mcp_configs() -> list[Path]:
    """The paths :func:`find_mcp_config` considers, in the order it considers them."""
    # <repo>/agent/spatialomicsgym/mcp_config_path.py -> <repo>/agent/MCP_server/mcp_config.yaml.
    candidates: list[Path] = [Path(__file__).resolve().parents[1].joinpath(*CANONICAL_REL)]

    pointer = (os.environ.get(POINTER_ENV_VAR) or "").strip()
    if pointer:
        candidates.append(Path(os.path.expanduser(pointer)))

    # A clone that has been through the wizard serves the generated config, so it outranks the
    # in-tree canonical one here -- the order chat_cli already uses. A deleted working
    # directory is a rung that does not apply, never a raise: this function feeds import-time
    # catalogue caches whose contract is to degrade, not to make the package unimportable.
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    if cwd is not None:
        # A checkout root (install/recipes/, agent/MCP_server/) or a seeded home (setup/, MCP_server/).
        candidates.append(recipes_dir(cwd) / GENERATED_REL[-1])
        candidates.append(platform_dir(cwd).joinpath(*CANONICAL_REL))

    # Last rung: the read-only copy a self-contained wheel carries under ``_platform/`` (absent
    # in checkouts and editable installs, so this appends nothing there). Last on purpose --
    # everything the operator can point at or stand in outranks the build-frozen copy, and the
    # rung's presence depends only on the install itself, never on this machine's home
    # directory, so the relocated-package answers above stay machine-independent.
    packaged = packaged_canonical_config()
    if packaged is not None:
        candidates.append(packaged)
    return candidates


def find_mcp_config() -> Path | None:
    """Return the first candidate that is a readable file, or ``None`` if there is none.

    ``None`` rather than a raise: the two catalogue caches are built at import time and a
    missing catalogue must degrade to an empty one rather than make the package unimportable.
    Callers that need a path say so themselves.
    """
    for candidate in candidate_mcp_configs():
        try:
            if candidate.is_file():
                return candidate
        except OSError:  # an unreadable parent directory is just a candidate that does not apply
            continue
    return None


def describe_search() -> str:
    """A one-line account of where we looked, for error messages that must be actionable."""
    looked = ", ".join(str(c) for c in candidate_mcp_configs())
    return f"looked for mcp_config.yaml in: {looked} (set {POINTER_ENV_VAR} to point at it)"
