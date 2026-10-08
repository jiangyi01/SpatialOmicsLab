"""Where the repository's parts live on disk -- computed here, and only here.

The checkout is laid out as top-level parts (``agent/``, ``install/``). Running code asks this module for a location instead of counting
``Path(__file__).parents[N]``, so moving this package deeper (under ``agent/``) changes one function, not
every caller.

:func:`repo_root` walks up from this file to the directory whose ``pyproject.toml`` names this project.
It is ``None`` in a non-editable install (site-packages has no such file above it); every location
helper then falls back to the copy the wheel build placed inside the package (``setup.py``).

Stdlib only: ``setup.py`` and other early code may load this by path.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

__all__ = [
    "repo_root",
    "agent_dir",
    "tools_dir",
    "tools_user_dir",
    "mcp_server_dir",
    "skills_dir",
    "benchmarks_dir",
    "install_dir",
    "recipes_dir",
    "test_dir",
]

_HERE = Path(__file__).resolve().parent

#: The ``[project]`` name line that identifies OUR pyproject, not one belonging to a project that merely
#: holds a virtualenv this package is installed into.
_PROJECT_NAME_RE = re.compile(r'^\s*name\s*=\s*["\']spatialomicsgym["\']\s*$', re.MULTILINE)

def _is_repo_root(path: Path) -> bool:
    pyproject = path / "pyproject.toml"
    try:
        return pyproject.is_file() and bool(_PROJECT_NAME_RE.search(pyproject.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError):
        return False


@functools.lru_cache(maxsize=1)
def repo_root() -> Path | None:
    """The checkout this package runs from, or ``None`` when it runs from an installed wheel."""
    # _HERE is <repo>/agent/spatialomicsgym in a checkout.
    for candidate in (_HERE, *_HERE.parents):
        if _is_repo_root(candidate):
            return candidate
    return None


# --- the other top-level parts (checkout layout) -------------------------------------------------------------
# Each returns the checkout location, or ``None`` in a non-editable install (callers that need a wheel fallback
# use ``platform_root``). ``agent_dir`` is the AGENT part (tools, tools_user, MCP_server live under it); it is not
# the repository root -- runtime state (.sog_setup, .env, work/, data/, spatial_library/) stays at repo_root().


def _part(*parts: str) -> Path | None:
    root = repo_root()
    return None if root is None else root.joinpath(*parts)


def agent_dir() -> Path | None:
    return _part("agent")


def tools_dir() -> Path | None:
    return _part("agent", "tools")


def tools_user_dir() -> Path | None:
    return _part("agent", "tools_user")


def mcp_server_dir() -> Path | None:
    return _part("agent", "MCP_server")


def skills_dir() -> Path | None:
    return _part("agent", "skills")


def benchmarks_dir() -> Path | None:
    return _part("agent", "benchmarks")


def install_dir() -> Path | None:
    return _part("install")


def recipes_dir() -> Path | None:
    """``install/recipes`` -- tool_specs/, the generated mcp_config.setup.yaml and env_overrides.env."""
    return _part("install", "recipes")


def test_dir() -> Path | None:
    """``test/`` -- untracked: the suite, smoke harness and test data. Absent on a fresh clone; callers say so."""
    return _part("test")
