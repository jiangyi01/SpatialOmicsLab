"""Wheel-build hook that ships the platform trees inside the package.

Everything declarative lives in ``pyproject.toml``; this file exists ONLY to register a
``build_py`` subclass that copies the MCP portal layer -- ``agent/tools/``, the canonical
``agent/MCP_server/mcp_config.yaml``, the ``install/recipes/tool_specs/`` corpus and the
``agent/tools_user/`` helper twins -- into ``spatialomicsgym/_platform/`` at wheel-build time.

The packaged copy is laid out FLAT, not like the repository: ``_platform/tools``,
``_platform/MCP_server``, ``_platform/tools_user`` and ``_platform/setup/tool_specs`` -- the
layout every sibling-directory resolver in the frozen portal files was written against, and the
one a seeded SOG_HOME copies. ``platform_root.PAYLOAD_PACKAGED_DIRS`` maps each repository
directory to its packaged name; ``platform_root.packaged_relpath`` applies it to one file.

Why a build-time copy and not a committed mirror: the hygiene suite pins that no second
``base_mcp.py`` exists in the tree (drift hazard), so the copy may exist only inside build
products. Why not ``copytree``: ``agent/tools/`` on a dev box carries multi-GB gitignored trees
(``tools/data``, ``tools/third_party``); the payload is an explicit per-directory spec.

The spec itself lives ONCE, in ``agent/spatialomicsgym/platform_root.py`` (``PLATFORM_PAYLOAD``,
``PAYLOAD_PACKAGED_DIRS`` + ``payload_files``). It is loaded here **by path**, never by importing the
package: the PEP 517 build environment has none of the package's dependencies installed, and must
not need them.

Editable installs (``pip install -e .``) skip the copy entirely -- the checkout serves the real
trees, and the resolvers' first rung finds them there.

``setup()`` runs under the ``__main__`` guard -- setuptools' PEP 517 backend executes this file
with ``__name__ == "__main__"`` (``build_meta.run_setup``), and the guard is what lets the test
suite import the hook logic by path without starting a build.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

_HERE = Path(__file__).resolve().parent


def _load_platform_root_by_path():
    module_path = _HERE / "agent" / "spatialomicsgym" / "platform_root.py"
    spec = importlib.util.spec_from_file_location("_sog_build_platform_root", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - a broken tree cannot build
        raise RuntimeError(f"cannot load the payload spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_platform_payload(pr, source_root: Path, dest_root: Path, version: str) -> list[str]:
    """Select the payload under ``source_root`` and materialize it, flat, at ``dest_root``.

    ``source_root`` is normally the repository (``agent/tools/x.py`` ...); each file lands at its
    packaged path (``tools/x.py``). A source already laid out flat -- a seeded home, an unpacked
    ``_platform`` -- is copied as it stands. Returns the packaged relpaths, which is exactly what the
    manifest records. Split out of the command class so the guard and copy semantics are
    unit-testable without running a build.
    """
    repository = pr.is_repository_layout(source_root)
    selected = pr.payload_files(source_root)
    # Every (directory, patterns) entry must contribute at least one file: a zero-file area
    # means the build is running from a tree missing a platform directory (a crippled sdist,
    # a partial checkout), and the wheel it would produce degrades silently at runtime --
    # exactly the failure class this hook exists to kill. Fail the build instead. Membership
    # is by exact parent directory (payload paths are always one level below their area), so
    # files in a nested area -- tool_specs/env/ under tool_specs -- cannot stand in for their
    # parent's.
    for rel_dir, _patterns in pr.PLATFORM_PAYLOAD:
        packaged_dir = pr.PAYLOAD_PACKAGED_DIRS[rel_dir]
        area = rel_dir if repository else packaged_dir
        if not any(path.rsplit("/", 1)[0] == area for path in selected):
            raise RuntimeError(
                f"platform payload area {rel_dir!r} (packaged as {packaged_dir!r}) selected no files "
                f"under {source_root}; refusing to build a wheel with a partial _platform tree"
            )

    # Rebuilds reuse ``build_lib`` (pip and ``python -m build --wheel`` both build in-tree):
    # a payload dir left by an earlier build would ship every file the current selection no
    # longer carries -- ghosts the freshly written manifest disclaims. Start from nothing.
    if dest_root.exists():
        shutil.rmtree(dest_root)

    shipped = []
    for rel in selected:
        packaged = pr.packaged_relpath(rel) if repository else rel
        target = dest_root / packaged
        target.parent.mkdir(parents=True, exist_ok=True)
        # copyfile follows symlinks: the tools_user twins ship as real bytes (the
        # byte-equality hygiene pin blesses identical copies of base_mcp/worker_utils).
        shutil.copyfile(source_root / rel, target)
        shipped.append(packaged)
    shipped.sort()

    manifest = {"version": version, "files": shipped}
    dest_root.mkdir(parents=True, exist_ok=True)
    (dest_root / pr.PLATFORM_MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return shipped



class build_py_with_platform_payload(build_py):
    """Standard ``build_py``, then the ``_platform/`` payload copy (non-editable builds only)."""

    def run(self) -> None:
        super().run()
        if getattr(self, "editable_mode", False):
            return  # the checkout serves the real trees; a copy would shadow nothing and drift
        pr = _load_platform_root_by_path()
        dest_root = Path(self.build_lib) / "spatialomicsgym" / "_platform"
        _copy_platform_payload(pr, _HERE, dest_root, self.distribution.get_version())


if __name__ == "__main__":
    setup(cmdclass={"build_py": build_py_with_platform_payload})
