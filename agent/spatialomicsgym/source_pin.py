"""The two source pins a scored arm is stamped with, computed by committed code.

WHY THIS EXISTS. Every SpatialBench arm is refused unless the agent source matches a recorded
``AGENT_SRC_HASH`` and the know-how corpus matches a recorded ``KNOW_HOW_HASH``. Both definitions
lived only in the external benchmark harness (``runners/spatialbench/onbench_runner.py``) and in
prose, so ``BASELINE.md`` had to record three landed changes as having *no* source pin: nothing in
this repository could compute one. A hash nobody can recompute is worse than none -- it reads as
provenance and cannot be checked.

THE DEFINITIONS ARE THE HARNESS'S, BYTE FOR BYTE, and ``test/test_the_source_pin_is_computed_by_code
_that_ships.py`` compares the two whenever the harness is present. Three details are load-bearing
and easy to "fix" by accident:

* ``git ls-files -- 'agent/spatialomicsgym/*.py' 'agent/tools/*.py'``: a git pathspec ``*``
  **crosses** ``/``, so this folds every ``.py`` at any depth under the package and the portal tree.
  The labels folded are the repository-relative paths git prints (``agent/spatialomicsgym/x.py``).
  A checkout from before the re-layout (``spatialomicsgym/`` and ``tools/`` at the top) is read with
  the old pathspec ``'spatialomicsgym/*.py' 'tools/*.py'`` and labels, so its recorded pins can still
  be recomputed. The re-layout itself moved the pin: the portal (``webui/``, now
  ``backend/sog_portal``) left the folded tree, and every label gained ``agent/``.
* Untracked-but-not-ignored ``.py`` fold as well, because the harness reads the **working tree**:
  an uncommitted edit moves the pin exactly as a commit does.
* The know-how glob is **non-recursive**, mirroring ``KnowHowLoader``: ``know_how/resource/`` never
  reaches a prompt, so it must not move the pin either. A guard with false positives is a guard
  the operator learns to bypass.

Usage::

    python -m spatialomicsgym.source_pin            # both pins for this checkout
    python -m spatialomicsgym.source_pin --json --repo /path/to/checkout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

#: The checkout this module was imported from -- the default a caller almost always means. The
#: repository root: ``<repo>/agent/spatialomicsgym/source_pin.py`` -> ``<repo>``.
REPO = Path(__file__).resolve().parents[2]


def _agent_prefix(repo: Path) -> str:
    """``"agent/"`` for a checkout laid out with the agent part under ``agent/``; ``""`` for one from
    before the re-layout, whose ``spatialomicsgym/`` and ``tools/`` sit at the top."""
    return "agent/" if (repo / "agent" / "spatialomicsgym").is_dir() else ""


def _git_lines(repo: Path, args: list[str]) -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def agent_src_files(repo: Path | str = REPO) -> list[str]:
    """Repo-relative paths the source pin folds, sorted: tracked plus untracked-unignored ``.py``.

    The pathspec is ``agent/spatialomicsgym/*.py agent/tools/*.py`` (``spatialomicsgym/*.py tools/*.py``
    on a pre-re-layout checkout); each path is returned exactly as git prints it, relative to ``repo``.
    """
    repo = Path(repo)
    prefix = _agent_prefix(repo)
    package, tools = f"{prefix}spatialomicsgym", f"{prefix}tools"
    files = set(_git_lines(repo, ["ls-files", "--", f"{package}/*.py", f"{tools}/*.py"]))
    others = _git_lines(repo, ["ls-files", "--others", "--exclude-standard", "--", package, tools])
    files |= {f for f in others if f.endswith(".py")}
    return sorted(files)


def _fold(entries: list[tuple[str, Path]]) -> str:
    h = hashlib.sha256()
    for label, path in entries:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = "unreadable"  # a state, not an absence -- a deleted-but-tracked file still counts
        h.update(f"{digest}  {label}\n".encode())
    return h.hexdigest()


def agent_src_hash(repo: Path | str = REPO) -> str:
    """``AGENT_SRC_HASH``: sha256 over ``"<sha256>  <relpath>\\n"`` lines of :func:`agent_src_files`."""
    repo = Path(repo)
    return _fold([(rel, repo / rel) for rel in agent_src_files(repo)])


def know_how_files(repo: Path | str = REPO) -> list[Path]:
    """The documents :func:`know_how_hash` folds: ``know_how/*.md``, non-recursive, sorted.

    ``agent/spatialomicsgym/know_how`` (``spatialomicsgym/know_how`` on a pre-re-layout checkout). The
    labels are bare filenames, so the re-layout alone does not move this pin.
    """
    repo = Path(repo)
    return sorted((repo / _agent_prefix(repo) / "spatialomicsgym" / "know_how").glob("*.md"))


def know_how_hash(repo: Path | str = REPO) -> str:
    """``KNOW_HOW_HASH``: sha256 over ``"<sha256>  <filename>\\n"`` lines of :func:`know_how_files`."""
    return _fold([(p.name, p) for p in know_how_files(repo)])


def pins(repo: Path | str = REPO) -> dict[str, str]:
    repo = Path(repo)
    return {"repo": str(repo), "AGENT_SRC_HASH": agent_src_hash(repo), "KNOW_HOW_HASH": know_how_hash(repo)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=str(REPO), help="checkout to hash (default: this one)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)
    result = pins(args.repo)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"AGENT_SRC_HASH: {result['AGENT_SRC_HASH']}")
        print(f"KNOW_HOW_HASH: {result['KNOW_HOW_HASH']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
