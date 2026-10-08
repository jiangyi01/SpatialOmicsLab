#!/usr/bin/env python3
"""Find and reclaim ``user_*`` conda envs that no created tool claims any more.

Why this exists, measured on this box on 2026-09-17:

* 20 ``user_*`` envs on disk, **46 GB** between them;
* ``tools_user/install_log.json`` holds **0** records;
* ``MCP_server/mcp_config_user.yaml`` wires **0** servers.

So every one of those envs is an orphan by the registry's own reckoning -- nothing can invoke
them -- and they are the leftovers of the finished 21-tool self-extension experiment.

They are not merely wasted disk. The creation pre-flight in
``know_how/add_new_mcp_tool.md`` is::

    check(
        "user_env_cap",
        len(user_envs) < default_config.max_user_envs,
        f"{len(user_envs)} / {default_config.max_user_envs} user envs",
    )

With ``max_user_envs = 20`` that is ``20 < 20`` -- False -- so **every new tool creation on this
box is refused**, by dead envs belonging to tools that no longer exist. Reclaiming them is what
makes tool creation work again, which is why the reclaim is part of the creation story rather
than a housekeeping script.

Read-only by default: it prints what it found and exits. ``--remove`` is the only thing that
deletes, it names every env before it acts, and it refuses to touch an env that any record or
config entry still claims.

Stdlib only, so it runs in the agent-core env and in a bare interpreter alike.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INSTALL_LOG = REPO / "tools_user" / "install_log.json"
USER_CONFIG = REPO / "MCP_server" / "mcp_config_user.yaml"
PREFIX = "user_"


def _user_config() -> Path:
    """``USER_CONFIG`` unless ``SOG_MCP_USER_CONFIG`` moves it -- the resolver every reader uses."""
    try:
        from spatialomicsgym.mcp_user_config import user_config_override

        moved = user_config_override()
    except Exception:
        moved = None
    return Path(moved) if moved else USER_CONFIG


def _envs_root() -> Path:
    """Where conda keeps its envs. ``CONDA_ENVS_PATH`` wins, else beside this interpreter."""
    named = os.environ.get("CONDA_ENVS_PATH") or os.environ.get("CONDA_ENV_PATH")
    if named:
        return Path(named.split(os.pathsep)[0])
    base = os.environ.get("CONDA_PREFIX") or sys.prefix
    here = Path(base)
    # sys.prefix inside an env is <root>/envs/<name>; one level up is the envs dir.
    return here.parent if here.parent.name == "envs" else here / "envs"


def user_envs() -> list[Path]:
    root = _envs_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(PREFIX))


def claimed() -> set[str]:
    """Tool ids any record or config entry still claims, so they are never reclaimed.

    Both sources are read defensively and a *failure to read is treated as a claim*: if the
    install log is corrupt we must not conclude that nothing is claimed and delete everything.
    """
    ids: set[str] = set()
    unreadable = False

    try:
        entries = json.loads(INSTALL_LOG.read_text(encoding="utf-8")) if INSTALL_LOG.is_file() else []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and entry.get("tool_id"):
                ids.add(str(entry["tool_id"]))
    except Exception:
        unreadable = True

    # The user config is YAML, and this module is stdlib-only, so the server names are read as
    # text rather than parsed. Over-reading here is safe: a false claim keeps an env, and keeping
    # an env is the recoverable mistake.
    try:
        # The file the agent really wires, SOG_MCP_USER_CONFIG included -- the constant above ignored
        # the override, so with it set an env the active config referenced looked unreferenced and
        # was offered for reclaim (hunt 2026-09-30, u14-mcp-wiring-7).
        user_config = _user_config()
        if user_config.is_file():
            for line in user_config.read_text(encoding="utf-8").splitlines():
                bare = line.strip()
                if bare.startswith(PREFIX) and bare.endswith(":"):
                    ids.add(bare[len(PREFIX) : -1])
    except Exception:
        unreadable = True

    if unreadable:
        print("! could not read a registry source; every env is treated as claimed", file=sys.stderr)
        ids.update(p.name[len(PREFIX) :] for p in user_envs())
    return ids


def _size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                stat = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            if not os.path.islink(os.path.join(root, name)):
                total += stat.st_size
    return total


def human(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def survey() -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """``(orphans, kept)`` -- each a list of ``(path, bytes)``."""
    live = claimed()
    orphans: list[tuple[Path, int]] = []
    kept: list[tuple[Path, int]] = []
    for env in user_envs():
        row = (env, _size_bytes(env))
        (kept if env.name[len(PREFIX) :] in live else orphans).append(row)
    return orphans, kept


def remove(env: Path) -> str:
    """Remove one env, preferring conda so its own bookkeeping stays consistent."""
    conda = shutil.which("conda") or shutil.which("micromamba")
    if conda:
        done = subprocess.run(
            [conda, "env", "remove", "-y", "-p", str(env)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if done.returncode == 0 and not env.exists():
            return "removed"
        # Fall through: conda can report success while leaving the directory, and can fail on an
        # env it never created. Either way the directory is the thing that holds the disk.
    try:
        shutil.rmtree(env, ignore_errors=False)
    except OSError as exc:
        return f"FAILED: {exc}"
    return "removed" if not env.exists() else "FAILED: still present"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remove", action="store_true", help="actually delete the orphaned envs")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt (for scripts)")
    args = parser.parse_args(argv)

    orphans, kept = survey()
    if kept:
        print(f"{len(kept)} env{'' if len(kept) == 1 else 's'} still claimed by a tool, left alone:")
        for env, size in kept:
            print(f"    {env.name:32s} {human(size):>10s}")
    if not orphans:
        print("No orphaned user tool envs.")
        return 0

    total = sum(size for _e, size in orphans)
    print(f"\n{len(orphans)} orphaned env{'' if len(orphans) == 1 else 's'}, {human(total)}:")
    for env, size in orphans:
        print(f"    {env.name:32s} {human(size):>10s}")
    print("\nOrphaned means: no record in install_log.json and no server in mcp_config_user.yaml,")
    print("so nothing in the portal or the agent can invoke them.")

    if not args.remove:
        print("\nNothing was deleted. Re-run with --remove to reclaim the space.")
        return 0

    if not args.yes:
        print(f"\nAbout to delete {len(orphans)} conda envs and free {human(total)}. This cannot be undone.")
        try:
            if input("Type 'reclaim' to continue: ").strip() != "reclaim":
                print("Nothing was deleted.")
                return 1
        except (EOFError, KeyboardInterrupt):
            print("\nNothing was deleted.")
            return 1

    freed = 0
    for env, size in orphans:
        outcome = remove(env)
        print(f"    {env.name:32s} {outcome}")
        if outcome == "removed":
            freed += size
    print(f"\nFreed {human(freed)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
