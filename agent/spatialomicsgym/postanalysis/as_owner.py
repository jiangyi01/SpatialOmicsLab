"""Run a post-analysis step as the account that can write the directory, when this process is root.

The portal runs as root and the agent's code runs as the unprivileged ``sog-agent`` uid
(:mod:`sog_portal.boundary`). Every run's output directory is therefore writable by that
uid, and anything it wrote there -- including a symlink at a name the engine or the report is about
to write -- was followed when root wrote into it. A path-based writer cannot be made safe over a
directory someone else can change: a check before the write loses the race, and a no-follow open of
the last component still follows a swapped directory above it (hunt 2026-09-30, u11-stcoscientist-1:
a manifest planted by the worker became root's write over ``admins.json``). So the L1 engine and the
L3 report run in a child process as that uid for such directories: whatever it follows, it could
already write.

L2's two writers -- ``review.write_review`` and ``review.register_new_artifacts`` -- run the same way.
They read the manifest before rewriting it, so as root they followed a ``manifest.json`` the agent had
swapped for a link and published the file it pointed at, a provider key store included, as the run's
manifest (hunt 2026-09-30, u18-postanalysis-1). So does L2's judging, ``review.review_manifest``: it
reads every table the manifest's ``source_outputs`` names, the agent can write that manifest, and as
root it read another account's prediction table and quoted its labels into this account's review
(hunt 2026-09-30, rp-u18).

Not root, or a directory only root can write: :func:`foreign_writer` answers ``None`` and callers keep
the in-process path, which is every CLI, notebook and benchmark run and every test.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

#: Wall-clock bound on one engine or report run in the child. The in-process path has none; a child
#: that hangs would otherwise hold the finished turn open forever.
CHILD_TIMEOUT_SECONDS = 1800


def _child_pythonpath() -> str:
    """The child's ``PYTHONPATH``: every directory holding one of this project's packages.

    In a checkout that is two -- ``agent/`` (``spatialomicsgym``) and ``install/`` (``sog_install``) -- since the re-layout put
    them in separate parts; the repository root alone imports none of them. An installed copy has one,
    the directory holding the package.
    """
    from spatialomicsgym import layout

    root = layout.repo_root()
    if root is None:
        return str(Path(__file__).resolve().parents[2])
    return os.pathsep.join(str(root / part) for part in ("agent", "install"))


def _agent_uid() -> int | None:
    try:
        import pwd

        from sog_portal.boundary import AGENT_USER

        return pwd.getpwnam(AGENT_USER).pw_uid
    except (ImportError, KeyError):
        return None


def foreign_writer(directory: Path | str) -> tuple[int, int] | None:
    """``(uid, gid)`` to write ``directory`` as, or ``None`` to write it in this process.

    ``None`` unless this process is root and someone else can write the directory: it belongs to
    another uid (run as that uid), or it is root's but group- or world-writable -- the portal's
    ``outputs/<account>/<chat>`` folders are ``root:sog-u-<account>`` 2770 -- in which case the child
    runs as the agent's uid with that group. A directory that is itself a symlink is refused (``None``
    would write through it), so it is run as the agent too when a boundary exists.
    """
    if not _is_root():
        return None
    try:
        info = os.lstat(directory)
    except OSError:
        return None
    return _writer_for(info)


def foreign_writer_of_fd(fd: int) -> tuple[int, int] | None:
    """:func:`foreign_writer` for a directory already open as ``fd``: the inode, not the path.

    A path names whatever is there when it is looked up, so a directory judged by path and then written
    by path can be two directories when someone else can rename its parent. Asking the descriptor the
    write will go through makes the judgement and the write about one directory (hunt 2026-09-30,
    rp-u18).
    """
    if not _is_root():
        return None
    try:
        info = os.fstat(fd)
    except OSError:
        return None
    return _writer_for(info)


def _is_root() -> bool:
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def _writer_for(info: os.stat_result) -> tuple[int, int] | None:
    if stat.S_ISLNK(info.st_mode):
        agent = _agent_uid()
        return (agent, info.st_gid) if agent is not None else None
    if info.st_uid != 0:
        return info.st_uid, info.st_gid
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        agent = _agent_uid()
        return (agent, info.st_gid) if agent is not None else None
    return None


def run_module_as(
    uid: int, gid: int, module: str, args: list[str], *, stdin_text: str | None = None
) -> subprocess.CompletedProcess:
    """``python -m <module> <args>`` as ``uid``/``gid``, with no provider keys in its environment.

    The environment is built from nothing, not inherited: the portal's holds every provider key, and
    the child reads only files. ``HOME`` and ``MPLCONFIGDIR`` point at a scratch directory the child
    owns, removed afterwards. ``stdin_text`` is handed to the child on its standard input -- how L2
    passes the verdict it computed to the child that writes it.
    """
    scratch = tempfile.mkdtemp(prefix="sog-postanalysis-")
    try:
        os.chown(scratch, uid, gid)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": scratch,
            "MPLCONFIGDIR": scratch,
            "MPLBACKEND": "Agg",
            "PYTHONPATH": _child_pythonpath(),
            "PYTHONNOUSERSITE": "1",
            "SOG_SKIP_DOTENV": "1",
            "LANG": os.environ.get("LANG") or "C.UTF-8",
            "LC_ALL": os.environ.get("LC_ALL") or "C.UTF-8",
        }
        return subprocess.run(
            [sys.executable, "-m", module, *args],
            user=uid,
            group=gid,
            extra_groups=[],
            env=env,
            cwd=scratch,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_SECONDS,
            check=False,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
