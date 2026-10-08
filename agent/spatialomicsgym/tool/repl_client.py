"""The server's side of the REPL worker: spawn it, talk to it, kill it when it must be killed.

One ``ReplProcess`` at a time, bound to an account, restarted when the account changes and after
a timeout kill -- never per cell, because "variables defined in one execution will be available
in subsequent executions" is the REPL's documented contract and the thing a researcher relies on
across a session. The account boundary is the one place that promise does not hold, and here it
is a fresh process rather than a cleared dict: a fresh process also drops imported modules,
monkeypatches and anything a cell did to the interpreter.

What this buys over the in-process REPL, whoever the worker runs as:

* a timeout is a real kill. ``run_with_timeout`` raises an async exception into a thread, which
  cannot stop a C call; a SIGKILL on the worker's process group stops the cell AND the tool
  portals it spawned, and the next cell is told in plain words that the session restarted.
* a crash is a crash of the worker, not of the portal.

And when the worker runs as another user (``sog_portal/boundary.py`` builds that plan), model-written
code can no longer read the account store, the operator list, the provider keys or the other
accounts' files. ``boundary()`` reports which of those this process actually has, from the
worker's own ``hello`` -- ``"uid"`` when its uid differs from ours, ``"process-only"`` otherwise --
and never from what was asked for.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from spatialomicsgym.tool.repl_protocol import PARTIAL_INTERVAL, FrameTimeout, read_frame, write_frame

#: The per-turn environment the server forwards to the worker before every cell. Also what the
#: MCP overlay forwards into each tool portal (``server._WORKER_ENV_FORWARDS`` aliases this).
FORWARDED_ENV: tuple[str, ...] = ("SOG_WORK_DIR", "SOG_SPATIAL_LIBRARY_REGISTRY")

#: Appended to ``utils.execution.timeout_message`` when the kill was real.
TIMEOUT_RESTART_NOTE = (
    " The Python session that ran it was stopped and restarted: every variable bound by earlier "
    "cells is gone -- reload what you need from the files you already wrote."
)

#: The observation when the worker died under a cell (a segfault, an OOM kill, ``os._exit``).
DIED_NOTE = (
    "Error in execution: the Python session died (exit {rc}) and was restarted with an empty "
    "namespace -- every variable bound by earlier cells is gone. Reload what you need from the "
    "files you already wrote, and if the cell that died loaded something very large, load less of "
    "it at a time rather than shrinking the data."
)

#: How long the worker may take to say hello (it imports the agent package and matplotlib) --
#: and how long a ``configure`` may take, since a configure is what a fresh spawn replays.
HELLO_TIMEOUT = 120.0
#: How long the server waits for the worker to confirm it is idle after a cell's reply. A reply
#: is a frame on a pipe the cell can write to; the idle check is the second half of trusting it.
IDLE_CHECK_TIMEOUT = 5.0
#: The most stderr the server keeps per worker lifetime; past it the drain counts and drops.
MAX_STDERR_BYTES = 8 * 1024 * 1024
#: How often a request waiting on a server-side upcall checks that the worker it would answer is
#: still alive. A killed worker can never receive the answer, so the wait ends there.
UPCALL_POLL_SECONDS = 0.25
#: Names that never cross into a same-user worker either: the CLI's worker does not call the model.
_SECRET_NAME_RE = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSW(OR)?D|PASSWD|CREDENTIAL|ACCESS_CODE|AUTH)", re.IGNORECASE)

_LOG = logging.getLogger("spatialomicsgym.repl_host")

#: Whether an upcall is being answered, and when the last one started or ended (monotonic). The
#: portal's stall watchdog reads this: a cell that is waiting on the SERVER -- a brokered env build
#: that takes minutes -- produces no graph chunk, and the watchdog killed the turn as "stalled"
#: although the cell's own deadline had correctly been moved past that time (u12-react-11).
_UPCALLS = {"running": 0, "last": 0.0}
_UPCALLS_LOCK = threading.Lock()


def last_upcall_activity() -> float:
    """``time.monotonic()`` now if an upcall is being answered, else when the worker was last heard from -- an upcall
    ending or a running cell's progress report (0.0: never)."""
    with _UPCALLS_LOCK:
        return time.monotonic() if _UPCALLS["running"] else float(_UPCALLS["last"])


def _activity_mark() -> None:
    """Record that the worker just reported progress on a running cell."""
    with _UPCALLS_LOCK:
        _UPCALLS["last"] = time.monotonic()


def _upcall_mark(delta: int) -> None:
    with _UPCALLS_LOCK:
        _UPCALLS["running"] = max(0, int(_UPCALLS["running"]) + delta)
        _UPCALLS["last"] = time.monotonic()


@dataclass
class SpawnPlan:
    """Everything a spawn needs. ``preexec`` runs in the child before exec (limits, uid drop)."""

    argv: list[str]
    env: dict[str, str]
    cwd: str
    preexec: Callable[[], None] | None = None
    account: str = ""
    expects_uid_drop: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _child_pythonpath(inherited: str = "") -> str:
    """The worker's ``PYTHONPATH``: ``agent/`` and ``install/`` of a checkout, then whatever was inherited.

    The worker runs ``-m spatialomicsgym.tool.repl_host`` from the caller's cwd. Since the re-layout the
    repository root no longer holds ``tools_user`` (the declarative prompt tools) or the other packages;
    they sit under ``agent/`` and ``install/``. An installed copy adds nothing and keeps the inherited value.
    """
    from spatialomicsgym import layout

    root = layout.repo_root()
    parts = [str(root / part) for part in ("agent", "install")] if root is not None else []
    parts += [p for p in inherited.split(os.pathsep) if p and p not in parts]
    return os.pathsep.join(parts)


def same_user_plan(account: str = "") -> SpawnPlan:
    """A worker as THIS user: crash isolation and a real kill, no privilege boundary.

    What the CLI gets under ``SOG_REPL_ISOLATION=process``, and what the portal falls back to
    when the box has no ``sog-agent`` (and says so). The environment is this process's minus the
    run journal, exactly as ``ingest._child_env`` builds a tool worker's.
    """
    env = {k: v for k, v in os.environ.items() if not _SECRET_NAME_RE.search(k) and k != "SOG_WEB_HISTORY"}
    env["PYTHONUNBUFFERED"] = "1"
    pythonpath = _child_pythonpath(env.get("PYTHONPATH", ""))
    if pythonpath:
        env["PYTHONPATH"] = pythonpath
    return SpawnPlan(
        argv=[sys.executable, "-m", "spatialomicsgym.tool.repl_host"],
        env=env,
        cwd=os.getcwd(),
        preexec=None,
        account=account,
        expects_uid_drop=False,
    )


def _proc_identity(pid: int) -> tuple[int, int, frozenset[int]] | None:
    """``(real uid, real gid, supplementary groups)`` of ``pid`` from ``/proc``, or None."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as fh:
            fields = dict(line.split(":", 1) for line in fh if ":" in line)
        return (
            int(fields["Uid"].split()[0]),
            int(fields["Gid"].split()[0]),
            frozenset(int(g) for g in fields.get("Groups", "").split()),
        )
    except (OSError, KeyError, ValueError, IndexError):
        return None


def _pids() -> list[int]:
    try:
        return [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []


def _matches(pid: int, uid: int, groups: frozenset[int]) -> bool:
    identity = _proc_identity(pid)
    return identity is not None and identity[0] == uid and bool(identity[2] & groups) and pid != os.getpid()


class _RequestLock:
    """``ReplProcess._lock``: a re-entrant lock that refuses, by name, the one wait that cannot end.

    The request thread holds it for the whole cell and answers the worker's upcalls on a side thread
    (:meth:`ReplProcess._run_upcall`). An upcall that calls back into the same ``ReplProcess`` --
    ``set_upcalls``, ``configure``, ``exec``, ``summary`` -- would wait here for a thread that is
    itself waiting for that upcall: a deadlock until the worker dies. Answered inline, as before, the
    RLock was re-entrant and the call worked (hunt 2026-09-30, uL2-concurrency-2). A non-blocking try
    (``kill``) is still allowed: it cannot wait.
    """

    def __init__(self) -> None:
        self._rlock = threading.RLock()
        self._local = threading.local()

    def mark_answering(self, on: bool) -> None:
        """Called on an upcall's own thread, around the server-side call."""
        self._local.answering = on

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if blocking and getattr(self._local, "answering", False):
            raise RuntimeError(
                "an upcall cannot call back into the REPL worker that is waiting for its answer; "
                "that worker's lock is held until the upcall returns"
            )
        return self._rlock.acquire(blocking, timeout)

    def release(self) -> None:
        self._rlock.release()

    def __enter__(self) -> _RequestLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class ReplProcess:
    """One worker, lazily spawned, replayed its ``configure`` on every spawn."""

    def __init__(
        self,
        plan_for: Callable[[str], SpawnPlan] = same_user_plan,
        *,
        upcalls: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        self._plan_for = plan_for
        self._upcalls: dict[str, Callable[..., Any]] = dict(upcalls or {})
        self._lock = _RequestLock()
        #: Guards only ``_proc`` itself, so ``kill()`` can signal from another thread while a
        #: cell holds ``_lock`` (the audit's S2: Stop could not end a running cell).
        self._proc_lock = threading.Lock()
        self._stderr_bytes = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._in_fd: int | None = None  # we read replies here
        self._out_fd: int | None = None  # we write requests here
        self._hello: dict[str, Any] = {}
        self._configured: dict[str, Any] | None = None
        self._seq = 0
        self.account: str = ""
        self.last_spawn_error: str = ""
        #: The dropped uid and the ACCOUNT's own supplementary groups the current worker runs with,
        #: read from the worker itself at spawn. What an account switch reaps by -- see
        #: ``_reap_previous_account``.
        self._worker_identity: tuple[int, frozenset[int]] | None = None

    # -- state ------------------------------------------------------------------------------- #
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def uid(self) -> int | None:
        value = self._hello.get("uid")
        return int(value) if isinstance(value, int) else None

    @property
    def hello(self) -> dict[str, Any]:
        return dict(self._hello)

    def boundary(self) -> str:
        """``"uid"`` when the worker runs as another user, else ``"process-only"``.

        Read from the worker's own report, never from the plan: a plan that asked for a drop the
        child could not perform must not be reported as a boundary.
        """
        if not self.alive or self.uid is None:
            return "process-only"
        return "uid" if self.uid != os.getuid() else "process-only"

    def set_upcalls(self, upcalls: dict[str, Callable[..., Any]]) -> None:
        with self._lock:
            self._upcalls = dict(upcalls)

    # -- lifecycle ----------------------------------------------------------------------------- #
    def set_account(self, account: str) -> None:
        """Bind the worker to an account; a different one gets a fresh process."""
        who = str(account or "")
        with self._lock:
            if who != self.account:
                if self.alive:
                    self.kill()
                self._reap_previous_account()
                # And forget the configuration: it is the previous account's -- its owner, its
                # declarative tools and their templates, its SOG_WORK_DIR and library registry --
                # and every later spawn replays it. Kept, the eager spawn for the next account started
                # that account's worker configured as the previous one until its first python cell
                # reconfigured it (u12-react-extra-19). The next python cell configures afresh.
                self._configured = None
            self.account = who

    def ensure(self) -> None:
        with self._lock:
            if not self.alive:
                self._spawn()

    def configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the tool set now (if alive) and remember it for every later spawn."""
        with self._lock:
            self._configured = dict(payload)
            if not self.alive:
                self._spawn()  # replays it
                return {"replayed": True}
            try:
                return self._request("configure", self._configured, deadline=time.monotonic() + HELLO_TIMEOUT)
            except FrameTimeout:
                # A worker that does not answer a configure is wedged (a cell that never returned,
                # a forged frame): a fresh process, and the configure replays on it.
                self.kill()
                self._spawn()
                return {"replayed": True, "restarted": True}

    def exec(
        self,
        code: str,
        timeout: float,
        *,
        env_failures: dict[str, str] | None = None,
        general_env_calls: int = 0,
        env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run one cell. Always returns a result dict; the failure modes are in ``output``."""
        env_failures = dict(env_failures or {})
        base = {"figures": [], "env_failures": env_failures, "general_env_calls": general_env_calls}
        with self._lock:
            try:
                self.ensure()
            except Exception as exc:
                return {**base, "output": f"Error in execution: the Python session could not be started ({exc})"}
            from spatialomicsgym.utils.execution import partial_room, timeout_observation

            budget = max(1.0, float(timeout))
            framing = {"reported_every": PARTIAL_INTERVAL, "after": TIMEOUT_RESTART_NOTE}
            payload = {
                "code": code,
                "env_failures": env_failures,
                "general_env_calls": general_env_calls,
                "env": dict(env or {}),
                # What the timeout observation below has room for, so the worker's reports fit it.
                "partial_room": partial_room(timeout, **framing),
            }
            # The worker's latest report of what the cell has printed (``repl_host._PartialReporter``).
            # At the deadline the process is killed with the cell's output in it, so this is the
            # only copy of what it had got to.
            partial: dict[str, str] = {}

            def _note(frame: dict[str, Any]) -> None:
                partial["text"] = str(frame.get("partial") or "")

            try:
                reply = self._request("exec", payload, deadline=time.monotonic() + budget, on_partial=_note)
                # A reply is a frame on a pipe the cell can write to. Before it is trusted, the
                # worker must answer a fresh nonce'd ping -- something a cell that is still running
                # (and only forged its reply) cannot do, because the request loop is what answers.
                self._request("ping", {}, deadline=time.monotonic() + IDLE_CHECK_TIMEOUT)
            except FrameTimeout:
                self.kill()
                output = timeout_observation(partial.get("text", ""), timeout, **framing)
                return {**base, "output": output, "timed_out": True}
            except _WorkerGone as gone:
                return {**base, "output": DIED_NOTE.format(rc=gone.rc), "died": True}
            if not reply.get("ok", False):
                return {**base, "output": f"Error in execution: {reply.get('error') or 'the worker refused the cell'}"}
            return {
                "output": str(reply.get("output") or ""),
                "figures": list(reply.get("figures") or []),
                "env_failures": dict(reply.get("env_failures") or env_failures),
                "general_env_calls": int(reply.get("general_env_calls") or general_env_calls),
                "namespace_summary": str(reply.get("namespace_summary") or ""),
            }

    def exec_shell(self, kind: str, code: str, timeout: float, *, env: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a ``#!BASH`` or ``#!R`` cell in the worker. Same budget and kill path as ``exec``.

        Kept beside ``exec`` rather than folded into it: the two carry different payloads and only
        the python one has a namespace, figures or an env-failure ledger to return. What they share
        -- the lock, the deadline, the nonce'd idle ping before a reply is trusted, and the SIGKILL
        of the process group on timeout -- is what matters, and is shared.
        """
        base: dict[str, Any] = {"figures": [], "output": ""}
        with self._lock:
            try:
                self.ensure()
            except Exception as exc:
                return {**base, "output": f"Error in execution: the worker could not be started ({exc})"}
            budget = max(1.0, float(timeout))
            # The per-turn env (``FORWARDED_ENV``) rides with the cell, as it does on ``exec``. Only
            # python cells carried it, so a ``#!BASH``/``#!R`` cell saw the previous turn's -- or the
            # previous ACCOUNT's -- SOG_WORK_DIR and library registry (u12-react-7).
            payload = {"kind": str(kind), "code": code, "timeout": budget, "env": dict(env or {})}
            try:
                reply = self._request("exec_shell", payload, deadline=time.monotonic() + budget)
                self._request("ping", {}, deadline=time.monotonic() + IDLE_CHECK_TIMEOUT)
            except FrameTimeout:
                self.kill()
                from spatialomicsgym.utils.execution import timeout_message

                return {**base, "output": timeout_message(timeout) + TIMEOUT_RESTART_NOTE, "timed_out": True}
            except _WorkerGone as gone:
                return {**base, "output": DIED_NOTE.format(rc=gone.rc), "died": True}
            if not reply.get("ok", False):
                return {**base, "output": f"Error in execution: {reply.get('error') or 'the worker refused the cell'}"}
            return {**base, "output": str(reply.get("output") or "")}

    def summary(self, limit: int = 20) -> str:
        with self._lock:
            if not self.alive:
                return ""
            try:
                return str(
                    self._request("summary", {"limit": limit}, deadline=time.monotonic() + 30).get("summary") or ""
                )
            except Exception:
                return ""

    def forget(self, name: str) -> bool:
        with self._lock:
            if not self.alive:
                return False
            try:
                return bool(self._request("forget", {"name": name}, deadline=time.monotonic() + 30).get("found"))
            except Exception:
                return False

    def set_env(self, mapping: dict[str, Any]) -> None:
        with self._lock:
            if not self.alive:
                return
            try:
                self._request("set_env", {"env": dict(mapping)}, deadline=time.monotonic() + 30)
            except Exception:
                pass

    def reset(self) -> None:
        """Forget everything: kill and respawn lazily. A cleared dict would keep the modules."""
        with self._lock:
            self.kill()

    def kill(self) -> None:
        """SIGKILL the worker's process group NOW, from any thread.

        The signal goes out under ``_proc_lock`` only: a Stop or a stall arrives on another thread
        while a cell holds ``_lock`` inside ``exec``, and a kill that waited for that lock waited
        for the cell to finish (the audit's S2 -- measured at 24 s of a 25 s cell). The descriptor
        cleanup needs ``_lock``; it happens here when nobody holds it, and otherwise on the exec
        thread, which reads EOF from the dead worker and reaps it.
        """
        with self._proc_lock:
            proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._lock.acquire(blocking=False):
            try:
                self._cleanup(proc)
            finally:
                self._lock.release()

    def _cleanup(self, proc: subprocess.Popen[bytes] | None) -> None:
        """Reap and forget ``proc`` if it is still the worker. Caller holds ``_lock``."""
        with self._proc_lock:
            if proc is None or self._proc is not proc:
                return
            try:
                proc.wait(timeout=10)
            except Exception:
                pass
            self._close_fds()
            self._proc = None
            self._hello = {}

    def shutdown(self) -> None:
        with self._lock:
            if self.alive:
                try:
                    self._request("shutdown", {}, deadline=time.monotonic() + 5)
                    self._proc.wait(timeout=5)  # type: ignore[union-attr]
                except Exception:
                    pass
            self.kill()

    # -- the wire ------------------------------------------------------------------------------ #
    def _spawn(self) -> None:
        plan = self._plan_for(self.account)
        self.last_spawn_error = ""
        try:
            proc = subprocess.Popen(
                list(plan.argv),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=plan.cwd or None,
                env=dict(plan.env),
                preexec_fn=plan.preexec,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as exc:
            self.last_spawn_error = f"{type(exc).__name__}: {exc}"
            raise
        with self._proc_lock:
            self._proc = proc
            self._stderr_bytes = 0
        self._out_fd = proc.stdin.fileno()  # type: ignore[union-attr]
        self._in_fd = proc.stdout.fileno()  # type: ignore[union-attr]
        threading.Thread(target=self._drain_stderr, args=(proc,), daemon=True, name="repl-host-stderr").start()
        try:
            first = read_frame(self._in_fd, time.monotonic() + HELLO_TIMEOUT)
        except FrameTimeout:
            self.kill()
            self.last_spawn_error = "the worker did not say hello in time"
            raise RuntimeError(self.last_spawn_error) from None
        if first is None or "hello" not in first:
            rc = proc.poll()
            self.kill()
            self.last_spawn_error = f"the worker exited before saying hello (exit {rc})"
            raise RuntimeError(self.last_spawn_error)
        self._hello = dict(first["hello"])
        if plan.expects_uid_drop and self.uid == os.getuid():
            _LOG.warning("repl worker expected a uid drop and did not get one (uid %s)", self.uid)
        identity = _proc_identity(proc.pid)
        if identity is not None and identity[0] not in (0, os.getuid()):
            self._worker_identity = (identity[0], identity[2] - {identity[1]})
        if self._configured is not None:
            self._request("configure", self._configured, deadline=time.monotonic() + HELLO_TIMEOUT)

    def _reap_previous_account(self) -> None:
        """Kill every process the previous account's worker left behind, before the next one starts.

        Every account's worker runs as the one ``sog-agent`` uid, and accounts differ only by a
        supplementary group -- which the kernel's ptrace check does not compare. ``kill`` signals
        the worker's process group, so a child a cell detached into its own session (``setsid``,
        a daemonised tool) survived the switch, and with ``ptrace_scope`` 0 it could attach to the
        next account's worker or read its ``/proc/<pid>/environ`` (hunt 2026-09-30,
        u17-cli-report-extra-24).

        Matched by uid AND the previous account's own group, both read from that worker at spawn:
        another portal or a test on this box runs ``sog-agent`` workers for other accounts, and
        those are not this process's to kill. A same-user worker (the CLI) has no identity and is
        never reaped by this. Refuses to go on if leftovers keep reappearing, rather than start the
        next account's worker beside them.
        """
        identity, self._worker_identity = self._worker_identity, None
        if identity is None or not identity[1]:
            return
        uid, groups = identity
        for _ in range(20):
            victims = [pid for pid in _pids() if _matches(pid, uid, groups)]
            if not victims:
                return
            for pid in victims:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            time.sleep(0.05)
        self.last_spawn_error = "processes left by the previous account could not be cleared"
        raise RuntimeError(self.last_spawn_error)

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        """Log the worker's stderr, bounded: a cell that writes gigabytes without a newline must
        not become one unbounded ``bytes`` object in the SERVER (the audit's S6)."""
        stream = proc.stderr
        if stream is None:
            return
        truncated = False
        try:
            for raw in iter(lambda: stream.readline(65536), b""):
                self._stderr_bytes += len(raw)
                if self._stderr_bytes > MAX_STDERR_BYTES:
                    if not truncated:
                        _LOG.warning("repl worker stderr truncated past %d bytes", MAX_STDERR_BYTES)
                        truncated = True
                    continue
                _LOG.info("%s", raw.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _request(
        self,
        op: str,
        payload: dict[str, Any],
        *,
        deadline: float | None,
        on_partial: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """One request, its reply -- servicing any upcalls the worker makes in between.

        Time spent answering an upcall is the SERVER's, not the cell's: a brokered environment
        build takes minutes, and a deadline that kept ticking through it reported a successful
        build as a timed-out cell with a restarted namespace (the audit's F8). The deadline moves
        forward by exactly the time each upcall took.

        A progress report for THIS request (``{"id": rid, "partial": ...}``) goes to ``on_partial``
        and does not move the deadline: it is the cell talking, not the server working.
        """
        if self._in_fd is None or self._out_fd is None:
            raise _WorkerGone(None)
        self._seq += 1
        rid = self._seq
        nonce = secrets.token_hex(16)
        try:
            write_frame(self._out_fd, {"id": rid, "nonce": nonce, "op": op, "payload": payload})
        except (BrokenPipeError, OSError):
            raise _WorkerGone(self._reap()) from None
        while True:
            frame = read_frame(self._in_fd, deadline)  # FrameTimeout propagates to the caller
            if frame is None:
                raise _WorkerGone(self._reap())
            if "upcall" in frame:
                started = time.monotonic()
                _upcall_mark(+1)
                try:
                    self._answer_upcall(frame)
                finally:
                    _upcall_mark(-1)
                if deadline is not None:
                    deadline += time.monotonic() - started
                continue
            if "partial" in frame and frame.get("id") == rid:
                # A cell that is reporting output is alive, whatever it has not yet returned: the
                # portal's stall watchdog reads this (``last_upcall_activity``) and must not give up
                # on a run that is visibly working -- it did, and the page's clock froze while the
                # turn ran on (2026-10-04, U7).
                _activity_mark()
                if on_partial is not None:
                    on_partial(frame)
                continue
            if frame.get("id") == rid and frame.get("nonce") == nonce:
                return frame
            # Not ours: a stale answer, or a frame a cell wrote onto the protocol descriptor. Either
            # way it is not the reply to THIS request, and only the reply carries the nonce.
            _LOG.warning("repl worker sent a frame that is not the reply: %s", {k: frame[k] for k in list(frame)[:3]})

    def _answer_upcall(self, frame: dict[str, Any]) -> None:
        name = str(frame.get("upcall") or "")
        args = frame.get("args") or {}
        fn = self._upcalls.get(name)
        if fn is None:
            result: Any = {"status": "error", "message": f"{name} is not available from this session"}
        else:
            result = self._run_upcall(name, fn, args)
        if self._out_fd is None:
            return
        try:
            write_frame(self._out_fd, {"id": frame.get("id"), "upcall_result": result})
        except Exception as exc:
            # The answer must reach the worker or the cell waits forever (the audit's S1); a
            # result too large for the wire becomes an error result, and a dead pipe is the
            # request loop's to notice on its next read.
            try:
                write_frame(
                    self._out_fd,
                    {
                        "id": frame.get("id"),
                        "upcall_result": {"status": "error", "message": f"{name} could not be answered: {exc}"},
                    },
                )
            except Exception:
                pass

    def _run_upcall(self, name: str, fn: Callable[..., Any], args: Any) -> Any:
        """``fn``'s answer to the worker's upcall, computed on a side thread. Caller holds ``_lock``.

        Answered inline, a Stop or a stall kill could not end the request: ``kill`` signals the
        worker, but this thread stayed inside ``fn`` -- a brokered env build, up to
        ``SOG_BROKER_TIMEOUT`` -- holding ``_lock``, and the next turn's ``set_account`` blocked on
        it inside the portal's turn lock, with no frame sent and no watchdog running (hunt
        2026-09-30, uL2-concurrency-2). The wait now ends when the worker dies: an answer has
        nowhere to go then, and the abandoned call finishes on its own thread, unread.
        """
        box: dict[str, Any] = {}
        done = threading.Event()
        context = contextvars.copy_context()

        def _call() -> None:
            self._lock.mark_answering(True)  # a call back into this process now refuses, not hangs
            try:
                box["result"] = context.run(fn, **args) if isinstance(args, dict) else context.run(fn, args)
            except Exception as exc:
                box["result"] = {"status": "error", "message": f"{name} failed: {type(exc).__name__}: {exc}"}
            finally:
                self._lock.mark_answering(False)
                done.set()

        with self._proc_lock:
            proc = self._proc
        threading.Thread(target=_call, daemon=True, name=f"repl-upcall-{name}").start()
        while not done.wait(UPCALL_POLL_SECONDS):
            if proc is None or proc.poll() is not None:
                _LOG.warning("repl worker died while the server answered %s; the answer is abandoned", name)
                raise _WorkerGone(self._reap())
        return box.get("result", {"status": "error", "message": f"{name} failed without an answer"})

    def _reap(self) -> int | None:
        """The worker is gone (EOF): its exit status, and the descriptors closed. Caller holds ``_lock``."""
        with self._proc_lock:
            proc = self._proc
        rc = None
        if proc is not None:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            try:
                rc = proc.wait(timeout=5)
            except Exception:
                rc = proc.poll()
        self._cleanup(proc)
        return rc

    def _close_fds(self) -> None:
        proc = self._proc  # caller holds _proc_lock
        # Not stderr: ``_drain_stderr`` owns it and closes it when it ends. Closing a buffered reader
        # waits for the thread parked in its ``readline``, and that read never ends while anything
        # still holds the pipe -- a child a cell detached inherits the worker's stderr -- so the
        # close, and with it ``kill`` (Stop, an account switch), hung forever (hunt 2026-09-30).
        for stream in (proc.stdin, proc.stdout) if proc is not None else ():
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
        self._in_fd = self._out_fd = None


class _WorkerGone(Exception):
    def __init__(self, rc: int | None) -> None:
        super().__init__(f"worker exited ({rc})")
        self.rc = rc
