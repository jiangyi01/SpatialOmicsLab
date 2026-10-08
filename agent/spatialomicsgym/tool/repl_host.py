"""The REPL worker: every ``<execute>`` cell runs HERE when ``repl_isolation`` is ``"process"``.

``python -m spatialomicsgym.tool.repl_host``, spawned by ``tool/repl_client.py`` -- from the portal
as the unprivileged ``sog-agent`` user (``sog_portal/boundary.py``), or as the same user when the box
has no such user (the client then reports ``process-only`` and never claims more). It speaks the
frames in ``repl_protocol`` over the descriptors it was started with, and it reuses
``support_tools.PythonREPL`` unchanged: the same persistent namespace, the same matplotlib
capture, the same ``Error:`` texts -- only the process, and the user, are different.

Why the protocol descriptors are moved first. A cell is arbitrary code: ``print`` in a thread the
REPL did not redirect, a C library writing to stdout, ``os.system("ls")``, ``input()``. If the
protocol shared fds 0/1 with the cell, any of those would corrupt or block a frame. So ``main``
``dup``s 0 and 1 for the protocol, points fd 0 at ``/dev/null`` (``input()`` sees EOF) and fd 1 at
fd 2 (stray stdout reaches the server's log with the rest of stderr), and no cell corrupts a frame
BY ACCIDENT after that.

What this does not claim: that a cell cannot lie to its own protocol. The dup'ed descriptors are
ordinary descriptors in the process the cell runs in, so a cell written to do so can write a frame
onto them. Every request therefore carries a nonce the reply must echo (a cell that never read the
request cannot forge its reply), the server confirms the worker is idle after a cell before it
trusts the result, and the server's own wall clock -- not the pipe -- decides when the process
group is killed. What a lying cell can never do is act as anyone but ``sog-agent``: the boundary
is the uid, and the protocol is only how results travel.

What the worker builds itself: every MCP wrapper, from the spec ``add_mcp`` recorded, with the
same module-level factory -- so the tool portals are spawned by THIS process, as THIS user, and
``resolve_server_env`` reads ``${SOG_WORK_DIR}`` from this environ, which the server sets per cell.
The duplicate-dispatch memo, the budget notice and the env-failure notice run here against an
``_AgentProxy``; the ledger they write comes back in the cell's reply and the server-side seam
merges it onto the real agent, so ``rescue.py`` and ``env_fallback.py`` read what they always read.

What the worker asks the server for (upcalls): anything that needs the server's secrets or the
server's user -- the LLM-backed ``make_prompt_tool``, and under the brokered tool-creation policy
an environment or a tool file. It binds a stub per name the server offers, whose body is one
frame out and one frame back.
"""

from __future__ import annotations

import importlib
import os
import re
import sys
import threading
import traceback
from typing import Any

from spatialomicsgym.tool.repl_protocol import PARTIAL_INTERVAL, FrameTooLarge, read_frame, write_frame

#: How this worker introduces itself; the client reads ``uid`` to decide what boundary it has.
HELLO_KEYS = ("pid", "uid", "gid", "groups", "cwd", "executable", "python")

_DENIED_RE = re.compile(r"\[Errno 13\] Permission denied: '([^']+)'")

#: The most figure bytes one cell's reply carries. A cell that saves twenty dense scatters at
#: 150 dpi could otherwise exceed the wire's frame ceiling, and the first version of this file
#: let that kill the worker -- namespace and all -- with advice about "loading less data".
MAX_FIGURE_BYTES = 16 * 1024 * 1024
FIGURE_BUDGET_VAR = "SOG_REPL_MAX_FIGURE_BYTES"


def _figure_budget() -> int:
    try:
        return max(0, int(os.environ.get(FIGURE_BUDGET_VAR) or MAX_FIGURE_BYTES))
    except ValueError:
        return MAX_FIGURE_BYTES


def _within_budget(figures: list[str]) -> tuple[list[str], int]:
    kept: list[str] = []
    total = 0
    budget = _figure_budget()
    for fig in figures:
        size = len(fig)
        if total + size > budget:
            continue
        kept.append(fig)
        total += size
    return kept, len(figures) - len(kept)


def _after(output: str, line: str) -> str:
    """``line`` after ``output``, on a line of its own: how this worker puts its notices after a cell."""
    return output.rstrip("\n") + "\n" + line + "\n"


class _AgentProxy:
    """What the wrapper factory, the notices and the general-env helper see instead of the agent."""

    def __init__(self) -> None:
        self.timeout_seconds: float | None = None
        #: The server's agent's, so the budget notice names the knobs where the real agent would.
        self.conversation_memory: bool = False
        self._env_failures: dict[str, str] = {}
        self._turn_owner: str = ""
        self._custom_functions: dict[str, Any] = {}


class _PartialReporter:
    """Sends ``{"id": rid, "partial": text}`` while one cell runs: the tail of its output so far.

    ``text`` is what ``PythonREPL`` published into the cell's slot -- the same bounded tail and
    warnings the in-process timeout observation carries, fitted to the ``room`` the server said
    its observation has for them. A frame is written under the host's write lock, and never once
    :meth:`stop` has been called, so every report for a cell reaches the server BEFORE that cell's
    reply. Harmless if forged: it is the cell's own output, the server reads it only to describe a
    cell it has just killed, and it holds that description to its limit itself.
    """

    def __init__(self, host: Host, rid: Any, slot: dict[str, Any], room: int | None = None) -> None:
        self._host, self._rid, self._slot, self._room = host, rid, slot, room
        self._halt = threading.Event()
        self._sent = ""
        self._thread = threading.Thread(target=self._run, name="repl-partial", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        from spatialomicsgym.utils.execution import read_partial_output

        while not self._halt.wait(PARTIAL_INTERVAL):
            text = read_partial_output(self._slot, self._room)
            if not text or text == self._sent:
                continue
            try:
                if not self._host._send({"id": self._rid, "partial": text}, unless=self._halt):
                    return
            except Exception:
                return  # a report that cannot be sent must not take the cell down
            self._sent = text


class Host:
    """One worker's state: the REPL, the proxy, and the names the last ``configure`` bound."""

    def __init__(self, proto_in: int, proto_out: int) -> None:
        from spatialomicsgym.tool.support_tools import PythonREPL

        self._in = proto_in
        self._out = proto_out
        self.repl = PythonREPL()
        self.agent = _AgentProxy()
        self._bound: set[str] = set()
        self._seq = 0
        #: A request the server sent while this worker was waiting for an upcall's answer: the
        #: server gave up on that upcall; the cell is told so, and the request is served next.
        self._pending: list[dict[str, Any]] = []
        #: Roots under which a cell's ``Errno 13`` gets the broker notice, and which have had it.
        self._protected_roots: list[str] = []
        self._noticed: set[str] = set()
        #: Every frame out goes through :meth:`_send`: the partial reporter writes from its own
        #: thread while the cell may be writing an upcall.
        self._write_lock = threading.Lock()
        #: The id of the request being served, for the partial reports that belong to it.
        self._rid: Any = None

    # -- lifecycle ----------------------------------------------------------------------------- #
    def hello(self) -> None:
        try:
            groups = sorted(os.getgroups())
        except Exception:
            groups = []
        self._send(
            {
                "hello": {
                    "pid": os.getpid(),
                    "uid": os.getuid(),
                    "gid": os.getgid(),
                    "groups": groups,
                    "cwd": os.getcwd(),
                    "executable": sys.executable,
                    "python": sys.version.split()[0],
                }
            },
        )

    def serve(self) -> int:
        while True:
            frame = self._pending.pop(0) if self._pending else read_frame(self._in, None)
            if frame is None:
                return 0
            op = str(frame.get("op") or "")
            rid = frame.get("id")
            nonce = frame.get("nonce")
            self._rid = rid
            handler = getattr(self, f"op_{op}", None)
            if handler is None:
                self._reply({"id": rid, "nonce": nonce, "ok": False, "error": f"unknown op {op!r}"})
                continue
            try:
                result = handler(frame.get("payload") or {})
            except Exception as exc:  # a handler that raises must not take the worker down
                self._reply(
                    {
                        "id": rid,
                        "nonce": nonce,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "trace": traceback.format_exc(),
                    }
                )
                continue
            self._reply({"id": rid, "nonce": nonce, "ok": True, **(result or {})})
            if op == "shutdown":
                return 0

    def _send(self, frame: dict[str, Any], *, unless: threading.Event | None = None) -> bool:
        """Write one frame, whole, under the write lock. ``False`` (and nothing written) if
        ``unless`` is set by the time the lock is held."""
        with self._write_lock:
            if unless is not None and unless.is_set():
                return False
            write_frame(self._out, frame)
            return True

    def _reply(self, frame: dict[str, Any]) -> None:
        """Send a reply; a reply too large for the wire becomes an error frame, not worker death."""
        try:
            self._send(frame)
        except FrameTooLarge as exc:
            self._send(
                {
                    "id": frame.get("id"),
                    "nonce": frame.get("nonce"),
                    "ok": False,
                    "error": f"the reply was too large for the wire ({exc}); figures were dropped",
                },
            )

    # -- the ops ------------------------------------------------------------------------------- #
    def op_ping(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"pid": os.getpid()}

    def op_shutdown(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {}

    def op_configure(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Bind the tools the server describes; unbind what the last configure bound and this one
        does not (a trashed user tool stops being callable here too)."""
        from spatialomicsgym.agent.mcp_integration import attach_kwarg_signature, make_mcp_wrapper

        knobs = payload.get("knobs") or {}
        self._apply_knobs(knobs)
        self.agent.timeout_seconds = knobs.get("timeout_seconds") or self.agent.timeout_seconds
        self.agent.conversation_memory = bool(knobs.get("conversation_memory", False))
        self.agent._turn_owner = str(payload.get("owner") or "")

        fresh: dict[str, Any] = {}
        unresolved: list[str] = []
        modules: dict[str, dict[str, Any]] = {}
        for spec in payload.get("tools") or []:
            try:
                fn = make_mcp_wrapper(
                    self.agent,
                    str(spec["cmd"]),
                    list(spec.get("args") or []),
                    str(spec["name"]),
                    str(spec.get("doc") or ""),
                    dict(spec.get("env_spec") or {}),
                )
                attach_kwarg_signature(fn, list(spec.get("required") or []), list(spec.get("optional") or []))
                fresh[str(spec["name"])] = fn
                module = str(spec.get("module") or "")
                if module.startswith("mcp_servers."):
                    modules.setdefault(module, {})[str(spec["name"])] = fn
            except Exception as exc:
                unresolved.append(f"{spec.get('name')}: {type(exc).__name__}: {exc}")
        for ref in payload.get("import_tools") or []:
            name = str(ref.get("name") or "")
            try:
                obj: Any = importlib.import_module(str(ref["module"]))
                for part in str(ref["qualname"]).split("."):
                    obj = getattr(obj, part)
                fresh[name] = obj
            except Exception as exc:
                unresolved.append(f"{name}: {type(exc).__name__}: {exc}")
        for record in payload.get("declarative") or []:
            try:
                from spatialomicsgym.agent.execution import _prompt_tool_callable
                from tools_user import declarative as _declarative

                name = str(record.get("name") or "")
                if name.isidentifier():
                    fresh[name] = _prompt_tool_callable(_declarative, record)
            except Exception as exc:
                unresolved.append(f"{record.get('name')}: {type(exc).__name__}: {exc}")
        for up in payload.get("upcalls") or []:
            name = str(up.get("name") or "")
            if name.isidentifier():
                fresh[name] = self._upcall_stub(name, str(up.get("doc") or ""), list(up.get("params") or []))

        # The general-env helper: seeded HERE, by the knob, because it is a subprocess runner and
        # must run as this user (its script inherits this environ, this uid, this cwd).
        try:
            from spatialomicsgym.tool import general_env

            if payload.get("seed_helper"):
                fresh[general_env.HELPER_NAME] = general_env.run_in_general_env
        except Exception as exc:
            unresolved.append(f"general_env: {type(exc).__name__}: {exc}")

        for stale in self._bound - set(fresh):
            self.repl._namespace.pop(stale, None)
        self.repl._namespace.update(fresh)
        self.agent._custom_functions = dict(fresh)
        self._bound = set(fresh)
        self._install_tool_modules(modules)
        self._apply_env(payload.get("env") or {})
        self._protected_roots = [str(r) for r in (payload.get("protected_roots") or []) if r]
        return {"bound": sorted(fresh), "unresolved": unresolved}

    def _install_tool_modules(self, modules: dict[str, dict[str, Any]]) -> None:
        """Make ``from mcp_servers.<server> import <tool>`` resolve here, as ``add_mcp`` does in-process.

        The retrieval-mode prompt tells the model to import every function from its module, and
        lists MCP tools under ``mcp_servers.<server>`` -- a module ``add_mcp`` registers in
        ``sys.modules`` of the process that wires. This worker bound bare names only, so a model
        that followed the instruction got ModuleNotFoundError (hunt 2026-09-30, u13-prompt-4).
        Rebuilt on every configure, so a tool that is gone is gone from its module too.
        """
        import types

        for name in [m for m in sys.modules if m.startswith("mcp_servers.") and m not in modules]:
            if getattr(sys.modules[name], "_sog_worker_shim", False):
                del sys.modules[name]
        for name, functions in modules.items():
            shim = types.ModuleType(name)
            shim._sog_worker_shim = True  # type: ignore[attr-defined]
            for tool, fn in functions.items():
                setattr(shim, tool, fn)
            sys.modules[name] = shim

    def op_exec(self, payload: dict[str, Any]) -> dict[str, Any]:
        from spatialomicsgym.agent.tool_call_memo import arm_tool_call_memo, disarm_tool_call_memo
        from spatialomicsgym.tool import general_env
        from spatialomicsgym.tool.support_tools import _summarize_namespace, _with_warnings
        from spatialomicsgym.utils.execution import OBSERVATION_CHARS, partial_output_slot

        self._apply_env(payload.get("env") or {})
        # The ledger is authoritative from the server: it is reset there at turn start.
        self.agent._env_failures = dict(payload.get("env_failures") or {})
        try:
            general_env.set_calls_this_turn(int(payload.get("general_env_calls") or 0))
        except Exception:
            pass
        self.repl.clear_captured_plots()
        arm_tool_call_memo()
        try:
            room = payload.get("partial_room")
            with partial_output_slot() as slot:
                reporter = _PartialReporter(self, self._rid, slot, room if isinstance(room, int) else None)
                reporter.start()
                try:
                    text, warned = self.repl.run_cell(str(payload.get("code") or ""))
                finally:
                    reporter.stop()
        finally:
            disarm_tool_call_memo()
        # The lines this worker puts after the cell are known BEFORE its warnings are sized. The REPL
        # gives the warnings whatever room the observation has left, so a notice appended after that
        # -- the boundary notice alone is some 900 characters -- pushed a cell that fitted with its
        # notice into the clip, for its warnings' sake.
        notices = [line for line in (self._broker_notice(text),) if line]
        try:
            summary = _summarize_namespace(self.repl._namespace)
        except Exception:
            summary = ""
        figures, dropped = _within_budget(list(self.repl.get_captured_plots()))
        if dropped:
            notices.append(
                f"[figures notice] {dropped} figure(s) were not carried into the transcript because together they "
                f"exceed {_figure_budget() // (1024 * 1024)} MB; the files you saved are untouched."
            )
        # Each on a line of its own after the warnings, whose block ends in one newline: its length + 1.
        output = _with_warnings(text, warned, OBSERVATION_CHARS - sum(len(line) + 1 for line in notices))
        for line in notices:
            output = _after(output, line)
        return {
            "output": output,
            "figures": figures,
            "env_failures": dict(self.agent._env_failures),
            "general_env_calls": general_env.calls_this_turn(),
            "namespace_summary": summary,
        }

    def op_exec_shell(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run a ``#!BASH`` / ``#!R`` cell HERE, in the worker, rather than in the server.

        This op exists because the two languages used to skip the boundary entirely. The portal
        runs as root; ``execution.py`` sent a python cell to ``op_exec`` (this process, uid 999,
        ``boundary.agent_env()``, no provider keys) but sent ``#!BASH`` and ``#!R`` straight to
        ``utils.execution.run_bash_script`` / ``run_r_code``, which start their child with
        ``env=os.environ.copy()`` and ``cwd=os.getcwd()`` **in the server process**. So one half of
        a turn was confined and the other half ran as root with every provider key in its
        environment -- and ``prompt_builder`` tells the model to use ``#!BASH`` by name.

        Nothing here re-implements the runners: they are imported and called unchanged. What makes
        the difference is only *where* they are called from. This process was started by
        ``boundary.spawn_plan`` -- setuid 999, supplementary group ``sog-u-<account>``, the env
        allowlist, ``RLIMIT_NPROC``, ``SOG_WORK_DIR`` -- so a shell cell inherits every one of those
        by construction rather than by a second copy of the policy. That includes this worker's
        process group: the client ends a cell on the budget and on Stop by SIGKILLing the group, and
        the runners leave their child in it for exactly that reason.
        """
        from spatialomicsgym.utils.execution import run_bash_script, run_r_code

        self._apply_env(payload.get("env") or {})
        kind = str(payload.get("kind") or "bash").strip().lower()
        code = str(payload.get("code") or "")
        timeout = float(payload.get("timeout") or 60.0)
        runner = run_r_code if kind == "r" else run_bash_script
        try:
            output = runner(code, timeout=timeout)
        except Exception as exc:  # a runner that raises must not take the worker down
            output = f"Error in execution: {type(exc).__name__}: {exc}"
        return {"output": self._with_broker_notice(str(output)), "kind": kind}

    def op_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        from spatialomicsgym.tool.support_tools import _summarize_namespace

        return {"summary": _summarize_namespace(self.repl._namespace, int(payload.get("limit") or 20))}

    def op_forget(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = str(payload.get("name") or "")
        found = name in self.repl._namespace
        self.repl._namespace.pop(name, None)
        self._bound.discard(name)
        return {"found": found}

    def op_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.repl._namespace.clear()
        self.repl.clear_captured_plots()
        self._bound = set()
        return {}

    def op_set_env(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._apply_env(payload.get("env") or {})
        return {}

    # -- helpers ------------------------------------------------------------------------------- #
    def _with_broker_notice(self, output: str) -> str:
        """``output``, then the broker notice on a line of its own when :meth:`_broker_notice` gives one."""
        notice = self._broker_notice(output)
        return _after(output, notice) if notice else output

    def _broker_notice(self, output: str) -> str:
        """The broker notice, the first time a cell is denied under a protected root; else ``""``.

        The denial itself is the REPL's own ``Error: [Errno 13] Permission denied: '<path>'``; the
        notice says what to call instead. Once per root per worker -- the same failure twice still
        carries the Errno 13, and the notice is in the transcript already.
        """
        if not self._protected_roots or "Errno 13" not in output:
            return ""
        match = _DENIED_RE.search(output)
        if not match:
            return ""
        path = match.group(1)
        for root in self._protected_roots:
            if path == root or path.startswith(root.rstrip("/") + "/"):
                if root in self._noticed:
                    return ""
                self._noticed.add(root)
                try:
                    from spatialomicsgym.agent.broker import notice

                    return notice(path)
                except Exception:
                    return ""
        return ""

    def _apply_env(self, env: dict[str, Any]) -> None:
        """``None`` deletes; everything else is set. This is how per-turn ``SOG_WORK_DIR`` arrives."""
        for key, value in env.items():
            if value is None:
                os.environ.pop(str(key), None)
            else:
                os.environ[str(key)] = str(value)

    def _apply_knobs(self, knobs: dict[str, Any]) -> None:
        """The configuration the notices and the helper read, mirrored from the server's."""
        try:
            from spatialomicsgym.config import default_config
        except Exception:
            return
        for key in (
            "timeout_seconds",
            "env_fallback_enabled",
            "unsolved_rescue_enabled",
            "general_env_python",
            "general_env_max_calls",
            "benchmarking_enabled",
        ):
            if key in knobs:
                try:
                    setattr(default_config, key, knobs[key])
                except Exception:
                    pass

    def _upcall_stub(self, name: str, doc: str, params: list[str]) -> Any:
        """A callable whose body is one frame out and one back. Positional arguments are bound
        to the parameter names the server sent, so ``make_prompt_tool("n", "d", "t")`` works
        exactly as it does in-process; the wire carries keywords only."""
        host = self

        def _stub(*args: Any, **kwargs: Any) -> Any:
            if len(args) > len(params):
                raise TypeError(f"{name}() takes at most {len(params)} positional arguments but {len(args)} were given")
            bound = dict(zip(params, args, strict=False))
            for key in bound:
                if key in kwargs:
                    raise TypeError(f"{name}() got multiple values for argument {key!r}")
            bound.update(kwargs)
            return host.upcall(name, bound)

        _stub.__name__ = name
        _stub.__qualname__ = name
        _stub.__doc__ = doc
        try:
            import inspect

            _stub.__signature__ = inspect.Signature(
                [inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD) for p in params]
            )
        except Exception:
            pass
        return _stub

    def upcall(self, name: str, args: dict[str, Any]) -> Any:
        """Ask the server to do ``name`` with ``args``; block until it answers.

        Single-threaded by construction: the cell holds the request loop, so the input descriptor
        is free, and the only frame the server sends now is the answer (or nothing, if it died).
        """
        self._seq += 1
        rid = self._seq
        self._send({"id": rid, "upcall": name, "args": args})
        while True:
            frame = read_frame(self._in, None)
            if frame is None:
                raise RuntimeError(f"the server went away while answering {name}()")
            if frame.get("id") == rid and "upcall_result" in frame:
                return frame["upcall_result"]
            if "op" in frame:
                # The server moved on to a new request: it gave up on this upcall (its cell budget
                # ran out). Serve that request after this cell ends, and end this cell now.
                self._pending.append(frame)
                raise RuntimeError(f"the server stopped waiting for {name}() -- the cell's time budget ran out")


def main(argv: list[str] | None = None) -> int:
    # Before any agent module is imported: the package loads .env at import time, and this process
    # must hold exactly the environment its spawner built (see ``boundary.agent_env``).
    os.environ["SOG_SKIP_DOTENV"] = "1"
    proto_in = os.dup(0)
    proto_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
    except Exception:
        pass
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass
    host = Host(proto_in, proto_out)
    host.hello()
    return host.serve()


if __name__ == "__main__":
    sys.exit(main())
