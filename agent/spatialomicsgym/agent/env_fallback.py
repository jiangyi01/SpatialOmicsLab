"""In-turn env fallback: when a tool's OWN environment fails, tell the model once, and how to go on.

THE FAILURE THIS ANSWERS. A tool whose conda env is broken on this machine -- a worker that dies
on ``ModuleNotFoundError``, a portal that never answers its handshake, an interpreter path that is
not there -- fails identically on every call. The loop saw an error, the model retried the same
call, and the repeated-error guard ended the turn. Nothing told the model that the arguments were
fine and the *environment* was not, so nothing told it that a different route existed.

INSTRUCT, NEVER SUBSTITUTE. The notice printed here does not run anything. It names the tool and
the reason, states that re-running the call cannot help, and offers ``run_in_general_env(code)``
(``spatialomicsgym/tool/general_env.py``) for an in-process redo -- and it *requires* the final
``<solution>`` to say that the tool could not run and to name the substitute method. The repo
already recorded why the silent version is wrong: ``spatialscope_worker.py`` refuses to fall back to
NNLS quietly because the result "looks like the real method" and is not.

WHERE IT FIRES. ``mcp_integration.make_mcp_wrapper.sync_tool_wrapper`` -- the same channel the
budget notice and the duplicate-call notice use: a line ``print()``ed into the running cell's
captured stdout, so it lands in the observation whether or not the model prints the tool's return
value, on the normal-return path (``base_mcp`` returns error dicts, never raises) and on the
exception path (portal crash, handshake timeout). Once per tool per turn.

A GPU-PATH FAILURE IS NOT ONE. A CUDA build this GPU cannot run, a driver older than the build, a
missing CUDA runtime library: the tool is installed, and the argument that picked the device picked
the broken path. Its notice names the CPU value of the device argument the tool itself declares and
asks for the same call again, and nothing is entered in the "do not call again" ledger. Only when
the tool declares no such argument, or its CPU re-run fails the same way, does the environment
notice above follow.

WHAT IT REFUSES TO CALL AN ENV FAILURE. Argument rejections, signal deaths, per-call budget
timeouts and every scientific error return ``None`` from :func:`classify_env_failure`: for those the
remedy is to fix the call or the data, and a notice offering a different environment would send
the model the wrong way. The E2BIG/ENOMEM/EACCES/ENOEXEC launch diagnostics say in so many words
that the tool "does not need reprovisioning", and they are excluded for the same reason.

GATE. :func:`recovery_active` is the RL-3 gate: false under ``benchmarking_enabled``, whatever the
knob says, so a scored observation is byte-identical with or without this module.
"""

from __future__ import annotations

import json
import re
from importlib.util import find_spec
from typing import Any

NOTICE_TAG = "[env notice]"

#: The exception lines an import failure ends on, at the start of a line in a stderr tail.
_IMPORT_TRACEBACK_TAIL_RE = re.compile(r"(?m)^(?:ModuleNotFoundError|ImportError)\b")
#: Any exception line at the start of a line -- the LAST one is what a tail ended on.
_EXCEPTION_LINE_RE = re.compile(r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt))\b[^\n]*")
#: The same failure quoted inside an error sentence (a worker that caught it and emitted JSON).
_IMPORT_IN_MESSAGE_RE = re.compile(
    r"No module named ['\"]|cannot import name ['\"]|\bModuleNotFoundError\b|\bImportError\b|undefined symbol:|DLL load failed"
)
#: The GPU path failing: a CUDA build this GPU cannot run, a driver older than the build, a CUDA
#: runtime library that is not there. The same tool asked for the CPU usually runs. The libraries
#: match with their suffixed parts too -- torch loads ``libcudnn_cnn_infer.so.8`` and
#: ``libcublasLt.so.11`` on first use, and ``\blibcudnn\b`` stops at the underscore.
_GPU_IN_MESSAGE_RE = re.compile(
    r"no kernel image is available|CUDA driver version is insufficient|\blibcu(?:dnn|blas|dart)\w*"
)
#: Environment failures that are not Python imports: an R library the env lacks, a shared object
#: the interpreter cannot load (the CUDA runtime under torch), a GPU the build was not made for.
_ENV_IN_MESSAGE_RE = re.compile(
    r"there is no package called|cannot open shared object file|" + _GPU_IN_MESSAGE_RE.pattern
)
#: Exception classes a worker's error MESSAGE starts with when the data or the call was wrong.
_DATA_ERROR_RE = re.compile(
    r"^\s*(?:KeyError|ValueError|IndexError|TypeError|AttributeError|AssertionError|FileNotFoundError"
    r"|NotADirectoryError|IsADirectoryError|PermissionError|ZeroDivisionError|StopIteration)\b"
)
_TIMED_OUT = "timed out"
#: The sentences ``tools/base_mcp.py`` and ``agent/mcp_integration.py`` use; matched by content,
#: not by structure, so a portal that phrases a launch failure itself is still recognised.
_LAUNCH_MISSING = "may be missing on this machine"
_COULD_NOT_LAUNCH = "could not be launched"
_NO_STDOUT = "produced no output on stdout"
_REJECTED_ARGS = "rejected the arguments"
_KILLED = "was killed by"
_BUDGET_EXPIRED = "per-call tool budget expired"
_HANDSHAKE = "did not answer its handshake"
_CONNECTION_CLOSED = "Connection closed"
_PORTAL_STDERR = "wrote this to stderr before the connection closed"
#: ``mcp_integration.PORTAL_STARTED_NOTE``: the portal got as far as its banner, so it did not die on import.
_PORTAL_STARTED = "its stderr holds only the startup banner"

#: The reason a GPU-path failure gets. Its own, because its remedy is not another environment: the
#: same tool asked for the CPU (hunt 2026-09-30, u14-mcp-wiring-12).
GPU_REASON = "its GPU path failed (a CUDA build, driver or runtime library this machine cannot use)"
#: The parameters that pick the device, in the order a tool that declares several lets them decide
#: (``device`` overrides ``use_gpu`` wherever both exist).
_DEVICE_KNOBS = ("device", "use_gpu", "gpu")

#: What a notice may say imports in THIS process -- probed at notice time with ``find_spec``.
_HERE_PACKAGES = (
    "scanpy",
    "anndata",
    "squidpy",
    "decoupler",
    "gseapy",
    "sklearn",
    "scipy",
    "numpy",
    "pandas",
    "statsmodels",
    "seaborn",
    "skimage",
    "umap",
    "torch",
)


def recovery_active() -> bool:
    """False whenever this layer must be invisible: benchmarking wins, then the user's knob."""
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return False
    if getattr(default_config, "benchmarking_enabled", False):
        return False
    return bool(getattr(default_config, "env_fallback_enabled", True))


def _payload(result: Any) -> dict[str, Any] | None:
    """The tool's result as the dict ``base_mcp`` produced, when it is one."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        text = result.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                loaded = json.loads(text)
            except ValueError:
                return None
            return loaded if isinstance(loaded, dict) else None
    return None


def _classify_payload(payload: dict[str, Any]) -> str | None:
    status = str(payload.get("status") or "").lower()
    if status == "ok":
        return None
    if status == "dep_missing":
        return "its worker reported a missing dependency (status 'dep_missing')"
    error = str(payload.get("error") or "")
    diagnostic = str(payload.get("diagnostic") or "")
    tail = str(payload.get("stderr_tail") or "")
    if _REJECTED_ARGS in error or _KILLED in error or _TIMED_OUT in error:
        # A timeout's partial stderr can hold an earlier, harmless import warning; the tool ran.
        return None
    if _COULD_NOT_LAUNCH in error:
        # Only the errno-less default diagnostic means "not installed here". E2BIG, ENOMEM, EACCES
        # and ENOEXEC each say the opposite, and re-routing those would hide a fixable call.
        return "its worker interpreter or script is missing on this machine" if _LAUNCH_MISSING in diagnostic else None
    if _NO_STDOUT in error:
        if not _env_failed(tail):
            return None
        if _gpu_failed(tail):
            return GPU_REASON
        return "its worker died before producing any output, on an import or library-load error its environment cannot satisfy"
    if status != "error":
        return None
    if _GPU_IN_MESSAGE_RE.search(error):
        return GPU_REASON
    if _ENV_IN_MESSAGE_RE.search(error):
        return (
            "its worker could not load a library its environment lacks (an R package, a shared object or a GPU build)"
        )
    if _IMPORT_IN_MESSAGE_RE.search(error):
        return "its worker failed on an import error (a package missing or broken in the tool's environment)"
    if _DATA_ERROR_RE.search(error):
        # The worker said what went wrong, and it was the data or the call. Whatever its stderr
        # logged on the way -- an optional accelerator's ModuleNotFoundError, say -- is not why.
        return None
    if _env_failed(tail):
        if _gpu_failed(tail):
            return GPU_REASON
        return "its worker failed on an import error (a package missing or broken in the tool's environment)"
    return None


def _last_exception_line(stderr_tail: str) -> str:
    """The LAST ``SomeError: ...`` line in the tail's final 1200 chars, or ``""``."""
    lines = [m.group(0) for m in _EXCEPTION_LINE_RE.finditer((stderr_tail or "")[-1200:])]
    return lines[-1] if lines else ""


def _env_failed(stderr_tail: str) -> bool:
    """Whether a stderr tail ENDS on an environment failure: its LAST exception line is an import
    error or a library/driver load error -- not merely some earlier line in the window.

    The last line, because a worker that logs an optional accelerator's ``ModuleNotFoundError`` on
    its way to a genuine ``KeyError`` must not be re-routed to another environment, and a portal
    that dies on a ``NameError`` or a ``SyntaxError`` at startup is a portal with a typo, not a
    missing package -- its last line is neither an import nor a library-load error, and that is
    the whole test. A tail with no exception line at all (an R process: ``Error in library(x) :
    there is no package called 'x'``) is judged on the environment sentences alone.
    """
    window = (stderr_tail or "")[-1200:]
    last = _last_exception_line(window)
    if not last:
        return bool(_ENV_IN_MESSAGE_RE.search(window))
    return bool(_IMPORT_TRACEBACK_TAIL_RE.search(last) or _ENV_IN_MESSAGE_RE.search(last))


def _gpu_failed(stderr_tail: str) -> bool:
    """Whether the line :func:`_env_failed` decided on is a GPU-path failure. Same window, same line."""
    window = (stderr_tail or "")[-1200:]
    last = _last_exception_line(window)
    return bool(_GPU_IN_MESSAGE_RE.search(last or window))


def _import_failed(stderr_tail: str) -> bool:
    """Kept as the name the tests and the miner know; the rule is :func:`_env_failed`."""
    return _env_failed(stderr_tail)


def classify_env_failure(result: Any) -> str | None:
    """A short reason when ``result`` says the tool's ENVIRONMENT failed; ``None`` for anything else.

    ``result`` is whatever the MCP wrapper is about to hand back -- a ``base_mcp`` error dict, the
    JSON text of one, or the message of the exception it is about to raise. Never raises.
    """
    try:
        payload = _payload(result)
        if payload is not None:
            return _classify_payload(payload)
        if not isinstance(result, str):
            return None
        # The exception's own message, and -- after the sentence ``mcp_integration`` appends --
        # whatever the portal wrote to stderr. Judged separately: the message says how the call
        # ended, the stderr says what the portal was doing, and a warning in the second that
        # mentions ImportError is not a verdict on the first.
        head, _, stderr = result.partition(_PORTAL_STDERR)
        if _BUDGET_EXPIRED in head or _REJECTED_ARGS in head or _KILLED in head or _TIMED_OUT in head:
            return None
        if _HANDSHAKE in head:
            return "its MCP server did not answer its handshake (the server's environment failed to start)"
        if "No such file or directory" in head and "/bin/python" in head:
            return "its MCP server's interpreter is missing on this machine (the tool's environment is not provisioned)"
        if _GPU_IN_MESSAGE_RE.search(head):
            # The call's own error (an isError result re-raised), so the tool ran and its GPU path
            # failed -- not a portal that never started, which the branches below judge.
            return GPU_REASON
        if _ENV_IN_MESSAGE_RE.search(head) or _IMPORT_IN_MESSAGE_RE.search(head):
            return "its MCP server failed on an import error"
        if _CONNECTION_CLOSED in head:
            if _PORTAL_STARTED in head:
                return None
            if not stderr.strip():
                return "its MCP server closed the connection before answering (it most likely died on import)"
            if _env_failed(stderr):
                return "its MCP server died on an import error while starting"
            return None
        return None
    except Exception:
        return None


def _here_packages() -> str:
    found = [name for name in _HERE_PACKAGES if _spec_ok(name)]
    return ", ".join(found) if found else "none of the analysis packages"


def _spec_ok(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except Exception:
        return False


def env_failure_notice(tool_name: str, reason: str) -> str:
    """The notice, ASCII, one paragraph. Names the tool, the reason, the way on, and the disclosure."""
    from spatialomicsgym.tool import general_env

    python, how = general_env.resolve_python()
    here = _here_packages()
    head = (
        f"{NOTICE_TAG} '{tool_name}' could not run: {reason}. This is the tool's own environment, not "
        f"your arguments -- re-running the same call will fail the same way, so do not call '{tool_name}' "
        "again this turn."
    )
    # The helper's per-turn budget is checked too: offering it when the budget is spent (or set to
    # 0) sent the model into calls that each answered "budget spent" (u14-mcp-wiring-13).
    spent = bool(python) and general_env.calls_this_turn() >= general_env.max_calls()
    if spent:
        way = (
            f" {general_env.HELPER_NAME} has no calls left this turn. If the step can be done with what "
            f"imports in THIS REPL ({here}), do it here; otherwise stop trying this tool and say what is missing."
        )
    elif python:
        there = ", ".join(general_env.available_packages(python)) or "its packages could not be listed"
        way = (
            f" To keep going, redo the step in-process: call {general_env.HELPER_NAME}(code) with a "
            "complete, self-contained Python script as a string. It runs in the general-purpose analysis "
            f"environment ({there}), inherits your environment variables (write outputs under the same "
            "output directory you already use), and returns the script's stdout -- or a line starting "
            f"with 'Error:' when it fails. Packages importable in THIS REPL: {here}."
        )
    else:
        way = (
            f" The general-purpose environment is not available on this machine ({how}). If the step "
            f"can be done with what imports in THIS REPL ({here}), do it here; otherwise stop trying "
            "this tool and say what is missing."
        )
    disclosure = (
        f" Whatever you run instead is NOT the same method as '{tool_name}': your final <solution> must "
        f"say that '{tool_name}' could not run in its environment and must name the substitute method "
        "you used -- the two are not the same method."
    )
    # ASCII whatever the interpreter path or the tool name contains: this text is the model's
    # template for its next <execute> block, and a non-ASCII byte there has crashed the executor.
    return (head + way + disclosure).encode("ascii", "replace").decode("ascii")


def _tool_parameters(agent: Any, tool_name: str) -> dict[str, dict[str, Any]]:
    """The wired tool's parameters by name -- ``{"type": ..., "default": ...}`` -- or ``{}``.

    Read off the wrapper: the schema ``add_mcp`` recorded on it (``_sog_spec``), else the signature
    ``attach_kwarg_signature`` gave it, which is all a REPL worker's rebuilt wrapper carries.
    """
    try:
        fn = (getattr(agent, "_custom_functions", None) or {}).get(tool_name)
        if fn is None:
            return {}
        spec = getattr(fn, "_sog_spec", None)
        if isinstance(spec, dict):
            out: dict[str, dict[str, Any]] = {}
            for entry in list(spec.get("required") or []) + list(spec.get("optional") or []):
                if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                    out.setdefault(
                        entry["name"], {"type": str(entry.get("type") or ""), "default": entry.get("default")}
                    )
            if out:
                return out
        import inspect

        params = inspect.signature(fn).parameters
        out = {}
        for name, param in params.items():
            if param.kind in (param.KEYWORD_ONLY, param.POSITIONAL_OR_KEYWORD):
                annotation = param.annotation
                kinds = [a for a in getattr(annotation, "__args__", (annotation,)) if a is not type(None)]
                out[name] = {"type": getattr(kinds[0], "__name__", "") if kinds else "", "default": None}
        return out
    except Exception:
        return {}


def cpu_argument(agent: Any, tool_name: str) -> str | None:
    """The argument that runs ``tool_name`` on the CPU, as ``name=value`` text; ``None`` if it has none.

    Only a knob the tool declares, with a value of the type it declares: ``device='cpu'`` for a device
    string (the tool's own spelling when its default already names the CPU, as ``'CPU'``),
    ``use_gpu=False``/``gpu=False`` for a switch, ``use_gpu='cpu'`` for the string-valued one, and
    ``gpu=-1`` for a GPU index, the convention its tools document for "no GPU".
    """
    params = _tool_parameters(agent, tool_name)
    for knob in _DEVICE_KNOBS:
        if knob not in params:
            continue
        kind = str(params[knob].get("type") or "").lower()
        default = params[knob].get("default")
        if kind in ("bool", "boolean"):
            return f"{knob}=False"
        if kind in ("int", "integer"):
            return f"{knob}=-1"
        if knob == "device" and isinstance(default, str) and default.strip().lower() == "cpu":
            return f"{knob}={default.strip()!r}"
        return f"{knob}='cpu'"
    return None


def gpu_failure_notice(tool_name: str, argument: str) -> str:
    """The GPU notice, ASCII, one paragraph: the same tool, on the CPU, and what follows if that fails."""
    text = (
        f"{NOTICE_TAG} '{tool_name}' could not run: {GPU_REASON}. The tool itself is installed -- its GPU "
        f"path is what failed here. Call '{tool_name}' again with {argument} and otherwise the same "
        "arguments; on the CPU it takes longer and gives the tool's own result. If that call fails the "
        "same way, the failure is the tool's environment rather than the device, and you will be told how "
        "to go on."
    )
    return text.encode("ascii", "replace").decode("ascii")


def _first_gpu_failure(agent: Any, tool_name: str) -> bool:
    """True the first time this turn ``tool_name`` gets the CPU notice; recorded on the agent.

    A set apart from ``_env_failures`` on purpose: that ledger is read as "do not call again" by the
    rescue prompt and by ``rescue.succeeded_tools``, and a tool told to re-run on the CPU must be
    neither. Reset with it in ``STCoscientist._reset_recovery_state``.
    """
    if agent is None:
        return True
    noticed = getattr(agent, "_gpu_retry_noticed", None)
    if not isinstance(noticed, set):
        noticed = set()
        try:
            agent._gpu_retry_noticed = noticed
        except Exception:
            return True
    if tool_name in noticed:
        return False
    noticed.add(tool_name)
    return True


def note_env_failure(agent: Any, tool_name: str, reason: str) -> bool:
    """Record the failure on the agent for this turn; True the FIRST time this tool is recorded.

    The dict is ``agent._env_failures``, reset by ``STCoscientist._reset_recovery_state`` at turn
    start and read by the rescue prompt ("tools whose env failed -- do not call them again").
    """
    if agent is None:
        return True
    failures = getattr(agent, "_env_failures", None)
    if not isinstance(failures, dict):
        failures = {}
        try:
            agent._env_failures = failures
        except Exception:
            return True
    if tool_name in failures:
        return False
    failures[tool_name] = reason
    return True


def failed_tools(agent: Any) -> dict[str, str]:
    """The tools whose env failed this turn, and why. Empty when none did."""
    failures = getattr(agent, "_env_failures", None)
    return dict(failures) if isinstance(failures, dict) else {}


def notice_for(agent: Any, tool_name: str, result: Any) -> str | None:
    """The notice to print for this tool call, or ``None``. Never raises; silent under benchmarking.

    The one entry point the MCP wrapper calls, on both of its exit paths.
    """
    try:
        if not recovery_active():
            return None
        reason = classify_env_failure(result)
        if reason is None:
            return None
        if reason == GPU_REASON and tool_name not in failed_tools(agent):
            # The GPU path failed, and the device is an argument: the remedy is the same tool on the
            # CPU, not "never call it again" plus a substitute method in another environment, which
            # is what this notice said (hunt 2026-09-30, u14-mcp-wiring-12). Not entered in the
            # env ledger. A tool with no device argument, or one whose CPU re-run failed the same
            # way, falls through to the environment notice below -- then the environment IS why.
            argument = cpu_argument(agent, tool_name)
            if argument is not None and _first_gpu_failure(agent, tool_name):
                return gpu_failure_notice(tool_name, argument)
        if not note_env_failure(agent, tool_name, reason):
            return None
        return env_failure_notice(tool_name, reason)
    except Exception:
        return None
