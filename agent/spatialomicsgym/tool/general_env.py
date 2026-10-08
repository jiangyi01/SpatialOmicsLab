"""Run a script in the general-purpose analysis env: the in-process fallback for a tool whose own env failed.

WHAT THIS IS FOR. Every analysis tool runs in its own conda env behind an MCP portal, and when that
env is broken on a machine -- a package that will not import, an interpreter that is not there --
the tool fails the same way every time it is called. The ReAct loop used to have nowhere to go:
the thin agent-core env the REPL runs in deliberately carries no heavy analysis stack, so "do it
here instead" was not an option, and the turn ended on a repeated error. The general env
(``spatialomicsgym_env_general``, built from the core recipe plus an analysis stack; see
``spatialomicsgym/spatialomicsgym_env/spatialomicsgym_env_general.yml``) is where a redo can go.

WHAT THIS IS NOT. It is not a substitute for the tool. ``spatialscope_worker.py`` records why:
falling back to a simpler method silently "produces misleading outputs that look like the real
method". So the model only ever learns this helper exists from the env notice
(``agent/env_fallback.py``), which requires the answer to say the tool could not run and to name
the substitute -- and the helper itself is injected into the REPL only while that layer is active,
never under benchmarking (``execution.inject_custom_functions``).

CONTRACT. :func:`run_in_general_env` never raises. On success it returns the script's stdout; on
any failure it returns a string beginning ``Error: run_in_general_env:`` -- the prefix
``_EXEC_ERROR_RE`` recognises -- so a failed helper call reads to the loop as a failed step and
counts toward the repeated-error guard like any other. The subprocess inherits ``os.environ`` whole,
so ``SOG_WORK_DIR`` and the per-chat output directory reach it unchanged. Calls are capped per turn
(``general_env_max_calls``, default 12) so a model cannot loop on the helper indefinitely.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from typing import Any

#: The name the helper is bound to in the REPL, and unbound from when the layer is inactive.
HELPER_NAME = "run_in_general_env"
_ERROR_PREFIX = "Error: run_in_general_env: "
#: What the timeout falls back to when the config cannot be read.
_DEFAULT_TIMEOUT_SECONDS = 600.0
#: Packages a notice may say are importable in the general env -- probed, never asserted.
PROBE_PACKAGES = (
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

_lock = threading.Lock()
_calls_this_turn = 0
#: The running turn's own per-call budget, when the agent handed one to ``reset_turn_budget``.
_turn_timeout: float | None = None
_probe_cache: dict[str, tuple[str, ...]] = {}


def reset_turn_budget(timeout_seconds: float | None = None) -> None:
    """Start a turn's call budget over. ``STCoscientist`` calls this at every turn start, with ITS
    per-call budget: the CLI's ``--timeout`` and ``STCoscientist(timeout_seconds=...)`` set the
    agent's and never ``default_config``'s, so this helper killed a redo at 600 s with a message
    telling the user to raise the very two settings they had raised (u12-react-10)."""
    global _calls_this_turn, _turn_timeout
    with _lock:
        _calls_this_turn = 0
        try:
            value = float(timeout_seconds) if timeout_seconds is not None else None
        except (TypeError, ValueError):
            value = None
        _turn_timeout = value if value is not None and value > 0 else None


def calls_this_turn() -> int:
    with _lock:
        return _calls_this_turn


def set_calls_this_turn(n: int) -> None:
    """Carry the budget across the process hop: the REPL worker counts its own calls and the
    server-side seam writes the number back, so a turn's cap holds wherever the cells ran."""
    global _calls_this_turn
    with _lock:
        _calls_this_turn = max(0, int(n))


def max_calls() -> int:
    """The per-turn cap, from the config; 12 when the config cannot be read."""
    try:
        from spatialomicsgym.config import default_config

        return max(0, int(getattr(default_config, "general_env_max_calls", 12)))
    except Exception:
        return 12


def _timeout_seconds() -> float:
    if _turn_timeout is not None:
        return _turn_timeout
    try:
        from spatialomicsgym.config import default_config

        value = float(getattr(default_config, "timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        return value if value > 0 else _DEFAULT_TIMEOUT_SECONDS
    except Exception:
        return _DEFAULT_TIMEOUT_SECONDS


def resolve_python() -> tuple[str | None, str]:
    """(interpreter or None, how). An explicit ``general_env_python`` config value wins; an explicit
    path that does not exist is reported, not skipped -- the same rule ``constants.general_python``
    applies to ``$SOG_GENERAL_PYTHON``, which that function also reads."""
    try:
        from spatialomicsgym.config import default_config

        explicit = (getattr(default_config, "general_env_python", None) or "").strip()
    except Exception:
        explicit = ""
    if explicit:
        if os.path.isfile(explicit):
            return explicit, f"general_env_python={explicit}"
        return None, f"general_env_python={explicit!r} names a file that does not exist on this machine"
    try:
        from sog_install.constants import general_python

        return general_python()
    except Exception as exc:  # the resolver is stdlib-only; this is belt and braces
        return None, f"the general env could not be located ({type(exc).__name__}: {exc})"


def available_packages(python: str | None = None) -> tuple[str, ...]:
    """Which of :data:`PROBE_PACKAGES` import under ``python`` -- probed once per interpreter.

    Probed rather than listed from the recipe, so a notice is true on the machine it is printed on:
    a recipe says what was asked for, the interpreter says what is there. An interpreter that cannot
    be probed (absent, broken) reports nothing rather than guessing.
    """
    if python is None:
        python, _ = resolve_python()
    if not python:
        return ()
    with _lock:
        cached = _probe_cache.get(python)
    if cached is not None:
        return cached
    code = "import importlib.util as u, sys\nprint(','.join(p for p in sys.argv[1:] if u.find_spec(p) is not None))"
    try:
        proc = subprocess.run(
            [python, "-c", code, *PROBE_PACKAGES],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        found = tuple(p for p in (proc.stdout or "").strip().split(",") if p) if proc.returncode == 0 else ()
    except Exception:
        found = ()
    with _lock:
        _probe_cache[python] = found
    return found


def _script_dir() -> str | None:
    """Where the script file goes: the run's work dir when there is one, so the model can find it."""
    work = (os.environ.get("SOG_WORK_DIR") or "").strip()
    if work and os.path.isdir(work):
        return work
    return None


def run_in_general_env(code: str, timeout: float | None = None) -> str:
    """Run ``code`` as a Python script in the general-purpose analysis environment.

    Use this only after a tool's own environment has failed (the env notice tells you when).
    ``code`` is a complete, self-contained script as a string -- it runs in a separate process, so
    nothing from this REPL is visible to it; read your inputs from files and write your outputs
    under the output directory you already use (the environment variables are inherited).
    Returns the script's stdout, or a line starting with ``Error:`` when it failed. Whatever it
    computes is NOT the same method as the tool that could not run: say so in your <solution>.
    """
    global _calls_this_turn
    if not isinstance(code, str) or not code.strip():
        return _ERROR_PREFIX + "code must be a non-empty string holding a complete Python script."
    cap = max_calls()
    with _lock:
        if _calls_this_turn >= cap:
            return (
                _ERROR_PREFIX + f"this turn's budget of {cap} general-env calls is spent (SOG_GENERAL_ENV_MAX_CALLS). "
                "Finish with what you have and say in your <solution> what is missing."
            )
        _calls_this_turn += 1
    python, how = resolve_python()
    if not python:
        return _ERROR_PREFIX + f"no general environment is available: {how}"
    budget = float(timeout) if timeout else _timeout_seconds()
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".py", prefix="general_env_", dir=_script_dir(), delete=False
        ) as fh:
            fh.write(code)
            path = fh.name
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        proc = subprocess.run(
            [python, path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=budget,
            env=env,
            cwd=os.getcwd(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return (
            _ERROR_PREFIX + f"the script did not finish within {int(budget)}s and was terminated. Any files it was "
            "writing are incomplete. Do NOT subsample the input to fit; the budget is raised via "
            "STCoscientist(timeout_seconds=...), the --timeout CLI flag, or SOG_TIMEOUT_SECONDS."
        )
    except Exception as exc:
        return _ERROR_PREFIX + f"the script could not be started ({type(exc).__name__}: {exc})"
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    out = proc.stdout or ""
    err = (proc.stderr or "").strip()
    if proc.returncode != 0:
        tail = err[-4000:] if err else "(nothing on stderr)"
        # What the script PRINTED before it failed travels with the failure: a script that wrote
        # its progress and its partial result to stdout, then exited 1, used to hand back only the
        # traceback, and the model re-ran the whole step to learn what it had already learned.
        printed = out.strip()
        if printed:
            tail += "\n[stdout before the failure]\n" + printed[-2000:]
        return _ERROR_PREFIX + f"the script exited with status {proc.returncode} under {python}\n{tail}"
    if not out.strip():
        out = "(the script ran and wrote nothing to stdout)"
    if err:
        out = out.rstrip("\n") + "\n[stderr]\n" + err[-2000:]
    return out


def describe(agent: Any = None) -> dict[str, Any]:
    """A small status dict for doctors and banners: where the interpreter is and what it has."""
    python, how = resolve_python()
    return {
        "python": python,
        "how": how,
        "packages": list(available_packages(python)) if python else [],
        "max_calls": max_calls(),
    }
