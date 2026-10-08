#!/usr/bin/env python3
"""
base_mcp.py - Shared utilities for all SpatialOmicsLab MCP tool wrappers.

Standardizes:
  - Logging (stderr with [prefix] tag)
  - Subprocess invocation (CLI args or JSON payload)
  - JSON output parsing (last line of stdout)
  - Error handling (return error dict, never raise)

Usage in a wrapper:

    from base_mcp import create_mcp, run_worker_cli, run_worker_json

    mcp = create_mcp("my-tool")

    @mcp.tool()
    def my_tool(arg1: str, arg2: int = 5) -> dict:
        return run_worker_cli("MY_TOOL", ["--arg1", arg1, "--arg2", str(arg2)])

    if __name__ == "__main__":
        mcp.run()
"""

from __future__ import annotations

import errno
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

_PARENT_WATCH_STARTED = False


def die_with_parent() -> None:
    """Take this portal's process group down with the process that started it. Idempotent.

    The MCP stdio client starts every portal with ``start_new_session=True``, so the portal and the
    tool worker it runs with ``subprocess.run`` sit in a process group of their own. When the REPL
    worker that called the tool is killed -- a cell past its budget, or Stop -- its ``killpg`` never
    reached them: the portal and a full-dataset worker ran on for hours while the model was told to
    re-run the step, and the re-run raced the orphan on the same output directory (hunt 2026-09-30,
    u12-react-1 / u14-mcp-wiring-1). So a portal that leads its own group kills that group when its
    parent goes: the kernel's parent-death signal on Linux, and a re-parenting watch everywhere.

    Only when this process LEADS its group. A portal started from a shell shares the shell's group,
    and killing that group would take down whoever launched it.
    """
    global _PARENT_WATCH_STARTED
    if _PARENT_WATCH_STARTED or os.name != "posix":
        return
    try:
        if os.getpgrp() != os.getpid():
            return
    except OSError:
        return
    _PARENT_WATCH_STARTED = True
    parent = os.getppid()

    def _kill_group(*_args: Any) -> None:
        try:
            os.killpg(os.getpid(), signal.SIGKILL)
        except OSError:
            os._exit(1)

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_pdeathsig = 1
        if libc.prctl(pr_set_pdeathsig, int(signal.SIGUSR2), 0, 0, 0) == 0:
            signal.signal(signal.SIGUSR2, _kill_group)
    except Exception:  # not Linux, no libc, or not the main thread: the watch below still covers it
        pass

    def _watch() -> None:
        while True:
            if os.getppid() != parent:
                _kill_group()
            time.sleep(2.0)

    threading.Thread(target=_watch, name="portal-parent-watch", daemon=True).start()


def create_mcp(name: str) -> Any:
    """Create a FastMCP instance with the given server name."""
    from fastmcp import FastMCP

    die_with_parent()
    return FastMCP(name)


def log(prefix: str, msg: str) -> None:
    """Log a message to stderr with a consistent [prefix] tag."""
    sys.stderr.write(f"[{prefix}] {msg}\n")
    sys.stderr.flush()


_DEPLOYMENT_WORK_ROOT = "/workspace/work"
_ADVERTISED_WORK_ROOT = "./work"


def default_output_dir(subdir: str = "") -> str:
    """Resolve a portal's fallback output directory for a caller that supplies none.

    ``output_dir`` is optional on most portals, so this value is what the tool actually writes to
    whenever the agent omits it. It used to be the absolute ``/workspace/work`` hardcoded into each
    portal signature, which no rebaser touches (``mcp_resolver._rebase_worker`` handles only
    ``*_WORKER``/``*_PYTHON``) -- so on any box without a writable ``/workspace`` the tool died with
    EACCES/ENOENT on a parameter the caller never mentioned.

    Resolution order, first usable wins:
      1. ``SOG_WORK_DIR`` -- lets a deployment point the scratch root anywhere.
      2. ``/workspace/work`` when it exists and is writable, so this deployment's runs keep landing
         exactly where they always have (a live scratch tree; redirecting it is not this fix's job).
      3. ``./work`` -- the relative path the shipped config advertises, correct on any clone.

    Never creates a directory: resolution happens at import time for the portal signatures, and a
    module import must not touch the filesystem. The worker creates the directory it is handed.
    """
    override = os.environ.get("SOG_WORK_DIR", "").strip()
    root = override or (
        _DEPLOYMENT_WORK_ROOT
        if os.path.isdir(_DEPLOYMENT_WORK_ROOT) and os.access(_DEPLOYMENT_WORK_ROOT, os.W_OK)
        else _ADVERTISED_WORK_ROOT
    )
    return os.path.join(root, subdir) if subdir else root


def _local_worker_script(pinned: str) -> str:
    """This clone's copy of ``pinned``'s basename, or "" when it has none.

    Searches the two directories that hold worker scripts, relative to *this* module's own
    location: the shipped ``tools/`` and the sibling ``tools_user/`` the agent writes its own tools
    into. ``tools/`` is searched first so a user-created tool can never shadow a shipped worker.
    Both are named explicitly rather than derived from ``dirname(__file__)`` alone, because
    ``tools_user/base_mcp.py`` is a symlink to this file -- a portal imported through it must still
    find the built-in workers next door.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    base = os.path.basename(pinned)
    for directory in (os.path.join(parent, "tools"), os.path.join(parent, "tools_user"), here):
        candidate = os.path.join(directory, base)
        if os.path.exists(candidate):
            return candidate
    return ""


def resolve_worker_script(worker_path: str) -> str:
    """Adopt this clone's copy of a worker script whose configured path is absent.

    The worker default baked into each portal is an absolute path from the machine that wrote it
    (85 of the 89 portals name this repository's own checkout root), and an ``{PREFIX}_WORKER``
    override written by ``sog-setup`` on one box is just as machine-specific once the config travels.
    Neither is rebased on the agent's launch path: ``mcp_integration`` rewrites ``command``/``args``
    but passes a server's ``env`` block through untouched, and the shipped canonical config supplies
    a ``*_WORKER`` override for fewer than half its servers. So on a clone at any other filesystem
    path the portal launches correctly and then hands the worker a path that does not exist.

    Same policy as ``mcp_integration._rebase_script_args``, applied one level down: only an
    **absolute** ``.py``/``.R`` path that is **missing** is reconsidered, and only in favour of the
    identically-named file sitting next to this module. A path that exists is returned
    byte-identical, and a basename with no local counterpart is left alone so ``_launch_error``
    names what was actually configured.

    Public because ``get_worker_paths`` is not the only way in. That helper derives both override
    names from one prefix (``{PREFIX}_PYTHON`` / ``{PREFIX}_WORKER``), and a portal whose
    interpreter override is spelled otherwise -- ``seurat`` reads ``SEURAT_RSCRIPT``, which
    ``sog_install.capture`` parses out of the source and ``sog_install.wiring`` writes back -- cannot route
    through it without renaming that var everywhere it is recorded. Such a portal calls this
    directly on its worker path and gets the identical repair.
    """
    if not worker_path or not isinstance(worker_path, str):
        return worker_path
    if not worker_path.endswith((".py", ".R")) or not os.path.isabs(worker_path):
        return worker_path
    if os.path.exists(worker_path):
        return worker_path
    return _local_worker_script(worker_path) or worker_path


def get_worker_paths(env_prefix: str, default_python: str, default_worker: str) -> tuple:
    """
    Resolve worker Python path and worker script path from env vars with fallback.

    Environment variables checked:
      - {ENV_PREFIX}_PYTHON  -> path to conda env python
      - {ENV_PREFIX}_WORKER  -> path to worker script

    A worker script that is absent here -- because the default, or the override, names another
    machine's checkout -- resolves to this clone's copy (see :func:`resolve_worker_script`). The
    interpreter gets the equivalent treatment at launch time in :func:`_resolve_worker_python`,
    which needs the filesystem state at dispatch rather than at import.

    Returns (python_path, worker_path).
    """
    python_path = os.environ.get(f"{env_prefix}_PYTHON", default_python)
    worker_path = os.environ.get(f"{env_prefix}_WORKER", default_worker)
    return python_path, resolve_worker_script(worker_path)


# Bidirectional env-name aliases for the post-rebrand heavy-env rename, sourced from the ONE
# sanctioned home -- ``sog_install.constants.LEGACY_ENV_ALIASES`` (branded -> its pre-rebrand
# name). The shipped config/portals pin the branded name; a box that upgraded in place kept the
# pre-rebrand env on disk (and vice-versa for a renamed box handed an older config), so a
# branded<->legacy map lets resolution work whichever name is present. A portal usually CANNOT import
# the package (it runs under the agent env's bare interpreter with a minimal environment -- no
# PYTHONPATH, no editable install), so the import branch falling back to ``{}`` made the alias map
# unreachable exactly where it matters: every portal on a renamed box dispatched svca-style tools at a
# dead interpreter path. The fallback AST-reads the SAME sanctioned constant from this checkout's
# ``install/sog_install/constants.py`` (pure data via ``ast.literal_eval`` -- nothing is executed);
# a missing/unreadable file degrades to ``{}`` as before. Never hardcodes the legacy name here
# (guarded by test_no_legacy_env_name).
def _env_aliases_from_source(constants_py: str | None = None) -> dict[str, str]:
    """AST-read ``LEGACY_ENV_ALIASES`` from the sibling checkout's constants.py (import-free)."""
    import ast

    if constants_py is None:
        # <repo>/agent/tools/base_mcp.py (or the tools_user/ symlink to it) -> <repo>/install/...
        constants_py = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "install",
            "sog_install",
            "constants.py",
        )
    try:
        with open(constants_py, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in tree.body:
            target = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target.id
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
            if target == "LEGACY_ENV_ALIASES" and node.value is not None:
                pairs = ast.literal_eval(node.value)
                return pairs if isinstance(pairs, dict) else {}
    except Exception:
        pass
    return {}


def _env_aliases() -> dict[str, str]:
    try:
        from sog_install.constants import LEGACY_ENV_ALIASES as _pairs
    except Exception:
        _pairs = _env_aliases_from_source()
    if not isinstance(_pairs, dict):
        return {}
    bidir: dict[str, str] = {}
    for branded, legacy in _pairs.items():
        b, leg = str(branded), str(legacy)
        bidir[b] = leg
        bidir[leg] = b
    return bidir


def _live_envs_root() -> str:
    """The directory this process's own conda envs live under (the parent of ``envs/``).

    Derived from ``sys.prefix`` -- the portal is launched with the agent env's interpreter, so its
    prefix is either ``<root>/envs/<name>`` (a named env) or ``<root>`` itself (the base env).
    Deliberately not ``CONDA_PREFIX``: that reflects whatever was last *activated* in the shell,
    which for a portal launched by ``sys.executable`` without activation is the wrong env or absent.
    """
    prefix = os.path.abspath(sys.prefix)
    marker = os.sep + "envs" + os.sep
    idx = prefix.find(marker)
    return prefix[:idx] if idx != -1 else prefix


def _resolve_worker_python(python_path: str) -> str:
    """Resolve a worker interpreter across a renamed env *or* a relocated conda root.

    The common case is a no-op: an interpreter that exists on this box is returned unchanged. Only
    when the pinned path is ABSENT are alternatives tried, in descending order of authority:

    1. The aliased ``.../envs/<NAME>/...`` segment under the **pinned** root (branded<->legacy, via
       ``_env_aliases``) -- a tool whose config pins the branded heavy env still dispatches on a box
       that only has the pre-rebrand env (the svca case), and vice-versa. The root an operator
       pinned stays authoritative, so this is tried before any relocation.
    2. The same env name under the root **this process is running from**. 83 portals bake
       ``/opt/conda/envs/<tool>/bin/python``; a deployment whose conda lives at ``~/miniconda3`` or
       under a Slurm module prefix has every one of those pointing at nothing, even when
       ``sog-setup`` built all the environments correctly.
    3. Both at once -- relocated conda *and* renamed env.

    The interpreter tail (``bin/python`` vs ``bin/Rscript``) is preserved throughout: it names the
    language, so a candidate that lacks it is not a substitute. If nothing resolves, the pinned path
    is returned as-is so the caller's error dict names what was actually configured.
    """
    if not python_path or not isinstance(python_path, str) or os.path.exists(python_path):
        return python_path
    parts = python_path.split(os.sep)
    try:
        i = parts.index("envs")
    except ValueError:
        return python_path
    if i + 1 >= len(parts):
        return python_path
    name = parts[i + 1]
    alias = _env_aliases().get(name)
    tail = parts[i + 2 :]
    pinned_root = os.sep.join(parts[:i])
    live_root = _live_envs_root()
    for root, env_name in ((pinned_root, alias), (live_root, name), (live_root, alias)):
        if not env_name:
            continue
        candidate = os.sep.join([root, "envs", env_name] + tail)
        if candidate != python_path and os.path.exists(candidate):
            return candidate
    return python_path


_UTF8_WORKER_LOCALE = "C.UTF-8"


def _is_utf8_locale(value: str) -> bool:
    """Does this locale name request a UTF-8 charmap? ``C.UTF-8`` and ``en_US.utf8`` both do."""
    return "utf-8" in value.lower().replace("utf8", "utf-8")


def _worker_env() -> dict[str, str]:
    """The caller's environment, with a UTF-8 ctype guaranteed for the worker.

    ``LC_ALL=C`` is the default in minimal containers, cron, systemd units and most Slurm jobs. What
    this sees is THIS portal's environment, and how much of the operator's that is depends on who
    started the portal. Run by hand or imported in-process, it is the caller's, locale included, and
    a C locale reaches the worker verbatim. Spawned by the agent over MCP -- the normal path -- it is
    the SDK's allowlist (``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``, ``TERM``, ``USER``) plus
    ``mcp_integration._FORWARDED_TO_TOOLS`` and the server's ``env:`` block. On 2026-09-30 none of
    those carried ``LC_ALL``/``LC_CTYPE``/``LANG`` or ``SOG_WORKER_LOCALE``, so on that path no
    locale was set and the override below fired whatever the operator's locale was (hunt
    2026-09-30, u14-mcp-wiring-extra-24; this paragraph used to say the C locale arrived "because
    subprocess inherits ``os.environ``").

    Python workers are immune -- PEP 540 turns on UTF-8 mode under any non-UTF-8 locale -- but R
    has no equivalent: under a C ctype R marks strings ``Encoding() == "unknown"``, and ``jsonlite::toJSON`` then
    renders every non-ASCII byte in R's ``<xx>`` escape notation. The same run's ``write.csv``
    passes those bytes through untouched, so one worker publishes ``Müller glia`` in its CSV header
    and ``M<c3><bc>ller glia`` in its stdout payload. The JSON still parses, so nothing errors; the
    corrupted name simply matches neither the ground truth nor the tool's own column headers
    downstream. 18 of the 22 R workers put user-derived names in their payload.

    A UTF-8 locale this process can see is left exactly as set -- only a non-UTF-8 (or absent)
    locale is overridden. The override has to land on ``LC_ALL`` because POSIX precedence puts it
    above both ``LC_CTYPE`` and ``LANG``; measured, ``LC_ALL=C`` plus ``LC_CTYPE=C.UTF-8`` still
    corrupts.

    ``SOG_WORKER_LOCALE`` names the replacement for hosts that lack ``C.UTF-8`` (macOS, older
    glibc); on the MCP path it arrives through ``mcp_integration._FORWARDED_TO_TOOLS``, with LANG/LC_ALL/LC_CTYPE
    (hunt 2026-09-30, u29a-mcp-transport-10). When
    the locale named does not exist R falls back to C with seven visible startup warnings -- noisy,
    but no longer silent.
    """
    env = dict(os.environ)
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = env.get(var, "")
        if not value:
            continue  # POSIX: an empty value defers to the next category
        if _is_utf8_locale(value):
            return env
        break  # the winning category is not UTF-8, and nothing below it can rescue the child
    env["LC_ALL"] = env.get("SOG_WORKER_LOCALE", "").strip() or _UTF8_WORKER_LOCALE
    return env


#: ``errno`` values that reach :func:`_launch_error` meaning something other than "not installed",
#: with the remedy that actually applies.
#:
#: THE BUG THIS EXISTS FOR. ``except OSError`` catches a family and the diagnostic asserted one
#: member of it, so every launch failure told the operator to rebuild a 1-30 GB conda environment
#: that was fine, and told the model the tool was unprovisioned. Driven: ``run_worker_json`` passes
#: the whole payload as a single argv element, so a 156 KB payload hits Linux's 128 KB
#: ``MAX_ARG_STRLEN`` and returns ``[Errno 7] Argument list too long`` -- from the very interpreter
#: that had just run a 13 KB call successfully. Nine portals use that path and four of them declare
#: an unbounded list parameter that lands in the payload; ``ENOMEM`` can hit any of the 88.
#:
#: The real remedy in those cases is "send fewer genes" or "the box is out of memory", and a model
#: has been observed abandoning a tool on weaker signals than a provisioning failure.
_LAUNCH_DIAGNOSTICS: dict[int, str] = {
    errno.E2BIG: (
        "The arguments were too large for the operating system to pass to the worker (Linux caps a "
        "single argument at 128 KB). Shorten the largest argument -- typically a gene or marker "
        "list -- or write it to a file and pass the path instead. This tool is installed correctly "
        "and does not need reprovisioning."
    ),
    errno.ENOMEM: (
        "The host could not allocate memory to start the worker process. Nothing is wrong with this "
        "tool's installation; retry when the machine is less loaded, or reduce the size of the input."
    ),
    errno.EACCES: (
        "The operating system refused permission to execute the worker "
        "(python={python_path!r}, worker={worker_path!r}). The files exist; check their execute "
        "permission and ownership rather than reprovisioning."
    ),
    errno.ENOEXEC: (
        "The operating system could not execute the worker interpreter "
        "(python={python_path!r}) -- it is present but not a runnable binary for this platform."
    ),
}

#: What is said when the errno is genuinely unrecognised, or absent.
_LAUNCH_DIAGNOSTIC_DEFAULT = (
    "The worker interpreter or script path may be missing on this machine "
    "(python={python_path!r}, worker={worker_path!r}). "
    "Re-run `sog-setup` to (re)provision this tool's conda environment."
)


def _launch_error(tool_name: str, python_path: str, worker_path: str, exc: Exception) -> dict[str, Any]:
    """Standard error dict for a worker that could not even be launched. Keeps base_mcp's contract
    -- 'return an error dict, never raise' -- when ``subprocess.run`` raises before the worker starts.

    The diagnostic is chosen by ``errno`` rather than asserted, because the same ``except OSError``
    catches "you never installed this" and "your gene list is 156 KB", and only one of those is
    fixed by re-running ``sog-setup``. See :data:`_LAUNCH_DIAGNOSTICS`.
    """
    template = _LAUNCH_DIAGNOSTICS.get(getattr(exc, "errno", None), _LAUNCH_DIAGNOSTIC_DEFAULT)
    return {
        "status": "error",
        "error": f"{tool_name} worker could not be launched: {exc}",
        "diagnostic": template.format(python_path=python_path, worker_path=worker_path),
    }


def _last_json_object(text: str) -> dict[str, Any] | None:
    """Recover the last top-level JSON *object* from ``text``, tolerating pretty-printed
    (multi-line ``json.dumps(indent=2)``) output and leading progress/log lines on stdout.

    The per-line scan in ``_parse_result`` only matches a result printed on ONE line; a worker
    that pretty-prints (somde/spatialprompt/cell2location/svgbit do) spreads the object over many
    lines, so every line is a fragment and the scan finds nothing -- turning a SUCCESS into a
    bogus parse error. First try the whole (stripped) buffer; if that fails because progress text
    precedes the block, walk each ``{`` with ``raw_decode`` and keep the last object that parses.
    Returns None when no JSON object is present.
    """
    if not text or not text.strip():
        return None
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    last: dict[str, Any] | None = None
    idx = 0
    while True:
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except ValueError:
            idx = brace + 1
            continue
        if isinstance(obj, dict):
            last = obj
        idx = max(end, brace + 1)
    return last


_ARGPARSE_ERROR_LINE = re.compile(r"^\S+: error: (.+)$")


def _argparse_rejection(stderr: str) -> str:
    """The reason argparse refused the call, or "" if this was not an argparse rejection.

    Workers reached through ``run_worker_cli`` validate argv with argparse, which prints a usage
    block and one ``prog: error: ...`` line to stderr and exits before any tool code runs. Read on
    its own, an empty stdout looks identical to a worker that crashed without emitting -- but the
    caller's remedy is the opposite, so the two must be told apart. Both halves of argparse's own
    signature are required: a bare ``usage:`` block (as ``--help`` prints) is not a refusal.
    """
    if not stderr or "usage:" not in stderr:
        return ""
    for line in reversed(stderr.strip().splitlines()):
        match = _ARGPARSE_ERROR_LINE.match(line.strip())
        if match:
            return match.group(1).strip()
    return ""


#: What the caller can do about each way the operating system takes a worker down. Keyed by signal
#: name so an unlisted signal falls through to the generic sentence rather than to silence.
_SIGNAL_ADVICE = {
    "SIGKILL": (
        "The operating system killed the process outright; on Linux an unexplained SIGKILL is "
        "almost always the out-of-memory killer. The worker's code is not at fault and did not "
        "run to completion. Retry on a host with more memory, or with the tool's own settings that "
        "hold less at once (a smaller batch_size, a sparse mode, fewer highly variable genes or latent "
        "dimensions, where the tool has them) -- the whole dataset still has to fit."
    ),
    "SIGTERM": (
        "Something asked the process to stop -- usually a time limit, a batch scheduler, or an "
        "operator. The worker's code is not at fault. Raise the time limit or reduce the work "
        "before retrying."
    ),
    "SIGSEGV": (
        "A native extension crashed the interpreter. This is a fault inside the tool rather than "
        "in the call; the stderr tail below is the only evidence of it."
    ),
    "SIGABRT": (
        "A native library aborted the process. This is a fault inside the tool rather than in the "
        "call; the stderr tail below is the only evidence of it."
    ),
}


def _signal_death(returncode: int) -> str:
    """The name of the signal that killed the worker, or "" if it exited under its own control.

    ``subprocess`` reports a child killed by signal N as a ``returncode`` of ``-N``, so this is the
    operating system speaking and nothing the worker printed can override it. A killed worker did
    not forget to emit -- it was stopped before it could, so the generic "check that the worker
    calls emit()" advice sends the caller to read code that is already correct. Measured over the
    recorded benchmark logs, nine of the twelve empty-stdout results were exactly this case.

    A positive code is a worker choosing its own exit status and is not a signal.
    """
    if not isinstance(returncode, int) or returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"signal {-returncode}"


def _killed_result(tool_name: str, killed: str, returncode: int, stderr: str, stdout: str = "") -> dict[str, Any]:
    """The error for a worker the operating system killed, with whatever it had written."""
    result: dict[str, Any] = {
        "status": "error",
        "error": f"{tool_name} worker was killed by {killed} before it produced a result.",
        "diagnostic": _SIGNAL_ADVICE.get(
            killed,
            f"The process was terminated by {killed} rather than exiting on its own. The "
            f"worker did not reach the end of its run; the stderr tail below is the only "
            f"evidence of how far it got.",
        ),
        "return_code": returncode,
    }
    if stdout:
        result["stdout_tail"] = stdout[-4000:]
    result["stderr_tail"] = stderr[-4000:] if stderr else ""
    return result


def _parse_result(stdout: str, stderr: str, returncode: int, tool_name: str) -> dict[str, Any]:
    """
    Parse worker output into a standardized result dict.

    - Scans stdout lines from bottom up to find the first valid JSON line.
    - If no line is valid JSON, returns an error dict with raw stdout for debugging.
    - On any failure, returns an error dict instead of raising.
    """
    if not stdout or not stdout.strip():
        # A killed worker is checked first because a negative return code is the kernel reporting
        # a signal, which nothing the worker printed earlier can outrank -- and unlike the two
        # readings below it, no advice for it can be inferred from stderr, which the kill leaves
        # empty of any marker of its own.
        killed = _signal_death(returncode)
        if killed:
            return _killed_result(tool_name, killed, returncode, stderr)

        # An argparse refusal and a worker that died before emitting both arrive here as an empty
        # stdout, but the caller's remedy is opposite: one is "fix the argument you passed", the
        # other is "this tool broke". Telling the agent the worker produced no output when its own
        # argument was turned away sends it to debug code it cannot see, while the sentence that
        # would have fixed the call in one turn sits unread in stderr_tail.
        rejected = _argparse_rejection(stderr)
        if rejected:
            return {
                "status": "error",
                "error": f"{tool_name} rejected the arguments: {rejected}",
                "diagnostic": "The call did not pass the worker's argument checks, so nothing ran. "
                "Correct the argument named above and call the tool again; a worker flag such as "
                "--problem-type is the tool parameter problem_type.",
                "return_code": returncode,
                "stderr_tail": stderr[-4000:] if stderr else "",
            }
        return {
            "status": "error",
            "error": f"{tool_name} worker produced no output on stdout (empty stdout).",
            "diagnostic": "The worker process wrote nothing to stdout. "
            "Check that the worker calls emit() or prints JSON before exiting.",
            "return_code": returncode,
            "stderr_tail": stderr[-4000:] if stderr else "",
        }

    lines = stdout.strip().splitlines()

    # Scan from bottom up to find the first valid JSON line.
    # Workers may leak progress messages, warnings, or print() calls to stdout
    # before the final JSON output.
    payload = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        # A worker result is always a JSON *object* (WorkerOutput.emit() dumps a dict).
        # A bare scalar/array line (e.g. a stray `print(json.dumps([...]))` or a lone
        # number/bool) is not a result — skip it so a non-dict last line can't crash
        # `payload.get(...)` below. Module contract: "return error dict, never raise".
        if isinstance(candidate, dict):
            payload = candidate
            break

    # Fallback for a worker that pretty-prints its result as a MULTI-LINE JSON object
    # (json.dumps(..., indent=2)) -- the per-line scan above can never match a multi-line object,
    # so without this a successful somde/spatialprompt/cell2location/svgbit run is misreported as
    # a parse error and its real result (files already written) is discarded.
    if payload is None:
        payload = _last_json_object(stdout)

    if payload is None:
        # The signal outranks stdout here too. A worker the OOM killer took after it had printed
        # anything -- a C library's progress line redirect_stdout cannot catch, or half a
        # pretty-printed result -- used to land below and be told to stop calling print() (hunt
        # 2026-09-30, u29a-mcp-transport-9).
        killed = _signal_death(returncode)
        if killed:
            return _killed_result(tool_name, killed, returncode, stderr, stdout)
        return {
            "status": "error",
            "error": (
                f"{tool_name} worker stdout has content ({len(lines)} line(s)) but no valid JSON object line was found."
            ),
            "diagnostic": "The worker printed to stdout but none of the lines are a valid JSON object. "
            "Ensure the worker uses WorkerOutput.emit() and avoids bare print() calls.",
            "return_code": returncode,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:] if stderr else "",
        }

    if returncode != 0 and payload.get("status") != "ok":
        payload.setdefault("status", "error")
        payload.setdefault("return_code", returncode)
        payload.setdefault("stderr_tail", stderr[-4000:] if stderr else "")

    return payload


#: The step budget the portal's agent worker hands each tool call (``agent/mcp_integration.py``), in
#: seconds. Set ONLY there -- a CLI run, a notebook and a scored trial never carry it -- so everything below
#: is inert outside the portal.
TOOL_BUDGET_ENV = "SOG_TOOL_BUDGET_S"
#: How long before the budget the worker is stopped: room for the portal to return its answer to the cell.
_BUDGET_MARGIN_S = 60
#: The shortest a budgeted worker is given, however small the budget.
_BUDGET_FLOOR_S = 30


def _budgeted(timeout: int | None) -> tuple[int | None, bool]:
    """``(timeout to use, whether it came from the step budget)``. An explicit timeout always wins.

    Why (2026-10-04, the shared-box audit's R3): the tool call and the cell that made it share ONE
    budget, so a worker that ran to the end of it took the cell with it -- the agent's worker process
    was killed and restarted and every variable of the session was lost, and the model retried the same
    call into the same wall. Stopped here a minute early instead, the call returns an ordinary error
    the session survives, and the error says what to do next.
    """
    if timeout is not None:
        return timeout, False
    raw = (os.environ.get(TOOL_BUDGET_ENV) or "").strip()
    try:
        budget = float(raw)
    except ValueError:
        return None, False
    if budget <= 0:
        return None, False
    return max(_BUDGET_FLOOR_S, int(budget - _BUDGET_MARGIN_S)), True


def _timed_out(tool_name: str, timeout: int | None, e: subprocess.TimeoutExpired, budgeted: bool) -> dict[str, Any]:
    partial_stdout = (e.stdout or b"").decode("utf-8", errors="replace")[-2000:] if e.stdout else ""
    partial_stderr = (e.stderr or b"").decode("utf-8", errors="replace")[-2000:] if e.stderr else ""
    error = f"{tool_name} worker timed out after {timeout}s"
    if budgeted:
        error = (
            f"{tool_name} was stopped after {timeout}s, shortly before this step's time limit, so the session "
            "and its variables are kept. Its partial outputs are incomplete: do not reuse them. Run it again "
            "with a lighter configuration (fewer training epochs or components) or choose a lighter method, "
            "and say in the answer that the run was shortened and how."
        )
    return {
        "status": "error",
        "error": error,
        "timeout": timeout,
        "stopped_before_budget": budgeted,
        "stdout_tail": partial_stdout,
        "stderr_tail": partial_stderr,
    }


def run_worker_cli(
    tool_name: str,
    python_path: str,
    worker_path: str,
    args: list[str],
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run a worker via CLI args: [python_path, worker_path, *args].

    Args:
        tool_name: Name for logging/error messages (e.g. "hotspot").
        python_path: Absolute path to the worker's Python interpreter.
        worker_path: Absolute path to the worker script.
        args: CLI arguments to pass to the worker.
        timeout: Maximum seconds to wait for the worker (None = no limit).

    Returns:
        Parsed JSON dict from the worker's stdout (last line).
    """
    python_path = _resolve_worker_python(python_path)
    cmd = [python_path, worker_path] + args
    log(tool_name, f"Running worker: {' '.join(cmd)}")
    timeout, budgeted = _budgeted(timeout)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env=_worker_env()
        )
    except subprocess.TimeoutExpired as e:
        return _timed_out(tool_name, timeout, e, budgeted)
    except OSError as e:
        # Missing/unrunnable interpreter or script -> return a clean error dict, don't raise.
        return _launch_error(tool_name, python_path, worker_path, e)

    stdout = (proc.stdout or "").strip()
    stderr = proc.stderr or ""

    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()

    return _parse_result(stdout, stderr, proc.returncode, tool_name)


#: Payload fields that carry a credential, and the environment variable the worker reads in their
#: place. ``run_worker_json`` puts the payload on the worker's command line, and a command line is
#: readable by every local account (``ps``, ``/proc/<pid>/cmdline``) for the worker's whole life --
#: a UCD run's upload and polling, minutes -- while its environment is readable only by its own uid.
#: So a field named here never rides argv: it is taken out of the payload and handed over in the
#: environment, where the worker's own resolver already looks (``ucdeconvolve_worker`` resolves
#: through ``ucd_token``, which reads ``UCD_TOKEN`` when the payload has no token). ucdeconvolve is
#: the one JSON portal that sends a credential (hunt 2026-09-30, u29a-mcp-transport-5).
_PAYLOAD_CREDENTIALS = {"token": "UCD_TOKEN"}


def run_worker_json(
    tool_name: str,
    python_path: str,
    worker_path: str,
    payload: dict[str, Any],
    timeout: int | None = None,
) -> dict[str, Any]:
    """
    Run a worker via JSON payload: [python_path, worker_path, --json, '<json>'].

    Args:
        tool_name: Name for logging/error messages.
        python_path: Absolute path to the worker's Python interpreter.
        worker_path: Absolute path to the worker script.
        payload: Dict to serialize as JSON and pass via --json flag.
        timeout: Maximum seconds to wait for the worker (None = no limit).

    Returns:
        Parsed JSON dict from the worker's stdout (last line).
    """
    python_path = _resolve_worker_python(python_path)
    env = _worker_env()
    payload = dict(payload)
    for field, var in _PAYLOAD_CREDENTIALS.items():
        secret = payload.get(field)
        if isinstance(secret, str) and secret.strip():
            env[var] = secret.strip()
            del payload[field]
    json_str = json.dumps(payload, ensure_ascii=False)
    cmd = [python_path, worker_path, "--json", json_str]
    log(tool_name, f"Running worker: {cmd[0]} {cmd[1]} --json <payload>")
    timeout, budgeted = _budgeted(timeout)

    try:
        proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        return _timed_out(tool_name, timeout, e, budgeted)
    except OSError as e:
        # Missing/unrunnable interpreter or script -> return a clean error dict, don't raise.
        return _launch_error(tool_name, python_path, worker_path, e)

    stdout = (proc.stdout or "").strip()
    stderr = proc.stderr or ""

    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()

    return _parse_result(stdout, stderr, proc.returncode, tool_name)
