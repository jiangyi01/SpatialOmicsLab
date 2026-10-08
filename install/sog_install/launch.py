"""
Launch the ST-Coscientist terminal chat (``stcoscientist``) from the setup layer.

One module, two callers that must behave identically:

* the wizard's skip / end-of-run handoff (:meth:`wizard.Wizard._offer_chat`), and
* the standalone ``sog-setup chat`` subcommand (:func:`main`).

Both resolve a conda env that can actually run the agent **and** wire MCP, then launch
``python -m spatialomicsgym.chat_cli --mcp`` in that env.

* The chat is interactive, so it is launched with a **direct** ``subprocess`` (inherited
  stdio, no timeout) — never through :meth:`Conda.run`, which captures output and enforces a
  timeout.
* The chat's ``--mcp`` is what wires the analysis tools, so it is passed explicitly.

Stdlib + sibling setup modules only, so this imports and runs before any conda env exists.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from . import base_env
from .envtools import Conda, CondaError, no_capture_output_args
from .prompts import PromptIO
from .state import SetupState

# An env must import these to (a) run the agent and (b) wire MCP tools. A bare
# ``pip install -e .`` env has spatialomicsgym + langchain but NOT mcp/nest_asyncio (those
# ship only in the base recipe), so it correctly fails this probe on a fresh device — which
# is what routes the skip path into building the recipe env instead of launching crippled.
AGENT_CORE_MODS = ["spatialomicsgym", "langchain", "mcp", "nest_asyncio"]


def _active_env_name() -> str | None:
    """The conda env ``sog-setup`` is running in.

    :class:`Conda` has no accessor for the active env; it comes from the environment — the
    exact source the wizard already uses (``wizard.py`` reads ``CONDA_DEFAULT_ENV``)."""
    name = os.environ.get("CONDA_DEFAULT_ENV")
    if name:
        return name
    prefix = (os.environ.get("CONDA_PREFIX") or sys.prefix or "").rstrip("/")
    return os.path.basename(prefix) or None


def _candidate_envs(state: SetupState | None) -> list[str]:
    """Ordered, de-duplicated base-env candidates (recorded → active → recipe defaults)."""
    out: list[str] = []
    for name in (
        state.basic_env_name if state else None,
        _active_env_name(),
        "sog",  # the wizard's NEW-mode default env name
        "spatialomicsgym_env",  # the documented minimal agent-core env
    ):
        if name and name not in out:
            out.append(name)
    return out


def resolve_base_env(conda: Conda, state: SetupState | None) -> str | None:
    """Pick an existing env that can import the agent AND wire MCP — or ``None``.

    Probes each candidate with :func:`base_env.scan_core_modules` (an absent env reports
    all-missing, so non-existent candidates are skipped for free). Returns ``None`` when
    nothing qualifies — the caller then builds the recipe base env (skip path) or reports a
    friendly error (``chat`` subcommand). This is exactly what makes a bare fresh-clone env
    trigger a base-env build rather than launching an agent that can't reach its tools."""
    for name in _candidate_envs(state):
        try:
            missing = base_env.scan_core_modules(conda, name, modules=AGENT_CORE_MODS)
        except CondaError:
            continue  # probe couldn't run — treat as not-a-candidate, keep looking
        if not missing:
            return name
    return None


def chat_command(conda: Conda, basic: str, *, mcp: bool = True, passthrough: list[str] | None = None) -> list[str]:
    """Build the ``conda run -n <basic> python -m spatialomicsgym.chat_cli …`` launch argv.

    Passes ``--mcp`` by default (no tools without it), unless the caller's ``passthrough``
    already carries an ``--mcp`` form (then don't double-add)."""
    passthrough = list(passthrough or [])
    argv = [conda.exe, "run", *no_capture_output_args(conda.exe), "-n", basic, "python", "-m", "spatialomicsgym.chat_cli"]
    if mcp and not any(a == "--mcp" or a.startswith("--mcp=") for a in passthrough):
        argv.append("--mcp")
    argv += passthrough
    return argv


def _run_foreground(cmd: list[str]) -> int:
    """Run the chat in the foreground (inherited stdio and terminal, no timeout); Ctrl-C returns ``130``."""
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 130


def offer_chat(
    conda: Conda,
    basic: str | None,
    *,
    io: PromptIO,
    dry_run: bool = False,
    assume_yes: bool = False,
    mcp: bool = True,
    non_interactive: bool = False,
    passthrough: list[str] | None = None,
) -> int:
    """Offer to start the terminal chat; always print the command first (the fallback).

    Returns an exit code: ``2`` (no usable env), the child's exit code when launched, or ``0``
    when the user declines / dry-run / non-interactive."""
    if not basic or not conda.env_exists(basic):
        if dry_run:  # a dry-run reports intent and always succeeds — never the real "no env" failure
            io.note("[dry-run] would start the chat once an agent env exists (`sog-setup` builds one).")
            return 0
        io.warn("no agent env yet to start the chat — run `sog-setup` to build one, then `sog-setup chat`.")
        return 2

    cmd = chat_command(conda, basic, mcp=mcp, passthrough=passthrough)

    # Printed first so it survives a decline/failure.
    io.section("Next: chat with ST-Coscientist")
    io.say("  An interactive terminal session with the agent and its analysis tools.")
    io.say(f"  start it anytime:   conda activate {basic} && stcoscientist --mcp")
    io.say("                      (or: sog-setup chat)")
    io.note(f"(equivalent: {' '.join(cmd)})")

    if dry_run:
        io.note("[dry-run] would start the chat; not now.")
        return 0
    if non_interactive:
        io.note("run the command above to start the chat when you're ready.")
        return 0
    if not (assume_yes or io.ask_yesno("Start the ST-Coscientist chat now?", default=True)):
        io.note("skipped — run `sog-setup chat` when you want it.")
        return 0
    return _run_foreground(cmd)


def main(argv: list[str] | None = None) -> int:
    """``sog-setup chat`` — start the terminal chat in the agent env, reusing the shared handoff logic."""
    ap = argparse.ArgumentParser(
        prog="sog-setup chat",
        description="Start the ST-Coscientist terminal chat in the agent env, with the analysis tools wired.",
    )
    ap.add_argument("--base", help="agent env to launch in (default: the recorded env, else auto-detected)")
    ap.add_argument("--no-mcp", action="store_true", help="start without wiring the analysis tools")
    ap.add_argument("--dry-run", action="store_true", help="print the launch command but don't start")
    # Unknown args forward verbatim to `stcoscientist` (e.g. --model / --source / --path / a question).
    args, passthrough = ap.parse_known_args(argv)

    conda = Conda()  # raises CondaError if no manager → the cli wrapper maps it to exit 3
    io = PromptIO(non_interactive=not sys.stdin.isatty())
    if args.base:
        if not conda.env_exists(args.base):
            io.warn(f"env '{args.base}' not found — omit --base to auto-detect an agent env, or pass an existing one.")
            return 2
        basic = args.base
    else:
        basic = resolve_base_env(conda, SetupState.load())
    return offer_chat(
        conda,
        basic,
        io=io,
        dry_run=args.dry_run,
        assume_yes=True,  # an explicit `sog-setup chat` is the request to start it
        mcp=not args.no_mcp,
        non_interactive=False,
        passthrough=passthrough,
    )


if __name__ == "__main__":
    from .cli import main as _cli_main

    raise SystemExit(_cli_main(["chat", *sys.argv[1:]]))
