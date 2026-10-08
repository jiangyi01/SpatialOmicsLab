"""
Command-line front door for the setup wizard.

``sog-setup`` (and ``python -m sog_install``) resolve here. Three
commands share one binary:

* **(default) / ``run``** — the guided wizard (:mod:`wizard`). All the run flags
  live on this command.
* **``doctor``** — a read-only health report for an existing base env
  (delegates to :func:`doctor.main`).
* **``capture``** — freeze the live tool envs into committed specs
  (delegates to :func:`capture.main`), run once on a host that already has them.
* **``pack`` / ``unpack``** — write / restore a one-file transplant bundle of the
  whole deployment, user-created tools included (delegate to :func:`bundle.main`).

The subcommand is peeled off the front of ``argv`` and its tail handed to the
owning module's own ``main`` — so ``doctor``/``capture`` keep their flags in one
place and never drift from this file.

Stdlib only.
"""

from __future__ import annotations

import argparse
import sys

from . import wizard
from .answers import AnswersError
from .prompts import PromptError
from .wizard import WizardOptions

_EPILOG = (
    "subcommands:\n"
    "  sog-setup            run the guided setup wizard (default)\n"
    "  sog-setup doctor     read-only health report for a base env\n"
    "  sog-setup conncheck  verify base-env <-> MCP <-> ST-Coscientist are wired together\n"
    "  sog-setup capture    freeze live tool envs into committed specs\n"
    "  sog-setup chat       start the ST-Coscientist terminal chat with the tools wired\n"
    "  sog-setup pack       write a one-file transplant bundle (code + user tools + env recipes)\n"
    "  sog-setup unpack     restore a bundle onto this machine\n"
    "  sog-setup reset      remove a run's tool envs + test artifacts, keep the logs\n\n"
    "the reproducible path is `sog-setup --answers scenario.yaml` (no prompts, no LLM)."
)


def _build_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sog-setup",
        description=(
            "Set up SpatialOmicsLab from a fresh clone: connect your LLM, pick the "
            "analysis tools you want, build one conda env per tool, and test each on "
            "a tiny dataset — resumable, and non-destructive to your running project."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--answers",
        metavar="FILE.yaml",
        help="scripted answers — runs with no prompts and no LLM (the reproducible/CI path)",
    )
    resume_grp = ap.add_mutually_exclusive_group()
    resume_grp.add_argument("--resume", action="store_true", help="continue a previous run without asking")
    resume_grp.add_argument("--restart", action="store_true", help="archive a previous run and start clean")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and make no changes (best paired with --answers)",
    )
    ap.add_argument("--only", metavar="A,B", help="restrict provisioning + testing to these server keys")
    ap.add_argument(
        "--yes",
        action="store_true",
        help="auto-confirm every propose→execute gate (other prompts stay interactive; "
        "use --answers for a fully unattended run)",
    )
    ap.add_argument("--no-net", action="store_true", help="skip the network preflight probe")
    ap.add_argument(
        "--keep-agent-config",
        action="store_true",
        help="don't rewrite agent/MCP_server/mcp_config.yaml — leave the agent config as-is "
        "(the install-aware install/recipes/mcp_config.setup.yaml is still written)",
    )
    ap.add_argument(
        "--skip-install",
        action="store_true",
        help="skip building tool envs and go straight to the ST-Coscientist chat "
        "(a small agent-core env is still built if none exists)",
    )
    return ap


def _run(argv: list[str]) -> int:
    args = _build_run_parser().parse_args(argv)
    opts = WizardOptions(
        answers_path=args.answers,
        resume=args.resume,
        restart=args.restart,
        dry_run=args.dry_run,
        assume_yes=args.yes,
        only=[s.strip() for s in (args.only or "").split(",") if s.strip()],
        check_net=not args.no_net,
        keep_agent_config=args.keep_agent_config,
        skip_install=args.skip_install,
    )
    try:
        return wizard.run(opts)
    except AnswersError as exc:
        # A malformed/missing --answers file (missing keys, bad YAML, unset ${ENV}, unknown provider).
        print(f"❌ answers file problem — {exc}", file=sys.stderr)
        return 2
    except PromptError as exc:
        # A prompt with no scripted answer + no default under --answers/non-interactive.
        print(f"❌ needed input that the answers file didn't provide: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted — state saved; re-run `sog-setup` to resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Last-resort catch-all: any UNEXPECTED error (an OSError from a full/read-only disk, or
        # anything wizard.run didn't already map to CondaError→3) must surface as a friendly one-liner
        # + a nonzero exit + a resume hint — never a raw traceback on a user's deploy box. Wizard
        # phases persist their state as they advance, so re-running resumes. ``redact`` masks any
        # secret an error string might echo. (KeyboardInterrupt/SystemExit are BaseException, so the
        # arms above still take precedence — this never swallows a Ctrl-C.)
        from .session_log import redact

        print(
            f"❌ unexpected error ({type(exc).__name__}): {redact(str(exc))}\n"
            "   your progress is saved — re-run `sog-setup` to resume, or `sog-setup doctor` to check the env.",
            file=sys.stderr,
        )
        return 1


def _run_subcommand(head: str, tail: list[str]) -> int:
    """Dispatch a subcommand (``doctor``/``conncheck``/``capture``/``reset``/``chat``) with the same friendly
    failure handling the wizard path already gets in :func:`_run`. Without this a KeyboardInterrupt during a
    subcommand dumps a raw traceback + exit 1 instead of the clean ``interrupted…`` + 130 — and, worse,
    an ordinary ``CondaError`` (``doctor``/``capture`` on a box with no conda/mamba on PATH; the wizard
    actively steers users to ``sog-setup doctor``) unwinds to a raw traceback because ``doctor.main``
    and ``capture.main`` wrap nothing. So catch here at the single dispatch chokepoint, covering all
    three subcommands at once: a conda problem → exit 3 (matching ``reset.main``'s own CondaError→3),
    anything else → a friendly one-liner + exit 1 (mirroring ``_run``'s catch-all). ``redact`` masks any
    secret an error string might echo. ``KeyboardInterrupt``/``SystemExit`` are ``BaseException`` so the
    Ctrl-C arm still takes precedence and neither this nor the generic arm ever swallows them."""
    from .envtools import CondaError
    from .session_log import redact

    try:
        if head == "doctor":
            from . import doctor

            return doctor.main(tail)
        if head == "conncheck":
            from . import conncheck

            return conncheck.main(tail)
        if head == "capture":
            from . import capture

            return capture.main(tail)
        if head == "chat":
            from . import launch

            return launch.main(tail)
        if head in ("pack", "unpack"):
            from . import bundle

            return bundle.main([head, *tail])  # bundle.main peels the verb itself
        from . import reset

        return reset.main(tail)
    except KeyboardInterrupt:
        print(f"\ninterrupted — stopped `sog-setup {head}`.", file=sys.stderr)
        return 130
    except CondaError as exc:
        print(
            f"❌ sog-setup {head}: {redact(str(exc))}\n"
            "   check that conda/mamba/micromamba is installed and on your PATH, then re-run.",
            file=sys.stderr,
        )
        return 3
    except Exception as exc:  # last-resort: a friendly line + nonzero, never a raw traceback on a deploy box
        print(
            f"❌ sog-setup {head} hit an unexpected error ({type(exc).__name__}): {redact(str(exc))}",
            file=sys.stderr,
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        head = argv[0]
        if head in ("doctor", "conncheck", "capture", "reset", "chat", "pack", "unpack"):
            return _run_subcommand(head, argv[1:])
        if head == "run":  # explicit alias for the default command
            argv = argv[1:]
    return _run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
