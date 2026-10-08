"""
``python -m spatialomicsgym.report <results_dir> [...]`` -- write ``report.html`` for runs that
already have a ``manifest.json``.

The engine writes the report as part of a run; this is the by-hand door for the other cases: a run
that finished before the report generator existed, a directory copied off a cluster, or a manifest
someone edited. With ``--all`` it walks a results ROOT and writes one report per run found. With no
path at all it walks this install's own output roots -- the same pair the web portal browses -- so
"report on everything I have run" needs no knowledge of where a run's output landed.

Exit status: 0 when every run found got its report and every root was searched whole; 1 when no run
was found, when some reports were not written, or when a directory could not be read or a walk
stopped at its directory budget; 2 when not one report was written. The reasons go to stderr.

Stdlib only, no plotting, no agent -- runs in the 1.6 GB minimal env.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .discover import MAX_DIRS, find_runs, normalized_roots
from .manifest import ManifestError
from .render import write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spatialomicsgym.report",
        description="Render report.html from a post-analysis manifest.json.",
        epilog=(
            "exit status: 0 every report written and every root searched; 1 no run found, some reports "
            "not written, or a directory unread or a walk stopped early; 2 no report written at all. "
            "Details go to stderr."
        ),
    )
    parser.add_argument(
        "path",
        nargs="*",
        help=(
            "a results directory (or, with --all, a results root). "
            "Omit it to report on every run under this install's own output roots."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="treat each PATH as a root and write a report for every run under it",
    )
    args = parser.parse_args(argv)

    # No PATH means "this install's results", which is the question a user who has just run the
    # agent actually has. Requiring them to name the roots first means knowing that a single run
    # splits across two of them -- exactly the knowledge ``paths`` exists to hold in one place, and
    # the same pair the web portal browses, so the two doors show the same set of runs.
    roots_given = list(args.path)
    walk_roots = bool(args.all) or not roots_given
    if not roots_given:
        from spatialomicsgym.chat_cli import _load_env_files
        from spatialomicsgym.paths import results_search_roots

        # The .env the other front doors read, so SOG_PATH / SOG_RESULTS_ROOTS set there count here.
        _load_env_files(".env")

        # With the agent's own root. Generated code writes under <SOG_PATH or ./data>/
        # spatialomicsgym_data/outputs, and called bare this searched only the tool work root, so
        # "report on everything I have run" found nothing the CLI's run card and the portal list
        # (u17-cli-report-17). The same default the agent constructor takes.
        data = os.environ.get("SOG_PATH") or os.environ.get("SOG_DATA_PATH") or "./data"
        roots_given = results_search_roots(os.path.join(os.path.abspath(data), "spatialomicsgym_data"))

    targets: list[Path] = []
    denied: set[str] = set()
    stopped: set[str] = set()
    if walk_roots:
        for root in normalized_roots(roots_given):
            # NOT the default `limit`. That default is MAX_RUNS, the portal's page size: it exists
            # because 300 cards is a lot for one HTTP response, and this door renders no page --
            # it writes one file per run for a command the user typed and is waiting on. Capping
            # here means writing 300 reports for 400 runs and exiting 0. MAX_DIRS is the honest
            # bound, because every run directory is itself visited before it is offered, so the
            # walk cannot reach more runs than it visits directories; the heap stays bounded by
            # the same ceiling that already bounds the walk.
            targets.extend(
                root / run["dir"]
                for run in find_runs(
                    [root],
                    limit=MAX_DIRS,
                    on_error=lambda e: denied.add(str(getattr(e, "filename", None) or e)),
                    on_limit=lambda _visited, root=root: stopped.add(str(root)),
                )
            )
    else:
        targets = [Path(p) for p in roots_given]
    # Runs under a directory this process cannot open are simply absent from the walk. Reporting
    # "no runs were found" for a root whose runs are all behind one is wrong in the direction that
    # costs the most: it reads as "nothing has been run here" and sends the user to re-run it.
    for path in sorted(denied):
        print(f"warning: could not read {path} -- any runs inside it were skipped", file=sys.stderr)
    # Same harm, other cause: the walk bounds how many directories it will visit so it cannot run
    # away on a network mount, and a root big enough to reach that bound is abandoned part-way.
    # Whatever it had not reached gets no report and, without this, no mention either.
    for path in sorted(stopped):
        print(
            f"warning: the scan of {path} stopped after {MAX_DIRS} directories -- "
            "the rest were not searched and have no report. Point this at a narrower directory.",
            file=sys.stderr,
        )
    if not targets:
        print("no runs with a manifest.json were found", file=sys.stderr)
        return 1

    failures = 0
    for target in targets:
        try:
            out = write_report(target)
        except ManifestError as exc:
            print(f"skipped {target}: {exc}", file=sys.stderr)
            failures += 1
        except OSError as exc:
            print(f"could not write a report in {target}: {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:  # one malformed run must not end a --all batch (u17-cli-report-16)
            print(f"could not render {target}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(out)
    # Partial failure is failure. This returned 0 unless EVERY target failed, so 49 unwritten
    # reports out of 50 -- or a root abandoned at MAX_DIRS, or a subtree it could not open -- exited
    # 0, and a caller checking $? believed every report was written; the warnings above went only to
    # stderr (hunt 2026-09-30, u17-cli-report-24). 2 keeps "nothing was written" apart from "some".
    if failures == len(targets):
        return 2
    return 1 if failures or denied or stopped else 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
