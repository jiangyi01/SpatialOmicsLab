"""
``python -m spatialomicsgym.postanalysis <source> [...]`` -- analyse output a tool has already
written.

The engine runs by itself at the end of an agent turn. This is the door for output that never came
from a turn: a benchmark sweep, a tool someone ran by hand, a directory copied off a cluster, a run
that finished before this package existed. Until now the only way in was to write Python --
``python -m spatialomicsgym.postanalysis`` answered "is a package and cannot be directly executed".

The sibling ``python -m spatialomicsgym.report`` is the by-hand door for *rendering* a manifest, and
could not help: writing the manifest is the step that had no door. So one command here runs both,
and a directory of tool output becomes figures, tables, a manifest and a page you can open.
``--no-report`` stops at the manifest for callers that only want the machine record.

Unlike the sibling this is not stdlib-only -- the analysis needs the dataframe and plotting stack.
The import is deferred into :func:`main` so ``--help`` does not pay for it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .manifest import DEFAULT_RESULTS_DIRNAME, STATUS_FAILED, TASK_TYPES


def _read_manifest(results_dir: Path) -> dict | None:
    """The manifest just written, or ``None`` if there is not one to read.

    Absent is a real answer, not an error: the benchmarking gate returns a path without writing, and
    an output directory that cannot be created is logged and returned the same way.
    """
    try:
        import json

        return json.loads((results_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m spatialomicsgym.postanalysis",
        description="Analyse a tool's output: write figures, tables, a manifest and a report.",
    )
    parser.add_argument("source", nargs="+", help="a file or directory a tool wrote")
    parser.add_argument(
        "--tool",
        metavar="NAME",
        help=(
            "the portal function that produced the output (run_spagcn). Lets the tool output "
            "registry supply the task type and the authoritative filenames instead of guessing "
            "from the data."
        ),
    )
    parser.add_argument(
        "--task",
        metavar="TYPE",
        choices=TASK_TYPES,
        help="override task detection outright. One of: " + ", ".join(TASK_TYPES),
    )
    parser.add_argument(
        "--results-dir",
        metavar="DIR",
        help=f"where to write. Defaults to <SOURCE>/{DEFAULT_RESULTS_DIRNAME}. One SOURCE only.",
    )
    parser.add_argument("--no-report", action="store_true", help="write the manifest but not report.html")
    args = parser.parse_args(argv)

    # One results directory cannot hold two runs: the second manifest would overwrite the first and
    # both would be reported written. Refused before anything is on disk, so it is a refusal and not
    # a half-done batch.
    if args.results_dir and len(args.source) > 1:
        parser.error("--results-dir takes a single SOURCE; the second run would overwrite the first")

    from .engine import _resolve_results_dir, run_post_analysis
    from .sources import leads_under_benchmark_results

    # The same collision arrives without ``--results-dir`` too, and by ordinary use: a results
    # directory defaults to the source's own folder, and a source that is a *file* uses its parent,
    # so ``postanalysis run/a.csv run/b.csv`` sends both to ``run/post_analysis``. Driven by hand it
    # printed that path twice, exited 0, and left one manifest describing only ``b.csv`` -- over
    # whatever analysis of ``a.csv`` was there before.
    #
    # The engine is asked where each source *would* go rather than the rule being restated here:
    # ``write=False`` exists for exactly this question, and a second spelling of a rule this package
    # already keeps in one place is the failure mode it has been bitten by before.
    destinations: dict[Path, str] = {}
    for raw in args.source:
        try:
            destination = _resolve_results_dir(Path(raw), args.results_dir, write=False)[0]
        except OSError:
            # Cannot say where it would go, so cannot claim it collides. The run itself will report
            # the problem; refusing the whole batch on a question we failed to ask would be worse.
            continue
        # A source under benchmarks/results is redirected to a fresh temp directory per run, so the
        # shared answer returned here is not where either of two such sources actually lands.
        if leads_under_benchmark_results(destination):
            continue
        if destination in destinations:
            parser.error(
                f"{destinations[destination]} and {raw} would both be analysed into {destination}, "
                "and only the last would survive; run them one at a time or pass --results-dir"
            )
        destinations[destination] = raw

    failures = 0
    for raw in args.source:
        source = Path(raw)
        try:
            results_dir = run_post_analysis(
                source, tool_name=args.tool, task_type=args.task, results_dir=args.results_dir
            )
        except Exception as exc:  # contract non-negotiable 4 says it does not raise
            # If it does anyway, that is this door's problem to report, not to propagate: the next
            # source in the batch is still worth analysing.
            print(f"post-analysis failed on {source}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        manifest = _read_manifest(results_dir)
        if manifest is None:
            # Printing the path and exiting 0 here would tell the shell an analysis exists where
            # ``ls`` finds nothing -- R80/A, one layer up. Both causes are legitimate, so this is a
            # report rather than a complaint, but it is not success.
            print(
                f"no analysis was written for {source}: there is no manifest in {results_dir}. "
                "Post-analysis writes nothing while benchmarking is enabled, so a scored run cannot "
                "be contaminated; an output directory that could not be created is in the log.",
                file=sys.stderr,
            )
            failures += 1
            continue

        print(results_dir)
        if manifest.get("status") == STATUS_FAILED:
            # The manifest exists and says why there is no analysis in it. The sibling CLI fails only
            # when *every* target failed, because ``--all`` walks a tree and most of what it finds is
            # legitimately not a run; here every source was named by hand and was meant to be
            # analysed, so one that was not is worth a non-zero exit and the reason on stderr.
            for warning in manifest.get("warnings") or ["the manifest does not say why"]:
                print(f"{source}: {warning}", file=sys.stderr)
            failures += 1

        if not args.no_report:
            from spatialomicsgym.report.manifest import ManifestError
            from spatialomicsgym.report.render import write_report

            try:
                # Written for a failed run too: the page is how a human reads the reason.
                print(write_report(results_dir))
            except (ManifestError, OSError) as exc:
                # The analysis is on disk and is the valuable half. A lost page is worth a message,
                # not a discarded run.
                print(f"analysed {source}, but could not write its report: {exc}", file=sys.stderr)
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
