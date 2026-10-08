"""Worker for the two 3D-diagnosis portals.

Runs on the agent environment's interpreter and dispatches to
``spatialomicsgym.spatial3d.pipelines``. Standard output carries the payload and nothing else, so
everything else goes to standard error for the whole run: one stray line from a numerical library
is otherwise indistinguishable from the result.

A refusal is a normal outcome here. No section column, no coordinates, no known section order --
each comes back as a payload naming what is missing and what would supply it, so the model's next
move is obvious rather than a guess.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

TOOL = "spatial3d"


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose a stack of serial sections.")
    p.add_argument("--task", required=True, choices=("diagnose", "inspect", "contract", "adapters"))
    p.add_argument("--data-path", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--slice-key", default="")
    p.add_argument("--z-key", default="")
    p.add_argument("--slice-order", default="")
    p.add_argument("--coords-key", default="spatial")
    p.add_argument("--n-genes", type=int, default=8)
    p.add_argument("--skip-expression", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    from worker_utils import WorkerOutput

    args = _parse(argv)
    failure: BaseException | None = None
    stdout, sys.stdout = sys.stdout, sys.stderr  # payload on the real stdout, everything else aside
    try:
        from spatialomicsgym.spatial3d import pipelines

        payload: dict[str, Any]
        if args.task == "diagnose":
            payload = pipelines.diagnose_3d_stack(
                args.data_path,
                args.output_dir or os.path.join(os.getcwd(), "work", "spatial3d"),
                slice_key=args.slice_key,
                z_key=args.z_key,
                slice_order=args.slice_order,
                coords_key=args.coords_key,
                n_genes=args.n_genes,
                skip_expression=args.skip_expression,
            )
        elif args.task == "inspect":
            payload = pipelines.inspect_3d_coordinates(args.data_path, coords_key=args.coords_key)
        elif args.task == "contract":
            payload = pipelines.explain_3d_contract()
        else:
            payload = pipelines.list_aligner_adapters()
    except Exception as exc:  # a crash still has to leave a payload the caller can read
        # The exception is carried out of the handler, so the traceback has to come with it:
        # traceback.format_exc() is empty once the handler has exited and the report would say
        # the literal "NoneType: None".
        payload = {"status": "error", "tool": TOOL, "error": f"{type(exc).__name__}: {exc}"}
        failure = exc
    finally:
        sys.stdout = stdout

    if payload.get("status") != "ok":
        # A refusal's hints -- the obs columns to pick a slice_key from, the obsm keys to pick a
        # coords_key from -- were dropped here, because emit_error carries a message and nothing
        # else; they now ride in the message (hunt 2026-09-30, u21-3d-22).
        message = str(payload.get("error", "the task did not complete"))
        hints = [f"{k}: {v}" for k, v in payload.items() if k not in ("status", "tool", "error")]
        if hints:
            message = f"{message} ({'; '.join(hints)})"
        # A static method taking the tool FIRST, and the exception must be handed over: once the
        # except handler has exited, traceback.format_exc() is empty and the report would carry the
        # literal "NoneType: None" in place of the traceback.
        WorkerOutput.emit_error(TOOL, message, task=args.task, exc=failure)
        return 1

    out = WorkerOutput(TOOL, args.task)
    out.set_data(**(payload.get("data") or {}))
    out.add_output_files(payload.get("output_files") or {})
    out.add_params(
        {
            "task": args.task,
            "data_path": args.data_path,
            "slice_key": args.slice_key,
            "z_key": args.z_key,
            "slice_order": args.slice_order,
            "coords_key": args.coords_key,
            "n_genes": args.n_genes,
            "skip_expression": bool(args.skip_expression),
        }
    )
    out.set_summary(headline=payload.get("summary", ""))
    if payload.get("next_step"):
        out.set_analysis(str(payload["next_step"]))
    for note in (payload.get("data") or {}).get("questions") or []:
        out.add_warning(str(note))
    out.emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
