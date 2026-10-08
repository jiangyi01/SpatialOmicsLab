"""Worker for the read-only visualization inspector.

Runs on the agent environment's interpreter, where anndata, numpy and pandas already are. It
opens a dataset backed, asks ``spatialomicsgym.viz`` what is there and what can be drawn, and
prints one JSON object. It writes no file and creates no directory.

Standard output belongs to the payload alone, so everything else goes to standard error for the
whole run and is restored afterwards. A stray progress line from a library is otherwise
indistinguishable from the result.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a dataset and report what can be drawn.")
    parser.add_argument("--task", required=True, choices=("inspect", "capabilities", "recommend", "validate"))
    parser.add_argument("--data-path", default="")
    parser.add_argument("--obs-keys", default="")
    parser.add_argument("--var-names", default="")
    parser.add_argument("--max-categories", type=int, default=200)
    parser.add_argument("--sample-spots", type=int, default=5000)
    parser.add_argument("--include-capabilities", action="store_true")
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--question", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--plot-id", default="")
    parser.add_argument("--params-json", default="")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    from spatialomicsgym.viz import pipelines

    if args.task == "inspect":
        if not args.data_path:
            raise ValueError("--data-path is required for this task")
        return pipelines.inspect_dataset(
            args.data_path,
            obs_keys=args.obs_keys,
            var_names=args.var_names,
            max_categories=args.max_categories,
            sample_spots=args.sample_spots,
            include_capabilities=bool(args.include_capabilities),
        )
    if args.task == "capabilities":
        return pipelines.list_visualization_capabilities(
            args.data_path, include_unsupported=bool(args.include_unsupported)
        )
    if args.task == "recommend":
        if not args.data_path:
            raise ValueError("--data-path is required for this task")
        return pipelines.recommend_visualizations(args.data_path, question=args.question, limit=args.limit)
    if not args.data_path or not args.plot_id:
        raise ValueError("--data-path and --plot-id are required for this task")
    return pipelines.validate_plot_request(args.data_path, args.plot_id, args.params_json)


def main(argv: list[str] | None = None) -> int:
    import json

    args = _parse(argv)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from worker_utils import WorkerOutput

        try:
            result = _run(args)
        except ImportError as exc:
            payload = WorkerOutput.error(
                "viz-inspector",
                f"the visualization package could not be imported here: {exc}",
                task=args.task,
                status="dep_missing",
            )
            payload["diagnostic"] = (
                "This portal runs on the agent environment's interpreter and needs the "
                "spatialomicsgym package importable there. Run `sog-setup doctor` to check that "
                "environment; no per-tool environment has to be built for this portal."
            )
            print(json.dumps(payload), file=real_stdout)
            return 1
        except Exception as exc:
            payload = WorkerOutput.error("viz-inspector", str(exc), task=args.task, exc=exc)
            print(json.dumps(payload), file=real_stdout)
            return 1

        out = WorkerOutput("viz-inspector", args.task)
        status = str(result.get("status") or "ok")
        if status != "ok":
            payload = WorkerOutput.error(
                "viz-inspector",
                str(result.get("error") or result.get("why") or "the request was refused"),
                task=args.task,
            )
            payload.update({k: v for k, v in result.items() if k not in ("status", "error")})
            print(json.dumps(payload), file=real_stdout)
            return 1

        out.add_params({k: v for k, v in vars(args).items() if v not in ("", None, False) and k != "task"})
        profile = result.get("profile") or {}
        dataset = profile.get("dataset") or {}
        if dataset:
            out.set_data(n_obs=dataset.get("n_obs"), n_vars=dataset.get("n_vars"))
        for key, value in result.items():
            if key == "status":
                continue
            out.add_extra(key, value)
        out.set_analysis(_analysis(args.task, result))
        print(json.dumps(out.to_dict()), file=real_stdout)
        return 0
    finally:
        sys.stdout = real_stdout


def _analysis(task: str, result: dict[str, Any]) -> str:
    """One paragraph the model reads. Says what was found, not what it means."""
    if task == "inspect":
        profile = result.get("profile") or {}
        dataset = profile.get("dataset") or {}
        derived = profile.get("derived") or {}
        rows = [r for r in (result.get("capabilities") or []) if r.get("available")]
        # A row that reads another tool's output is callable only once that output exists, so it is
        # not drawable "as it stands"; it was counted as if it were (hunt 2026-09-30, u20b-viz-rest-29).
        available = [r for r in rows if not r.get("needs_result")]
        pending = [r for r in rows if r.get("needs_result")]
        bits = [
            f"{dataset.get('n_obs', '?')} observations by {dataset.get('n_vars', '?')} variables",
            f"modality {derived.get('modality', 'unknown')}",
        ]
        matrix = profile.get("matrix") or {}
        if matrix.get("integral") is True:
            bits.append("X holds counts")
        elif matrix.get("has_negative"):
            bits.append(
                "X is scaled and holds negative values, so a counts layer is the right source for a magnitude plot"
            )
        elif matrix.get("integral") is None:
            bits.append("the matrix type could not be established from the sample")
        if available:
            bits.append(f"{len(available)} plot types can be drawn from it as it stands")
        if pending:
            bits.append(f"{len(pending)} more can be drawn once another tool's result table is passed in")
        return ". ".join(bits) + "."
    if task == "recommend":
        names = ", ".join(r["plot_id"] for r in (result.get("recommended") or []))
        return f"Suggested, in order: {names}." if names else "Nothing can be drawn from this dataset as it stands."
    if task == "validate":
        if result.get("ok"):
            return f"{result.get('plot_id')} can be drawn from this dataset."
        return f"{result.get('plot_id')} cannot be drawn: {result.get('why', '')}"
    rows = result.get("capabilities") or []
    return f"{len(rows)} plot types in the catalogue."


if __name__ == "__main__":
    raise SystemExit(main())
