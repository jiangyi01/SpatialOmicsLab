"""Worker for the spatial visualization portal.

Runs on the agent environment's interpreter and dispatches to ``spatialomicsgym.viz.pipelines``.
Standard output carries the payload and nothing else, so everything else is redirected to
standard error for the whole run: a stray line from a plotting library is otherwise
indistinguishable from the result.

A refusal is a normal outcome here, not a crash. A dataset with no coordinates, a differential
expression result with no p-values, a deconvolution that has not been run -- each comes back as
a payload naming what is missing, what would produce it and what can be drawn instead, so the
model's next move is obvious rather than a guess.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

TOOL = "spatial-viz"


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw a figure from a dataset or a result table.")
    p.add_argument(
        "--task",
        required=True,
        choices=(
            "spatial_expression",
            "spatial_annotation",
            "embedding",
            "qc",
            "markers",
            "differential",
            "deconvolution",
            "pathway",
            "organisation",
            "trajectory",
            "communication",
            "compose",
            "export",
            "pipeline",
            "update",
            "spatial_3d",
            "section_grid",
            "alignment_qc",
        ),
    )
    p.add_argument("--data-path", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--genes", default="")
    p.add_argument("--color", default="")
    p.add_argument("--obs-key", default="")
    p.add_argument("--obsm-key", default="")
    p.add_argument("--groupby", default="")
    p.add_argument("--group", default="")
    p.add_argument("--basis", default="auto")
    p.add_argument("--layer", default="")
    p.add_argument("--use-raw", action="store_true")
    p.add_argument("--library-id", default="")
    p.add_argument("--image-key", default="")
    p.add_argument("--image-alpha", type=float, default=1.0)
    p.add_argument("--spot-alpha", type=float, default=1.0)
    p.add_argument("--point-size", type=float, default=0.0)
    p.add_argument("--color-map", default="")
    p.add_argument("--vmin", default="")
    p.add_argument("--vmax", default="")
    p.add_argument("--normalize", default="auto")
    p.add_argument("--share-scale", action="store_true")
    p.add_argument("--split-panels", action="store_true")
    p.add_argument("--legend-loc", default="right margin")
    p.add_argument("--ncols", type=int, default=3)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument("--kind", default="")
    p.add_argument("--fc-threshold", type=float, default=1.0)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--compute-if-missing", action="store_true")
    p.add_argument("--on-tissue", action="store_true")
    p.add_argument("--proportions-csv", default="")
    p.add_argument("--title", default="")
    # "" is "not given"; every producer reads it as png, and the pipeline as png or, at publication
    # depth, svg. A default of "png" made an explicit png indistinguishable from none.
    p.add_argument("--figure-format", default="")
    p.add_argument("--dpi", type=int, default=0)
    p.add_argument("--figure-id", default="")
    p.add_argument("--figure-spec", default="")
    p.add_argument("--standardize", action="store_true")
    p.add_argument("--split-by", default="")
    p.add_argument("--results-path", default="")
    p.add_argument("--collection", default="")
    p.add_argument("--method", default="")
    p.add_argument("--pathways", default="")
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--n-neighbors", type=int, default=6)
    p.add_argument("--pseudotime-key", default="")
    p.add_argument("--n-bins", type=int, default=20)
    p.add_argument("--source-column", default="")
    p.add_argument("--target-column", default="")
    p.add_argument("--score-column", default="")
    p.add_argument("--figure-specs", default="")
    p.add_argument("--export-format", default="svg")
    p.add_argument("--include-values", action="store_true")
    p.add_argument("--bundle", action="store_true")
    p.add_argument("--depth", default="overview")
    p.add_argument("--question", default="")
    p.add_argument("--set-params", default="")
    p.add_argument("--unset-params", default="")
    p.add_argument("--allow-reread", action="store_true")
    # Three dimensions. --z-spacing has no default that means anything: 0.0 is "not given", and the
    # producer refuses rather than stacking sections at z = 0, 1, 2 when the real cut is irregular.
    p.add_argument("--view", default="scatter")
    p.add_argument("--axis", default="z")
    p.add_argument("--coords-key", default="")
    p.add_argument("--section-key", default="")
    p.add_argument("--z-spacing", type=float, default=0.0)
    p.add_argument("--before-key", default="")
    p.add_argument("--after-key", default="")
    p.add_argument("--pair", default="")
    p.add_argument("--mode", default="before_after")
    return p.parse_args(argv)


def _dispatch(args: argparse.Namespace) -> dict[str, Any]:
    from spatialomicsgym.viz import pipelines as P

    if args.task == "spatial_expression":
        return P.plot_spatial_expression(
            args.data_path,
            genes=args.genes,
            obs_key=args.obs_key,
            obsm_key=args.obsm_key,
            output_dir=args.output_dir,
            layer=args.layer,
            use_raw=args.use_raw,
            library_id=args.library_id,
            image_key=args.image_key,
            image_alpha=args.image_alpha,
            spot_alpha=args.spot_alpha,
            point_size=args.point_size,
            color_map=args.color_map,
            vmin=args.vmin,
            vmax=args.vmax,
            normalize=args.normalize,
            share_scale=args.share_scale,
            ncols=args.ncols,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "spatial_annotation":
        return P.plot_spatial_annotation(
            args.data_path,
            obs_key=args.obs_key,
            output_dir=args.output_dir,
            library_id=args.library_id,
            image_key=args.image_key,
            image_alpha=args.image_alpha,
            spot_alpha=args.spot_alpha,
            point_size=args.point_size,
            split_panels=args.split_panels,
            ncols=args.ncols,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "embedding":
        return P.plot_embedding(
            args.data_path,
            color=args.color,
            basis=args.basis,
            output_dir=args.output_dir,
            layer=args.layer,
            use_raw=args.use_raw,
            groupby=args.groupby,
            legend_loc=args.legend_loc,
            point_size=args.point_size,
            color_map=args.color_map,
            vmin=args.vmin,
            vmax=args.vmax,
            normalize=args.normalize,
            share_scale=args.share_scale,
            ncols=args.ncols,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "qc":
        return P.generate_qc_report(
            args.data_path,
            output_dir=args.output_dir,
            groupby=args.groupby,
            compute_if_missing=args.compute_if_missing,
            on_tissue=args.on_tissue,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "markers":
        return P.plot_marker_expression(
            args.data_path,
            genes=args.genes,
            groupby=args.groupby,
            output_dir=args.output_dir,
            kind=args.kind or "dotplot",
            layer=args.layer,
            use_raw=args.use_raw,
            normalize=args.normalize,
            standardize=args.standardize,
            top_n=args.top_n,
            split_by=args.split_by,
            color_map=args.color_map,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "pathway":
        return P.plot_pathway_results(
            args.results_path,
            data_path=args.data_path,
            output_dir=args.output_dir,
            kind=args.kind or "enrichment",
            collection=args.collection,
            method=args.method,
            pathways=args.pathways,
            obsm_key=args.obsm_key,
            top_n=args.top_n,
            fdr_threshold=args.fdr_threshold,
            library_id=args.library_id,
            ncols=args.ncols,
            color_map=args.color_map,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "organisation":
        return P.plot_spatial_statistics(
            args.data_path,
            results_path=args.results_path,
            output_dir=args.output_dir,
            kind=args.kind or "graph",
            obs_key=args.obs_key,
            n_neighbors=args.n_neighbors,
            top_n=args.top_n,
            library_id=args.library_id,
            color_map=args.color_map,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "trajectory":
        return P.plot_trajectory(
            args.data_path,
            output_dir=args.output_dir,
            kind=args.kind or "pseudotime",
            pseudotime_key=args.pseudotime_key,
            basis=args.basis,
            genes=args.genes,
            groupby=args.groupby,
            layer=args.layer,
            use_raw=args.use_raw,
            normalize=args.normalize,
            n_bins=args.n_bins,
            library_id=args.library_id,
            point_size=args.point_size,
            color_map=args.color_map,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "communication":
        return P.plot_cell_communication(
            args.results_path,
            output_dir=args.output_dir,
            kind=args.kind or "interactions",
            source_column=args.source_column,
            target_column=args.target_column,
            score_column=args.score_column,
            top_n=args.top_n,
            color_map=args.color_map,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "compose":
        return P.compose_figure(
            args.figure_specs,
            output_dir=args.output_dir,
            ncols=args.ncols,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "export":
        return P.export_visualization(
            args.figure_spec,
            output_dir=args.output_dir,
            data_path=args.data_path,
            export_format=args.export_format,
            dpi=args.dpi,
            include_values=args.include_values,
            bundle=args.bundle,
        )
    if args.task == "pipeline":
        return P.run_visualization_pipeline(
            args.data_path,
            output_dir=args.output_dir,
            depth=args.depth,
            question=args.question,
            # Passed through: rewriting "png" to "" made an explicit png ask at publication depth
            # come back as svg (hunt 2026-09-30, u20a-viz-pipelines-35).
            figure_format=args.figure_format,
            dpi=args.dpi,
        )
    if args.task == "differential":
        return P.plot_differential_expression(
            args.data_path,
            group=args.group,
            output_dir=args.output_dir,
            kind=args.kind or "volcano",
            top_n=args.top_n,
            fc_threshold=args.fc_threshold,
            p_threshold=args.p_threshold,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "deconvolution":
        return P.plot_deconvolution(
            args.data_path,
            obsm_key=args.obsm_key,
            proportions_csv=args.proportions_csv,
            output_dir=args.output_dir,
            kind=args.kind or "maps",
            library_id=args.library_id,
            ncols=args.ncols,
            top_n=args.top_n,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "spatial_3d":
        return P.plot_spatial_3d(
            args.data_path,
            view=args.view,
            genes=args.genes,
            obs_key=args.obs_key,
            layer=args.layer,
            use_raw=args.use_raw,
            coords_key=args.coords_key,
            section_key=args.section_key,
            z_spacing=args.z_spacing,
            axis=args.axis,
            n_bins=args.n_bins,
            output_dir=args.output_dir,
            point_size=args.point_size,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "section_grid":
        return P.plot_section_grid(
            args.data_path,
            genes=args.genes,
            obs_key=args.obs_key,
            layer=args.layer,
            use_raw=args.use_raw,
            section_key=args.section_key,
            ncols=args.ncols,
            output_dir=args.output_dir,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    if args.task == "alignment_qc":
        return P.plot_alignment_qc(
            args.data_path,
            mode=args.mode,
            coords_key=args.coords_key,
            before_key=args.before_key,
            after_key=args.after_key,
            section_key=args.section_key,
            pair=args.pair,
            output_dir=args.output_dir,
            title=args.title,
            figure_format=args.figure_format,
            dpi=args.dpi,
            figure_id=args.figure_id,
        )
    return P.update_visualization(
        args.figure_spec,
        set_params=args.set_params,
        unset_params=args.unset_params,
        output_dir=args.output_dir,
        data_path=args.data_path,
        allow_reread=args.allow_reread,
    )


def main(argv: list[str] | None = None) -> int:
    import json

    args = _parse(argv)
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        from worker_utils import WorkerOutput

        try:
            result = _dispatch(args)
        except ImportError as exc:
            payload = WorkerOutput.error(
                TOOL,
                f"the visualization package could not be imported here: {exc}",
                task=args.task,
                status="dep_missing",
            )
            payload["diagnostic"] = (
                "This portal runs on the agent environment's interpreter and needs the "
                "spatialomicsgym package importable there. No per-tool environment has to be "
                "built for it; run `sog-setup doctor` to check that environment."
            )
            print(json.dumps(payload), file=real_stdout)
            return 1
        except Exception as exc:
            payload = WorkerOutput.error(TOOL, str(exc), task=args.task, exc=exc)
            # A LayerError says what to do in .fix; reporting str(exc) alone dropped it (hunt
            # 2026-09-30, skeptic note on u20a-viz-pipelines-31).
            fix = getattr(exc, "fix", "")
            if isinstance(fix, str) and fix:
                payload["fix"] = fix
            print(json.dumps(payload), file=real_stdout)
            return 1

        status = str(result.get("status") or "ok")
        if status in ("refused", "needs_redraw", "error"):
            # A refusal is an outcome, not a crash, and it is reported as an error status so the
            # smoke driver grades it honestly -- while keeping every field the model needs to
            # make its next call.
            payload = WorkerOutput.error(
                TOOL,
                str(result.get("why") or result.get("error") or "the request was refused"),
                task=args.task,
                status="error" if status != "needs_redraw" else "needs_redraw",
            )
            payload.update({k: v for k, v in result.items() if k not in ("status",)})
            print(json.dumps(payload), file=real_stdout)
            return 1

        out = WorkerOutput(TOOL, args.task)
        out.add_params({k: v for k, v in vars(args).items() if v not in ("", None, False) and k != "task"})
        files: dict[str, Any] = {}
        if result.get("figure"):
            files["figure"] = str(result["figure"])
        if result.get("manifest"):
            files["manifest"] = str(result["manifest"])
        if result.get("output_dir"):
            files["output_dir"] = str(result["output_dir"])
        if files:
            out.add_output_files(files)
        out.set_summary(figure_id=result.get("figure_id", ""), plot_id=result.get("plot_id", ""))
        for message in result.get("warnings") or []:
            out.add_warning(str(message))
        for key, value in result.items():
            if key in ("status", "warnings", "figure", "manifest", "output_dir"):
                continue
            out.add_extra(key, value)
        out.set_analysis(str(result.get("caption") or ""))
        print(json.dumps(out.to_dict()), file=real_stdout)
        return 0
    finally:
        sys.stdout = real_stdout


if __name__ == "__main__":
    raise SystemExit(main())
