#!/usr/bin/env python3
"""
result_collector.py - Collect, evaluate, and summarize SpatialOmicsLab MCP tool results.

This module:
1. Collects JSON results from individual tool runs
2. Runs evaluation (with optional gold standard) on each result
3. Generates a comprehensive summary report (Markdown + JSON)

Usage:

    from result_collector import ResultCollector

    collector = ResultCollector(output_dir="/workspace/work/benchmark_run")

    # After each tool run, add the result:
    collector.add_result(tool_result_dict)

    # When all tools are done, generate the report:
    report = collector.generate_report()
    collector.save_report()

Or as a standalone script to re-evaluate existing results:

    python result_collector.py --results-dir /workspace/work/benchmark_run
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from typing import Any

# Ensure tools dir is on path
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from worker_utils import default_output_dir, json_safe

# Task type classification for tools
TOOL_TASK_TYPES = {
    # Clustering / spatial domain
    "scanpy_spatial": "clustering",
    "miso": "clustering",
    "deepst": "clustering",
    "graphst": "clustering",  # has both clustering and deconvolution
    "stagate": "clustering",
    "cellcharter": "clustering",
    "stlearn": "clustering",
    "prost": "clustering",  # has both svg and clustering
    "spaceflow": "clustering",
    "stage": "other",  # generation/recovery - no standard eval
    # SVG identification
    "hotspot": "svg",
    "somde": "svg",
    "spatialde": "svg",
    "spagft": "svg",
    "svgbit": "svg",
    # Deconvolution / cell-type mapping
    "cell2location": "deconvolution",
    "tangram": "deconvolution",
    "tacco": "deconvolution",
    "stride": "deconvolution",
    "bayestme": "deconvolution",
    "starfysh": "deconvolution",
    "ucdeconvolve": "deconvolution",
    "spatialprompt": "deconvolution",
    # Communication / other
    "commot": "other",
    "moscot": "other",
    "paste": "other",
    "st_gears": "other",
    "istar": "other",
    "spotgf": "other",
    "xfuse": "other",
    "seurat": "other",
    "mist": "other",
    "spaotsc": "other",
}


class ResultCollector:
    """Collect and summarize MCP tool run results."""

    def __init__(self, output_dir: str = default_output_dir("benchmark_results")):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: list[dict[str, Any]] = []
        self.start_time = time.time()

    def add_result(
        self,
        result: dict[str, Any],
        gold_standard: dict[str, Any] | None = None,
        run_eval: bool = True,
    ) -> dict[str, Any]:
        """
        Add a tool result and optionally run evaluation.

        Args:
            result: JSON dict from a worker (with status, tool, output_files, etc.)
            gold_standard: Optional gold-standard data for evaluation
            run_eval: Whether to run evaluation metrics

        Returns:
            The result dict with evaluation attached (if run_eval=True).
        """
        tool = result.get("tool", "unknown")
        task_type = self._infer_task_type(result)

        if run_eval and "evaluation" not in result:
            try:
                from eval_metrics import evaluate_tool_output

                eval_result = evaluate_tool_output(
                    tool_name=tool,
                    task_type=task_type,
                    output_dir=result.get("output_dir", ""),
                    output_files=result.get("output_files", {}),
                    summary=result.get("summary", {}),
                    gold_standard=gold_standard,
                )
                result["evaluation"] = eval_result
            except Exception as e:
                result["evaluation"] = {"error": str(e), "task_type": task_type}

        result["_collected_at"] = datetime.now().isoformat()
        self.results.append(result)

        # Save individual result
        result_file = os.path.join(self.output_dir, f"{tool}_result.json")
        with open(result_file, "w") as f:
            json.dump(json_safe(result), f, indent=2, default=str)

        return result

    def _infer_task_type(self, result: dict[str, Any]) -> str:
        """Infer task type from tool name and task field."""
        tool = result.get("tool", "")
        task = result.get("task", "")

        # Check explicit task-type mappings
        if tool in TOOL_TASK_TYPES:
            return TOOL_TASK_TYPES[tool]

        # Infer from task name
        task_lower = task.lower()
        if any(kw in task_lower for kw in ("cluster", "domain", "segment")):
            return "clustering"
        if any(kw in task_lower for kw in ("svg", "variable_gene", "spatially_variable")):
            return "svg"
        if any(kw in task_lower for kw in ("deconv", "cell_type", "annotation", "mapping")):
            return "deconvolution"
        return "other"

    def generate_report(self) -> dict[str, Any]:
        """Generate a comprehensive summary report."""
        elapsed = time.time() - self.start_time

        report: dict[str, Any] = {
            "title": "SpatialOmicsLab MCP Tools Benchmark Report",
            "generated_at": datetime.now().isoformat(),
            "total_elapsed_sec": round(elapsed, 1),
            "n_tools_run": len(self.results),
            "n_success": sum(1 for r in self.results if r.get("status") == "ok"),
            "n_failed": sum(1 for r in self.results if r.get("status") != "ok"),
            "by_task_type": {},
            "tool_results": [],
        }

        # Group by task type
        by_type: dict[str, list[dict]] = {}
        for r in self.results:
            tt = self._infer_task_type(r)
            by_type.setdefault(tt, []).append(r)

        for tt, results in by_type.items():
            type_summary = self._summarize_task_type(tt, results)
            report["by_task_type"][tt] = type_summary

        # Per-tool summary
        for r in self.results:
            tool_summary = {
                "tool": r.get("tool"),
                "task": r.get("task"),
                "status": r.get("status"),
                "data": r.get("data"),
            }
            ev = r.get("evaluation", {})
            if ev:
                tool_summary["eval_metrics"] = ev.get("metrics", {})
                tool_summary["eval_interpretation"] = ev.get("interpretation", "")
            report["tool_results"].append(tool_summary)

        return report

    def _summarize_task_type(self, task_type: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize results for a task type."""
        summary: dict[str, Any] = {
            "n_tools": len(results),
            "tools": [r.get("tool") for r in results],
        }

        if task_type == "clustering":
            aris = []
            nmis = []
            for r in results:
                ev = r.get("evaluation", {}).get("metrics", {})
                if "ARI" in ev:
                    aris.append((r.get("tool"), ev["ARI"]))
                if "NMI" in ev:
                    nmis.append((r.get("tool"), ev["NMI"]))
            if aris:
                aris.sort(key=lambda x: x[1], reverse=True)
                summary["ARI_ranking"] = [{"tool": t, "ARI": a} for t, a in aris]
                summary["best_ARI"] = aris[0]
                summary["mean_ARI"] = round(sum(a for _, a in aris) / len(aris), 4)
                # The denominator travels with the number. ``n_tools`` above counts every result in
                # this section and the ``tools`` line names all of them, but a tool that crashed, or
                # whose evaluation declined to score, carries no ARI and is not in this mean. The two
                # counts are printed adjacent, so the one the mean was taken over has to be stated.
                summary["n_scored_ARI"] = len(aris)
            if nmis:
                summary["mean_NMI"] = round(sum(n for _, n in nmis) / len(nmis), 4)
                # Its own count, not ARI's: a result can carry one metric and not the other.
                summary["n_scored_NMI"] = len(nmis)

        elif task_type == "svg":
            f1s = []
            morans = []
            for r in results:
                ev = r.get("evaluation", {}).get("metrics", {})
                if "f1" in ev:
                    f1s.append((r.get("tool"), ev["f1"]))
                mi = ev.get("morans_i", {})
                if isinstance(mi, dict) and "mean_morans_i" in mi:
                    morans.append((r.get("tool"), mi["mean_morans_i"]))
            if f1s:
                f1s.sort(key=lambda x: x[1], reverse=True)
                summary["F1_ranking"] = [{"tool": t, "F1": f} for t, f in f1s]
                summary["best_F1"] = f1s[0]
                # A ranking with no mean under it is short in the same silent way: the table is the
                # only cue that two of the five named tools are missing from it.
                summary["n_scored_F1"] = len(f1s)
            if morans:
                morans.sort(key=lambda x: x[1], reverse=True)
                summary["morans_ranking"] = [{"tool": t, "mean_morans_i": m} for t, m in morans]
                summary["n_scored_morans_i"] = len(morans)

        elif task_type == "deconvolution":
            rmses = []
            pearsons = []
            for r in results:
                ev = r.get("evaluation", {}).get("metrics", {})
                if "RMSE" in ev:
                    rmses.append((r.get("tool"), ev["RMSE"]))
                if "mean_pearson_r" in ev and ev["mean_pearson_r"] is not None:
                    pearsons.append((r.get("tool"), ev["mean_pearson_r"]))
            if rmses:
                rmses.sort(key=lambda x: x[1])  # lower is better
                summary["RMSE_ranking"] = [{"tool": t, "RMSE": r} for t, r in rmses]
                summary["best_RMSE"] = rmses[0]
                summary["n_scored_RMSE"] = len(rmses)
            if pearsons:
                pearsons.sort(key=lambda x: x[1], reverse=True)
                summary["pearson_ranking"] = [{"tool": t, "pearson_r": p} for t, p in pearsons]
                # Its own count again, and a third value in general: the loop above additionally
                # drops a mean_pearson_r of None, which a tool that scored an RMSE can still have.
                summary["n_scored_pearson_r"] = len(pearsons)

        return summary

    def save_report(self, filename: str = "benchmark_report") -> str:
        """Save report as both JSON and Markdown."""
        report = self.generate_report()

        # JSON report. json_safe, not default=str: a metric that could not be computed arrives here
        # as a NaN, which json serialises to the bare token ``NaN`` -- valid to Python's reader and
        # to nothing else, taking the whole document down with it. _summarize_task_type guards
        # mean_pearson_r against None but not against NaN, so it reaches this line.
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w") as f:
            json.dump(json_safe(report), f, indent=2, default=str)

        # Markdown report
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        md = self._render_markdown(report)
        with open(md_path, "w") as f:
            f.write(md)

        # All results JSON
        all_path = os.path.join(self.output_dir, "all_results.json")
        with open(all_path, "w") as f:
            json.dump(json_safe(self.results), f, indent=2, default=str)

        print("Report saved to:", file=sys.stderr)
        print(f"  JSON: {json_path}", file=sys.stderr)
        print(f"  Markdown: {md_path}", file=sys.stderr)
        print(f"  All results: {all_path}", file=sys.stderr)

        return md_path

    def _render_markdown(self, report: dict[str, Any]) -> str:
        """Render the report as Markdown."""
        lines = []
        lines.append(f"# {report['title']}")
        lines.append("")
        lines.append(f"**Generated:** {report['generated_at']}")
        lines.append(f"**Total time:** {report['total_elapsed_sec']}s")
        lines.append(
            f"**Tools run:** {report['n_tools_run']} ({report['n_success']} success, {report['n_failed']} failed)"
        )
        lines.append("")

        def _of_run(key: str, summary: dict[str, Any]) -> str:
            """`` (over 3 of 5 tools)`` -- the denominator travels with the number it belongs to.

            The section header counts every result and the ``Tools:`` line names every one of them,
            but each mean and each ranking below covers only the results that carried that metric.
            Empty when the count was not recorded, so a hand-built summary still renders.
            """
            n = summary.get(f"n_scored_{key}")
            return f" (over {n} of {summary.get('n_tools', '?')} tools)" if n is not None else ""

        def _ranked(key: str, label: str, summary: dict[str, Any]) -> list[str]:
            """The same disclosure for the rankings that publish no mean at all.

            A short table is the only cue that three of the five tools named above it are missing
            from it, and a reader has to count rows to notice. Say it instead.
            """
            n = summary.get(f"n_scored_{key}")
            if n is None:
                return []
            return [f"*{n} of {summary.get('n_tools', '?')} tools ranked; the rest carried no {label}.*", ""]

        # By task type
        for tt, summary in report.get("by_task_type", {}).items():
            lines.append(f"## {tt.title()} Tools ({summary['n_tools']})")
            lines.append("")
            lines.append(f"Tools: {', '.join(summary['tools'])}")
            lines.append("")

            if "ARI_ranking" in summary:
                lines.append("### Clustering Quality (ARI Ranking)")
                lines.append("")
                lines.append("| Rank | Tool | ARI |")
                lines.append("|------|------|-----|")
                for i, item in enumerate(summary["ARI_ranking"], 1):
                    lines.append(f"| {i} | {item['tool']} | {item['ARI']} |")
                lines.append("")
                lines.append(f"**Mean ARI:** {summary.get('mean_ARI', 'N/A')}{_of_run('ARI', summary)}")
                if summary.get("mean_NMI"):
                    lines.append(f"**Mean NMI:** {summary['mean_NMI']}{_of_run('NMI', summary)}")
                lines.append("")

            if "F1_ranking" in summary:
                lines.append("### SVG Detection Quality (F1 Ranking)")
                lines.append("")
                lines.append("| Rank | Tool | F1 |")
                lines.append("|------|------|-----|")
                for i, item in enumerate(summary["F1_ranking"], 1):
                    lines.append(f"| {i} | {item['tool']} | {item['F1']} |")
                lines.append("")
                lines.extend(_ranked("F1", "F1", summary))

            if "morans_ranking" in summary:
                lines.append("### Spatial Autocorrelation (Moran's I)")
                lines.append("")
                lines.append("| Rank | Tool | Mean Moran's I |")
                lines.append("|------|------|----------------|")
                for i, item in enumerate(summary["morans_ranking"], 1):
                    lines.append(f"| {i} | {item['tool']} | {item['mean_morans_i']} |")
                lines.append("")
                lines.extend(_ranked("morans_i", "Moran's I", summary))

            if "RMSE_ranking" in summary:
                lines.append("### Deconvolution Quality (RMSE Ranking)")
                lines.append("")
                lines.append("| Rank | Tool | RMSE |")
                lines.append("|------|------|------|")
                for i, item in enumerate(summary["RMSE_ranking"], 1):
                    lines.append(f"| {i} | {item['tool']} | {item['RMSE']} |")
                lines.append("")
                lines.extend(_ranked("RMSE", "RMSE", summary))

            if "pearson_ranking" in summary:
                lines.append("### Deconvolution Correlation (Pearson r)")
                lines.append("")
                lines.append("| Rank | Tool | Pearson r |")
                lines.append("|------|------|-----------|")
                for i, item in enumerate(summary["pearson_ranking"], 1):
                    lines.append(f"| {i} | {item['tool']} | {item['pearson_r']} |")
                lines.append("")
                lines.extend(_ranked("pearson_r", "Pearson r", summary))

        # Per-tool detail
        lines.append("## Per-Tool Results")
        lines.append("")
        lines.append("| Tool | Task | Status | Key Metric | Interpretation |")
        lines.append("|------|------|--------|------------|----------------|")
        for tr in report.get("tool_results", []):
            tool = tr.get("tool", "?")
            task = tr.get("task", "?")
            status = tr.get("status", "?")
            metrics = tr.get("eval_metrics", {})
            interp = tr.get("eval_interpretation", "")[:80]

            # Pick best metric to show
            key_metric = ""
            if "ARI" in metrics:
                key_metric = f"ARI={metrics['ARI']}"
            elif "f1" in metrics:
                key_metric = f"F1={metrics['f1']}"
            elif "RMSE" in metrics:
                key_metric = f"RMSE={metrics['RMSE']}"
            elif "n_clusters" in metrics:
                key_metric = f"clusters={metrics['n_clusters']}"

            lines.append(f"| {tool} | {task} | {status} | {key_metric} | {interp} |")

        lines.append("")
        return "\n".join(lines)


def main():
    """CLI: re-evaluate results from a directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect and summarize SpatialOmicsLab tool results")
    parser.add_argument("--results-dir", required=True, help="Directory with *_result.json files")
    parser.add_argument("--output", default=None, help="Output dir for report (default: same as results-dir)")
    args = parser.parse_args()

    output_dir = args.output or args.results_dir
    collector = ResultCollector(output_dir=output_dir)

    # Load existing results
    for fname in sorted(os.listdir(args.results_dir)):
        if fname.endswith("_result.json"):
            with open(os.path.join(args.results_dir, fname)) as f:
                result = json.load(f)
            collector.results.append(result)
            print(f"Loaded: {fname}", file=sys.stderr)

    if not collector.results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    report_path = collector.save_report()
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
