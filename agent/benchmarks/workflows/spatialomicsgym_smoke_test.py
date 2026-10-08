"""Main orchestrator for SpatialOmicsLab-driven MCP smoke tests.

Loads enabled MCP tools from ``mcp_config.yaml``, runs each tool through the
SpatialOmicsLab STCoscientist agent, and collects structured results with pass/fail verdicts.

Usage::

    python -m benchmarks.workflows.spatialomicsgym_smoke_test --mode direct
    python -m benchmarks.workflows.spatialomicsgym_smoke_test --mode full --workers 3

``--workers`` above 1 runs each tool in its own process: the agent's stdout capture and its REPL
namespace are process-global, so agents sharing one process read and overwrite each other's.
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from benchmarks.config.benchmark_config import repository_root
from benchmarks.reporting.result_schema import ResultSchema
from benchmarks.strategies.base_strategy import BenchmarkResult
from benchmarks.turn_outcome import degrade_note, tool_was_invoked
from benchmarks.workflows.prompt_generator import classify_tool, generate_prompt, get_dataset_for_tool
from benchmarks.workflows.result_recorder import ResultRecorder, SmokeTestRecord

#: The agent tree (``<checkout>/agent``): ``MCP_server/`` and ``benchmarks/`` live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
#: The repository root: the agent's ``data/``, ``test/test_data`` and ``benchmark_outputs/`` live here.
REPO_ROOT = repository_root(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# 1. load_enabled_tools
# ---------------------------------------------------------------------------


def load_enabled_tools(mcp_config_path: str | Path) -> list[dict[str, Any]]:
    """Load enabled MCP server tools from a ``mcp_config.yaml`` file.

    Parameters
    ----------
    mcp_config_path : str | Path
        Path to the MCP configuration YAML file.

    Returns
    -------
    list[dict[str, Any]]
        List of tool info dicts with keys ``server_key``, ``spatialomicsgym_name``,
        ``description``, and ``parameters``.
    """
    config_path = Path(mcp_config_path)
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    tools: list[dict[str, Any]] = []
    for server_key, server_cfg in config.get("mcp_servers", {}).items():
        if not server_cfg.get("enabled", False):
            continue
        for tool_def in server_cfg.get("tools", []):
            tools.append(
                {
                    "server_key": server_key,
                    "spatialomicsgym_name": tool_def.get("spatialomicsgym_name", server_key),
                    "description": tool_def.get("description", ""),
                    "parameters": tool_def.get("parameters", {}),
                }
            )

    return tools


# ---------------------------------------------------------------------------
# 2. run_single_tool_smoke_test
# ---------------------------------------------------------------------------


def run_single_tool_smoke_test(
    tool_info: dict[str, Any],
    mode: str = "direct",
    project_root: str | Path = PROJECT_ROOT,
    mcp_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    llm: str = "gpt-5",
    temperature: float = 1.0,
    timeout_seconds: int = 1800,
) -> SmokeTestRecord:
    """Run a single MCP tool through the SpatialOmicsLab agent and return a test record.

    Parameters
    ----------
    tool_info : dict[str, Any]
        Tool descriptor as returned by :func:`load_enabled_tools`.
    mode : str
        ``"direct"`` (minimal prompt) or ``"full"`` (richer prompt).
    project_root : str | Path
        Agent tree used to resolve relative paths (``MCP_server/``, ``benchmarks/``); ``data/``,
        ``test/test_data`` and ``benchmark_outputs/`` are read from its repository root.
    mcp_config_path : str | Path | None
        Path to the MCP config; defaults to ``<project_root>/MCP_server/mcp_config.yaml``.
    output_dir : str | Path | None
        Directory for saving outputs.
    llm : str
        LLM model identifier for the STCoscientist agent.
    temperature : float
        Sampling temperature for the LLM.
    timeout_seconds : int
        Per-tool timeout in seconds.

    Returns
    -------
    SmokeTestRecord
        Completed record with status, logs, and timing information.
    """
    from spatialomicsgym.config import default_config

    project_root = Path(project_root)
    repo_root = repository_root(project_root)
    if mcp_config_path is None:
        mcp_config_path = project_root / "MCP_server" / "mcp_config.yaml"
    if output_dir is None:
        output_dir = repo_root / "benchmark_outputs" / "smoke_test"

    tool_name = tool_info["spatialomicsgym_name"]
    record = SmokeTestRecord(tool_name=tool_name, dataset_name="", mode=mode)
    start = time.time()

    try:
        # -- dataset & prompt --------------------------------------------------
        tool_params = tool_info.get("parameters", {})
        dataset_info = get_dataset_for_tool(tool_name, tool_params=tool_params, project_root=str(project_root))
        # The dataset category the prompt was built from. split("/")[-2] of the output dir was the
        # tool's own name in every row (hunt 2026-09-30, u33a-bench-runner-2).
        record.dataset_name = classify_tool(tool_name, tool_params)

        prompt = generate_prompt(
            tool_name=tool_name,
            description=tool_info.get("description", ""),
            params=tool_params,
            dataset=dataset_info,
            mode=mode,
        )
        record.prompt = prompt

        # -- configure agent ---------------------------------------------------
        default_config.temperature = temperature
        default_config.llm = llm
        default_config.timeout_seconds = timeout_seconds

        data_dir = repo_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # One tool's REPL bindings (adata, output_dir) must not reach the next: with one worker every
        # tool runs in this process, one after another, in the one process-global namespace
        # (hunt 2026-09-30, u33a-bench-runner-3, review). benchmark_runner resets between pairs too.
        from spatialomicsgym.tool.support_tools import reset_repl_namespace

        reset_repl_namespace()

        from spatialomicsgym.agent.stcoscientist import STCoscientist

        agent = STCoscientist(path=str(data_dir), llm=llm)

        # -- register MCP tools -----------------------------------------------
        agent.add_mcp(str(mcp_config_path))

        # -- run agent with captured output ------------------------------------
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            log, response = agent.go(prompt)

        record.stdout_log = stdout_buf.getvalue()
        record.stderr_log = stderr_buf.getvalue()
        record.spatialomicsgym_call_record = {"log_length": len(log), "response_preview": str(response)[:2000]}

        # -- analyse results ---------------------------------------------------
        # This verdict could not fail (hunt 2026-09-30, u33a-bench-runner-2). "Invoked" was the
        # tool's name anywhere in the log -- whose first entry is the prompt, which names the tool
        # -- and a failure phrase only counted when the answer held none of "output", "results",
        # "saved"..., which nearly every answer does. A turn a 429 stopped before the model said a
        # word was recorded PASS, and summary.md declared the integration ready. The verdict now
        # reads what the agent did (benchmarks/turn_outcome.py) and whether the turn finished, and
        # judges the answer a person would read rather than the ReAct wrapper around it.
        from spatialomicsgym import clean_answer

        degraded = degrade_note(agent)
        answer_text = clean_answer(str(response)).lower()

        # Explicit failure phrases in the agent's answer (not the shared log)
        _failure_indicators = (
            "failed to",
            "could not complete",
            "exception occurred",
            "tool execution failed",
            "unable to execute",
            "fatal error",
        )
        has_failure = any(phrase in answer_text for phrase in _failure_indicators)

        if degraded:
            record.status = "DEGRADED"
            record.error_message = f"The turn did not finish: {degraded}"
        elif not tool_was_invoked(agent, log, tool_name):
            record.status = "FAIL_PROMPT"
            record.error_message = f"Tool '{tool_name}' was never invoked by the agent."
        elif has_failure:
            record.status = "FAIL_TOOL"
            record.error_message = "Explicit failure detected in agent response."
        else:
            record.status = "PASS"

        record.parsed_output = {"response": str(response)[:5000]}

    except TimeoutError:
        record.status = "TIMEOUT"
        record.error_message = f"Tool execution exceeded {timeout_seconds}s timeout."
        record.error_traceback = traceback.format_exc()
    except Exception as exc:
        record.status = "FAIL_MCP"
        record.error_message = str(exc)
        record.error_traceback = traceback.format_exc()
    finally:
        record.elapsed_seconds = time.time() - start

    return record


# ---------------------------------------------------------------------------
# 3. generate_summary_md
# ---------------------------------------------------------------------------


def generate_summary_md(results: list[SmokeTestRecord], run_id: str) -> str:
    """Generate a Markdown summary of smoke test results.

    Parameters
    ----------
    results : list[SmokeTestRecord]
        Completed smoke test records.
    run_id : str
        Unique run identifier.

    Returns
    -------
    str
        Markdown-formatted summary.
    """
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = total - passed

    lines: list[str] = [
        "# SpatialOmicsLab MCP Smoke Test Summary",
        "",
        f"**Run ID:** `{run_id}`  ",
        f"**Timestamp:** {datetime.now().isoformat()}  ",
        f"**Total:** {total} | **Passed:** {passed} | **Failed:** {failed}  ",
        "",
        "## Results",
        "",
        "| Tool | Status | Elapsed (s) | Dataset | Error |",
        "|------|--------|-------------|---------|-------|",
    ]

    for r in results:
        error_preview = r.error_message[:80].replace("|", "/") if r.error_message else ""
        lines.append(f"| {r.tool_name} | {r.status} | {r.elapsed_seconds:.1f} | {r.dataset_name} | {error_preview} |")

    # Failure details
    failures = [r for r in results if r.status != "PASS"]
    if failures:
        lines.append("")
        lines.append("## Failure Details")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.tool_name} ({r.status})")
            lines.append("")
            lines.append(f"**Error:** {r.error_message}")
            lines.append("")
            if r.error_traceback:
                lines.append("```")
                lines.append(r.error_traceback[:1000])
                lines.append("```")
            lines.append("")

    # Readiness assessment
    lines.append("## Readiness Assessment")
    lines.append("")
    if total == 0:
        lines.append("No tools were tested.")
    elif passed == total:
        lines.append("All tools passed smoke testing. The MCP integration is ready for full benchmarks.")
    elif passed / total >= 0.8:
        lines.append(
            f"{passed}/{total} tools passed ({100 * passed / total:.0f}%). "
            f"Most tools are operational; review failures before proceeding."
        )
    else:
        lines.append(
            f"Only {passed}/{total} tools passed ({100 * passed / total:.0f}%). "
            f"Significant issues detected -- investigate failures before running full benchmarks."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. generate_summary_html
# ---------------------------------------------------------------------------


def generate_summary_html(md_content: str) -> str:
    """Wrap Markdown content in a basic HTML page with table styling.

    Parameters
    ----------
    md_content : str
        Markdown-formatted text.

    Returns
    -------
    str
        HTML document string.
    """
    # Escape HTML special chars in the markdown body
    escaped = md_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SpatialOmicsLab Smoke Test Summary</title>
<style>
  body {{ font-family: sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.8rem; text-align: left; }}
  th {{ background: #f4f4f4; }}
  pre {{ background: #f8f8f8; padding: 1rem; overflow-x: auto; }}
  code {{ background: #eee; padding: 0.1rem 0.3rem; border-radius: 3px; }}
</style>
</head>
<body>
<pre>{escaped}</pre>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 5. run_spatialomicsgym_smoke_test  (main entry point)
# ---------------------------------------------------------------------------


def _run_tools(
    tools: list[dict[str, Any]],
    workers: int,
    kwargs: dict[str, Any],
):
    """Yield ``(tool, record or exception)`` for each tool, in completion order.

    Each tool needs its own process when they run at once. The agent's output capture is
    ``contextlib.redirect_stdout``, which swaps the one process-wide ``sys.stdout``: two threads
    restoring each other's buffers left it a dead StringIO for the rest of the run, so every later
    [PASS]/[FAIL] line vanished, and every agent executed cells in the one process-global REPL
    namespace, binding over the others' ``adata`` mid-run (hunt 2026-09-30, u33a-bench-runner-3).
    One worker runs here, in this process, as before.
    """
    if workers <= 1:
        for tool in tools:
            try:
                yield tool, run_single_tool_smoke_test(tool_info=tool, **kwargs)
            except Exception as exc:
                yield tool, exc
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_single_tool_smoke_test, tool_info=tool, **kwargs): tool for tool in tools}
        for future in as_completed(futures):
            try:
                yield futures[future], future.result()
            except Exception as exc:
                yield futures[future], exc


def run_spatialomicsgym_smoke_test(
    mode: str = "direct",
    workers: int = 3,
    config_path: str | Path | None = None,
) -> ResultSchema:
    """Run the full SpatialOmicsLab MCP smoke test suite.

    Parameters
    ----------
    mode : str
        ``"direct"`` for minimal prompts, ``"full"`` for detailed prompts.
    workers : int
        Number of tools run at once, each in its own process (1 runs them here, in turn).
    config_path : str | Path | None
        Optional YAML file with run configuration overrides. Expected keys:
        ``llm``, ``temperature``, ``timeout_seconds``, ``output_dir``.

    Returns
    -------
    ResultSchema
        Structured results suitable for benchmark integration.
    """
    # -- defaults & config overrides -------------------------------------------
    run_cfg: dict[str, Any] = {
        "llm": "gpt-5",
        "temperature": 1.0,
        "timeout_seconds": 1800,
        "output_dir": str(REPO_ROOT / "benchmark_outputs" / "smoke_test"),
    }

    if config_path is not None:
        with open(config_path, encoding="utf-8") as fh:
            overrides = yaml.safe_load(fh) or {}
        run_cfg.update(overrides)

    output_dir = Path(run_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    mcp_config_path = PROJECT_ROOT / "MCP_server" / "mcp_config.yaml"
    enabled_tools = load_enabled_tools(mcp_config_path)

    if not enabled_tools:
        print("No enabled tools found in mcp_config.yaml -- nothing to test.")
        return ResultSchema(run_id="smoke-empty", strategy="smoke_test", results=[], summary={"total": 0})

    run_id = f"smoke-{uuid.uuid4().hex[:8]}"
    recorder = ResultRecorder(output_dir)
    records: list[SmokeTestRecord] = []

    # -- execution -------------------------------------------------------------
    run_kwargs = {
        "mode": mode,
        "project_root": PROJECT_ROOT,
        "mcp_config_path": str(mcp_config_path),
        "output_dir": str(output_dir),
        "llm": run_cfg["llm"],
        "temperature": run_cfg["temperature"],
        "timeout_seconds": run_cfg["timeout_seconds"],
    }
    for tool, outcome in _run_tools(enabled_tools, workers, run_kwargs):
        if isinstance(outcome, BaseException):
            record = SmokeTestRecord(
                tool_name=tool["spatialomicsgym_name"],
                dataset_name="",
                mode=mode,
                status="FAIL_MCP",
                error_message=str(outcome),
                error_traceback="".join(traceback.format_exception(outcome)),
            )
        else:
            record = outcome

        records.append(record)
        recorder.save(record)

        tag = "PASS" if record.status == "PASS" else "FAIL"
        print(f"[{tag}] {record.tool_name}: {record.status} ({record.elapsed_seconds:.1f}s)")

    # -- summaries & reports ---------------------------------------------------
    summary_data: dict[str, Any] = {
        "total": len(records),
        "passed": sum(1 for r in records if r.status == "PASS"),
        "failed": sum(1 for r in records if r.status != "PASS"),
        "by_status": {},
    }
    for r in records:
        summary_data["by_status"][r.status] = summary_data["by_status"].get(r.status, 0) + 1

    # Write summary.json
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary_data, indent=2, default=str), encoding="utf-8")

    # Write summary.md
    md_content = generate_summary_md(records, run_id)
    (output_dir / "summary.md").write_text(md_content, encoding="utf-8")

    # Write summary.html
    html_content = generate_summary_html(md_content)
    (output_dir / "summary.html").write_text(html_content, encoding="utf-8")

    # -- build ResultSchema for benchmark integration --------------------------
    benchmark_results = [
        BenchmarkResult(
            tool_name=r.tool_name,
            dataset_name=r.dataset_name,
            task_type="smoke_test",
            status="success" if r.status == "PASS" else "failure",
            elapsed_seconds=r.elapsed_seconds,
            error_message=r.error_message,
        )
        for r in records
    ]

    return ResultSchema(
        run_id=run_id,
        strategy="smoke_test",
        config_snapshot=run_cfg,
        results=benchmark_results,
        summary=summary_data,
    )


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for SpatialOmicsLab MCP smoke tests."""
    parser = argparse.ArgumentParser(
        description="Run SpatialOmicsLab-driven MCP smoke tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m benchmarks.workflows.spatialomicsgym_smoke_test --mode direct\n"
            "  python -m benchmarks.workflows.spatialomicsgym_smoke_test --mode full --workers 3\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "full"],
        default="direct",
        help="Prompt generation mode: 'direct' (minimal) or 'full' (detailed). Default: direct",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of tools run at once, each in its own process. Default: 3",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config file with overrides (llm, temperature, timeout_seconds, output_dir).",
    )

    args = parser.parse_args()
    schema = run_spatialomicsgym_smoke_test(mode=args.mode, workers=args.workers, config_path=args.config)
    print(f"\nCompleted run {schema.run_id}: {schema.summary.get('passed', 0)}/{schema.summary.get('total', 0)} passed")


if __name__ == "__main__":
    main()
