"""Runner for executing individual MCP tools through their portals."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.runners.base_runner import BenchmarkRunner
from benchmarks.runners.pipeline_runner import next_step_dataset
from benchmarks.strategies.base_strategy import BenchmarkResult

if TYPE_CHECKING:
    from benchmarks.config.benchmark_config import BenchmarkConfig
    from benchmarks.data.data_registry import DatasetEntry

#: Parameter names a tool declares for the spatial slide, the output directory, the single-cell
#: reference and the reference's label column -- in the order they are tried.
_SLIDE_PARAMS = (
    "st_h5ad",
    "spatial_h5ad",
    "spatial_h5ad_path",
    "h5ad_path",
    "data_path",
    "adata_path",
    "spatial_data_path",
    "counts_h5ad",
)
_OUTPUT_PARAMS = ("output_dir", "results_dir", "save_path", "out_dir")
_REFERENCE_PARAMS = ("sc_h5ad", "sc_h5ad_path", "scrna_h5ad", "sc_data_path", "sc_data")
_LABEL_PARAMS = (
    "cell_type_key",
    "celltype_key",
    "annotation_key",
    "labels_key",
    "sc_label_key",
    "celltype_col",
    "label_key",
)
_CLUSTER_COUNT_PARAMS = ("n_clusters", "n_domains", "K", "n_clust", "target_n_clusters")


class MCPToolRunner(BenchmarkRunner):
    """Execute MCP tools the way the agent does: one call through the tool's MCP portal.

    This used to run the server command from mcp_config (``python tools/X_mcp_server.py``) with
    ``--data_path``/``--output_dir`` appended. A portal parses no argv -- it starts a FastMCP stdio
    server on the inherited stdin -- so from a terminal every tool hung until ``timeout_seconds``
    and was recorded "timeout", and with stdin at EOF it exited with no JSON: "failure". It also
    read the server's ``environment`` block, which mcp_config spells ``env``, so worker paths and
    interpreters were never applied. The Full and Smoke workflows built on it could not produce a
    single success (hunt 2026-09-30, u33a-bench-runner-20). It now goes through
    ``mcp_integration.make_mcp_wrapper``, the agent's own dispatch, with the arguments mapped onto
    the parameters the tool declares; a tool whose required parameters the dataset entry cannot
    fill is reported as such rather than called with guesses.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        super().__init__(config)
        self._mcp_config: dict[str, Any] | None = None

    def _load_mcp_config(self) -> dict[str, Any]:
        """Load and cache the MCP configuration."""
        if self._mcp_config is None:
            import yaml

            config_path = self.config.resolve_path(self.config.mcp_config_path)
            with open(config_path) as f:
                self._mcp_config = yaml.safe_load(f)
        return self._mcp_config

    def _find_tool_config(self, tool_name: str) -> dict[str, Any] | None:
        """Find tool configuration in mcp_config.yaml by spatialomicsgym_name."""
        mcp_config = self._load_mcp_config()
        for server_key, server_config in mcp_config.get("mcp_servers", {}).items():
            for tool in server_config.get("tools", []):
                if tool.get("spatialomicsgym_name") == tool_name:
                    env = server_config.get("env") or {}
                    return {
                        "server_key": server_key,
                        "command": server_config.get("command", []),
                        "args": server_config.get("args", []),
                        "tool_config": tool,
                        "env": env,
                        # The old key, kept for any caller that read it; same block.
                        "environment": env,
                    }
        return None

    @staticmethod
    def _tool_arguments(
        tool_name: str,
        tool_config: dict[str, Any],
        dataset: DatasetEntry,
        input_path: Path,
        output_dir: Path,
        data_dir: Path,
    ) -> dict[str, Any]:
        """The call's keyword arguments, mapped onto the parameters the tool declares.

        Raises ValueError naming what is missing when the dataset entry cannot fill a parameter the
        tool requires.
        """
        params = tool_config.get("parameters") or {}
        metadata = dataset.metadata or {}
        arguments: dict[str, Any] = {}
        slide = next((p for p in _SLIDE_PARAMS if p in params), None)
        if slide is None:
            raise ValueError(f"{tool_name} declares no h5ad slide parameter this runner can fill: {sorted(params)}")
        arguments[slide] = str(input_path)
        output = next((p for p in _OUTPUT_PARAMS if p in params), None)
        if output:
            arguments[output] = str(output_dir)
        if "input_mode" in params:
            spec = params["input_mode"] if isinstance(params["input_mode"], dict) else {}
            offered = [str(v) for v in (spec.get("enum") or [])]
            if "h5ad" not in offered and spec.get("default") != "h5ad":
                raise ValueError(f"{tool_name} has no h5ad input_mode; this runner stages only an h5ad")
            arguments["input_mode"] = "h5ad"
        reference = next((p for p in _REFERENCE_PARAMS if p in params), None)
        if reference and metadata.get("sc_reference_path"):
            arguments[reference] = str(data_dir / metadata["sc_reference_path"])
        label = next((p for p in _LABEL_PARAMS if p in params), None)
        if label and reference in arguments and metadata.get("sc_reference_celltype_key"):
            arguments[label] = metadata["sc_reference_celltype_key"]
        count = next((p for p in _CLUSTER_COUNT_PARAMS if p in params), None)
        if count and metadata.get("n_clusters") is not None:
            arguments[count] = int(metadata["n_clusters"])
        missing = [
            name
            for name, spec in params.items()
            if isinstance(spec, dict) and spec.get("required") and name not in arguments
        ]
        if missing:
            raise ValueError(f"{tool_name} requires {missing}, which this runner cannot fill from {dataset.name}")
        return arguments

    def _dispatcher(self, tool_config: dict[str, Any], tool_name: str, timeout_seconds: float):
        """The agent's own synchronous MCP call for *tool_name*, bounded at *timeout_seconds*."""
        from types import SimpleNamespace

        from spatialomicsgym.agent.mcp_integration import (
            _local_tools_dir,
            _rebase_script_args,
            _resolve_portal_interp,
            make_mcp_wrapper,
        )

        command = [str(c) for c in (tool_config.get("command") or [])]
        if not command:
            raise ValueError(f"{tool_name}: its mcp_config server has no command")
        args = _rebase_script_args(command[1:] + [str(a) for a in (tool_config.get("args") or [])], _local_tools_dir())
        budget = SimpleNamespace(timeout_seconds=timeout_seconds)
        return make_mcp_wrapper(
            budget, _resolve_portal_interp(command[0]), args, tool_name, "", env_spec=tool_config.get("env") or {}
        )

    def run_tool(
        self,
        tool_name: str,
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """Run a single MCP tool on a dataset through its portal and parse the JSON result."""
        result = BenchmarkResult(
            tool_name=tool_name,
            dataset_name=dataset.name,
            task_type=dataset.task_type,
            config_snapshot=config.to_dict(),
        )

        tool_config = self._find_tool_config(tool_name)
        if tool_config is None:
            result.status = "error"
            result.error_message = f"Tool '{tool_name}' not found in MCP config"
            return result

        data_dir = config.resolve_path(config.data_dir)
        input_path = dataset.resolve_input(data_dir)
        output_dir = Path(config.resolve_path(config.output_dir)) / tool_name / dataset.name

        if not input_path.exists():
            result.status = "error"
            result.error_message = f"Input file not found: {input_path}"
            return result

        try:
            arguments = self._tool_arguments(
                tool_name, tool_config["tool_config"], dataset, input_path, output_dir, Path(data_dir)
            )
            call = self._dispatcher(tool_config, tool_name, config.timeout_seconds)
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            return result

        output_dir.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        try:
            raw = call(**arguments)
        except Exception as e:
            result.elapsed_seconds = time.time() - start_time
            message = str(e)
            result.status = "timeout" if "timed out" in message.lower() or "timeout" in message.lower() else "failure"
            result.error_message = message[:1000]
            return result
        result.elapsed_seconds = time.time() - start_time

        output_json = self._parse_output(raw if isinstance(raw, str) else json.dumps(raw, default=str))
        if not output_json:
            result.status = "failure"
            result.error_message = "Could not parse JSON output from the tool"
            return result
        if str(output_json.get("status", "")).lower() in ("error", "failed", "failure"):
            result.status = "failure"
            result.error_message = str(output_json.get("error") or output_json.get("message") or output_json)[:1000]
            return result
        result.status = "success"
        result.metrics = output_json.get("evaluation", output_json.get("summary", {}))
        files = output_json.get("output_files", {})
        result.output_files = [str(f) for f in (files.values() if isinstance(files, dict) else files or [])]
        return result

    def run_pipeline(
        self,
        steps: list[dict[str, Any]],
        dataset: DatasetEntry,
        config: BenchmarkConfig,
    ) -> list[BenchmarkResult]:
        """Run a multi-step pipeline, each step on the previous step's h5ad output (see next_step_dataset)."""
        results: list[BenchmarkResult] = []
        current_dataset = dataset
        for step in steps:
            tool_name = step.get("tool", "")
            result = self.run_tool(tool_name, current_dataset, config)
            results.append(result)
            if not result.is_success():
                break
            current_dataset = next_step_dataset(current_dataset, result)
        return results

    @staticmethod
    def _parse_output(stdout: str) -> dict[str, Any] | None:
        """Parse the tool's JSON result: the whole text, else the last JSON object line."""
        text = (stdout or "").strip()
        try:
            whole = json.loads(text)
            if isinstance(whole, dict):
                return whole
        except (json.JSONDecodeError, ValueError):
            pass
        lines = text.splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None
