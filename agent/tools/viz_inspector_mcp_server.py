"""MCP portal: read a dataset and say what can be drawn from it. Writes nothing.

Four pure reads, deliberately on their own portal. None of them takes an output directory, none
of them creates a file, and none of them can therefore leave a directory behind that the
post-analysis engine would nominate for review. That is what lets this portal's data-requirements
block say "Writes nothing." and mean it, and it is why the producing tools live next door instead
of here.

The worker runs on the agent environment's own interpreter: everything these functions need --
anndata, numpy, pandas -- is already there, so there is no environment to build and nothing to
provision. The spec records that with ``build_strategy: none``, the shape the dataset-search
portal already uses.
"""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "viz-inspector"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "VIZ_INSPECTOR",
    "/opt/conda/envs/spatialomicsgym_env/bin/python",
    "/workspace/epic-fermat/agent/tools/viz_inspector_worker.py",
)

mcp = create_mcp(TOOL_NAME)

#: A comma-separated list longer than this is refused before the worker is spawned. Every
#: argument crosses the process boundary as one argv element, and the kernel's ceiling on that is
#: 128 KB -- a refusal that names the limit is a working tool, while an E2BIG from a healthy
#: interpreter reads as a broken installation and makes the model abandon the tool.
INLINE_LIST_MAX_CHARS = 40_000


def _too_long(name: str, value: str) -> dict[str, Any] | None:
    if len(value or "") <= INLINE_LIST_MAX_CHARS:
        return None
    return {
        "status": "error",
        "tool": TOOL_NAME,
        "error": (
            f"{name} is {len(value):,} characters, above the {INLINE_LIST_MAX_CHARS:,}-character "
            "limit for one argument. Ask about fewer names at a time."
        ),
        "diagnostic": "This tool is installed correctly and does not need reprovisioning.",
    }


@mcp.tool()
def inspect_dataset(
    data_path: str,
    obs_keys: str = "",
    var_names: str = "",
    max_categories: int = 200,
    sample_spots: int = 5000,
    include_capabilities: bool = True,
) -> dict[str, Any]:
    """Report what a dataset holds and what can be drawn from it. Writes nothing."""
    for name, value in (("obs_keys", obs_keys), ("var_names", var_names)):
        refusal = _too_long(name, value)
        if refusal:
            return refusal
    args = [
        "--task",
        "inspect",
        "--data-path",
        data_path,
        "--obs-keys",
        obs_keys,
        "--var-names",
        var_names,
        "--max-categories",
        str(int(max_categories)),
        "--sample-spots",
        str(int(sample_spots)),
    ]
    if include_capabilities:
        args.append("--include-capabilities")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def list_visualization_capabilities(data_path: str = "", include_unsupported: bool = True) -> dict[str, Any]:
    """List every plot this toolkit can draw, and for a named dataset which of them it can draw."""
    args = ["--task", "capabilities", "--data-path", data_path]
    if include_unsupported:
        args.append("--include-unsupported")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def recommend_visualizations(data_path: str, question: str = "", limit: int = 8) -> dict[str, Any]:
    """Suggest a short, ordered set of figures worth drawing for this dataset."""
    refusal = _too_long("question", question)
    if refusal:
        return refusal
    args = [
        "--task",
        "recommend",
        "--data-path",
        data_path,
        "--question",
        question,
        "--limit",
        str(int(limit)),
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def validate_plot_request(data_path: str, plot_id: str, params_json: str = "") -> dict[str, Any]:
    """Check one plot request against a dataset before drawing it, and say what is missing."""
    refusal = _too_long("params_json", params_json)
    if refusal:
        return refusal
    args = [
        "--task",
        "validate",
        "--data-path",
        data_path,
        "--plot-id",
        plot_id,
        "--params-json",
        params_json,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
