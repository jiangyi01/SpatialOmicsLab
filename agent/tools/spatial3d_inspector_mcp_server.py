"""MCP portal: read what an object already says about its own 3D coordinates. Writes nothing.

Split from the diagnosis portal for the reason the visualization toolkit splits its inspector from
its producers: a caller orienting itself should not have to pick an output directory, and a
function that cannot write cannot be blamed for a file that appeared. Everything here is a read.
"""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spatial3d-inspector"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIAL3D",
    "/opt/conda/envs/spatialomicsgym_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spatial3d_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def inspect_3d_coordinates(data_path: str, coords_key: str = "spatial") -> dict[str, Any]:
    """List the coordinate keys an object holds, their shapes, and what the contract says is wrong.

    The fast orienting read before a diagnosis: which obsm keys exist, which of them are
    three-column frames, whether a provenance block is present, and which contract rules the
    object currently breaks.
    """
    return run_worker_cli(
        TOOL_NAME,
        WORKER_PYTHON,
        WORKER_SCRIPT,
        ["--task", "inspect", "--data-path", data_path, "--coords-key", coords_key],
    )


@mcp.tool()
def explain_3d_contract() -> dict[str, Any]:
    """The 3D coordinate keys, roles, units and rules, so a caller need not infer them.

    Includes the one rule that is easy to get wrong and expensive to get wrong: obsm['spatial']
    stays two columns and is never overwritten, because PASTE asserts it and a recorded run died
    on a three-column spatial.
    """
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, ["--task", "contract"])


@mcp.tool()
def list_aligner_adapters() -> dict[str, Any]:
    """Where each shipped alignment tool leaves its answer, or the reason it cannot be read.

    No two of the ten agree. This is the table that says which obsm key to open for a given tool,
    whether the coordinates it was given can still be recovered from its output, and whether it
    exposed a random seed at all.
    """
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, ["--task", "adapters"])


if __name__ == "__main__":
    mcp.run()
