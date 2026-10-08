"""MCP portal: diagnose a stack of serial sections before anything moves a coordinate.

Ten alignment portals already ship in this repository. This one is the layer around them: it says
whether a stack needs aligning at all, and on what evidence. The answer is one of four -- already
aligned, rigidly misaligned, non-rigidly deformed, or not determined -- and it never arrives as a
bare letter. Every criterion comes with the value measured, the threshold compared against, and
where that threshold itself was measured.

Two things this portal will not do. It does not modify the data it is given: a diagnosis that
edited its subject would be a different kind of tool, and the promise is held by a test that
digests the object either side of a run. And it has no default section spacing, anywhere -- a
stack assembled on an assumed spacing produces a 3D neighbour graph that is either several
disconnected 2D graphs or one undifferentiated mush, and neither announces itself. An unknown
spacing comes back as a question.

The worker runs on the agent environment's own interpreter: numpy, scipy, pandas and anndata are
already there, so there is no environment to build.
"""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "spatial3d"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPATIAL3D",
    "/opt/conda/envs/spatialomicsgym_env/bin/python",
    "/workspace/epic-fermat/agent/tools/spatial3d_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def diagnose_3d_stack(
    data_path: str,
    output_dir: str,
    slice_key: str = "",
    z_key: str = "",
    slice_order: str = "",
    coords_key: str = "spatial",
    n_genes: int = 8,
    skip_expression: bool = False,
) -> dict[str, Any]:
    """Classify a stack of serial sections as A, B, C or unknown, with the evidence.

    Reads the data and writes only its own report. Returns the class, the per-pair metric table,
    the numbers behind the verdict, and -- where a fact could not be established -- the question
    whose answer would establish it.
    """
    return run_worker_cli(
        TOOL_NAME,
        WORKER_PYTHON,
        WORKER_SCRIPT,
        [
            "--task",
            "diagnose",
            "--data-path",
            data_path,
            "--output-dir",
            output_dir or default_output_dir("spatial3d"),
            "--slice-key",
            slice_key,
            "--z-key",
            z_key,
            "--slice-order",
            slice_order,
            "--coords-key",
            coords_key,
            "--n-genes",
            str(n_genes),
        ]
        + (["--skip-expression"] if skip_expression else []),
    )


if __name__ == "__main__":
    mcp.run()
