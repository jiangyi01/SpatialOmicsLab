"""MCP portal: CAST, non-rigid alignment of serial sections at single-cell resolution.

CAST learns a graph-neural embedding of each section (CAST_MARK), then registers the embeddings
onto a reference with an affine fit followed by a free-form deformation (CAST_STACK). It is the
non-rigid branch of the Phase-2 decision tree that STalign otherwise holds alone, and unlike
STalign it consumes and produces AnnData rather than bare point clouds.

``ffd_iterations = 0`` gives an affine-only result, which is the knob to reach for when the
diagnosis said class B rather than class C: the free-form stage is what a rigid misalignment does
not need and can be harmed by.

Measured on this box, CPU-only: 61 seconds for two sections totalling 1,415 cells at upstream's
default 400 epochs. The spec declares ``gpu: true`` because CUDA is what it is written for and a
GPU-shaped failure on a GPU-less box should not count as a hard failure -- not because CPU is
impractical.
"""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "cast"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "CAST",
    "/opt/conda/envs/cast/bin/python",
    "/workspace/epic-fermat/agent/tools/cast_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def cast_align_slices(
    slice_h5ads: list[str],
    output_dir: str,
    reference_index: int = 0,
    epochs: int = 400,
    affine_iterations: int = 500,
    ffd_iterations: int = 400,
    graph_strategy: str = "delaunay",
    n_hvg: int = 3000,
    random_seed: int = 0,
    z_spacing: float = 0.0,
) -> dict[str, Any]:
    """Register serial sections onto a reference with CAST, allowing non-rigid deformation.

    Parameters
    ----------
    slice_h5ads:
        Two or more .h5ad files, in physical section order, each with obsm['spatial']. Only genes
        present in every slice are used, since the embedding is joint; background spots
        (obs['in_tissue'] == 0) are left out.
    output_dir:
        Where the aligned slices and the displacement table are written. CAST's own intermediate
        artefacts, whose filenames are hardcoded upstream, go to a cast_intermediate/ subdirectory
        so they cannot be mistaken for the answer.
    reference_index:
        Which slice stays fixed, as an index into slice_h5ads. Every other section is warped onto
        it, so this choice changes the result.
    epochs:
        Training epochs for the graph-neural embedding. Upstream's default.
    affine_iterations:
        Iterations of the affine stage.
    ffd_iterations:
        Iterations of the free-form deformation. 0 gives an affine-only, rigid-ish alignment --
        the right setting when the diagnosis said class B.
    graph_strategy:
        'delaunay' (this portal's default) or 'convex' (upstream's). Both work; delaunay is pure
        scipy and networkx and measured ~20x faster here, while convex routes through libpysal and
        emits a deprecation warning. The divergence is recorded in official_defaults.yaml.
    n_hvg:
        Highly variable genes kept before embedding, chosen once across all slices so every slice
        is embedded over the same genes in the same order.
    random_seed:
        Seeds numpy and torch.
    z_spacing:
        Physical distance between consecutive sections, in the coordinates' own units. 0.0 means
        NOT DECLARED, not zero apart: CAST registers in plane and cannot supply a z itself. Given,
        the aligned coordinates are written as a three-column obsm['spatial_3d_aligned']; absent,
        as two columns under obsm['spatial_aligned']. Either way obsm['spatial'] keeps the
        coordinates CAST was given.
    """
    slices = (
        [p.strip() for p in slice_h5ads.split(",") if p.strip()]
        if isinstance(slice_h5ads, str)
        else [str(p) for p in (slice_h5ads or [])]
    )
    if len(slices) < 2:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "cast_align_slices requires at least two slice_h5ads, in section order.",
        }
    if not 0 <= reference_index < len(slices):
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": f"reference_index {reference_index} is outside 0..{len(slices) - 1}.",
        }
    args = [
        "--output-dir",
        output_dir or default_output_dir("cast"),
        "--reference-index",
        str(reference_index),
        "--epochs",
        str(epochs),
        "--affine-iterations",
        str(affine_iterations),
        "--ffd-iterations",
        str(ffd_iterations),
        "--graph-strategy",
        graph_strategy,
        "--n-hvg",
        str(n_hvg),
        "--random-seed",
        str(random_seed),
        "--z-spacing",
        str(z_spacing),
    ]
    for path in slices:
        args += ["--slice-h5ad", path]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
