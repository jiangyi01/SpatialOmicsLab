"""MCP portal: PASTE2, for serial sections that only PARTIALLY overlap.

PASTE assumes two sections cover the same tissue and solves a full optimal-transport problem.
When they do not -- a section torn during cutting, a block trimmed between slices, a region that
simply runs out along the axis -- that assumption drags the whole fit, because every spot must be
matched to something. PASTE2 takes an overlap fraction and solves a partial problem instead.

The overlap fraction is the parameter that makes this tool different from PASTE, so it is the one
to get right. ``s = 0.0`` is the sentinel for "estimate it per pair" rather than a request for no
overlap; zero is not a legal fraction, so the two cannot be confused. The value actually used is
reported per pair, because a pair aligned at 0.9 and a pair aligned at 0.3 are different claims
about the tissue.
"""

from typing import Any

from base_mcp import create_mcp, default_output_dir, get_worker_paths, run_worker_cli

TOOL_NAME = "paste2"

WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "PASTE2",
    "/opt/conda/envs/paste2/bin/python",
    "/workspace/epic-fermat/agent/tools/paste2_worker.py",
)

mcp = create_mcp(TOOL_NAME)


def _slices(slice_h5ads: Any) -> list[str]:
    """Accept a list or a comma-separated string.

    PASTE's portal rejected a string where a list was required, and the recorded consequence is in
    ``test/test_data/mcp_run_results/paste_pairwise_align_result.json``: a pydantic ``list_type``
    error the model has no way to act on. Widening costs nothing and the config still declares an
    array.
    """
    if isinstance(slice_h5ads, str):
        return [p.strip() for p in slice_h5ads.split(",") if p.strip()]
    return [str(p) for p in (slice_h5ads or [])]


@mcp.tool()
def paste2_partial_align(
    slice_h5ads: list[str],
    output_dir: str,
    s: float = 0.0,
    alpha: float = 0.1,
    dissimilarity: str = "glmpca",
    armijo: bool = False,
    norm: bool = True,
    random_seed: int = 0,
    z_spacing: float = 0.0,
) -> dict[str, Any]:
    """Align serial sections that overlap only partially, with PASTE2's partial optimal transport.

    Parameters
    ----------
    slice_h5ads:
        Two or more .h5ad files, in physical section order, each with obsm['spatial'].
    output_dir:
        Where the aligned slices, the coupling matrices and the overlap table are written.
    s:
        Overlap fraction in (0, 1]. 0.0 means ESTIMATE IT per adjacent pair with
        ``select_overlap_fraction``; zero is not a legal fraction, so the sentinel is unambiguous.
        The estimate always runs GLM-PCA (latent_dim=50) and 20 partial OT solves per pair,
        whatever ``dissimilarity`` is chosen; give s to skip it. The value used is reported per
        pair either way.
    alpha:
        Weight on the spatial term against the expression term, as in PASTE.
    dissimilarity:
        'glmpca' is upstream's default and is the slowest by an order of magnitude; 'kl' and
        'euclidean' are the fast alternatives for the alignment itself (not for the s=0.0
        estimate, which is GLM-PCA regardless).
    armijo:
        Use an Armijo line search in the solver.
    norm:
        Rescale the spatial distances so their maximum equals the maximum expression
        dissimilarity, which keeps alpha weighing comparable terms. It does not touch the
        expression matrices.
    random_seed:
        PASTE2 exposes no seed of its own; this seeds numpy, which is what glmpca's initialisation
        draws from.
    z_spacing:
        Physical distance between consecutive sections, in the coordinates' own units. 0.0 means
        NOT DECLARED, not zero apart: PASTE2 aligns in plane and cannot supply a z itself. Given,
        the aligned coordinates are written as a three-column obsm['spatial_3d_aligned']; absent,
        as two columns under obsm['spatial_aligned']. Either way obsm['spatial'] keeps the
        coordinates PASTE2 was given -- without that there is no before frame and the alignment
        cannot be validated.
    """
    slices = _slices(slice_h5ads)
    if len(slices) < 2:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "paste2_partial_align requires at least two slice_h5ads, in section order.",
        }
    args = [
        "--mode",
        "partial",
        "--output-dir",
        output_dir or default_output_dir("paste2"),
        "--s",
        str(s),
        "--alpha",
        str(alpha),
        "--dissimilarity",
        dissimilarity,
        "--random-seed",
        str(random_seed),
        "--z-spacing",
        str(z_spacing),
    ]
    if armijo:
        args.append("--armijo")
    if not norm:
        args.append("--no-norm")
    for path in slices:
        args += ["--slice-h5ad", path]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def paste2_estimate_overlap(
    slice_h5ads: list[str],
    output_dir: str,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Measure how much of each adjacent pair actually overlaps. Aligns nothing.

    A value near 1 means PASTE's full-overlap assumption holds and PASTE is the cheaper tool; a low
    value is the evidence that it does not. Writes only the table. Each pair costs a GLM-PCA fit and
    20 partial OT solves; background spots (obs['in_tissue'] == 0) are left out first.

    Parameters
    ----------
    slice_h5ads:
        Two or more .h5ad files, in physical section order.
    output_dir:
        Where overlap_fractions.csv is written.
    alpha:
        Weight on the spatial term, as in the alignment itself.
    """
    slices = _slices(slice_h5ads)
    if len(slices) < 2:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": "paste2_estimate_overlap requires at least two slice_h5ads.",
        }
    args = [
        "--mode",
        "estimate",
        "--output-dir",
        output_dir or default_output_dir("paste2"),
        "--alpha",
        str(alpha),
    ]
    for path in slices:
        args += ["--slice-h5ad", path]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
