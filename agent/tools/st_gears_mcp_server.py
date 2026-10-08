#!/usr/bin/env python3
"""ST-GEARS 3D reconstruction MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "st-gears"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "ST_GEARS",
    "/opt/conda/envs/st_gears/bin/python",
    "/workspace/epic-fermat/agent/tools/st_gears_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def st_gears_reconstruct_3d(
    st_h5ad: str,
    output_dir: str,
    slice_key: str = "slice_id",
    group_key: str = "annotation",
    binning_on: bool = True,
    bin_step: int = 2,
    uniform_weight: bool = False,
    filter_by_label: bool = True,
    tune_alpha_li: list[float] | None = None,
    num_itermax: int = 200,
    fil_pc: int = 20,
    pixel_size: float | None = None,
    sigma: float = 1.0,
    start_idx: int | None = None,
    end_idx: int | None = None,
    use_gpu: bool = False,
    seed: int = 0,
    drop_unlabeled: bool = False,
    allow_nearest_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run the full ST-GEARS 3D reconstruction pipeline on a multi-section
    spatial AnnData (.h5ad): optional binning, serial_align anchors, rigid
    stacking, elastic registration and, when binning ran, interpolation back
    to every original spot.

    Parameters
    ----------
    st_h5ad:
        Multi-section spatial AnnData with obsm['spatial'] (two columns x, y,
        or three x, y, z), an obs section-id column (slice_key) and an obs
        cluster/annotation column (group_key). ST-GEARS itself indexes a third
        coordinate column; for a two-column input the worker hands it a working
        copy whose z is the slice ordinal and writes the two-column original
        back unchanged. Spots with obs['in_tissue'] == 0 (background glass, as
        CELLxGENE Visium exports carry) are left out of the alignment and the
        output, and counted in params.in_tissue_filter; a section with no
        in-tissue spot is refused.
    output_dir:
        Where 'st_gears_aligned.h5ad' and 'st_gears_run_metadata.json' are
        written.
    slice_key:
        obs column of section ids. Sections are ordered numerically when every
        id is a number, and in natural order otherwise ('S2' before 'S10').
    group_key:
        obs column of clusters/annotations used for anchors. A missing label
        (NaN/empty) is refused unless drop_unlabeled=True.
    binning_on, bin_step:
        Granularity adjusting. bin_step is in the coordinates' own units:
        pick it from the spot spacing (a step below the spacing bins nothing).
        Binning densifies each section's spots x genes matrix whatever the
        step (st_gears.binning calls X.todense()); the worker estimates that
        memory first and refuses with the numbers when it cannot fit.
    uniform_weight, filter_by_label, tune_alpha_li, num_itermax:
        Passed to st_gears.serial_align. tune_alpha_li that is not a list of
        numbers is an error (it used to be replaced by the default list).
    fil_pc, sigma:
        Passed to the rigid and elastic steps.
    pixel_size:
        Edge length of one elastic-field pixel, in coordinate units. When
        None, the median nearest-neighbour spacing of the spots (bins when
        binning is on) handed to ST-GEARS -- upstream's guidance is 'a rough
        average of spots distance'. The value used is reported in
        params.pixel_size and params.pixel_size_source. The field is a dense
        grid triangulated whole, so a pixel_size far below the spot spacing
        on pixel-unit coordinates is refused, with its grid size and memory
        estimate, when that grid cannot fit in memory.
    start_idx, end_idx:
        0-based positions, in the ordered sections, of the first and last
        section to align (start_idx < end_idx). Only the sections in that range
        are aligned and written.
    use_gpu:
        Let serial_align use CUDA; a request on a machine without CUDA runs on
        CPU.
    seed:
        numpy seed.
    drop_unlabeled:
        False (default): spots with a missing group_key label or slice_key id
        stop the run with their count. True: they are left out of the
        alignment and the output, and the count is reported.
    allow_nearest_fallback:
        With binning on, ST-GEARS's linear interpolation gives no coordinate to
        spots outside the convex hull of the binned grid: their rows in
        spatial_3d_aligned / st_gears_xyz / spatial_elas_reuse are NaN,
        obs['st_gears_xy_source'] is 'none', and the payload counts them
        (summary.n_spots_without_aligned_xy). True gives those spots the
        displacement of the nearest bin in spatial_3d_aligned / st_gears_xyz
        (spatial_elas_reuse stays as ST-GEARS wrote it), marks them 'nearest',
        and records params.used_fallback=True. No effect with binning off.

    Returns the worker payload: obsm['spatial_3d_aligned'] (aligned x, y and
    the slice ordinal as z) in the aligned h5ad, params.method, per-section
    counts of unaligned and filled spots, and the effective pixel_size.
    """
    if tune_alpha_li is None:
        tune_alpha_li = [0.8, 0.2, 0.05, 0.013]

    args: list[str] = [
        "--st-h5ad",
        st_h5ad,
        "--output-dir",
        output_dir,
        "--slice-key",
        slice_key,
        "--group-key",
        group_key,
        "--binning-on",
        "true" if binning_on else "false",
        "--bin-step",
        str(bin_step),
        "--uniform-weight",
        "true" if uniform_weight else "false",
        "--filter-by-label",
        "true" if filter_by_label else "false",
        "--tune-alpha-li",
        ",".join(str(a) for a in tune_alpha_li),
        "--num-itermax",
        str(num_itermax),
        "--fil-pc",
        str(fil_pc),
        "--sigma",
        str(sigma),
        "--use-gpu",
        "true" if use_gpu else "false",
        "--seed",
        str(seed),
    ]
    if pixel_size is not None:
        args += ["--pixel-size", str(pixel_size)]
    if start_idx is not None:
        args += ["--start-idx", str(start_idx)]
    if end_idx is not None:
        args += ["--end-idx", str(end_idx)]
    args += ["--drop-unlabeled", "true" if drop_unlabeled else "false"]
    args += ["--allow-nearest-fallback", "true" if allow_nearest_fallback else "false"]

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
