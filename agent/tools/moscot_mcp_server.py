#!/usr/bin/env python3
"""MOSCOT MCP wrapper for SpatialOmicsLab."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "moscot"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "MOSCOT",
    "/opt/conda/envs/moscot/bin/python",
    "/workspace/epic-fermat/agent/tools/moscot_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def moscot_run(
    problem_type: str,
    adata_path: str,
    adata_spatial_path: str | None = None,
    adata_sc_path: str | None = None,
    time_key: str | None = None,
    batch_key: str | None = None,
    spatial_key: str = "spatial",
    policy: str | None = None,
    reference_batch: str | None = None,
    sc_attr: str | None = None,
    epsilon: float = 1e-3,
    alpha: float = 0.8,
    batch_size: int | None = None,
    device: str = "cpu",
    output_dir: str | None = None,
    out_tag: str = "moscot_run",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run a MOSCOT problem (TemporalProblem, AlignmentProblem or MappingProblem) in a dedicated worker.

    What runs, per ``problem_type``:

    - ``temporal``: X gets normalize_total + log1p, then a worker-computed PCA
      (n_comps = min(30, n_obs - 1, n_vars - 1)) is stored as obsm['X_moscot_pca'] and passed as
      ``TemporalProblem.prepare(time_key=time_key, policy=policy, joint_attr='X_moscot_pca')``.
      ``policy`` is 'sequential' (default), 'triu' or 'tril'; 'explicit' is refused because it needs
      time-point pairs this tool does not take. The couplings are saved only in
      moscot_temporal_problem.pkl.
    - ``alignment``: the same preprocessing, then ``AlignmentProblem.prepare(batch_key=batch_key,
      spatial_key=spatial_key, policy=policy, joint_attr='X_moscot_pca')`` and ``ap.align()`` into
      obsm['moscot_spatial_warp'] of adata_alignment_aligned.h5ad, one merged object whose sections
      are told apart by obs[batch_key] (recorded as params.batch_key). ``reference_batch`` is the
      section held fixed: the hub of policy='star' (passed to prepare as well) and the target of
      align(). Omitted, the first level of obs[batch_key] in order of appearance is used and
      reported as params.reference_batch_used. obs[batch_key] is stored as string categories
      (moscot's align() needs them), so '0' names an integer section 0; an unknown name is refused
      before solving.
      ``alpha`` is not forwarded: AlignmentProblem.solve runs with moscot's own alpha (0.5,
      params.solver_alpha).
    - ``mapping``: with ``sc_attr`` omitted, BOTH AnnDatas' X get the same
      normalize_total(target_sum=1e4) + log1p (moscot builds the expression term from one joint PCA
      over their shared genes) and the worker's PCA of the sc AnnData is passed as
      sc_attr='X_moscot_pca'; with ``sc_attr`` given neither X is touched. Then
      ``MappingProblem.solve(alpha=alpha, ...)``. The prediction is moscot_mapping_cell_to_spot.csv
      (columns cell, batch, spot, transport_mass, mass_fraction: for every cell, the spot that
      receives most of its transported mass); adata_mapping_spatial.h5ad and
      adata_mapping_sc_with_pca.h5ad are the inputs as they entered the solver.

    Background spots: when the spatial AnnData (the input of temporal/alignment, the spatial side of
    mapping) has obs['in_tissue'], spots marked 0 are left out after loading and counted in
    params.in_tissue_filter, with a warning.

    ``drop_unlabeled`` (default False): observations with no ``time_key`` (temporal) or ``batch_key``
    (alignment, mapping) value belong to no moscot subproblem -- alignment used to write (0, 0) as
    their warped coordinates. By default such a run is refused with the count; True drops those
    rows before solving and reports how many.

    Parameters that do not apply to the chosen problem type are accepted and listed in
    params.ignored. ``epsilon``, ``batch_size`` (None lets moscot decide) and ``device`` ('cpu',
    'gpu', 'cuda[:N]' or 'tpu') reach ``.solve()`` for every problem type. A GPU request on a
    machine where this worker finds no GPU runs on the CPU: params.device is the request,
    params.device_used what solve() received, and the payload warns. Parameters mirror the MCP
    JSON schema in mcp_config.yaml.
    """
    args = [
        "--problem-type",
        problem_type,
        "--adata-path",
        adata_path,
        "--spatial-key",
        spatial_key,
        "--epsilon",
        str(epsilon),
        "--alpha",
        str(alpha),
        "--device",
        device,
        "--out-tag",
        out_tag,
    ]
    if adata_spatial_path is not None:
        args += ["--adata-spatial-path", adata_spatial_path]
    if adata_sc_path is not None:
        args += ["--adata-sc-path", adata_sc_path]
    if time_key is not None:
        args += ["--time-key", time_key]
    if batch_key is not None:
        args += ["--batch-key", batch_key]
    if policy is not None:
        args += ["--policy", policy]
    if reference_batch is not None:
        args += ["--reference-batch", reference_batch]
    if sc_attr is not None:
        args += ["--sc-attr", sc_attr]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if output_dir is not None:
        args += ["--output-dir", output_dir]
    if drop_unlabeled:
        args += ["--drop-unlabeled"]

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
