#!/usr/bin/env python3
"""SPACEL (Splane/Scube) MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "spacel"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "SPACEL",
    "/opt/conda/envs/spacel_env/bin/python3.9",
    "/workspace/epic-fermat/agent/tools/spacel_worker.py",
)

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_spacel_splane(
    spatial_h5ad_path: str,
    output_dir: str,
    n_clusters: int = 7,
    resolution: float = 1.0,
    celltype_key: str = "",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run SPACEL Splane for spatial domain identification on a single
    spatial transcriptomics AnnData (.h5ad).

    Splane clusters a graph-convolutional embedding of per-spot CELL-TYPE
    PROPORTIONS, not of gene expression. The input must therefore carry
    proportions: either ``uns['celltypes']`` naming one float obs column per
    cell type (e.g. written by a deconvolution step), or ``celltype_key``
    naming an obs label column to one-hot encode as a coarse stand-in.

    Spots with ``obs['in_tissue'] == 0`` (background outside the tissue, which
    CELLxGENE Visium exports carry) are left out and counted in
    params.in_tissue_filter; they get no domain and are absent from the outputs.
    data.n_spots is the number supplied, data.n_spots_used the number clustered.

    Parameters
    ----------
    spatial_h5ad_path:
        Path to the spatial AnnData (.h5ad) file.
    output_dir:
        Directory to write Splane outputs (spacel_splane_annotated.h5ad with
        obs['splane_cluster'], spacel_splane_clusters.csv).
    n_clusters:
        Number of spatial domains to identify (KMeans on the latent features).
    resolution:
        Accepted for backward compatibility and IGNORED: Splane has no
        resolution knob. Reported under params.ignored.
    celltype_key:
        obs column of per-spot cell-type labels to one-hot encode into
        Splane's proportion input. Leave empty when the h5ad already carries
        uns['celltypes'] with proportion columns.
    drop_unlabeled:
        A spot with no label (NaN/empty in obs[celltype_key], or a NaN
        proportion in the uns['celltypes'] columns) is not a cell type. False
        (default) stops the run and says how many there are; True leaves those
        spots out and reports the count in data.n_spots_unlabeled_dropped.
    """
    args = [
        "--task",
        "splane",
        "--spatial-h5ad",
        spatial_h5ad_path,
        "--output-dir",
        output_dir,
        "--n-clusters",
        str(n_clusters),
        "--resolution",
        str(resolution),
    ]
    if celltype_key:
        args += ["--celltype-key", celltype_key]
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def run_spacel_scube(
    spatial_h5ad_paths: str,
    output_dir: str,
    cluster_key: str = "",
    allow_joint_leiden: bool = False,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """
    Run SPACEL Scube for 3D alignment of multiple spatial transcriptomics
    slices. Takes comma-separated h5ad paths.

    Scube aligns slices on a label column shared by all of them, so the
    label ids must mean the same thing in every slice. Spots with
    ``obs['in_tissue'] == 0`` (background) are left out of every slice and
    counted in params.in_tissue_filter; the aligned slices do not contain them.

    Parameters
    ----------
    spatial_h5ad_paths:
        Comma-separated paths to two or more spatial AnnData (.h5ad) files.
    output_dir:
        Directory to write Scube outputs.
    cluster_key:
        obs column present in every slice to align on. Empty: a conventional
        label column shared by all slices is used if one exists.
    allow_joint_leiden:
        When no shared label column exists, cluster all slices jointly (one
        Leiden over the concatenated slices) and align on those labels. Off by
        default, because the alignment then depends on those clusters. When it
        runs, params.used_fallback is true.
    drop_unlabeled:
        A spot whose label in the aligned-on column is missing (NaN/empty) has
        no label to align. False (default) stops the run and says how many
        there are; True leaves those spots out of their slice and reports the
        count in data.n_spots_unlabeled_dropped.
    """
    args = [
        "--task",
        "scube",
        "--spatial-h5ad-paths",
        spatial_h5ad_paths,
        "--output-dir",
        output_dir,
    ]
    if cluster_key:
        args += ["--cluster-key", cluster_key]
    if allow_joint_leiden:
        args.append("--allow-joint-leiden")
    if drop_unlabeled:
        args.append("--drop-unlabeled")
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
