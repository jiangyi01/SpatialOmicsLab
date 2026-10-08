#!/usr/bin/env python3
"""SMART spatial mapping MCP wrapper for SpatialOmicsLab."""

from __future__ import annotations

import os
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "smart_spatial"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "SMART",
    "/opt/conda/envs/smart_env/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/smart_spatial_worker.R",
)

# Upstream ``SMART_base(iterations = 2000)``. This portal has always run 500 -- the value was
# hard-coded in the worker and never reported -- and 500 stays the default so recorded runs remain
# comparable. It is now a parameter, and the payload's ``params.iterations`` says what ran.
DEFAULT_ITERATIONS = 500
UPSTREAM_ITERATIONS = 2000

mcp = create_mcp(TOOL_NAME)


@mcp.tool()
def run_smart(
    spatial_counts_csv: str,
    marker_genes_csv: str,
    output_dir: str,
    ref_counts_csv: str = "",
    n_topics: int = 10,
    seed: int = 42,
    iterations: int = DEFAULT_ITERATIONS,
    round_counts: bool = False,
) -> dict[str, Any]:
    """
    Run SMART keyword-assisted topic model deconvolution on spatial data.

    SMART fits a keyATM base model (``SMART_base``) on the spatial counts. Each cell type in the
    marker gene list seeds one keyword topic, ``n_topics`` minus that many unsupervised topics
    are added, and the per-spot topic proportions are returned as cell type proportions. No
    single-cell reference is read: the topics come from the marker list alone. The payload's
    ``params.method`` names what ran and ``params.used_fallback`` is always False (there is no
    substitute path).

    Parameters
    ----------
    spatial_counts_csv:
        Path to spatial gene expression counts CSV (genes x spots, as written by
        ``convert_h5ad_to_csv``; spots x genes is detected from the marker gene names). Must be
        raw integer counts -- see ``round_counts``.
    marker_genes_csv:
        Path to marker genes CSV. Two layouts: a gene column plus a cell-type column (one row per
        marker gene; accepted headings: gene/Gene/genes/Genes/gene_name/marker and
        cell_type/celltype/cluster/Cluster/CellType/cell_type_name/type) seeds one topic per
        cell type; a single gene column seeds one topic with every gene. A file with several
        columns and no recognised gene column is refused by name. A marker seeds its topic only
        if it is on the spatial panel and counted in at least one spot (keyATM cannot seed a
        topic with a word that never occurs); a cell type with no such marker gets no topic and
        no proportions column, and the payload lists it under ``data.dropped_marker_types`` with
        a warning. ``data.n_marker_genes_off_panel`` / ``data.n_marker_genes_without_counts``
        count the marker genes that could not be used.
    output_dir:
        Directory for SMART output files: ``smart_proportions.csv`` (spots x topics with a
        ``spot`` column), ``smart_beta.csv`` (topics x genes) and, written by SMART itself,
        ``inst_<seed>/base_model.rds`` (the fitted keyATM object).
    ref_counts_csv:
        Not read. SMART seeds its topics from ``marker_genes_csv`` and never opens a single-cell
        reference; the parameter is kept so existing callers keep working. A non-empty value is
        forwarded and comes back under ``params.ignored`` with a warning -- it is no longer
        checked for existence, so a wrong path here cannot fail the run.
    n_topics:
        Total number of topics (keyword-seeded + unsupervised extras). When it is below the
        number of marker cell types (0 included), one topic per usable marker cell type is fitted
        with no extras and the payload warns; ``summary.n_topics_fit`` is the effective count.
    seed:
        Random seed for reproducibility (also names the ``inst_<seed>/`` model directory).
    iterations:
        keyATM Gibbs-sampling iterations passed to ``SMART_base`` (at least 1). Upstream's
        default is 2000; this portal defaults to 500, the value it has always run. Reported in
        ``params.iterations``. Run time grows linearly with it.
    round_counts:
        keyATM models non-negative integer counts. With the default False, a matrix holding
        non-integer or negative values (normalised or log-transformed expression) is refused
        with a message naming this knob; whole-number floats pass untouched. With True the
        values are rounded to the nearest integer and negatives clamped to 0, and the payload
        reports ``data.n_values_rounded`` / ``data.n_values_clamped`` with a warning.
    """
    os.makedirs(output_dir, exist_ok=True)

    args = [
        "--spatial-counts-csv",
        spatial_counts_csv,
        "--marker-genes-csv",
        marker_genes_csv,
        "--output-dir",
        output_dir,
        "--n-topics",
        str(n_topics),
        "--seed",
        str(seed),
        "--iterations",
        str(iterations),
    ]

    if round_counts:
        args.extend(["--round-counts", "true"])

    # Forwarded so the worker can record it under params.ignored; the worker never opens it.
    if ref_counts_csv:
        args.extend(["--ref-counts-csv", ref_counts_csv])

    return run_worker_cli(TOOL_NAME, WORKER_RSCRIPT, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
