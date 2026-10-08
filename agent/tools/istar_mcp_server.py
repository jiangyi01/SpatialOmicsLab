#!/usr/bin/env python3
"""iStar MCP wrapper for SpatialOmicsLab."""

import os
import re
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "istar"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "ISTAR",
    "/opt/conda/envs/istar/bin/python",
    "/workspace/epic-fermat/agent/tools/istar_worker.py",
)

mcp = create_mcp(TOOL_NAME)

#: Where the iStar checkout is looked for when ISTAR_SRC is unset: the order of
#: ``istar_worker.resolve_istar_root``, which runs the scripts from it. A copy, because this server
#: runs in another env and does not import the worker; test/test_istar_finds_its_demo_through_the_
#: fallback_checkout_when_istar_src_is_unset.py fails when the two differ (hunt 2026-09-30, rp-u30).
_ISTAR_ROOT_FALLBACKS = (os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "istar"), "/opt/istar")
_ISTAR_SRC_REF = re.compile(r"\$\{ISTAR_SRC\}|\$ISTAR_SRC(?![A-Za-z0-9_])")


def _istar_root() -> str | None:
    for candidate in (os.environ.get("ISTAR_SRC") or "", *_ISTAR_ROOT_FALLBACKS):
        if candidate and os.path.isfile(os.path.join(candidate, "rescale.py")):
            return candidate
    return None


def expand_istar_prefix(prefix: str) -> tuple[str, str | None]:
    """``(prefix with $ISTAR_SRC, ~ and other variables expanded, the iStar root used or None)``.

    The descriptions give the demo as ``prefix='$ISTAR_SRC/data/demo/'``; ISTAR_SRC is not in this
    server's environment, so the literal string became ``<cwd>/$ISTAR_SRC/data/demo`` and the demo read
    as missing (hunt 2026-09-30, u30-uncovered-mcp-6). It now means the checkout the worker runs.
    """
    root = _istar_root() if _ISTAR_SRC_REF.search(prefix or "") else None
    expanded = _ISTAR_SRC_REF.sub(lambda _m: root, prefix) if root else prefix
    return os.path.expanduser(os.path.expandvars(expanded)), root


@mcp.tool()
def istar_check_dataset(prefix: str) -> dict[str, Any]:
    """
    Check whether an iStar dataset prefix directory is ready to run the pipeline.

    Inspects the directory for required raw input files:
      he-raw.jpg, cnts.tsv, locs-raw.tsv, pixel-size-raw.txt, radius-raw.txt
    """
    names_istar_src = bool(_ISTAR_SRC_REF.search(prefix or ""))
    prefix, istar_root = expand_istar_prefix(prefix)
    if not prefix.endswith("/") and not prefix.endswith("\\"):
        prefix = prefix + "/"
    prefix_abs = os.path.abspath(prefix)
    dir_exists = os.path.isdir(prefix_abs)

    required_names = [
        "he-raw.jpg",
        "cnts.tsv",
        "locs-raw.tsv",
        "pixel-size-raw.txt",
        "radius-raw.txt",
    ]
    required_files: dict[str, Any] = {}
    for name in required_names:
        path = os.path.join(prefix_abs, name)
        exists = os.path.isfile(path)
        required_files[name] = {"path": path, "exists": exists}

    ready = dir_exists and all(info["exists"] for info in required_files.values())
    if ready:
        message = "Dataset is ready to run iStar."
    elif names_istar_src and istar_root is None:
        message = (
            "Dataset is NOT ready: the prefix names $ISTAR_SRC and no iStar checkout was found (ISTAR_SRC is "
            f"unset and none of {', '.join(_ISTAR_ROOT_FALLBACKS)} holds rescale.py). Set ISTAR_SRC to the "
            "iStar checkout, or pass the dataset directory itself."
        )
    else:
        message = "Dataset is NOT ready; see 'required_files' for missing items."
    return {
        "prefix": prefix_abs,
        "exists": dir_exists,
        "required_files": required_files,
        "ready": ready,
        "message": message,
    }


@mcp.tool()
def istar_full_pipeline(
    prefix: str,
    output_dir: str,
    device: str = "cpu",
    pixel_size: float = 0.5,
    n_genes: int = 1000,
    impute_epochs: int = 400,
    n_clusters: int = 10,
    filter_size: int = 8,
    min_cluster_size: int = 20,
    run_celltype: bool = False,
    celltype_marker_file: str | None = None,
    run_signature_scores: bool = False,
    signature_file: str | None = None,
    allow_mask_fallback: bool = False,
) -> dict[str, Any]:
    """
    Run the full iStar pipeline: iStar's own scripts, run in the order of its run.sh.

    Parameters
    ----------
    prefix:
        Directory prefix for the dataset (he-raw.jpg, cnts.tsv, locs-raw.tsv, pixel-size-raw.txt,
        radius-raw.txt).
    output_dir:
        Directory where all outputs will be written. The raw inputs are copied here first.
    device:
        'cpu' or 'cuda' for feature extraction and imputation.
    pixel_size:
        Desired analysis pixel size (microns per pixel), written to pixel-size.txt.
    n_genes:
        Number of most variable genes to impute (select_genes.py --n-top). The count actually
        imputed is reported as summary.n_genes_imputed.
    impute_epochs:
        Number of imputation epochs.
    n_clusters:
        Number of k-means clusters for gene-feature clustering. Regions smaller than
        min_cluster_size are merged into their neighbours afterwards, so summary.n_clusters (the
        clusters found) can be lower; the request is summary.n_clusters_requested.
    filter_size:
        Filter size for smoothing.
    min_cluster_size:
        Minimum connected region size, in superpixels.
    run_celltype:
        If True, run optional cell-type inference (pixannot.py + enrich.py). The marker panel is
        checked before the pipeline starts, and again against the imputed genes before imputation.
    celltype_marker_file:
        Path to cell-type marker panel, a TSV with 'gene' and 'label' columns (only used if
        run_celltype).
    run_signature_scores:
        If True, compute signature scores (marker_score.py).
    signature_file:
        Path to signature score panel, one gene per line (only used if run_signature_scores).
    allow_mask_fallback:
        What to do when iStar's tissue detection (get_mask.py) finds no tissue. False (default): the
        run stops with an error before imputation. True: the whole image, background included, is
        treated as tissue; the payload then sets params.used_fallback=true and
        summary.tissue_mask='all_true_fallback'.

    Outputs (output_files; a path iStar did not write is null)
    -------
    imputed_dir: cnts-super/ (one super-resolution array per gene); imputed_plots_dir:
    cnts-super-plots/; clusters_gene_dir / clusters_gene_labels: clusters-gene/ and its
    labels.pickle; cluster_expression_dir: cnts-clustered/by-clusters/; differential_dir:
    cnts-clustered/by-clusters/contrast/; plot_spots_dir: spots/; model_states_dir: states/;
    celltype_results_dir: markers/cell-type/ (with celltype_labels, celltype_label_names and
    celltype_enrichment = clusters-gene/enrichment.csv); signature_results_dir: markers/, holding
    signature_scores (signature-score.pickle) and signature_scores_plot (signature-score.png).
    """
    kw = {
        "prefix": prefix,
        "output_dir": output_dir,
        "device": device,
        "pixel_size": pixel_size,
        "n_genes": n_genes,
        "impute_epochs": impute_epochs,
        "n_clusters": n_clusters,
        "filter_size": filter_size,
        "min_cluster_size": min_cluster_size,
        "run_celltype": run_celltype,
        "celltype_marker_file": celltype_marker_file,
        "run_signature_scores": run_signature_scores,
        "signature_file": signature_file,
        "allow_mask_fallback": allow_mask_fallback,
    }
    # Convert kwargs to CLI flags: key -> --key (snake_case -> kebab-case)
    args = []
    for k, v in kw.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        args.extend([flag, str(v)])

    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
