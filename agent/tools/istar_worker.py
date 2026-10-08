#!/usr/bin/env python
"""
iStar worker script for SpatialOmicsLab MCP integration.

Runs INSIDE the /opt/conda/envs/istar environment and executes the
full iStar pipeline, closely following the reference run.sh:

  - copy raw inputs from prefix -> output_dir
  - write pixel-size.txt
  - rescale & preprocess histology
  - extract histology features
  - get tissue mask
  - select top variable genes
  - rescale coordinates & radius
  - impute super-resolution expression
  - plot imputed expression
  - cluster by gene features
  - aggregate & differential analysis
  - plot spot-level expression
  - optional: cell type inference and signature scores

Every step is one of iStar's own scripts, run as a subprocess; nothing here reimplements a step.
The outputs therefore sit where those scripts put them (``output_files`` names them):

  cnts-super/<gene>.pickle            super-resolution expression, one array per imputed gene
  cnts-super-plots/<gene>.png         plots of those arrays
  clusters-gene/labels.pickle         gene-feature cluster label per superpixel (-1 = not tissue)
  cnts-clustered/by-clusters/         per-cluster mean / variance / count / sum tables
  cnts-clustered/by-clusters/contrast/  differential expression, by cluster and by metric
  spots/<gene>.png                    spot-level expression plots
  markers/cell-type/                  cell-type labels (run_celltype)
  markers/signature-score.{pickle,png}  signature score (run_signature_scores; a file prefix upstream)

Conventions:
- All logs go to stderr, prefixed with [istar-worker].
- The ONLY thing printed to stdout is a single JSON line summarizing the run.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import traceback

from worker_utils import WorkerOutput, record_method, resolve_compute

#: What this worker runs, for ``params.method``: iStar's own scripts in run.sh order.
METHOD = (
    "iStar (upstream run.sh steps: HIPT histology features, get_mask tissue detection, "
    "super-resolution imputation, k-means on the imputed gene embeddings, cluster differential analysis)"
)

#: get_mask.py's ``min_connected``: tissue regions smaller than this many superpixels are merged away.
GET_MASK_MIN_CONNECTED = 4000

#: ``params.method`` when ``allow_mask_fallback`` let an all-True mask stand in for an empty one.
MASK_FALLBACK_METHOD = METHOD + "; tissue mask replaced by an all-True mask because get_mask.py found no tissue"


def log(msg: str) -> None:
    print(f"[istar-worker] {msg}", file=sys.stderr, flush=True)


def run_step(cmd, cwd: str | None = None, step_name: str | None = None) -> None:
    """Run a subprocess step, log stdout/stderr, and raise on non-zero exit."""
    name = step_name or " ".join(str(x) for x in cmd)
    log(f"Starting step: {name}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.stdout:
        for line in proc.stdout.splitlines():
            log(f"[{name} stdout] {line}")
    if proc.stderr:
        for line in proc.stderr.splitlines():
            log(f"[{name} stderr] {line}")

    if proc.returncode != 0:
        raise RuntimeError(f"Step '{name}' failed with return code {proc.returncode}")


def ensure_trailing_slash(path: str) -> str:
    if path.endswith("/") or path.endswith("\\"):
        return path
    return path + "/"


def resolve_istar_root() -> str:
    """Locate the iStar checkout whose ``.py`` scripts this worker drives as subprocesses.

    iStar is not pip-installable and nothing in ``setup/`` clones it, so the directory is a property
    of the machine. Every step below runs ``python <root>/<script>.py``, and a wrong root surfaces
    only as ``can't open file '.../rescale.py'`` behind a return code -- an error about the symptom
    that names no way to fix it. So: ``ISTAR_SRC`` first, then the layout this was written on, and a
    message that names the variable when none of them holds the scripts.
    """
    candidates = [
        os.environ.get("ISTAR_SRC") or "",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "istar"),
        "/opt/istar",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "rescale.py")):
            return candidate
    searched = ", ".join(c for c in candidates if c)
    raise FileNotFoundError(
        "Could not find the iStar source checkout (no rescale.py under any of: "
        f"{searched}). Clone https://github.com/daviddaiweizhang/istar and set ISTAR_SRC to the "
        "directory holding its scripts."
    )


#: ``$ISTAR_SRC`` or ``${ISTAR_SRC}`` in a prefix (not ``$ISTAR_SRC_X``).
_ISTAR_SRC_REF = re.compile(r"\$\{ISTAR_SRC\}|\$ISTAR_SRC(?![A-Za-z0-9_])")


def expand_istar_prefix(raw: str, istar_root: str) -> str:
    """A prefix with ``$ISTAR_SRC`` read as the iStar checkout found here, then ``~`` and other variables.

    The tool descriptions give the demo as ``prefix='$ISTAR_SRC/data/demo/'`` and the model passes it
    as written. ISTAR_SRC is not in the tool's environment, so ``abspath`` alone made it
    ``<cwd>/$ISTAR_SRC/data/demo`` and the demo was "not found" (hunt 2026-09-30, u30-uncovered-mcp-6).
    The variable now means the same root ``resolve_istar_root`` runs the scripts from.
    """
    expanded = _ISTAR_SRC_REF.sub(lambda _m: istar_root, raw or "")
    return os.path.expanduser(os.path.expandvars(expanded))


# ----------------------------------------------------------------------------- inputs


def read_marker_panel(path: str) -> dict:
    """``{label: [genes]}`` from a pixannot.py marker panel: a TSV with ``gene`` and ``label`` columns.

    pixannot.py reads the same file with ``pd.read_csv(sep='\\t')[['gene', 'label']]``; checking it
    here, before any step runs, turns a KeyError hours into the run into an error at the start.
    """
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        columns = list(reader.fieldnames or [])
        if "gene" not in columns or "label" not in columns:
            raise ValueError(
                f"celltype_marker_file {path} must be a tab-separated table with a 'gene' and a 'label' column "
                f"(pixannot.py reads exactly those two); its header is {columns}."
            )
        panel: dict = {}
        for row in reader:
            gene = (row.get("gene") or "").strip()
            label = (row.get("label") or "").strip()
            if gene and label:
                panel.setdefault(label, []).append(gene)
    if not panel:
        raise ValueError(f"celltype_marker_file {path} has a header but no gene/label rows.")
    return panel


def read_gene_list(path: str) -> list:
    """Non-empty lines of a one-gene-per-line file (gene-names.txt, a signature panel)."""
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def validate_optional_inputs(
    run_celltype: bool, celltype_marker_file: str | None, run_signature_scores: bool, signature_file: str | None
) -> None:
    """Refuse a cell-type or signature request that cannot run, before the expensive steps do.

    These checks used to sit after imputation, clustering and differential analysis had finished,
    so a missing panel threw all of that work away.
    """
    if run_celltype:
        if not celltype_marker_file:
            raise ValueError("run_celltype is True but celltype_marker_file is not provided.")
        if not os.path.isfile(celltype_marker_file):
            raise FileNotFoundError(f"celltype_marker_file not found: {os.path.abspath(celltype_marker_file)}")
        read_marker_panel(celltype_marker_file)
    if run_signature_scores:
        if not signature_file:
            raise ValueError("run_signature_scores is True but signature_file is not provided.")
        if not os.path.isfile(signature_file):
            raise FileNotFoundError(f"signature_file not found: {os.path.abspath(signature_file)}")
        if not read_gene_list(signature_file):
            raise ValueError(f"signature_file {signature_file} lists no genes.")


def check_panels_against_imputed_genes(
    gene_names_txt: str,
    celltype_marker_file: str | None,
    signature_file: str | None,
    n_genes: int,
) -> dict:
    """Which marker/signature genes iStar will impute, checked before imputation runs.

    pixannot.py and marker_score.py score a label from the markers that are among the imputed genes
    (``gene-names.txt``) and drop the others with a line on their stdout. A label with none of its
    markers imputed makes them stack an empty list and die -- after the whole pipeline has run. So:
    a label (or a signature) left with no imputed gene stops the run here, naming the knob, and the
    genes that are dropped are reported in the payload rather than only in a subprocess log.
    """
    imputed = set(read_gene_list(gene_names_txt))
    report: dict = {}
    if celltype_marker_file:
        panel = read_marker_panel(celltype_marker_file)
        not_imputed = {}
        empty = []
        for label, genes in panel.items():
            missing = [g for g in genes if g not in imputed]
            if missing:
                not_imputed[label] = missing
            if len(missing) == len(genes):
                empty.append(label)
        if empty:
            raise ValueError(
                f"None of the marker genes for cell type(s) {empty} are among the {len(imputed)} genes iStar "
                f"imputes (gene-names.txt: the n_genes={n_genes} most variable genes of cnts.tsv), so pixannot.py "
                "cannot score them. Raise n_genes (if cnts.tsv holds those genes), or drop those labels from "
                "celltype_marker_file."
            )
        report["celltype_markers_not_imputed"] = not_imputed
    if signature_file:
        genes = read_gene_list(signature_file)
        missing = [g for g in genes if g not in imputed]
        if len(missing) == len(genes):
            raise ValueError(
                f"None of the {len(genes)} genes in signature_file are among the {len(imputed)} genes iStar "
                f"imputes (gene-names.txt: the n_genes={n_genes} most variable genes of cnts.tsv), so "
                "marker_score.py has nothing to score. Raise n_genes (if cnts.tsv holds those genes), or use genes "
                "that are imputed."
            )
        report["signature_genes_not_imputed"] = missing
    return report


def count_table_dims(cnts_tsv: str) -> tuple:
    """``(n_spots, n_genes)`` of iStar's cnts.tsv: a header of gene names after the spot column."""
    n_rows = 0
    with open(cnts_tsv, encoding="utf-8") as fh:
        header = fh.readline()
        for line in fh:
            if line.strip():
                n_rows += 1
    n_genes = max(len(header.rstrip("\r\n").split("\t")) - 1, 0)
    return n_rows, n_genes


# ----------------------------------------------------------------------------- tissue mask


def check_tissue_mask(mask_path: str, allow_fallback: bool) -> dict:
    """Read get_mask.py's mask and refuse an empty one unless the caller allowed the substitute.

    iStar restricts clustering, the cluster differential analysis and its plots to this mask. When
    HIPT finds no tissue the mask is all False; the worker used to overwrite it with an all-True
    mask -- the whole image, background included, treated as tissue -- and say so only on stderr,
    while ``output_files['mask_small']`` presented the substitute as iStar's mask.
    """
    import numpy as np
    from PIL import Image

    mask = np.array(Image.open(mask_path)) > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    n_total = int(mask.size)
    n_tissue = int(mask.sum())
    if n_tissue:
        return {"tissue_mask": "get_mask", "n_tissue": n_tissue, "n_total": n_total}
    if not allow_fallback:
        small = ""
        if n_total < 4 * GET_MASK_MIN_CONNECTED:
            small = (
                f" get_mask.py merges connected regions smaller than {GET_MASK_MIN_CONNECTED} superpixels into their "
                f"surroundings, and this image has only {n_total} (16x16-pixel blocks of the image rescaled to "
                "pixel_size), so it can come back empty whatever it shows; a smaller pixel_size gives it more."
            )
        raise ValueError(
            f"iStar's tissue detection (get_mask.py) found no tissue: {mask_path} marks 0 of {n_total} "
            "superpixels as tissue, and clustering, the cluster differential analysis and the plots are all "
            "restricted to that mask. Check that he-raw.jpg is the H&E image the spots were captured on, at the "
            f"pixel size in pixel-size-raw.txt.{small} To run with the whole image, background included, treated "
            "as tissue, pass allow_mask_fallback=True; the payload then records it as a fallback."
        )
    # get_mask.py restores the margins it trims, so its mask is already the embedding grid's shape.
    full = np.ones(mask.shape, dtype=bool)
    partial = mask_path + ".partial"
    Image.fromarray(full).save(partial, format="PNG")
    os.replace(partial, mask_path)
    log(f"WARNING: tissue mask was empty; replaced with an all-True mask {full.shape} (allow_mask_fallback=True)")
    return {"tissue_mask": "all_true_fallback", "n_tissue": int(full.size), "n_total": int(full.size)}


# ----------------------------------------------------------------------------- outputs


def copy_dir_contents(src: str, dst: str) -> list:
    """Copy everything under ``src`` into ``dst``: what ``cp -r src/* dst/`` does in a shell.

    run.sh lifts pixannot.py's threshold010/ results into markers/cell-type/ with that shell glob.
    Passed to subprocess as an argument list, the ``*`` is never expanded, cp is asked for a file
    literally named ``*`` and fails -- which killed every run_celltype run after all the expensive
    steps had finished. Each file is written to ``.partial`` and moved into place.
    """
    copied = []
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        source = os.path.join(src, name)
        target = os.path.join(dst, name)
        if os.path.isdir(source):
            copied.extend(copy_dir_contents(source, target))
        else:
            partial = target + ".partial"
            shutil.copy2(source, partial)
            os.replace(partial, target)
            copied.append(target)
    return copied


def istar_output_files(run_prefix: str, run_celltype: bool, run_signature_scores: bool) -> dict:
    """Where iStar's scripts write their results, keyed for ``output_files``.

    The keys ``imputed_dir``, ``differential_dir`` and ``plot_spots_dir`` used to point at
    ``imputed/``, ``differential/`` and ``plots/``, which no iStar script creates. The paths below
    are the ones the scripts write (impute.py, plot_imputed.py, aggregate_imputed.py /
    reorganize_imputed.py, differential.py, plot_spots.py, pixannot.py, enrich.py, marker_score.py).
    """
    j = os.path.join
    clustered = j(run_prefix, "cnts-clustered", "by-clusters")
    files = {
        "embeddings_hist": j(run_prefix, "embeddings-hist.pickle"),
        "embeddings_gene": j(run_prefix, "embeddings-gene.pickle"),
        "mask_small": j(run_prefix, "mask-small.png"),
        "gene_names": j(run_prefix, "gene-names.txt"),
        "clusters_gene_dir": j(run_prefix, "clusters-gene"),
        "clusters_gene_labels": j(run_prefix, "clusters-gene", "labels.pickle"),
        "differential_dir": j(clustered, "contrast"),
        "cluster_expression_dir": clustered,
        "imputed_dir": j(run_prefix, "cnts-super"),
        "imputed_plots_dir": j(run_prefix, "cnts-super-plots"),
        "plot_spots_dir": j(run_prefix, "spots"),
        "model_states_dir": j(run_prefix, "states"),
        "celltype_results_dir": None,
        "celltype_labels": None,
        "celltype_label_names": None,
        "celltype_enrichment": None,
        "signature_results_dir": None,
        "signature_scores": None,
        "signature_scores_plot": None,
    }
    if run_celltype:
        celltype_dir = j(run_prefix, "markers", "cell-type")
        files["celltype_results_dir"] = celltype_dir
        files["celltype_labels"] = j(celltype_dir, "labels.pickle")
        files["celltype_label_names"] = j(celltype_dir, "label-names.txt")
        # enrich.py writes into its first argument, the gene-cluster directory.
        files["celltype_enrichment"] = j(run_prefix, "clusters-gene", "enrichment.csv")
    if run_signature_scores:
        # marker_score.py takes an output *prefix* and writes <prefix>.pickle and <prefix>.png.
        files["signature_results_dir"] = j(run_prefix, "markers")
        files["signature_scores"] = j(run_prefix, "markers", "signature-score.pickle")
        files["signature_scores_plot"] = j(run_prefix, "markers", "signature-score.png")
    return files


def keep_existing(files: dict) -> tuple:
    """``(files, missing_keys)``: a path nothing wrote is reported as None, never as a result."""
    present = {}
    missing = []
    for key, path in files.items():
        if path is not None and not os.path.exists(path):
            missing.append(key)
            path = None
        present[key] = path
    return present, missing


def cluster_sizes(labels_path: str) -> dict:
    """``{cluster: n_superpixels}`` from cluster.py's labels.pickle (-1 marks non-tissue)."""
    import pickle

    import numpy as np

    with open(labels_path, "rb") as fh:
        labels = np.asarray(pickle.load(fh))
    values, counts = np.unique(labels[labels >= 0], return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values.tolist(), counts.tolist())}


def describe_run(
    n_imputed: int,
    n_genes: int,
    n_table_genes: int,
    impute_epochs: int,
    sizes: dict | None,
    n_clusters: int,
    min_cluster_size: int,
    mask_info: dict,
    run_celltype: bool,
    run_signature_scores: bool,
) -> str:
    """The analysis text, from what the run produced rather than what it was asked for."""
    parts = [
        f"iStar pipeline completed: imputed {n_imputed} genes at super-resolution ({impute_epochs} epochs) "
        "from HIPT histology features."
    ]
    if n_imputed < n_genes:
        parts.append(f"n_genes={n_genes} was requested; cnts.tsv holds {n_table_genes} genes.")
    if sizes is None:
        parts.append(f"Gene-feature clustering ran with n_clusters={n_clusters}; its labels could not be read back.")
    else:
        n_found = len(sizes)
        n_pixels = sum(sizes.values())
        parts.append(f"Found {n_found} gene-feature clusters over {n_pixels} tissue superpixels.")
        if n_found != n_clusters:
            parts.append(
                f"k-means was asked for n_clusters={n_clusters}; {n_found} remain after connected regions smaller "
                f"than min_cluster_size={min_cluster_size} superpixels were merged into their neighbours."
            )
        else:
            parts.append(f"The count is the requested n_clusters={n_clusters}, not a number chosen from the data.")
    if mask_info["tissue_mask"] == "all_true_fallback":
        parts.append(
            "FALLBACK: get_mask.py found no tissue, so an all-True mask was used (allow_mask_fallback=True); "
            "background pixels were clustered and aggregated as tissue."
        )
    else:
        share = 100.0 * mask_info["n_tissue"] / max(mask_info["n_total"], 1)
        parts.append(f"The tissue mask from get_mask.py covers {share:.1f}% of the image.")
    if run_celltype:
        parts.append(
            "Cell types were inferred with pixannot.py (threshold 0.10) and tested for enrichment per cluster."
        )
    if run_signature_scores:
        parts.append("A signature score was computed with marker_score.py.")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="iStar worker for SpatialOmicsLab MCP (run iStar full pipeline).")
    parser.add_argument(
        "--prefix",
        required=True,
        help="Dataset prefix directory with raw inputs (e.g. agent/tools/third_party/istar/data/demo/).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where the full iStar pipeline will be run.",
    )
    parser.add_argument("--device", default="cpu", help="Computation device: 'cpu' or 'cuda'.")
    parser.add_argument("--pixel-size", type=float, default=0.5, help="Target pixel size written to pixel-size.txt.")
    parser.add_argument("--n-genes", type=int, default=1000, help="Number of most variable genes to impute.")
    parser.add_argument(
        "--impute-epochs", type=int, default=400, help="Number of epochs for imputation (impute.py --epochs)."
    )
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of clusters for gene-feature clustering.")
    parser.add_argument("--filter-size", type=int, default=8, help="Filter size for spatial smoothing in clustering.")
    parser.add_argument("--min-cluster-size", type=int, default=20, help="Minimum cluster size.")
    parser.add_argument("--run-celltype", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--celltype-marker-file", default=None)
    parser.add_argument("--run-signature-scores", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--signature-file", default=None)
    parser.add_argument(
        "--allow-mask-fallback",
        type=lambda s: s.lower() == "true",
        default=False,
        help="If get_mask.py finds no tissue, treat the whole image as tissue instead of stopping (a fallback).",
    )

    args = parser.parse_args()

    # Normalize paths. The prefix is resolved inside the try below, once the iStar root is known.
    output_dir = ensure_trailing_slash(os.path.abspath(args.output_dir))
    # The token is formatted into `--device=...` for two child processes (extract_features.py and
    # impute.py), so an unvalidated 'GPU' or a 'cuda' on a driverless box surfaces only as an opaque
    # non-zero exit from a subprocess. Resolve it here, against this machine's real hardware.
    device = resolve_compute(args.device).device
    pixel_size = args.pixel_size
    n_genes = args.n_genes
    impute_epochs = args.impute_epochs
    n_clusters = args.n_clusters
    filter_size = args.filter_size
    min_cluster_size = args.min_cluster_size
    run_celltype = args.run_celltype
    celltype_marker_file = args.celltype_marker_file
    run_signature_scores = args.run_signature_scores
    signature_file = args.signature_file
    allow_mask_fallback = args.allow_mask_fallback

    # Where we actually run iStar (all results live here)
    run_prefix = output_dir

    try:
        istar_root = resolve_istar_root()
        prefix = ensure_trailing_slash(os.path.abspath(expand_istar_prefix(args.prefix, istar_root)))
        log(f"iStar root: {istar_root}")
        log(f"Input prefix (raw data): {prefix}")
        log(f"Output dir (run_prefix): {run_prefix}")
        log(f"Device: {device}")
        log(
            f"pixel_size={pixel_size}, n_genes={n_genes}, "
            f"impute_epochs={impute_epochs}, n_clusters={n_clusters}, "
            f"filter_size={filter_size}, min_cluster_size={min_cluster_size}"
        )
        log(f"run_celltype={run_celltype}, celltype_marker_file={celltype_marker_file}")
        log(f"run_signature_scores={run_signature_scores}, signature_file={signature_file}")
        log(f"allow_mask_fallback={allow_mask_fallback}")

        # The optional panels are checked before any step runs, not after the pipeline has finished.
        validate_optional_inputs(run_celltype, celltype_marker_file, run_signature_scores, signature_file)
        celltype_marker_file_abs = os.path.abspath(celltype_marker_file) if run_celltype else None
        signature_file_abs = os.path.abspath(signature_file) if run_signature_scores else None

        # Ensure output directory exists
        os.makedirs(run_prefix, exist_ok=True)

        # Copy required raw files from prefix to run_prefix
        required_raw = [
            "he-raw.jpg",
            "cnts.tsv",
            "locs-raw.tsv",
            "pixel-size-raw.txt",
            "radius-raw.txt",
        ]
        for fname in required_raw:
            src = os.path.join(prefix, fname)
            dst = os.path.join(run_prefix, fname)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"Required raw file not found in prefix: {src}")
            if src != dst:
                log(f"Copying {src} -> {dst}")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                subprocess.run(["cp", src, dst], check=True)

        # 1) Write pixel-size.txt
        pixel_size_txt = os.path.join(run_prefix, "pixel-size.txt")
        log(f"Writing pixel size {pixel_size} to {pixel_size_txt}")
        with open(pixel_size_txt, "w", encoding="utf-8") as f:
            f.write(f"{pixel_size}\n")

        # 2) Preprocess histology image
        run_step(
            [sys.executable, os.path.join(istar_root, "rescale.py"), run_prefix, "--image"],
            step_name="rescale_image",
        )
        run_step(
            [sys.executable, os.path.join(istar_root, "preprocess.py"), run_prefix, "--image"],
            step_name="preprocess_image",
        )

        # 3) Extract histology features
        run_step(
            [
                sys.executable,
                os.path.join(istar_root, "extract_features.py"),
                run_prefix,
                f"--device={device}",
            ],
            step_name="extract_features",
        )

        # 4) Auto detect tissue mask
        embeddings_hist = os.path.join(run_prefix, "embeddings-hist.pickle")
        mask_small = os.path.join(run_prefix, "mask-small.png")
        run_step(
            [sys.executable, os.path.join(istar_root, "get_mask.py"), embeddings_hist, mask_small],
            step_name="get_mask",
        )

        # 4b) An empty mask stops the run unless allow_mask_fallback permits the all-True substitute.
        mask_info = check_tissue_mask(mask_small, allow_mask_fallback)

        # 5) Select most variable genes
        cnts_tsv = os.path.join(run_prefix, "cnts.tsv")
        gene_names_txt = os.path.join(run_prefix, "gene-names.txt")
        run_step(
            [
                sys.executable,
                os.path.join(istar_root, "select_genes.py"),
                f"--n-top={n_genes}",
                cnts_tsv,
                gene_names_txt,
            ],
            step_name="select_genes",
        )
        # The panels can only be scored on imputed genes; find out now, before imputation.
        panel_report = check_panels_against_imputed_genes(
            gene_names_txt, celltype_marker_file_abs, signature_file_abs, n_genes
        )

        # 6) Rescale coordinates and spot radius
        run_step(
            [sys.executable, os.path.join(istar_root, "rescale.py"), run_prefix, "--locs", "--radius"],
            step_name="rescale_locs_radius",
        )

        # 7) Train imputation model and predict at super-resolution
        run_step(
            [
                sys.executable,
                os.path.join(istar_root, "impute.py"),
                run_prefix,
                f"--epochs={impute_epochs}",
                f"--device={device}",
            ],
            step_name="impute",
        )

        # 8) Visualize imputed gene expression
        run_step(
            [sys.executable, os.path.join(istar_root, "plot_imputed.py"), run_prefix],
            step_name="plot_imputed",
        )

        # 9) Segment image by gene features (gene-based clustering)
        embeddings_gene = os.path.join(run_prefix, "embeddings-gene.pickle")
        clusters_gene_dir = os.path.join(run_prefix, "clusters-gene")
        run_step(
            [
                sys.executable,
                os.path.join(istar_root, "cluster.py"),
                f"--filter-size={filter_size}",
                f"--min-cluster-size={min_cluster_size}",
                f"--n-clusters={n_clusters}",
                f"--mask={mask_small}",
                embeddings_gene,
                clusters_gene_dir + "/",
            ],
            step_name="cluster_gene_features",
        )

        # 10) Differential analysis by clusters
        run_step(
            [sys.executable, os.path.join(istar_root, "aggregate_imputed.py"), run_prefix],
            step_name="aggregate_imputed",
        )
        run_step(
            [sys.executable, os.path.join(istar_root, "reorganize_imputed.py"), run_prefix],
            step_name="reorganize_imputed",
        )
        run_step(
            [sys.executable, os.path.join(istar_root, "differential.py"), run_prefix],
            step_name="differential",
        )

        # 11) Visualize spot-level gene expression
        run_step(
            [sys.executable, os.path.join(istar_root, "plot_spots.py"), run_prefix],
            step_name="plot_spots",
        )

        # 12) Optional: cell type inference
        if run_celltype:
            celltype_outdir = os.path.join(run_prefix, "markers", "cell-type")
            run_step(
                [
                    sys.executable,
                    os.path.join(istar_root, "pixannot.py"),
                    run_prefix,
                    celltype_marker_file_abs,
                    celltype_outdir + "/",
                ],
                step_name="pixannot",
            )
            # Lift the threshold010 results into the main folder, as run.sh's `cp -r .../threshold010/* ...`.
            threshold_dir = os.path.join(celltype_outdir, "threshold010")
            if not os.path.isdir(threshold_dir):
                raise RuntimeError(
                    f"pixannot.py finished without writing {threshold_dir}, the threshold-0.10 labels enrich.py reads."
                )
            copied = copy_dir_contents(threshold_dir, celltype_outdir)
            log(f"Copied {len(copied)} threshold010 file(s) into {celltype_outdir}")
            # Enrichment
            run_step(
                [
                    sys.executable,
                    os.path.join(istar_root, "enrich.py"),
                    clusters_gene_dir + "/",
                    celltype_outdir + "/",
                ],
                step_name="celltype_enrich",
            )

        # 13) Optional: signature scores
        if run_signature_scores:
            # marker_score.py treats its third argument as a file prefix: <prefix>.pickle / <prefix>.png.
            signature_prefix = os.path.join(run_prefix, "markers", "signature-score")
            run_step(
                [
                    sys.executable,
                    os.path.join(istar_root, "marker_score.py"),
                    run_prefix,
                    signature_file_abs,
                    signature_prefix,
                ],
                step_name="marker_score",
            )

        # Build JSON summary from what the run wrote.
        n_spots, n_table_genes = count_table_dims(cnts_tsv)
        n_imputed = len(read_gene_list(gene_names_txt))
        files, missing = keep_existing(istar_output_files(run_prefix, run_celltype, run_signature_scores))
        sizes = None
        warnings = []
        try:
            sizes = cluster_sizes(os.path.join(clusters_gene_dir, "labels.pickle"))
        except Exception as exc:  # the labels are reported, not required; say why they are absent
            warnings.append(f"could not read clusters-gene/labels.pickle ({type(exc).__name__}: {exc})")
        if missing:
            warnings.append(
                "iStar did not write these expected outputs, so they are reported as null: " + ", ".join(missing)
            )

        out = WorkerOutput("istar", task="full_pipeline")
        out.set_data(n_spots=n_spots, n_genes=n_table_genes)
        out.add_params(
            {
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
                "input_prefix": prefix,
                "output_dir": run_prefix,
            }
        )
        if mask_info["tissue_mask"] == "all_true_fallback":
            record_method(
                out,
                MASK_FALLBACK_METHOD,
                used_fallback=True,
                why="allow_mask_fallback=True; background pixels are clustered and aggregated as tissue",
            )
        else:
            record_method(out, METHOD)
        out.add_output_files(files)
        steps_run = [
            "feature_extraction",
            "tissue_mask",
            "gene_selection",
            "imputation",
            "imputed_plots",
            "clustering",
            "differential",
            "spot_plots",
        ]
        if run_celltype:
            steps_run.append("celltype_inference")
        if run_signature_scores:
            steps_run.append("signature_scores")
        out.set_summary(
            n_clusters=len(sizes) if sizes is not None else None,
            n_clusters_requested=n_clusters,
            cluster_sizes=sizes,
            n_genes_imputed=n_imputed,
            tissue_mask=mask_info["tissue_mask"],
            tissue_superpixels=mask_info["n_tissue"],
            mask_superpixels=mask_info["n_total"],
            steps_completed=steps_run,
            **panel_report,
        )
        out.set_analysis(
            describe_run(
                n_imputed,
                n_genes,
                n_table_genes,
                impute_epochs,
                sizes,
                n_clusters,
                min_cluster_size,
                mask_info,
                run_celltype,
                run_signature_scores,
            )
        )
        out.add_warnings(warnings)
        out.emit()

    except Exception as e:
        log("Exception during iStar pipeline execution:")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("istar", str(e), task="full_pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
