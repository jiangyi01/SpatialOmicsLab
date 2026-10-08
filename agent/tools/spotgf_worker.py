#!/usr/bin/env python
"""
SpotGF worker script for SpatialOmicsLab MCP integration.

Runs inside /opt/conda/envs/SpotGF and does the heavy lifting:

- Calls SpotGF.py on a GEM-like SRT file.
- Forwards SpotGF logs to stderr.
- Collects the output files this run wrote and returns a JSON summary on stdout.

IMPORTANT:
- All progress / logging is printed to stderr (prefixed with [spotgf-worker]).
- The ONLY thing printed to stdout is a single line of JSON with a summary
  of the run, so that the FastMCP wrapper can parse it safely.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any

from worker_utils import WorkerOutput, record_ignored

# We are already running in /opt/conda/envs/SpotGF, so just use this interpreter
PYTHON_EXEC = sys.executable

# The files SpotGF.py writes into its output directory (it chdirs there first), by the key the payload
# lists each under. The names are fixed, so a file of the same name left by an earlier run into the same
# directory cannot be told apart by name: each is compared by stat before and after the run instead.
FIXED_OUTPUTS = (
    ("scores_file", "SpotGF_scores.txt"),
    ("auto_threshold_gem", "SpotGF_auto_threshold.gem"),
    ("alpha_shape_png", "alpha_shape.png"),
    ("spatial_automatic_png", "Spatial_automatic.png"),
    ("spatial_proportion_png", "Spatial_proportion.png"),
    ("violinplot_n_genes_by_counts_png", "Violinplot_n_genes_by_counts.png"),
    ("violinplot_total_counts_png", "Violinplot_total_counts.png"),
)
PROPORTION_GEM_KEY = "proportion_gem_0"

# SpotGF.expression_figure draws a spatial figure only for a denoised GEM with more than this many genes.
FIGURE_MIN_GENES = 200

# SpotGF.generate_GFgem ends, whatever the switches, with scanpy's calculate_qc_metrics(percent_top=(50, 100,
# 200)) on the raw GEM, and scanpy raises "Positions outside range of features" unless the data has at least
# max(percent_top) genes. That is after the scores and every denoised GEM were computed, so the worker
# checks the count first.
INPUT_MIN_GENES = 200

# Why visualize=False cannot be honoured while the auto threshold runs. SpotGF.generate_GFgem assigns
# ``adata_auto`` / ``adata_prop`` only inside ``if visualize == True`` (from expression_figure), then
# reads ``adata_auto.obs`` for its violin plots whenever ``auto_threshold == True``.
VISUALIZE_IGNORED_WHY = (
    "SpotGF builds its auto-threshold violin plots from the data its spatial-figure step returns, so "
    "upstream stops with NameError after writing the GEMs when auto_threshold=True and visualize=False; "
    "the figures were drawn. Pass auto_threshold=False to run without figures."
)

# Why lower / upper / max_iterations do nothing when alpha is set. SpotGF.calculate_GFscore reads them only in
# ``if alpha == 0: alpha_use = alphashape.optimizealpha(all_cell, max_iterations, lower, upper)``; with any other
# alpha the tissue outline is ``alphashape.alphashape(all_cell, alpha)`` and the three are never used.
ALPHA_SEARCH_IGNORED_WHY = (
    "they bound alphashape.optimizealpha, the search SpotGF runs for the tissue-outline alpha only when "
    "alpha == 0; this run passed alpha={alpha}, which SpotGF used as the outline alpha directly, so no "
    "search ran. Pass alpha=0 for the searched alpha these limits bound."
)
ALPHA_SEARCH_PARAMS = ("lower", "upper", "max_iterations")

# SpotGF scores only the genes in more than this many GEM rows (preparedata: value_counts() > 10); the rest
# are carried into every denoised GEM unscored.
SCORED_MIN_ROWS = 10

# The one upstream stop that comes after every result is on disk. With auto_threshold=True,
# generate_GFgem ends by building violin plots from ``adata_auto.obs`` / ``adata_prop.obs`` -- the data
# expression_figure returns, which is None for a denoised GEM of FIGURE_MIN_GENES genes or fewer -- so a
# small denoised GEM stops it with "'NoneType' object has no attribute 'obs'" after SpotGF_scores.txt and
# both GEMs were written. Matched on the traceback's own source line, so no other AttributeError qualifies.
VIOLIN_STOP_LINE = re.compile(r"adata_(?:auto|prop)\.obs")
VIOLIN_STOP_ERROR = "'NoneType' object has no attribute 'obs'"


def log(msg: str) -> None:
    print(f"[spotgf-worker] {msg}", file=sys.stderr, flush=True)


def resolve_spotgf_script() -> str:
    """Locate ``SpotGF.py`` in the hand-cloned SpotGF checkout this worker shells out to.

    SpotGF ships no console entry point, so the run is ``python <checkout>/SpotGF.py``. Nothing in
    ``setup/`` creates that checkout, which makes its location a property of the machine rather than
    of this repo -- hence ``SPOTGF_SRC`` first, with the layout this was written on as the default.
    Resolved lazily so importing the worker still works on a box that has not cloned SpotGF.
    """
    candidates = [
        os.environ.get("SPOTGF_SRC") or "",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "SpotGF"),
        "/opt/SpotGF",
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "SpotGF.py")):
            return os.path.join(candidate, "SpotGF.py")
    searched = ", ".join(c for c in candidates if c)
    raise FileNotFoundError(
        f"Could not find SpotGF.py (looked under: {searched}). Clone the SpotGF repository and set "
        "SPOTGF_SRC to the directory containing SpotGF.py."
    )


def upstream_bool(value: bool) -> str:
    """The argv value SpotGF.py reads back as ``value``.

    SpotGF.py declares ``-auto_threshold`` and ``-v`` with argparse ``type=bool``, which calls ``bool()``
    on the string: ``bool("False")`` is True, so passing "False" switched both on. The empty string is the
    one value ``bool()`` reads as False.
    """
    return "True" if value else ""


def proportion_gem_name(proportion: float) -> str:
    """The name SpotGF.py gives the proportion GEM: ``'SpotGF_proportion_' + str(proportion) + '.gem'``.

    Upstream parses ``-p`` with ``type=float`` and the worker passes ``str(float)``, so the round trip is
    exact: 0.5 -> ``SpotGF_proportion_0.5.gem``, 1 -> ``SpotGF_proportion_1.0.gem``.
    """
    return "SpotGF_proportion_" + str(float(proportion)) + ".gem"


def _stat_key(path: str):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def snapshot(paths) -> dict:
    """stat of each path before the run (None when absent)."""
    return {p: _stat_key(p) for p in paths}


def written_by_this_run(path: str, before: dict) -> bool:
    """True when ``path`` exists now and was absent, or different, before the run."""
    now = _stat_key(path)
    return now is not None and before.get(path) != now


def _sniff_delimiter(path: str) -> str:
    """The delimiter SpotGF.open_gem would use: csv.Sniffer on the first 1024 characters."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        sample = fh.read(1024)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return "\t"


def gem_dimensions(path: str):
    """(n_genes, n_positions, n_scorable) of an input GEM, read the way SpotGF.py reads it (sniffed delimiter,
    '#' comments). ``n_scorable`` is the number of genes SpotGF scores: those in more than SCORED_MIN_ROWS GEM
    rows (binning moves rows, it does not merge them before that count)."""
    import pandas as pd

    df = pd.read_csv(path, sep=_sniff_delimiter(path), comment="#", usecols=lambda c: c in ("geneID", "x", "y"))
    n_genes = n_scorable = None
    if "geneID" in df.columns:
        rows_per_gene = df["geneID"].value_counts()
        n_genes = int(len(rows_per_gene))
        n_scorable = int((rows_per_gene > SCORED_MIN_ROWS).sum())
    n_spots = int(df.groupby(["x", "y"]).ngroups) if {"x", "y"} <= set(df.columns) else None
    return n_genes, n_spots, n_scorable


def n_genes_in_output(path: str) -> int:
    """Distinct genes in a GEM or score table SpotGF wrote (tab-separated, with a geneID column)."""
    import pandas as pd

    return int(pd.read_csv(path, sep="\t", usecols=["geneID"])["geneID"].nunique())


def plain_input(gem_path: str, output_dir: str):
    """Return (path SpotGF can read, scratch path to remove or None).

    SpotGF.open_gem sniffs the delimiter from ``open(path, 'r').read(1024)`` before it reaches its own
    ``.gz`` branch, so a gzipped GEM stops it with UnicodeDecodeError. A ``.gz`` input is decompressed to a
    hidden scratch file in ``output_dir`` first, and that file is removed after the run.
    """
    if not gem_path.endswith(".gz"):
        return gem_path, None
    fd, scratch = tempfile.mkstemp(prefix=".spotgf_input_", suffix=".gem", dir=output_dir)
    with os.fdopen(fd, "wb") as dst, gzip.open(gem_path, "rb") as src:
        shutil.copyfileobj(src, dst, 1 << 20)
    return scratch, scratch


def is_violin_step_stop(stderr: str) -> bool:
    """True when SpotGF stopped building its violin plots from a figure it did not draw (see VIOLIN_STOP_LINE)."""
    text = stderr or ""
    return VIOLIN_STOP_ERROR in text and VIOLIN_STOP_LINE.search(text) is not None


def failure_message(returncode: int, stderr: str) -> str:
    """The upstream exit code, the tail of its stderr, and what to change for the one known upstream stop."""
    tail = "\n".join((stderr or "").strip().splitlines()[-15:])[-2000:]
    msg = f"SpotGF.py exited with code {returncode}."
    if "Positions outside range of features" in (stderr or ""):
        msg += (
            f" SpotGF's final QC step (scanpy calculate_qc_metrics, percent_top up to {INPUT_MIN_GENES}) needs at "
            f"least {INPUT_MIN_GENES} distinct genes in the input GEM."
        )
    if "'NoneType' object has no attribute 'obs'" in (stderr or ""):
        msg += (
            f" SpotGF draws a spatial figure only for a denoised GEM with more than {FIGURE_MIN_GENES} genes, and "
            "with auto_threshold=True it then reads that figure's data for its violin plots, so a denoised GEM "
            f"of {FIGURE_MIN_GENES} genes or fewer stops it there. The GEMs and SpotGF_scores.txt are written "
            "before that step; pass auto_threshold=False to finish without the violin plots."
        )
    if tail:
        msg += f" SpotGF stderr (last lines):\n{tail}"
    return msg


def run_spotgf_pipeline(
    gem_path: str,
    output_dir: str,
    binsize: int,
    proportion: float,
    lower: float,
    upper: float,
    max_iterations: int,
    auto_threshold: bool,
    visualize: bool,
    spot_size: int,
    alpha: float,
) -> dict[str, Any]:
    if not 0.0 < float(proportion) <= 1.0:
        raise ValueError(
            f"proportion must be in (0, 1]; got {proportion}. It is the share of the scored genes the "
            "proportion GEM keeps."
        )

    os.makedirs(output_dir, exist_ok=True)

    gem_path = os.path.abspath(gem_path)
    output_dir = os.path.abspath(output_dir)

    out = WorkerOutput("spotgf", task="denoising")
    visualize_effective = bool(visualize) or bool(auto_threshold)
    if visualize_effective and not visualize:
        record_ignored(out, "visualize", VISUALIZE_IGNORED_WHY)
    # Upstream parses -a with type=float and tests ``alpha == 0``; the same comparison here.
    alpha_searched = float(alpha) == 0.0
    if not alpha_searched:
        record_ignored(out, list(ALPHA_SEARCH_PARAMS), ALPHA_SEARCH_IGNORED_WHY.format(alpha=alpha))

    log(f"Input GEM: {gem_path}")
    log(f"Output dir: {output_dir}")
    log(
        "Parameters: "
        f"binsize={binsize}, proportion={proportion}, lower={lower}, "
        f"upper={upper}, max_iterations={max_iterations}, "
        f"auto_threshold={auto_threshold}, visualize={visualize} (effective {visualize_effective}), "
        f"spot_size={spot_size}, alpha={alpha}"
    )

    # Input dimensions, read the way SpotGF reads the file; also the one upstream limit that can be checked
    # before the run instead of after it.
    n_genes_gem = n_spots_gem = n_scorable = None
    try:
        n_genes_gem, n_spots_gem, n_scorable = gem_dimensions(gem_path)
    except Exception as exc:  # descriptive only: SpotGF reports an unreadable file itself
        out.add_warning(f"could not count the genes/positions of the input GEM: {type(exc).__name__}: {exc}")
    if n_genes_gem is not None and n_genes_gem < INPUT_MIN_GENES:
        raise ValueError(
            f"SpotGF needs at least {INPUT_MIN_GENES} distinct genes in the input GEM and this one has "
            f"{n_genes_gem}: SpotGF ends every run with scanpy's calculate_qc_metrics(percent_top=(50, 100, 200)) "
            "on the raw data, which stops on fewer genes after all the scoring is done."
        )
    # generate_GFgem keeps int(n_scored * proportion) genes and reads the last of them, so a selection of 0
    # stops it with IndexError -- after the optimal-transport scoring, the slow part, is done.
    if n_scorable is not None and n_scorable == 0:
        raise ValueError(
            f"No gene in this GEM appears in more than {SCORED_MIN_ROWS} rows, and SpotGF scores only those, so it "
            "has no gene to score or keep. The GEM needs genes observed at more positions."
        )
    if n_scorable is not None and int(n_scorable * float(proportion)) < 1:
        raise ValueError(
            f"proportion={proportion} keeps int({n_scorable} x {proportion}) = 0 of the {n_scorable} genes SpotGF "
            f"scores (those in more than {SCORED_MIN_ROWS} GEM rows), and SpotGF stops on an empty selection "
            f"(IndexError) after all the scoring. Pass proportion of at least 1/{n_scorable} "
            f"(about {1.0 / n_scorable:.4g})."
        )

    expected = {key: os.path.join(output_dir, name) for key, name in FIXED_OUTPUTS}
    expected[PROPORTION_GEM_KEY] = os.path.join(output_dir, proportion_gem_name(proportion))
    before = snapshot(expected.values())
    earlier_proportion_gems = set(glob.glob(os.path.join(output_dir, "SpotGF_proportion_*.gem")))

    run_input, scratch = plain_input(gem_path, output_dir)
    try:
        cmd = [
            PYTHON_EXEC,
            resolve_spotgf_script(),
            "-i",
            run_input,
            "-o",
            output_dir,
            "-b",
            str(binsize),
            "-lower",
            str(lower),
            "-upper",
            str(upper),
            "-max_iterations",
            str(max_iterations),
            "-p",
            str(float(proportion)),
            "-auto_threshold",
            upstream_bool(auto_threshold),
            "-v",
            upstream_bool(visualize_effective),
            "-s",
            str(spot_size),
            "-a",
            str(alpha),
        ]

        log(f"Running SpotGF: {' '.join(repr(c) if c == '' else c for c in cmd)}")

        proc = subprocess.run(cmd, text=True, capture_output=True)
    finally:
        if scratch is not None:
            try:
                os.remove(scratch)
            except OSError:
                pass

    # Forward SpotGF stdout/stderr to our stderr for debugging
    if proc.stdout:
        log("=== SpotGF stdout ===")
        sys.stderr.write(proc.stdout)
        sys.stderr.flush()
    if proc.stderr:
        log("=== SpotGF stderr ===")
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()

    # Only what this run wrote is this run's output; a same-named file from an earlier run is not.
    produced: dict[str, Any] = {}
    stale = []
    for key, path in expected.items():
        if written_by_this_run(path, before):
            produced[key] = os.path.abspath(path)
        else:
            produced[key] = None
            if os.path.exists(path):
                stale.append(os.path.basename(path))

    # A stop in the violin-plot step leaves every result on disk; anything else, or a result missing, is a
    # failure. The partial run is reported as one (summary.upstream_completed, a warning, the analysis).
    upstream_completed = proc.returncode == 0
    if not upstream_completed:
        results_written = produced["scores_file"] is not None and produced[PROPORTION_GEM_KEY] is not None
        if auto_threshold:
            results_written = results_written and produced["auto_threshold_gem"] is not None
        if not (is_violin_step_stop(proc.stderr) and results_written):
            raise RuntimeError(failure_message(proc.returncode, proc.stderr))
    stale.extend(
        sorted(
            os.path.basename(p) for p in earlier_proportion_gems if os.path.abspath(p) != expected[PROPORTION_GEM_KEY]
        )
    )

    scores_file = produced["scores_file"]
    auto_gem = produced["auto_threshold_gem"]
    proportion_gem = produced[PROPORTION_GEM_KEY]

    if stale:
        out.add_warning(
            f"output_dir already held {len(stale)} SpotGF file(s) this run did not write ({', '.join(stale)}); "
            "they were left in place and are not listed as this run's outputs."
        )
    if auto_threshold and auto_gem is None:
        out.add_warning("auto_threshold=True, but SpotGF wrote no SpotGF_auto_threshold.gem in this run.")
    if proportion_gem is None:
        out.add_warning(f"SpotGF wrote no {proportion_gem_name(proportion)} in this run.")

    counts = {}
    for name, path in (
        ("n_genes_scored", scores_file),
        ("n_genes_kept_auto_threshold", auto_gem),
        ("n_genes_kept_proportion", proportion_gem),
    ):
        if path is None:
            continue
        try:
            counts[name] = n_genes_in_output(path)
        except Exception as exc:
            out.add_warning(f"could not count the genes in {os.path.basename(path)}: {type(exc).__name__}: {exc}")

    if not upstream_completed:
        small = [
            f"{label} ({counts[key]} genes)"
            for key, label in (
                ("n_genes_kept_auto_threshold", "the auto-threshold GEM"),
                ("n_genes_kept_proportion", "the proportion GEM"),
            )
            if key in counts and counts[key] <= FIGURE_MIN_GENES
        ]
        out.add_warning(
            f"SpotGF stopped (exit code {proc.returncode}, {VIOLIN_STOP_ERROR}) in its last step, the violin plots "
            "comparing the denoised GEMs with the raw data: it builds them from its spatial figures' data, and draws "
            f"a spatial figure only for a denoised GEM with more than {FIGURE_MIN_GENES} genes"
            + (
                f"; {' and '.join(small)} {'has' if len(small) == 1 else 'have'} {FIGURE_MIN_GENES} or fewer"
                if small
                else ""
            )
            + ". SpotGF_scores.txt and every denoised GEM were written before that step and are listed; "
            "Violinplot_*.png were not drawn. auto_threshold=False skips the violin step and the auto-threshold GEM."
        )

    missing_figures = []
    if visualize_effective:
        if auto_threshold and auto_gem is not None and produced["spatial_automatic_png"] is None:
            missing_figures.append("Spatial_automatic.png")
        if proportion_gem is not None and produced["spatial_proportion_png"] is None:
            missing_figures.append("Spatial_proportion.png")
    if missing_figures:
        out.add_warning(
            f"SpotGF draws a spatial figure only for a denoised GEM with more than {FIGURE_MIN_GENES} genes; "
            f"it wrote no {', '.join(missing_figures)} in this run."
        )

    data_kwargs = {}
    if n_spots_gem is not None:
        data_kwargs["n_spots"] = n_spots_gem
    if n_genes_gem is not None:
        data_kwargs["n_genes"] = n_genes_gem
    if data_kwargs:
        out.set_data(**data_kwargs)
    out.add_output_files({key: produced[key] for key, _ in FIXED_OUTPUTS})
    if proportion_gem is not None:
        out.add_output_file(PROPORTION_GEM_KEY, proportion_gem)
    params = {
        "input_gem": gem_path,
        "output_dir": output_dir,
        "binsize": binsize,
        "proportion": proportion,
        "lower": lower,
        "upper": upper,
        "max_iterations": max_iterations,
        "auto_threshold": auto_threshold,
        "visualize": visualize,
        "visualize_effective": visualize_effective,
        "spot_size": spot_size,
        "alpha": alpha,
        # Whether SpotGF searched for the outline alpha (alpha == 0), the only case lower / upper /
        # max_iterations take part in; otherwise they are listed in params.ignored.
        "alpha_searched": alpha_searched,
    }
    if scratch is not None:
        params["input_decompressed"] = True
    out.add_params(params)
    out.set_summary(
        has_auto_threshold_gem=auto_gem is not None,
        n_proportion_gems=1 if proportion_gem is not None else 0,
        has_scores_file=scores_file is not None,
        upstream_completed=upstream_completed,
        **counts,
    )
    if not upstream_completed:
        out.set_summary(upstream_stopped_in="violin plots")

    text = (
        f"SpotGF denoising {'completed' if upstream_completed else 'wrote its scores and denoised GEMs'} for input "
        f"GEM {os.path.basename(gem_path)} (binsize={binsize}, proportion={proportion}). "
    )
    if "n_genes_scored" in counts:
        text += (
            f"SpotGF scored {counts['n_genes_scored']} genes (those in more than 10 GEM rows); genes in 10 rows "
            "or fewer are not scored and are kept in every denoised GEM. "
        )
    if "n_genes_kept_proportion" in counts:
        text += (
            f"The proportion GEM keeps the top {proportion} of the scored genes by SpotGF score plus those "
            f"unscored genes: {counts['n_genes_kept_proportion']} genes. "
        )
    if "n_genes_kept_auto_threshold" in counts:
        text += f"The auto-threshold GEM keeps {counts['n_genes_kept_auto_threshold']} genes. "
    text += (
        f"Produced {1 if auto_gem else 0} auto-threshold GEM and "
        f"{1 if proportion_gem else 0} proportion-filtered GEM in this run"
        + (" (auto_threshold=False)." if not auto_threshold else ".")
    )
    if not upstream_completed:
        text += (
            f" SpotGF then stopped in its violin-plot step (a denoised GEM of {FIGURE_MIN_GENES} genes or fewer "
            "has no spatial figure to build the violins from): the scores and GEMs above were written before it, "
            "and no violin plot was drawn."
        )
    if stale:
        text += f" {len(stale)} older SpotGF file(s) in output_dir were not written by this run and are not listed."
    out.set_analysis(text)

    out.emit()


def main() -> None:
    parser = argparse.ArgumentParser(description="SpotGF worker for SpatialOmicsLab MCP")
    parser.add_argument("--gem-path", required=True, help="Path to input GEM / SRT file.")
    parser.add_argument("--output-dir", required=True, help="Directory for SpotGF outputs.")
    parser.add_argument("--binsize", type=int, default=70)
    parser.add_argument("--proportion", type=float, default=0.5)
    parser.add_argument("--lower", type=float, default=0.0)
    parser.add_argument("--upper", type=float, default=100000.0)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--auto-threshold", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--visualize", type=lambda s: s.lower() == "true", default=True)
    parser.add_argument("--spot-size", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.0)

    args = parser.parse_args()

    try:
        run_spotgf_pipeline(
            gem_path=args.gem_path,
            output_dir=args.output_dir,
            binsize=args.binsize,
            proportion=args.proportion,
            lower=args.lower,
            upper=args.upper,
            max_iterations=args.max_iterations,
            auto_threshold=args.auto_threshold,
            visualize=args.visualize,
            spot_size=args.spot_size,
            alpha=args.alpha,
        )
    except Exception as e:
        log("ERROR occurred in SpotGF worker:")
        log(traceback.format_exc())
        WorkerOutput.emit_error("spotgf", str(e), task="denoising")
        sys.exit(1)


if __name__ == "__main__":
    main()
