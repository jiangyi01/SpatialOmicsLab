#!/usr/bin/env python

"""
spagft_worker.py

Worker that runs the official SpaGFT package (``SpaGFT.detect_svg``) on one spatial transcriptomics
AnnData (.h5ad) to find spatially variable genes.

- Executed inside the SpaGFT conda env: /opt/conda/envs/spagft_env
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

What is read, and what is not:

* ``layer_key`` names the count matrix: ``adata.layers[layer_key]`` when the object has that layer,
  otherwise ``adata.X``; ``use_raw_counts`` selects ``adata.raw.X`` instead (``params.expression_source``
  says which). The matrix is then ``normalize_total`` + ``log1p``-transformed, as SpaGFT's tutorials do,
  so it has to hold counts (the shared ``worker_utils.choose_counts_matrix`` rule): negative or
  non-finite values (scaled / z-scored data) are refused, naming ``use_raw_counts`` when ``adata.raw``
  holds counts -- on a CELLxGENE Skin slide the transform turned 7.27M stored values into NaN and
  SpaGFT then crashed inside ``preprocessing.normalize`` -- and a non-negative non-integer matrix
  (already normalised) runs with a warning that it is not raw counts. ``params.x_matrix_kind`` says what
  ``X`` held.
* ``spatial_key`` names the coordinates the KNN graph is built on: ``adata.obsm[spatial_key]``, or
  two obs columns joined by a comma (``'array_row,array_col'``, SpaGFT's tutorial convention for
  Visium). Earlier revisions handed SpaGFT ``obs['array_row'/'array_col']`` whenever those columns
  existed, so on every Visium object the graph was built on the array grid whatever ``spatial_key``
  said; the grid is now read only when it is the key asked for.
* Spots flagged ``obs['in_tissue'] == 0`` are background, not tissue, and are left out before the
  graph is built (counted in ``data``, ``params`` and the analysis text). Nothing is subsampled.
* SpaGFT sets its own KNN size (``ceil(sqrt(n_spots) / 2)``, 4 when there are at most 500 spots),
  its own low/high-frequency mode counts (``ceil(sqrt(n_spots))`` each) and its own gene filter
  (detected in at least one spot), and draws no random number. ``n_neighbors``, ``n_eigs``,
  ``low_freq_fraction``, ``min_detected_frac`` and ``seed`` therefore do not reach it: they are
  listed in ``params.ignored`` with a warning, and the values SpaGFT used are in ``params``.
  Those five configure only the from-scratch substitute below.

The substitute -- a low-frequency energy ratio on a normalized-Laplacian KNN graph, written here and
not part of SpaGFT -- runs only with ``--allow-gft-fallback`` and only when SpaGFT (with ``kneed`` and
``statsmodels``, which ``detect_svg`` imports) cannot be imported. The payload then says so in
``params.method`` / ``params.used_fallback``. Without the flag the run fails and names what is missing.
It scores the same ``normalize_total`` + ``log1p`` matrix, with each gene centred, so a gene's
expression level is not read as low-frequency energy; it writes ``spagft_graph_spectrum.npz`` and no
``predicted_genes.json`` (it applies no significance test).

``n_top_genes`` (at least 1) caps the top table and ``predicted_genes.json``; a spot with no finite
coordinate among those analysed is refused with the count rather than handed to the KNN graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_svg_analysis,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    make_names_unique_and_report,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
)

METHOD_OFFICIAL = "SpaGFT.detect_svg (graph Fourier transform; kneedle cutoff on gft_score)"
METHOD_FALLBACK = (
    "from-scratch low-frequency energy ratio on a normalized-Laplacian KNN graph (not SpaGFT; "
    "SpaGFT could not be imported)"
)
# The knobs that configure only the from-scratch substitute; SpaGFT.detect_svg sets its own.
FALLBACK_ONLY_PARAMS = ("n_neighbors", "n_eigs", "low_freq_fraction", "min_detected_frac", "seed")
# The obsm key the validated coordinates are handed to SpaGFT under (detect_svg reads
# ``adata.obsm[spatial_info]`` when ``spatial_info`` is an obsm key).
GRAPH_COORDS_KEY = "_sog_spagft_graph_xy"
# MERFISH negative-control barcodes: never a gene, so never a predicted SVG.
CONTROL_PROBE_PREFIX = "Blank-"
# Default count-matrix layer; its absence is the ordinary case (most objects keep counts in X).
DEFAULT_LAYER_KEY = "counts"

_GIB = float(1 << 30)
# Bytes per KNN edge while SpaGFT builds the graph (sklearn kneighbors_graph + networkx), measured
# in spagft_env on 3,000-20,000-spot grids.
_BYTES_PER_GRAPH_EDGE = 100


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    sys.stderr.write(f"[spagft-worker] {msg}\n")
    sys.stderr.flush()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SpaGFT worker: identify spatially-varying genes via graph Fourier analysis."
    )

    parser.add_argument(
        "--task",
        type=str,
        default="svg",
        help="Task name (reserved for future extension). Currently only 'svg' is implemented.",
    )
    parser.add_argument(
        "--st-h5ad",
        type=str,
        required=True,
        help="Path to spatial AnnData (.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save SpaGFT results.",
    )
    parser.add_argument(
        "--layer-key",
        type=str,
        default=DEFAULT_LAYER_KEY,
        help="AnnData layer holding raw counts; adata.X is used when the object has no such layer.",
    )
    parser.add_argument(
        "--spatial-key",
        type=str,
        default="spatial",
        help=(
            "AnnData .obsm key with the 2D coordinates the KNN graph is built on (e.g. 'spatial'), or two obs "
            "columns joined by a comma (e.g. 'array_row,array_col')."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=20,
        help="KNN size of the from-scratch substitute only; SpaGFT sets its own (listed in params.ignored).",
    )
    parser.add_argument(
        "--n-eigs",
        type=int,
        default=50,
        help="Eigenpairs of the from-scratch substitute only; SpaGFT sets its own (listed in params.ignored).",
    )
    parser.add_argument(
        "--low-freq-fraction",
        type=float,
        default=0.1,
        help="Low-frequency share of the from-scratch substitute only (listed in params.ignored under SpaGFT).",
    )
    parser.add_argument(
        "--min-detected-frac",
        type=float,
        default=0.05,
        help="Gene detection filter of the from-scratch substitute only; SpaGFT keeps genes seen in >= 1 spot.",
    )
    parser.add_argument(
        "--n-top-genes",
        type=int,
        default=200,
        help="Number of top-ranked significant genes written to the top table and predicted_genes.json (>= 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the from-scratch substitute only; SpaGFT draws no random number (params.ignored).",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        default=False,
        help=(
            "Analyse adata.raw.X instead of X or the layer_key layer (CELLxGENE exports keep the counts there); "
            "refused when the object has no adata.raw or it does not hold counts."
        ),
    )
    parser.add_argument(
        "--allow-gft-fallback",
        action="store_true",
        default=False,
        help=(
            "Accept the from-scratch low-frequency energy ratio (not SpaGFT) when SpaGFT cannot be "
            "imported. Off by default: the run then fails and names what is missing."
        ),
    )

    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- small helpers


def _import_spagft():
    """``(SpaGFT module or None, why)``.

    ``detect_svg`` imports ``kneed`` and ``statsmodels`` lazily, at the very end of the run; a missing
    one surfaced there as an ImportError that the old ``except ImportError`` around the whole official
    call caught -- after SpaGFT had done all its work -- and answered by silently running the substitute
    under SpaGFT's name. Checked up front instead, so the gate below decides.
    """
    try:
        import kneed  # type: ignore  # noqa: F401
        import SpaGFT  # type: ignore
        from statsmodels.stats import multitest  # type: ignore  # noqa: F401
    except Exception as exc:  # an unimportable build raises more than ImportError (numpy ABI, ...)
        return None, f"{type(exc).__name__}: {exc}"
    return SpaGFT, ""


def upstream_graph_settings(n_spots: int, ratio_neighbors: float = 1.0) -> dict:
    """The KNN size and mode counts ``SpaGFT.detect_svg`` derives, by its own formulas (gft.py).

    ``ratio_low_freq = ratio_high_freq = 'infer'`` gives ``ceil(sqrt(n))`` modes each;
    ``ratio_neighbors`` gives ``K = ceil(sqrt(n) / 2 * ratio)``, overridden to 4 at <= 500 spots.
    """
    root = math.sqrt(int(n_spots))
    k = int(math.ceil(root / 2.0 * float(ratio_neighbors)))
    if int(n_spots) <= 500:
        k = 4
    n_modes = int(math.ceil(root))
    return {"num_neighbors": k, "n_low_frequency_modes": n_modes, "n_high_frequency_modes": n_modes}


def estimate_detect_svg_bytes(n_spots: int, n_genes: int, itemsize: int) -> int:
    """Peak bytes ``SpaGFT.detect_svg`` needs beyond the sparse input.

    Intrinsic to the method, not to this wrapper: detect_svg densifies the whole spots x genes
    matrix (``adata.X.toarray()``), z-scores a copy of it (``preprocessing.scale``) and multiplies it by
    the float64 Fourier modes, which casts it to float64 once more -- ``n x g x (2 * itemsize + 8)``
    bytes (measured at 3.8-3.9x the float32 dense size in spagft_env). Add the KNN graph and the
    ARPACK eigenvector blocks.
    """
    s = upstream_graph_settings(n_spots)
    dense = int(n_spots) * int(n_genes) * (2 * int(itemsize) + 8)
    graph = int(n_spots) * s["num_neighbors"] * _BYTES_PER_GRAPH_EDGE
    modes = int(n_spots) * (s["n_low_frequency_modes"] + s["n_high_frequency_modes"]) * 8 * 2
    return int(dense + graph + modes)


def check_detect_svg_memory(n_spots: int, n_genes: int, itemsize: int, available=None):
    """Refuse, with the numbers, when SpaGFT's dense intermediates cannot fit. Never subsamples.

    ``available`` defaults to ``worker_utils.available_memory_bytes()``: the smaller of ``MemAvailable``
    and the room under the cgroup limit, with the cgroup's page cache (the ``active_file`` /
    ``inactive_file`` LRU in ``memory.stat``) counted as reclaimable, so a container sitting at its limit
    on cache alone -- reading the input h5ad fills it -- is not refused. Returns ``(needed_bytes,
    available_bytes_or_None)``; when available memory cannot be read the check passes and says so in
    the log.
    """
    need = estimate_detect_svg_bytes(n_spots, n_genes, itemsize)
    if available is None:
        available = available_memory_bytes()
    if available is None:
        log(f"memory check skipped: available memory cannot be read here (estimated need {need / _GIB:.1f} GiB)")
        return need, None
    if need > available:
        raise MemoryError(
            f"SpaGFT.detect_svg densifies the whole {int(n_spots)} spots x {int(n_genes)} genes expression "
            f"matrix and z-scores a copy of it; with the KNN graph and Fourier modes that needs about "
            f"{need / _GIB:.1f} GiB, but about {available / _GIB:.1f} GiB is available here (MemAvailable / "
            "room under the cgroup limit, page cache counted as reclaimable). Nothing was computed. The dense "
            "matrix is SpaGFT's own, so no parameter of this tool shrinks it, and this worker does not drop "
            "spots or genes to make it fit; run it where that much memory is free."
        )
    return need, available


def in_tissue_mask(obs):
    """``(keep_mask or None, note)`` for the Space Ranger ``in_tissue`` flag.

    Space Ranger / CELLxGENE exports can carry every array spot, and the off-tissue ones are
    background: a spatial-variability test over them finds the tissue edge (on the Fetal12W heart
    sample 3,009 of 4,992 spots are off the tissue). None when there is no usable flag: no column, or
    values that are not a clean 0/1 flag (then ``note`` says so and nothing is dropped on a guess).
    """
    import numpy as np
    import pandas as pd

    if "in_tissue" not in obs.columns:
        return None, ""
    flag = pd.to_numeric(pd.Series(np.asarray(obs["in_tissue"]), dtype=object), errors="coerce")
    values = set(flag.dropna().unique().tolist())
    if flag.isna().any() or not values <= {0, 1}:
        return None, (
            "obs['in_tissue'] is present but is not a 0/1 flag, so no spot was excluded by it; "
            "every spot was analysed as tissue"
        )
    keep = flag.to_numpy() == 1
    if not keep.any():
        raise ValueError(
            f"obs['in_tissue'] marks all {len(keep)} spots as off the tissue (in_tissue == 0), so nothing is left "
            "to analyse. Fix the flag, or drop the column if every spot is tissue."
        )
    return keep, ""


def _first_non_count(X):
    """A value showing ``X`` is not raw counts (negative or non-integer), or None. Samples <= 200k stored values."""
    import numpy as np
    from scipy.sparse import issparse

    vals = X.data if issparse(X) else np.asarray(X).ravel()
    vals = np.asarray(vals[:200000], dtype=np.float64)
    bad = vals[(vals < 0) | (np.abs(vals - np.round(vals)) > 1e-6)]
    return float(bad[0]) if bad.size else None


def _detected_gene_count(X) -> int:
    """Genes with a non-zero value in at least one spot -- the genes SpaGFT's min_cells=1 filter keeps."""
    import numpy as np
    from scipy.sparse import issparse

    if issparse(X):
        return int((np.asarray((X != 0).sum(axis=0)).ravel() > 0).sum())
    return int((np.count_nonzero(np.asarray(X), axis=0) > 0).sum())


def _write_csv_atomic(df, path: str, index: bool = False) -> None:
    tmp = path + ".partial"
    df.to_csv(tmp, index=index)
    os.replace(tmp, path)


def _write_json_atomic(obj, path: str) -> None:
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def graph_coordinates(adata, spatial_key: str, check_finite: bool = True):
    """``(coords, label)``: the coordinates ``spatial_key`` names, and how the payload should name them.

    ``spatial_key`` is an obsm key (validated by ``spatial_coords``: a missing or 3-column key is
    refused). Two obs column names joined by a comma -- ``'array_row,array_col'``, SpaGFT's tutorial
    convention for Visium, which the manual benchmark runner uses -- select those columns instead.
    That is the only way the obs grid is read: it is never substituted for an obsm key.

    With ``check_finite`` a spot with no finite coordinate is refused, with the count, whichever way
    the coordinates were named (an obsm NaN used to reach sklearn's KNN as an anonymous
    "Input contains NaN" after all the preprocessing). ``load_input`` validates the key with
    ``check_finite=False`` before it drops the off-tissue spots and checks the rest afterwards, so a
    background spot with no position does not refuse the tissue.
    """
    import numpy as np

    obsm = getattr(adata, "obsm", {})
    if spatial_key not in obsm and "," in spatial_key:
        cols = [c.strip() for c in spatial_key.split(",")]
        missing = [c for c in cols if c not in adata.obs.columns]
        if len(cols) != 2 or missing:
            raise KeyError(
                f"spatial_key={spatial_key!r} names obs columns {cols}, which needs exactly two columns that obs "
                f"has; missing: {missing}. obs columns: {list(adata.obs.columns)[:20]}; obsm keys: {list(obsm.keys())}"
            )
        coords = np.column_stack([np.asarray(adata.obs[c].astype(float), dtype=np.float64) for c in cols])
        label = "obs[{}]".format(", ".join(repr(c) for c in cols))
    else:
        coords, _ = spatial_coords(adata, spatial_key, want=2, tool="SpaGFT")
        label = f"obsm['{spatial_key}']"
    if check_finite:
        n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
        if n_bad:
            raise ValueError(
                f"{label} gives {n_bad} of {coords.shape[0]} analysed spot(s) no finite coordinate (NaN or inf), so "
                "they cannot be placed in the KNN graph. Nothing was computed. Fix those coordinates, or pass "
                "spatial_key naming coordinates every analysed spot has."
            )
    return coords, label


def _refuse_non_counts(kind: str, source: str, raw_kind) -> None:
    """Negative or non-finite values are not counts: normalize_total + log1p turns them into NaN."""
    if kind not in ("negative", "nonfinite"):
        return
    what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
    hint = (
        " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
        if raw_kind == "counts"
        else " Pass layer_key naming a raw-count layer, or supply an h5ad whose X (or adata.raw with "
        "use_raw_counts=True) holds raw counts."
    )
    raise ValueError(
        f"The analysed matrix ({source}) holds {what}, not counts. SpaGFT's preprocessing (normalize_total + "
        "log1p) treats it as counts and turns those values into NaN, on which SpaGFT fails; nothing was "
        "computed." + hint
    )


def load_input(st_h5ad: str, spatial_key: str, layer_key: str, use_raw_counts: bool = False) -> dict:
    """Read, deduplicate, drop off-tissue spots and put the requested count matrix in ``adata.X``.

    Returns the AnnData (``X`` = the matrix to analyse) plus what was done to it, for the payload.
    The matrix is ``adata.raw.X`` under ``use_raw_counts`` (``choose_counts_matrix``), else the
    ``layer_key`` layer when the object has it, else ``X``; negative or non-finite values are refused.
    """
    import scanpy as sc

    log(f"Loading spatial AnnData: {st_h5ad}")
    adata = sc.read_h5ad(st_h5ad)
    renamed = make_names_unique_and_report(adata)
    # Refuse a missing or 3-column key before any work; finiteness is checked on the spots analysed.
    graph_coordinates(adata, spatial_key, check_finite=False)

    warnings = []
    n_spots_supplied = int(adata.n_obs)
    keep, note = in_tissue_mask(adata.obs)
    if note:
        warnings.append(note)
    n_off_tissue = 0 if keep is None else int((~keep).sum())
    if n_off_tissue:
        log(f"Leaving out {n_off_tissue} of {n_spots_supplied} spots with in_tissue == 0")
        adata = adata[keep].copy()

    layers = list(adata.layers.keys())
    raw = getattr(adata, "raw", None)
    raw_kind = expression_matrix_kind(raw.X) if raw is not None else None
    x_matrix_kind = expression_matrix_kind(adata.X)
    ignored = []
    if use_raw_counts:
        # Refuses an object without adata.raw, or whose adata.raw is not counts.
        adata, _info = choose_counts_matrix(adata, True)
        # Report the renames of the matrix analysed (raw.X's genes), not X's plus raw.X's.
        renamed["n_genes_renamed"] = 0
        make_names_unique_and_report(adata, into=renamed, axes=("var",))
        source = "raw.X"
        if layer_key and layer_key in layers:
            ignored.append("layer_key")
    elif layer_key and layer_key in adata.layers:
        adata.X = adata.layers[layer_key]
        source = f"layers['{layer_key}']"
    else:
        source = "X"
        # The default naming a layer the object lacks is the ordinary case (counts kept in X) and is
        # worth a word only when other layers exist; a layer the caller NAMED and did not get is
        # always said, including on an object with no layers at all (that used to pass in silence).
        if layer_key and layer_key != "X" and (layers or layer_key != DEFAULT_LAYER_KEY):
            warnings.append(
                f"layer_key={layer_key!r} is not a layer of this object (layers: {layers}); adata.X was analysed "
                "instead. Pass layer_key naming the raw-count layer to analyse that one, or layer_key='X'."
            )
    log(f"Expression matrix: {source}")

    kind = x_matrix_kind if source == "X" else expression_matrix_kind(adata.X)
    _refuse_non_counts(kind, source, raw_kind)
    if kind == "nonnegative_noninteger":
        bad = _first_non_count(adata.X)
        held = f"it holds {bad:g}" if bad is not None else "it holds non-integer values"
        hint = (
            " adata.raw holds raw counts: pass use_raw_counts=True to run on them."
            if raw_kind == "counts" and source != "raw.X"
            else " Pass layer_key naming a raw-count layer if the object has one."
        )
        warnings.append(
            f"the analysed matrix ({source}) is not raw counts ({held}); normalize_total + log1p were "
            "applied to it anyway, which re-transforms an already normalized matrix." + hint
        )

    coords, coords_label = graph_coordinates(adata, spatial_key)
    return {
        "adata": adata,
        "coords": coords,
        "coords_label": coords_label,
        "expression_source": source,
        "n_spots_supplied": n_spots_supplied,
        "n_off_tissue": n_off_tissue,
        "warnings": warnings,
        "renamed": renamed,
        # What adata.X held (the matrix analysed may be a layer or adata.raw.X), and the knobs the matrix
        # choice left unread.
        "x_matrix_kind": x_matrix_kind,
        "use_raw_counts": bool(use_raw_counts),
        "ignored": ignored,
    }


def _refuse_nonfinite_after_log1p(adata) -> None:
    """A NaN or inf after normalize_total + log1p is refused with its count, before SpaGFT sees it."""
    import numpy as np
    from scipy.sparse import issparse

    values = adata.X.data if issparse(adata.X) else np.asarray(adata.X)
    n_bad = int(np.count_nonzero(~np.isfinite(values)))
    if n_bad:
        raise ValueError(
            f"normalize_total + log1p left {n_bad} non-finite value(s) in the analysed matrix; it does not hold "
            "counts SpaGFT's preprocessing can normalise. Nothing was computed."
        )


def _record_matrix_choice(out, loaded: dict) -> None:
    """params.x_matrix_kind / use_raw_counts, the in-tissue record, and a layer_key use_raw_counts overrode."""
    out.add_params({"x_matrix_kind": loaded["x_matrix_kind"], "use_raw_counts": loaded["use_raw_counts"]})
    record_in_tissue(out, loaded["n_spots_supplied"], loaded["n_off_tissue"])
    if loaded["ignored"]:
        record_ignored(
            out, loaded["ignored"], "use_raw_counts=True analyses adata.raw.X, so the layer_key layer was not read"
        )


# ----------------------------------------------------------------------------- the official SpaGFT path


def _run_official_spagft(
    SpaGFT,
    st_h5ad: str,
    output_dir: str,
    layer_key: str,
    spatial_key: str,
    n_top_genes: int,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Call ``SpaGFT.detect_svg`` on the requested matrix and coordinates.

    Manual reference: /workspace/hands_by_myself/runners/spagft_identify_svg.py (same detect_svg
    arguments; that runner builds its graph on obs array_row/array_col, this one on obsm[spatial_key]).
    Significance filter precedence: cutoff_gft_score -> fdr<0.05 -> pvalue<0.05.
    """
    import numpy as np
    import scanpy as sc

    loaded = load_input(st_h5ad, spatial_key, layer_key, use_raw_counts)
    adata = loaded["adata"]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _refuse_nonfinite_after_log1p(adata)

    # The graph is built on the coordinates the caller named. SpaGFT reads obsm[spatial_info] when
    # spatial_info is an obsm key; the old ['array_row', 'array_col'] made it read the obs grid instead.
    adata.obsm[GRAPH_COORDS_KEY] = np.asarray(loaded["coords"], dtype=np.float64)
    # Fourier modes a previous SpaGFT run stored in the file belong to that run's graph; detect_svg
    # would reuse them whenever their count matches.
    stale_modes = adata.uns.pop("FMs_after_select", None) is not None

    n_spots = int(adata.n_obs)
    n_genes_supplied = int(adata.n_vars)
    n_genes_detected = _detected_gene_count(adata.X)
    settings = upstream_graph_settings(n_spots)
    check_detect_svg_memory(n_spots, n_genes_detected, int(np.dtype(adata.X.dtype).itemsize))

    log(
        f"Calling SpaGFT.detect_svg on {n_spots} spots x {n_genes_supplied} genes "
        f"(graph on {loaded['coords_label']}, K={settings['num_neighbors']}) ..."
    )
    score_df = SpaGFT.detect_svg(
        adata,
        ratio_low_freq="infer",
        ratio_high_freq="infer",
        ratio_neighbors=1.0,
        spatial_info=GRAPH_COORDS_KEY,
    )
    log(f"SpaGFT returned {len(score_df)} rows; columns={list(score_df.columns)}")
    n_genes_used = int(len(score_df))

    if "cutoff_gft_score" in score_df.columns:
        sig = score_df[score_df["cutoff_gft_score"].astype(bool)]
        sig_basis = "cutoff_gft_score"
    elif "fdr" in score_df.columns:
        sig = score_df[score_df["fdr"].astype(float) < 0.05]
        sig_basis = "fdr<0.05"
    elif "pvalue" in score_df.columns:
        sig = score_df[score_df["pvalue"].astype(float) < 0.05]
        sig_basis = "pvalue<0.05"
    else:
        raise RuntimeError(
            "SpaGFT.detect_svg returned no significance column (cutoff_gft_score, fdr or pvalue); columns: "
            f"{list(score_df.columns)}. Nothing was written: a significance this worker invented would be "
            "reported under SpaGFT's name."
        )

    # MERFISH negative-control barcodes are not genes: out of the significant set, the count, the top
    # table and predicted_genes.json alike (they used to be dropped from the count only).
    is_control = np.asarray(sig.index.astype(str).str.startswith(CONTROL_PROBE_PREFIX), dtype=bool)
    n_control_sig = int(is_control.sum())
    sig = sig[~is_control]
    sig_genes = sig.index.astype(str).tolist()
    log(f"Significant SVGs: {len(sig_genes)} (basis={sig_basis}; {n_control_sig} control probe(s) left out)")

    score_df = score_df.copy()
    score_df["gene"] = score_df.index.astype(str)
    cols = ["gene"] + [c for c in score_df.columns if c != "gene"]
    scores_csv = os.path.join(output_dir, "spagft_svg_scores.csv")
    _write_csv_atomic(score_df[cols], scores_csv)
    log(f"Saved official SpaGFT scores to {scores_csv}")

    # Top-N significant genes (sorted by gft_score descending when present).
    if "gft_score" in sig.columns:
        sig_sorted = sig.copy().sort_values("gft_score", ascending=False)
    else:
        sig_sorted = sig.copy()
    n_top = min(max(1, n_top_genes), len(sig_sorted))
    top_csv = os.path.join(output_dir, "spagft_top_svg_genes.csv")
    top_table = sig_sorted.head(n_top).copy()
    top_table["gene"] = top_table.index.astype(str)
    cols2 = ["gene"] + [c for c in top_table.columns if c != "gene"]
    _write_csv_atomic(top_table[cols2], top_csv)
    log(f"Saved top {n_top} significant genes to {top_csv}")

    # predicted_genes.json: the standardised SVG-prediction artifact. It holds the same top-N
    # significant genes as spagft_top_svg_genes.csv (n_top_genes caps it); the manual runner writes
    # the whole significant set, which is what a large n_top_genes reproduces.
    top_gene_names = top_table["gene"].astype(str).tolist()
    pg_path = os.path.join(output_dir, "predicted_genes.json")
    _write_json_atomic({"predicted_genes": top_gene_names}, pg_path)
    log(f"Wrote {pg_path} with {len(top_gene_names)} genes")

    n_off = int(loaded["n_off_tissue"])
    n_genes_renamed = int(loaded["renamed"].get("n_genes_renamed", 0))
    out = WorkerOutput("spagft", task="svg_identification")
    out.set_data(
        n_spots=n_spots,
        n_spots_supplied=int(loaded["n_spots_supplied"]),
        n_spots_out_of_tissue_excluded=n_off,
        n_genes=n_genes_supplied,
        n_genes_used=n_genes_used,
    )
    out.add_output_files(
        {
            "scores_csv": scores_csv,
            "top_genes_csv": top_csv,
            "predicted_genes_json": pg_path,
        }
    )
    out.add_params(
        {
            "implementation": "official_SpaGFT",
            "ratio_low_freq": "infer",
            "ratio_high_freq": "infer",
            "ratio_neighbors": 1.0,
            "spatial_info": loaded["coords_label"],
            "spatial_key": spatial_key,
            "layer_key": layer_key,
            "expression_source": loaded["expression_source"],
            "num_neighbors": settings["num_neighbors"],
            "n_low_frequency_modes": settings["n_low_frequency_modes"],
            "n_high_frequency_modes": settings["n_high_frequency_modes"],
            "significance_basis": sig_basis,
            "n_top_genes": int(n_top_genes),
            "n_control_probes_excluded": n_control_sig,
            "n_spots_out_of_tissue_excluded": n_off,
            "n_genes_dropped_undetected": int(n_genes_supplied - n_genes_used),
            "precomputed_fourier_modes_discarded": bool(stale_modes),
        }
    )
    record_method(out, METHOD_OFFICIAL, used_fallback=False)
    _record_matrix_choice(out, loaded)
    record_ignored(
        out,
        list(FALLBACK_ONLY_PARAMS),
        "they configure only the from-scratch substitute (allow_gft_fallback). SpaGFT.detect_svg sets its own KNN "
        f"size (ceil(sqrt(n_spots)/2), 4 at <= 500 spots: K={settings['num_neighbors']} here) and frequency-mode "
        f"counts (ceil(sqrt(n_spots)) low and high: {settings['n_low_frequency_modes']} each here), keeps every gene "
        "detected in at least one spot, and draws no random number; the values it used are in params",
    )
    out.add_warnings(loaded["warnings"])
    out.set_summary(
        n_significant=int(len(sig_genes)),
        top_genes=top_gene_names,
    )

    analysis = build_svg_analysis(
        n_genes_used,
        int(len(sig_genes)),
        top_gene_names,
        method_name="SpaGFT",
        n_genes_renamed=n_genes_renamed,
    )
    analysis += describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        reason="SpaGFT's own filter, which keeps only genes detected in at least one analysed spot",
    )
    if n_off:
        analysis += (
            f" {n_off} of the {loaded['n_spots_supplied']} spots were left out because obs['in_tissue'] marks them "
            f"off the tissue; the KNN graph and the test cover the {n_spots} tissue spots."
        )
    if n_control_sig:
        analysis += (
            f" {n_control_sig} negative-control probe(s) ({CONTROL_PROBE_PREFIX}*) passed SpaGFT's cutoff and were "
            "left out of the significant set: they are not genes."
        )
    out.set_analysis(analysis)
    return out.to_dict()


# ----------------------------------------------------------------------------- the opt-in substitute


def _run_gft_fallback(
    st_h5ad: str,
    output_dir: str,
    layer_key: str,
    spatial_key: str,
    n_neighbors: int,
    n_eigs: int,
    low_freq_fraction: float,
    min_detected_frac: float,
    n_top_genes: int,
    seed: int,
    why: str,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """The from-scratch low-frequency energy ratio. Not SpaGFT; reached only with allow_gft_fallback.

    Each gene's expression is library-size normalised and log-transformed (``normalize_total`` +
    ``log1p``, the same preprocessing the SpaGFT path applies and the non-count warning describes)
    and centred, then projected on the ``n_eigs`` smoothest eigenvectors of the normalised KNN
    Laplacian; the score is the share of that energy in the lowest ``low_freq_fraction`` of them.
    Earlier revisions projected the raw, uncentred counts: the gene's mean then sat in the constant
    lowest mode, every well-expressed gene scored ~1.0 and the ranking followed expression level, not
    space (spatially random genes outranked a gene expressed in one region only), and a library-depth
    gradient read as a spatial pattern of every gene.
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy import sparse
    from scipy.sparse import linalg as splinalg
    from sklearn.neighbors import NearestNeighbors

    log("==== from-scratch GFT substitute (SpaGFT did not run) ====")
    log(f"why          : {why}")
    log(f"layer_key    : {layer_key}")
    log(f"spatial_key  : {spatial_key}")
    log(f"n_neighbors  : {n_neighbors}")
    log(f"n_eigs       : {n_eigs}")
    log(f"low_freq_frac: {low_freq_fraction:.4f}")
    log(f"min_det_frac : {min_detected_frac:.4f}")
    log(f"n_top_genes  : {n_top_genes}")
    log(f"seed         : {seed}")

    np.random.seed(seed)

    loaded = load_input(st_h5ad, spatial_key, layer_key, use_raw_counts)
    adata = loaded["adata"]
    coords = loaded["coords"]
    n_cells, n_genes = adata.n_obs, adata.n_vars
    log(f"Analysing {n_cells} spots and {n_genes} genes.")
    if not 1 <= int(n_neighbors) < n_cells:
        raise ValueError(
            f"n_neighbors={n_neighbors} cannot build the substitute's KNN graph on {n_cells} analysed spot(s): it "
            f"needs 1 <= n_neighbors < {n_cells}. Nothing was computed."
        )

    warnings = []
    X = adata.X
    X = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()

    # Gene filtering by detection fraction, on the supplied matrix (sparse throughout). An exact count
    # over n: a float32 mean of ones rounds 1.0 down (0.999992 at 900 spots), so a gene detected in
    # every spot used to fail min_detected_frac=1.0.
    n_detected = np.asarray((X != 0).sum(axis=0), dtype=np.int64).ravel()
    detected_frac = n_detected / float(n_cells)
    mean_expr = np.asarray(X.mean(axis=0), dtype=np.float64).ravel()

    keep_mask = detected_frac >= min_detected_frac
    n_keep = int(keep_mask.sum())
    log(f"Keeping {n_keep}/{n_genes} genes with detected_frac >= {min_detected_frac:.3f}.")
    if n_keep == 0:
        raise RuntimeError(f"No genes passed the detection fraction threshold ({min_detected_frac:.3f}).")

    # The same preprocessing as the SpaGFT path (and what load_input's non-count warning says was done).
    adata.X = X
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _refuse_nonfinite_after_log1p(adata)
    X_sub = sparse.csr_matrix(adata.X)[:, keep_mask]
    gene_names = np.array(adata.var_names)[keep_mask]

    # Symmetric unweighted KNN graph on the spot coordinates.
    log(f"Building KNN graph (n_neighbors={n_neighbors})...")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n_cells), indices.shape[1] - 1)
    cols = indices[:, 1:].ravel()
    A = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    A = A.maximum(A.T)

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = 1.0 / np.sqrt(deg)
    inv_sqrt_deg[~np.isfinite(inv_sqrt_deg)] = 0.0
    D_inv_sqrt = sparse.diags(inv_sqrt_deg)
    L = sparse.eye(n_cells, format="csr") - D_inv_sqrt @ A @ D_inv_sqrt

    k = min(max(1, n_eigs), max(1, n_cells - 1))
    if k != int(n_eigs):
        warnings.append(
            f"n_eigs={n_eigs} cannot be computed on a {n_cells}-spot graph (1 <= n_eigs <= {n_cells - 1}); the "
            f"substitute used {k} eigenpairs (params.n_eigs)."
        )
    lf = min(max(1, int(round(k * low_freq_fraction))), k)
    if lf >= k:
        raise ValueError(
            f"low_freq_fraction={low_freq_fraction:g} of {k} eigenpairs puts all {k} in the low-frequency band, so "
            "every gene would score 1.0 and the ranking would be arbitrary. Nothing was computed. Lower "
            "low_freq_fraction or raise n_eigs."
        )
    log(f"Computing {k} smallest eigenpairs of the normalized Laplacian...")
    eigvals, eigvecs = splinalg.eigsh(L, k=k, which="SM")
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    log(f"Using {lf}/{k} eigenvectors as low-frequency components.")

    # U^T (X - 1 mu^T) = U^T X - (U^T 1) mu^T: each gene centred without densifying the matrix, so its
    # mean does not sit in the constant mode and count as low-frequency energy.
    gene_means = np.asarray(X_sub.mean(axis=0), dtype=np.float64).ravel()
    F = np.asarray((X_sub.T @ eigvecs).T, dtype=np.float64)  # (k, n_keep)
    F = F - np.outer(eigvecs.sum(axis=0), gene_means)
    energy_total = (F * F).sum(axis=0)
    energy_low = (F[:lf, :] * F[:lf, :]).sum(axis=0)
    score = energy_low / (energy_total + 1e-12)

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "spagft_lowfreq_energy": energy_low,
            "spagft_total_energy": energy_total,
            "spagft_score": score,
            "mean_expression": mean_expr[keep_mask],
            "detected_fraction": detected_frac[keep_mask],
        }
    )
    df.sort_values("spagft_score", ascending=False, inplace=True)

    scores_csv = os.path.join(output_dir, "spagft_svg_scores.csv")
    _write_csv_atomic(df, scores_csv)
    n_top = min(max(1, n_top_genes), df.shape[0])
    top_df = df.head(n_top)
    top_csv = os.path.join(output_dir, "spagft_top_svg_genes.csv")
    _write_csv_atomic(top_df, top_csv)
    spectrum_path = os.path.join(output_dir, "spagft_graph_spectrum.npz")
    tmp = spectrum_path + ".partial"
    with open(tmp, "wb") as fh:
        np.savez(fh, eigenvalues=eigvals)
    os.replace(tmp, spectrum_path)
    log(f"Saved substitute scores, top {n_top} genes and the Laplacian spectrum to {output_dir}")

    top_gene_names = top_df["gene"].astype(str).tolist()
    n_off = int(loaded["n_off_tissue"])

    out = WorkerOutput("spagft", task="svg_identification")
    out.set_data(
        n_spots=int(n_cells),
        n_spots_supplied=int(loaded["n_spots_supplied"]),
        n_spots_out_of_tissue_excluded=n_off,
        n_genes=int(n_genes),
        n_genes_used=int(n_keep),
    )
    out.add_output_files(
        {
            "scores_csv": scores_csv,
            "top_genes_csv": top_csv,
            "spectrum_npz": spectrum_path,
        }
    )
    out.add_params(
        {
            "implementation": "from_scratch_gft",
            "spatial_info": loaded["coords_label"],
            "spatial_key": spatial_key,
            "layer_key": layer_key,
            "expression_source": loaded["expression_source"],
            "n_neighbors": int(n_neighbors),
            "n_eigs": int(k),
            "n_low_freq": int(lf),
            "low_freq_fraction": float(low_freq_fraction),
            "min_detected_frac": float(min_detected_frac),
            "seed": seed,
            "n_top_genes": int(n_top_genes),
            "n_spots_out_of_tissue_excluded": n_off,
            "normalization": "normalize_total(target_sum=1e4) + log1p",
            "genes_centred": True,
        }
    )
    record_method(out, METHOD_FALLBACK, used_fallback=True, why=why)
    _record_matrix_choice(out, loaded)
    out.add_warnings(loaded["warnings"])
    out.add_warnings(warnings)
    # A low-frequency energy ratio, not SpaGFT's q-value. n_top is min(requested, n), so reporting it
    # as n_significant would make the discovery rate a restatement of the request.
    out.set_summary(
        n_significant=None,
        n_top_reported=int(n_top),
        top_genes=top_gene_names,
    )
    analysis = build_svg_analysis(
        int(n_keep),
        None,
        top_gene_names,
        method_name="The from-scratch GFT substitute (SpaGFT did not run)",
        n_genes_renamed=int(loaded["renamed"].get("n_genes_renamed", 0)),
    )
    analysis += describe_reduction(
        "genes", int(n_genes), int(n_keep), reason=f"the min_detected_frac={min_detected_frac:g} detection filter"
    )
    if n_off:
        analysis += (
            f" {n_off} of the {loaded['n_spots_supplied']} spots were left out because obs['in_tissue'] marks them "
            "off the tissue."
        )
    out.set_analysis(analysis)
    return out.to_dict()


# ----------------------------------------------------------------------------- entry points


def run_spagft_svg(
    st_h5ad: str,
    output_dir: str,
    layer_key: str,
    spatial_key: str,
    n_neighbors: int,
    n_eigs: int,
    low_freq_fraction: float,
    min_detected_frac: float,
    n_top_genes: int,
    seed: int,
    allow_gft_fallback: bool = False,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run the official ``SpaGFT.detect_svg``.

    The from-scratch substitute runs only when SpaGFT cannot be imported AND ``allow_gft_fallback`` is
    True; an error raised inside SpaGFT is never answered with the substitute.
    """
    if int(n_top_genes) < 1:
        # It used to be clamped to 1 in silence while params echoed the value asked for.
        raise ValueError(
            f"n_top_genes={n_top_genes} asks for no gene; it is how many top-ranked genes are written to "
            "spagft_top_svg_genes.csv (and predicted_genes.json), so it must be at least 1. Nothing was computed. "
            "Pass a value at least as large as the number of genes to keep every significant gene."
        )
    spagft, why = _import_spagft()
    if spagft is None and not allow_gft_fallback:
        raise ImportError(
            f"SpaGFT cannot run in this environment ({why}). Nothing was computed. Install SpaGFT with kneed and "
            "statsmodels in the tool env, or pass allow_gft_fallback=True to accept a from-scratch low-frequency "
            "energy ratio that is not SpaGFT (the result then says so in params.method and params.used_fallback)."
        )

    os.makedirs(output_dir, exist_ok=True)
    if spagft is not None:
        return _run_official_spagft(spagft, st_h5ad, output_dir, layer_key, spatial_key, n_top_genes, use_raw_counts)

    log(f"SpaGFT unavailable ({why}); allow_gft_fallback=True, running the from-scratch substitute.")
    return _run_gft_fallback(
        st_h5ad,
        output_dir,
        layer_key,
        spatial_key,
        n_neighbors,
        n_eigs,
        low_freq_fraction,
        min_detected_frac,
        n_top_genes,
        seed,
        why,
        use_raw_counts=use_raw_counts,
    )


def main() -> None:
    args = parse_args()

    # Redirect stdout to stderr during processing so logs don't pollute JSON
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    error_msg = ""
    result = None
    try:
        try:
            result = run_spagft_svg(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                layer_key=args.layer_key,
                spatial_key=args.spatial_key,
                n_neighbors=args.n_neighbors,
                n_eigs=args.n_eigs,
                low_freq_fraction=args.low_freq_fraction,
                min_detected_frac=args.min_detected_frac,
                n_top_genes=args.n_top_genes,
                seed=args.seed,
                allow_gft_fallback=args.allow_gft_fallback,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            log("ERROR while running SpaGFT worker:")
            traceback.print_exc(file=sys.stderr)
            result = None
            error_msg = str(e)
            error_exc = e
    finally:
        # Restore stdout
        sys.stdout = orig_stdout

    # Print a single JSON line to stdout
    if result is None:
        WorkerOutput.emit_error("spagft", error_msg, task="svg_identification", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
