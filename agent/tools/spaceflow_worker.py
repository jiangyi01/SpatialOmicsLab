#!/usr/bin/env python3
"""
SpaceFlow worker for SpatialOmicsLab MCP (runs in /opt/conda/envs/spaceflow_env).

Protocol:
- stdout: final JSON only (SpaceFlow's own ``print`` progress is redirected to stderr)
- stderr: logs/progress/tracebacks

What runs, in order:
0) Spots flagged ``obs['in_tissue'] == 0`` (background glass) are left out and reported
   (``params.in_tissue_filter``). In ``generic_counts_coords`` mode the flag is the coords file's
   ``in_tissue`` column when it has one, the counts h5ad's own ``obs['in_tissue']`` when it has one, and
   both together (a spot is tissue only where both say so) when both do -- never a column of 1s the
   file did not carry (``params.in_tissue_source``); spots of the counts with no row in the coords file
   are left out and counted (``params.n_spots_without_coordinates``). SpaceFlow normalises the matrix
   as counts, so it is chosen by ``worker_utils.choose_counts_matrix``: X (negative or NaN values are
   refused, fractional ones run with a warning) or, with ``use_raw_counts``, ``adata.raw.X``
   (``params.expression_source``). The genes with no count in any remaining spot are left out too
   (``params.n_genes_without_counts_dropped``). Those genes add nothing to a spot's library size and
   cannot be highly variable, but they all share a mean of 0, which collapses cell_ranger's
   mean-percentile bins: on whole-transcriptome slides (11,414 of 36,601 genes on V1_Human_Lymph_Node)
   SpaceFlow's own preprocessing could not run without this.
1) ``SpaceFlow.preprocessing_data()`` on that matrix (sparse stays sparse; SpaceFlow's
   ``train()`` densifies only the highly-variable-gene subset): normalize_total, log1p, scanpy HVGs
   with flavor='cell_ranger', PCA, and SpaceFlow's alpha-complex spatial graph.
   cell_ranger bins genes by mean at fixed percentiles, and pandas raises "Bin edges must be unique"
   when many genes still share one mean. The run then STOPS, unless the
   caller passed ``allow_preprocessing_fallback=True``: only then does this wrapper's substitute run
   (scanpy HVGs by ``hvg_flavor`` on a copy of the supplied counts, and a symmetric kNN spatial graph
   in place of the alpha complex), populating ``adata_preprocessed``/``spatial_graph`` itself. The
   payload says so under ``params.method``/``params.used_fallback``.
2) ``SpaceFlow.train()`` -- upstream 1.0.4's own source with two audited one-token corrections (see
   ``_corrected_train``): the accelerated spatial-regularization term paired the coordinates of one
   random sample with themselves, which made it non-spatial; and the "best" weights it restored were
   a live alias of the current ones. ``params.train_corrections`` lists what was corrected.
3) ``SpaceFlow.segmentation()`` (Leiden on the embedding), optionally binary-searched on resolution.
4) Optional pseudo-Spatiotemporal Map: SpaceFlow's recipe (kNN graph on the embedding, diffusion
   pseudotime from the spot most distant from all others), with that root found over every spot.

Outputs:
- spaceflow_annotated.h5ad
- spaceflow_embedding.csv
- domains.tsv
- (optional) pSM_values.tsv
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    available_memory_bytes,
    build_cluster_analysis,
    choose_counts_matrix,
    describe_reduction,
    find_tissue_positions,
    id_mismatch_msg,
    keep_in_tissue,
    read_coords_csv,
    read_tissue_positions,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    sniff_tabular_sep,
    unsupported_choice_msg,
)

# ---- what SpaceFlow itself fixes -----------------------------------------------------------------

#: ``preprocessing_data(n_neighbors=10)``: the kNN radius SpaceFlow's alpha-complex graph is cut at.
UPSTREAM_ALPHA_NEIGHBORS = 10
#: Above this many spots ``train()`` estimates the regularization from sampled pairs whatever
#: ``regularization_acceleration`` says (``if regularization_acceleration or n > 5000``).
UPSTREAM_EXACT_REGULARIZATION_MAX_SPOTS = 5000

# ---- this wrapper's substitute preprocessing (runs only with allow_preprocessing_fallback=True) --

FALLBACK_HVG_FLAVORS = ("seurat", "cell_ranger", "seurat_v3")
#: What the substitute has always run in spaceflow_env: the old default asked for seurat_v3, whose
#: dependency (scikit-misc) the env does not ship, and switched to seurat without saying so.
DEFAULT_HVG_FLAVOR = "seurat"
#: Private layer the raw counts wait in for seurat_v3, which ranks counts rather than log data.
_HVG_COUNTS_LAYER = "_spaceflow_hvg_counts"

COORD_TYPES = ("array", "pixel")
INPUT_MODES = ("visium_h5_spatial", "spaceranger_outs", "h5ad", "generic_counts_coords")

# ---- audited corrections to SpaceFlow 1.0.4 ``train()`` ---------------------------------------------

#: ``SpaceFlow.py:238-242`` draws two independent random spot samples for the accelerated
#: regularization estimate and pairs the EMBEDDINGS as (sample_1, sample_2) but the COORDINATES as
#: (sample_1, sample_1). Every spatial distance is then PairwiseDistance's eps, normalised to 1.0, and
#: ``penalty = mean((1 - z_dist) * sp_dist)`` becomes ``mean(1 - z_dist)``: it pushes all embeddings
#: apart and carries no spatial information at all. The accelerated path is the default and is forced
#: above 5000 spots, so every run took it. The correction indexes the second coordinate sample with
#: ``cell_random_subset_2``, exactly as the embedding sample on the line above already is.
_COORD_PAIR_TYPO = re.compile(
    r"(c1\s*,\s*c2\s*=\s*torch\.index_select\(\s*coords\s*,\s*0\s*,\s*cell_random_subset_1\s*\)\s*,"
    r"\s*torch\.index_select\(\s*coords\s*,\s*0\s*,\s*)cell_random_subset_1(\s*\))"
)
_COORD_PAIR_INDEPENDENT = re.compile(r"torch\.index_select\(\s*coords\s*,\s*0\s*,\s*cell_random_subset_2\s*\)")
#: ``best_params = model.state_dict()`` holds tensors that share storage with the live weights, so
#: every optimizer step rewrites the "best" snapshot and ``model.load_state_dict(best_params)`` at the
#: end restores the LAST epoch, not the lowest-loss one early stopping was tracking. Cloning fixes it.
_BEST_PARAMS_ALIAS = re.compile(r"best_params\s*=\s*model\.state_dict\(\)")
_BEST_PARAMS_SNAPSHOT = "best_params = {_k: _v.detach().clone() for _k, _v in model.state_dict().items()}"

CORRECTION_COORD_PAIRS = (
    "the sampled spatial regularization pairs coordinates as it pairs embeddings, from two independent "
    "draws (SpaceFlow 1.0.4 paired each sampled spot's coordinates with themselves, so the term carried "
    "no spatial information)"
)
CORRECTION_BEST_WEIGHTS = (
    "the lowest-loss weights are restored after training (SpaceFlow 1.0.4 restored the last epoch's, "
    "through a live reference to the weights)"
)

#: SpaceFlow trains in float32 (``torch.tensor(expr).float()``).
_BYTES_PER_FLOAT32 = 4


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def emit_json(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _read_label_file(path: str, obs_names, what: str) -> pd.Series | None:
    """Read one of SpaceFlow's own label files, aligned to ``obs_names``, or return None.

    SpaceFlow's ``segmentation(domain_label_save_filepath=...)`` and
    ``pseudo_Spatiotemporal_Map(pSM_values_save_filepath=...)`` write ONE bare value per line in
    ``adata.obs_names`` order -- no barcode column and no header. Four readers here were written
    for a two-column ``barcode<TAB>value`` layout and gated on ``shape[1] >= 2``, which is False on
    every real run; because a False guard raises nothing, all four failed in silence.

    Both layouts are accepted. Two columns are matched by barcode, which is order-independent and
    is what the callers originally intended. One column can only be paired by position, so it is
    refused unless its length matches ``obs_names`` -- attaching labels to the wrong spots is worse
    than attaching none. ``what`` names the caller in the log; every refusal leaves a record.
    """
    try:
        frame = pd.read_csv(path, sep="\t", header=None)
    except Exception as e:
        log(f"[SpaceFlow worker] WARNING: could not read {what} from {path}: {e}")
        return None

    index = pd.Index(obs_names).astype(str)

    if frame.shape[1] >= 2:
        keyed = frame.iloc[:, :2]
        keyed.columns = ["barcode", "value"]
        keyed = keyed.astype({"barcode": str}).drop_duplicates("barcode").set_index("barcode")
        missing = index.difference(keyed.index)
        if len(missing) > 0:
            log(
                f"[SpaceFlow worker] WARNING: {what} in {path} covers {len(keyed)} barcodes but "
                f"{len(missing)} of {len(index)} spots are absent (e.g. {list(missing[:3])}); not attaching"
            )
            return None
        return pd.Series(keyed.loc[index, "value"].values, index=index, name=what)

    if len(frame) != len(index):
        log(
            f"[SpaceFlow worker] WARNING: {what} in {path} has one column and {len(frame)} rows, "
            f"but there are {len(index)} spots; a positional pairing would be wrong, so not attaching"
        )
        return None

    return pd.Series(frame.iloc[:, 0].values, index=index, name=what)


def _load_visium_h5_spatial(visium_h5_path: str, visium_spatial_dir: str, coord_type: str = "array"):
    import scanpy as sc

    adata = sc.read_10x_h5(visium_h5_path)
    adata.var_names_make_unique()

    tp_path = find_tissue_positions(visium_spatial_dir)
    if tp_path is None:
        raise FileNotFoundError(f"Cannot find {' or '.join(TISSUE_POSITIONS_NAMES)} under: {visium_spatial_dir}")
    tp = read_tissue_positions(tp_path)

    common = np.intersect1d(adata.obs_names.astype(str), tp["barcode"].values)
    if common.size == 0:
        raise ValueError(
            id_mismatch_msg("barcodes", "count matrix", adata.obs_names, Path(tp_path).name, tp["barcode"].values)
        )
    adata = adata[common].copy()
    tp = tp.set_index("barcode").loc[adata.obs_names.astype(str)].copy()

    adata.obs["in_tissue"] = tp["in_tissue"].astype(int).values
    adata.obs["array_row"] = tp["array_row"].astype(int).values
    adata.obs["array_col"] = tp["array_col"].astype(int).values

    if coord_type == "pixel":
        coords = np.vstack([tp["pxl_col_in_fullres"].values, tp["pxl_row_in_fullres"].values]).T.astype(float)
    else:
        coords = np.vstack([tp["array_col"].values, tp["array_row"].values]).T.astype(float)

    adata.obsm["spatial"] = coords
    return adata


def _load_spaceranger_outs(spaceranger_dir: str, coord_type: str = "array"):
    outs = Path(spaceranger_dir)
    h5 = outs / "filtered_feature_bc_matrix.h5"
    sp = outs / "spatial"
    if not h5.exists() or not sp.exists():
        raise FileNotFoundError("Space Ranger outs/ must contain filtered_feature_bc_matrix.h5 and spatial/")
    return _load_visium_h5_spatial(str(h5), str(sp), coord_type=coord_type)


def _load_h5ad(h5ad_path: str):
    import anndata as ad

    return ad.read_h5ad(h5ad_path)


def _coords_file_columns(path: str) -> list:
    """The column names in a coordinate file's header, read with the separator the file uses.

    ``read_coords_csv`` fills a missing ``in_tissue`` column with 1, so its frame cannot say whether the
    file carried the flag; the header can.
    """
    return [str(c) for c in pd.read_csv(path, sep=sniff_tabular_sep(path), nrows=0).columns]


def _in_tissue_flags(values) -> np.ndarray:
    """``in_tissue`` values as 0/1 ints, by ``keep_in_tissue``'s rule: 1/True/"1"/"true" are tissue."""
    text = pd.Series(np.asarray(values, dtype=object)).astype(str).str.strip().str.lower()
    flag = pd.to_numeric(text.replace({"true": "1", "false": "0"}), errors="coerce")
    return np.asarray(flag == 1).astype(int)


def _load_generic_counts_coords(counts_h5ad_path: str, coords_csv: str, report: dict | None = None):
    """The counts h5ad joined with its coordinate file, in the counts' own spot order.

    ``obs['in_tissue']`` is the coords file's column when the file has one, the h5ad's own when only
    it has one, and both together (tissue only where both say so) when both do. It used to be the
    coords file's column always -- a column of 1s ``read_coords_csv`` fills in when the file has none,
    which overwrote a CELLxGENE h5ad's own flag and let its background glass through. Spots of the
    counts with no coordinate row are left out; ``report`` (when given) receives how many, the spot
    count of the counts file and where the in-tissue flag came from.
    """
    import anndata as ad

    adata = ad.read_h5ad(counts_h5ad_path)
    n_spots_in_counts = int(adata.n_obs)

    csv_has_in_tissue = "in_tissue" in _coords_file_columns(coords_csv)
    df = read_coords_csv(coords_csv)
    duplicated = df["barcode"].duplicated(keep=False)
    if duplicated.any():
        shown = sorted(set(df.loc[duplicated, "barcode"]))[:5]
        raise ValueError(
            f"coords_csv has more than one row for {int(df.loc[duplicated, 'barcode'].nunique())} barcode(s) "
            f"(e.g. {shown}); each spot needs exactly one coordinate row."
        )
    df = df.set_index("barcode")

    barcodes = pd.Index(adata.obs_names.astype(str))
    has_coords = np.asarray(barcodes.isin(df.index))
    if not has_coords.any():
        raise ValueError(
            id_mismatch_msg("barcodes", "counts_h5ad_path", adata.obs_names, "coords_csv.barcode", df.index)
        )
    n_spots_without_coordinates = int((~has_coords).sum())
    if n_spots_without_coordinates:
        adata = adata[has_coords].copy()
    df = df.loc[adata.obs_names.astype(str)].copy()

    h5ad_has_in_tissue = "in_tissue" in adata.obs.columns
    if csv_has_in_tissue and h5ad_has_in_tissue:
        adata.obs["in_tissue"] = _in_tissue_flags(df["in_tissue"].values) & _in_tissue_flags(
            adata.obs["in_tissue"].values
        )
        in_tissue_source = "coords_csv and counts_h5ad obs['in_tissue'] (tissue only where both say so)"
    elif csv_has_in_tissue:
        adata.obs["in_tissue"] = _in_tissue_flags(df["in_tissue"].values)
        in_tissue_source = "coords_csv"
    elif h5ad_has_in_tissue:
        in_tissue_source = "counts_h5ad obs['in_tissue']"
    else:
        in_tissue_source = None
    adata.obsm["spatial"] = np.vstack([df["x"].values, df["y"].values]).T.astype(float)
    if report is not None:
        report.update(
            {
                "n_spots_in_counts": n_spots_in_counts,
                "n_spots_without_coordinates": n_spots_without_coordinates,
                "in_tissue_source": in_tissue_source,
            }
        )
    return adata


def _load_input(payload: dict[str, Any], input_mode: str, coord_type: str, report: dict | None = None):
    """``(adata, coordinates_source, coord_type_used)`` for the requested input mode.

    ``coord_type`` picks array vs pixel columns out of a Space Ranger ``tissue_positions`` file, so it
    only means something in the two modes that read one. The h5ad and generic modes take the
    coordinates they are handed as stored, and say so rather than echoing a choice nobody applied.
    ``report`` (generic mode) receives what the coordinate join left out; see
    ``_load_generic_counts_coords``.
    """
    if input_mode in ("visium_h5_spatial", "spaceranger_outs") and coord_type not in COORD_TYPES:
        raise ValueError(unsupported_choice_msg("coord_type", coord_type, COORD_TYPES))
    if input_mode == "visium_h5_spatial":
        adata = _load_visium_h5_spatial(
            visium_h5_path=str(payload["visium_h5_path"]),
            visium_spatial_dir=str(payload["visium_spatial_dir"]),
            coord_type=coord_type,
        )
        return adata, f"tissue_positions {coord_type} coordinates", True
    if input_mode == "spaceranger_outs":
        adata = _load_spaceranger_outs(spaceranger_dir=str(payload["spaceranger_dir"]), coord_type=coord_type)
        return adata, f"tissue_positions {coord_type} coordinates", True
    if input_mode == "h5ad":
        adata = _load_h5ad(str(payload["h5ad_path"]))
        if "spatial" not in adata.obsm:
            raise ValueError("h5ad input must contain adata.obsm['spatial'] for SpaceFlow.")
        return adata, "obsm['spatial'] as stored", False
    if input_mode == "generic_counts_coords":
        adata = _load_generic_counts_coords(
            counts_h5ad_path=str(payload["counts_h5ad_path"]),
            coords_csv=str(payload["coords_csv"]),
            report=report,
        )
        return adata, "coords_csv x,y as stored", False
    # A literal list, not list(INPUT_MODES): the smoke-harness census reads the modes a worker dispatches
    # from this call's literal argument (test/test_the_smoke_harness_keeps_the_modes_it_reaches.py).
    raise ValueError(
        unsupported_choice_msg(
            "input_mode", input_mode, ["visium_h5_spatial", "spaceranger_outs", "h5ad", "generic_counts_coords"]
        )
    )


def _is_bin_edge_error(message: str) -> bool:
    return "Bin edges must be unique" in message or "duplicates" in message


def _short_error(message: str) -> str:
    """pandas prints every bin edge after "Bin edges must be unique"; the payload needs the sentence."""
    if "Bin edges must be unique" in message:
        return "pandas.cut: Bin edges must be unique"
    first = message.strip().splitlines()[0] if message.strip() else message
    return first[:300]


def _preprocessing_refusal(error: str, n_top_genes: int, n_genes_given: int = 0, n_without_counts: int = 0) -> str:
    """The stop message when cell_ranger's bins collapse on a panel that has no all-zero gene left.

    Genes with no counts -- the usual cause -- are already removed before SpaceFlow sees the matrix,
    so the message says so instead of asking an MCP caller to do by hand what this tool has done.
    """
    removed = (
        f"The {n_without_counts} genes with no counts in any spot, the usual cause, had already been left out, "
        if n_without_counts
        else "No gene lacked counts, so none was left out, "
    )
    return (
        "SpaceFlow's own preprocessing failed in scanpy highly_variable_genes(flavor='cell_ranger', "
        f"n_top_genes={n_top_genes}): {_short_error(error)}. cell_ranger bins genes by mean expression at fixed "
        "percentiles, and the bins collapse when many genes share one mean. "
        + removed
        + f"and the {n_genes_given} genes SpaceFlow was given still collapsed them. SpaceFlow was not run. "
        "Pass allow_preprocessing_fallback=True to train SpaceFlow on this tool's substitute preprocessing "
        "instead (scanpy highly variable genes by hvg_flavor, and a k-nearest-neighbour spatial graph in "
        "place of SpaceFlow's alpha complex); that run reports params.used_fallback=true."
    )


def _genes_with_counts(X, chunk_rows: int = 4096) -> np.ndarray:
    """Boolean mask over the genes: True where at least one spot holds a non-zero value.

    Sparse-aware: a CSR/CSC matrix is read through its stored values (explicit zeros do not count) and
    nothing is densified; a dense matrix is scanned ``chunk_rows`` spots at a time.
    """
    import scipy.sparse as sp  # type: ignore

    if X is None:
        raise ValueError("the input has no expression matrix (adata.X is empty); SpaceFlow needs counts in X.")
    n_vars = int(X.shape[1])
    if sp.issparse(X):
        fmt = X.format
        nonzero = np.asarray(X.data) != 0
        if fmt == "csr":
            seen = np.zeros(n_vars, dtype=bool)
            seen[np.asarray(X.indices)[nonzero]] = True
            return seen
        if fmt == "csc":
            per_column = np.diff(np.concatenate(([0], np.cumsum(nonzero, dtype=np.int64)))[np.asarray(X.indptr)])
            return per_column > 0
        coo = X.tocoo()
        seen = np.zeros(n_vars, dtype=bool)
        seen[np.asarray(coo.col)[np.asarray(coo.data) != 0]] = True
        return seen
    seen = np.zeros(n_vars, dtype=bool)
    for start in range(0, int(X.shape[0]), int(chunk_rows)):
        seen |= np.any(np.asarray(X[start : start + int(chunk_rows)]) != 0, axis=0)
    return seen


def _fallback_preprocess_scanpy_subset(adata, n_top_genes: int, flavor: str = "seurat"):
    """This wrapper's substitute for ``preprocessing_data``: a NEW, highly-variable-gene subset.

    ``adata`` must hold the counts as supplied -- SpaceFlow's own ``preprocessing_data`` normalises and
    logs ``self.adata`` in place BEFORE its HVG step raises, so handing that object here normalised and
    logged the data a second time. The copy is normalised (target_sum=1e4) and log1p'd as SpaceFlow
    does, then subset to ``n_top_genes`` by ``flavor``. ``seurat_v3`` ranks the raw counts (kept in a
    private layer until then) and needs scikit-misc; a missing or broken install stops the run. The
    flavour is never switched for the caller.
    """
    import scanpy as sc

    if flavor not in FALLBACK_HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", flavor, FALLBACK_HVG_FLAVORS))
    flavor = require_hvg_flavor(flavor)

    ad = adata.copy()
    try:
        ad.var_names_make_unique()
    except Exception:
        pass

    hvg_kwargs: dict[str, Any] = {}
    if flavor == "seurat_v3":
        ad.layers[_HVG_COUNTS_LAYER] = ad.X.copy()
        hvg_kwargs["layer"] = _HVG_COUNTS_LAYER

    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    try:
        sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes, flavor=flavor, subset=True, **hvg_kwargs)
    except Exception as e:
        # Broad on purpose: scikit-misc can be importable yet ABI-broken against this env's numpy
        # (ValueError, not ImportError). Whatever the cause, the caller's flavour did not run, so the
        # run stops here; it never switches flavour.
        if flavor == "seurat_v3":
            raise RuntimeError(
                f"hvg_flavor='seurat_v3' could not run ({type(e).__name__}: {e}). It needs a working "
                "scikit-misc (skmisc.loess), which spaceflow_env does not ship; install it against this "
                "environment's numpy, or pass hvg_flavor='seurat' explicitly. The flavour is not switched for you."
            ) from e
        raise

    if _HVG_COUNTS_LAYER in ad.layers:
        del ad.layers[_HVG_COUNTS_LAYER]
    return ad


def _knn_neighbours_per_spot(n_spots: int, k: int) -> int:
    """Neighbours each spot gets in ``_build_spatial_knn_graph``: the query counts the spot itself."""
    k_eff = max(2, min(int(k), max(2, int(n_spots) - 1)))
    return k_eff - 1


def _build_spatial_knn_graph(coords: np.ndarray, k: int = 50):
    """
    Build a symmetric kNN adjacency as a scipy.sparse CSR matrix.
    SpaceFlow expects self.spatial_graph to exist; this is a compatible substitute.

    coords: (n, d). ``k`` counts the spot itself (sklearn returns it as its own first neighbour), so each
    spot is joined to its ``k - 1`` nearest other spots before symmetrisation.
    """
    import scipy.sparse as sp  # type: ignore
    from sklearn.neighbors import NearestNeighbors

    n = coords.shape[0]
    k_eff = max(2, min(int(k), max(2, n - 1)))

    nn = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nn.fit(coords)
    dists, idx = nn.kneighbors(coords)

    # build adjacency: i -> neighbors
    rows = np.repeat(np.arange(n), k_eff)
    cols = idx.reshape(-1)
    data = np.ones(rows.shape[0], dtype=np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # symmetrize and remove self-loops
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def _corrected_train(spaceflow_cls):
    """``(train, corrections, pairing)``: upstream ``train`` with the audited corrections applied.

    The corrections are made to upstream's OWN source, token for token (``_COORD_PAIR_TYPO``,
    ``_BEST_PARAMS_ALIAS``), and compiled in upstream's module namespace at upstream's line numbers --
    so everything else ``train`` does is exactly what the installed SpaceFlow does, a traceback points
    at the real file, and a SpaceFlow release that has changed the code is left alone rather than
    patched blind. ``pairing`` says what the installed code does with coordinate pairs:
    ``"independent"`` (corrected here, or already correct upstream) or ``"unverified"`` (source not
    recognised, so nothing was checked). ``train`` is called as ``train(sf, **kwargs)``.
    """
    import inspect

    upstream = spaceflow_cls.train
    try:
        lines, first_line = inspect.getsourcelines(upstream)
        filename = inspect.getsourcefile(upstream) or "<SpaceFlow.train>"
    except (OSError, TypeError):
        return upstream, [], "unverified"

    source = textwrap.dedent("".join(lines))
    patched, n_pairs = _COORD_PAIR_TYPO.subn(r"\1cell_random_subset_2\2", source)
    corrections = []
    if n_pairs:
        corrections.append(CORRECTION_COORD_PAIRS)
        pairing = "independent"
    elif _COORD_PAIR_INDEPENDENT.search(source):
        pairing = "independent"
    else:
        pairing = "unverified"
    patched, n_alias = _BEST_PARAMS_ALIAS.subn(_BEST_PARAMS_SNAPSHOT, patched)
    if n_alias:
        corrections.append(CORRECTION_BEST_WEIGHTS)
    if not corrections:
        return upstream, [], pairing

    tree = ast.parse(patched, filename=filename)
    ast.increment_lineno(tree, first_line - 1)
    namespace: dict[str, Any] = {}
    exec(compile(tree, filename, "exec"), vars(sys.modules[spaceflow_cls.__module__]), namespace)
    return namespace[upstream.__name__], corrections, pairing


def _mem_available_bytes():
    """Memory this run can still allocate, in bytes, or None where nothing can be read.

    The shared reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and the
    room under the cgroup memory limit, page cache counted as reclaimable. /proc/meminfo alone reports
    the HOST's free memory inside a memory-limited container, so the refusal below never fired there
    and the kernel killed the run instead.
    """
    return available_memory_bytes()


def _train_bytes(n_spots, n_genes, x_itemsize, z_dim, n_coord_dims, accelerated, edge_subset_sz) -> int:
    """A lower bound on what ``train()`` allocates, intrinsic to SpaceFlow rather than to this wrapper.

    ``train`` densifies the preprocessed HVG matrix (``X.todense()``) and copies it into a float32
    tensor. The regularization then holds either the sampled pairs (embeddings and coordinates of
    ``edge_subset_sz`` pairs, and their gradients) or, on the exact path, two ``n x n`` distance
    matrices and their gradients.
    """
    dense = int(n_spots) * int(n_genes) * (int(x_itemsize) + _BYTES_PER_FLOAT32)
    if accelerated:
        pairs = int(edge_subset_sz) * (2 * int(z_dim) + 2 * int(n_coord_dims)) * _BYTES_PER_FLOAT32 * 2
    else:
        pairs = int(n_spots) * int(n_spots) * _BYTES_PER_FLOAT32 * 4
    return dense + pairs


def _check_train_budget(n_spots, n_genes, x_itemsize, z_dim, n_coord_dims, accelerated, edge_subset_sz, n_top_genes):
    """Refuse with the numbers before ``train()`` allocates what the machine does not have."""
    need = _train_bytes(n_spots, n_genes, x_itemsize, z_dim, n_coord_dims, accelerated, edge_subset_sz)
    available = _mem_available_bytes()
    if available is None or need <= available:
        return need
    gib = 1024.0**3
    pairs = (
        f"{edge_subset_sz} sampled spot pairs per epoch (edge_subset_sz)"
        if accelerated
        else f"two dense {n_spots} x {n_spots} distance matrices"
    )
    raise MemoryError(
        f"SpaceFlow trains on a dense {n_spots} x {n_genes} copy of the preprocessed expression plus its "
        f"float32 tensor, and on {pairs}: at least {need / gib:.1f} GiB, but this machine reports "
        f"{available / gib:.1f} GiB available. Lower n_top_genes (now {n_top_genes}) to shrink the gene "
        "axis, lower edge_subset_sz, or run on a machine with more memory. Every spot is analysed; the "
        "slide is never cut down."
    )


def _most_distant_spot(X, chunk_bytes: int = 256 * 1024 * 1024) -> int:
    """Index of the row with the largest summed Euclidean distance to every row -- over ALL rows.

    SpaceFlow's pSM takes its diffusion-pseudotime root as ``argmax(distance_matrix(X, X).sum(1))``.
    From 5000 spots up it computes that on a random 5000-spot subsample and stores the position WITHIN
    THE SUBSAMPLE as the root index into the full set -- an arbitrary spot. The full sum is computed
    here in row blocks (``chunk_bytes`` of distances at a time), so no n x n matrix is ever held and
    no spot is left out of the choice.
    """
    X = np.asarray(X, dtype=np.float64)
    n = int(X.shape[0])
    squared = np.einsum("ij,ij->i", X, X)
    rows = max(1, int(chunk_bytes // (8 * max(n, 1))))
    sums = np.empty(n, dtype=np.float64)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        d2 = squared[start:stop, None] + squared[None, :] - 2.0 * (X[start:stop] @ X.T)
        np.maximum(d2, 0.0, out=d2)
        np.sqrt(d2, out=d2)
        sums[start:stop] = d2.sum(axis=1)
    return int(np.argmax(sums))


def _pseudo_spatiotemporal_map(embedding, n_neighbors: int):
    """SpaceFlow's pSM, with its root found over every spot. ``(pseudotime, root_index)``.

    Upstream ``pseudo_Spatiotemporal_Map`` builds a kNN graph on the embedding, takes the spot most
    distant from all others as the root, and returns scanpy's diffusion pseudotime from it. It also
    runs UMAP, Leiden (at ``resolution``) and PAGA on a throwaway object whose results nothing reads --
    ``sc.tl.dpt`` with ``n_branchings=0`` uses only the neighbour graph, the diffusion map and the root
    -- so those are not run here, and ``psm_resolution`` is reported as having no effect.
    """
    import anndata
    import scanpy as sc

    adata = anndata.AnnData(np.asarray(embedding))
    sc.pp.neighbors(adata, n_neighbors=int(n_neighbors), use_rep="X")
    root = _most_distant_spot(adata.X)
    adata.uns["iroot"] = root
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    return adata.obs["dpt_pseudotime"].to_numpy(), root


def _write_text_atomic(values, path: str, fmt: str) -> None:
    tmp = f"{path}.partial"
    np.savetxt(tmp, values, fmt=fmt, header="", footer="", comments="")
    os.replace(tmp, path)


def _save_embedding_csv(adata, emb: np.ndarray, out_csv: str) -> None:
    df = pd.DataFrame(emb, index=adata.obs_names)
    tmp = f"{out_csv}.partial"
    df.to_csv(tmp)
    os.replace(tmp, out_csv)


def _write_h5ad_atomic(adata, path: str) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def run_spaceflow(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()

    output_dir = ensure_dir(str(payload["output_dir"]))
    sample_id = str(payload.get("sample_id") or "sample")

    input_mode = str(payload.get("input_mode", "h5ad"))
    coord_type = str(payload.get("coord_type", "array"))

    n_top_genes = int(payload.get("n_top_genes", 3000))
    # The substitute preprocessing is opt-in. ``allow_fallback_preprocessing`` is the worker-side
    # spelling it had before the portal declared the knob, when it defaulted to on.
    allow_preprocessing_fallback = bool(
        payload.get("allow_preprocessing_fallback", payload.get("allow_fallback_preprocessing", False))
    )
    hvg_flavor = str(payload.get("hvg_flavor", payload.get("fallback_hvg_flavor", DEFAULT_HVG_FLAVOR)))
    fallback_knn_k = int(payload.get("fallback_knn_k", 50))
    if allow_preprocessing_fallback and hvg_flavor not in FALLBACK_HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, FALLBACK_HVG_FLAVORS))

    spatial_regularization_strength = float(payload.get("spatial_regularization_strength", 0.1))
    z_dim = int(payload.get("z_dim", 50))
    lr = float(payload.get("lr", 1e-3))
    epochs = int(payload.get("epochs", 1000))
    max_patience = int(payload.get("max_patience", 50))
    min_stop = int(payload.get("min_stop", 100))
    random_seed = int(payload.get("random_seed", 42))
    gpu_requested = int(payload.get("gpu", 0))  # -1 => cpu
    regularization_acceleration = bool(payload.get("regularization_acceleration", True))
    edge_subset_sz = int(payload.get("edge_subset_sz", 1000000))

    seg_n_neighbors = int(payload.get("seg_n_neighbors", 50))
    seg_resolution = float(payload.get("seg_resolution", 1.0))
    target_n_clusters = int(payload.get("target_n_clusters", 0))

    run_psm = bool(payload.get("run_psm", False))
    psm_n_neighbors = int(payload.get("psm_n_neighbors", 20))
    psm_resolution = float(payload.get("psm_resolution", 1.0))

    make_plots = bool(payload.get("make_plots", False))
    use_raw_counts = bool(payload.get("use_raw_counts", False))
    # The two Space Ranger modes read the count matrix itself, which has no adata.raw.
    reads_count_matrix = input_mode in ("visium_h5_spatial", "spaceranger_outs")

    log(f"[SpaceFlow worker] input_mode={input_mode} sample_id={sample_id}")

    # ---- device. Upstream trains on f"cuda:{gpu}" whenever CUDA exists, so gpu=-1 on a GPU box was
    # "cuda:-1" and a crash; hiding CUDA before torch is imported is what makes -1 mean CPU. On a box
    # without CUDA upstream silently trains on CPU, and the payload says so rather than echoing 0.
    if gpu_requested < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = resolve_compute(gpu_requested, quiet=True).device
    gpu_effective = int(device.split(":", 1)[1]) if device.startswith("cuda:") else (0 if device == "cuda" else -1)

    # ---- load input
    load_report: dict[str, Any] = {}
    adata, coordinates_source, coord_type_used = _load_input(payload, input_mode, coord_type, load_report)
    n_spots_without_coordinates = int(load_report.get("n_spots_without_coordinates", 0))

    # Background spots (obs['in_tissue'] == 0, e.g. the glass around a CELLxGENE Visium export) are not
    # tissue: they are left out and reported, as every spot-clustering tool here does.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    n_spots_in_tissue = int(adata.n_obs)
    # The slide the caller handed over: in generic mode that is every spot of the counts h5ad,
    # including any the coordinate join left out.
    n_spots_given = int(load_report.get("n_spots_in_counts", n_spots_supplied))

    # SpaceFlow's preprocessing normalises and log1p's the matrix as counts. A scaled X (negative
    # values) or NaN is refused, naming use_raw_counts when adata.raw holds counts; use_raw_counts runs
    # on adata.raw.X; a fractional (already normalised) X runs as before, with a warning.
    adata, counts_choice = choose_counts_matrix(adata, use_raw_counts and not reads_count_matrix)

    # SpaceFlow's own preprocessing subsets its object IN PLACE and the substitute works on a COPY,
    # so after the call `sf.adata` means the analysed panel on one path and the supplied panel on the
    # other. Record the supplied panel here -- that of the matrix chosen above -- where it is unambiguous.
    n_genes_supplied = int(adata.n_vars)

    # Genes with no count in any spot add nothing to normalize_total and cannot be highly variable, but
    # they all share a mean of 0 and collapse cell_ranger's mean-percentile bins ("Bin edges must be
    # unique"): 11,414 of 36,601 genes on V1_Human_Lymph_Node. Leaving them out is what lets SpaceFlow's
    # own preprocessing run on a whole-transcriptome slide; the cut is reported, never silent.
    detected = _genes_with_counts(adata.X)
    n_genes_without_counts = int(detected.size - int(detected.sum()))
    if not detected.any():
        raise ValueError(
            f"none of the {n_genes_supplied} genes has a count in any of the {n_spots_in_tissue} spots analysed; "
            "SpaceFlow has no expression to train on. Check that X holds the counts."
        )
    if n_genes_without_counts:
        log(f"[SpaceFlow worker] leaving out {n_genes_without_counts} of {n_genes_supplied} genes with no counts")
        adata = adata[:, detected].copy()
    n_genes_given = int(adata.n_vars)

    # ---- SpaceFlow. The matrix goes in in its supplied storage (a sparse matrix stays sparse), less the
    # background spots and count-less genes above: its preprocessing (normalize_total, log1p, cell_ranger
    # HVGs, PCA) is sparse-aware and train() densifies only the HVG subset, so densifying the whole panel
    # here (36.7 GB for a 507,684 x 18,085 VisiumHD slide) bought nothing.
    from SpaceFlow import SpaceFlow as SpaceFlowClass

    # The substitute needs the counts as supplied, and SpaceFlow's preprocessing normalises and logs
    # its object in place before the step that can fail -- so when the substitute is allowed,
    # SpaceFlow gets a copy and `adata` stays untouched.
    sf = SpaceFlowClass.SpaceFlow(adata=adata.copy() if allow_preprocessing_fallback else adata)

    preproc_mode = "spaceflow_builtin"
    hvg_flavor_ran = "cell_ranger"
    spatial_graph_desc = f"alpha complex (SpaceFlow graph_alpha, n_neighbors={UPSTREAM_ALPHA_NEIGHBORS})"
    method = (
        "SpaceFlow (its own preprocessing: cell_ranger highly variable genes and alpha-complex spatial "
        "graph; Deep Graph Infomax embedding with spatial regularization; Leiden segmentation)"
    )
    used_fallback = False
    fallback_why = ""
    try:
        with contextlib.redirect_stdout(sys.stderr):
            sf.preprocessing_data(n_top_genes=n_top_genes)
    except ValueError as e:
        msg = str(e)
        if not _is_bin_edge_error(msg):
            raise
        if not allow_preprocessing_fallback:
            raise ValueError(_preprocessing_refusal(msg, n_top_genes, n_genes_given, n_genes_without_counts)) from e
        log(f"[SpaceFlow worker] WARNING: preprocessing_data failed ({msg}); running the allowed substitute.")

        # 1) adata_preprocessed, from the counts as supplied
        ad_pre = _fallback_preprocess_scanpy_subset(adata, n_top_genes=n_top_genes, flavor=hvg_flavor)

        # 2) spatial_graph from the spatial coords
        if "spatial" not in ad_pre.obsm:
            raise RuntimeError("Fallback preprocessing requires adata.obsm['spatial'] to build spatial graph.") from e
        coords = np.asarray(ad_pre.obsm["spatial"])
        G = _build_spatial_knn_graph(coords, k=fallback_knn_k)

        # 3) the attributes SpaceFlow.train() reads
        sf.adata_preprocessed = ad_pre
        sf.spatial_graph = G

        neighbours = _knn_neighbours_per_spot(ad_pre.n_obs, fallback_knn_k)
        preproc_mode = f"fallback_scanpy_{hvg_flavor}"
        hvg_flavor_ran = hvg_flavor
        spatial_graph_desc = (
            f"symmetric k-nearest-neighbour graph, {neighbours} nearest spots each "
            f"(fallback_knn_k={fallback_knn_k} counts the spot itself); substitute for SpaceFlow's alpha complex"
        )
        used_fallback = True
        fallback_why = f"SpaceFlow's own preprocessing raised {_short_error(msg)}"
        method = (
            "SpaceFlow Deep Graph Infomax embedding and Leiden segmentation on this tool's substitute "
            f"preprocessing: scanpy {hvg_flavor} highly variable genes and a {neighbours}-nearest-neighbour "
            "spatial graph in place of SpaceFlow's cell_ranger genes and alpha-complex graph"
        )

    # adata_preprocessed is what train() and segmentation() actually run on -- set by the library on
    # its own path and by the substitute above on the other, so one read answers for both.
    analysed = getattr(sf, "adata_preprocessed", None)
    if analysed is None:
        analysed = sf.adata
    n_spots_used = int(analysed.n_obs)
    n_genes_used = int(analysed.n_vars)

    accelerated = bool(regularization_acceleration or n_spots_used > UPSTREAM_EXACT_REGULARIZATION_MAX_SPOTS)
    coords_used = np.asarray(analysed.obsm["spatial"]) if "spatial" in analysed.obsm else np.zeros((n_spots_used, 2))
    x_itemsize = int(getattr(getattr(analysed.X, "dtype", None), "itemsize", 8) or 8)
    _check_train_budget(
        n_spots_used,
        n_genes_used,
        x_itemsize,
        z_dim,
        coords_used.shape[1] if coords_used.ndim == 2 else 2,
        accelerated,
        edge_subset_sz,
        n_top_genes,
    )

    # ---- train: upstream's own train() with the audited corrections. It saves a copy of the
    # embedding to `embedding_save_filepath`, "./embedding.tsv" by default -- the portal's working
    # directory, i.e. the repository root, overwritten by every run. It goes to a scratch directory.
    train, train_corrections, pairing = _corrected_train(SpaceFlowClass.SpaceFlow)
    with tempfile.TemporaryDirectory(prefix=".spaceflow_train_", dir=output_dir) as scratch:
        with contextlib.redirect_stdout(sys.stderr):
            train(
                sf,
                embedding_save_filepath=os.path.join(scratch, "embedding.tsv"),
                spatial_regularization_strength=spatial_regularization_strength,
                z_dim=z_dim,
                lr=lr,
                epochs=epochs,
                max_patience=max_patience,
                min_stop=min_stop,
                random_seed=random_seed,
                gpu=max(0, gpu_requested),
                regularization_acceleration=regularization_acceleration,
                edge_subset_sz=edge_subset_sz,
            )

    # embedding
    emb = np.asarray(sf.embedding)
    emb_csv = str(Path(output_dir) / "spaceflow_embedding.csv")
    _save_embedding_csv(sf.adata, emb, emb_csv)

    # segmentation (optionally binary-searched to land on target_n_clusters)
    domains_tsv = str(Path(output_dir) / "domains.tsv")

    def _segment(resolution: float) -> int:
        with contextlib.redirect_stdout(sys.stderr):
            sf.segmentation(
                domain_label_save_filepath=domains_tsv,
                n_neighbors=seg_n_neighbors,
                resolution=resolution,
            )
        labels = _read_label_file(domains_tsv, sf.adata.obs_names, "domain labels")
        return -1 if labels is None else int(labels.nunique())

    if target_n_clusters > 0:
        # Binary search on resolution; SpaceFlow's segmentation uses Leiden internally.
        low, high = 0.05, 5.0
        best_resolution = seg_resolution
        best_diff = 10**9
        for _ in range(20):
            mid = (low + high) / 2
            k = _segment(mid)
            if k < 0:
                break
            diff = abs(k - target_n_clusters)
            if diff < best_diff:
                best_diff = diff
                best_resolution = mid
                if k == target_n_clusters:
                    break
            if k < target_n_clusters:
                low = mid
            else:
                high = mid
            if high - low < 1e-3:
                break
        # Final pass with the chosen resolution (also rewrites domains_tsv)
        _segment(best_resolution)
        seg_resolution = best_resolution
    else:
        _segment(seg_resolution)

    # Attach domain labels. Upstream segmentation() swallows its own AttributeError/NameError with a
    # print, so a failed segmentation leaves no file -- a run without its domains is not a result.
    domain_labels = _read_label_file(domains_tsv, sf.adata.obs_names, "domain labels")
    if domain_labels is None:
        raise RuntimeError(
            f"SpaceFlow segmentation produced no usable domain labels in {domains_tsv} (see the log above); "
            "the embedding was trained but no spatial domains exist to report."
        )
    sf.adata.obs["spaceflow_domain"] = domain_labels.values

    # optional pSM
    psm_tsv: str | None = None
    psm_root_barcode: str | None = None
    if run_psm:
        psm_tsv = str(Path(output_dir) / "pSM_values.tsv")
        psm_values, psm_root = _pseudo_spatiotemporal_map(emb, psm_n_neighbors)
        _write_text_atomic(psm_values, psm_tsv, fmt="%.5f")
        sf.adata.obs["spaceflow_pSM"] = np.asarray(psm_values)
        psm_root_barcode = str(sf.adata.obs_names[psm_root])

    seg_pdf: str | None = None
    if make_plots:
        seg_pdf = str(Path(output_dir) / "domain_segmentation.pdf")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                sf.plot_segmentation(segmentation_figure_save_filepath=seg_pdf)
        except Exception as e:
            log(f"[SpaceFlow worker] WARNING: plot_segmentation failed: {e}")
            seg_pdf = None

    # The sample identifier travels with the annotated object, not only with this payload.
    sf.adata.uns["spaceflow_run"] = {
        "sample_id": sample_id,
        "method": method,
        "preprocessing_mode": preproc_mode,
        "device": device,
    }
    out_h5ad = str(Path(output_dir) / "spaceflow_annotated.h5ad")
    _write_h5ad_atomic(sf.adata, out_h5ad)

    runtime = round(time.time() - t0, 3)

    # The in-tissue cut is reported by record_in_tissue below; this note covers any spot SpaceFlow
    # itself left out of the in-tissue set, so the same cut is not announced twice.
    spot_note = describe_reduction("spots", n_spots_in_tissue, n_spots_used)
    hvg_cut = f"the highly-variable-gene selection (n_top_genes={n_top_genes})"
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_used,
        (
            f"leaving out the {n_genes_without_counts} genes with no counts in any spot and then {hvg_cut}"
            if n_genes_without_counts
            else hvg_cut
        ),
    )

    out = WorkerOutput("spaceflow", task="spatial_domains")
    out.set_data(
        n_spots=n_spots_given,
        n_genes=n_genes_supplied,
        n_spots_used=n_spots_used,
        n_genes_used=n_genes_used,
    )
    reduction_notes = [note.strip() for note in (spot_note, gene_note) if note]
    if reduction_notes:
        out.add_warnings(reduction_notes)
    files = {
        "spaceflow_annotated_h5ad": out_h5ad,
        "embedding_csv": emb_csv,
        "domains_tsv": domains_tsv,
    }
    if psm_tsv:
        files["psm_tsv"] = psm_tsv
    if seg_pdf:
        files["segmentation_pdf"] = seg_pdf
    out.add_output_files(files)
    if accelerated:
        regularization_pairs = f"{edge_subset_sz} random spot pairs per epoch, {pairing}"
    else:
        regularization_pairs = "every spot pair (exact)"
    out.add_params(
        {
            "sample_id": sample_id,
            "input_mode": input_mode,
            "coord_type": coord_type if coord_type_used else None,
            "coordinates": coordinates_source,
            "n_top_genes": n_top_genes,
            "hvg_flavor": hvg_flavor_ran,
            "spatial_graph": spatial_graph_desc,
            "z_dim": z_dim,
            "epochs": epochs,
            "lr": lr,
            "spatial_regularization_strength": spatial_regularization_strength,
            "regularization_acceleration": accelerated,
            "regularization_pairs": regularization_pairs,
            "train_corrections": list(train_corrections),
            "seg_n_neighbors": seg_n_neighbors,
            "seg_resolution": seg_resolution,
            "run_psm": bool(run_psm),
            "gpu": gpu_effective,
            "gpu_requested": gpu_requested,
            "device": device,
            "preprocessing_mode": preproc_mode,
            "allow_preprocessing_fallback": allow_preprocessing_fallback,
            "fallback_knn_k": fallback_knn_k if "fallback" in preproc_mode else None,
            # Genes with no count in any analysed spot, left out before SpaceFlow's preprocessing.
            "n_genes_without_counts_dropped": n_genes_without_counts,
            "use_raw_counts": use_raw_counts,
        }
    )
    record_expression_source(out, counts_choice)
    if use_raw_counts and reads_count_matrix:
        record_ignored(
            out,
            "use_raw_counts",
            f"input_mode={input_mode!r} reads the Space Ranger count matrix itself, which has no adata.raw",
        )
    if input_mode == "generic_counts_coords":
        out.add_params(
            {
                "in_tissue_source": load_report.get("in_tissue_source"),
                "n_spots_without_coordinates": n_spots_without_coordinates,
            }
        )
    if n_spots_without_coordinates:
        out.add_warning(
            f"{n_spots_without_coordinates} of the {n_spots_given} spots in counts_h5ad_path have no row in "
            "coords_csv and were left out: SpaceFlow needs a position for every spot it trains on."
        )
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    if run_psm:
        out.add_params({"psm_n_neighbors": psm_n_neighbors, "psm_root": psm_root_barcode})
    record_method(out, method, used_fallback=used_fallback, why=fallback_why)
    if not coord_type_used:
        record_ignored(
            out,
            "coord_type",
            f"input_mode={input_mode!r} uses {coordinates_source}; coord_type only chooses the array or pixel "
            "columns of a Space Ranger tissue_positions file (input_mode 'visium_h5_spatial' or 'spaceranger_outs')",
        )
    if not used_fallback and hvg_flavor != DEFAULT_HVG_FLAVOR:
        record_ignored(
            out,
            "hvg_flavor",
            "SpaceFlow's own preprocessing ran, and it always selects genes with flavor='cell_ranger'; "
            "hvg_flavor applies only to the substitute preprocessing",
        )
    if run_psm:
        record_ignored(
            out,
            "psm_resolution",
            f"SpaceFlow's pSM is diffusion pseudotime from a root spot; the Leiden clustering it ran at "
            f"resolution {psm_resolution:g} fed a PAGA graph that nothing read, so the value cannot change the pSM",
        )
    if accelerated and pairing == "unverified":
        out.add_warning(
            "SpaceFlow.train() is not the 1.0.4 source this tool audits, so whether its sampled spatial "
            "regularization pairs coordinates independently was not checked."
        )
    if not regularization_acceleration and accelerated:
        out.add_warning(
            f"regularization_acceleration=False was not honoured: SpaceFlow estimates the spatial regularization "
            f"from sampled pairs above {UPSTREAM_EXACT_REGULARIZATION_MAX_SPOTS} spots, and this slide has {n_spots_used}."
        )

    # Count domains from the segmentation output: the labels already read above -- the file has not
    # been rewritten since -- so the reported count cannot disagree with what was attached.
    n_clusters = int(domain_labels.nunique())
    cluster_sizes = {str(k): int(v) for k, v in domain_labels.astype(str).value_counts().sort_index().items()}
    n_requested = target_n_clusters if target_n_clusters > 0 else None
    if n_requested is not None and n_clusters != n_requested:
        out.add_warning(
            f"target_n_clusters={n_requested} was requested but the resolution search settled on "
            f"seg_resolution={seg_resolution:.4g}, which gives {n_clusters} domains."
        )

    summary_kwargs: dict[str, Any] = {"runtime_sec": runtime, "n_clusters": n_clusters, "cluster_sizes": cluster_sizes}
    if n_requested is not None:
        summary_kwargs["n_clusters_requested"] = n_requested
    out.set_summary(**summary_kwargs)
    out.add_output_file("output_dir", output_dir)

    fallback_sentence = ""
    if used_fallback:
        fallback_sentence = (
            f" {fallback_why}; as allow_preprocessing_fallback permits, SpaceFlow trained on this tool's "
            f"substitute instead: scanpy {hvg_flavor} highly variable genes and a k-nearest-neighbour spatial "
            "graph in place of SpaceFlow's alpha complex."
        )
    corrections_sentence = ""
    if train_corrections:
        corrections_sentence = (
            " Corrected in SpaceFlow's train() before training: " + "; ".join(train_corrections) + "."
        )
    psm_sentence = ""
    if run_psm:
        psm_sentence = (
            f" pSM: diffusion pseudotime rooted at spot {psm_root_barcode}, the spot most distant from all others."
        )
    tissue_sentence = ""
    if n_spots_without_coordinates:
        tissue_sentence += (
            f" {n_spots_without_coordinates} of the {n_spots_given} spots in the counts file have no coordinate "
            "row and were left out."
        )
    if n_spots_off_tissue:
        tissue_sentence += (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots were left out because obs['in_tissue'] marks "
            "them as background outside the tissue."
        )
    out.set_analysis(
        f"SpaceFlow spatial domain segmentation completed in {runtime:.1f}s on {device}. "
        f"Trained on {n_spots_used} spots with {n_genes_used} genes. "
        + build_cluster_analysis(
            cluster_sizes, cluster_key="spatial domain", total_spots=n_spots_used, n_requested=n_requested
        )
        + fallback_sentence
        + corrections_sentence
        + psm_sentence
        + tissue_sentence
        + spot_note
        + gene_note
    )
    return out.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    payload = json.loads(args.json)
    if not isinstance(payload, dict):
        raise ValueError("--json must be a JSON object")
    if payload.get("__tool__") != "spaceflow_spatial_domains":
        raise ValueError(f"Unsupported __tool__: {payload.get('__tool__')}")

    result = run_spaceflow(payload)
    emit_json(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[SpaceFlow worker] ERROR: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        WorkerOutput.emit_error("spaceflow", str(e), task="spatial_domains")
        sys.exit(1)
