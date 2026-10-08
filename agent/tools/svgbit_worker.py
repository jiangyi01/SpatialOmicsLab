#!/usr/bin/env python3
"""
SVGBit worker (runs inside /opt/conda/envs/svgbit).

Contract
- Input: JSON string via --json
- Output: ONLY final JSON to stdout
- Logs/progress: stderr

Tools
- __tool__ = "svgbit_run"

Input modes
- visium_10x: counts_h5 + spatial_dir. Coordinates are the tissue-positions table's
  pxl_col_in_fullres / pxl_row_in_fullres.
- h5ad: adata_path. Coordinates are read from obs['x'] / obs['y'], else obs['center_x'] /
  obs['center_y'], else a two-column obsm['spatial'] (a three-column one is refused, not flattened).
  params.coord_source says which.

The gene axis is cut in this order before svgbit ranks anything, and every cut is reported
(params.n_genes_after_*, data.n_genes_used = the genes actually ranked, and a warning):
  1. min_counts: genes non-zero in fewer than min_counts spots are dropped, whatever max_genes is.
  2. max_genes > 0: the max_genes highest-variance survivors are kept.
  3. low_variance_var > 0: svgbit.filters.low_variance_filter keeps genes with variance above it.
  4. quantile < 1: svgbit.filters.quantile_filter keeps genes whose mean is strictly below that
     quantile of all gene means (0.99 drops the ~1% most highly expressed). quantile = 1.0 (or an
     explicit null in a direct worker call) skips it: upstream's strict '<' would still drop the
     highest-mean gene at 1.0. A value outside (0, 1] is refused.

Spots flagged obs['in_tissue'] == 0 (h5ad mode) or in_tissue == 0 in the tissue-positions table
(visium_10x mode) are background outside the tissue and are left out before anything is ranked, by the
shared worker_utils.keep_in_tissue rule; params.in_tissue_filter, a warning and the analysis say how
many (data.n_spots is the count ranked, params.in_tissue_filter.n_spots_supplied the count supplied).
CELLxGENE Visium exports carry every array spot, and 56-70% of them are background on the library's
samples.

Which matrix is ranked: normalize=True (the default) runs svgbit's log-CPM normalizer, which treats the
matrix as counts, so the shared worker_utils.choose_counts_matrix rule applies after the background is
left out: a matrix with negative or non-finite values (scaled / z-scored data) is refused, naming
use_raw_counts when adata.raw holds counts (log-CPM of z-scores made 45.9% of a CELLxGENE Skin slide
NaN/inf, which STDataset then filled with 0 before ranking), and a non-negative non-integer one runs with
a warning. use_raw_counts=True ranks adata.raw.X (h5ad mode; the 10x count matrix is counts already).
normalize=False ranks the matrix as supplied, which accepts normalised or scaled data by design; only
non-finite values are refused there. params.expression_source and params.x_matrix_kind say which matrix
ran and what X held. min_counts counts the spots where a gene is non-zero (a negative value is non-zero).

The worker's stdout carries only its JSON: the pipeline runs with stdout redirected to stderr.

A filter or normalizer that was asked for and is missing from the installed svgbit stops the run.
svgbit's density step builds a dense n_spots x n_spots neighbour matrix per pool worker; its size is
estimated before the run (memory read with worker_utils.available_memory_bytes, page cache counted as
reclaimable), which stops with the numbers when it cannot fit (no tissue spot is ever dropped for it).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from worker_utils import (
    TISSUE_POSITIONS_NAMES,
    WorkerOutput,
    available_memory_bytes,
    build_svg_analysis,
    choose_counts_matrix,
    describe_reduction,
    expression_matrix_kind,
    find_tissue_positions,
    keep_in_tissue,
    make_names_unique_and_report,
    read_tissue_positions,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    spatial_coords,
    unsupported_choice_msg,
)

METHOD_NAME = "SVGbit (svgbit.run: KNN weights, local Moran's I hotspots, AI/Di density, hotspot clustering)"

# The portal's defaults. A key absent from a direct worker call runs at these AND is reported as
# these: the old tri-state (absent -> not passed -> svgbit's own n_svgs=1000 / n_svg_clusters=8 /
# cores=cpu_count()) was published as 200 / 5 / 1, three numbers the run never used.
DEFAULT_K = 6
DEFAULT_MAX_GENES = 2000
DEFAULT_MIN_COUNTS = 1
DEFAULT_QUANTILE = 0.99
DEFAULT_N_SVGS = 1000
DEFAULT_N_SVG_CLUSTERS = 5
DEFAULT_CORES = 1
DEFAULT_TOP_K = 20


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr, flush=True)


def _json_out(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and Path(p).exists():
            return p
    return None


def _make_unique(names) -> list:
    """Make names unique with suffix .1 .2 ... (pandas-private-free)."""
    seen = {}
    out = []
    for n in list(names):
        n = str(n)
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
    return out


#: Every numeric payload key the worker reads through ``_int_param`` / ``_float_param``. A literal tuple
#: so a reader of the source (and the smoke-driver key census in test/) can see which keys are read;
#: the two readers refuse any other key, so a typo in a call site cannot silently read nothing.
NUMERIC_PAYLOAD_KEYS = (
    "k",
    "max_genes",
    "min_counts",
    "low_variance_var",
    "n_svgs",
    "n_svg_clusters",
    "cores",
    "top_k_genes",
)


def _int_param(params: dict[str, Any], key: str, default: int) -> int:
    """``params[key]`` as an int, or ``default`` when the key is absent or null.

    One reader for the value the run uses and the value the payload reports, so the two cannot
    drift apart again.
    """
    if key not in NUMERIC_PAYLOAD_KEYS:
        raise KeyError(f"_int_param: {key!r} is not in NUMERIC_PAYLOAD_KEYS")
    value = params.get(key)
    return int(default) if value is None else int(value)


def _float_param(params: dict[str, Any], key: str, default: float) -> float:
    if key not in NUMERIC_PAYLOAD_KEYS:
        raise KeyError(f"_float_param: {key!r} is not in NUMERIC_PAYLOAD_KEYS")
    value = params.get(key)
    return float(default) if value is None else float(value)


def _quantile_param(params: dict[str, Any]) -> float | None:
    """The quantile_filter setting: absent -> the default, explicit null -> no filter."""
    if "quantile" not in params:
        return DEFAULT_QUANTILE
    value = params.get("quantile")
    if value is None:
        return None
    q = float(value)
    if not (0.0 < q <= 1.0):
        raise ValueError(
            f"quantile={q} is outside (0, 1]. svgbit's quantile_filter keeps genes whose mean is strictly "
            "below this quantile of all gene means, so quantile<=0 keeps no gene and a value above 1 is "
            "not a quantile. Pass quantile=1.0 to run without the filter."
        )
    return q


def _load_visium_10x(counts_h5: str, spatial_dir: str):
    import scanpy as sc

    eprint(f"[SVGBit] Reading 10x h5: {counts_h5}")
    adata = sc.read_10x_h5(counts_h5)

    try:
        # Records the count under adata.uns so run_svgbit_pipeline can put it on the payload:
        # this tool's answer IS a gene ranking, and a deduplicated symbol is one we invented.
        make_names_unique_and_report(adata)
    except Exception:
        pass

    # Both Space Ranger spellings, in the outs/ directory or the spatial/ inside it. Hand-rolled
    # here until the shared reader existed, which cost this worker the root-level lookup its peers
    # have and left it reading the v2 header row as a spot called "barcode" on a re-export.
    pos_csv = find_tissue_positions(spatial_dir)
    if not pos_csv:
        raise FileNotFoundError(
            f"Could not find {' or '.join(TISSUE_POSITIONS_NAMES)} under {spatial_dir} or its spatial/ subdirectory"
        )

    df = read_tissue_positions(pos_csv).set_index("barcode", drop=True)

    missing = [b for b in adata.obs_names if b not in df.index]
    if missing:
        raise ValueError(
            f"Spatial positions missing for {len(missing)} barcodes (example: {missing[0]}). "
            f"Check counts_h5 and spatial_dir belong to the same sample."
        )

    xy = df.loc[adata.obs_names, ["pxl_col_in_fullres", "pxl_row_in_fullres"]].copy()
    xy.columns = ["x", "y"]
    adata.obs["x"] = xy["x"].to_numpy()
    adata.obs["y"] = xy["y"].to_numpy()
    # Space Ranger's own tissue flag, so a raw_feature_bc_matrix.h5 (every array spot) is cut to the
    # tissue by the same rule as an h5ad; a filtered matrix holds in-tissue spots only and is unchanged.
    adata.obs["in_tissue"] = df.loc[adata.obs_names, "in_tissue"].to_numpy()

    # The image sits beside the spot table, so look in the directory the search landed in. Rebuilding
    # it from the argument was right for only one of the two spellings above, and reported no image
    # at all to a caller who passed the spatial/ directory.
    spatial_path = Path(pos_csv).parent
    auto_he = _first_existing(
        str(spatial_path / "tissue_hires_image.png"),
        str(spatial_path / "tissue_lowres_image.png"),
    )

    eprint(f"[SVGBit] Loaded: obs={adata.n_obs}, vars={adata.n_vars}, auto_he={auto_he}")
    return adata, auto_he


def _h5ad_coordinates(adata) -> str:
    """Put the spot coordinates in obs['x'] / obs['y'] and return where they came from.

    obsm['spatial'] goes through ``spatial_coords``, which refuses a three-column key instead of
    silently taking its first two columns (that would lay a serial stack on one plane).
    """
    import numpy as np

    if "x" in adata.obs and "y" in adata.obs:
        return "obs['x'], obs['y']"
    if "center_x" in adata.obs and "center_y" in adata.obs:
        adata.obs["x"] = adata.obs["center_x"]
        adata.obs["y"] = adata.obs["center_y"]
        return "obs['center_x'], obs['center_y']"
    if "spatial" in adata.obsm:
        xy, _note = spatial_coords(adata, "spatial", 2, tool="svgbit")
        adata.obs["x"] = np.asarray(xy[:, 0], dtype=np.float64)
        adata.obs["y"] = np.asarray(xy[:, 1], dtype=np.float64)
        return "obsm['spatial']"
    raise ValueError("h5ad needs obs['x','y'], obs['center_x','center_y'], or obsm['spatial']")


def _subset_genes_by_variance(adata, max_genes: int, min_counts: int, report: dict | None = None):
    """Drop genes seen in fewer than ``min_counts`` spots, then keep the ``max_genes`` most variable.

    The min_counts cut applies whatever ``max_genes`` is. It used to sit below an early return taken
    whenever ``max_genes <= 0`` or the panel already had ``max_genes`` genes or fewer, so on a
    targeted panel or with ``max_genes=0`` it never ran while ``params`` still echoed it -- the
    1,186-gene SpinalCord panel then ranked 45 genes that are zero in every spot, at AI 0, as if they
    had been tested and found not spatially variable.

    ``report`` receives ``n_after_min_counts`` and ``n_after_variance_cap``. The variance ranking and
    the gene order it produces are unchanged, so a run whose survivors are unchanged is unchanged.
    """
    import numpy as np
    import scipy.sparse as sp

    report = {} if report is None else report
    X = adata.X
    # Non-zero, as documented: ``X > 0`` counted a negative value (normalize=False on scaled data) as
    # absent. On counts the two are the same.
    if sp.issparse(X):
        counts = np.asarray((X != 0).sum(axis=0)).ravel()
    else:
        counts = np.asarray((np.asarray(X) != 0).sum(axis=0)).ravel()

    keep = counts >= int(min_counts)
    n_kept = int(keep.sum())
    report["n_after_min_counts"] = n_kept
    if n_kept == 0:
        raise ValueError(
            f"min_counts={min_counts} removed every gene: none of the {adata.n_vars} genes is non-zero in "
            f"{min_counts} or more of the {adata.n_obs} spots. Lower min_counts (0 keeps every gene)."
        )

    if max_genes is None or int(max_genes) <= 0 or n_kept <= int(max_genes):
        report["n_after_variance_cap"] = n_kept
        if n_kept == adata.n_vars:
            return adata
        return adata[:, keep].copy()

    if sp.issparse(X):
        mu = np.asarray(X.mean(axis=0)).ravel()
        ex2 = np.asarray(X.power(2).mean(axis=0)).ravel()
        var = ex2 - mu**2
    else:
        var = np.asarray(np.asarray(X).var(axis=0)).ravel()

    idx = np.where(keep)[0]
    var_kept = var[idx]
    top = idx[np.argsort(-var_kept)[: int(max_genes)]]
    report["n_after_variance_cap"] = int(len(top))
    return adata[:, top].copy()


def _to_count_coord_df(adata) -> tuple[Any, Any]:
    """The spots x genes counts and the x/y coordinates, as svgbit's STDataset takes them.

    A sparse ``X`` stays sparse: STDataset converts its counts to a sparse DataFrame anyway
    (``make_sparse=True``), so ``X.toarray()`` here only added a dense float64 copy that stayed alive
    for the whole run. The values handed over are the same either way.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    index = adata.obs_names.astype(str)
    columns = _make_unique(adata.var_names)
    X = adata.X
    if sp.issparse(X):
        count_df = pd.DataFrame.sparse.from_spmatrix(sp.csc_matrix(X), index=index, columns=columns)
    else:
        count_df = pd.DataFrame(np.asarray(X), index=index, columns=columns)

    coord_df = pd.DataFrame(
        {"x": adata.obs["x"].to_numpy(), "y": adata.obs["y"].to_numpy()},
        index=index,
    )
    coord_df = coord_df.loc[count_df.index]

    xy = np.asarray(coord_df.to_numpy(), dtype=np.float64)
    bad = ~np.isfinite(xy).all(axis=1)
    if bad.any():
        example = str(coord_df.index[np.where(bad)[0][0]])
        raise ValueError(
            f"{int(bad.sum())} of {len(coord_df)} spots have a missing or non-finite x/y coordinate "
            f"(first: {example!r}). svgbit's KNN graph needs a finite position for every spot."
        )
    return count_df, coord_df


def _resolve_stdataset_class():
    """
    Robustly locate the STDataset class across svgbit versions.

    Common patterns seen in the wild:
      - from svgbit.core.STDataset import STDataset   (class)
      - import svgbit.core.STDataset as STDataset     (module containing class STDataset)
      - from svgbit.core.STDataset.STDataset import STDataset (class)
    """
    import importlib
    import types

    candidates = [
        ("svgbit.core.STDataset", "STDataset"),
        ("svgbit.core.STDataset.STDataset", "STDataset"),
        ("svgbit.core", "STDataset"),
    ]

    errs = []
    for modname, attr in candidates:
        try:
            mod = importlib.import_module(modname)
            obj = getattr(mod, attr, None)
            if obj is None:
                continue

            # If we got a module (not callable), try to fetch class inside it.
            if isinstance(obj, types.ModuleType):
                inner = getattr(obj, "STDataset", None) or getattr(obj, "Dataset", None)
                if callable(inner):
                    return inner
                errs.append(f"{modname}.{attr} is module; no callable STDataset/Dataset inside")
                continue

            # If callable (class/function), return it.
            if callable(obj):
                return obj

            errs.append(f"{modname}.{attr} found but not callable: {type(obj)}")
        except Exception as e:
            errs.append(f"{modname}.{attr} import failed: {e!r}")

    raise RuntimeError(f"Could not resolve callable STDataset class from svgbit. Tried: {candidates}. Details: {errs}")


def _construct_dataset(count_df, coord_df):
    STDatasetCls = _resolve_stdataset_class()

    errs = []
    # svgbit's own spelling first. Trying ``coord_df=`` first made every run fail once inside
    # STDataset.__init__ and print an "Exception ignored in STDataset.__del__" traceback to stderr.
    for kwargs in [
        {"count_df": count_df, "coordinate_df": coord_df},
        {"count_df": count_df, "coord_df": coord_df},
        {"counts": count_df, "coord": coord_df},
    ]:
        try:
            return STDatasetCls(**kwargs)
        except Exception as e:
            errs.append(repr(e))

    try:
        return STDatasetCls(count_df, coord_df)
    except Exception as e:
        errs.append(repr(e))

    raise RuntimeError(f"Failed to construct STDataset (tried multiple signatures). Errors: {errs}")


def _n_genes_in(dataset) -> int:
    """How many genes an STDataset holds right now (its count_df columns)."""
    count_df = getattr(dataset, "count_df", None)
    if count_df is not None:
        return int(count_df.shape[1])
    return int(dataset.n_genes)


def _maybe_apply_filters(dataset, low_variance_var: float, quantile: float | None, report: dict | None = None):
    """Run the svgbit gene filters the caller asked for, and record how many genes each one left.

    ``quantile_filter`` keeps genes whose mean is strictly below the quantile
    (svgbit/core/filters.py), so it drops genes even at quantile=1.0 -- the highest-mean gene
    always goes. 1.0 (or None) therefore means "no quantile filter" and the call is skipped. A
    filter that was asked for and is missing from the installed svgbit stops the run; it used to be
    skipped in silence while params still said it ran.
    """
    report = {} if report is None else report
    want_low_variance = low_variance_var is not None and float(low_variance_var) > 0
    want_quantile = quantile is not None and float(quantile) < 1.0
    report["low_variance_filter_applied"] = False
    report["quantile_filter_applied"] = False
    if not (want_low_variance or want_quantile):
        return dataset

    import svgbit

    filt = getattr(svgbit, "filters", None)

    if want_low_variance:
        lvf = getattr(filt, "low_variance_filter", None)
        if not callable(lvf):
            raise RuntimeError(
                f"low_variance_var={low_variance_var} asks for svgbit.filters.low_variance_filter, which this "
                "svgbit installation does not provide. Pass low_variance_var=0 to run without it."
            )
        eprint(f"[SVGBit] Applying low_variance_filter(var={low_variance_var})")
        dataset = lvf(dataset, float(low_variance_var))
        report["low_variance_filter_applied"] = True
        report["n_after_low_variance_filter"] = _n_genes_in(dataset)

    if want_quantile:
        qf = getattr(filt, "quantile_filter", None)
        if not callable(qf):
            raise RuntimeError(
                f"quantile={quantile} asks for svgbit.filters.quantile_filter, which this svgbit installation "
                "does not provide. Pass quantile=1.0 to run without it."
            )
        eprint(f"[SVGBit] Applying quantile_filter(quantile={quantile})")
        dataset = qf(dataset, float(quantile))
        report["quantile_filter_applied"] = True
        report["n_after_quantile_filter"] = _n_genes_in(dataset)

    return dataset


def _maybe_normalize(dataset, normalize: bool, report: dict | None = None):
    """Apply svgbit's log-CPM normalizer when asked; record which one ran (None when not asked).

    A requested normalization the installed svgbit cannot perform stops the run. It used to print a
    stderr warning and rank raw counts while the payload still said normalize=true.
    """
    report = {} if report is None else report
    report["normalizer"] = None
    if not normalize:
        return dataset

    import svgbit

    norms = getattr(svgbit, "normalizers", None)
    candidates = [
        "logcpm_normalizer",
        "logcpm_normailzer",  # typo in some versions/docs
    ]
    for name in candidates:
        fn = getattr(norms, name, None)
        if callable(fn):
            eprint(f"[SVGBit] Applying normalization via svgbit.normalizers.{name}()")
            report["normalizer"] = f"svgbit.normalizers.{name}"
            return fn(dataset)

    raise RuntimeError(
        "normalize=true asks for svgbit.normalizers.logcpm_normalizer, which this svgbit installation does "
        "not provide. Pass normalize=false to rank the counts as supplied."
    )


def _run_svgbit(
    dataset,
    k: int,
    n_svgs: int | None,
    n_svg_clusters: int | None,
    cores: int | None,
    report: dict | None = None,
):
    """Call ``svgbit.run``. A knob its signature lacks is listed in ``report['ignored']``."""
    import inspect

    import svgbit

    report = {} if report is None else report
    run_fn = getattr(svgbit, "run", None)
    if not callable(run_fn):
        raise RuntimeError("svgbit.run not found; svgbit installation/API mismatch in this env.")

    try:
        sig = inspect.signature(run_fn)
        params = sig.parameters
        supports = lambda p: p in params
    except Exception:
        supports = lambda p: True

    kwargs = {}
    ignored = []
    for name, value in (("k", k), ("n_svgs", n_svgs), ("n_svg_clusters", n_svg_clusters), ("cores", cores)):
        if value is None:
            continue
        if supports(name):
            kwargs[name] = int(value)
        else:
            ignored.append(name)
    report["ignored"] = ignored

    eprint("[SVGBit] Running SVGBit ...")
    out = run_fn(dataset, **kwargs) if kwargs else run_fn(dataset)
    return out if out is not None else dataset


def _as_dataframe(obj):
    import numpy as np
    import pandas as pd

    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, dict):
        return pd.DataFrame.from_dict(obj, orient="index")
    if hasattr(obj, "to_pandas"):
        try:
            return obj.to_pandas()
        except Exception:
            pass
    try:
        return pd.DataFrame(obj)
    except Exception:
        try:
            arr = np.asarray(obj)
            if arr.ndim == 1:
                return pd.DataFrame({"value": arr})
        except Exception:
            pass
    return None


def _atomic_to_csv(df, path: str) -> None:
    """Write ``df`` to ``path`` through a ``.partial`` sibling, so a killed run leaves no half file."""
    tmp = path + ".partial"
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _safe_to_csv(obj, path: str, errors: list | None = None) -> str | None:
    df = _as_dataframe(obj)
    if df is None:
        return None
    try:
        # Ensure gene names stored in the index are exposed as a named column
        if df.index.name is None and "gene" not in df.columns:
            df = df.reset_index()
            df.columns = ["gene"] + list(df.columns[1:])
        elif df.index.name is not None:
            # A named index is the row identity -- the spot barcodes of hotspot_df / Di when the h5ad's
            # obs index is named (SI_P12's is 'barcode'). index=False below used to drop it, leaving
            # those tables with no way to tell which row is which spot.
            name = str(df.index.name)
            if name in df.columns:
                name = name + "_index"
            df = df.rename_axis(name).reset_index()
        _atomic_to_csv(df, path)
        return path
    except Exception as exc:
        eprint(f"[SVGBit] Could not write {path} ({type(exc).__name__}: {exc})")
        if errors is not None:
            errors.append(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
        return None


def _extract_top_table(dataset, top_k: int):
    import pandas as pd

    AI = getattr(dataset, "AI", None)
    Di = getattr(dataset, "Di", None)

    try:
        if AI is not None:
            ai = AI if isinstance(AI, pd.Series) else pd.Series(AI)

            # Di is only a tiebreaker for equal AI. SVGbit returns it as a spot-by-gene matrix
            # (200 x 495 on a real Visium slide), and pd.Series() on a 2-D object raises
            # "Data must be 1-dimensional" -- which the bare except below used to swallow,
            # throwing away the perfectly usable AI ranking with it and reporting zero SVGs.
            di = None
            if Di is not None:
                try:
                    di = Di if isinstance(Di, pd.Series) else pd.Series(Di)
                    di = di.reindex(ai.index)
                except Exception as exc:
                    eprint(f"[SVGBit] Di unusable as a tiebreaker ({type(exc).__name__}: {exc}); ranking on AI alone")
                    di = None

            df = pd.DataFrame({"gene": ai.index.astype(str), "AI": ai.to_numpy()})
            if di is not None:
                df["Di"] = di.to_numpy()

            sort_cols = ["AI"] + (["Di"] if "Di" in df.columns else [])
            df = df.sort_values(sort_cols, ascending=False)
            df.attrs["ranked_by"] = "AI" + (" (Di tiebreak)" if "Di" in df.columns else "")
            return df.head(top_k), df
    except Exception as exc:
        # Silence here is how a whole missing ranking went unnoticed: the worker still exits 0
        # with status "ok", so stderr is the only place the loss can surface.
        eprint(f"[SVGBit] Could not rank genes from AI ({type(exc).__name__}: {exc}); trying hotspot_df")

    hotspot_df = getattr(dataset, "hotspot_df", None)
    hdf = _as_dataframe(hotspot_df)
    if hdf is not None:
        score_col = None
        for c in ["AI", "score", "hotspot", "hotspot_score", "stat", "pval", "qval"]:
            if c in hdf.columns:
                score_col = c
                break
        if score_col is not None:
            df = hdf.copy()
            if "gene" not in df.columns:
                df = df.reset_index().rename(columns={"index": "gene"})
            asc = True if score_col in ["pval", "qval"] else False
            df = df.sort_values([score_col], ascending=asc)
            df.attrs["ranked_by"] = f"hotspot_df column {score_col!r}"
            return df.head(top_k), df

    return None, None


# ----------------------------------------------------------------------------- memory


def _density_step_bytes(n_spots: int, cores: int) -> int:
    """A floor on the memory svgbit's density step needs, in bytes.

    ``density.hotspot_AI`` turns the KNN weights into a dense spot-by-spot int64 DataFrame
    (``utils.pysal_to_pandas``: ``W.full()`` is a float64 n x n array, then ``astype(int)``) and hands
    it to the worker pool inside every task it pickles. So the parent holds two n x n arrays while
    converting and each of ``cores`` workers unpickles its own copy. The matrix is intrinsic to the
    method -- svgbit reads each hotspot's neighbours off its columns -- so it is estimated, not avoided.
    """
    n = int(n_spots)
    return 8 * n * n * (2 + max(1, int(cores)))


def _check_density_memory(n_spots: int, cores: int, available: int | None = None) -> int:
    """Refuse, with the numbers, a slide whose dense neighbour matrix cannot fit. Returns the floor.

    ``available`` defaults to the shared ``worker_utils.available_memory_bytes``: the smaller of
    MemAvailable and the room under the cgroup limit, with the cgroup's page cache (``active_file`` +
    ``inactive_file``) counted as reclaimable. The reader this worker used to carry took the room as
    ``memory.max - memory.current``, and ``memory.current`` counts that cache, so right after the h5ad
    was read a memory-limited container looked full and a run that fits was refused.
    """
    need = _density_step_bytes(n_spots, cores)
    avail = available_memory_bytes() if available is None else int(available)
    if avail is None or need <= avail:
        return need
    per_copy = 8 * int(n_spots) * int(n_spots)
    at_one = _density_step_bytes(n_spots, 1)
    advice = (
        f"Each pool worker holds its own copy, so cores=1 needs at least {at_one / 1e9:.1f} GB."
        if int(cores) > 1 and at_one <= avail
        else "Run it on a machine with that much free memory."
    )
    raise MemoryError(
        f"svgbit's density step builds a dense {n_spots} x {n_spots} spot-neighbour matrix "
        f"({per_copy / 1e9:.1f} GB per copy). With cores={cores} it needs at least {need / 1e9:.1f} GB "
        f"(two copies in the parent, one per worker) and {avail / 1e9:.1f} GB is available (MemAvailable / "
        f"cgroup room, page cache counted as reclaimable). {advice}"
    )


# ----------------------------------------------------------------------------- pipeline


def _gene_cut_reason(max_genes: int, min_counts: int, low_variance_var: float, quantile, stages: dict) -> str:
    """Name every gene cut that removed something, with its own count, in the order it ran."""
    parts = []
    before = stages.get("n_supplied", 0)
    steps = [
        ("n_after_min_counts", f"min_counts={min_counts} (genes non-zero in fewer spots)"),
        ("n_after_variance_cap", f"the top-{max_genes} variance prefilter (max_genes={max_genes})"),
        ("n_after_low_variance_filter", f"svgbit's low_variance_filter(var={low_variance_var})"),
        (
            "n_after_quantile_filter",
            f"svgbit's quantile_filter(quantile={quantile}), which drops the genes whose mean is at or "
            "above that quantile of all gene means -- the most highly expressed genes",
        ),
    ]
    for key, label in steps:
        if key not in stages:
            continue
        after = int(stages[key])
        if after < before:
            parts.append(f"{label}: {before - after}")
        before = after
    return "; ".join(parts)


def _choose_matrix(adata, normalize: bool, use_raw_counts: bool):
    """``(adata, info)``: the matrix svgbit ranks, by the shared ``choose_counts_matrix`` rule.

    normalize=True runs svgbit's log-CPM on the matrix, i.e. treats it as counts: negative or non-finite
    values are refused (naming use_raw_counts when adata.raw holds counts) and a non-integer matrix is
    warned about. use_raw_counts=True ranks adata.raw.X either way. normalize=False without it ranks the
    matrix as supplied -- normalised or scaled data is accepted by design there -- and refuses only
    non-finite values, which STDataset would otherwise fill with 0 without a word.
    """
    if normalize or use_raw_counts:
        return choose_counts_matrix(adata, use_raw_counts)
    kind = expression_matrix_kind(adata.X)
    if kind == "nonfinite":
        raise ValueError(
            "X holds NaN or infinite values; svgbit's STDataset fills NaN with 0 and keeps infinities, so the "
            "ranking would rest on values that were never measured. Supply a finite matrix."
        )
    return adata, {"expression_source": "X", "x_matrix_kind": kind, "warning": None}


def run_svgbit_pipeline(params: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import scanpy as sc

    # SVGbit calls its spatially variable genes from esda's conditional-randomization Moran's I,
    # which draws 999 permutations out of numpy's global RNG and takes no seed of its own. Seed
    # before anything reaches the library: unseeded, two runs of one slide differed by up to 0.215
    # in AI (a score bounded in [0, 1]) and returned different gene lists. svgbit's local_moran
    # forks a Pool, so the children inherit this state -- but Pool.map chunks genes by worker count,
    # so a fixed seed reproduces a run only at the same `cores`.
    random_seed = int(params.get("random_seed", 0))
    np.random.seed(random_seed)

    output_dir = _ensure_dir(str(params["output_dir"]))

    # Each knob is read once: the value the run uses is the value the payload reports.
    k = _int_param(params, "k", DEFAULT_K)
    max_genes = _int_param(params, "max_genes", DEFAULT_MAX_GENES)
    min_counts = _int_param(params, "min_counts", DEFAULT_MIN_COUNTS)
    low_variance_var = _float_param(params, "low_variance_var", 0.0)
    quantile = _quantile_param(params)
    normalize = bool(params.get("normalize", True))
    use_raw_counts = bool(params.get("use_raw_counts", False))
    n_svgs = _int_param(params, "n_svgs", DEFAULT_N_SVGS)
    n_svg_clusters = _int_param(params, "n_svg_clusters", DEFAULT_N_SVG_CLUSTERS)
    cores = _int_param(params, "cores", DEFAULT_CORES)
    top_k = _int_param(params, "top_k_genes", DEFAULT_TOP_K)
    if top_k < 0:
        # DataFrame.head(-N) is every row except the last N, so a negative value used to report nearly
        # the whole ranking as "the N highest-scoring genes".
        raise ValueError(f"top_k_genes={top_k} is negative; it is the number of top genes returned inline (0 or more).")

    input_mode = params.get("input_mode", "visium_10x")
    auto_he = None
    raw_ignored = False

    if input_mode == "visium_10x":
        adata, auto_he = _load_visium_10x(params["counts_h5"], params["spatial_dir"])
        adata, n_spots_supplied, n_off_tissue = keep_in_tissue(adata, "spots")
        # The Space Ranger matrix is counts and has no adata.raw: use_raw_counts has nothing to select.
        raw_ignored = use_raw_counts
        adata, counts_info = _choose_matrix(adata, normalize, False)
        coord_source = "tissue positions pxl_col_in_fullres, pxl_row_in_fullres"
    elif input_mode == "h5ad":
        adata = sc.read_h5ad(params["adata_path"])
        # Background first, so it neither gets coordinates checked nor enters the n^2 density step.
        adata, n_spots_supplied, n_off_tissue = keep_in_tissue(adata, "spots")
        # Then the matrix (counts or refuse), over the in-tissue spots; obs and obsm are kept.
        adata, counts_info = _choose_matrix(adata, normalize, use_raw_counts)
        coord_source = _h5ad_coordinates(adata)
        try:
            make_names_unique_and_report(adata)
        except Exception:
            pass
        auto_he = params.get("he_image")
        eprint(f"[SVGBit] Loaded h5ad: obs={adata.n_obs}, vars={adata.n_vars}")
    else:
        raise ValueError(unsupported_choice_msg("input_mode", input_mode, ["visium_10x", "h5ad"]))
    if n_off_tissue:
        eprint(f"[SVGBit] Left out {n_off_tissue} of {n_spots_supplied} spots with in_tissue == 0 (background)")

    # Before hours of Moran permutations, not after: the dense matrix is allocated in the last step.
    _check_density_memory(int(adata.n_obs), cores)

    # What the user handed us: the subset below rebinds adata, after which adata.n_vars is the
    # count of survivors and the size of the supplied panel is unrecoverable.
    n_genes_supplied = int(adata.n_vars)
    gene_stages: dict[str, Any] = {"n_supplied": n_genes_supplied}
    adata = _subset_genes_by_variance(adata, max_genes=max_genes, min_counts=min_counts, report=gene_stages)
    eprint(f"[SVGBit] After min_counts / max_genes: obs={adata.n_obs}, vars={adata.n_vars}")

    count_df, coord_df = _to_count_coord_df(adata)
    dataset = _construct_dataset(count_df, coord_df)
    del count_df  # STDataset keeps its own copy

    dataset = _maybe_apply_filters(dataset, low_variance_var=low_variance_var, quantile=quantile, report=gene_stages)

    # The genes svgbit ranks: every cut above has happened. Publishing adata.n_vars here overstated
    # the ranking by whatever svgbit's own filters removed (12 genes of 1,186 at quantile=0.99).
    n_genes_ranked = _n_genes_in(dataset)
    if n_genes_ranked < 2:
        raise ValueError(
            f"{n_genes_ranked} gene(s) of the {n_genes_supplied} supplied survived the gene filters "
            f"({_gene_cut_reason(max_genes, min_counts, low_variance_var, quantile, gene_stages)}); svgbit "
            "clusters the ranked genes and needs at least 2. Relax min_counts, low_variance_var or quantile."
        )

    run_notes: dict[str, Any] = {}
    dataset = _maybe_normalize(dataset, normalize, report=run_notes)

    dataset = _run_svgbit(
        dataset,
        k=k,
        n_svgs=n_svgs,
        n_svg_clusters=n_svg_clusters,
        cores=cores,
        report=run_notes,
    )

    write_errors: list[str] = []
    out_paths: dict[str, str | None] = {}
    out_paths["hotspot_df_csv"] = _safe_to_csv(
        getattr(dataset, "hotspot_df", None), os.path.join(output_dir, "hotspot_df.csv"), write_errors
    )
    out_paths["AI_csv"] = _safe_to_csv(getattr(dataset, "AI", None), os.path.join(output_dir, "AI.csv"), write_errors)
    out_paths["Di_csv"] = _safe_to_csv(getattr(dataset, "Di", None), os.path.join(output_dir, "Di.csv"), write_errors)
    out_paths["svg_cluster_csv"] = _safe_to_csv(
        getattr(dataset, "svg_cluster", None), os.path.join(output_dir, "svg_cluster.csv"), write_errors
    )

    top_df, ranked_df = _extract_top_table(dataset, top_k=top_k)
    if ranked_df is None:
        # A tool whose whole answer is a gene ranking must not report success without one.
        raise RuntimeError(
            f"svgbit.run returned no gene ranking (dataset.AI is empty) for {n_genes_ranked} genes x "
            f"{int(adata.n_obs)} spots, so there is no result to report."
        )
    ranked_by = ranked_df.attrs.get("ranked_by", "AI")

    top_table = []
    try:
        # Ensure gene names are in a named column, not just the index
        if "gene" not in ranked_df.columns:
            ranked_df = ranked_df.reset_index()
            ranked_df.columns = ["gene"] + list(ranked_df.columns[1:])
        ranked_path = os.path.join(output_dir, "svg_ranked.csv")
        _atomic_to_csv(ranked_df, ranked_path)
        out_paths["svg_ranked_csv"] = ranked_path
    except Exception as exc:
        eprint(f"[SVGBit] Could not write svg_ranked.csv ({type(exc).__name__}: {exc})")
        write_errors.append(f"svg_ranked.csv: {type(exc).__name__}: {exc}")
    if top_df is not None:
        try:
            top_table = top_df.to_dict(orient="records")
        except Exception:
            top_table = []

    # Extract top gene names for summary
    top_gene_names = []
    if top_df is not None and "gene" in top_df.columns:
        top_gene_names = top_df["gene"].astype(str).tolist()

    # _extract_top_table ranks by AI (then Di) and slices head(top_k). Those are continuous
    # scores with no null model, so the slice size is the caller's request, not a count of
    # significant genes -- reporting it as one made the discovery rate track top_k_genes.
    n_top_reported = len(top_gene_names)

    out_obj = WorkerOutput("svgbit", task="svg_identification")
    out_obj.set_data(
        # The spots ranked. When background spots (in_tissue == 0) were left out above, the count
        # supplied is params.in_tissue_filter.n_spots_supplied (record_in_tissue below).
        n_spots=int(adata.n_obs),
        n_genes=n_genes_supplied,
        n_genes_used=n_genes_ranked,
    )
    out_obj.add_output_files({k: v for k, v in out_paths.items() if v})
    out_obj.add_params(
        {
            "k": k,
            "max_genes": max_genes,
            "min_counts": min_counts,
            "low_variance_var": low_variance_var,
            "quantile": quantile,
            "normalize": normalize,
            "use_raw_counts": use_raw_counts,
            "n_svgs": n_svgs,
            "n_svg_clusters": n_svg_clusters,
            "cores": cores,
            # Recorded beside cores on purpose: the seed reproduces this run only at the same
            # worker count, so publishing one without the other would overstate what it pins.
            "random_seed": random_seed,
        }
    )
    # What actually happened, beside what was asked for.
    stage_params = {
        key: int(gene_stages[key])
        for key in (
            "n_after_min_counts",
            "n_after_variance_cap",
            "n_after_low_variance_filter",
            "n_after_quantile_filter",
        )
        if key in gene_stages
    }
    out_obj.add_params(
        {
            "coord_source": coord_source,
            "n_genes_after_min_counts": stage_params.get("n_after_min_counts"),
            "n_genes_after_variance_cap": stage_params.get("n_after_variance_cap"),
            "low_variance_filter_applied": bool(gene_stages.get("low_variance_filter_applied")),
            "n_genes_after_low_variance_filter": stage_params.get("n_after_low_variance_filter"),
            "quantile_filter_applied": bool(gene_stages.get("quantile_filter_applied")),
            "n_genes_after_quantile_filter": stage_params.get("n_after_quantile_filter"),
            "normalizer": run_notes.get("normalizer"),
            # svgbit clusters the top n_svgs genes by AI; a panel smaller than that clusters them all.
            "n_svgs_clustered": min(n_svgs, n_genes_ranked),
            "ranked_by": ranked_by,
        }
    )
    record_method(out_obj, METHOD_NAME)
    record_in_tissue(out_obj, n_spots_supplied, n_off_tissue)
    record_expression_source(out_obj, counts_info)
    if raw_ignored:
        record_ignored(
            out_obj,
            ["use_raw_counts"],
            "input_mode=visium_10x reads the Space Ranger count matrix, which is counts and has no adata.raw",
        )
    if run_notes.get("ignored"):
        record_ignored(out_obj, run_notes["ignored"], "the installed svgbit.run takes no such argument")
    if not str(ranked_by).startswith("AI"):
        out_obj.add_warning(f"the gene ranking came from {ranked_by}, not from svgbit's AI score")
    for problem in write_errors:
        out_obj.add_warning(f"an output table could not be written: {problem}")
    out_obj.set_summary(
        n_significant=None,
        n_top_reported=n_top_reported,
        top_genes=top_gene_names,
        top_svg_genes=top_table,
    )
    gene_note = describe_reduction(
        "genes",
        n_genes_supplied,
        n_genes_ranked,
        reason=_gene_cut_reason(max_genes, min_counts, low_variance_var, quantile, gene_stages),
    )
    if gene_note:
        out_obj.add_warnings(gene_note.strip())
    # record_in_tissue above already warns; the sentence the agent quotes says it too.
    spot_note = describe_reduction(
        "spots",
        int(n_spots_supplied),
        int(adata.n_obs),
        reason="the in_tissue flag (in_tissue == 0 marks background outside the tissue)",
    )
    out_obj.set_analysis(
        build_svg_analysis(
            n_genes_ranked,
            None,
            top_gene_names,
            method_name="SVGBit",
            n_genes_renamed=int(adata.uns.get("identifier_renames", {}).get("n_genes_renamed", 0)),
        )
        + spot_note
        + gene_note
    )
    if auto_he:
        out_obj.add_extra("auto_he_image", auto_he)
    return out_obj.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JSON string payload")
    args = ap.parse_args()

    try:
        params = json.loads(args.json)
        tool = params.get("__tool__")
        if tool != "svgbit_run":
            raise ValueError(f"Unknown __tool__: {tool}")

        # stdout carries only the JSON: svgbit and its libraries print (STDataset announces renamed
        # genes with print()), so the pipeline runs with stdout sent to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            payload = run_svgbit_pipeline(params)
        _json_out(payload)

    except Exception as e:
        WorkerOutput.emit_error("svgbit", str(e), task="svg_identification")
        sys.exit(1)


if __name__ == "__main__":
    main()
