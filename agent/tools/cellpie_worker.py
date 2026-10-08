#!/usr/bin/env python
"""
cellpie_worker.py

Worker script for CellPie intNMF -- a reference-free topic model of a spatial slide.

- Executed inside the cellpie conda env: /opt/conda/envs/cellpie_env
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

What runs
---------
CellPie's ``intNMF`` jointly factorises the spot x gene expression matrix and a spot x image-feature
matrix (``obsm['features']``) into ``n_components`` non-negative topics. The output columns are
``topic_0..topic_{n-1}``: latent expression programmes learned from the slide alone. No scRNA-seq
reference takes part -- ``--sc-h5ad`` / ``--cell-type-key`` are accepted for compatibility and are
reported under ``params.ignored``.

The image modality is decided in this order and recorded in ``params.method`` / ``params.image_modality``:

1. ``obsm['features']`` present -> upstream's joint mode (``mod1_skew=1``).
2. ``--allow-expression-only-fallback`` -> upstream's expression-only mode (``mod1_skew=2``), with a
   constant one-column image matrix because ``intNMF.fit`` unconditionally reads ``obsm['features']``.
   The constant's value is not inert (upstream's first-epoch theta rescale and the phi_im
   initialisation read it), so it is fixed at 1.0 and published as
   ``params.expression_only_image_constant``.
3. ``--allow-pca-image-fallback`` -> |PCA of the log-normalised expression| stands in for image
   features, so the "integrative" NMF factorises expression twice. Kept only because earlier runs did
   this silently; it is a fallback and is labelled as one.
4. Otherwise the run stops with a message naming the two knobs.

The stand-in image matrix of modes 2 and 3 is not left in ``obsm['features']`` of the annotated h5ad
(a later run would read it as supplied image features); the |PCA| stand-in is kept as
``obsm['cellpie_pca_image_standin']`` and ``uns['cellpie_method']`` / ``uns['cellpie_image_modality']``
say what ran.

Spots and counts
----------------
Background spots (``obs['in_tissue'] == 0``, as CELLxGENE Visium exports carry them: 3009 of 4992 on
Heart Fetal12W) are left out right after loading (``worker_utils.keep_in_tissue``; reported in
``params.in_tissue_filter``, ``data.n_spots_supplied`` and a warning), so glass is not a topic.
CellPie's recipe normalises X as counts (normalize_total + log1p), so the in-tissue matrix then goes
through ``worker_utils.choose_counts_matrix``: negative or non-finite values (a z-scored X) are
refused, naming ``use_raw_counts`` when ``adata.raw`` holds counts; a non-integer X runs with a
warning; ``--use-raw-counts`` factorises ``adata.raw.X``. (An earlier revision clipped negative values
to 0 and factorised what was left.)

NOTE: CellPie is imported via sys.path from the cloned repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
)

# Default location for the CellPie clone: this clone's own tools/third_party, not a fixed box.
# Nothing sets CELLPIE_REPO -- the setup resolver only rebases *_WORKER/*_PYTHON -- so an absolute
# default here is the sole value the worker sees, and it must follow whatever checkout is running it.
CELLPIE_REPO = os.environ.get(
    "CELLPIE_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "CellPie"),
)

# The honest names of what ran. ``params.method`` carries exactly one of these.
METHOD_JOINT = "CellPie intNMF (joint expression + obsm['features'] image features, mod1_skew=1)"
METHOD_EXPRESSION_ONLY = "CellPie intNMF expression-only (mod1_skew=2; no image features)"
METHOD_PCA_IMAGE = "CellPie intNMF with |PCA of log-normalised expression| standing in for image features (fallback)"

DEFAULT_EPOCHS = 20
# The value of the one-column image matrix the expression-only mode hands intNMF.fit. With
# mod1_skew=2 the image term drops out of the theta HALS update, but not out of upstream's
# first-epoch theta rescale (the mean of the expression and image ratios), the phi_im random
# initialisation (scaled by the image matrix's mean) or the summed loss that drives early stopping.
# So the value changes the fit; it stays 1.0 so earlier results reproduce, and it is published.
EXPRESSION_ONLY_IMAGE_CONSTANT = 1.0
DEFAULT_HVG_FLAVOR = "seurat"
PCA_N_TOP_GENES = 2000
PCA_N_COMPS = 50
# Upstream ``fit`` builds a dense spots x genes DataFrame and a CSR copy of it beside the input.
DENSE_COPIES = 2.5
# What is done to X before ``fit``, said once and published as ``params.preprocessing``.
PREPROCESSING = (
    "normalize_total(target_sum=1e4) + log1p on X, every gene kept; image features used as supplied (upstream's "
    "preprocess_data_visium also drops genes seen in <100 spots and MinMax-scales obsm['features']; neither is done here)"
)


def log(msg: str) -> None:
    sys.stderr.write(f"[cellpie-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CellPie worker: reference-free intNMF topic model.")
    parser.add_argument("--spatial-h5ad", type=str, required=True, help="Path to spatial AnnData (.h5ad).")
    parser.add_argument(
        "--sc-h5ad",
        type=str,
        default="",
        help="Accepted for compatibility only; CellPie is reference-free and never reads it (reported as ignored).",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--n-components", type=int, default=10, help="Number of NMF topics.")
    parser.add_argument(
        "--cell-type-key",
        type=str,
        default="cell_type",
        help="Accepted for compatibility only; no reference is read, so it has no effect (reported as ignored).",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="intNMF optimisation epochs (upstream default 20)."
    )
    parser.add_argument(
        "--allow-expression-only-fallback",
        action="store_true",
        help="Without obsm['features'], run upstream's expression-only mode (mod1_skew=2) instead of stopping.",
    )
    parser.add_argument(
        "--allow-pca-image-fallback",
        action="store_true",
        help="Without obsm['features'], let |PCA of log-normalised expression| stand in for image features.",
    )
    parser.add_argument(
        "--hvg-flavor",
        type=str,
        default=DEFAULT_HVG_FLAVOR,
        help="HVG flavour for the PCA image-feature fallback only: 'seurat' (default) or 'seurat_v3' (needs scikit-misc).",
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Factorise adata.raw.X (raw counts) instead of X, e.g. for a CELLxGENE h5ad whose X is log-normalised "
        "or scaled. Refused when the h5ad has no adata.raw or it does not hold counts.",
    )
    return parser.parse_args()


def _ensure_repo() -> str:
    """Ensure CellPie repo is cloned and return the repo path."""
    import subprocess

    if os.path.isdir(CELLPIE_REPO) and (
        os.path.exists(os.path.join(CELLPIE_REPO, "CellPie"))
        or os.path.exists(os.path.join(CELLPIE_REPO, "cellpie"))
        or os.path.exists(os.path.join(CELLPIE_REPO, "setup.py"))
    ):
        log(f"CellPie repo found at {CELLPIE_REPO}")
        return CELLPIE_REPO

    log(f"Cloning CellPie to {CELLPIE_REPO}...")
    os.makedirs(os.path.dirname(CELLPIE_REPO), exist_ok=True)
    subprocess.run(
        ["git", "clone", "https://github.com/ManchesterBioinference/CellPie.git", CELLPIE_REPO],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    log("CellPie repo cloned successfully")
    return CELLPIE_REPO


def _import_intnmf(repo_path: str):
    """CellPie's main class is intNMF in CellPie/cellpie_main.py."""
    try:
        from CellPie.cellpie_main import intNMF

        log("Imported intNMF via 'from CellPie.cellpie_main import intNMF'")
        return intNMF
    except ImportError:
        pass
    try:
        from CellPie import cellpie_main

        log("Imported intNMF via CellPie.cellpie_main.intNMF")
        return cellpie_main.intNMF
    except ImportError:
        pass
    import importlib.util

    modpath = os.path.join(repo_path, "CellPie", "cellpie_main.py")
    if os.path.exists(modpath):
        spec = importlib.util.spec_from_file_location("cellpie_main", modpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        log(f"Imported intNMF from {modpath}")
        return mod.intNMF
    raise ImportError(f"Could not import intNMF from {repo_path}. Expected CellPie.cellpie_main.intNMF.")


def _mem_available_bytes():
    """Memory this run can still allocate, in bytes, or None where nothing can be read.

    The shared reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and the
    room under the cgroup memory limit, page cache counted as reclaimable. /proc/meminfo alone reports
    the HOST's free memory inside a memory-limited container, so the refusal below never fired there
    and the kernel killed the run instead (2026-09-30, hunt round 2).
    """
    return available_memory_bytes()


def _dense_bytes(n_obs: int, n_vars: int, itemsize: int) -> int:
    """Bytes of the dense spots x genes matrix upstream ``intNMF.fit`` materialises (times its copies)."""
    return int(n_obs) * int(n_vars) * int(itemsize) * DENSE_COPIES


def _check_dense_budget(n_obs: int, n_vars: int, itemsize: int) -> None:
    """Fail with the numbers before upstream densifies, rather than be killed while it does."""
    need = _dense_bytes(n_obs, n_vars, itemsize)
    avail = _mem_available_bytes()
    if avail is not None and need > avail:
        raise MemoryError(
            f"CellPie's intNMF.fit densifies the expression matrix (a dense {n_obs} x {n_vars} DataFrame plus a "
            f"CSR copy): about {need / 1e9:.1f} GB, but only {avail / 1e9:.1f} GB is available. Reduce the gene "
            "set before calling (e.g. keep highly variable genes and write a new h5ad; report that selection) or "
            "run on a larger machine. The worker never subsamples spots."
        )


def _to_csr_or_dense(X):
    """The only sparse type upstream ``fit`` recognises is ``csr_matrix``; anything else it hands to
    ``pd.DataFrame(data=...)``, which raises on csc/csr_array. Dense stays dense (np.matrix -> ndarray)."""
    import numpy as np
    from scipy import sparse

    if sparse.issparse(X):
        return sparse.csr_matrix(X)
    return np.asarray(X)


def _image_feature_copy(features):
    """A float copy of the supplied ``obsm['features']`` for upstream ``fit``, which zeroes its
    negative entries in place -- the caller's matrix is written back untouched.

    Sparse stays sparse, as the ``csr_matrix`` upstream converts it to anyway (image features can be
    wide; densifying them would be the wrapper's doing, not the method's). Dense input (ndarray or
    DataFrame) becomes an ndarray. A floating dtype is kept; anything else becomes float32, because
    upstream initialises phi_im with ``astype(IM_mat.dtype)`` and an integer dtype truncates it.
    """
    import numpy as np
    from scipy import sparse

    if sparse.issparse(features):
        feats = sparse.csr_matrix(features)
        dtype = feats.dtype if np.issubdtype(feats.dtype, np.floating) else np.float32
        feats = feats.astype(dtype, copy=True)
        feats.sum_duplicates()  # one stored value per entry, so the negative count is the matrix's
        return feats
    feats = np.asarray(features)
    dtype = feats.dtype if np.issubdtype(feats.dtype, np.floating) else np.float32
    return np.array(feats, dtype=dtype, copy=True)


def _stored_values(M):
    """The explicitly stored values of a sparse matrix, or a dense array itself."""
    import numpy as np
    from scipy import sparse

    return M.data if sparse.issparse(M) else np.asarray(M)


def _pca_image_features(adata_st, hvg_flavor: str):
    """|PCA of the (already log-normalised) expression| -- the fallback stand-in for image features.

    ``seurat_v3`` scores dispersion on raw counts, so when it is asked for it runs on the raw
    counts layer kept before normalisation; ``seurat`` runs on the log data it was designed for.
    A missing scikit-misc stops the run with the package named -- no silent flavour switch.
    """
    import numpy as np
    import scanpy as sc

    require_hvg_flavor(hvg_flavor)
    n_top = min(PCA_N_TOP_GENES, adata_st.n_vars)
    try:
        if hvg_flavor == "seurat_v3":
            sc.pp.highly_variable_genes(adata_st, n_top_genes=n_top, flavor="seurat_v3", layer="counts")
        else:
            sc.pp.highly_variable_genes(adata_st, n_top_genes=n_top, flavor=hvg_flavor)
    except Exception as e:
        # scikit-misc can be installed yet ABI-broken against the env's numpy ("numpy.dtype size
        # changed" from a user-site build); that surfaces here as a ValueError, not an ImportError.
        raise RuntimeError(
            f"highly_variable_genes(flavor='{hvg_flavor}') failed: {e}. "
            + (
                "seurat_v3 needs a working scikit-misc (import name skmisc) in the cellpie env; fix it or pass "
                "hvg_flavor='seurat'."
                if hvg_flavor == "seurat_v3"
                else "Pass a flavour scanpy accepts ('seurat' or 'seurat_v3')."
            )
        ) from e
    n_comps = min(PCA_N_COMPS, adata_st.n_vars - 1, adata_st.n_obs - 1)
    sc.pp.pca(adata_st, n_comps=n_comps)
    feats = np.abs(np.asarray(adata_st.obsm["X_pca"]))  # NMF needs non-negative
    log(f"Created |PCA| stand-in image features: shape={feats.shape}")
    return feats


def _atomic_write_csv(df, path: str) -> None:
    tmp = path + ".partial"
    df.to_csv(tmp)
    os.replace(tmp, path)


def _atomic_write_h5ad(adata, path: str) -> None:
    tmp = path + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, path)


def _no_features_message() -> str:
    return (
        "CellPie factorises expression jointly with image features, and the spatial h5ad has no "
        "obsm['features'] (spots x image-feature matrix, e.g. from CellPie's extract_features). Either add "
        "them, or choose explicitly: allow_expression_only_fallback=True runs upstream's expression-only mode "
        "(mod1_skew=2); allow_pca_image_fallback=True lets |PCA of expression| stand in for image features. "
        "Both are recorded in params.method with used_fallback=True."
    )


def run_cellpie_pipeline(
    spatial_h5ad: str,
    output_dir: str,
    n_components: int,
    sc_h5ad: str = "",
    cell_type_key: str = "cell_type",
    epochs: int = DEFAULT_EPOCHS,
    allow_expression_only_fallback: bool = False,
    allow_pca_image_fallback: bool = False,
    hvg_flavor: str = DEFAULT_HVG_FLAVOR,
    use_raw_counts: bool = False,
) -> dict[str, Any]:
    """Run CellPie intNMF on the in-tissue spots of the slide and publish spot x topic proportions."""
    import matplotlib
    import numpy as np
    import pandas as pd
    import scanpy as sc

    matplotlib.use("Agg")

    if int(n_components) < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if int(epochs) < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")

    # The reference is not an input -- it is never opened -- so a path to it is not checked either.
    preflight_check(inputs={"spatial_h5ad": spatial_h5ad}, output_dir=output_dir)

    out = WorkerOutput("cellpie", task="deconvolution")

    # The reference is never opened: intNMF is fitted on the slide alone and nothing in it shapes
    # the topics. Saying so beats loading a multi-GB h5ad to count labels the output never carries.
    if sc_h5ad:
        record_ignored(
            out,
            ["sc_h5ad_path", "cell_type_key"],
            "CellPie intNMF is a reference-free topic model; the scRNA-seq reference was not read and its labels "
            "do not shape the topic_* columns",
        )
    elif cell_type_key != "cell_type":
        record_ignored(out, "cell_type_key", "no reference is read by CellPie, so a label column has no effect")

    log(f"Loading spatial AnnData from {spatial_h5ad}")
    adata_st = sc.read_h5ad(spatial_h5ad)
    renamed_st = make_names_unique_and_report(adata_st)

    # Background glass (obs['in_tissue'] == 0) is not tissue and must not become a topic; obsm['features']
    # is subset with the spots.
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")
    if n_spots_off_tissue:
        log(f"Leaving out {n_spots_off_tissue} of {n_spots_supplied} spots with in_tissue == 0 (background)")
    # CellPie's recipe normalises X as counts: a scaled X is refused, a raw one can be asked for.
    adata_st, counts_info = choose_counts_matrix(adata_st, use_raw_counts)
    if counts_info["expression_source"] != "X":
        log(f"use_raw_counts: factorising adata.raw.X ({adata_st.n_vars} genes)")
        make_names_unique_and_report(adata_st, into=renamed_st)
    x_integer_valued = counts_info["x_matrix_kind"] in ("counts", "empty")
    if counts_info.get("warning"):
        counts_info = dict(
            counts_info,
            warning="X is not integer-valued (it may already be normalised or log-transformed); normalize_total + "
            "log1p was applied on top of it as given. " + counts_info["warning"],
        )

    n_spots = int(adata_st.n_obs)
    n_genes_st = int(adata_st.n_vars)
    log(f"Spatial data: n_spots={n_spots} in tissue (of {n_spots_supplied} supplied), n_genes={n_genes_st}")

    # ---- image modality: chosen before anything is cloned, densified or fitted --------------------
    supplied_features = None
    if "features" in adata_st.obsm:
        supplied_features = adata_st.obsm["features"]
        feats = _image_feature_copy(supplied_features)  # a copy: upstream zeroes negatives in place
        if feats.ndim != 2 or feats.shape[0] != n_spots:
            raise ValueError(
                f"obsm['features'] must be a spots x image-features matrix; got shape {feats.shape} for {n_spots} spots"
            )
        if not np.all(np.isfinite(_stored_values(feats))):
            raise ValueError("obsm['features'] contains NaN/inf; CellPie cannot factorise it")
        mode = "joint"
    elif allow_expression_only_fallback:
        mode = "expression_only"
    elif allow_pca_image_fallback:
        mode = "pca_image"
        require_hvg_flavor(hvg_flavor)
    else:
        raise ValueError(_no_features_message())
    if hvg_flavor != DEFAULT_HVG_FLAVOR and mode != "pca_image":
        record_ignored(out, "hvg_flavor", "only the PCA image-feature fallback selects highly variable genes")

    # Upstream densifies inside fit; know whether that fits before starting (or cloning anything).
    adata_st.X = _to_csr_or_dense(adata_st.X)
    _check_dense_budget(n_spots, n_genes_st, max(4, int(getattr(adata_st.X, "dtype", np.dtype("f4")).itemsize)))

    repo_path = _ensure_repo()
    # Add repo to sys.path so "from CellPie import CellPie" works
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    intNMF = _import_intnmf(repo_path)

    # CellPie's recipe normalises counts. A matrix that is already normalised or logged gets the
    # recipe on top of it; that cannot be undone here, but it is said (counts_info's warning). A matrix
    # with negative values never gets here: choose_counts_matrix refused it above.
    # Upstream CellPie's preprocess_data_visium normalises + log1p's the expression modality whatever
    # the image modality is.
    if mode == "pca_image" and hvg_flavor == "seurat_v3":
        adata_st.layers["counts"] = adata_st.X.copy()  # seurat_v3 scores dispersion on counts, not log data
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st.X = _to_csr_or_dense(adata_st.X)

    if mode == "joint":
        n_negative_features = int(np.count_nonzero(_stored_values(feats) < 0))
        if n_negative_features:
            out.add_warning(f"{n_negative_features} negative image-feature values are set to 0 by CellPie's fit")
        adata_st.obsm["features"] = feats
        mod1_skew = 1
        method, used_fallback, why = METHOD_JOINT, False, ""
        image_modality = f"obsm['features'] ({feats.shape[1]} image features supplied)"
    elif mode == "expression_only":
        # intNMF.fit reads obsm['features'] unconditionally. With mod1_skew=2 the image term has weight
        # (2 - mod1_skew) = 0 in the theta HALS update only; the constant's value still enters the
        # first-epoch theta rescale, the phi_im initialisation and the early-stopping loss, so it is a
        # fixed, published value (see EXPRESSION_ONLY_IMAGE_CONSTANT), not an inert placeholder.
        adata_st.obsm["features"] = np.full((n_spots, 1), EXPRESSION_ONLY_IMAGE_CONSTANT, dtype=np.float32)
        mod1_skew = 2
        method, used_fallback = METHOD_EXPRESSION_ONLY, True
        why = "the spatial h5ad has no obsm['features']; allow_expression_only_fallback=True"
        image_modality = (
            f"none -- expression-only (mod1_skew=2) with a constant {EXPRESSION_ONLY_IMAGE_CONSTANT} one-column "
            "image matrix; its value enters upstream's first-epoch theta rescaling, so it is fixed at "
            f"{EXPRESSION_ONLY_IMAGE_CONSTANT}"
        )
    else:
        adata_st.obsm["features"] = _pca_image_features(adata_st, hvg_flavor)
        mod1_skew = 1
        method, used_fallback = METHOD_PCA_IMAGE, True
        why = "the spatial h5ad has no obsm['features']; allow_pca_image_fallback=True"
        image_modality = (
            f"|PCA| of log-normalised expression ({adata_st.obsm['features'].shape[1]} components, "
            f"hvg_flavor={hvg_flavor}) -- expression factorised twice"
        )

    # ---- fit ----------------------------------------------------------------------------------
    log(f"Running CellPie intNMF: n_topics={n_components}, epochs={epochs}, mod1_skew={mod1_skew}")
    model = intNMF(adata_st, n_topics=int(n_components), epochs=int(epochs), mod1_skew=mod1_skew)
    model.fit(adata_st)

    # ---- theta: spots x topics ------------------------------------------------------------------
    theta = getattr(model, "theta", None)
    if theta is None:
        raise RuntimeError(
            "CellPie intNMF finished without a theta (spots x topics) matrix; there are no proportions to report."
        )
    if isinstance(theta, pd.DataFrame):
        theta = theta.values
    elif hasattr(theta, "toarray"):
        theta = theta.toarray()
    theta = np.asarray(theta, dtype=float)  # also flattens np.matrix
    if theta.ndim != 2 or theta.shape[0] != n_spots:
        raise RuntimeError(f"CellPie returned theta of shape {theta.shape}; expected ({n_spots}, {n_components})")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("CellPie returned a theta with NaN/inf entries; the factorisation did not converge")
    n_topics = int(theta.shape[1])
    row_sums = theta.sum(axis=1, keepdims=True)
    n_spots_without_signal = int((row_sums[:, 0] == 0).sum())
    row_sums[row_sums == 0] = 1
    proportions = pd.DataFrame(
        theta / row_sums,
        index=adata_st.obs_names,
        columns=[f"topic_{i}" for i in range(n_topics)],
    )

    proportions_path = os.path.join(output_dir, "cellpie_proportions.csv")
    _atomic_write_csv(proportions, proportions_path)
    log(f"Saved proportions: {proportions.shape}")

    dominant_counts = {}
    if n_topics > 1:
        dominant = proportions.idxmax(axis=1)
        dominant_counts = {str(k): int(v) for k, v in dominant.value_counts().to_dict().items()}
    if n_spots_without_signal:
        out.add_warning(f"{n_spots_without_signal} spots have an all-zero topic loading (proportions left at 0)")

    # A stand-in is not image data. Leaving it in obsm['features'] would make the next CellPie run on
    # this file read it as supplied image features and label the result the joint method.
    if mode == "joint":
        adata_st.obsm["features"] = supplied_features  # exactly what the caller supplied
    else:
        standin = np.asarray(adata_st.obsm["features"])
        del adata_st.obsm["features"]
        if mode == "pca_image":
            adata_st.obsm["cellpie_pca_image_standin"] = standin

    annotated_path = os.path.join(output_dir, "cellpie_annotated.h5ad")
    adata_st.obsm["cellpie_proportions"] = proportions.values
    adata_st.uns["cellpie_celltypes"] = list(proportions.columns)  # historical key; the values are topic names
    adata_st.uns["cellpie_topics"] = list(proportions.columns)
    adata_st.uns["cellpie_method"] = method
    adata_st.uns["cellpie_image_modality"] = image_modality
    _atomic_write_h5ad(adata_st, annotated_path)
    log(f"Saved annotated AnnData to {annotated_path}")

    # ---- payload ------------------------------------------------------------------------------
    out.set_data(n_spots=n_spots, n_spots_supplied=n_spots_supplied, n_genes_spatial=n_genes_st)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_expression_source(out, counts_info)
    out.add_output_files({"proportions_csv": proportions_path, "annotated_h5ad": annotated_path})
    out.add_params(
        {
            "n_components": int(n_components),
            "n_topics": n_topics,
            "epochs": int(epochs),
            "mod1_skew": mod1_skew,
            "image_modality": image_modality,
            "preprocessing": PREPROCESSING,
            "x_integer_valued": bool(x_integer_valued),
            "use_raw_counts": bool(use_raw_counts),
            "reference_used": False,
            "allow_expression_only_fallback": bool(allow_expression_only_fallback),
            "allow_pca_image_fallback": bool(allow_pca_image_fallback),
        }
    )
    if mode == "pca_image":
        out.add_param("hvg_flavor", hvg_flavor)
    if mode == "expression_only":
        out.add_param("expression_only_image_constant", EXPRESSION_ONLY_IMAGE_CONSTANT)
    record_method(out, method, used_fallback=used_fallback, why=why)
    out.add_params(identifier_rename_params(renamed_st))
    out.set_summary(
        n_topics=n_topics,
        n_components=int(n_components),
        dominant_counts=dominant_counts,
    )
    analysis = (
        f"{method} factorised {n_spots} spots into {n_topics} topics (topic_0..topic_{n_topics - 1}) over "
        f"{n_genes_st} genes. Topics are latent expression programmes learned from this slide alone -- they are "
        "not cell types and no scRNA-seq reference shaped them."
    )
    if dominant_counts:
        top = sorted(dominant_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
        analysis += " Most prevalent: " + ", ".join(f"{t} ({c} spots dominant)" for t, c in top) + "."
    if used_fallback:
        analysis += f" Image modality: {image_modality}."
    if n_spots_off_tissue:
        analysis += (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots supplied have obs['in_tissue'] == 0 (background) "
            "and were left out; they are not in the proportions CSV or the annotated h5ad."
        )
    if counts_info["expression_source"] != "X":
        analysis += " The expression factorised is adata.raw.X (use_raw_counts=True)."
    out.set_analysis(analysis + identifier_rename_note(renamed_st, subject="spatial data"))
    return out.to_dict()


def main() -> None:
    args = parse_args()

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    error_msg = ""
    try:
        try:
            result = run_cellpie_pipeline(
                spatial_h5ad=args.spatial_h5ad,
                output_dir=args.output_dir,
                n_components=args.n_components,
                sc_h5ad=args.sc_h5ad,
                cell_type_key=args.cell_type_key,
                epochs=args.epochs,
                allow_expression_only_fallback=args.allow_expression_only_fallback,
                allow_pca_image_fallback=args.allow_pca_image_fallback,
                hvg_flavor=args.hvg_flavor,
                use_raw_counts=args.use_raw_counts,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            result = None
            error_msg = str(e)
            error_exc = e
    finally:
        sys.stdout = orig_stdout

    if result is None:
        WorkerOutput.emit_error("cellpie", error_msg, task="deconvolution", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
