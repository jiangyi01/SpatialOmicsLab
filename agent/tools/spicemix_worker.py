#!/usr/bin/env python3
"""
SpiceMix worker: spatial factorization for spatial transcriptomics.

- Runs inside /opt/conda/envs/spicemix_env
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

SpiceMix learns K metagene factors from spatial expression data,
jointly factorizing the expression matrix while incorporating
spatial neighborhood structure.

The worker converts h5ad input into SpiceMix's expected file-based
format, runs the factorization, and exports results back to
standard formats (CSV, h5ad).

What SpiceMix is given (its README, "Preparing input files"):

- Expression is ``log(1 + 1e4 * E / sum_g E)``. An ``X`` of non-negative integer counts is
  normalised that way here (library size = the spot's total over every gene supplied); any other
  non-negative ``X`` is taken to be normalised already and passed as it is. Negative or non-finite
  values are refused: SpiceMix models non-negative expression.
- Genes detected in fewer than ``min_spots_per_gene`` spots are left out (default 1: only genes that
  are zero in every spot, which carry no signal). The count used is reported beside the count read.
- The matrix is handed over as ``expression_<r>.pkl`` (float64, which ``load_data.load_expression``
  accepts beside text), not a ``%.18e`` text dump several times its size. SpiceMix holds it densely
  as a float64 torch tensor -- that is the method -- so its size is checked against the memory
  available before it is built.
- The neighbour graph is the undirected k-nearest-neighbour graph (k = 6): every kNN pair once.
- Spots with ``obs['in_tissue'] == 0`` (the background glass a CELLxGENE Visium export carries) are
  left out before anything else and reported (``params.in_tissue_filter`` and a warning).
- Every file this worker writes -- SpiceMix's inputs and every output the payload lists -- is written
  to ``<name>.partial`` and moved into place, so a run killed mid-write never leaves a truncated file
  under a name a reader trusts.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
sys.path.insert(
    0,
    os.environ.get("SPICEMIX_SRC")
    or os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "SpiceMix", "SpiceMix"),
)

# Monkey-patch numpy to restore removed np.float and np.int aliases
# required by SpiceMix's load_data.py (uses np.float and np.int which
# were removed in numpy 2.0).
import numpy as _np

if not hasattr(_np, "float"):
    _np.float = _np.float64
if not hasattr(_np, "int"):
    _np.int = _np.int64

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    cpu_budget,
    default_output_dir,
    describe_reduction,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_in_tissue,
    record_method,
    resolve_compute,
)

#: SpiceMix's input spec (README, "Preparing input files"): E' = log(1 + 1e4 * E / sum_g E).
NORMALIZE_TARGET_SUM = 1e4
#: The transform above, spelled the way params and the analysis report it.
COUNTS_TRANSFORM = "log(1 + 1e4 * E / sum_g E)"
#: Dense float64 copies of the spots x genes expression alive at the peak of SpiceMix's
#: ``load_dataset``: the unpickled matrix, its rescaled copy (``G / GG * K * Y / Y.sum(1).mean()``)
#: while the first is still referenced, and the torch tensor built from it.
DENSE_COPIES = 3
#: Neighbours per spot in the kNN graph SpiceMix's spatial prior runs over.
N_NEIGHBORS = 6


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[spicemix-worker] {msg}", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _redirect_stdout_to_stderr():
    """Redirect stdout to stderr so library prints do not pollute JSON output."""
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


def _build_neighbor_edges(coords, n_neighbors=N_NEIGHBORS):
    """The undirected k-nearest-neighbour graph of ``coords`` as an E x 2 array of (i < j) pairs.

    Every kNN pair is kept once, whichever end listed the other. The previous loop kept a pair only
    when the lower-numbered spot listed the higher one, so a neighbour relation that is not mutual
    was dropped whenever it ran the other way (150 of 1,061 pairs on 300 uniformly scattered spots).
    Self is dropped by identity, not by position: with duplicated coordinates a twin can precede a
    spot in its own neighbour list.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 2 or n_neighbors < 1:
        return np.zeros((0, 2), dtype=int)
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n), metric="euclidean")
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    rows = np.repeat(np.arange(n), indices.shape[1])
    cols = indices.ravel()
    keep = rows != cols
    if not keep.any():
        return np.zeros((0, 2), dtype=int)
    pairs = np.stack([np.minimum(rows[keep], cols[keep]), np.maximum(rows[keep], cols[keep])], axis=1)
    return np.unique(pairs, axis=0).astype(int)


def _memory_available_bytes():
    """What this process can still allocate: ``worker_utils.available_memory_bytes``, or None.

    The smaller of MemAvailable and the room left under the cgroup memory limit (page cache counted
    as reclaimable). Reading /proc/meminfo alone saw the host's memory from inside a memory-limited
    container, so the refusal below passed and the run was OOM-killed with no payload.
    """
    return available_memory_bytes()


def _check_dense_budget(n_spots, n_genes, available=None):
    """Refuse, with the numbers, an expression matrix SpiceMix cannot hold in memory.

    SpiceMix's model is over the dense spots x genes matrix (``torch.tensor(Y, float64)`` in
    ``load_dataset``); that is the method, not this wrapper. Every in-tissue spot is kept: the knob that
    shrinks the matrix is the gene axis, ``min_spots_per_gene``. ``available`` defaults to
    :func:`_memory_available_bytes` (MemAvailable, or the room under the cgroup limit if smaller).
    """
    if available is None:
        available = _memory_available_bytes()
    need = int(n_spots) * int(n_genes) * 8 * DENSE_COPIES
    if available is not None and need > available:
        gib = 1024.0**3
        raise MemoryError(
            f"SpiceMix holds the expression of every spot as a dense float64 matrix: {n_spots} spots x "
            f"{n_genes} genes is about {need / gib:.1f} GiB at the peak of loading ({DENSE_COPIES} copies), and "
            f"{available / gib:.1f} GiB is available here (MemAvailable, or the room left under the cgroup memory "
            "limit). Raise min_spots_per_gene (genes detected in fewer spots are left out, which shrinks the gene "
            "axis of that matrix) or run where more memory is available; every in-tissue spot is factorised."
        )
    return need


def _is_integer_valued(values, chunk=10_000_000):
    """True when every value is a whole number. Read in chunks so the check never copies it all."""
    import numpy as np

    flat = np.asarray(values).ravel()
    for start in range(0, flat.size, chunk):
        block = flat[start : start + chunk]
        if block.size and not np.all(np.equal(np.mod(block, 1), 0)):
            return False
    return True


def _spicemix_expression(X, min_spots_per_gene=1, available=None):
    """Build the spots x genes matrix SpiceMix is given, and say what was done to it.

    Returns ``(Y, keep, info)``: ``Y`` the dense float64 matrix over the kept genes, ``keep`` the
    boolean gene mask over the genes supplied, ``info`` what params/data/analysis report.
    """
    import numpy as np
    import scipy.sparse as sp

    if min_spots_per_gene < 0:
        raise ValueError(f"min_spots_per_gene must be 0 or more, got {min_spots_per_gene}.")

    sparse = sp.issparse(X)
    if sparse:
        X = sp.csr_matrix(X)
        stored = X.data
    else:
        X = np.asarray(X)
        stored = X
    n_spots, n_genes = X.shape
    if stored.size:
        if not np.all(np.isfinite(stored)):
            raise ValueError(
                "adata.X holds NaN or infinite values; SpiceMix cannot factorise them. Supply finite counts or "
                "log-normalised expression."
            )
        low = float(stored.min())
        if low < 0:
            raise ValueError(
                f"adata.X holds negative values (minimum {low:g}). SpiceMix models non-negative expression "
                f"(its README asks for {COUNTS_TRANSFORM}); scaled or centred data cannot be factorised. Supply raw "
                "counts, which are normalised here, or non-negative log-normalised expression."
            )
    counts = _is_integer_valued(stored)

    detected = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = detected >= int(min_spots_per_gene)
    n_used = int(keep.sum())
    if n_used < 2:
        # SpiceMix factorises a spots x genes matrix; one gene is not a matrix it can factorise (and its
        # own genes-file reader returns a 0-d array for one name, whose len() then raises).
        raise ValueError(
            f"{n_used} gene(s) are detected in at least min_spots_per_gene={min_spots_per_gene} of the {n_spots} "
            f"spots ({n_genes} genes supplied, the most widely detected in "
            f"{int(detected.max()) if detected.size else 0} spots); SpiceMix needs at least 2. Lower "
            "min_spots_per_gene."
        )
    _check_dense_budget(n_spots, n_used, available=available)

    if sparse:
        library = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        Y = X[:, np.flatnonzero(keep)].toarray().astype(np.float64, copy=False)
    else:
        library = X.sum(axis=1, dtype=np.float64).ravel() if X.size else np.zeros(n_spots)
        # X[:, keep] is already a copy (boolean indexing), so Y never aliases adata.X, which the
        # in-place normalisation below would otherwise rewrite before the annotated h5ad is saved.
        Y = np.asarray(X[:, keep], dtype=np.float64)

    # A spot whose row sums to zero (over every gene supplied) reaches SpiceMix as an all-zero row.
    n_zero_spots = int(np.sum(library <= 0))
    if counts:
        scale = np.zeros_like(library)
        np.divide(NORMALIZE_TARGET_SUM, library, out=scale, where=library > 0)
        Y *= scale[:, None]
        np.log1p(Y, out=Y)
        transform = COUNTS_TRANSFORM
    else:
        transform = (
            "none (adata.X is not integer counts, so it was taken to be normalised already and used as supplied)"
        )

    info = {
        "expression_input": "counts" if counts else "as_supplied",
        "expression_transform": transform,
        "n_genes_input": int(n_genes),
        "n_genes_used": n_used,
        "min_spots_per_gene": int(min_spots_per_gene),
        "n_spots_zero_counts": n_zero_spots,
    }
    return Y, keep, info


def _gpu_was_requested(request) -> bool:
    """Whether ``--device`` asked for a GPU, in any spelling ``resolve_compute`` reads as one."""
    text = str(request).strip().lower()
    return text in ("gpu", "cuda", "true", "yes", "on") or text.startswith(("cuda", "gpu")) or text.isdigit()


def _write_atomic(path, write) -> None:
    """``write(<path>.partial)``, then ``os.replace`` onto ``path``; a failed write leaves no partial file.

    Nothing is ever written in place: a run killed mid-write would leave a truncated table, h5ad or
    image under the name the payload lists, and a later reader would trust it.
    """
    path = str(path)
    tmp = path + ".partial"
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _write_pickle_atomic(obj, path) -> None:
    """``<path>.partial`` then ``os.replace``: SpiceMix never finds a half-written matrix."""

    def _dump(tmp):
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=4)

    _write_atomic(path, _dump)


def _write_text_atomic(path, lines) -> None:
    def _write(tmp):
        with open(tmp, "w") as fh:
            for line in lines:
                fh.write(line)
                fh.write("\n")

    _write_atomic(path, _write)


def _prepare_spicemix_dataset(adata, dataset_dir, repli_name="0", min_spots_per_gene=1, available=None):
    """Convert an AnnData object into SpiceMix's expected file layout.

    SpiceMix expects:
      dataset_dir/files/expression_{repli}.pkl  (N x G float64 matrix; load_expression reads .pkl or .txt)
      dataset_dir/files/neighborhood_{repli}.txt  (E x 2 edge list)
      dataset_dir/files/genes_{repli}.txt  (gene names)
      dataset_dir/files/meta_{repli}.csv  (metadata)

    Returns what ``_spicemix_expression`` reports, plus ``keep`` (the gene mask) and ``genes``.
    """
    import numpy as np
    import pandas as pd

    # Spot coordinates first: a file without them is refused before any expression work is done.
    if "spatial" in adata.obsm:
        coords = np.array(adata.obsm["spatial"])
    elif "X_spatial" in adata.obsm:
        coords = np.array(adata.obsm["X_spatial"])
    else:
        # Alone of the four tools in this family, SpiceMix already reads the X_spatial spelling a
        # Seurat conversion writes, so only a file with neither key reaches here. It used to get
        # column_stack([arange(n), zeros(n)]) -- every spot on the x axis -- which is then fed
        # straight to _build_neighbor_edges and optimised with use_spatial=[True], making the
        # spatial neighbours barcode neighbours. Both spellings are named, since either satisfies
        # the branches above.
        raise ValueError(
            f"Spot coordinates not found: neither adata.obsm['spatial'] nor adata.obsm['X_spatial'] "
            f"is present. Available obsm keys: {list(adata.obsm.keys())}"
        )

    files_dir = dataset_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Expression matrix (dense, N x G_used): SpiceMix's own model is dense, so this one is intrinsic.
    Y, keep, info = _spicemix_expression(adata.X, min_spots_per_gene=min_spots_per_gene, available=available)
    expr_path = files_dir / f"expression_{repli_name}.pkl"
    _write_pickle_atomic(Y, expr_path)
    log(f"Saved expression ({Y.shape}, {info['expression_transform']}) to {expr_path}")
    shape = Y.shape
    del Y

    # Gene names. SpiceMix reads this file with np.loadtxt(dtype=str), which splits on whitespace: one
    # name containing a space ("HLA A" style) makes a two-column row and load_dataset dies with "the
    # number of columns changed". SpiceMix only counts the names it reads, so the file carries each name
    # with whitespace runs joined by "_"; the real names label every output (metagenes CSV, top genes).
    genes_path = files_dir / f"genes_{repli_name}.txt"
    gene_names = [str(g) for g, k in zip(adata.var_names, keep) if k]
    _write_text_atomic(genes_path, ["_".join(g.split()) or "_" for g in gene_names])
    log(f"Saved {len(gene_names)} gene names to {genes_path}")

    # Spatial neighbors
    n_neighbors = min(N_NEIGHBORS, adata.n_obs - 1)
    edges = _build_neighbor_edges(coords, n_neighbors=n_neighbors)
    neighbor_path = files_dir / f"neighborhood_{repli_name}.txt"
    _write_atomic(neighbor_path, lambda tmp: np.savetxt(tmp, edges, fmt="%d", delimiter="\t"))
    log(f"Saved {len(edges)} edges to {neighbor_path}")

    # Metadata
    meta_df = pd.DataFrame({"cell type": ["unknown"] * adata.n_obs})
    if "cell_type" in adata.obs.columns:
        meta_df["cell type"] = adata.obs["cell_type"].values
    elif "CellType" in adata.obs.columns:
        meta_df["cell type"] = adata.obs["CellType"].values
    elif "cluster" in adata.obs.columns:
        meta_df["cell type"] = adata.obs["cluster"].values
    meta_path = files_dir / f"meta_{repli_name}.csv"
    _write_atomic(meta_path, lambda tmp: meta_df.to_csv(tmp, index=False))
    log(f"Saved metadata to {meta_path}")

    info = dict(info)
    info.update({"shape": shape, "keep": keep, "genes": gene_names, "n_edges": int(len(edges)), "coords": coords})
    return info


def _run_spicemix(
    spatial_h5ad_path: str,
    output_dir: str,
    K: int,
    n_epochs: int,
    device: str,
    min_spots_per_gene: int = 1,
) -> WorkerOutput:
    """Run SpiceMix factorization and return WorkerOutput (caller emits)."""
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load h5ad ───────────────────────────────────────────────────────
    log(f"Loading AnnData from {spatial_h5ad_path}")
    adata = sc.read_h5ad(spatial_h5ad_path)
    renamed = make_names_unique_and_report(adata)
    # Background spots (obs['in_tissue'] == 0) are not tissue: left out here and reported below.
    adata, n_spots_supplied, n_off_tissue = keep_in_tissue(adata)
    n_spots = adata.n_obs
    n_genes = adata.n_vars
    log(
        f"Loaded: n_obs={n_spots}, n_vars={n_genes}"
        + (f" ({n_off_tissue} of {n_spots_supplied} spots off the tissue left out)" if n_off_tissue else "")
    )

    # ── Prepare SpiceMix dataset ────────────────────────────────────────
    dataset_dir = out_dir / "spicemix_dataset"
    repli_name = "0"
    prep = _prepare_spicemix_dataset(adata, dataset_dir, repli_name=repli_name, min_spots_per_gene=min_spots_per_gene)
    keep = prep["keep"]
    genes_used = prep["genes"]
    n_genes_used = int(prep["n_genes_used"])
    log(f"Prepared dataset with shape {prep['shape']} ({n_genes_used} of {n_genes} genes)")

    # ── Configure and run SpiceMix ──────────────────────────────────────
    from model import SpiceMix as SpiceMixModel

    result_path = out_dir / "spicemix_result.h5"
    # SpiceMix appends to its HDF5 result file on every saved iteration, so it writes to
    # <name>.partial and the finished file is moved into place after the last iteration: a run killed
    # mid-optimisation never leaves a half-written result under the name the payload lists. A stale
    # .partial from a killed run makes SpiceMix's writer fail with "Unable to create link (name
    # already exists)", so it is removed first; a finished result from an earlier run stays until
    # this run's replaces it.
    result_partial = Path(str(result_path) + ".partial")
    if result_partial.exists():
        try:
            result_partial.unlink()
        except OSError as e:
            log(f"Warning: could not remove stale {result_partial}: {e}")
    # SpiceMix passes context["device"] straight to torch, so an unrecognised spelling ('GPU', the
    # word this fleet's own portals document) raises inside the model, and 'cuda' on a box with no
    # driver dies there too. resolve_compute normalises every spelling the agent may produce and
    # never returns a CUDA device unless CUDA is really present.
    device_requested = device
    compute = resolve_compute(device)
    device = compute.device
    context = {"device": device, "dtype": torch.float64}

    log(f"Initializing SpiceMix (K={K}, device={device})")
    np.random.seed(0)
    # Was a hardcoded 4, which oversubscribes a 2-CPU container by 2x. cpu_budget() reads the
    # allowance actually in force (affinity mask + cgroup quota) rather than the machine. Capped at
    # the previous literal on purpose: this can only ever lower the thread count, never raise it, so
    # a box that was already sized correctly behaves exactly as before.
    torch.set_num_threads(cpu_budget(cap=4))

    obj = SpiceMixModel(
        K=K,
        lambda_Sigma_x_inv=1e-6,
        power_Sigma_x_inv=2,
        repli_list=[repli_name],
        betas=np.ones(1),
        context=context,
        context_Y=context,
        path2result=result_partial,
    )

    log("Loading dataset into SpiceMix...")
    obj.load_dataset(dataset_dir)

    # Use kmeans initialization (louvain requires a separate 'louvain' package
    # that may not be installed; kmeans is always available via scikit-learn).
    log("Initializing (kmeans + NMF warmup)...")
    obj.initialize(method="kmeans", random_state=0)

    # NMF warmup (no spatial)
    init_nmf_iter = min(10, n_epochs // 2) if n_epochs > 5 else 2
    log(f"Running {init_nmf_iter} NMF warmup iterations...")
    for i in range(init_nmf_iter):
        iiter_tag = 0 if i == init_nmf_iter - 1 else -1
        obj.estimate_weights(iiter=iiter_tag, use_spatial=[False])
        obj.estimate_parameters(iiter=iiter_tag, use_spatial=[False])
    obj.initialize_Sigma_x_inv()

    # Main optimization with spatial
    main_iter = max(1, n_epochs - init_nmf_iter)
    log(f"Running {main_iter} spatial iterations...")
    for iiter in range(1, main_iter + 1):
        if iiter % 10 == 0 or iiter == main_iter:
            log(f"  Iteration {iiter}/{main_iter}")
        obj.estimate_parameters(iiter=iiter, use_spatial=[True])
        obj.estimate_weights(iiter=iiter, use_spatial=[True])

    log("SpiceMix optimization complete.")
    if result_partial.exists():
        os.replace(str(result_partial), str(result_path))

    # ── Extract and save results ────────────────────────────────────────

    # Factor loadings (X): N x K per replicate
    X_factors = obj.Xs[0].cpu().numpy()  # shape (N, K)
    log(f"Factor loadings shape: {X_factors.shape}")

    # Metagene dictionary (M): G x K
    M_metagenes = obj.M.cpu().numpy()  # shape (G, K)
    log(f"Metagene dictionary shape: {M_metagenes.shape}")

    # Save factor loadings as CSV
    loadings_df = pd.DataFrame(
        X_factors,
        index=adata.obs_names,
        columns=[f"factor_{k}" for k in range(K)],
    )
    loadings_path = out_dir / "spicemix_factor_loadings.csv"
    _write_atomic(loadings_path, loadings_df.to_csv)
    log(f"Saved factor loadings to {loadings_path}")

    # Save metagene dictionary as CSV: one row per gene SpiceMix was given, in the order it was given them.
    if M_metagenes.shape[0] != len(genes_used):
        raise RuntimeError(
            f"SpiceMix returned a metagene dictionary over {M_metagenes.shape[0]} genes for the {len(genes_used)} "
            "genes it was given; the rows cannot be named."
        )
    metagenes_df = pd.DataFrame(
        M_metagenes,
        index=genes_used,
        columns=[f"factor_{k}" for k in range(K)],
    )
    metagenes_path = out_dir / "spicemix_metagenes.csv"
    _write_atomic(metagenes_path, metagenes_df.to_csv)
    log(f"Saved metagenes to {metagenes_path}")

    # Store in adata and save h5ad
    adata.obsm["X_spicemix"] = X_factors
    # Which genes the factorisation ran on (the rest were detected in fewer than min_spots_per_gene spots).
    adata.var["spicemix_used"] = np.asarray(keep, dtype=bool)
    # Assign dominant factor as a cluster label
    dominant_factor = np.argmax(X_factors, axis=1)
    adata.obs["spicemix_factor"] = pd.Categorical([f"factor_{f}" for f in dominant_factor])
    h5ad_path = out_dir / "spicemix_annotated.h5ad"
    _write_atomic(h5ad_path, adata.write_h5ad)
    log(f"Saved annotated h5ad to {h5ad_path}")

    # ── Top genes per factor ────────────────────────────────────────────
    top_genes_per_factor = {}
    n_top = min(10, n_genes_used)
    for k in range(K):
        top_idx = np.argsort(M_metagenes[:, k])[::-1][:n_top]
        top_genes_per_factor[f"factor_{k}"] = [str(genes_used[i]) for i in top_idx]

    # ── Factor assignment statistics ────────────────────────────────────
    factor_counts = {}
    for f in range(K):
        factor_counts[f"factor_{f}"] = int(np.sum(dominant_factor == f))

    # ── Save factor visualization ───────────────────────────────────────
    plot_path = out_dir / "spicemix_factors.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # The coordinates the neighbour graph was built from -- obsm['spatial'] or obsm['X_spatial'].
        coords = np.asarray(prep["coords"], dtype=float)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            n_show = min(K, 6)
            fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4))
            if n_show == 1:
                axes = [axes]
            for k in range(n_show):
                ax = axes[k]
                sc_plot = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=X_factors[:, k],
                    cmap="viridis",
                    s=2,
                    alpha=0.8,
                )
                ax.set_title(f"Factor {k}")
                ax.set_aspect("equal")
                ax.axis("off")
                plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            # format= is required: the .partial suffix names no image type.
            _write_atomic(plot_path, lambda tmp: plt.savefig(tmp, format="png", dpi=150, bbox_inches="tight"))
            plt.close(fig)
            log(f"Saved factor plot to {plot_path}")
        else:
            log("No spatial coordinates for plotting")
            plot_path = None
    except Exception as e:
        log(f"Warning: could not generate factor plot: {e}")
        plot_path = None

    # ── Emit output ─────────────────────────────────────────────────────
    out = WorkerOutput("spicemix", task="factorization")
    # n_spots is the in-tissue spots factorised; the background cut is in params.in_tissue_filter.
    # n_genes is the count the factorisation ran on; the count read is a different quantity.
    out.set_data(
        n_spots=n_spots,
        n_genes=n_genes_used,
        n_genes_input=n_genes,
        n_spots_zero_counts=int(prep["n_spots_zero_counts"]),
        n_neighbor_edges=int(prep["n_edges"]),
    )
    out.add_output_file("factor_loadings_csv", str(loadings_path))
    out.add_output_file("metagenes_csv", str(metagenes_path))
    out.add_output_file("annotated_h5ad", str(h5ad_path))
    out.add_output_file("result_h5", str(result_path))
    if plot_path:
        out.add_output_file("factors_plot", str(plot_path))
    out.add_output_file("output_dir", str(out_dir))

    out.add_params(
        {
            "K": K,
            "n_epochs": n_epochs,
            "device": device,
            "init_nmf_iter": init_nmf_iter,
            "main_iter": main_iter,
            "lambda_Sigma_x_inv": 1e-6,
            "init": "kmeans",
            "n_neighbors": N_NEIGHBORS,
            "min_spots_per_gene": int(prep["min_spots_per_gene"]),
            "expression_input": prep["expression_input"],
            "expression_transform": prep["expression_transform"],
        }
    )
    record_method(out, "SpiceMix (kmeans initialisation, NMF warm-up, then spatial iterations)")
    record_in_tissue(out, n_spots_supplied, n_off_tissue)
    out.add_params(identifier_rename_params(renamed))
    if device == "cpu" and _gpu_was_requested(device_requested):
        # resolve_compute says this on stderr only, which a successful run never returns.
        out.add_params({"device_requested": str(device_requested)})
        out.add_warning(
            f"device={device_requested!r} was requested, but no CUDA device is available: SpiceMix ran on CPU."
        )
    if prep["n_spots_zero_counts"]:
        out.add_warning(
            f"{prep['n_spots_zero_counts']} of {n_spots} spots have no counts in any gene supplied; their expression "
            "row is all zero, so their factor loadings come from the spatial prior (their neighbours) alone."
        )
    if prep["expression_input"] != "counts":
        out.add_warning(
            "adata.X is not integer counts, so it was used as supplied (taken to be normalised already); SpiceMix "
            f"expects {COUNTS_TRANSFORM}. Supply raw counts to have it normalised that way."
        )

    out.set_summary(
        n_factors=K,
        factor_assignment_counts=factor_counts,
        top_genes_per_factor=top_genes_per_factor,
        final_Q=float(obj.Q) if obj.Q is not None else None,
    )

    # Build analysis text
    n_nonempty = sum(1 for c in factor_counts.values() if c > 0)
    top_factor = max(factor_counts, key=factor_counts.get)
    top_factor_n = factor_counts[top_factor]
    top_factor_pct = round(top_factor_n / n_spots * 100, 1) if n_spots > 0 else 0

    if prep["expression_input"] == "counts":
        expression_sentence = f"The counts in X were normalised as SpiceMix specifies, {COUNTS_TRANSFORM}."
    else:
        expression_sentence = (
            "X is not integer counts, so it was used as supplied (taken to be normalised already); "
            f"SpiceMix expects {COUNTS_TRANSFORM}."
        )
    analysis_lines = [
        f"SpiceMix learned {K} spatial metagene factors from {n_spots} spots x {n_genes_used} genes "
        f"({init_nmf_iter} NMF warm-up + {main_iter} spatial iterations for n_epochs={n_epochs}).",
        expression_sentence,
        f"{n_nonempty}/{K} factors have at least one dominant spot.",
        f"Most prevalent: {top_factor} ({top_factor_n} spots, {top_factor_pct}%).",
    ]
    if obj.Q is not None:
        analysis_lines.append(f"Final objective (Q): {float(obj.Q):.2f}.")

    # Only a threshold of 0 or 1 drops nothing but all-zero genes; above that a dropped gene may be expressed.
    m = int(prep["min_spots_per_gene"])
    if m <= 1:
        cut = f"min_spots_per_gene={m} (genes zero in every spot, which carry no signal)"
    else:
        cut = f"min_spots_per_gene={m} (genes detected in fewer than {m} spots)"
    reduction = describe_reduction("genes", n_genes, n_genes_used, reason=cut)
    out.set_analysis(" ".join(analysis_lines) + reduction + identifier_rename_note(renamed))
    return out


def _cli_main() -> None:
    parser = argparse.ArgumentParser(description="SpiceMix spatial factorization worker")
    parser.add_argument("--spatial-h5ad", required=True, help="Path to spatial AnnData h5ad")
    parser.add_argument("--output-dir", default=default_output_dir(), help="Output directory")
    parser.add_argument("--K", type=int, default=10, help="Number of factors (metagenes)")
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of optimization epochs")
    parser.add_argument("--device", default="cpu", help="Compute device (cpu or cuda)")
    parser.add_argument(
        "--min-spots-per-gene",
        type=int,
        default=1,
        help="Leave out genes detected in fewer spots (default 1: only genes zero in every spot); 0 keeps all",
    )

    args = parser.parse_args()

    # Preflight checks
    try:
        preflight_check(
            inputs={"spatial_h5ad": args.spatial_h5ad},
            output_dir=args.output_dir,
            packages=["torch", "scanpy"],
        )
    except (FileNotFoundError, PermissionError, ImportError) as e:
        WorkerOutput.emit_error("spicemix", str(e), task="factorization")
        sys.exit(1)

    run_error = None
    with _redirect_stdout_to_stderr():
        try:
            worker_out = _run_spicemix(
                spatial_h5ad_path=args.spatial_h5ad,
                output_dir=args.output_dir,
                K=args.K,
                n_epochs=args.n_epochs,
                device=args.device,
                min_spots_per_gene=args.min_spots_per_gene,
            )
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback

            traceback.print_exc(file=sys.stderr)
            run_error = e

    # Emit JSON to real stdout (after redirect context is closed)
    if run_error is not None:
        WorkerOutput.emit_error("spicemix", str(run_error), task="factorization", exc=run_error)
        sys.exit(1)
    worker_out.emit()


if __name__ == "__main__":
    _cli_main()
