#!/usr/bin/env python3
"""
scDOT-based cell-spot mapping worker: entropic optimal transport on a cosine cost.

Runs inside /opt/conda/envs/scdot conda env.
Code at /opt/conda/envs/scdot/scDOT/src/

WHAT RUNS HERE, HONESTLY. Upstream scDOT (gersteinlab/scDOT, ``src/net.py``) is two modules trained
jointly: an NNLS deconvolution of each spot against per-cell-type marker profiles
(``st.uns['markers']``, ``nls_projgrad``) and a differentiable optimal-transport layer whose cost
matrix ``M`` is an ``nn.Parameter`` tuned by Adam so the OT plan agrees with the NNLS proportions.
This worker runs ONLY the optimal-transport half, once, on the fixed cosine cost: no marker set is
built, no NNLS is solved, and nothing is trained. That is plain entropy-regularised OT (Sinkhorn)
solved by scDOT's ``OptimalTransportLayer``; the payload names it as such in ``params.method`` and
the analysis text, and the coupling equals ``ddn.sinkhorn(cost, gamma)`` column-normalised. It is
not a fallback -- nothing else ever ran here -- so ``params.used_fallback`` is False.

MEMORY. The transport plan is a dense n_spots x n_cells matrix. That is intrinsic to the method
(Sinkhorn scales the full plan every iteration), so it is kept; what is NOT intrinsic is anything
larger than it. The cost is computed sparse-aware (L2-normalised rows, one sparse-dense product,
``1 - sim`` in place), the one-hot cell-type matrix is n_cells x n_types, and the plan is
normalised in place, so the peak is 12 bytes per spot-cell pair. Before anything that size exists
the worker compares that figure with the memory it can still allocate (``worker_utils.
available_memory_bytes``: ``MemAvailable``, and the room under a cgroup memory limit with the
cgroup's page cache counted as reclaimable; free GPU memory too when the solve runs on CUDA) and
refuses with the numbers rather than allocating. The worker never subsamples.

BACKGROUND SPOTS. A spatial file whose ``obs['in_tissue']`` marks background spots (0) -- CELLxGENE
Visium exports carry every array spot -- is mapped on its in-tissue spots only
(``worker_utils.keep_in_tissue``); ``params.in_tissue_filter`` and a warning say how many were left
out, and every output row is an in-tissue spot.

Input: spatial h5ad + scRNA h5ad. Spatial coordinates are NOT read -- the mapping compares
expression profiles only -- so ``--spatial-key`` is accepted for compatibility and reported under
``params.ignored``.
Output: cell-spot coupling matrix (spots x cells), per-spot cell-type proportions (coupling @ one-hot,
row-normalised), and the spatial h5ad with ``obsm['deconvolution']``.

All logs go to stderr; stdout is JSON-only (final result).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add scDOT source to path
# <TOOL>_SRC seam (as spatialscope_worker.py does): this checkout exists only where it was
# installed, and a sys.path entry that does not exist fails silently -- the run dies later in an
# ImportError naming an upstream module, with no way to redirect it. The literal stays the default.
sys.path.insert(0, os.environ.get("SCDOT_SRC") or "/opt/conda/envs/scdot/scDOT/src")

from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    build_deconv_analysis,
    describe_reduction,
    id_mismatch_msg,
    keep_in_tissue,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    unsupported_choice_msg,
)
from worker_utils import (
    drop_unlabeled as split_unlabeled,
)

#: What this worker runs. Not "scDOT": upstream scDOT is NNLS + a trained OT cost (see the module
#: docstring); this is one Sinkhorn solve on a fixed cosine cost through scDOT's OT layer.
METHOD_NAME = (
    "entropic optimal transport (Sinkhorn, scDOT's OptimalTransportLayer) on a fixed cosine spot-cell "
    "cost; scDOT's NNLS marker deconvolution and joint training are not run"
)
#: The short form the analysis sentence leads with.
METHOD_SHORT = "Entropic-OT cell-spot mapping (scDOT OT layer only, no NNLS or joint training)"

#: HVG flavours the single-cell reference can be reduced with. ``seurat_v3`` ranks raw counts and
#: needs scikit-misc; the other two rank the log-normalised matrix.
HVG_FLAVORS = ("seurat_v3", "seurat", "cell_ranger")

#: Bytes per (spot, cell) pair of the dense state the worker holds at its peak. Two moments reach it
#: and they never overlap: building the cost (float64 cosine distances + their float32 copy, 8 + 4)
#: and starting the Sinkhorn solve (float32 cost + the two float32 temporaries of
#: ``torch.exp(-gamma * (M - min))``, 4 + 4 + 4). The float32 plan is then normalised in place.
DENSE_BYTES_PER_ENTRY = 12


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def _ensure_dir(p: str) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _celltype_onehot(labels, index) -> pd.DataFrame:
    """One-hot cell-type matrix: one row per cell (in ``index`` order), one column per label.

    Replaces a networkx cell-to-type graph whose ``adjacency_matrix(...).todense()`` was an
    ``(n_cells + n_types)^2`` int64 array -- 43 GB for a 73k-cell reference, 625 GB for a 280k-cell
    one -- built to read back an ``n_cells x n_types`` block of it. That path also relied on
    ``reset_index`` naming its column ``"index"`` (a KeyError when ``obs.index.name`` is set) and
    merged a cell with a type when a barcode equalled a label. Same rows, same values, O(n_cells *
    n_types) memory.

    Columns keep the old order -- ``np.unique`` of the labels as given, so integer labels sort 1, 2,
    10 rather than "1", "10", "2" -- and are named by ``str(label)`` so the frame can be written to
    h5ad. Labels numpy cannot order among themselves (mixed types) fall back to sorting by text.
    """
    raw = np.asarray(labels, dtype=object)
    try:
        order = list(np.unique(np.asarray(labels)))
    except TypeError:
        order = sorted(set(raw.tolist()), key=str)
    names = list(dict.fromkeys(str(v) for v in order))
    position = {v: i for i, v in enumerate(names)}
    text = [str(v) for v in raw.tolist()]
    onehot = np.zeros((len(text), len(names)), dtype=np.float64)
    if text:
        onehot[np.arange(len(text)), [position[t] for t in text]] = 1.0
    return pd.DataFrame(onehot, index=pd.Index(index), columns=names)


def _dense_plan_bytes(n_spots: int, n_cells: int) -> int:
    """Bytes of the dense n_spots x n_cells state the worker holds at its peak.

    Twelve bytes per (spot, cell) pair, at two moments that are each that size and never overlap:
    building the cost (the float64 cosine distances plus their float32 copy) and starting the
    Sinkhorn solve (the float32 cost plus the two float32 temporaries of ``exp(-gamma * (M - min))``).
    The transport plan itself is float32 and is normalised in place.
    """
    return int(n_spots) * int(n_cells) * DENSE_BYTES_PER_ENTRY


def _memory_budget_bytes():
    """Memory this run can still allocate, in bytes, or None when nothing can be read.

    ``worker_utils.available_memory_bytes`` -- the one reader every worker's preflight uses. This
    worker once carried its own probe that took ``memory.max - memory.current`` (cgroup v2) or
    ``limit_in_bytes - usage_in_bytes`` (v1) as the headroom. Both usage figures count the page cache,
    and a memory-limited container sits near its limit on cache alone after reading two h5ads, so
    that probe refused transport plans that fit; the shared reader counts the cache as reclaimable.
    """
    return available_memory_bytes()


def _gpu_free_bytes(device: str):
    """Free memory on a CUDA device, or None when it is not a CUDA device or torch cannot say."""
    if not str(device).startswith("cuda"):
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return int(free)
    except Exception:
        return None


def _check_dense_plan_fits(n_spots: int, n_cells: int, device: str = "cpu") -> int:
    """Refuse, with the numbers, before allocating a transport plan the machine cannot hold.

    The plan is intrinsic to optimal transport, so there is no smaller setting of this tool that
    avoids it and the worker never subsamples the inputs to make it fit. On a CUDA device the solve
    also needs the same ~12 bytes per pair of GPU memory; there ``device='cpu'`` is the knob.
    """
    need = _dense_plan_bytes(n_spots, n_cells)
    budget = _memory_budget_bytes()
    if budget is not None and need > budget:
        raise MemoryError(
            f"the optimal-transport plan is a dense {n_spots} spots x {n_cells} reference cells matrix and "
            f"this run needs ~{need / 1e9:.1f} GB for it at peak (12 bytes per spot-cell pair: float64 cost "
            f"+ float32 copy, then float32 cost + two Sinkhorn temporaries), but only ~{budget / 1e9:.1f} GB "
            "is available. The dense plan is intrinsic to the method and this tool does not subsample; run it "
            f"where at least {need / 1e9:.1f} GB is free."
        )
    gpu_free = _gpu_free_bytes(device)
    if gpu_free is not None and need > gpu_free:
        raise MemoryError(
            f"the Sinkhorn solve on {device} needs ~{need / 1e9:.1f} GB of GPU memory for a dense {n_spots} x "
            f"{n_cells} plan (float32 cost + two temporaries) but {device} has ~{gpu_free / 1e9:.1f} GB free. "
            "Pass device='cpu' to solve it in host memory instead; the data is never subsampled."
        )
    return need


def _cosine_cost(st_X, sc_X, st_names=None, sc_names=None) -> np.ndarray:
    """C-contiguous float32 n_spots x n_cells cosine-distance matrix, built without densifying the inputs.

    Numerically the ``ot.dist(st_X, sc_X, metric="cosine")`` result upstream uses (``1 - <u, v> /
    (|u| |v|)``, computed in float64 and handed to the OT layer as float32), but from L2-normalised
    rows and one sparse-dense product. Peak is 12 bytes per spot-cell pair (the float64 distances and
    their float32 copy), the figure ``_dense_plan_bytes`` budgets; the old path held a sparse product,
    its dense copy and ``1 - sim`` at once. The only operand densified is the side with fewer rows,
    and only when both sides are sparse.

    A row with no counts on the analysed panel has no cosine distance (0/0): cdist returns NaN there
    and Sinkhorn then turns the WHOLE plan into NaN -- every spot, every cell -- which the old worker
    published with status ok. That case is refused by name instead.
    """
    import scipy.sparse as sps

    def _l2_rows(X, what, names):
        if sps.issparse(X):
            X = sps.csr_matrix(X, dtype=np.float64)
            norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        else:
            X = np.asarray(X, dtype=np.float64)
            norms = np.sqrt(np.einsum("ij,ij->i", X, X))
        zero = norms == 0
        if zero.any():
            shown = ""
            if names is not None:
                empty = [str(n) for n in np.asarray(names)[zero][:5]]
                shown = f" (e.g. {', '.join(empty)})"
            raise ValueError(
                f"{int(zero.sum())} of {len(norms)} {what}{shown} have no counts on the {X.shape[1]}-gene analysed "
                "panel, so their cosine distance is undefined (0/0) and the whole transport plan would be NaN. "
                "Remove observations with no counts before calling, or raise n_hvg so the shared panel covers them."
            )
        if sps.issparse(X):
            return sps.csr_matrix(sps.diags(1.0 / norms) @ X)
        return X / norms[:, None]

    st_n = _l2_rows(st_X, "spots", st_names)
    sc_n = _l2_rows(sc_X, "reference cells", sc_names)
    if sps.issparse(st_n) and sps.issparse(sc_n):
        # sparse @ sparse would build a near-dense sparse product (12+ bytes per pair) before its dense
        # copy; densify the smaller side instead so the product is written dense once.
        if st_n.shape[0] <= sc_n.shape[0]:
            st_n = st_n.toarray()
        else:
            sc_n = sc_n.toarray()
    if sps.issparse(st_n):
        sim = st_n @ sc_n.T  # sparse @ dense -> dense, spots x cells
    elif sps.issparse(sc_n):
        sim = (sc_n @ st_n.T).T  # sparse @ dense -> dense cells x spots, viewed as spots x cells
    else:
        sim = st_n @ sc_n.T
    sim = np.asarray(sim)
    del st_n, sc_n
    # 1 - sim in place: the float64 buffer becomes the cost, no second n_spots x n_cells array.
    np.subtract(1.0, sim, out=sim)
    # Rounding can push an exact match to -1e-16; cdist clips the same way.
    np.clip(sim, 0.0, 2.0, out=sim)
    return np.ascontiguousarray(sim, dtype=np.float32)


def _h5ad_safe_names(names):
    """(names safe as HDF5 dataset names, {original: rewritten} for the ones that changed).

    ``/`` is HDF5's group separator; everything else a label may hold is a legal dataset name. A
    rewrite that lands on another label's spelling gets a numeric suffix, so no two columns merge.
    """
    safe, renames = [], {}
    taken = {str(n) for n in names if "/" not in str(n)}
    for name in names:
        original = str(name)
        if "/" not in original:
            safe.append(original)
            continue
        base = original.replace("/", "_")
        new, k = base, 1
        while new in taken:
            k += 1
            new = f"{base}_{k}"
        taken.add(new)
        safe.append(new)
        renames[original] = new
    return safe, renames


# The transport plan is float32. Run to exhaustion at ot_eps=0 through scDOT's own OptimalTransportLayer
# (random costs, 14x36 up to 2812x3000 and 500x20000), its float64 marginal error stopped at 0.8-1.9 x
# float32 eps x the larger marginal; 4 leaves headroom for the block summation of bigger plans.
FLOAT32_FLOOR_ULPS = 4.0


def _float32_marginal_floor(n_spots: int, n_cells: int) -> float:
    """The marginal error below which a float32 plan cannot be driven, however many iterations run."""
    return FLOAT32_FLOOR_ULPS * float(np.finfo(np.float32).eps) * max(1.0 / n_spots, 1.0 / n_cells)


def _not_converged_warning(
    marginal_error: float, ot_eps: float, ot_maxiters: int, float32_floor: float, n_spots: int, n_cells: int
) -> str:
    """Say why the plan missed ot_eps, and which knob moves it.

    Two different situations report ``sinkhorn_converged`` False. Either the error is still above what
    float32 resolves (Sinkhorn needed more iterations), or it is already at that resolution and
    ot_eps asks for less than a float32 plan can hold -- where more iterations change nothing.
    """
    larger = f"1/{min(n_spots, n_cells)}"
    if marginal_error <= float32_floor:
        return (
            f"Sinkhorn's plan has a marginal error of {marginal_error:.3g}, which is as close as a float32 plan "
            f"gets to marginals of {larger} (about {float32_floor:.2g}), but ot_eps={ot_eps} asks for less, so "
            f"the plan is reported not converged. Raising ot_maxiters (now {ot_maxiters}) will not change "
            f"this; set ot_eps to {float32_floor:.2g} or more to have it judged against what float32 can reach."
        )
    message = (
        f"Sinkhorn stopped at ot_maxiters={ot_maxiters} with a marginal error of {marginal_error:.3g} "
        f"(ot_eps={ot_eps}): the plan is not converged. Raise ot_maxiters."
    )
    if float(ot_eps) < float32_floor:
        message += (
            f" Note that ot_eps is below what a float32 plan resolves at marginals of {larger} (about "
            f"{float32_floor:.2g}), so even then the plan will stop short of ot_eps; set ot_eps to "
            f"{float32_floor:.2g} or more."
        )
    return message


def _write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


def _write_h5ad_atomic(adata, path: Path) -> None:
    """Write the AnnData as ``<path>.partial`` and rename over ``path`` once it is complete."""
    tmp = str(path) + ".partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


def run_scdot(
    sc_h5ad: str,
    st_h5ad: str,
    output_dir: str,
    annotation_key: str = "cell_type",
    spatial_key: str = "spatial",
    n_hvg: int = 2000,
    ot_gamma: float = 1.0,
    ot_maxiters: int = 1000,
    ot_eps: float = 1e-6,
    device: str = "cpu",
    random_seed: int = 0,
    hvg_flavor: str = "seurat_v3",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """Map single cells to spots by entropic OT on a cosine cost (scDOT's OT layer only).

    See the module docstring for what upstream scDOT does that this does not. ``spatial_key`` is
    accepted and never read (the method uses expression only). ``hvg_flavor`` picks how the
    reference is reduced to ``n_hvg`` genes; a flavour whose dependency is missing stops the run with
    the package name, it is never swapped for another flavour. ``drop_unlabeled`` leaves out
    reference cells whose label is NaN/empty instead of refusing them.
    """

    import scanpy as sc
    import torch

    _ensure_dir(output_dir)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    outdir = Path(output_dir)

    if hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", hvg_flavor, HVG_FLAVORS))
    # Name the missing package now, before two h5ads are loaded and preprocessed for nothing.
    try:
        require_hvg_flavor(hvg_flavor)
    except ImportError:
        raise
    except Exception as exc:
        # Installed but unimportable -- the ~/.local scikit-misc built against another numpy raises
        # ValueError here, which require_hvg_flavor does not translate. Same remedy, same knob.
        raise ImportError(
            f"hvg_flavor={hvg_flavor!r} needs scikit-misc, which is installed but cannot be imported in this "
            f"environment ({type(exc).__name__}: {exc}). Repair it, or pass hvg_flavor='seurat' explicitly."
        ) from exc

    # Import scDOT's OT layer. scDOT/src/ddn.py exports it at the top level; the git ddn package the
    # recipe installs keeps the byte-identical class in ddn.pytorch.optimal_transport, so a recipe-built
    # env without the src tree failed here (hunt 2026-09-30, u38c-specs-b-10).
    try:
        from ddn import OptimalTransportLayer
    except ImportError:
        from ddn.pytorch.optimal_transport import OptimalTransportLayer

    # ---- Load data ----
    eprint(f"[scDOT] Loading scRNA-seq data: {sc_h5ad}")
    adata_sc = sc.read_h5ad(sc_h5ad)
    adata_sc.var_names_make_unique()
    eprint(f"[scDOT] Loaded scRNA: {adata_sc.n_obs} cells x {adata_sc.n_vars} genes")
    # What the user handed us on each modality, captured before the preprocessing below cuts both
    # gene axes down to their intersection. Both panels have to survive into the payload, or a
    # caller reads the shared survivor count as the panel they supplied.
    n_genes_sc_supplied = int(adata_sc.n_vars)
    n_cells_sc_supplied = int(adata_sc.n_obs)

    eprint(f"[scDOT] Loading spatial data: {st_h5ad}")
    adata_st = sc.read_h5ad(st_h5ad)
    adata_st.var_names_make_unique()
    eprint(f"[scDOT] Loaded spatial: {adata_st.n_obs} spots x {adata_st.n_vars} genes")
    n_genes_st_supplied = int(adata_st.n_vars)
    # Background glass (obs['in_tissue'] == 0) is not tissue: it is left out before anything is
    # filtered or mapped, and the count travels in the payload (params.in_tissue_filter + a warning).
    adata_st, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata_st, "spots")
    if n_spots_off_tissue:
        eprint(
            f"[scDOT] Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 "
            "(background); mapping the in-tissue spots only"
        )

    # ---- Validate annotation key ----
    if annotation_key not in adata_sc.obs:
        raise ValueError(
            f"annotation_key='{annotation_key}' not found in scRNA obs. Available keys: {list(adata_sc.obs.columns)}"
        )
    # A missing label is not a cell type. Refuse by default; drop and report when allowed.
    keep_labelled, n_unlabeled = split_unlabeled(
        adata_sc.obs[annotation_key].to_numpy(), drop_unlabeled, what="reference cells"
    )
    if n_unlabeled:
        adata_sc = adata_sc[keep_labelled].copy()
        eprint(f"[scDOT] Dropped {n_unlabeled} reference cells with no '{annotation_key}' label (drop_unlabeled)")

    # ---- Preprocessing ----
    eprint("[scDOT] Preprocessing data ...")

    # Normalize scRNA
    sc.pp.filter_genes(adata_sc, min_counts=10)
    non_mito = [g for g in adata_sc.var_names if not g.startswith("MT-")]
    adata_sc = adata_sc[:, non_mito].copy()
    adata_sc.layers["counts"] = adata_sc.X.copy()
    sc.pp.normalize_total(adata_sc, target_sum=1e5)
    sc.pp.log1p(adata_sc)
    # No ``adata_sc.raw = adata_sc``: upstream's tutorial line kept a second full copy of the
    # log-normalised reference that nothing here reads (the reference is never written out).

    n_top = min(n_hvg, adata_sc.n_vars)
    eprint(f"[scDOT] Selecting {n_top} highly variable genes on the reference (flavor={hvg_flavor}) ...")
    # Narrow guard around the one call whose failure used to be swallowed: scanpy raises inside
    # ``flavor="seurat_v3"`` when scikit-misc is absent or broken, and this worker used to answer by
    # running ``flavor="seurat"`` on log data and saying nothing. Now the caller's flavour is the only
    # one that runs; if it cannot, the error names the package (or the real cause) and the knob.
    try:
        if hvg_flavor == "seurat_v3":
            sc.pp.highly_variable_genes(
                adata_sc, n_top_genes=n_top, subset=True, layer="counts", flavor=hvg_flavor, span=1
            )
        else:
            sc.pp.highly_variable_genes(adata_sc, n_top_genes=n_top, subset=True, flavor=hvg_flavor)
    except Exception as exc:
        require_hvg_flavor(hvg_flavor)  # raises ImportError naming scikit-misc when that is the cause
        raise RuntimeError(
            f"highly_variable_genes(flavor='{hvg_flavor}') failed on the single-cell reference and no other "
            f"flavour was substituted: {exc}. Pass hvg_flavor='seurat' to rank log-normalised dispersion instead."
        ) from exc

    # Normalize spatial
    sc.pp.filter_genes(adata_st, min_cells=3)
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    # Upstream's spatial-side call: no subset=, so it only annotates var['highly_variable'] (kept in
    # the annotated h5ad) and selects nothing. The spatial panel is cut by filter_genes above only.
    sc.pp.highly_variable_genes(adata_st, n_top_genes=min(n_hvg, adata_st.n_vars))

    # Intersect genes
    intersect = np.intersect1d(adata_sc.var_names, adata_st.var_names)
    eprint(f"[scDOT] Intersecting genes: {len(intersect)}")
    if len(intersect) == 0:
        raise ValueError(id_mismatch_msg("genes", "scRNA", adata_sc.var_names, "spatial", adata_st.var_names))

    adata_sc = adata_sc[:, intersect].copy()
    adata_st = adata_st[:, intersect].copy()

    # ---- Build cell type matrix ----
    eprint(f"[scDOT] Building cell type matrix from '{annotation_key}' ...")
    ct = _celltype_onehot(adata_sc.obs[annotation_key].to_numpy(), adata_sc.obs_names)
    adata_sc.obsm["cell_type"] = ct

    # --device was resolved, logged and recorded in the output params, but the one tensor in this
    # worker never left the CPU, so asking for a GPU did nothing. scDOT's OptimalTransportLayer
    # takes no device of its own -- it allocates alongside whatever it is handed -- so placing the
    # cost matrix is what places the solve.
    resolved_device = resolve_compute(device).device
    eprint(f"[scDOT] Using device: {resolved_device}")

    # ---- Memory estimate (before anything n_spots x n_cells exists), then the OT cost matrix ----
    n_spots, n_cells = int(adata_st.n_obs), int(adata_sc.n_obs)
    dense_bytes = _check_dense_plan_fits(n_spots, n_cells, resolved_device)
    eprint(f"[scDOT] Dense OT state: {n_spots} spots x {n_cells} cells, ~{dense_bytes / 1e9:.2f} GB at peak")

    eprint("[scDOT] Computing OT cost matrix (cosine distance, sparse-aware) ...")
    M = _cosine_cost(adata_st.X, adata_sc.X, adata_st.obs_names, adata_sc.obs_names)  # float32, C order
    M_tensor = torch.from_numpy(M).unsqueeze(0).to(resolved_device)
    del M  # on the CPU the tensor shares this buffer; on a GPU the host copy is no longer needed

    # ---- Run optimal transport ----
    eprint(f"[scDOT] Running entropic OT / Sinkhorn (gamma={ot_gamma}, maxiters={ot_maxiters}) ...")
    ot_layer = OptimalTransportLayer(
        gamma=ot_gamma,
        eps=ot_eps,
        maxiters=ot_maxiters,
        method="approx",
    )

    with torch.no_grad():
        coupling_tensor = ot_layer(M_tensor)
    # The cost has done its job; free it before the plan is normalised and written.
    del M_tensor

    # .cpu() before .numpy(): a no-op on the CPU path, and required on the one this fix opens up.
    coupling = coupling_tensor.squeeze(0).cpu().numpy()
    del coupling_tensor
    n_nonfinite = int(coupling.size - np.count_nonzero(np.isfinite(coupling)))
    if n_nonfinite:
        # A cell whose kernel column exp(-ot_gamma * cost) underflows to zero makes Sinkhorn divide
        # 0/0, and the NaN then spreads to every entry. Refuse rather than publish NaN proportions.
        raise ValueError(
            f"the Sinkhorn transport plan has {n_nonfinite} non-finite entries of {coupling.size} "
            f"(ot_gamma={ot_gamma}): exp(-ot_gamma * cost) underflowed for some cells. Lower ot_gamma "
            "(upstream's default is 1.0)."
        )
    # How far the plan is from its marginals (each spot 1/n_spots, each cell 1/n_cells) when the solve
    # stopped, summed in float64. Converged means at or below ot_eps itself -- the absolute tolerance
    # upstream's Sinkhorn loop stops on -- with no floor: a fixed 1e-6 floor once passed ot_eps=1e-12
    # as met, and at 73k cells (marginal 1.4e-5) it would pass a 7% miss. The plan is float32, so no
    # number of iterations drives the error below float32_floor; the warning tells that case apart.
    marginal_error = max(
        float(np.abs(coupling.sum(axis=1, dtype=np.float64) - 1.0 / n_spots).max()),
        float(np.abs(coupling.sum(axis=0, dtype=np.float64) - 1.0 / n_cells).max()),
    )
    converged = marginal_error <= float(ot_eps)
    float32_floor = _float32_marginal_floor(n_spots, n_cells)
    # Normalize: columns (cells) sum to 1 -- in place, so no second n_spots x n_cells array.
    col_sums = coupling.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    coupling /= col_sums
    eprint(f"[scDOT] OT coupling shape: {coupling.shape} (spots x cells)")

    # ---- Deconvolution ----
    eprint("[scDOT] Computing deconvolution (cell type proportions per spot) ...")
    ct_np = ct.to_numpy()
    P = coupling @ ct_np
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P = P / row_sums

    deconv_df = pd.DataFrame(P, columns=ct.columns, index=adata_st.obs_names)
    # HDF5 reads "/" as a group separator, so a reference label such as "Treg/Tfr" beside "Treg" made
    # write_h5ad fail AFTER both CSVs were written, and the run ended in an error. The CSV and the
    # payload keep the reference's own spelling; only the h5ad copy is rewritten, and says so.
    h5ad_names, h5ad_renames = _h5ad_safe_names(deconv_df.columns)
    adata_st.obsm["deconvolution"] = deconv_df.set_axis(h5ad_names, axis=1)
    adata_st.uns["deconvolution_cell_types"] = [str(c) for c in deconv_df.columns]

    # ---- Save outputs (each written beside its final name and renamed into place) ----
    # Coupling matrix
    coupling_csv = outdir / "scdot_coupling.csv"
    _write_csv_atomic(
        pd.DataFrame(
            coupling,
            index=adata_st.obs_names,
            columns=adata_sc.obs_names,
        ),
        coupling_csv,
    )
    eprint(f"[scDOT] Saved coupling matrix to {coupling_csv}")

    # Deconvolution proportions
    deconv_csv = outdir / "scdot_deconvolution.csv"
    _write_csv_atomic(deconv_df, deconv_csv)
    eprint(f"[scDOT] Saved deconvolution to {deconv_csv}")

    # Annotated spatial h5ad
    spatial_h5ad_out = outdir / "scdot_spatial_annotated.h5ad"
    _write_h5ad_atomic(adata_st, spatial_h5ad_out)
    eprint(f"[scDOT] Saved annotated spatial h5ad to {spatial_h5ad_out}")

    # ---- Compute summary stats ----
    celltypes = list(deconv_df.columns)
    n_celltypes = len(celltypes)
    dominant = deconv_df.idxmax(axis=1)
    dominant_counts = dominant.value_counts().to_dict()

    # ---- Build output ----
    # Both modalities were cut to the SAME surviving panel (the intersection taken after each side's
    # own HVG/detection filter), so n_genes_used and n_genes_sc_used are equal by construction --
    # that is why publishing only the survivor count made one number stand in for two panels.
    n_genes_analysed = int(len(intersect))
    st_note = describe_reduction(
        "spatial genes",
        n_genes_st_supplied,
        n_genes_analysed,
        "the worker's own preprocessing, which drops genes detected in fewer than 3 spots and then keeps "
        "only the genes the single-cell reference also retained",
    )
    sc_note = describe_reduction(
        "single-cell genes",
        n_genes_sc_supplied,
        n_genes_analysed,
        f"the worker's own preprocessing, which drops genes with fewer than 10 counts and genes named "
        f"'MT-*' (upstream's human mitochondrial prefix), keeps the {n_hvg} most highly variable "
        f"(flavor={hvg_flavor}), and then keeps only the genes the spatial slide also retained",
    )
    cells_note = describe_reduction(
        "reference cells",
        n_cells_sc_supplied,
        n_cells,
        f"drop_unlabeled=True, which leaves out cells whose '{annotation_key}' label is missing",
    )

    out = WorkerOutput("scdot", task="cell_spot_mapping")
    out.set_data(
        n_spots=n_spots,
        n_spots_supplied=int(n_spots_supplied),
        n_genes=n_genes_st_supplied,
        n_genes_used=n_genes_analysed,
        n_cells_sc=n_cells_sc_supplied,
        n_cells_sc_used=n_cells,
        n_genes_sc=n_genes_sc_supplied,
        n_genes_sc_used=n_genes_analysed,
    )
    out.add_output_files(
        {
            "coupling_csv": str(coupling_csv),
            "deconvolution_csv": str(deconv_csv),
            "spatial_annotated_h5ad": str(spatial_h5ad_out),
        }
    )
    out.add_params(
        {
            "annotation_key": annotation_key,
            "n_hvg": n_hvg,
            "hvg_flavor": hvg_flavor,
            "drop_unlabeled": bool(drop_unlabeled),
            "n_cells_sc_unlabeled_dropped": int(n_unlabeled),
            "ot_gamma": ot_gamma,
            "ot_maxiters": ot_maxiters,
            "ot_eps": ot_eps,
            # The device the solve ran on, not the one asked for: a GPU request on a CPU-only box
            # degrades to 'cpu' inside resolve_compute, and the payload has to say which happened.
            "device": resolved_device,
            "device_requested": device,
            "random_seed": random_seed,
            "dense_plan_bytes_estimated": int(dense_bytes),
            "sinkhorn_marginal_error": marginal_error,
            "sinkhorn_converged": bool(converged),
            "n_cell_types_renamed_in_h5ad": len(h5ad_renames),
        }
    )
    if h5ad_renames:
        first, rewritten = next(iter(h5ad_renames.items()))
        out.add_params({"cell_type_renames_in_h5ad": dict(h5ad_renames)})
        out.add_warning(
            f"{len(h5ad_renames)} cell-type name(s) contain '/', which HDF5 reads as a group separator, so "
            f"obsm['deconvolution'] in scdot_spatial_annotated.h5ad spells them with '_' ({first!r} is "
            f"{rewritten!r} there; uns['deconvolution_cell_types'] keeps the originals). "
            "scdot_deconvolution.csv and this payload keep the reference's spelling."
        )
    if not converged:
        out.add_warning(_not_converged_warning(marginal_error, ot_eps, ot_maxiters, float32_floor, n_spots, n_cells))
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
    record_ignored(
        out,
        "spatial_key",
        "the optimal-transport mapping compares expression profiles only and never reads spatial "
        "coordinates, so this key has no effect on the result",
    )
    out.set_summary(
        n_celltypes=n_celltypes,
        celltypes=celltypes,
        dominant_counts=dominant_counts,
        coupling_shape=list(coupling.shape),
    )
    # A successful run never carries stderr into the payload (base_mcp attaches stderr_tail only on a
    # non-zero exit), so a cut this large has to travel in the payload itself.
    out.add_warnings([n.strip() for n in (st_note, sc_note, cells_note) if n])
    out.set_analysis(
        build_deconv_analysis(
            n_celltypes,
            dominant_counts,
            total_spots=n_spots,
            method_name=METHOD_SHORT,
        )
        + " Proportions are the column-normalised OT plan summed per cell type; scDOT's NNLS step and "
        "training were not run."
        + (
            f" {n_spots_off_tissue} of {n_spots_supplied} spots were background (obs['in_tissue'] == 0) and "
            f"were left out; the {n_spots} in-tissue spots were mapped."
            if n_spots_off_tissue
            else ""
        )
        + st_note
        + sc_note
        + cells_note
    )

    return out.to_dict()


def main():
    ap = argparse.ArgumentParser(
        description="Cell-spot mapping by entropic optimal transport on a cosine cost (scDOT's OT layer only)"
    )
    ap.add_argument("--sc-h5ad", required=True, help="Path to scRNA-seq AnnData (.h5ad)")
    ap.add_argument("--st-h5ad", required=True, help="Path to spatial AnnData (.h5ad)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--annotation-key", default="cell_type", help="obs column for cell type labels")
    ap.add_argument(
        "--spatial-key",
        default="spatial",
        help="Accepted for compatibility and reported under params.ignored: the mapping never reads coordinates",
    )
    ap.add_argument("--n-hvg", type=int, default=2000, help="Number of highly variable genes")
    ap.add_argument(
        "--hvg-flavor",
        default="seurat_v3",
        help=f"scanpy HVG flavour for the reference, one of {', '.join(HVG_FLAVORS)}; seurat_v3 needs "
        "scikit-misc and is never swapped for another flavour",
    )
    ap.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Drop reference cells whose label is NaN/empty instead of refusing them",
    )
    ap.add_argument("--ot-gamma", type=float, default=1.0, help="OT regularization parameter")
    ap.add_argument("--ot-maxiters", type=int, default=1000, help="Max iterations for Sinkhorn")
    ap.add_argument("--ot-eps", type=float, default=1e-6, help="Convergence tolerance for Sinkhorn")
    ap.add_argument("--device", default="cpu", help="Device: 'cpu', 'gpu'/'cuda', 'cuda:N', or 'auto'")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    try:
        result = run_scdot(
            sc_h5ad=args.sc_h5ad,
            st_h5ad=args.st_h5ad,
            output_dir=args.output_dir,
            annotation_key=args.annotation_key,
            spatial_key=args.spatial_key,
            n_hvg=args.n_hvg,
            ot_gamma=args.ot_gamma,
            ot_maxiters=args.ot_maxiters,
            ot_eps=args.ot_eps,
            device=args.device,
            random_seed=args.seed,
            hvg_flavor=args.hvg_flavor,
            drop_unlabeled=args.drop_unlabeled,
        )
        print(json.dumps(result, default=str))

    except Exception as e:
        eprint(f"[scDOT] ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("scdot", str(e), task="cell_spot_mapping")
        sys.exit(1)


if __name__ == "__main__":
    main()
