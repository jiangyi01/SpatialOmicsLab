#!/usr/bin/env python3
"""
GPSA worker: spatial alignment of two tissue slices with Gaussian Process Spatial Alignment
(upstream ``gpsa.VariationalGPSA``, slice 1 held fixed as the template).

- Runs inside /opt/conda/envs/gpsa
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

What a run does, in order:

* leaves out each slice's background spots (``obs['in_tissue'] == 0``, which CELLxGENE Visium
  exports carry) as it is read (``worker_utils.keep_in_tissue``), and counts them in
  ``params.in_tissue_filter`` / ``params.in_tissue_dropped_per_slice``: glass spots would enter the
  joint HVG selection, the expression GP and the warp as though they were tissue. The outputs hold
  the in-tissue spots only;
* reads the expression of both slices from ``X`` or, with ``--layer-key``, from
  ``layers[layer_key]`` (a layer missing from either slice is an error, never a silent ``X``), or
  with ``--use-raw-counts`` from ``adata.raw`` (``worker_utils.choose_counts_matrix``). The run
  normalises that matrix as counts (``normalize_total`` + ``log1p``), so negative or non-finite
  values are refused and non-integer values run with a warning;
* keeps the genes the two slices share, in slice 1's order (a set intersection's order follows
  string hashing, which Python randomises per process, so ``random_seed`` did not reproduce a run);
* picks ``n_top_genes`` highly variable genes jointly over both slices with ``--hvg-flavor``:
  ``seurat_v3`` ranks the raw counts it was designed for, ``seurat``/``cell_ranger`` rank the
  log-normalised matrix. A flavour that cannot run stops the run; no other flavour is substituted;
* rescales both slices' coordinates jointly into [0, 1] -- the frame GPSA is trained in -- trains
  full-batch (every spot, every step: upstream GPSA has no mini-batch path, so ``--batch-size`` is
  accepted and reported as ignored), then maps the aligned coordinates back into the units of
  ``obsm[spatial_key]`` before writing them. The [0, 1] answer is kept beside it under
  ``obsm['spatial_aligned_normalized']``.

The expression GPSA models is a dense ``spots x HVGs`` float32 matrix; that is intrinsic to the
method, so the run estimates it first and refuses, with the numbers, when it cannot fit.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    expression_matrix_kind,
    id_mismatch_msg,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    spatial_coords,
    unsupported_choice_msg,
)

#: What runs. Upstream ``gpsa.VariationalGPSA`` with slice 1 as the fixed view, trained full-batch.
METHOD_NAME = "GPSA (gpsa.VariationalGPSA, slice 1 fixed as the template, full-batch training)"

#: HVG flavours the joint gene selection accepts. ``seurat_v3`` ranks raw counts and needs
#: scikit-misc; the other two rank the log-normalised matrix.
HVG_FLAVORS = ("seurat_v3", "seurat", "cell_ranger")
DEFAULT_HVG_FLAVOR = "seurat_v3"

#: Upper bound on the inducing points of each GP. Both sets live in the spatial domain -- the warp
#: GP's per slice, the expression GP's in the common coordinate system -- so both are bounded by the
#: spot counts, never by the gene count.
MAX_INDUCING = 50

#: What ``obsm['spatial_aligned']`` and the aligned-coordinate CSV are expressed in.
COORDINATE_FRAME = "input"

#: float32 copies of the ``spots x HVGs`` expression held at the peak of a training step: the numpy
#: matrix, its torch copy, the modelled mean ``F`` ([S=1, n, p]), the temporaries of
#: ``torch.distributions.Normal.log_prob`` over it, and their gradients in the backward pass.
DENSE_COPIES = 8


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[gpsa-worker] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- pure helpers


def _shared_genes(names1, names2):
    """The genes both slices carry, in slice 1's order.

    ``list(set(a) & set(b))`` returned them in string-hash order, which Python randomises per
    process: the columns of the expression matrix -- and so which random initial weight each gene
    received -- changed from run to run under the same ``random_seed``.
    """
    in2 = {str(g) for g in names2}
    return [str(g) for g in names1 if str(g) in in2]


def _joint_unit_box(X1, X2):
    """``(x_min, x_range)`` of both slices together: the frame GPSA is trained in."""
    import numpy as np

    X_all = np.vstack([np.asarray(X1, dtype=np.float64), np.asarray(X2, dtype=np.float64)])
    x_min = X_all.min(axis=0)
    x_range = X_all.max(axis=0) - x_min
    x_range[x_range == 0] = 1.0  # a flat axis stays where it is instead of dividing by zero
    return x_min, x_range


def _to_unit_box(X, x_min, x_range):
    import numpy as np

    return (np.asarray(X, dtype=np.float64) - x_min) / x_range


def _from_unit_box(X_norm, x_min, x_range):
    """Invert :func:`_to_unit_box`: coordinates back in the units of ``obsm[spatial_key]``."""
    import numpy as np

    return np.asarray(X_norm, dtype=np.float64) * x_range + x_min


def _expression_source(adata, layer_key, which):
    """The matrix a slice's expression is read from: ``X``, or ``layers[layer_key]`` when named."""
    if not layer_key:
        return adata.X
    layers = getattr(adata, "layers", None) or {}
    if layer_key not in layers:
        raise ValueError(
            f"layer_key={layer_key!r} is not a layer of {which}. Available layers: "
            f"{sorted(str(k) for k in layers.keys())}. Pass layer_key='' to read adata.X."
        )
    return layers[layer_key]


def _looks_like_counts(X, chunk: int = 10_000_000) -> bool:
    """True when every stored value is a non-negative integer. Read in chunks, never densified."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(X):
        data = np.asarray(X.data).ravel()
        for start in range(0, data.size, chunk):
            block = data[start : start + chunk]
            if block.size and (block.min() < 0 or not np.all(np.equal(np.mod(block, 1), 0))):
                return False
        return True
    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return True
    rows = max(1, chunk // max(1, arr.shape[1]))
    for start in range(0, arr.shape[0], rows):
        block = np.asarray(arr[start : start + rows], dtype=np.float64)
        if block.size and (block.min() < 0 or not np.all(np.equal(np.mod(block, 1), 0))):
            return False
    return True


def _memory_available_bytes():
    """Memory this process can still allocate, or None when the platform cannot say.

    The fleet's one reader (``worker_utils.available_memory_bytes``): the smaller of MemAvailable and
    the room under the cgroup limit, page cache counted as reclaimable. MemAvailable alone is the whole
    host's figure, so in a memory-limited container the dense-budget check passed and training was
    OOM-killed with no message. Kept under this name so :func:`_check_dense_budget` reads it here.
    """
    return available_memory_bytes()


def _counts_source(adata, which, layer_key, use_raw_counts):
    """``(adata, info)``: the matrix this slice's expression is read from, checked as counts.

    GPSA's input is ``log1p(normalize_total(.))`` of that matrix, so it must hold counts. X (or
    ``adata.raw`` with ``use_raw_counts``) goes through the shared rule,
    ``worker_utils.choose_counts_matrix``; a named layer is held to the same rule by hand, since the
    shared helper reads X. ``info`` is the helper's shape, plus ``matrix_kind`` for the matrix used.
    """
    if layer_key:
        layer = _expression_source(adata, layer_key, which)
        kind = expression_matrix_kind(layer)
        if kind in ("negative", "nonfinite"):
            what = "negative values (scaled or z-scored data)" if kind == "negative" else "NaN or infinite values"
            raise ValueError(
                f"{which}: layers['{layer_key}'] holds {what}, not counts, and GPSA normalises its expression as "
                "counts (normalize_total, log1p). Name a layer of raw counts, or pass layer_key='' to read X."
            )
        warning = None
        if kind == "nonnegative_noninteger":
            warning = (
                f"layers['{layer_key}'] holds non-integer values (normalised or log-transformed data?), and GPSA "
                "normalises its expression as counts, so the result was computed on a matrix normalised twice."
            )
        info = {
            "expression_source": f"layers['{layer_key}']",
            "x_matrix_kind": expression_matrix_kind(adata.X),
            "warning": warning,
            "matrix_kind": kind,
        }
        return adata, info
    try:
        adata, info = choose_counts_matrix(adata, use_raw_counts)
    except ValueError as exc:
        raise ValueError(f"{which}: {exc}") from exc
    info = dict(info)
    info["matrix_kind"] = expression_matrix_kind(adata.X) if use_raw_counts else info["x_matrix_kind"]
    return adata, info


def _record_counts_source(out, infos):
    """``params.expression_source`` / ``x_matrix_kind`` for the two slices, and their warnings once."""
    kinds = [str(info["x_matrix_kind"]) for info in infos]
    record_expression_source(
        out,
        {
            "expression_source": infos[0]["expression_source"],
            "x_matrix_kind": kinds[0] if len(set(kinds)) == 1 else "mixed",
            "warning": None,
        },
    )
    out.add_params(
        {
            "x_matrix_kind_per_slice": kinds,
            "expression_matrix_kind_per_slice": [str(info["matrix_kind"]) for info in infos],
        }
    )
    warned = [(i + 1, info["warning"]) for i, info in enumerate(infos) if info.get("warning")]
    if warned and len({w for _, w in warned}) == 1:
        out.add_warning(f"slice(s) {', '.join(str(i) for i, _ in warned)}: {warned[0][1]}")
    else:
        for i, warning in warned:
            out.add_warning(f"slice {i}: {warning}")


def _dense_bytes(n_spots, n_genes):
    """Bytes of the dense expression state GPSA holds at its peak for ``n_spots x n_genes``."""
    return int(n_spots) * int(n_genes) * 4 * DENSE_COPIES


def _check_dense_budget(n_spots, n_genes, available=None):
    """Refuse up front, with the numbers, an expression matrix that cannot be trained in memory.

    GPSA's likelihood is over the dense ``spots x HVGs`` matrix; that is the method, not this
    wrapper. Without the check the run dies inside torch with a bare allocation error, or is
    OOM-killed mid-training with no message at all. Every spot is kept: the knob that shrinks the
    matrix is the gene axis, ``n_top_genes``.
    """
    if available is None:
        available = _memory_available_bytes()
    need = _dense_bytes(n_spots, n_genes)
    if available is not None and need > available:
        gib = 1024.0**3
        raise RuntimeError(
            f"GPSA models the expression of every spot as one dense matrix: {n_spots} spots x {n_genes} genes "
            f"is about {need / gib:.1f} GiB at the peak of a training step, and this machine reports "
            f"{available / gib:.1f} GiB available. Lower n_top_genes (the gene axis of that matrix) or run on "
            "a machine with more memory; every spot of both slices is aligned."
        )
    return need


def _n_inducing(n1, n2):
    """Inducing points for both GPs. The expression GP's are drawn from slice 2's spots without
    replacement upstream, so they can never outnumber them; they were sized by the gene count."""
    return min(MAX_INDUCING, int(n1), int(n2))


def _write_csv_atomic(df, path) -> None:
    """``<path>.partial`` then ``os.replace``: a reader never sees a half-written table."""
    tmp = f"{path}.partial"
    df.to_csv(tmp)
    os.replace(tmp, str(path))


def _write_h5ad_atomic(adata, path) -> None:
    tmp = f"{path}.partial"
    adata.write_h5ad(tmp)
    os.replace(tmp, str(path))


def _validate(args) -> None:
    """Refuse values the method cannot run with before two slices are read."""
    for name, value in (
        ("n_spatial_dims", args.n_spatial_dims),
        ("n_latent_gps", args.n_latent_gps),
        ("num_epochs", args.num_epochs),
        ("n_top_genes", args.n_top_genes),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1 (got {value}).")
    if args.hvg_flavor not in HVG_FLAVORS:
        raise ValueError(unsupported_choice_msg("hvg_flavor", args.hvg_flavor, HVG_FLAVORS))
    if getattr(args, "use_raw_counts", False) and args.layer_key:
        raise ValueError(
            f"use_raw_counts=True reads adata.raw and layer_key={args.layer_key!r} names a layer: the "
            "expression can come from one of them. Pass layer_key='' to use adata.raw, or leave "
            "use_raw_counts off to use the layer."
        )


# --------------------------------------------------------------------------- the run


def run_alignment(args):
    """Run GPSA alignment on two spatial slices."""
    _validate(args)
    hvg_flavor = args.hvg_flavor
    use_raw_counts = bool(getattr(args, "use_raw_counts", False))
    # Name a missing scikit-misc now, before two slices are loaded and preprocessed for nothing.
    require_hvg_flavor(hvg_flavor)

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from gpsa import VariationalGPSA

    log("Starting GPSA spatial alignment")
    log(f"  slice1_h5ad     = {args.slice1_h5ad}")
    log(f"  slice2_h5ad     = {args.slice2_h5ad}")
    log(f"  output_dir      = {args.output_dir}")
    log(f"  n_spatial_dims  = {args.n_spatial_dims}")
    log(f"  n_latent_gps    = {args.n_latent_gps}")
    log(f"  num_epochs      = {args.num_epochs}")
    log(f"  learning_rate   = {args.learning_rate}")
    log(f"  batch_size      = {args.batch_size} (ignored: GPSA trains full-batch)")
    log(f"  n_top_genes     = {args.n_top_genes}")
    log(f"  hvg_flavor      = {hvg_flavor}")
    log(f"  layer_key       = {args.layer_key or '(X)'}")
    log(f"  use_raw_counts  = {use_raw_counts}")
    log(f"  random_seed     = {args.random_seed}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Set seeds
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Load slices. Background spots go first: everything after this -- the counts check, the HVGs,
    # the GP, the written files -- is about tissue.
    spatial_key = args.spatial_key
    layer_key = args.layer_key or ""
    log("Loading slice 1...")
    adata1 = sc.read_h5ad(args.slice1_h5ad)
    adata1, n1_supplied, n1_off = keep_in_tissue(adata1, "spots of slice 1")
    if n1_off:
        log(f"  slice 1: left out {n1_off} of {n1_supplied} spots with obs['in_tissue'] == 0 (background)")

    log("Loading slice 2...")
    adata2 = sc.read_h5ad(args.slice2_h5ad)
    adata2, n2_supplied, n2_off = keep_in_tissue(adata2, "spots of slice 2")
    if n2_off:
        log(f"  slice 2: left out {n2_off} of {n2_supplied} spots with obs['in_tissue'] == 0 (background)")

    # Validate spatial coordinates and the expression source before any work is done
    for i, ad in enumerate([adata1, adata2], 1):
        if spatial_key not in ad.obsm:
            raise ValueError(f"Slice {i} is missing obsm['{spatial_key}']. Available keys: {list(ad.obsm.keys())}")
    adata1, counts_info1 = _counts_source(adata1, "slice 1", layer_key, use_raw_counts)
    adata2, counts_info2 = _counts_source(adata2, "slice 2", layer_key, use_raw_counts)
    expression_source = counts_info1["expression_source"]

    renamed1 = make_names_unique_and_report(adata1)
    log(f"  Slice 1: n_obs={adata1.n_obs}, n_vars={adata1.n_vars}")
    renamed2 = make_names_unique_and_report(adata2)
    log(f"  Slice 2: n_obs={adata2.n_obs}, n_vars={adata2.n_vars}")

    # Find common genes, in slice 1's order
    common_genes = _shared_genes(adata1.var_names, adata2.var_names)
    log(f"Common genes between slices: {len(common_genes)}")
    if len(common_genes) == 0:
        raise ValueError(id_mismatch_msg("genes", "slice 1", adata1.var_names, "slice 2", adata2.var_names))

    adata1 = adata1[:, common_genes].copy()
    adata2 = adata2[:, common_genes].copy()
    if layer_key:
        adata1.X = _expression_source(adata1, layer_key, "slice 1").copy()
        adata2.X = _expression_source(adata2, layer_key, "slice 2").copy()
    counts_like = _looks_like_counts(adata1.X) and _looks_like_counts(adata2.X)

    # Joint HVG selection over both slices, on the matrix the flavour expects
    adata1.obs["_gpsa_slice"] = "slice1"
    adata2.obs["_gpsa_slice"] = "slice2"
    adata_concat = sc.concat([adata1, adata2], label="_gpsa_batch", keys=["slice1", "slice2"])
    adata_concat.obs_names_make_unique()
    if hvg_flavor == "seurat_v3":
        hvg_input = f"unnormalised {expression_source} (seurat_v3 ranks raw counts)"
    else:
        sc.pp.normalize_total(adata_concat, target_sum=1e4)
        sc.pp.log1p(adata_concat)
        hvg_input = f"log1p(normalize_total(target_sum=1e4)) of {expression_source}"

    n_hvg = min(int(args.n_top_genes), len(common_genes))
    log(f"Selecting {n_hvg} HVGs jointly (flavor={hvg_flavor!r}) on {hvg_input}...")
    try:
        sc.pp.highly_variable_genes(adata_concat, n_top_genes=n_hvg, flavor=hvg_flavor, span=1.0)
    except Exception as exc:
        # Kept narrow on purpose (test/test_worker_hvg_seurat_v3_fallback.py): a broken or absent
        # scikit-misc raises in here. It is named, and no other flavour is substituted.
        if hvg_flavor == "seurat_v3":
            require_hvg_flavor(hvg_flavor)  # raises ImportError naming scikit-misc when that is the cause
        raise RuntimeError(
            f"highly_variable_genes(flavor={hvg_flavor!r}) failed and no other flavour was substituted: {exc}. "
            "hvg_flavor='seurat' ranks log-normalised dispersion instead and needs no scikit-misc."
        ) from exc
    hvg_mask = np.asarray(adata_concat.var["highly_variable"], dtype=bool)
    hvg_set = {str(g) for g in adata_concat.var_names[hvg_mask]}
    hvg_genes = [g for g in common_genes if g in hvg_set]
    log(f"Selected {len(hvg_genes)} highly variable genes")
    if not hvg_genes:
        raise RuntimeError(
            f"highly_variable_genes(flavor={hvg_flavor!r}, n_top_genes={n_hvg}) selected no gene out of "
            f"{len(common_genes)} shared genes; nothing was substituted. Check that {expression_source} holds "
            "expression, or pass another hvg_flavor."
        )

    # The model's expression input, and what each output h5ad's X holds
    sc.pp.normalize_total(adata1, target_sum=1e4)
    sc.pp.log1p(adata1)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)

    n1, n2 = int(adata1.n_obs), int(adata2.n_obs)
    n_total = n1 + n2
    p = len(hvg_genes)
    dense_bytes = _check_dense_budget(n_total, p)

    adata1_hvg = adata1[:, hvg_genes]
    adata2_hvg = adata2[:, hvg_genes]

    # Extract expression matrices (dense: GPSA's likelihood is over the full spots x genes matrix)
    import scipy.sparse as sp

    def _to_dense(X):
        if sp.issparse(X):
            return X.toarray().astype(np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    Y1 = _to_dense(adata1_hvg.X)
    Y2 = _to_dense(adata2_hvg.X)

    # Extract spatial coordinates
    X1, note1 = spatial_coords(adata1, spatial_key, want=args.n_spatial_dims, tool="GPSA", project=True)
    X2, note2 = spatial_coords(adata2, spatial_key, want=args.n_spatial_dims, tool="GPSA", project=True)
    projection_notes = []
    for which, note in (("slice 1", note1), ("slice 2", note2)):
        if note:
            projection_notes.append(f"{which}: {note} Raise n_spatial_dims to keep it.")
            log(projection_notes[-1])
    for which, X in (("1", X1), ("2", X2)):
        bad = int((~np.isfinite(X)).any(axis=1).sum())
        if bad:
            raise ValueError(
                f"Slice {which}: {bad} of {len(X)} spots have a non-finite coordinate in obsm['{spatial_key}']. "
                "GPSA cannot place them; fix or remove those spots in the input first."
            )

    # Normalize spatial coordinates to [0, 1] range jointly -- the frame GPSA is trained in
    x_min, x_range = _joint_unit_box(X1, X2)
    X1_norm = _to_unit_box(X1, x_min, x_range)
    X2_norm = _to_unit_box(X2, x_min, x_range)

    log(f"Data prepared: n1={n1}, n2={n2}, n_genes={p}, n_spatial_dims={args.n_spatial_dims}")

    # Build data dict for GPSA - format: { modality: { n_samples_list, spatial_coords, outputs } }
    Y = np.vstack([Y1, Y2])
    X = np.vstack([X1_norm, X2_norm])

    modality_name = "expression"
    data_dict = {
        modality_name: {
            "n_samples_list": [n1, n2],
            "spatial_coords": torch.tensor(X, dtype=torch.float32),
            "outputs": torch.tensor(Y, dtype=torch.float32),
        }
    }
    del Y, Y1, Y2

    m_X_per_view = _n_inducing(n1, n2)
    m_G = _n_inducing(n1, n2)
    n_latent_gps_dict = {modality_name: args.n_latent_gps}

    # Initialize GPSA model. Upstream places the inducing points with sklearn KMeans, whose
    # multi-threaded Lloyd step sums per-thread partials in completion order: under one seed the
    # starting points differed in the last float bit from run to run and training amplified that
    # into different alignments. One thread for this step only makes random_seed reproduce a run.
    from threadpoolctl import threadpool_limits

    log("Initializing VariationalGPSA model...")
    with threadpool_limits(limits=1):
        model = VariationalGPSA(
            data_dict,
            n_spatial_dims=args.n_spatial_dims,
            m_X_per_view=m_X_per_view,
            m_G=m_G,
            n_latent_gps=n_latent_gps_dict,
            fixed_view_idx=0,  # fix slice1 as reference
        )

    # Build X_spatial dict for forward pass
    X_spatial = {modality_name: torch.tensor(X, dtype=torch.float32)}

    # Training: full-batch, every spot of both slices at every step (upstream has no mini-batch path)
    log(f"Training GPSA model for {args.num_epochs} epochs (full-batch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    final_loss = float("nan")
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()

        # Forward pass returns (G_means, G_samples, F_latent, F_observed)
        G_means, G_samples, F_latent, F_observed = model.forward(X_spatial, model.view_idx, model.Ns)

        loss = model.loss_fn(data_dict, F_observed)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.item())
        if not math.isfinite(final_loss):
            raise RuntimeError(
                f"GPSA training diverged: the loss is {final_loss} at epoch {epoch + 1}. "
                f"Lower learning_rate (now {args.learning_rate}); nothing was written."
            )
        if (epoch + 1) % 100 == 0 or epoch == 0:
            log(f"  Epoch {epoch + 1}/{args.num_epochs}, loss={final_loss:.4f}")

    log("Training complete")

    # Extract aligned coordinates
    model.eval()
    with torch.no_grad():
        G_means_final, _, _, _ = model.forward(X_spatial, model.view_idx, model.Ns)

    aligned_norm = np.asarray(G_means_final[modality_name].detach().cpu().numpy(), dtype=np.float64)
    if not np.all(np.isfinite(aligned_norm)):
        raise RuntimeError(
            "GPSA returned non-finite aligned coordinates. Lower learning_rate or num_epochs; nothing was written."
        )
    aligned_norm1 = aligned_norm[:n1]
    aligned_norm2 = aligned_norm[n1:]
    # Back into the units of obsm[spatial_key], so 'spatial_aligned' sits beside 'spatial' in one frame
    aligned_X1 = _from_unit_box(aligned_norm1, x_min, x_range)
    aligned_X2 = _from_unit_box(aligned_norm2, x_min, x_range)

    log(f"Aligned coordinates shape: slice1={aligned_X1.shape}, slice2={aligned_X2.shape}")

    # Save aligned coordinates as CSV (input units)
    coord_cols = [f"aligned_{d}" for d in range(args.n_spatial_dims)]

    aligned_df1 = pd.DataFrame(aligned_X1, index=adata1.obs_names, columns=coord_cols)
    aligned_df1["slice"] = "slice1"
    aligned_df2 = pd.DataFrame(aligned_X2, index=adata2.obs_names, columns=coord_cols)
    aligned_df2["slice"] = "slice2"
    aligned_df = pd.concat([aligned_df1, aligned_df2])

    aligned_csv = out_dir / "gpsa_aligned_coordinates.csv"
    _write_csv_atomic(aligned_df, aligned_csv)
    log(f"Saved aligned coordinates to {aligned_csv}")

    # Save annotated h5ad with aligned coordinates
    frame_record = {
        "coordinate_frame": COORDINATE_FRAME,
        "spatial_key": spatial_key,
        "normalized_key": "spatial_aligned_normalized",
        "normalization_min": [float(v) for v in x_min],
        "normalization_range": [float(v) for v in x_range],
        "fixed_slice": "slice1",
    }
    for ad, aligned, aligned_n in ((adata1, aligned_X1, aligned_norm1), (adata2, aligned_X2, aligned_norm2)):
        ad.obsm["spatial_aligned"] = aligned
        ad.obsm["spatial_aligned_normalized"] = aligned_n
        ad.uns["gpsa_alignment"] = dict(frame_record)

    h5ad_out1 = out_dir / "gpsa_slice1_aligned.h5ad"
    h5ad_out2 = out_dir / "gpsa_slice2_aligned.h5ad"
    _write_h5ad_atomic(adata1, h5ad_out1)
    _write_h5ad_atomic(adata2, h5ad_out2)
    log(f"Saved aligned slice1 h5ad to {h5ad_out1}")
    log(f"Saved aligned slice2 h5ad to {h5ad_out2}")

    # Mean displacement per slice, in input units and in the [0, 1] training frame
    disp1 = float(np.mean(np.linalg.norm(aligned_X1 - X1, axis=1)))
    disp2 = float(np.mean(np.linalg.norm(aligned_X2 - X2, axis=1)))
    disp1_n = float(np.mean(np.linalg.norm(aligned_norm1 - X1_norm, axis=1)))
    disp2_n = float(np.mean(np.linalg.norm(aligned_norm2 - X2_norm, axis=1)))
    unit = f"units of obsm['{spatial_key}']"

    log(f"Mean displacement ({unit}): slice1={disp1:.4f}, slice2={disp2:.4f}")
    log(f"Final training loss: {final_loss:.4f}")

    out = WorkerOutput("gpsa", task="align_slices")
    out.set_data(
        n_spots_slice1=n1,
        n_spots_slice2=n2,
        n_spots_total=n_total,
        n_common_genes=len(common_genes),
        n_hvg_genes=len(hvg_genes),
    )
    out.add_output_files(
        {
            "aligned_coordinates_csv": str(aligned_csv),
            "slice1_aligned_h5ad": str(h5ad_out1),
            "slice2_aligned_h5ad": str(h5ad_out2),
        }
    )
    out.add_params(
        {
            "n_spatial_dims": args.n_spatial_dims,
            "n_latent_gps": args.n_latent_gps,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "n_top_genes": args.n_top_genes,
            "hvg_flavor": hvg_flavor,
            "layer_key": layer_key,
            "spatial_key": spatial_key,
            "random_seed": args.random_seed,
        }
    )
    out.add_params(
        {
            "expression_source": expression_source,
            "hvg_input": hvg_input,
            "n_hvg_used": len(hvg_genes),
            "training": "full-batch",
            "coordinate_frame": COORDINATE_FRAME,
            "normalization_min": [float(v) for v in x_min],
            "normalization_range": [float(v) for v in x_range],
            "dense_expression_bytes_estimate": int(dense_bytes),
        }
    )
    record_method(out, METHOD_NAME, used_fallback=False)
    record_in_tissue(out, n1_supplied + n2_supplied, n1_off + n2_off)
    if n1_off or n2_off:
        out.add_params({"in_tissue_dropped_per_slice": [int(n1_off), int(n2_off)]})
    _record_counts_source(out, [counts_info1, counts_info2])
    out.add_params({"use_raw_counts": use_raw_counts})
    record_ignored(
        out,
        "batch_size",
        "upstream GPSA has no mini-batch training; every step uses every spot of both slices",
    )
    if hvg_flavor == "seurat_v3" and not counts_like:
        out.add_warning(
            f"hvg_flavor='seurat_v3' ranks raw counts, but {expression_source} holds non-integer or negative "
            "values, so the gene ranking is not the one seurat_v3 defines. Pass layer_key naming a raw-count "
            "layer, or hvg_flavor='seurat' for log-normalised input."
        )
    for note in projection_notes:
        out.add_warning(note)
    if projection_notes:
        out.add_params({"spatial_projection": projection_notes})
    out.add_params(identifier_rename_params(renamed1, suffix="slice1"))
    out.add_params(identifier_rename_params(renamed2, suffix="slice2"))
    out.set_summary(
        final_loss=final_loss,
        mean_displacement_slice1=disp1,
        mean_displacement_slice2=disp2,
        displacement_unit=unit,
        mean_displacement_slice1_normalized=disp1_n,
        mean_displacement_slice2_normalized=disp2_n,
        n_inducing_spatial=m_X_per_view,
        n_inducing_expression_gp=m_G,
    )
    out.set_analysis(
        f"GPSA aligned 2 spatial slices ({n1} + {n2} spots, slice 1 fixed as the template) on "
        f"{len(hvg_genes)} HVGs of {len(common_genes)} shared genes, chosen jointly by flavor '{hvg_flavor}' on "
        f"{hvg_input}, over {args.num_epochs} full-batch epochs (batch_size has no effect). "
        f"Final loss: {final_loss:.4f}. Mean coordinate displacement ({unit}): slice1={disp1:.4f}, "
        f"slice2={disp2:.4f}. Aligned coordinates are in the units of obsm['{spatial_key}'] -- the joint [0, 1] "
        f"training frame was inverted -- in gpsa_aligned_coordinates.csv and obsm['spatial_aligned'] of each "
        f"output h5ad; the [0, 1] values are kept under obsm['spatial_aligned_normalized']. Each output h5ad "
        f"holds the {len(common_genes)} shared genes with X = log1p(normalize_total(1e4)) of {expression_source}."
        + (
            f" {n1_off + n2_off} background spots (obs['in_tissue'] == 0) were left out before alignment "
            f"(slice 1: {n1_off} of {n1_supplied}, slice 2: {n2_off} of {n2_supplied}); the outputs hold the "
            "in-tissue spots only."
            if (n1_off or n2_off)
            else ""
        )
        + (" " + " ".join(projection_notes) if projection_notes else "")
        + identifier_rename_note(renamed1, subject="first slice")
        + identifier_rename_note(renamed2, subject="second slice")
    )
    return out.to_dict()


def main():
    parser = argparse.ArgumentParser(description="GPSA worker: align spatial transcriptomics slices.")
    parser.add_argument("--slice1-h5ad", required=True, help="Path to first slice AnnData (.h5ad).")
    parser.add_argument("--slice2-h5ad", required=True, help="Path to second slice AnnData (.h5ad).")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--n-spatial-dims", type=int, default=2, help="Number of spatial dimensions.")
    parser.add_argument("--n-latent-gps", type=int, default=3, help="Number of latent Gaussian processes.")
    parser.add_argument("--num-epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for Adam.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Accepted and ignored: GPSA trains full-batch (reported under params.ignored).",
    )
    parser.add_argument("--n-top-genes", type=int, default=2000, help="Number of HVGs for alignment.")
    parser.add_argument(
        "--hvg-flavor",
        default=DEFAULT_HVG_FLAVOR,
        help="Scanpy HVG flavour: seurat_v3 (raw counts, needs scikit-misc), seurat or cell_ranger (log data).",
    )
    parser.add_argument(
        "--layer-key", default="", help="AnnData layer holding the expression, in both slices. Empty uses adata.X."
    )
    parser.add_argument(
        "--use-raw-counts",
        action="store_true",
        help="Read both slices' expression from adata.raw (raw counts) instead of X; refused with --layer-key, "
        "or when a slice has no adata.raw or it does not hold counts.",
    )
    parser.add_argument("--spatial-key", default="spatial", help="obsm key with spatial coordinates.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed.")

    args = parser.parse_args()

    # Redirect stdout to stderr during processing
    import json

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr

    result = None
    error_msg = None
    error_exc = None
    try:
        result = run_alignment(args)
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc(file=sys.stderr)
        error_msg = str(e)
        error_exc = e
    finally:
        sys.stdout = orig_stdout

    # Print a single JSON line to stdout
    if result is None:
        WorkerOutput.emit_error("gpsa", error_msg, task="align_slices", exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
