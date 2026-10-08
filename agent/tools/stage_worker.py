#!/usr/bin/env python
"""
STAGE worker script for SpatialOmicsLab MCP integration (generic, tutorial-agnostic).

Runs inside /opt/conda/envs/stage_env and calls upstream STAGE (``STAGE.STAGE``) once:

    STAGE.STAGE(
        adata,
        save_path=...,
        data_type=...,
        experiment=...,
        coord_sf=...,
        down_ratio=...,       # recovery
        train_epoch=...,
        seed=...,
        batch_size=...,
        learning_rate=...,
        w_recon=...,
        w_w=...,
        w_l1=...,
        relu=...,
        device=...,
        sec_name=...,         # 3d_model
        select_section=...,   # 3d_model
        gap=...,              # 3d_model
        path1=...             # STAGE's scratch directory, always inside save_path
    )

Return signatures (from STAGE docs):
  - experiment='recovery'   -> (adata_sample, adata_stage)
  - experiment='generation' -> adata_stage
  - experiment='3d_model'   -> (adata_stage, adata_simu, adata_all)

What the worker prepares, as the STAGE tutorials do (``tools/third_party/STAGE/tutorials``):

* ``obsm['coord']``. Used as supplied when present. Otherwise, for 10x and ST_KTH, it is built from
  ``obs[['array_col', 'array_row']]`` -- the tutorials' layout. ``generation`` places each new spot
  at a fixed fraction of an array-grid step from a measured one (``STAGE.utils.generation_coord_10x``
  adds 2/3, 1 and +-1/3; ``generation_coord_ST`` adds 0.5), so it needs grid units. Pixel
  coordinates put every "generated" spot within a pixel of a measured one; they are refused, not
  used. ``recovery`` and ``3d_model`` take ``obsm['spatial']`` (x, y, z for 3d_model).
* ``coord_sf``. Upstream asks for the largest scaled coordinate to be about 1. When left unset the
  tutorial value (77 / 35 / 6000) is used if it does that, otherwise the largest |x|,|y|; the value
  and its source are published.
* Genes. Unless ``var['highly_variable']`` is present, highly variable genes are selected with
  ``hvg_flavor`` (seurat_v3 on counts, as in the tutorials) and X is normalised and log1p'd. STAGE
  models only those genes; ``data.n_genes_used`` says how many. A seurat_v3 that cannot run stops the
  run and names scikit-misc; no other flavour is substituted.

* Memory. Upstream holds the training matrix dense (``dataset.get_data`` densifies
  ``X[:, highly_variable]``) and decodes every output position into a dense float32 matrix. That is
  the method, not a worker choice; the worker estimates the peak and, when it exceeds the memory
  available (``worker_utils.available_memory_bytes``: MemAvailable, or the room left under a cgroup
  limit once what the container already uses is counted), refuses with the numbers before training
  (``params.dense_peak_estimate_gib``).

* Spots. ``obs['in_tissue'] == 0`` marks background outside the tissue (CELLxGENE Visium exports
  carry every array spot). Those spots are left out right after loading, before training or
  generation (``worker_utils.keep_in_tissue``), and reported in ``params.in_tissue_filter``, a warning
  and the analysis; ``data.n_spots`` counts the measured spots STAGE modelled, ``data.n_spots_input``
  the spots supplied.

* Matrix. When the worker normalises (no ``var['highly_variable']`` supplied), X is read as counts
  under ``worker_utils.choose_counts_matrix``: a negative or non-finite X is refused, naming
  ``use_raw_counts`` when ``adata.raw`` holds counts; a non-negative non-integer X runs with a
  warning. ``use_raw_counts=True`` reads ``adata.raw.X`` and runs the tutorial preprocessing on it.
  An input that carries its own ``var['highly_variable']`` is used as supplied -- normalised data is
  accepted there by design -- and is not refused.

* pandas. Upstream's generation offsets were written for pandas 1.x column assignment; under the
  env's pandas 2 they write through one shared buffer and generation returns the measured spots,
  moved, as its "generated" ones. Generation therefore runs under pandas Copy-on-Write, and a
  result that moved ``obsm['coord']`` or generated nothing is refused rather than published.

``output_files`` lists only files this run created or rewrote under ``output_dir``; files that were
already there (an earlier run, another tool) are not republished as this run's results. Upstream
writes scratch files relative to the working directory, so STAGE runs with ``save_path`` as its
working directory and nothing lands outside ``output_dir``.

All human-readable logs go to stderr with '[stage-worker]' prefix.
Only ONE JSON line is printed to stdout at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import traceback
from typing import Any

import anndata as ad  # type: ignore
from worker_utils import (
    WorkerOutput,
    available_memory_bytes,
    choose_counts_matrix,
    expression_matrix_kind,
    keep_in_tissue,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    require_hvg_flavor,
    resolve_compute,
    spatial_coords,
    unsupported_choice_msg,
)

# STAGE_SRC first, then the vendored checkout, then whatever stage_env has installed. The vendored copy is upstream
# 73954b4 (the commit the recipe installs) with one local fix: model.py sets X_dim on the recovery branch, which
# upstream never does, so `experiment='recovery'` runs instead of stopping with UnboundLocalError.
STAGE_SRC = os.environ.get("STAGE_SRC") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "third_party", "STAGE"
)
if os.path.isfile(os.path.join(STAGE_SRC, "STAGE", "model.py")) and STAGE_SRC not in sys.path:
    sys.path.insert(0, STAGE_SRC)

# The data_type -> experiment pairs upstream STAGE.STAGE accepts (model.py raises on any other pair,
# but only after the HVG step has run). Checked first so a wrong pair costs milliseconds.
VALID_EXPERIMENTS = {
    "10x": ("generation", "recovery"),
    "ST_KTH": ("generation",),
    "Slide-seq": ("3d_model",),
}

# The coord_sf each STAGE tutorial uses for its platform's coordinate units.
TUTORIAL_COORD_SF = {"10x": 77.0, "ST_KTH": 35.0, "Slide-seq": 6000.0}

# Spacing between neighbouring spots in the grid units STAGE's generation offsets assume: Visium
# array (array_col, array_row) neighbours sit at (+-1, +-1), legacy ST array neighbours at 1.
GRID_PITCH = {"10x": 2**0.5, "ST_KTH": 1.0}
# A median nearest-neighbour spacing outside this band is not grid units (pixels are ~100-300).
GRID_PITCH_BAND = (0.5, 3.0)
# A worker-chosen coord_sf is kept when it puts the largest |x|,|y| inside this band ("about 1").
SCALED_MAX_BAND = (0.25, 4.0)

DEFAULT_HVG_FLAVOR = "seurat_v3"
DEFAULT_SECTIONS = (1, 3, 5, 6, 8)


def log(msg: str) -> None:
    print(f"[stage-worker] {msg}", file=sys.stderr, flush=True)


def snapshot_files(root: str) -> dict[str, tuple]:
    """``{relative path: (mtime_ns, size)}`` for every file under ``root``."""
    snap: dict[str, tuple] = {}
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            path = os.path.join(dirpath, f)
            try:
                st = os.stat(path)
            except OSError:
                continue
            snap[os.path.relpath(path, root)] = (st.st_mtime_ns, st.st_size)
    return snap


def files_written_since(root: str, before: dict[str, tuple]) -> list[str]:
    """Files under ``root`` that are new, or rewritten, since ``before`` was taken."""
    after = snapshot_files(root)
    return sorted(k for k, v in after.items() if before.get(k) != v)


def write_h5ad_atomic(adata: Any, path: str) -> None:
    tmp = path + ".partial"
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def parse_select_section(s: str | None) -> list[str] | None:
    """The comma-separated section IDs, as tokens; ``None`` when none were given."""
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts or None


def check_experiment(data_type: str, experiment: str) -> None:
    """Refuse a data_type/experiment pair upstream STAGE.STAGE would refuse, before any work."""
    allowed = VALID_EXPERIMENTS.get(data_type)
    if allowed is None:
        raise ValueError(unsupported_choice_msg("data_type", data_type, sorted(VALID_EXPERIMENTS)))
    if experiment not in allowed:
        pairs = "; ".join(f"{d}: {', '.join(e)}" for d, e in VALID_EXPERIMENTS.items())
        raise ValueError(
            unsupported_choice_msg(
                "experiment",
                experiment,
                list(allowed),
                extra=f"Upstream STAGE runs each experiment on one platform only ({pairs}); "
                f"data_type={data_type!r} was given.",
            )
        )


def median_nn_spacing(xy: Any) -> float:
    """Median distance from each distinct position to its nearest neighbour."""
    import numpy as np
    from scipy.spatial import cKDTree  # type: ignore

    pts = np.unique(np.asarray(xy, dtype=float), axis=0)
    if pts.shape[0] < 2:
        return float("nan")
    dist, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(dist[:, 1]))


def _numeric_obs_column(adata: Any, name: str) -> Any:
    import numpy as np
    import pandas as pd

    values = pd.to_numeric(pd.Series(np.asarray(adata.obs[name], dtype=object)), errors="coerce").to_numpy(float)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise ValueError(
            f"obs['{name}'] has {bad} non-numeric or missing value(s); it cannot serve as an array coordinate."
        )
    return values


def stage_coordinates(adata: Any, data_type: str, experiment: str) -> dict[str, Any]:
    """The coordinates STAGE will read from ``obsm['coord']``, where they came from, and their spacing.

    Raises for ``generation`` when the coordinates are not in array-grid units: upstream adds fixed
    fractions of a grid step to them, which on pixel coordinates is less than a pixel.
    """
    import numpy as np

    want = 3 if experiment == "3d_model" else 2
    obs_cols = set(map(str, adata.obs.columns))
    if "coord" in adata.obsm:
        coords, _ = spatial_coords(adata, "coord", want=want, tool="stage")
        source = "obsm['coord'] (supplied)"
    elif data_type in GRID_PITCH and {"array_row", "array_col"} <= obs_cols:
        coords = np.column_stack([_numeric_obs_column(adata, "array_col"), _numeric_obs_column(adata, "array_row")])
        source = "obs[['array_col', 'array_row']] (array grid, the STAGE tutorial layout)"
    elif "spatial" in adata.obsm:
        coords, _ = spatial_coords(adata, "spatial", want=want, tool="stage")
        source = "obsm['spatial']"
    else:
        raise KeyError(
            "STAGE needs spot coordinates: obsm['coord'], obs['array_row'] + obs['array_col'] (10x / ST_KTH), "
            f"or obsm['spatial']. This file has obsm keys {sorted(adata.obsm.keys())} and none of those."
        )
    coords = np.asarray(coords, dtype=float)
    n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
    if n_bad:
        raise ValueError(f"{n_bad} of {coords.shape[0]} spots have a missing or infinite coordinate in {source}.")

    spacing = None if experiment == "3d_model" else median_nn_spacing(coords[:, :2])
    if experiment == "generation":
        lo, hi = GRID_PITCH_BAND
        if not (spacing is not None and lo <= spacing <= hi):
            if data_type == "10x":
                offsets = "2/3, 1 and +-1/3 of an array step (STAGE.utils.generation_coord_10x)"
                expected = "about 1.4 (Visium array_col/array_row neighbours sit at +-1, +-1)"
                how = (
                    "obs['array_row'] and obs['array_col'] (the Visium tissue_positions columns), or "
                    "obsm['coord'] in array units laid out as (array_col, array_row) as in the STAGE tutorials"
                )
            else:
                offsets = "half an array step (STAGE.utils.generation_coord_ST)"
                expected = "about 1 (ST array x/y)"
                how = "obs['array_row'] and obs['array_col'], or obsm['coord'] in ST array units"
            raise ValueError(
                f"STAGE generation places each new spot {offsets} away from a measured spot, so it needs "
                f"coordinates in array-grid units, where neighbouring spots are {expected} apart. The "
                f"coordinates available here, {source}, have a median nearest-neighbour spacing of "
                f"{spacing:.4g}: every generated spot would sit within a small fraction of a spot pitch of a "
                f"measured one and its profile would be a copy of that spot's. Supply {how}. "
                "experiment='recovery' predicts held-out measured spots and does not need grid units."
            )
    return {"coords": coords, "source": source, "nn_spacing": spacing}


def effective_coord_sf(coords: Any, data_type: str, requested: float | None) -> dict[str, Any]:
    """The coord_sf STAGE will divide x and y by, where it came from, and a warning if it is off-scale."""
    import numpy as np

    xy_max = float(np.max(np.abs(coords[:, :2]))) if coords.size else 0.0
    lo, hi = SCALED_MAX_BAND
    if requested is not None:
        requested = float(requested)
        if not requested > 0:
            raise ValueError(f"coord_sf must be a positive number, got {requested!r}.")
        scaled = xy_max / requested
        warning = None
        if not (lo <= scaled <= hi):
            warning = (
                f"coord_sf={requested:g} scales the largest coordinate ({xy_max:g}) to {scaled:.3g}; upstream "
                "STAGE recommends about 1 (model.py docstring). The requested value was used."
            )
        return {"coord_sf": requested, "source": "requested", "warning": warning, "scaled_max": scaled}
    default = TUTORIAL_COORD_SF[data_type]
    scaled = xy_max / default
    if (lo <= scaled <= hi) or xy_max <= 0:
        return {
            "coord_sf": default,
            "source": f"tutorial value for data_type={data_type!r}",
            "warning": None,
            "scaled_max": scaled,
        }
    return {
        "coord_sf": xy_max,
        "source": (
            f"derived: the tutorial value {default:g} would scale the largest coordinate ({xy_max:g}) to "
            f"{scaled:.3g}, so the largest |x|,|y| was used and the scaled maximum is 1"
        ),
        "warning": None,
        "scaled_max": 1.0,
    }


def resolve_sections(adata: Any, sec_name: str, raw: str | None) -> list:
    """``select_section`` spelled as ``obs[sec_name]`` spells it; refuses IDs the column lacks."""
    import numpy as np
    import pandas as pd

    if sec_name not in adata.obs.columns:
        raise KeyError(
            f"sec_name={sec_name!r} is not an obs column; 3d_model needs the section of every bead. "
            f"obs columns: {list(map(str, adata.obs.columns))[:30]}"
        )
    col = adata.obs[sec_name]
    tokens = parse_select_section(raw)
    tokens = tokens if tokens is not None else [str(s) for s in DEFAULT_SECTIONS]
    base_dtype = col.cat.categories.dtype if isinstance(col.dtype, pd.CategoricalDtype) else col.dtype
    wanted: list = []
    if pd.api.types.is_numeric_dtype(base_dtype):
        for t in tokens:
            try:
                v = float(t)
            except ValueError:
                raise ValueError(
                    f"select_section entry {t!r} is not a number, but obs[{sec_name!r}] holds numbers."
                ) from None
            wanted.append(int(v) if v.is_integer() else v)
        present = set(np.asarray(col).tolist())
    else:
        wanted = list(tokens)
        present = {str(v) for v in np.asarray(col, dtype=object).tolist()}
    missing = [w for w in wanted if w not in present]
    if missing:
        raise ValueError(
            f"select_section names section(s) {missing} that obs[{sec_name!r}] does not contain "
            f"(it holds {sorted(present, key=str)[:30]}). Pass select_section as a comma-separated list of those "
            "values."
        )
    return wanted


def z_gap_warning(adata: Any, coords: Any, sec_name: str, gap: float) -> str | None:
    """Upstream defines gap as half the z-distance between adjacent real sections; say when it is not."""
    import numpy as np
    import pandas as pd

    z = pd.Series(coords[:, 2], index=range(coords.shape[0]))
    by_section = z.groupby(np.asarray(adata.obs[sec_name], dtype=object)).median().sort_values()
    steps = np.diff(by_section.to_numpy(float))
    steps = steps[steps > 0]
    if steps.size == 0:
        return None
    step = float(np.median(steps))
    if abs(2 * gap - step) > 0.25 * step:
        return (
            f"gap={gap:g} puts each simulated section {gap:g} below a real one along z, but the real sections "
            f"are {step:.4g} apart (median step), so the simulated sections are not halfway between them. "
            "Upstream defines gap as half the distance between adjacent real sections."
        )
    return None


def _hvg_failure(flavor: str, exc: BaseException) -> str:
    msg = f"highly_variable_genes(flavor={flavor!r}) failed ({type(exc).__name__}: {exc})."
    if flavor == "seurat_v3":
        msg += (
            " seurat_v3 needs scikit-misc (import name skmisc) and raw counts in X. The worker does not "
            "switch flavour on its own: repair scikit-misc, or pass hvg_flavor='seurat' explicitly."
        )
    return msg


def select_hvgs(adata: Any, sc: Any, data_type: str, experiment: str, hvg_flavor: str) -> dict[str, Any]:
    """Tutorial preprocessing (HVGs, normalize_total, log1p) unless var['highly_variable'] is supplied."""
    import numpy as np

    if "highly_variable" in adata.var.columns:
        n_used = int(np.asarray(adata.var["highly_variable"]).astype(bool).sum())
        log(f"Found adata.var['highly_variable'] ({n_used} genes); skipping HVG selection and normalisation.")
        return {
            "hvg_source": "supplied var['highly_variable']",
            "hvg_flavor": None,
            "hvg_n_top_genes": None,
            "hvg_n_top_genes_tutorial": None,
            "n_genes_used": n_used,
            "normalization": "none: X used as supplied, because var['highly_variable'] was present",
            "warning": None,
        }
    require_hvg_flavor(hvg_flavor)
    # The tutorials: 5000 HVGs for 10x generation (mouse brain) and Slide-seq 3D, 3000 otherwise.
    n_top_tutorial = 5000 if data_type == "Slide-seq" or (data_type == "10x" and experiment == "generation") else 3000
    # A panel narrower than the tutorial's count keeps every gene; the value passed is the one reported.
    n_top = min(n_top_tutorial, int(adata.n_vars))
    # What X holds was decided before this step (choose_stage_matrix); its warning is published there.
    warning = None
    log(f"Selecting up to {n_top} HVGs with flavor={hvg_flavor!r}, then normalize_total + log1p.")
    if hvg_flavor != "seurat_v3":
        # 'seurat' and 'cell_ranger' read log-normalised data; seurat_v3 reads counts.
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, flavor=hvg_flavor, n_top_genes=n_top)
    except Exception as exc:
        raise RuntimeError(_hvg_failure(hvg_flavor, exc)) from exc
    if hvg_flavor == "seurat_v3":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    n_used = int(np.asarray(adata.var["highly_variable"]).astype(bool).sum())
    return {
        "hvg_source": "computed by the worker",
        "hvg_flavor": hvg_flavor,
        "hvg_n_top_genes": n_top,
        "hvg_n_top_genes_tutorial": n_top_tutorial,
        "n_genes_used": n_used,
        "normalization": "normalize_total(target_sum=1e4) + log1p by the worker",
        "warning": warning,
    }


def choose_stage_matrix(adata: Any, use_raw_counts: bool) -> tuple:
    """``(adata, info, note)``: the matrix STAGE is prepared from, and what was decided.

    The worker normalises X as counts (HVG selection, normalize_total, log1p) unless the input
    carries its own ``var['highly_variable']``; then X is used as supplied, which accepts normalised
    data by design, so it is recorded and never refused. Otherwise ``worker_utils.choose_counts_matrix``
    applies the shared rule. ``use_raw_counts=True`` always reads ``adata.raw.X`` and runs the tutorial
    preprocessing on it; a ``highly_variable`` column is not carried onto the raw counts, because the
    worker-selected genes are normalised and the supplied ones are not.
    """
    supplied_flags = "highly_variable" in adata.var.columns
    if not use_raw_counts and supplied_flags:
        info = {"expression_source": "X", "x_matrix_kind": expression_matrix_kind(adata.X), "warning": None}
        return adata, info, None
    chosen, info = choose_counts_matrix(adata, use_raw_counts)
    note = None
    if use_raw_counts and (supplied_flags or "highly_variable" in chosen.var.columns):
        if "highly_variable" in chosen.var.columns:
            chosen.var = chosen.var.drop(columns="highly_variable")
        note = (
            "use_raw_counts=True: highly variable genes were selected on adata.raw's counts and normalised by "
            "the worker; the var['highly_variable'] flags of the input were not used."
        )
    return chosen, info, note


def dense_peak_bytes(X: Any, n_train: int, n_out: int, n_genes: int) -> int:
    """Upper-leaning bytes of the dense matrices upstream STAGE builds; intrinsic to it, not the worker's.

    ``dataset.get_data`` runs ``sp.coo_matrix(adata.X[:, highly_variable].T).todense()`` on the
    training spots and wraps the result in a DataFrame: the column slice, a COO copy (value plus
    int32 row and column per stored entry), the dense matrix and the frame are counted, each as if
    it were a copy. The decoder then writes one float32 row per output position.
    """
    import math

    itemsize = int(getattr(getattr(X, "dtype", None), "itemsize", 8) or 8)
    if hasattr(X, "nnz"):
        density = float(X.nnz) / max(1, int(X.shape[0]) * int(X.shape[1]))
        per_entry = 2.0 * itemsize + density * (2.0 * itemsize + 12.0)
    else:
        per_entry = 4.0 * itemsize + 8.0
    train = int(math.ceil(per_entry * n_train * n_genes))
    return train + 4 * int(n_out) * int(n_genes)


def check_dense_memory(
    X: Any, n_train: int, n_out: int, n_genes: int, available: int | None, experiment: str
) -> int | None:
    """Refuse, with the numbers, a run whose dense matrices cannot fit; return the estimate."""
    if n_genes < 1:
        raise ValueError(
            "no gene is flagged in var['highly_variable'], and STAGE models only flagged genes. Flag the genes "
            "to model, or drop the column so the worker selects highly variable genes itself."
        )
    need = dense_peak_bytes(X, n_train, n_out, n_genes)
    if available is not None and need > available:
        gib = float(1 << 30)
        raise MemoryError(
            f"STAGE {experiment} holds its training matrix dense (upstream dataset.get_data densifies "
            f"X[:, highly_variable]: {n_train} spots x {n_genes} genes) and decodes up to {n_out} positions x "
            f"{n_genes} genes into a dense float32 matrix: about {need / gib:.2f} GiB at peak, and this machine "
            f"has {available / gib:.2f} GiB available. The run was not started. Run it where that much memory "
            "is free; the modelled genes are the ones var['highly_variable'] flags, so an input that carries "
            "its own flags controls the width. The worker does not drop spots."
        )
    return need


def pandas_has_copy_on_write() -> bool:
    import pandas as pd

    try:
        pd.get_option("mode.copy_on_write")
    except Exception:
        return False
    return True


def copy_on_write(enabled: bool) -> Any:
    """pandas Copy-on-Write for the duration of the STAGE call, or a no-op.

    Upstream ``utils.generation_coord_10x`` / ``generation_coord_ST`` build four DataFrames with
    ``pd.DataFrame(adata.obsm['coord'])`` and shift them with ``df.iloc[:, j] = df.iloc[:, j] + d``.
    Written for pandas 1.x, where that assignment replaced the column. Under pandas 2 (the env pins
    2.0.3) it writes in place, and the four frames share one buffer: every shift lands on all of
    them and on ``obsm['coord']``. On float coordinates generation then returns exactly the measured
    spots, moved by (+2, +2/3); on integer coordinates the three new spots per spot sit on a vertical
    line through it. Copy-on-Write gives each frame its own data, which is the behaviour upstream was
    written against.
    """
    import contextlib

    import pandas as pd

    return pd.option_context("mode.copy_on_write", True) if enabled else contextlib.nullcontext()


def recovery_unavailable_msg(exc: BaseException) -> str:
    return (
        "upstream STAGE cannot run experiment='recovery' as installed: STAGE/model.py sets X_dim (the latent "
        "width) only in its generation/higher_res and 3d_model branches, so the recovery branch stops with "
        f"'{exc}' before training. The worker does not substitute another method; experiment='generation' "
        "runs, and recovery needs a STAGE build that sets X_dim for it."
    )


def check_generation(adata: Any, coords_before: Any, n_out: int, n_measured: int) -> None:
    """Refuse a generation that moved the measured spots or generated nothing."""
    import numpy as np

    after = np.asarray(adata.obsm["coord"], dtype=float)
    if after.shape != coords_before.shape or not np.array_equal(after, coords_before):
        raise RuntimeError(
            "upstream STAGE changed obsm['coord'] in place during generation, so the measured spots and the "
            "generated positions are both displaced (pandas in-place column assignment on a shared array). "
            "The result is not published."
        )
    if n_out <= n_measured:
        raise RuntimeError(
            f"STAGE generation returned {n_out} positions for {n_measured} measured spots, so nothing was "
            "generated. The result is not published."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic STAGE worker for SpatialOmicsLab MCP")
    parser.add_argument("--h5ad", required=True, help="Input AnnData (.h5ad)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--data-type",
        required=True,
        choices=["10x", "ST_KTH", "Slide-seq"],
        help="Platform type, passed to STAGE.STAGE (data_type)",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["recovery", "generation", "3d_model"],
        help="Analysis objective, passed to STAGE.STAGE (experiment)",
    )
    parser.add_argument("--coord-sf", type=float, default=None, help="coord_sf")
    parser.add_argument("--down-ratio", type=float, default=None, help="down_ratio")
    parser.add_argument("--train-epoch", type=int, default=10, help="train_epoch")
    parser.add_argument("--batch-size", type=int, default=None, help="batch_size")
    parser.add_argument("--learning-rate", type=float, default=None, help="learning_rate")
    parser.add_argument("--w-recon", type=float, default=None, help="w_recon")
    parser.add_argument("--w-w", type=float, default=None, help="w_w")
    parser.add_argument("--w-l1", type=float, default=None, help="w_l1")
    parser.add_argument("--use-gpu", action="store_true", help="Use a GPU if one is available.")
    parser.add_argument(
        "--device",
        default="",
        help="Compute device: 'cpu', 'gpu'/'cuda', or 'cuda:N'. Overrides --use-gpu when set.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--sec-name", type=str, default=None, help="Section column for 3D")
    parser.add_argument(
        "--select-section",
        type=str,
        default=None,
        help="Comma-separated section IDs for 3D (e.g. '1,3,5,6,8')",
    )
    parser.add_argument("--gap", type=float, default=None, help="Gap between sections for 3D")
    parser.add_argument(
        "--no-relu",
        dest="relu",
        action="store_false",
        help="Disable ReLU (default: use ReLU)",
    )
    parser.add_argument(
        "--hvg-flavor",
        default=DEFAULT_HVG_FLAVOR,
        help="scanpy HVG flavour when var['highly_variable'] is absent: seurat_v3 (counts), seurat, cell_ranger.",
    )
    parser.add_argument(
        "--use-raw-counts",
        dest="use_raw_counts",
        action="store_true",
        default=False,
        help="Read adata.raw.X instead of X and run the tutorial preprocessing on those counts.",
    )

    args = parser.parse_args()

    st_h5ad = os.path.abspath(args.h5ad)
    output_dir = os.path.abspath(args.output_dir)
    data_type = args.data_type
    experiment = args.experiment

    os.makedirs(output_dir, exist_ok=True)

    try:
        log("STAGE worker starting.")
        log(f"h5ad      : {st_h5ad}")
        log(f"output_dir: {output_dir}")
        log(f"data_type : {data_type}")
        log(f"experiment: {experiment}")

        check_experiment(data_type, experiment)
        if args.hvg_flavor not in ("seurat_v3", "seurat", "cell_ranger"):
            raise ValueError(
                unsupported_choice_msg("hvg_flavor", args.hvg_flavor, ["seurat_v3", "seurat", "cell_ranger"])
            )
        if not os.path.exists(st_h5ad):
            raise FileNotFoundError(f"Input AnnData file not found: {st_h5ad}")

        # Import heavy deps inside STAGE env
        try:
            import numpy as np  # type: ignore
            import scanpy as sc  # type: ignore
            import STAGE  # type: ignore
            import torch  # type: ignore
        except Exception as e:
            log("ERROR importing STAGE / torch / scanpy:")
            log(str(e))
            traceback.print_exc(file=sys.stderr)
            # The cause goes in the message itself: it is the line a caller reads first, and "which
            # module" is the whole diagnosis (stage_env shipped without tqdm, which STAGE/model.py imports).
            WorkerOutput.emit_error(
                "stage",
                f"Failed to import STAGE environment dependencies ({type(e).__name__}: {e}).",
                task=experiment,
            )
            sys.exit(1)

        # Device. Was hardcoded to card zero: a boolean --use-gpu cannot name any other one, so
        # every concurrent STAGE run on a multi-GPU box piled onto the same card. `--device` wins
        # when given; otherwise the legacy boolean still decides, so the default stays CPU.
        device_str = resolve_compute(args.device or args.use_gpu).device
        log(f"Using device: {device_str}")

        # Seed
        try:
            import random

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            # Keyed off the device actually chosen, not off what the box happens to have: seeding
            # CUDA's generators is meaningless when the run was deliberately put on the CPU.
            if device_str.startswith("cuda"):
                torch.cuda.manual_seed_all(args.seed)
        except Exception as e:
            log(f"WARNING: failed to fully set random seeds: {e!r}")

        # Load AnnData
        log("Loading AnnData...")
        adata = ad.read_h5ad(st_h5ad)
        log(f"AnnData loaded: {adata.n_obs} spots × {adata.n_vars} genes")

        # Background spots (obs['in_tissue'] == 0) are glass, not tissue: STAGE neither trains on them
        # nor generates positions around them.
        adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
        if n_spots_off_tissue:
            log(
                f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0 "
                f"(background); {adata.n_obs} in-tissue spots remain."
            )

        # The worker normalises X as counts unless var['highly_variable'] is supplied.
        adata, counts_info, counts_note = choose_stage_matrix(adata, args.use_raw_counts)
        log(f"Expression matrix: {counts_info['expression_source']} ({counts_info['x_matrix_kind']})")
        n_obs_in, n_vars_in = int(adata.n_obs), int(adata.n_vars)

        # ---- Tutorial-aligned preprocessing ----

        # Parameters that apply to one experiment only: accepted, not applied elsewhere, and said so.
        ignored: list[str] = []
        down_ratio = args.down_ratio
        sec_name = args.sec_name
        gap = args.gap
        if experiment != "recovery" and down_ratio is not None:
            ignored.append("down_ratio")
            down_ratio = None
        if experiment != "3d_model":
            for name in ("sec_name", "select_section", "gap"):
                if getattr(args, name) is not None:
                    ignored.append(name)
            sec_name, gap = None, None

        # 1) Coordinates, checked before the expensive steps.
        coord_info = stage_coordinates(adata, data_type, experiment)
        coords = coord_info["coords"]
        adata.obsm["coord"] = coords
        log(f"coordinates: {coord_info['source']}; median neighbour spacing {coord_info['nn_spacing']}")

        select_section = None
        z_warning = None
        if experiment == "3d_model":
            sec_name = sec_name if sec_name is not None else "section"
            gap = gap if gap is not None else 0.05
            select_section = resolve_sections(adata, sec_name, args.select_section)
            z_warning = z_gap_warning(adata, coords, sec_name, gap)

        # 2) Genes: STAGE models only var['highly_variable'] (upstream dataset.get_data).
        with contextlib.redirect_stdout(sys.stderr):
            hvg = select_hvgs(adata, sc, data_type, experiment, args.hvg_flavor)

        # 3) Upstream densifies the training matrix and decodes every output position densely; that
        #    is the method, so the worker estimates it and refuses with the numbers rather than
        #    letting the kernel kill the run.
        n_obs = int(adata.n_obs)
        if experiment == "3d_model":
            # Real sections, simulated sections, and both again in adata_all.
            n_train, n_out = int(adata.obs[sec_name].isin(select_section).sum()), 4 * n_obs
        elif experiment == "generation":
            n_train, n_out = n_obs, 4 * n_obs  # up to three new positions per measured spot
        else:
            n_train, n_out = 2 * n_obs, n_obs  # recovery densifies the sample and the full matrix
        dense_estimate = check_dense_memory(
            adata.X, n_train, n_out, hvg["n_genes_used"], available_memory_bytes(), experiment
        )

        # ---------------------------------------

        sf = effective_coord_sf(coords, data_type, args.coord_sf)
        coord_sf = sf["coord_sf"]
        train_epoch = args.train_epoch
        requested_epochs = train_epoch
        batch_size = args.batch_size
        learning_rate = args.learning_rate
        w_recon = args.w_recon if args.w_recon is not None else 0.1
        w_w = args.w_w if args.w_w is not None else 0.1
        w_l1 = args.w_l1 if args.w_l1 is not None else 0.1
        relu = args.relu

        # batch_size defaults
        if batch_size is None:
            if data_type == "10x":
                batch_size = 512
            elif data_type == "ST_KTH":
                batch_size = 64
            elif data_type == "Slide-seq":
                batch_size = 8192

        # learning_rate defaults
        if learning_rate is None:
            if data_type == "ST_KTH":
                learning_rate = 1e-4
            else:
                learning_rate = 1e-3

        # down_ratio default only matters for recovery
        if experiment == "recovery" and down_ratio is None:
            down_ratio = 0.5

        # The 3D tutorial trains for 1000 epochs; the portal default (10000) is read as "unset".
        if experiment == "3d_model" and train_epoch == 10000:
            train_epoch = 1000

        # Build save_path within output_dir
        save_path = os.path.join(output_dir, "stage_run")
        os.makedirs(save_path, exist_ok=True)
        log(f"save_path: {save_path}")

        # Construct keyword arguments for STAGE.STAGE
        stage_kwargs: dict[str, Any] = {
            "save_path": save_path,
            "data_type": data_type,
            "experiment": experiment,
            "coord_sf": coord_sf,
            "train_epoch": train_epoch,
            "seed": args.seed,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "w_recon": w_recon,
            "w_w": w_w,
            "w_l1": w_l1,
            "relu": relu,
            "device": device_str,
            # STAGE's scratch directory; its default ('file_tmp') is relative to the working directory.
            "path1": os.path.join(save_path, "file_tmp"),
        }

        # Extra arguments depending on experiment
        if experiment == "recovery":
            stage_kwargs["down_ratio"] = down_ratio
        if experiment == "3d_model":
            stage_kwargs["sec_name"] = sec_name
            stage_kwargs["select_section"] = select_section
            stage_kwargs["gap"] = gap

        log("Calling STAGE.STAGE with arguments:")
        for k, v in stage_kwargs.items():
            log(f"  {k} = {v}")

        before = snapshot_files(output_dir)
        # Upstream dataset.get_data writes 'input_data/used_gene.txt' relative to the working
        # directory whatever path1 says (generation and 3d_model pass it none). Run STAGE from
        # save_path so that file lands with the results instead of wherever the portal was started.
        try:
            prev_cwd = os.getcwd()
        except OSError:
            prev_cwd = None
        coords_before = np.array(adata.obsm["coord"], dtype=float, copy=True)
        cow = experiment == "generation" and pandas_has_copy_on_write()
        os.chdir(save_path)
        try:
            # stdout carries only the worker's JSON; upstream's progress goes to stderr.
            with copy_on_write(cow), contextlib.redirect_stdout(sys.stderr):
                result = STAGE.STAGE(adata, **stage_kwargs)
        except UnboundLocalError as exc:
            if experiment == "recovery" and "X_dim" in str(exc):
                raise RuntimeError(recovery_unavailable_msg(exc)) from exc
            raise
        finally:
            if prev_cwd is not None:
                os.chdir(prev_cwd)

        counts: dict[str, int] = {}
        if experiment == "recovery":
            adata_sample, adata_stage = result
            write_h5ad_atomic(adata_sample, os.path.join(save_path, "adata_sample.h5ad"))
            write_h5ad_atomic(adata_stage, os.path.join(save_path, "adata_stage.h5ad"))
            counts["n_spots_sampled"] = int(adata_sample.n_obs)
        elif experiment == "generation":
            adata_stage = result
            n_measured = int(np.unique(coords_before[:, :2], axis=0).shape[0])
            check_generation(adata, coords_before, int(adata_stage.n_obs), n_measured)
            write_h5ad_atomic(adata_stage, os.path.join(save_path, "adata_stage.h5ad"))
            counts["n_positions_generated"] = int(adata_stage.n_obs) - n_measured
        elif experiment == "3d_model":
            adata_stage, adata_simu, adata_all = result
            write_h5ad_atomic(adata_stage, os.path.join(save_path, "adata_stage.h5ad"))
            write_h5ad_atomic(adata_simu, os.path.join(save_path, "adata_simu.h5ad"))
            write_h5ad_atomic(adata_all, os.path.join(save_path, "adata_all.h5ad"))
            counts["n_spots_simulated"] = int(adata_simu.n_obs)
        else:
            raise ValueError(unsupported_choice_msg("experiment", experiment, ["generation", "recovery", "3d_model"]))

        # Summarize outputs: only what this run wrote.
        output_files = files_written_since(output_dir, before)
        n_preexisting = len([f for f in before if f not in set(output_files)])
        log(f"STAGE run completed. {len(output_files)} files written under output_dir.")

        out = WorkerOutput("stage", task=experiment)
        record_method(out, f"STAGE.STAGE (upstream STAGE, experiment={experiment}, data_type={data_type})")
        out.set_data(
            n_spots=n_obs_in,
            n_spots_input=int(n_spots_supplied),
            n_spots_used=n_obs_in,
            n_genes=n_vars_in,
            n_genes_used=hvg["n_genes_used"],
            n_spots_output=int(adata_stage.n_obs),
            **counts,
        )
        out.add_output_files({f: os.path.join(output_dir, f) for f in output_files})
        out.add_params(
            {
                "input_h5ad": st_h5ad,
                "data_type": data_type,
                "experiment": experiment,
                "save_path": save_path,
                "coord_sf": coord_sf,
                "coord_sf_source": sf["source"],
                "coord_source": coord_info["source"],
                "coord_nn_spacing": coord_info["nn_spacing"],
                "down_ratio": down_ratio,
                "train_epoch": train_epoch,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "w_recon": w_recon,
                "w_w": w_w,
                "w_l1": w_l1,
                "relu": relu,
                "device": device_str,
                "seed": args.seed,
                "sec_name": sec_name,
                "select_section": select_section,
                "gap": gap,
                "hvg_flavor": hvg["hvg_flavor"],
                "hvg_source": hvg["hvg_source"],
                "hvg_n_top_genes": hvg["hvg_n_top_genes"],
                "hvg_n_top_genes_tutorial": hvg["hvg_n_top_genes_tutorial"],
                "normalization": hvg["normalization"],
                "pandas_copy_on_write": cow,
                "dense_peak_estimate_gib": round(dense_estimate / float(1 << 30), 3),
                "use_raw_counts": bool(args.use_raw_counts),
            }
        )
        record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)
        record_expression_source(out, counts_info)
        if counts_note:
            out.add_warning(counts_note)
        if ignored:
            record_ignored(
                out,
                ignored,
                f"experiment={experiment!r} does not use them (down_ratio is for recovery; sec_name, "
                "select_section and gap are for 3d_model).",
            )
        if hvg["hvg_flavor"] is None and args.hvg_flavor != DEFAULT_HVG_FLAVOR:
            record_ignored(
                out, "hvg_flavor", "the input already carries var['highly_variable'], so no HVG selection ran."
            )
        for w in (sf["warning"], hvg["warning"], z_warning):
            if w:
                out.add_warning(w)
        if train_epoch != requested_epochs:
            out.add_note(
                f"train_epoch={requested_epochs} is the portal default and was read as unset; 3d_model used the "
                f"3D tutorial's {train_epoch}."
            )
        if n_preexisting:
            out.add_note(
                f"{n_preexisting} file(s) already under output_dir before this run were left alone and are not "
                "listed in output_files."
            )
        out.set_summary(
            n_output_files=len(output_files),
            data_type=data_type,
            experiment=experiment,
            train_epoch=train_epoch,
            device=device_str,
            n_genes_used=hvg["n_genes_used"],
            n_spots_output=int(adata_stage.n_obs),
        )
        if hvg["hvg_source"] == "supplied var['highly_variable']":
            genes = f"the {hvg['n_genes_used']} genes flagged in the supplied var['highly_variable']"
        elif hvg["hvg_n_top_genes"] < hvg["hvg_n_top_genes_tutorial"]:
            genes = (
                f"{hvg['n_genes_used']} of {n_vars_in} genes: the input has fewer genes than the "
                f"{hvg['hvg_n_top_genes_tutorial']} highly variable genes the tutorial selects, so "
                f"{hvg['hvg_flavor']} was asked for all {hvg['hvg_n_top_genes']}"
            )
        else:
            genes = (
                f"{hvg['n_genes_used']} highly variable genes ({hvg['hvg_flavor']}, top {hvg['hvg_n_top_genes']}) "
                f"out of {n_vars_in}"
            )
        if experiment == "generation":
            produced = (
                f"adata_stage.h5ad holds {int(adata_stage.n_obs)} positions: the measured spots plus "
                f"{counts['n_positions_generated']} generated between them, in the units of obsm['coord']."
            )
        elif experiment == "recovery":
            produced = (
                f"STAGE trained on {counts['n_spots_sampled']} down-sampled spots (down_ratio={down_ratio}) and "
                f"predicted all {int(adata_stage.n_obs)} spots in adata_stage.h5ad."
            )
        else:
            produced = (
                f"adata_stage.h5ad holds {int(adata_stage.n_obs)} beads of the real sections and adata_simu.h5ad "
                f"{counts['n_spots_simulated']} beads of simulated sections (trained on sections {select_section})."
            )
        if n_spots_off_tissue:
            spots = (
                f"{n_obs_in} in-tissue spots x {n_vars_in} genes ({n_spots_off_tissue} of the {n_spots_supplied} "
                "spots supplied have obs['in_tissue'] == 0, background outside the tissue, and were left out)"
            )
        else:
            spots = f"{n_obs_in} spots x {n_vars_in} genes"
        if counts_info["expression_source"] != "X":
            spots += ", read from adata.raw (use_raw_counts=True)"
        out.set_analysis(
            f"STAGE {experiment} experiment completed for {data_type} data. "
            f"Input: {spots}; STAGE modelled {genes}. "
            f"Coordinates: {coord_info['source']}, coord_sf={coord_sf:g} ({sf['source']}). "
            f"{produced} Trained for {train_epoch} epochs. "
            f"This run wrote {len(output_files)} files under {output_dir}."
        )
        out.emit()

    except Exception as e:
        tb = traceback.format_exc()
        log("EXCEPTION during STAGE worker run:")
        log(tb)
        WorkerOutput.emit_error("stage", str(e), task=experiment)
        sys.exit(1)


if __name__ == "__main__":
    main()
