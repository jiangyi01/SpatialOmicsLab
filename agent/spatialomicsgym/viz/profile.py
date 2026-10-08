"""What a dataset is, decided once, before anything is drawn.

Every plot in this toolkit asks the same questions -- is there a coordinate system, is there an
image and a scalefactor to go with it, which obs columns are categorical, does X hold counts or
something already transformed, which upstream results are present and *complete enough to draw*
-- and the wrong way to answer them is for each plot to open the file and guess. This module
answers them once, from one backed read, and returns a dict the capability registry evaluates
predicates against.

Three properties, each of which is a decision rather than an implementation detail.

**It reuses the repository's existing probes rather than adding a second opinion.** The
counts-versus-normalized question is ``data_validation._has_raw_counts``'s spread sample; the sparsity
question is ``tuning.adaptive._measure_sparsity``; the cell-type column is ``_find_cell_type_column``
given the obs frame, and the coordinate columns are the readiness check's alias table. Each of those has a
recorded bug history -- a backed sparse
matrix rejects fancy indexing, an unsorted sample index never once succeeded on a backed file --
and a fresh heuristic here would re-earn every one of them. The sample and the tables are read from
``spatialomicsgym.utils.obs_aliases``, where they moved (2026-10-01) so that reading them does not
import the agent package; ``data_validation`` re-exports the same objects.

**"Could not tell" is a third answer, not a synonym for no.** ``_has_raw_counts`` deliberately
answers ``True`` when its sample was empty or the object was too large to sample safely, because
it is a *gate* and an unverifiable check must not block a tool. For a figure the direction
inverts: a plot drawn from a matrix nobody characterised, captioned as if it had been, is worse
than no plot. So ``matrix.integral`` is tri-state -- ``True``, ``False`` or ``None`` -- and
``None`` forces a caption clause rather than being rendered as a yes.

**Nothing here costs more than a bounded read.** No full-matrix statistic, no PCA, no per-gene
mean over all genes. The largest read is a spread sample of a few hundred rows, and the file is
opened exactly once.

The module imports nothing heavy at module scope: it is read by the capability registry, which is
read by a portal, and the portal must stay importable in an environment that has no anndata.
"""

from __future__ import annotations

import os
from typing import Any

SCHEMA = "sog.viz_profile/1"

#: A categorical obs column with more levels than this is reported with its count only. Sixty is
#: not arbitrary: it is the number of distinct colours the shared categorical palette can give
#: before it starts pooling, so a column above it cannot be drawn as a legend anyway.
DEFAULT_MAX_CATEGORIES = 200

#: Rows of X read for the counts probe. The probe reads five spread blocks rather than one prefix,
#: because the first rows of a spatial object are one corner of the tissue.
DEFAULT_SAMPLE_SPOTS = 5000

#: Above this many observations a scatter is rasterized inside a vector export. Measured, not
#: guessed: 120k points is a 16.9 MB SVG unrasterized and 3.66 MB rasterized, and the report
#: renderer refuses a figure above 6 MB.
RASTERIZE_ABOVE = 50_000

#: Above this many observations the core path draws a density summary or a disclosed subsample
#: rather than one mark per observation.
SUBSAMPLE_ABOVE = 300_000

#: obs column names that name a cluster or a spatial domain rather than a curated cell type. Kept
#: separate from ``obs_aliases._CELL_TYPE_ALIASES``, which answers a different question: a
#: cluster id is a partition, a cell type is a claim about biology, and a plot that labels the
#: first as the second is the cheapest way to overstate a result.
_CLUSTER_HINTS = ("spatial_domain", "domain", "cluster", "clusters", "leiden", "louvain", "kmeans")

#: The coordinate frames a before/after figure reads, in the order ``plot_alignment_qc`` tries them.
#: One table for the catalogue and the producer: the profile counted ANY two of 'spatial' and the
#: spatial_3d*/spatial_aligned*/moscot_spatial* keys, so a STARmap object (spatial + spatial_3d) or a
#: Phase-1 stack (spatial + spatial_3d_raw) was offered a before/after the producer then refused --
#: it has a "before" and no "after" (hunt 2026-09-30, u20b-viz-rest-8).
BEFORE_FRAME_KEYS = ("spatial_3d_raw", "spatial")
AFTER_FRAME_KEYS = ("spatial_3d_aligned", "spatial_aligned", "moscot_spatial_warp")

#: obs columns that name a section, in the order they are tried -- ``layers.SECTION_COLUMNS``'s list,
#: kept here because this module imports nothing at module scope.
_SECTION_COLUMNS = (
    "library_id",
    "slice_id",
    "section",
    "section_id",
    "brain_section_label",
    "Bregma",
    "z_index",
    "batch",
)


def frame_pair(frames: list[str] | tuple[str, ...]) -> tuple[str, str]:
    """``(before, after)`` frame keys among *frames*, by the shared table; ``""`` where absent."""
    have = {str(f) for f in frames or ()}
    before = next((k for k in BEFORE_FRAME_KEYS if k in have), "")
    after = next((k for k in AFTER_FRAME_KEYS if k in have), "")
    return before, after


def coords_look_like_array_indices(xy: Any) -> bool:
    """Small, non-negative, whole coordinates are an array index -- the Visium lattice -- not a position.

    One rule, read by the profile and by the tissue renderer, which labels such axes as indices.
    """
    import numpy as np

    try:
        arr = np.asarray(xy, dtype=float)
    except Exception:
        return False
    if arr.ndim != 2 or arr.shape[1] < 2:
        return False
    xy2 = arr[:, :2]
    finite = xy2[np.isfinite(xy2).all(axis=1)]
    if not len(finite):
        return False
    whole = bool(np.allclose(finite, np.round(finite)))
    small = bool(finite.max() < 1000 and finite.min() >= 0)
    return bool(whole and small)


#: Keys under ``uns`` that mean an analysis has already run. The value is what a plot needs from
#: it; presence alone is never enough, which is the point of ``_rank_genes_fields``.
_ANALYSIS_KEYS = (
    "rank_genes_groups",
    "neighbors",
    "pca",
    "paga",
    "moranI",
    "nhood_enrichment",
    "co_occurrence",
    "ripley_L",
    "ripley_F",
    "ripley_G",
)
#: The analyses squidpy stores under ``<cluster_key>_<analysis>`` (``annotation_nhood_enrichment``), so the bare key
#: is never there. Only these accept a prefix: ``pca_neighbors`` is not a neighbour graph.
_PREFIXED = ("nhood_enrichment", "co_occurrence", "ripley_L", "ripley_F", "ripley_G")


def _stat(path: str) -> tuple[int, int]:
    try:
        st = os.stat(path)
    except OSError:
        return (0, 0)
    return (int(st.st_size), int(st.st_mtime_ns))


def fingerprint(path: str, n_obs: int, n_vars: int) -> str:
    """A change detector for one dataset, not a content hash.

    Sixteen hexadecimal characters, deliberately: the portal's redactor masks any hexadecimal run
    of thirty-two or more, so a sha256 here would reach the reader as ``[redacted]`` and the
    revision tool would then report that the figure's dataset is gone.

    Hashing the bytes of a two-gigabyte object on every plot is not acceptable, and it is not
    needed. Size, modification time and shape together catch every case that matters -- the file
    was replaced, re-analysed, truncated or regenerated -- at the cost of one ``stat``.
    """
    import hashlib

    size, mtime = _stat(path)
    seed = f"{os.path.realpath(path)}|{size}|{mtime}|{n_obs}|{n_vars}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def _integral_tristate(adata: Any) -> tuple[bool | None, int, tuple[float, float] | None]:
    """Does X hold whole numbers? ``True`` / ``False`` / ``None`` for "the sample proved nothing".

    Returns the answer, how many nonzero values were examined, and their range. The count is
    reported because a caption that says "treated as counts" when nothing was sampled is exactly
    the claim this tri-state exists to stop.
    """
    try:
        import numpy as np

        from spatialomicsgym.utils.obs_aliases import _nonzero_value_sample
    except Exception:
        return (None, 0, None)
    try:
        values = _nonzero_value_sample(adata)
    except Exception:
        return (None, 0, None)
    if values is None:
        return (None, 0, None)
    try:
        n = int(getattr(values, "size", 0))
    except Exception:
        return (None, 0, None)
    if n == 0:
        # An empty sample is the case `_has_raw_counts` answers True to on purpose, because it is
        # a gate. Here it is the honest None.
        return (None, 0, None)
    try:
        lo, hi = float(np.min(values)), float(np.max(values))
        integral = bool(np.allclose(values, np.round(values)))
    except Exception:
        return (None, n, None)
    return (integral, n, (lo, hi))


def column_facts(series: Any, name: str, max_categories: int) -> dict[str, Any]:
    """One obs column, classified. Categorical and numeric are answered differently on purpose.

    Public because the explorer classifies obs columns by this same rule, so the panel and the
    figures cannot disagree about which columns draw as categories.
    """
    import numpy as np
    import pandas as pd

    out: dict[str, Any] = {"name": name, "dtype": str(series.dtype)}
    try:
        na_fraction = float(pd.isna(series).mean())
    except Exception:
        na_fraction = 0.0
    out["na_fraction"] = round(na_fraction, 6)

    is_categorical = bool(
        isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object or str(series.dtype) == "bool"
    )
    if not is_categorical and pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        out["role"] = "numeric"
        if len(values):
            out["min"] = float(values.min())
            out["q50"] = float(values.median())
            out["max"] = float(values.max())
            out["constant"] = bool(values.min() == values.max())
        else:
            out["constant"] = True
        # An integer-looking column with few distinct values is a cluster id stored as a number.
        # Saying so is what stops a domain map being drawn with a continuous colourbar.
        try:
            distinct = int(values.nunique())
        except Exception:
            distinct = -1
        out["n_distinct"] = distinct
        out["looks_like_labels"] = bool(
            0 < distinct <= 64 and np.allclose(values.to_numpy(), np.round(values.to_numpy()))
        )
        # The draw rule. ``layers.obs_values`` paints an integral column with 64 or fewer levels as
        # categories, so the catalogue must count it the same way: an int64 ``spatial_domain`` was
        # 'numeric' here and its domain map was refused for want of a categorical column (hunt
        # 2026-09-30, u20b-viz-rest-17). ``role`` stays the storage type; this is how it draws.
        out["draws_as"] = "categorical" if out["looks_like_labels"] else "numeric"
        out["n_levels"] = distinct
        lowered = name.lower()
        out["looks_like_cluster"] = bool(out["looks_like_labels"] and any(h in lowered for h in _CLUSTER_HINTS))
        return out

    out["role"] = "categorical"
    out["draws_as"] = "categorical"
    try:
        levels = list(pd.Series(series).astype("object").dropna().unique())
    except Exception:
        levels = []
    out["n_levels"] = len(levels)
    if len(levels) <= max_categories:
        out["levels"] = [str(v) for v in levels]
        out["truncated"] = False
    else:
        out["levels"] = []
        out["truncated"] = True
    lowered = name.lower()
    out["looks_like_cluster"] = any(h in lowered for h in _CLUSTER_HINTS)
    return out


def _rank_genes_fields(uns: Any) -> dict[str, Any]:
    """Which fields a stored differential-expression result actually carries.

    Presence of ``rank_genes_groups`` is not the question. A volcano needs fold changes on one
    axis and corrected p-values on the other, and a real, shipped, processed dataset in this
    repository stores only ``names`` and ``scores`` -- the Wilcoxon z statistic, which is not an
    effect size. Reporting the field set is what lets a plot refuse with a reason instead of
    plotting the statistic it happens to find.
    """
    out: dict[str, Any] = {"present": False, "fields": [], "groups": [], "groupby": "", "params": {}}
    try:
        block = uns["rank_genes_groups"]
    except Exception:
        return out
    if block is None:
        return out
    out["present"] = True
    try:
        out["fields"] = sorted(str(k) for k in block.keys() if str(k) != "params")
    except Exception:
        out["fields"] = []
    try:
        names = block["names"]
        out["groups"] = [str(g) for g in (names.dtype.names or ())]
    except Exception:
        out["groups"] = []
    try:
        params = dict(block.get("params", {}) or {})
        out["groupby"] = str(params.get("groupby", ""))
        out["params"] = {str(k): str(v) for k, v in params.items()}
    except Exception:
        pass
    return out


def _spatial_facts(adata: Any) -> dict[str, Any]:
    """Coordinates, units, and whether an image can actually be registered against them."""
    import numpy as np

    out: dict[str, Any] = {
        "has_obsm_spatial": False,
        "obs_coord_columns": None,
        "n_dims": 0,
        "bbox": None,
        "span": None,
        "coords_look_like_array_indices": False,
        "coords_3d_keys": [],
        "coordinate_frames": [],
        "section_candidates": [],
        "section_levels": 0,
        "spatial_3d_provenance": {},
        "z_column": "",
        "section_z_buildable": False,
        "z_span": None,
        "z_unique": 0,
        "z_is_layered": False,
        "library_ids": [],
        "image_keys": [],
        "image_shapes": {},
        "scalefactors": {},
        "overlay_ready": False,
        "spot_diameter_fullres": None,
    }
    # Multi-slice facts. Added 2026-09-22: `library_ids` below is derived from the keys of
    # uns['spatial'], which a MERGED serial-section object does not have at all -- so on every
    # dataset the 3D work is about, that list is empty and anything keyed off it silently draws
    # nothing. The section axis on real data is an obs column.
    try:
        import numpy as _np

        from spatialomicsgym.viz.layers import is_coordinate_key, z_column

        three_col = []
        for key in list(getattr(adata, "obsm", {}) or {}):
            # A key that NAMES coordinates. Every three-column obsm counted, so a 2D Visium with a
            # 3-component UMAP was offered four 3D plots, drawn from the UMAP (hunt 2026-09-30,
            # u20b-viz-rest-6). ``layers.spatial_coords_3d`` reads by the same rule.
            if not is_coordinate_key(str(key), adata):
                continue
            try:
                arr = _np.asarray(adata.obsm[key])
            except Exception:
                continue
            if arr.ndim == 2 and arr.shape[1] == 3:
                three_col.append(str(key))
        out["coords_3d_keys"] = sorted(three_col)
        # A before/after figure needs TWO frames, and "has 3D coordinates" does not imply it: an
        # aligner that overwrote obsm['spatial'] leaves a perfectly good 3D stack and nothing to
        # compare it against. Counted here so the catalogue can refuse the row rather than offer a
        # plot whose producer will refuse -- which is the catalogue lying, not the producer.
        frames = [
            key
            for key in list(getattr(adata, "obsm", {}) or {})
            if str(key) == "spatial" or str(key).startswith(("spatial_3d", "spatial_aligned", "moscot_spatial"))
        ]
        out["coordinate_frames"] = sorted(str(k) for k in frames)
        before, after = frame_pair(out["coordinate_frames"])
        out["frame_pair"] = {"before": before, "after": after}

        block = adata.uns.get("spatial_3d")
        out["spatial_3d_provenance"] = dict(block) if isinstance(block, dict) else {}

        # The contract's own slice column first, as ``layers.section_labels`` reads it: a stack
        # whose sections live in a column the fixed list does not name was refused before the
        # producer that would have honoured it (hunt 2026-09-30, u20b-viz-rest-16).
        names = list(_SECTION_COLUMNS)
        recorded_key = str(block.get("slice_key") or "") if isinstance(block, dict) else ""
        if recorded_key:
            names = [recorded_key, *[n for n in names if n != recorded_key]]
        candidates = []
        for column in names:
            if column in adata.obs.columns:
                levels = int(adata.obs[column].astype(str).nunique())
                if 2 <= levels <= 200:
                    candidates.append({"column": column, "n_levels": levels})
            if len(candidates) >= 5:
                break
        out["section_candidates"] = candidates
        out["section_levels"] = candidates[0]["n_levels"] if candidates else 0
        # Whether ``layers.spatial_coords_3d`` can build a z from the section axis without being
        # told a spacing: only a RECORDED spacing. Numeric section labels counted too, so a concat of
        # two 2D samples -- batch "0"/"1", DLPFC library_id "151507"/"151673" -- was offered three 3D
        # plots and drawn as a stack at those values (hunt 2026-09-30, u20b-viz-rest-7, the review of
        # its first fix). A position column (Bregma, slice_z, z, a recorded z_key) is the z below.
        recorded = out["spatial_3d_provenance"].get("z_spacing")
        out["section_z_buildable"] = bool(
            candidates and isinstance(recorded, (int, float)) and not isinstance(recorded, bool) and recorded > 0
        )

        # The z column by ``layers.z_column``'s rule, the one the 3D reader builds its z from. Only
        # obs 'slice_z'/'z' were read here, so a Bregma stack was refused (u20b-viz-rest-7).
        z_source = None
        z_key = z_column(adata)
        if z_key:
            try:
                z_source = _np.asarray(adata.obs[z_key].astype(str), dtype=float)
                out["z_column"] = z_key
            except Exception:
                z_source = None
        if z_source is not None and len(z_source):
            planes = sorted({float(v) for v in z_source if v == v})
            out["z_span"] = [float(min(planes)), float(max(planes))] if planes else None
            out["z_unique"] = len(planes)
            out["z_is_layered"] = bool(1 < len(planes) <= 500)
    except Exception:
        pass

    coords = None
    try:
        if "spatial" in adata.obsm:
            coords = np.asarray(adata.obsm["spatial"], dtype=float)
            out["has_obsm_spatial"] = True
    except Exception:
        coords = None
    if coords is None:
        try:
            from spatialomicsgym.utils.obs_aliases import _find_spatial_coord_columns

            pair = _find_spatial_coord_columns(list(adata.obs.columns))
        except Exception:
            pair = None
        if pair:
            out["obs_coord_columns"] = list(pair)
            try:
                coords = np.column_stack(
                    [adata.obs[pair[0]].to_numpy(dtype=float), adata.obs[pair[1]].to_numpy(dtype=float)]
                )
            except Exception:
                coords = None
    if coords is not None and coords.ndim == 2 and coords.shape[1] >= 2:
        out["n_dims"] = int(coords.shape[1])
        xy = coords[:, :2]
        finite = xy[np.isfinite(xy).all(axis=1)]
        if len(finite):
            lo = finite.min(axis=0)
            hi = finite.max(axis=0)
            out["bbox"] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
            out["span"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
            # A coordinate pair that is small, non-negative and whole is an array index -- the
            # Visium row/column lattice -- not a pixel position. Pairing it with a scalefactor
            # produces a tissue map that is a hundred and seventy times too small and still
            # renders, which is the single hardest spatial defect to see by eye.
            try:
                out["coords_look_like_array_indices"] = coords_look_like_array_indices(finite)
            except Exception:
                pass

    # `uns['spatial']` is a mapping in a Space Ranger object and an arbitrary array in an object
    # some agent wrote. An unguarded `.items()` on the second shape has already raised out of the
    # recommender once, which is why this is an isinstance check and not a try.
    spatial_uns = None
    try:
        spatial_uns = adata.uns.get("spatial")
    except Exception:
        spatial_uns = None
    if isinstance(spatial_uns, dict):
        out["library_ids"] = [str(k) for k in spatial_uns.keys()]
        first = spatial_uns.get(out["library_ids"][0]) if out["library_ids"] else None
        if isinstance(first, dict):
            images = first.get("images")
            if isinstance(images, dict):
                out["image_keys"] = [str(k) for k in images.keys()]
                for key, img in images.items():
                    shape = getattr(img, "shape", None)
                    if shape is not None:
                        out["image_shapes"][str(key)] = [int(v) for v in shape]
            factors = first.get("scalefactors")
            if isinstance(factors, dict):
                out["scalefactors"] = {str(k): float(v) for k, v in factors.items() if isinstance(v, (int, float))}
                diameter = out["scalefactors"].get("spot_diameter_fullres")
                out["spot_diameter_fullres"] = float(diameter) if diameter else None

    # An image is only usable if its own scalefactor is there too. They are chosen as a pair or
    # not at all: hires with the lowres factor is a 3.3x error that leaves every spot inside the
    # frame, just huddled in one corner.
    for key in out["image_keys"]:
        if f"tissue_{key}_scalef" in out["scalefactors"]:
            out["overlay_ready"] = bool(out["has_obsm_spatial"] and not out["coords_look_like_array_indices"])
            break
    return out


def obsm_facts(adata: Any) -> list[dict[str, Any]]:
    """Embeddings, score matrices and proportion matrices, told apart by shape and content.

    Public for the same reason as :func:`column_facts`: the explorer reads obsm by this rule.
    """
    import numpy as np

    facts: list[dict[str, Any]] = []
    try:
        keys = list(adata.obsm.keys())
    except Exception:
        return facts
    for key in keys:
        try:
            value = adata.obsm[key]
            shape = tuple(int(v) for v in getattr(value, "shape", ()))
        except Exception:
            continue
        entry: dict[str, Any] = {"key": str(key), "shape": list(shape), "family": "other", "columns": None}
        if len(shape) == 2:
            name = str(key).lower()
            if name.startswith("x_") or name in ("umap", "tsne", "pca", "spatial"):
                entry["family"] = "embedding" if name != "spatial" else "coordinates"
            columns = None
            try:
                columns = [str(c) for c in value.columns]  # a DataFrame-backed obsm
            except Exception:
                columns = None
            entry["columns"] = columns
            # Rows that sum to one are proportions -- a deconvolution result -- and that is what
            # makes a per-cell-type map and a dominant-type map drawable without being told.
            if shape[1] >= 2 and shape[0] > 0:
                try:
                    block = np.asarray(value[: min(200, shape[0])], dtype=float)
                    sums = block.sum(axis=1)
                    if np.all(block >= -1e-9) and np.allclose(sums, 1.0, atol=1e-3):
                        entry["family"] = "proportions"
                        entry["rows_sum_to_one"] = True
                except Exception:
                    pass
        facts.append(entry)
    return facts


#: The names these had before they were public; kept, so nothing that imported them breaks.
_column_facts = column_facts
_obsm_facts = obsm_facts


def profile_for_viz(
    data_path: str,
    *,
    obs_keys: list[str] | tuple[str, ...] = (),
    var_names: list[str] | tuple[str, ...] = (),
    max_categories: int = DEFAULT_MAX_CATEGORIES,
    sample_spots: int = DEFAULT_SAMPLE_SPOTS,
) -> dict[str, Any]:
    """Everything a plot needs to know about one dataset, from one backed read.

    Never raises. An unreadable file comes back as a profile whose ``dataset.readable`` is false
    and whose ``read_error`` is a sentence, because a capability report that crashes tells the
    agent less than one that says the file could not be opened.
    """
    from spatialomicsgym.utils.file_io import read_h5ad_backed

    profile: dict[str, Any] = {
        "schema": SCHEMA,
        "dataset": {"path": os.path.realpath(data_path), "readable": False, "read_error": ""},
        "warnings": [],
    }
    try:
        with read_h5ad_backed(data_path) as adata:
            _fill(profile, adata, list(obs_keys), list(var_names), max_categories, sample_spots)
    except Exception as exc:
        profile["dataset"]["read_error"] = f"{type(exc).__name__}: {exc}"
        return profile
    return profile


def _fill(
    profile: dict[str, Any],
    adata: Any,
    obs_keys: list[str],
    var_names: list[str],
    max_categories: int,
    sample_spots: int,
) -> None:
    n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    size, _mtime = _stat(profile["dataset"]["path"])
    profile["dataset"].update(
        {
            "readable": True,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "size_bytes": size,
            "fingerprint": fingerprint(profile["dataset"]["path"], n_obs, n_vars),
            "estimated_dense_gb": round(n_obs * n_vars * 4 / 1024**3, 4),
        }
    )

    integral, n_sampled, value_range = (None, 0, None)
    if sample_spots > 0:
        integral, n_sampled, value_range = _integral_tristate(adata)
    sparsity = None
    # ``sample_spots=0`` is documented as "reads no expression data at all", and the sparsity probe
    # still read a window of X (hunt 2026-09-30, u20b-viz-rest-30). Zero now means zero.
    if sample_spots > 0:
        try:
            from spatialomicsgym.tuning.adaptive import _measure_sparsity

            sparsity = _measure_sparsity(adata)
        except Exception:
            sparsity = None
    try:
        layers = [str(k) for k in adata.layers.keys()]
    except Exception:
        layers = []
    profile["matrix"] = {
        "dtype": str(getattr(adata.X, "dtype", "")),
        "sparsity": None if sparsity is None else round(float(sparsity), 6),
        "sparsity_measured": sparsity is not None,
        "integral": integral,
        "n_sampled": n_sampled,
        "value_range": list(value_range) if value_range else None,
        "has_negative": bool(value_range is not None and value_range[0] < 0),
        "layers": layers,
        "has_raw": adata.raw is not None,
    }
    if integral is None:
        profile["warnings"].append(
            "Could not establish whether the matrix holds counts: "
            f"{n_sampled} non-zero values were sampled. Any figure drawn from it says so."
        )
    if profile["matrix"]["has_negative"]:
        profile["warnings"].append(
            "The matrix holds negative values, so it has been scaled or z-scored. Colour on it is "
            "standard deviations, not expression; a counts layer is the right source for a "
            "magnitude plot."
        )

    columns = list(adata.obs.columns)
    # Roles are decided over EVERY column; ``obs_keys`` only narrows which columns are listed in
    # full. Deriving the roles from the narrowed list made inspect_dataset(obs_keys='total_counts')
    # report a dataset with a 'leiden' column as unable to draw any categorical plot (hunt
    # 2026-09-30, u20b-viz-rest-18).
    facts = [column_facts(adata.obs[c], str(c), max_categories) for c in columns]
    wanted = {str(c) for c in obs_keys if c in columns} if obs_keys else None
    profile["obs"] = [f for f in facts if wanted is None or f["name"] in wanted]
    profile["n_obs_columns"] = len(columns)
    profile["obs_index"] = {
        f["name"]: {"role": f["role"], "draws_as": f.get("draws_as", f["role"]), "n_levels": f.get("n_levels", 0)}
        for f in facts
    }

    # A column draws as categorical by ``layers.obs_values``' rule -- an integral number column with
    # 64 or fewer levels included -- and the roles below follow it (u20b-viz-rest-17).
    categorical = [c for c in facts if c.get("draws_as") == "categorical" and 1 < c.get("n_levels", 0)]
    numeric = [c for c in facts if c.get("draws_as") == "numeric" and not c.get("constant")]
    numeric_stored = [c for c in facts if c.get("role") == "numeric" and not c.get("constant")]
    try:
        from spatialomicsgym.utils.obs_aliases import _find_cell_type_column

        # The obs frame, not just its names: an exact alias still decides alone, but the Xenium
        # kidney file keeps its labels in obs['celltype_plot'], which no alias spells, and every
        # cell-type panel was missing from its overview until the values could be read as well.
        cell_type = _find_cell_type_column(columns, adata.obs)
    except Exception:
        cell_type = None
    cluster = next((c["name"] for c in categorical if c.get("looks_like_cluster")), None)
    profile["obs_roles"] = {
        "cell_type": cell_type,
        "cluster": cluster,
        "categorical": [c["name"] for c in categorical],
        "numeric": [c["name"] for c in numeric],
        # By name over every stored number: a QC metric on a small panel can have few enough distinct
        # values to draw as categories, and it is still the QC metric.
        "qc": [
            c["name"] for c in numeric_stored if c["name"] in ("total_counts", "n_genes_by_counts", "pct_counts_mt")
        ],
        "over_palette": [c["name"] for c in categorical if c.get("n_levels", 0) > 60],
    }

    profile["spatial"] = _spatial_facts(adata)
    if profile["spatial"].get("coords_look_like_array_indices"):
        # Tissue maps are drawn on these now rather than refused (a policy decision, hunt
        # 2026-09-30, u20b-viz-rest-26), so every figure from this object carries the caveat.
        profile["warnings"].append(
            "The coordinates look like array row and column indices, not positions. Tissue maps are "
            "drawn on that index lattice with their axes labelled as indices; on a Visium array one "
            "row step is about 1.73 column steps, so the section appears compressed along its rows. "
            "A histology overlay is refused on them, because no scalefactor applies to an index."
        )
    profile["obsm"] = obsm_facts(adata)
    profile["obsp"] = sorted(str(k) for k in getattr(adata, "obsp", {}) or {})

    # The SAMPLE answers "what do these names look like"; the FULL index answers "how many are
    # mitochondrial". Sampling the second gave zero on a real Visium file, because its MT- genes
    # sort past the first two thousand rows -- and a QC report that says there are no
    # mitochondrial genes is worse than one that does not mention them.
    all_names = [str(v) for v in adata.var_names]
    index = all_names[: min(2000, n_vars)]
    ensembl = sum(1 for v in index if v.upper().startswith("ENS"))
    upper = sum(1 for v in index if v.isupper())
    profile["var"] = {
        "index_looks_like": "ensembl" if index and ensembl > len(index) * 0.5 else "symbol",
        "unique": bool(adata.var_names.is_unique),
        "n_duplicates": int(n_vars - len(set(all_names))),
        "species_guess": "human" if index and upper > len(index) * 0.5 else "mouse_or_other",
        "columns": [str(c) for c in adata.var.columns],
        "n_mito": sum(1 for v in all_names if v.upper().startswith(("MT-", "MT."))),
        "n_ribo": sum(1 for v in all_names if v.upper().startswith(("RPS", "RPL"))),
    }
    if not profile["var"]["unique"]:
        profile["warnings"].append(
            f"{profile['var']['n_duplicates']} gene symbols appear more than once, so a symbol "
            "does not identify one row. A plot asked for such a symbol reports the ambiguity "
            "rather than taking the first match."
        )
    if var_names:
        present: dict[str, bool] = {}
        lookup = {str(v).upper(): str(v) for v in adata.var_names}
        # .raw has its own, usually longer, gene list -- the standard object keeps every gene there
        # and only the highly variable ones in var -- and a figure can read it, so a gene that is
        # only in raw is present (hunt 2026-09-30, u20b-viz-rest-1).
        try:
            raw = adata.raw
            if raw is not None:
                lookup.update({str(v).upper(): str(v) for v in raw.var_names})
        except Exception:
            pass
        for name in var_names:
            present[str(name)] = str(name).upper() in lookup
        profile["var"]["requested_present"] = present

    uns_keys = []
    try:
        uns_keys = [str(k) for k in adata.uns.keys()]
    except Exception:
        pass
    profile["uns_analyses"] = {
        key: key in uns_keys or (key in _PREFIXED and any(k.endswith("_" + key) for k in uns_keys))
        for key in _ANALYSIS_KEYS
    }
    profile["uns_analyses"]["rank_genes_groups_detail"] = _rank_genes_fields(adata.uns)

    spatial = profile["spatial"]
    modality = "spatial" if (spatial["has_obsm_spatial"] or spatial["obs_coord_columns"]) else "single_cell"
    span = max(spatial["span"]) if spatial.get("span") else 0.0
    profile["derived"] = {
        "modality": modality,
        "platform": _platform_guess(profile),
        "default_point_size": round(float(max(2.0, min(40.0, 6000.0 / max(1, n_obs)))), 2),
        "needs_rasterize": n_obs > RASTERIZE_ABOVE,
        "needs_subsample": n_obs > SUBSAMPLE_ABOVE,
        "suggested_layer": _suggested_layer(profile),
        "coordinate_span": round(float(span), 2),
    }


def _suggested_layer(profile: dict[str, Any]) -> str:
    """Which slot a magnitude plot should read, given what the matrix turned out to be.

    Never silent: whatever this returns is recorded in the figure's spec and printed in its
    caption, so a reader can see that the counts layer was used rather than the scaled X.
    """
    matrix = profile["matrix"]
    if matrix["has_negative"] and "counts" in matrix["layers"]:
        return "counts"
    if matrix["integral"] is True:
        return "X (counts; normalized on read)"
    if matrix["has_negative"]:
        return "X (scaled - refuse a magnitude plot)"
    return "X"


def _platform_guess(profile: dict[str, Any]) -> dict[str, Any]:
    """A platform, and the evidence for it.

    Two rules already exist in this repository and they disagree by construction -- one reads the
    ``uns`` key, the other reads the image and the gene count. Reporting both pieces of evidence
    beside the answer is what stops a capability predicate silently deciding on the loser.
    """
    evidence: list[str] = []
    spatial = profile["spatial"]
    n_vars = profile["dataset"].get("n_vars", 0)
    guess = "unknown"
    for lib in spatial.get("library_ids", []):
        lowered = lib.lower()
        for name in ("visium", "xenium", "merfish", "slideseq", "stereo", "cosmx"):
            if name in lowered:
                guess = name
                evidence.append(f"the spatial library id contains {name!r}")
                break
    if guess == "unknown" and spatial.get("image_keys") and n_vars > 10000:
        guess = "visium"
        evidence.append("an embedded histology image with a transcriptome-wide gene count")
    if guess == "unknown" and spatial.get("has_obsm_spatial") and 0 < n_vars < 2000:
        guess = "imaging_based"
        evidence.append(f"coordinates with a {n_vars}-gene panel, which is an imaging-based platform")
    return {"name": guess, "evidence": evidence}
