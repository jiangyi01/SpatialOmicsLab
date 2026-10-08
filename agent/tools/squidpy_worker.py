#!/usr/bin/env python3
"""
Squidpy worker: spatial graph analysis for spatial transcriptomics.

- Runs inside /opt/conda/envs/moscot (override with SQUIDPY_PYTHON env var)
- All logs/progress go to stderr.
- stdout is reserved for a single final JSON line.

Supported tasks (via --task):
  spatial_neighbors   - Build spatial neighborhood graph
  nhood_enrichment    - Neighborhood enrichment analysis
  co_occurrence       - Cell type co-occurrence
  spatial_autocorr    - Spatial autocorrelation (Moran's I / Geary's C)
  ripley              - Ripley's spatial statistics (F, G, L)
  centrality_scores   - Graph centrality metrics

Every task leaves out the spots that ``obs['in_tissue']`` marks 0 before it runs. CELLxGENE Visium
exports carry every array spot, and on the library's samples 56-70% of them are background glass with
real ambient counts: a graph, a label statistic or a Moran's I over them measures the tissue-versus-
background contrast, not the tissue. The cut is reported (params.in_tissue_filter, a warning and a
NOTE in the analysis), and every count in the payload is after it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    describe_reduction,
    drop_unlabeled,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    per_section,
    record_ignored,
    record_in_tissue,
    spatial_frame,
)

#: The obsm key the worker hands squidpy: the coordinates :func:`worker_utils.spatial_frame` returned
#: (micrometres in 3D, and in 2D whenever the key declares its units). It is scratch, dropped before any
#: h5ad is written, so an annotated output holds exactly the coordinate keys its input held.
SOG_UM = "_sog_um"

#: ``params.mode`` values. "2d" is a plain one-section run; the other two are the program's vocabulary.
MODE_3D = "3d"
MODE_PER_SECTION = "per-section-2d"
MODE_2D = "2d"


def log(msg: str) -> None:
    """Print log messages to stderr with a prefix."""
    print(f"[squidpy-worker] {msg}", file=sys.stderr)


def _coords_key(args) -> str:
    """The obsm key the run reads: ``--coords-key`` when given, else the older ``--spatial-key``.

    Both name the same thing; when a caller sets both to different keys the run stops rather than
    picking one in silence.
    """
    spatial_key = getattr(args, "spatial_key", None) or "spatial"
    coords_key = getattr(args, "coords_key", None)
    if coords_key and spatial_key != "spatial" and coords_key != spatial_key:
        raise ValueError(
            f"coords_key='{coords_key}' and spatial_key='{spatial_key}' name two different coordinate keys; "
            "pass one of them (coords_key is the newer name for the same thing)."
        )
    return coords_key or spatial_key


#: ``uns`` key of the run's ``{mode, frame}`` record (``frame`` = ``Frame.to_dict()`` as JSON text).
SOG_CCC_UNS = "sog_ccc"


def _read_frame(adata, args, task: str, three_d_ok: bool = True):
    """``(frame, mode, section_key)``: the coordinates this run reads, put where squidpy will read them.

    Every task reads its coordinates through :func:`worker_utils.spatial_frame`, which refuses a
    three-column ``spatial``, a 3D run on an undeclared or rank-index z, and a 2D run over a stack of
    sections with no ``section_key``. The returned coordinates go to ``obsm[SOG_UM]`` and squidpy is
    pointed there, so a 3D graph is built in micrometres. ``section_key`` is the caller's (None when
    not given); in 2D it makes the run per section. ``three_d_ok=False`` (ripley) makes the refusals offer
    only the per-section way out.
    """
    dims = int(getattr(args, "dims", 2) or 2)
    section_key = getattr(args, "section_key", None) or None
    coords, frame = spatial_frame(adata, _coords_key(args), dims, section_key, "squidpy_" + task, three_d_ok)
    adata.obsm[SOG_UM] = coords
    if dims == 3:
        mode = MODE_3D
    elif section_key:
        mode = MODE_PER_SECTION
    else:
        mode = MODE_2D
    if section_key:
        import pandas as pd

        if not isinstance(adata.obs[section_key].dtype, pd.CategoricalDtype):
            adata.obs[section_key] = adata.obs[section_key].astype(str).astype("category")
    # The result h5ad's own record of the run's mode and frame: a task with no graph (co-occurrence, Ripley) leaves
    # the explorer nothing else to read them from (R2-3). The frame is JSON text: it holds None, which h5ad cannot.
    adata.uns[SOG_CCC_UNS] = {"mode": mode, "frame": json.dumps(frame.to_dict())}
    return frame, mode, section_key


def _frame_params(frame, mode: str, section_key) -> dict:
    """``params.mode`` / ``params.frame``: which coordinates the result is about, in the run's provenance."""
    return {
        "mode": mode,
        "frame": frame.to_dict(),
        "dims": frame.dims,
        "section_key": section_key,
        "n_sections": len(frame.sections) if frame.sections else None,
    }


def _mode_note(frame, mode: str, section_key) -> str:
    """One sentence for the analysis saying where the graph or the distances live."""
    if mode == MODE_3D:
        return (
            f" Built in 3D on obsm['{frame.key}'] in micrometres (z from {frame.z_source}); cross-section is "
            f"defined by obs['{section_key or frame.section_key}'], and a link between sections is inferred "
            "cross-section communication, not a measured one."
        )
    if mode == MODE_PER_SECTION:
        n = len(frame.sections) if frame.sections else 0
        return f" Run per section in 2D ({n} sections of obs['{section_key}']); nothing crosses a section."
    return ""


def _cross_section_edges(adata, section_key):
    """``(n_cross, fraction)``: graph edges whose two cells carry different labels in obs[section_key]."""
    import numpy as np

    graph = adata.obsp["spatial_connectivities"].tocoo()
    labels = adata.obs[section_key].astype(str).to_numpy()
    n_edges = int(graph.nnz)
    n_cross = int(np.count_nonzero(labels[graph.row] != labels[graph.col]))
    return n_cross, (float(n_cross) / n_edges if n_edges else 0.0)


def _report_frame(out, adata, frame, mode, section_key, graph=True) -> None:
    """Provenance of the frame, plus the cross-section edge count of a graph task given a section key."""
    out.add_params(_frame_params(frame, mode, section_key))
    if graph and section_key and "spatial_connectivities" in adata.obsp:
        n_cross, fraction = _cross_section_edges(adata, section_key)
        out.set_data(n_cross_section_edges=n_cross, cross_section_edge_fraction=fraction)


def _refuse_grid_in_3d(coord_type, mode: str) -> None:
    if mode == MODE_3D and str(coord_type) == "grid":
        raise ValueError(
            "coord_type='grid' is the 2D Visium lattice and has no third axis; a 3D graph needs "
            "coord_type='generic' (k nearest neighbours or Delaunay in micrometres), or run per section with "
            "`dims=2, section_key=<column>`."
        )


def _load_adata(h5ad_path: str, spatial_key: str = "spatial"):
    """Load AnnData and validate spatial coordinates exist."""
    import scanpy as sc

    log(f"Loading AnnData from {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    # The counts are left in adata.uns["identifier_renames"]; every task reads them back for its
    # payload, because each of the six keys a CSV or an annotated h5ad by these very names.
    make_names_unique_and_report(adata)
    log(f"Loaded: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    if spatial_key not in adata.obsm:
        available = list(adata.obsm.keys())
        raise ValueError(f"Spatial key '{spatial_key}' not found in adata.obsm. Available: {available}")
    return adata


def _atomic_to_csv(frame, path, **kwargs) -> None:
    """Write a CSV beside its final name and move it into place: a reader never sees half a table."""
    tmp = str(path) + ".partial"
    try:
        frame.to_csv(tmp, **kwargs)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_h5ad(adata, path) -> None:
    """``adata.write_h5ad`` to ``<path>.partial``, then rename; a killed run leaves no truncated h5ad.

    The scratch coordinate key :data:`SOG_UM` is dropped first: the frame it came from is named in
    ``params.frame``, and a copy of it in the output would be a second, unexplained coordinate key.
    """
    if SOG_UM in adata.obsm:
        del adata.obsm[SOG_UM]
    tmp = str(path) + ".partial"
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, str(path))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _present_labels(column) -> list:
    """The labels a column actually carries: used categories in category order, else first-seen order."""
    import pandas as pd

    if isinstance(column.dtype, pd.CategoricalDtype):
        counts = column.value_counts()
        return [c for c in column.cat.categories if int(counts.get(c, 0)) > 0]
    return list(pd.unique(column.dropna()))


def _keep_in_tissue(adata, cluster_key=None):
    """``(adata, n_spots_supplied, n_spots_off_tissue, labels_only_off_tissue)``.

    The shared ``keep_in_tissue`` leaves out the spots ``obs['in_tissue']`` marks 0 (background glass;
    CELLxGENE Visium exports keep every array spot, and the library's Heart sample has 3,009 of 4,992
    off the tissue, all labelled 'unknown'). Before it, every squidpy task analysed them as tissue:
    the neighbour graph joined the tissue edge to the glass, nhood/co-occurrence scored the
    background as a compartment, and Moran's I measured tissue-versus-background contrast.

    ``cluster_key`` names the label column of a task that reads one. A label carried only by
    background spots labels nothing once they are gone; left as a category it would reach squidpy as an
    all-zero row (a NaN z-score, a 0/0 centrality) and be reported as "a category that labels no cell",
    which it did -- off the tissue. Those labels are returned so the payload can name them, and removed
    from the categories if the subset kept them (anndata 0.11/0.12 already prunes every unused category
    when it subsets, so on those versions this is a no-op; older ones kept them).
    """
    before = None
    if cluster_key and cluster_key in adata.obs.columns:
        before = _present_labels(adata.obs[cluster_key])
    adata, n_supplied, n_off = keep_in_tissue(adata, "spots")
    emptied = []
    if n_off:
        log(f"Leaving out {n_off} of {n_supplied} spots with obs['in_tissue'] == 0 (background outside the tissue)")
        if before is not None:
            import pandas as pd

            after = set(_present_labels(adata.obs[cluster_key]))
            emptied = [c for c in before if c not in after]
            col = adata.obs[cluster_key]
            if isinstance(col.dtype, pd.CategoricalDtype):
                stale = [c for c in emptied if c in col.cat.categories]
                if stale:
                    adata.obs[cluster_key] = col.cat.remove_categories(stale)
    return adata, n_supplied, n_off, emptied


def _in_tissue_note(n_supplied: int, n_off: int, emptied=(), cluster_key: str = "") -> str:
    """The analysis sentence for the in-tissue cut, or nothing when no spot was left out."""
    if not n_off:
        return ""
    note = describe_reduction(
        "spots",
        int(n_supplied),
        int(n_supplied - n_off),
        reason="the in-tissue filter (obs['in_tissue'] == 0 marks background outside the tissue)",
    )
    if emptied:
        shown = ", ".join(str(c) for c in list(emptied)[:10])
        note += (
            f" Labels of obs['{cluster_key}'] carried only by those background spots were removed with them "
            f"({len(emptied)}: {shown})."
        )
    return note


def _validate_cluster_key(adata, cluster_key: str, required: bool = True, allow_drop_unlabeled: bool = False):
    """Ensure the cluster_key column exists in adata.obs, is categorical, and has no missing labels.

    Returns ``(adata, n_cells_dropped_unlabeled)``. The object comes back unchanged (normalised in
    place) unless unlabelled cells were dropped, in which case it is a copy without them.

    ``required=False`` is for tasks that do not consume the key at all — graph construction
    depends only on ``obsm['spatial']``. Failing those on a missing column made every raw Visium
    file (no obs column named ``cluster``, which is the point: clusters are computed later)
    unable to build the graph that the other five squidpy tasks list as their prerequisite.
    The categorical normalisation below still runs whenever the column is present, so runs that
    succeed today are unaffected. The missing-label check is skipped on that path too: a label
    nobody reads cannot be wrong.

    A missing label (NaN / None / "" / "nan") is not a class. squidpy's ``nhood_enrichment`` and
    ``co_occurrence`` crashed on one with a bare ``KeyError: nan``; ``ripley`` went further and
    scored the unlabelled cells as a pseudo-cluster (sklearn's LabelEncoder accepts NaN as a
    class). With ``allow_drop_unlabeled`` False the run stops and the message names the count and
    the knob; with it True the rows are left out and the count is returned for the payload.
    """
    if cluster_key not in adata.obs:
        available = list(adata.obs.columns)
        if not required:
            log(f"cluster_key='{cluster_key}' not in adata.obs (available: {available}); not needed for this task")
            return adata, 0
        raise ValueError(f"cluster_key='{cluster_key}' not found in adata.obs. Available: {available}")
    import pandas as pd

    n_dropped = 0
    if required:
        keep, n_dropped = drop_unlabeled(
            adata.obs[cluster_key].values, allow_drop_unlabeled, what=f"cells (obs['{cluster_key}'])"
        )
        if n_dropped:
            log(f"Dropping {n_dropped} of {adata.n_obs} cells with no label in obs['{cluster_key}'] (drop_unlabeled)")
            adata = adata[keep].copy()
    # Ensure categorical
    if not isinstance(adata.obs[cluster_key].dtype, pd.CategoricalDtype):
        log(f"Converting obs['{cluster_key}'] to categorical")
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    if n_dropped:
        # A literal "nan"/"" label was a category before the drop; an unused category would give
        # squidpy an all-zero row and a NaN z-score under a name that no longer labels anything.
        adata.obs[cluster_key] = adata.obs[cluster_key].cat.remove_unused_categories()
    return adata, n_dropped


def _dropped_note(n_dropped: int, cluster_key: str) -> str:
    """One sentence for the analysis when unlabelled cells were left out, else nothing."""
    if not n_dropped:
        return ""
    return (
        f" {n_dropped} cells with no label in obs['{cluster_key}'] were left out before the method ran "
        "(drop_unlabeled=True); every count above is after that cut."
    )


def _observed_categories(adata, cluster_key: str) -> list:
    """The categories of obs[cluster_key] that label at least one cell, in category order."""
    col = adata.obs[cluster_key]
    counts = col.value_counts()
    return [c for c in col.cat.categories if int(counts.get(c, 0)) > 0]


def _top_offdiag_pair(matrix, categories, largest: bool = True):
    """``(ct1, ct2, value, (i, j))`` for the largest/smallest finite off-diagonal cell, else None.

    The diagonal is excluded by masking it, not by writing 0 into it: a zeroed diagonal still
    competes, so when every heterotypic z-score is negative (two segregated compartments) the
    "most enriched pair" came back as a self pair with z=0, and when every one is positive the
    "most depleted" did. A category with no cells gives squidpy a NaN row and column, and
    ``np.argmax`` returns the first NaN it meets, so the headline pair was a label that labels
    nothing, with z=nan. Only finite values compete here.
    """
    import numpy as np

    m = np.array(matrix, dtype=float, copy=True)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] < 2:
        return None
    np.fill_diagonal(m, np.nan)
    finite = np.isfinite(m)
    if not finite.any():
        return None
    filled = np.where(finite, m, -np.inf if largest else np.inf)
    flat = int(np.argmax(filled)) if largest else int(np.argmin(filled))
    i, j = np.unravel_index(flat, m.shape)
    return categories[int(i)], categories[int(j)], float(m[i, j]), (int(i), int(j))


def _build_spatial_neighbors(adata, coord_type, n_neighs, n_rings, spatial_key, delaunay=False, library_key=None):
    """Build the spatial neighbor graph as a shared prerequisite step.

    ``library_key`` (a per-section 2D run) makes squidpy build one graph per section and join them
    block-diagonally, so no edge crosses a section.
    """
    import squidpy as sq

    log(
        f"Building spatial neighbors: coord_type={coord_type}, n_neighs={n_neighs}, "
        f"n_rings={n_rings}, delaunay={delaunay}, spatial_key={spatial_key}, library_key={library_key}"
    )
    sq.gr.spatial_neighbors(
        adata,
        coord_type=coord_type,
        n_neighs=n_neighs,
        n_rings=n_rings,
        delaunay=delaunay,
        spatial_key=spatial_key,
        library_key=library_key,
    )
    log("Spatial neighbor graph built successfully")


def _unused_graph_knobs(coord_type, n_neighs, n_rings, delaunay):
    """``[(name, why)]`` for the graph knobs squidpy.gr.spatial_neighbors does not read here.

    squidpy 1.6.5 (gr/_build.py): ``n_rings`` is read only on the grid path (``_build_grid``); the
    generic path calls ``_build_connectivity`` without it. ``n_neighs`` is not read once
    ``delaunay=True`` on either path: ``_build_connectivity`` takes the Delaunay triangulation's
    neighbours and never builds the k-nearest-neighbour tree. The default ``n_rings=1`` is the
    portal's, not a choice, so it is not reported on a generic graph.
    """
    unused = []
    if str(coord_type) != "grid" and n_rings != 1:
        unused.append(
            (
                "n_rings",
                f"squidpy reads n_rings only when coord_type='grid'; with coord_type='{coord_type}' "
                f"n_rings={n_rings} was not used",
            )
        )
    if delaunay:
        unused.append(
            (
                "n_neighs",
                f"with delaunay=True squidpy takes each spot's neighbours from the Delaunay triangulation, "
                f"so their number is set by the geometry and n_neighs={n_neighs} was not used",
            )
        )
    return unused


def _graph_description(coord_type, n_neighs, n_rings, delaunay) -> str:
    """What squidpy built, in the terms of the knobs it actually read."""
    if delaunay:
        base = "Delaunay triangulation"
        return base + (f" on the grid path, {n_rings} rings" if str(coord_type) == "grid" and n_rings > 1 else "")
    if str(coord_type) == "grid":
        return f"grid graph, {n_neighs} neighbouring tiles per ring, {n_rings} ring(s)"
    return f"{n_neighs}-nearest-neighbour graph on generic coordinates"


# ─── Task: spatial_neighbors ────────────────────────────────────────────


def run_spatial_neighbors(args):
    """Build spatial neighborhood graph and save annotated h5ad."""

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    adata, n_spots_supplied, n_off_tissue, _ = _keep_in_tissue(adata)
    # Advisory: this task never reads the key (sq.gr.spatial_neighbors has no such parameter).
    adata, _ = _validate_cluster_key(adata, args.cluster_key, required=False)
    if int(getattr(args, "dims", 2) or 2) == 3 and not getattr(args, "section_key", None):
        raise ValueError(
            "squidpy_spatial_neighbors with dims=3 needs section_key: the run reports the fraction of graph "
            "edges that join two sections (data.cross_section_edge_fraction), and a section is defined by its "
            "label, never by assuming a column of the frame is depth. Pass section_key=<the obs column naming "
            "each cell's section>."
        )
    frame, frame_mode, section_key = _read_frame(adata, args, "spatial_neighbors")
    _refuse_grid_in_3d(args.coord_type, frame_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _build_spatial_neighbors(
        adata,
        args.coord_type,
        args.n_neighs,
        args.n_rings,
        SOG_UM,
        args.delaunay,
        library_key=section_key if frame_mode == MODE_PER_SECTION else None,
    )

    h5ad_out = out_dir / "squidpy_spatial_neighbors.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    n_edges = adata.obsp["spatial_connectivities"].nnz if "spatial_connectivities" in adata.obsp else 0

    renamed = adata.uns.get("identifier_renames", {})
    graph = _graph_description(args.coord_type, args.n_neighs, args.n_rings, args.delaunay)

    out = WorkerOutput("squidpy", task="spatial_neighbors")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars))
    out.add_output_files({"annotated_h5ad": str(h5ad_out)})
    out.add_params(
        {
            "coord_type": args.coord_type,
            "n_neighs": args.n_neighs,
            "n_rings": args.n_rings,
            "delaunay": args.delaunay,
            "spatial_key": args.spatial_key,
        }
    )
    # The four graph knobs are echoed above as they were passed; the ones squidpy did not read under
    # this coord_type / delaunay combination are listed as ignored, so an echoed n_rings=3 on a
    # generic graph is not read as a three-ring neighbourhood.
    for name, why in _unused_graph_knobs(args.coord_type, args.n_neighs, args.n_rings, args.delaunay):
        record_ignored(out, name, why)
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    out.set_summary(n_edges=n_edges, graph=graph)
    avg_neighbors = n_edges / adata.n_obs if adata.n_obs > 0 else 0
    cross = ""
    if section_key:
        n_cross, fraction = _cross_section_edges(adata, section_key)
        cross = f" {n_cross} of the {n_edges} edges ({fraction:.4%}) join cells of two different sections."
    out.set_analysis(
        f"Built spatial neighbor graph ({graph}) for {adata.n_obs} spots with {n_edges} edges "
        f"(avg {avg_neighbors:.1f} neighbors per spot)."
        + _mode_note(frame, frame_mode, section_key)
        + cross
        + _in_tissue_note(n_spots_supplied, n_off_tissue)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── Task: nhood_enrichment ─────────────────────────────────────────────


def _n_note(n_pairs):
    """Render the observed-pair support for a z-score, or nothing when squidpy did not report it."""
    return "" if n_pairs is None else f", n={n_pairs} observed pairs"


def run_nhood_enrichment(args):
    """Compute neighborhood enrichment z-scores between cell types."""
    import numpy as np
    import pandas as pd
    import squidpy as sq

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    adata, n_spots_supplied, n_off_tissue, off_tissue_labels = _keep_in_tissue(adata, args.cluster_key)
    allow_drop = bool(getattr(args, "drop_unlabeled", False))
    adata, n_dropped = _validate_cluster_key(adata, args.cluster_key, allow_drop_unlabeled=allow_drop)
    frame, frame_mode, section_key = _read_frame(adata, args, "nhood_enrichment")
    _refuse_grid_in_3d(args.coord_type, frame_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build spatial neighbors first
    _build_spatial_neighbors(
        adata,
        args.coord_type,
        args.n_neighs,
        1,
        SOG_UM,
        library_key=section_key if frame_mode == MODE_PER_SECTION else None,
    )

    log(f"Running nhood_enrichment with n_perms={args.n_perms}, seed={args.seed}")
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=args.cluster_key,
        n_perms=args.n_perms,
        seed=args.seed,
    )

    # Extract z-score matrix
    nhood_key = f"{args.cluster_key}_nhood_enrichment"
    zscore_matrix = adata.uns[nhood_key]["zscore"]
    categories = list(adata.obs[args.cluster_key].cat.categories)
    zscore_df = pd.DataFrame(zscore_matrix, index=categories, columns=categories)

    zscore_csv = out_dir / "squidpy_nhood_enrichment_zscore.csv"
    _atomic_to_csv(zscore_df, zscore_csv)
    log(f"Saved z-score matrix to {zscore_csv}")

    # The z-score alone cannot distinguish a strong effect from a small-sample artifact: squidpy
    # also returns the observed number of neighbor pairs behind each cell, so export it too.
    count_csv = None
    count_matrix = adata.uns[nhood_key].get("count")
    if count_matrix is not None:
        count_df = pd.DataFrame(np.asarray(count_matrix), index=categories, columns=categories)
        count_csv = out_dir / "squidpy_nhood_enrichment_count.csv"
        _atomic_to_csv(count_df, count_csv)
        log(f"Saved observed neighbor-pair count matrix to {count_csv}")
    else:
        log("squidpy returned no 'count' matrix for this run; only z-scores are available")

    h5ad_out = out_dir / "squidpy_nhood_enrichment.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    # Find top enriched and depleted pairs over the finite off-diagonal z-scores. The helper copies,
    # so adata.uns is never edited (a later writer of this object would record the edit).
    enriched = _top_offdiag_pair(zscore_matrix, categories, largest=True)
    depleted = _top_offdiag_pair(zscore_matrix, categories, largest=False)
    top_enriched = enriched[:3] if enriched else (None, None, None)
    top_depleted = depleted[:3] if depleted else (None, None, None)
    observed = set(_observed_categories(adata, args.cluster_key))
    empty_categories = [c for c in categories if c not in observed]

    def _support(pair):
        """Observed neighbor pairs behind a cell of the z-score matrix, or None if unavailable."""
        if count_matrix is None or pair is None:
            return None
        return int(np.asarray(count_matrix)[pair[3]])

    top_enriched_n = _support(enriched)
    top_depleted_n = _support(depleted)

    def _pair_text(pair, n_pairs):
        if pair is None:
            return "none (no off-diagonal pair has a finite z-score)"
        return f"{pair[0]} <-> {pair[1]} (z={pair[2]:.2f}{_n_note(n_pairs)})"

    empty_note = (
        f" Categories of obs['{args.cluster_key}'] that label no cell "
        f"({len(empty_categories)}: {', '.join(str(c) for c in empty_categories[:10])}) have NaN rows and "
        "columns in the z-score CSV and were not ranked."
        if empty_categories
        else ""
    )

    renamed = adata.uns.get("identifier_renames", {})

    out = WorkerOutput("squidpy", task="nhood_enrichment")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars), n_cells_dropped_unlabeled=int(n_dropped))
    output_files = {
        "annotated_h5ad": str(h5ad_out),
        "zscore_csv": str(zscore_csv),
    }
    if count_csv is not None:
        output_files["count_csv"] = str(count_csv)
    out.add_output_files(output_files)
    out.add_params(
        {
            "cluster_key": args.cluster_key,
            "n_perms": args.n_perms,
            "n_neighs": args.n_neighs,
            "seed": args.seed,
            "drop_unlabeled": allow_drop,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    out.set_summary(
        n_celltypes=len(categories),
        celltypes=categories,
        top_enriched_pair={
            "ct1": top_enriched[0],
            "ct2": top_enriched[1],
            "zscore": top_enriched[2],
            "n_pairs_observed": top_enriched_n,
        },
        top_depleted_pair={
            "ct1": top_depleted[0],
            "ct2": top_depleted[1],
            "zscore": top_depleted[2],
            "n_pairs_observed": top_depleted_n,
        },
        null_model="global permutation of cluster labels over all cells",
        empty_categories=[str(c) for c in empty_categories],
        labels_only_off_tissue=[str(c) for c in off_tissue_labels],
    )
    out.set_analysis(
        f"Neighborhood enrichment computed for {len(categories)} cell types across {adata.n_obs} spots. "
        f"Most enriched neighbor pair: {_pair_text(enriched, top_enriched_n)}. "
        f"Most depleted pair: {_pair_text(depleted, top_depleted_n)}. "
        "Self-self pairs on the diagonal are excluded from both. z-scores are computed against a "
        f"global permutation of cluster labels across all cells (n_perms={args.n_perms}), a null "
        "that removes spatial structure at every scale -- so pairs that share an anatomical "
        "compartment score highly whether or not their association is specific to this sample."
        + _mode_note(frame, frame_mode, section_key)
        + empty_note
        + _in_tissue_note(n_spots_supplied, n_off_tissue, off_tissue_labels, args.cluster_key)
        + _dropped_note(n_dropped, args.cluster_key)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── Task: co_occurrence ────────────────────────────────────────────────


def _co_occurrence_one(adata, cluster_key, n_steps):
    """``(occ, interval, categories)`` of squidpy.gr.co_occurrence over ``obsm[SOG_UM]``.

    squidpy returns ``interval`` = the n_steps distance thresholds and ``occ`` with one bin per
    consecutive pair of thresholds, so the third axis has n_steps - 1 entries, not n_steps. The
    thresholds are in the frame's units: micrometres in 3D.
    """
    import squidpy as sq

    sq.gr.co_occurrence(adata, cluster_key=cluster_key, spatial_key=SOG_UM, interval=n_steps)
    result = adata.uns[f"{cluster_key}_co_occurrence"]
    return result["occ"], result["interval"], list(adata.obs[cluster_key].cat.categories)


def _mean_over_bins(occ):
    import numpy as np

    return np.mean(occ, axis=2) if getattr(occ, "ndim", 0) == 3 else np.asarray(occ)


#: co_occurrence's cost, estimated before squidpy starts (the 2026-10-06 real case: 7,001 s on 309,599 cells in 3D,
#: said nowhere before the run -- F-61). squidpy 1.6.5 splits the cells into ``s`` chunks (one while the n x n float32
#: distance matrix is under 2,000 MiB, else enough for at most 2,048 cells each), scans every chunk pair i <= j -- that
#: is (n^2 + sum of the chunk sizes squared) / 2 cell pairs, n^2 for one chunk -- and for each of the n_steps - 1
#: distance bins thresholds them and counts the pairs within, then does groups^2 work per chunk pair. So a plane costs
#: about ``FIXED + PAIR * pairs * (n_steps - 1) + GROUP * chunk_pairs * (n_steps - 1) * groups^2`` seconds, PAIR
#: depending on whether squidpy keeps the cells in one chunk.
#: Calibrated 2026-10-06 on this host (squidpy 1.6.5, the moscot env) on the real case and on random subsets of the
#: Zhuang 10-section object's CCF micrometres (30 domains): 2,589 / 5,000 / 10,000 / 20,000 cells x 50 thresholds in
#: one chunk took 1.8 / 6.1 / 26.3 / 132.1 s (20,000 x 10 thresholds: 28.1 s; x 200 random groups: 133.4 s, so the
#: groups barely count at these sizes); 40,000 cells in 20 chunks 189.8 s; and the recorded 3D run, 309,599 cells in
#: 152 chunks, 7,001 s. One chunk costs about 6.0e-9 s per cell pair per bin (the model gives 4.0 / 9.4 / 31.4 /
#: 119.6 / 23.6 s; FIXED is the numba compile a fresh worker pays, which the warmed calibration did not), chunked runs
#: about 3.7e-9 (the model gives 154 s and 8,749 s: random subsets cost more per pair than the file-ordered stack,
#: whose distant chunks hold fewer close pairs). Within about 25 % on every point above 5 s.
COOCCURRENCE_FIXED_S = 2.0
COOCCURRENCE_PAIR_ONE_CHUNK_S = 6.0e-9
COOCCURRENCE_PAIR_CHUNKED_S = 3.7e-9
COOCCURRENCE_GROUP_S = 1.0e-9
#: The budget a run's estimate must fit unless ``--max-estimated-s`` or the host's env knob says otherwise.
DEFAULT_COOCCURRENCE_MAX_SECONDS = 1800.0
COOCCURRENCE_BUDGET_ENV = "SOG_SQUIDPY_COOCCURRENCE_MAX_SECONDS"
#: squidpy's own rule for ``n_splits`` (``squidpy.gr._ppatterns.co_occurrence``): float32 distances, 2,000 MiB, 2,048.
_COOC_SPLIT_MIB = 2000
_COOC_SPLIT_CELLS = 2048


def _cooccurrence_chunks(n_cells):
    """The chunk sizes squidpy splits ``n_cells`` into (``np.array_split`` over its ``n_splits``)."""
    n = int(n_cells)
    if n <= 0:
        return []
    splits = 1
    if n * n * 4 / 1024 / 1024 > _COOC_SPLIT_MIB:
        while n / splits > _COOC_SPLIT_CELLS:
            splits += 1
    splits = max(min(splits, n), 1)
    base, extra = divmod(n, splits)
    return [base + 1] * extra + [base] * (splits - extra)


def co_occurrence_seconds(n_cells, n_steps, n_groups):
    """The estimated wall time, in seconds, of squidpy's co_occurrence on one plane of ``n_cells`` cells."""
    chunks = _cooccurrence_chunks(n_cells)
    if not chunks:
        return 0.0
    n = float(sum(chunks))
    pairs = (n * n + float(sum(c * c for c in chunks))) / 2.0
    chunk_pairs = len(chunks) * (len(chunks) + 1) / 2.0
    bins = max(1, int(n_steps) - 1)
    k = float(max(1, int(n_groups)))
    per_pair = COOCCURRENCE_PAIR_ONE_CHUNK_S if len(chunks) == 1 else COOCCURRENCE_PAIR_CHUNKED_S
    return COOCCURRENCE_FIXED_S + per_pair * pairs * bins + COOCCURRENCE_GROUP_S * chunk_pairs * bins * k * k


def _cooccurrence_budget(max_estimated_s):
    """``(seconds, source)``: the argument, else the host's env knob, else the default. <= 0 means no budget."""
    if max_estimated_s is not None:
        return float(max_estimated_s), "max_estimated_s"
    raw = (os.environ.get(COOCCURRENCE_BUDGET_ENV) or "").strip()
    if raw:
        try:
            return float(raw), COOCCURRENCE_BUDGET_ENV
        except ValueError:
            pass
    return DEFAULT_COOCCURRENCE_MAX_SECONDS, "default"


def _duration(seconds):
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _cooccurrence_plan(adata, cluster_key, n_steps, frame_mode, section_key, max_estimated_s):
    """The run's cost estimate, before squidpy starts: cells per plane (a section each per section, else one plane),
    the thresholds, the groups, the total, and the budget it must fit."""
    if frame_mode == MODE_PER_SECTION:
        sizes = adata.obs[section_key].astype(str).value_counts(sort=False)
        planes = {str(k): int(v) for k, v in sizes.items()}
        groups = {
            str(label): int(adata.obs.loc[adata.obs[section_key].astype(str) == label, cluster_key].nunique())
            for label in planes
        }
    else:
        planes = {"": int(adata.n_obs)}
        groups = {"": int(adata.obs[cluster_key].nunique())}
    per = {label: co_occurrence_seconds(n, n_steps, groups[label]) for label, n in planes.items()}
    largest = max(planes, key=lambda k: planes[k])
    budget, source = _cooccurrence_budget(max_estimated_s)
    return {
        "estimated_s": round(sum(per.values()), 1),
        "n_planes": len(planes),
        "n_cells_largest_plane": planes[largest],
        "largest_section": largest or None,
        "n_steps": int(n_steps),
        "n_groups": int(adata.obs[cluster_key].nunique()),
        "max_estimated_s": budget,
        "budget_source": source,
        "model": (
            f"seconds ~ {COOCCURRENCE_FIXED_S:g} + p * cell_pairs * (n_steps - 1) + {COOCCURRENCE_GROUP_S:.3g} * "
            "chunk_pairs * (n_steps - 1) * groups^2 per plane; cell_pairs = (cells^2 + sum of squidpy's chunk "
            f"sizes^2) / 2, p = {COOCCURRENCE_PAIR_ONE_CHUNK_S:.3g} in one chunk (up to about 22,900 cells) else "
            f"{COOCCURRENCE_PAIR_CHUNKED_S:.3g} (calibrated on one host)"
        ),
    }


def _cooccurrence_refusal(plan, frame_mode):
    """The refusal when the estimate exceeds a positive budget, every number stated; else None."""
    budget = plan["max_estimated_s"]
    if budget <= 0 or plan["estimated_s"] <= budget:
        return None
    if plan["n_planes"] > 1:
        where = (
            f"{plan['n_planes']} sections run one by one; the largest, '{plan['largest_section']}', has "
            f"{plan['n_cells_largest_plane']:,} cells"
        )
    else:
        frame = "one 3D frame" if frame_mode == MODE_3D else "one plane"
        where = f"{plan['n_cells_largest_plane']:,} cells in {frame}"
    return (
        f"squidpy_co_occurrence: this run is estimated at about {_duration(plan['estimated_s'])} ({where}, "
        f"{plan['n_steps']} distance thresholds, {plan['n_groups']} groups). squidpy scans every pair of cells at each "
        "distance bin, so the cost grows with cells^2 x intervals; the estimate is calibrated on one host and "
        f"approximate. It is over the budget of {_duration(budget)} (max_estimated_s, or "
        f"{COOCCURRENCE_BUDGET_ENV} for the host), so nothing was run. To run it: fewer cells -- a subset of "
        "adjacent sections or a bounding box written to its own file (halving the cells quarters the cost), or per "
        "section with `dims=2, section_key=<column>` (each section's own pairs only, no cross-section distances) -- or "
        "fewer intervals (n_steps), or raise max_estimated_s."
    )


def run_co_occurrence(args):
    """Compute cell type co-occurrence probabilities at varying distances.

    In 3D the distances are micrometres in the aligned frame. Per section (``dims=2`` with a
    ``section_key``) squidpy runs once per section through :func:`worker_utils.per_section` -- one
    pairwise-distance computation over a stack would measure distances between overlaid sections --
    and the mean CSV holds one block per section, led by a ``section`` column.
    """
    import numpy as np
    import pandas as pd

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    adata, n_spots_supplied, n_off_tissue, off_tissue_labels = _keep_in_tissue(adata, args.cluster_key)
    allow_drop = bool(getattr(args, "drop_unlabeled", False))
    adata, n_dropped = _validate_cluster_key(adata, args.cluster_key, allow_drop_unlabeled=allow_drop)
    frame, frame_mode, section_key = _read_frame(adata, args, "co_occurrence")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cooccurrence_csv = out_dir / "squidpy_co_occurrence_mean.csv"
    co_key = f"{args.cluster_key}_co_occurrence"

    plan = _cooccurrence_plan(
        adata, args.cluster_key, args.n_steps, frame_mode, section_key, getattr(args, "max_estimated_s", None)
    )
    refusal = _cooccurrence_refusal(plan, frame_mode)
    if refusal:
        raise ValueError(refusal)
    log(
        f"Running co_occurrence with interval={args.n_steps} (mode={frame_mode}); estimated "
        f"{_duration(plan['estimated_s'])} (budget {_duration(plan['max_estimated_s'])}, {plan['budget_source']})"
    )
    if frame_mode == MODE_PER_SECTION:
        runs = []  # (label, occ, interval, categories), in the file's section order

        def run_one(sub, label):
            sub.obs[args.cluster_key] = sub.obs[args.cluster_key].cat.remove_unused_categories()
            occ, interval, cats = _co_occurrence_one(sub, args.cluster_key, args.n_steps)
            runs.append((label, occ, interval, cats))
            block = pd.DataFrame(_mean_over_bins(occ), index=cats, columns=cats)
            block.index.name = "celltype"
            return block.reset_index()

        table = per_section(adata, section_key, run_one, tool="squidpy_co_occurrence")
        _atomic_to_csv(table, cooccurrence_csv, index=False)
        adata.uns[co_key + "_per_section"] = {
            "sections": np.asarray([r[0] for r in runs], dtype=object),
            "interval": np.vstack([np.asarray(r[2], dtype=float) for r in runs]),
            "occ": {str(i): np.asarray(r[1]) for i, r in enumerate(runs)},
            "categories": {str(i): np.asarray(r[3], dtype=object) for i, r in enumerate(runs)},
        }
        first_occ, first_interval = runs[0][1], runs[0][2]
        ranked = []
        for label, occ, _interval, cats in runs:
            best_here = _top_offdiag_pair(_mean_over_bins(occ), cats, largest=True)
            if best_here:
                ranked.append((label,) + tuple(best_here[:3]))
        top = max(ranked, key=lambda r: r[3]) if ranked else None
        top_pair = {"section": top[0], "ct1": top[1], "ct2": top[2], "score": top[3]} if top else None
        categories = list(adata.obs[args.cluster_key].cat.categories)
    else:
        occ_matrix, first_interval, categories = _co_occurrence_one(adata, args.cluster_key, args.n_steps)
        first_occ = occ_matrix
        mean_occ = _mean_over_bins(occ_matrix)
        _atomic_to_csv(pd.DataFrame(mean_occ, index=categories, columns=categories), cooccurrence_csv)
        # Find top co-occurring pair over the finite off-diagonal cells (the helper masks the diagonal
        # on a copy; the CSV above and adata.uns keep the full matrix).
        best = _top_offdiag_pair(mean_occ, categories, largest=True)
        top_pair = {"ct1": best[0], "ct2": best[1], "score": best[2]} if best else None
        ranked = None
    log(f"Saved mean co-occurrence to {cooccurrence_csv}")

    n_thresholds = int(len(first_interval)) if hasattr(first_interval, "__len__") else int(args.n_steps)
    n_bins = int(first_occ.shape[2]) if getattr(first_occ, "ndim", 0) == 3 else 1
    n_types = len(categories)

    h5ad_out = out_dir / "squidpy_co_occurrence.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    renamed = adata.uns.get("identifier_renames", {})

    out = WorkerOutput("squidpy", task="co_occurrence")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars), n_cells_dropped_unlabeled=int(n_dropped))
    out.add_output_files(
        {
            "annotated_h5ad": str(h5ad_out),
            "co_occurrence_csv": str(cooccurrence_csv),
        }
    )
    out.add_params(
        {
            "cluster_key": args.cluster_key,
            "n_steps": args.n_steps,
            "spatial_key": args.spatial_key,
            "drop_unlabeled": allow_drop,
            "cost_estimate": plan,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key, graph=False)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    empty_pair = {"ct1": None, "ct2": None, "score": None}
    out.set_summary(
        n_celltypes=n_types,
        celltypes=categories,
        n_distance_steps=n_thresholds,
        n_distance_thresholds=n_thresholds,
        n_distance_bins=n_bins,
        top_co_occurring_pair=top_pair or empty_pair,
        labels_only_off_tissue=[str(c) for c in off_tissue_labels],
    )
    if ranked is not None:
        out.set_summary(
            top_co_occurring_pair_per_section=[
                {"section": r[0], "ct1": r[1], "ct2": r[2], "score": r[3]} for r in ranked
            ]
        )
    if top_pair:
        where = f" in section {top_pair['section']}" if "section" in top_pair else ""
        strongest = (
            f"Strongest co-occurrence: {top_pair['ct1']} <-> {top_pair['ct2']}{where} "
            f"(mean score={top_pair['score']:.3f})."
        )
    else:
        strongest = "Strongest co-occurrence: none (no off-diagonal pair has a finite mean score)."
    csv_shape = (
        "The mean-co-occurrence CSV holds one block per section (column 'section'), each "
        if frame_mode == MODE_PER_SECTION
        else "The mean-co-occurrence CSV "
    )
    out.set_analysis(
        f"Co-occurrence analysis computed for {n_types} cell types across {adata.n_obs} spots. "
        f"{csv_shape}averages over {n_bins} distance bins ({n_thresholds} distance thresholds, "
        f"n_steps={args.n_steps}). "
        + strongest
        + _mode_note(frame, frame_mode, section_key)
        + _in_tissue_note(n_spots_supplied, n_off_tissue, off_tissue_labels, args.cluster_key)
        + _dropped_note(n_dropped, args.cluster_key)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── Task: spatial_autocorr ─────────────────────────────────────────────


def _bh_adjust(pvals):
    """Benjamini-Hochberg over the finite p-values only; NaN stays NaN.

    squidpy already writes ``<p>_fdr_bh`` columns, but it runs ``multipletests`` over the whole
    vector, and one NaN (a constant gene has NaN Moran's I and NaN p) turns the entire adjusted
    column into NaN -- a recorded Visium run had 32,285 of 32,285 rows empty. The adjusted values
    are computed here, in memory, for the significance count and the SVG filter; the CSV keeps
    squidpy's columns exactly as squidpy wrote them.
    """
    import numpy as np

    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    m = int(finite.sum())
    if m == 0:
        return out
    pf = p[finite]
    order = np.argsort(pf, kind="mergesort")
    ranked = pf[order] * m / np.arange(1, m + 1)
    adj = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    res = np.empty(m)
    res[order] = adj
    out[finite] = res
    return out


def _autocorr_pvalues(autocorr_df, n_perms, use_fdr):
    """``(p-values, CSV column read, correction)`` for the significance count and the SVG filter.

    ``n_perms`` decides the test: given, the permutation p-value ``pval_sim`` is what the caller
    asked to compute and it drives the result; omitted, the analytic ``pval_norm`` does. The
    earlier code always read ``pval_norm`` and fell back to a column named ``pval`` that squidpy
    never writes, so ``n_perms`` changed the CSV and nothing else. ``use_fdr`` applies BH over the
    chosen column (see ``_bh_adjust``). The column named is the CSV column whose values were read;
    it is never squidpy's ``*_fdr_bh`` column, which can be NaN in every row.
    """
    base = "pval_sim" if n_perms is not None else "pval_norm"
    if base not in autocorr_df.columns:
        raise ValueError(
            f"squidpy returned no '{base}' column (columns: {list(autocorr_df.columns)}); "
            "the squidpy.gr.spatial_autocorr contract this worker reads has changed."
        )
    raw = autocorr_df[base].to_numpy(dtype=float)
    if not use_fdr:
        return raw, base, "none"
    return _bh_adjust(raw), base, "fdr_bh over the finite p-values"


def run_spatial_autocorr(args):
    """Compute spatial autocorrelation (Moran's I or Geary's C) for all genes."""
    import numpy as np
    import squidpy as sq

    # head(-1) keeps all but the last gene, so a negative top_n would publish nearly every gene as the
    # predicted SVG list; refuse it before any work. 0 keeps every gene that passes the filter.
    top_n_arg = getattr(args, "top_n", 200)
    if top_n_arg is not None and int(top_n_arg) < 0:
        raise ValueError(f"top_n must be >= 0 (0 keeps every gene that passes the p-value filter); got {top_n_arg}.")

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    # Background spots carry ambient counts well above zero (Heart Fetal12W: median 968 against 3,265 in
    # the tissue), so a gene's tissue-versus-glass contrast scored as spatial autocorrelation.
    adata, n_spots_supplied, n_off_tissue, _ = _keep_in_tissue(adata)
    frame, frame_mode, section_key = _read_frame(adata, args, "spatial_autocorr")
    _refuse_grid_in_3d(args.coord_type, frame_mode)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build spatial neighbors
    _build_spatial_neighbors(
        adata,
        args.coord_type,
        args.n_neighs,
        1,
        SOG_UM,
        library_key=section_key if frame_mode == MODE_PER_SECTION else None,
    )

    n_perms = args.n_perms if args.n_perms is not None else None
    seed = getattr(args, "seed", 0)
    use_fdr = bool(getattr(args, "use_fdr", False))
    use_hvg = bool(getattr(args, "use_highly_variable", False))

    # Which genes are tested is stated, never inferred: with ``genes=None`` squidpy silently
    # restricts the test to var['highly_variable'] whenever that column exists, and the payload
    # showed n_genes_tested < n_genes with no reason. The whole panel is the default; the subset
    # runs only when asked for, and is then reported as a reduction.
    if use_hvg:
        if "highly_variable" not in adata.var:
            raise ValueError(
                "use_highly_variable=True but adata.var has no 'highly_variable' column; run an HVG "
                "selection first or leave use_highly_variable at False to test every gene."
            )
        hvg_mask = adata.var["highly_variable"].astype(bool).to_numpy()
        if not hvg_mask.any():
            raise ValueError("use_highly_variable=True but no gene is flagged in adata.var['highly_variable'].")
        genes = np.asarray(adata.var_names[hvg_mask])
        genes_source = "var['highly_variable'] (use_highly_variable=True)"
    else:
        genes = np.asarray(adata.var_names)
        genes_source = "all var_names"

    log(
        f"Running spatial_autocorr with mode={args.mode}, n_perms={n_perms}, n_jobs={args.n_jobs}, "
        f"seed={seed}, genes={len(genes)} ({genes_source})"
    )
    sq.gr.spatial_autocorr(
        adata,
        mode=args.mode,
        genes=genes,
        n_perms=n_perms,
        n_jobs=args.n_jobs,
        seed=seed,
    )

    # Results are stored in adata.uns['moranI'] or adata.uns['gearyC']
    result_key = "moranI" if args.mode == "moran" else "gearyC"
    autocorr_df = adata.uns[result_key]

    autocorr_csv = out_dir / f"squidpy_{result_key}.csv"
    _atomic_to_csv(autocorr_df, autocorr_csv)
    log(f"Saved autocorrelation results to {autocorr_csv}")

    h5ad_out = out_dir / f"squidpy_spatial_autocorr_{args.mode}.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    # Summary statistics. squidpy folds its p-values one-tailed in whichever direction the statistic
    # departs from its expectation, so a strongly *negatively* autocorrelated gene also gets a tiny p.
    # A spatially variable gene is the positive direction: I above E[I] = -1/(n-1), or C below 1.
    pvals, pval_col, correction = _autocorr_pvalues(autocorr_df, n_perms, use_fdr)
    pval_label = f"BH-adjusted {pval_col}" if use_fdr else pval_col
    stat_col = "I" if args.mode == "moran" else "C"
    stat = autocorr_df[stat_col].to_numpy(dtype=float)
    if args.mode == "moran":
        expected = -1.0 / (adata.n_obs - 1) if adata.n_obs > 1 else 0.0
        positive = stat > expected
    else:
        positive = stat < 1.0
    with np.errstate(invalid="ignore"):
        n_significant = int(((pvals < 0.05) & positive).sum())

    sort_ascending = args.mode == "geary"
    sorted_df = autocorr_df.sort_values(stat_col, ascending=sort_ascending)
    top_genes = list(sorted_df.head(10).index)

    # Curated SVG prediction list — used by SVG benchmarking pipeline. A non-positive threshold
    # disables the p-value filter, as the tool has always documented (the old ``< threshold``
    # guard turned 0 into "no gene passes" and wrote an empty list as the authoritative answer).
    pvalue_threshold = getattr(args, "pvalue_threshold", 0.05)
    top_n = getattr(args, "top_n", 200)
    filter_applied = pvalue_threshold is not None and pvalue_threshold > 0
    if filter_applied:
        with np.errstate(invalid="ignore"):
            passing = (pvals < pvalue_threshold) & positive
        filtered_df = autocorr_df.loc[passing].sort_values(stat_col, ascending=sort_ascending)
    else:
        filtered_df = sorted_df
    predicted_genes = list(filtered_df.head(top_n).index) if top_n else list(filtered_df.index)
    predicted_genes_json = out_dir / "predicted_genes.json"
    tmp_json = str(predicted_genes_json) + ".partial"
    with open(tmp_json, "w") as fh:
        json.dump([str(g) for g in predicted_genes], fh, indent=2)
    os.replace(tmp_json, str(predicted_genes_json))
    log(f"Saved {len(predicted_genes)} predicted SVGs to {predicted_genes_json}")

    renamed = adata.uns.get("identifier_renames", {})

    out = WorkerOutput("squidpy", task="spatial_autocorr")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars))
    out.add_output_files(
        {
            "annotated_h5ad": str(h5ad_out),
            "autocorr_csv": str(autocorr_csv),
            "predicted_genes_json": str(predicted_genes_json),
        }
    )
    out.add_params(
        {
            # params.mode is the coordinate mode ("3d" / "per-section-2d" / "2d"); the statistic is here.
            "autocorr_mode": args.mode,
            "n_neighs": args.n_neighs,
            "n_perms": n_perms,
            "seed": seed,
            "n_jobs": args.n_jobs,
            "coord_type": args.coord_type,
            "top_n": top_n,
            "pvalue_threshold": pvalue_threshold,
            "pvalue_filter_applied": bool(filter_applied),
            "pvalue_column": pval_col,
            "pvalue_correction": correction,
            "use_fdr": use_fdr,
            "use_highly_variable": use_hvg,
            "genes_tested_source": genes_source,
        }
    )
    # The portal forwards cluster_key to every task; this one tests genes, not labels, and
    # squidpy.gr.spatial_autocorr has no such argument. A column the caller named is reported as
    # ignored rather than silently accepted (the default 'cluster' is the portal's, not a choice, and
    # a warning on every run would train the reader to skip the one that matters).
    if getattr(args, "cluster_key", "cluster") != "cluster":
        record_ignored(
            out,
            "cluster_key",
            f"spatial_autocorr tests genes for spatial autocorrelation; it takes no label column, so "
            f"cluster_key='{args.cluster_key}' was not read",
        )
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    out.set_summary(
        n_genes_tested=int(len(autocorr_df)),
        n_significant=n_significant,
        n_predicted_genes=len(predicted_genes),
        top_genes=top_genes,
        metric=result_key,
        pvalue_column=pval_col,
        pvalue_correction=correction,
    )
    metric_label = "Moran's I" if args.mode == "moran" else "Geary's C"
    test_label = "permutation" if n_perms is not None else "analytic"
    corr_label = "BH-adjusted over the finite values" if use_fdr else "uncorrected"
    perm_note = (
        f" With n_perms={n_perms} the smallest attainable permutation p-value is 1/{n_perms + 1}."
        if n_perms is not None
        else ""
    )
    filter_label = (
        f"filtered at {pval_label} < {pvalue_threshold} (positive autocorrelation only) and "
        if filter_applied
        else "with the p-value filter disabled (pvalue_threshold <= 0) and "
    )
    out.set_analysis(
        f"Spatial autocorrelation ({metric_label}) computed for {len(autocorr_df)} genes "
        f"across {adata.n_obs} spots. {n_significant} genes show positive spatial autocorrelation at "
        f"{pval_label} < 0.05 ({corr_label} {test_label} p-values, CSV column {pval_col}).{perm_note} "
        f"predicted_genes.json holds {len(predicted_genes)} genes, {filter_label}ranked by {metric_label}. "
        f"Top spatially variable genes: {', '.join(str(g) for g in top_genes[:5])}."
        + _mode_note(frame, frame_mode, section_key)
        + describe_reduction("genes", int(adata.n_vars), int(len(autocorr_df)), reason=f"the {genes_source} subset")
        + _in_tissue_note(n_spots_supplied, n_off_tissue)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── Task: ripley ───────────────────────────────────────────────────────


def run_ripley(args):
    """Compute Ripley's statistics (F, G, or L) for spatial point patterns.

    squidpy's Ripley is 2D only (it draws its null from a 2D convex hull), so ``dims=3`` is refused, and a
    stack of sections runs per section through :func:`worker_utils.per_section`: the CSV then holds one
    block per section, led by a ``section`` column.
    """
    import anndata
    import pandas as pd
    import squidpy as sq

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    adata, n_spots_supplied, n_off_tissue, off_tissue_labels = _keep_in_tissue(adata, args.cluster_key)
    allow_drop = bool(getattr(args, "drop_unlabeled", False))
    adata, n_dropped = _validate_cluster_key(adata, args.cluster_key, allow_drop_unlabeled=allow_drop)
    if int(getattr(args, "dims", 2) or 2) == 3:
        raise ValueError(
            "squidpy ripley is 2D only; run per section with `dims=2, section_key=<column>`. "
            "(Its null pattern is drawn in a two-dimensional hull, so a 3D Ripley statistic would compare a "
            "volume against a plane.)"
        )
    frame, frame_mode, section_key = _read_frame(adata, args, "ripley", three_d_ok=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # squidpy.gr.ripley reads exactly two things: obs[cluster_key] and obsm[spatial_key]. It used to
    # be handed the whole object, and because its internals copy obs into fresh DataFrames (which
    # fail on array-valued columns), the worker stripped obs down to the cluster column and coerced
    # every obsm entry to a bare array -- then wrote THAT as the "annotated" h5ad, destroying every
    # other obs column the input had. Run it on a minimal object instead and copy the result back.
    ripley_key = f"{args.cluster_key}_ripley_{args.mode}"

    def _ripley_on(part):
        """squidpy.gr.ripley on a minimal object of ``part``'s labels and frame coordinates."""
        labels = part.obs[[args.cluster_key]].copy()
        labels[args.cluster_key] = labels[args.cluster_key].cat.remove_unused_categories()
        mini = anndata.AnnData(obs=labels, obsm={SOG_UM: part.obsm[SOG_UM]})
        # No ``seed``: squidpy re-seeds its Poisson process from the same value on every one of the
        # n_simulations draws, so a fixed seed collapses the whole envelope onto one simulated pattern.
        sq.gr.ripley(
            mini,
            cluster_key=args.cluster_key,
            mode=args.mode,
            spatial_key=SOG_UM,
            n_steps=args.n_steps,
            n_simulations=args.n_simulations,
            n_neigh=1,  # workaround for squidpy 1.8.1 bug: n_neigh>1 produces 2D distances that break pandas
        )
        return mini.uns[ripley_key]

    log(f"Running Ripley's {args.mode} function with n_steps={args.n_steps}, n_simulations={args.n_simulations}")
    if frame_mode == MODE_PER_SECTION:
        stat_key = f"{args.mode}_stat"

        def run_one(sub, _label):
            result = _ripley_on(sub)
            stat = result.get(stat_key) if isinstance(result, dict) else result
            if not isinstance(stat, pd.DataFrame):
                raise ValueError(f"squidpy.gr.ripley returned no '{stat_key}' table for a section")
            return stat

        # The per-section table under the key the explorer reads, as squidpy's own one-section result is.
        ripley_result = {stat_key: per_section(adata, section_key, run_one, tool="squidpy_ripley")}
    else:
        ripley_result = _ripley_on(adata)
    adata.uns[ripley_key] = ripley_result

    ripley_csv = out_dir / f"squidpy_ripley_{args.mode}.csv"
    # ripley returns a dict with keys: {mode}_stat (DataFrame), sims_stat (DataFrame), bins (1D), pvalues (2D)
    # Save the main statistics DataFrame
    if isinstance(ripley_result, pd.DataFrame):
        _atomic_to_csv(ripley_result, ripley_csv, index=False)
    elif isinstance(ripley_result, dict):
        stat_key = f"{args.mode}_stat"
        if stat_key in ripley_result and isinstance(ripley_result[stat_key], pd.DataFrame):
            _atomic_to_csv(ripley_result[stat_key], ripley_csv, index=False)
        else:
            # Only the 1D-safe entries can be tabulated
            safe = {
                k: v
                for k, v in ripley_result.items()
                if isinstance(v, (pd.DataFrame, pd.Series)) or (hasattr(v, "ndim") and v.ndim <= 1)
            }
            _atomic_to_csv(pd.DataFrame(safe), ripley_csv, index=False)
    else:
        _atomic_to_csv(pd.DataFrame({"result": [str(ripley_result)]}), ripley_csv, index=False)
    log(f"Saved Ripley's {args.mode} results to {ripley_csv}")

    h5ad_out = out_dir / f"squidpy_ripley_{args.mode}.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    # sklearn's LabelEncoder inside squidpy.gr.ripley fits on the observed labels, so a category that
    # labels no cell has no curve in the CSV; count and name only the ones that were scored.
    categories = _observed_categories(adata, args.cluster_key)

    renamed = adata.uns.get("identifier_renames", {})

    out = WorkerOutput("squidpy", task="ripley")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars), n_cells_dropped_unlabeled=int(n_dropped))
    out.add_output_files(
        {
            "annotated_h5ad": str(h5ad_out),
            "ripley_csv": str(ripley_csv),
        }
    )
    out.add_params(
        {
            "cluster_key": args.cluster_key,
            # params.mode is the coordinate mode ("per-section-2d" / "2d"); the statistic is here.
            "ripley_mode": args.mode,
            "n_steps": args.n_steps,
            "n_simulations": args.n_simulations,
            "spatial_key": args.spatial_key,
            "drop_unlabeled": allow_drop,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key, graph=False)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    out.set_summary(
        n_celltypes=len(categories),
        celltypes=categories,
        ripley_mode=args.mode,
        labels_only_off_tissue=[str(c) for c in off_tissue_labels],
    )
    mode_names = {"F": "F (empty space)", "G": "G (nearest neighbor)", "L": "L (Besag's L)"}
    mode_label = mode_names.get(args.mode, args.mode)
    out.set_analysis(
        f"Ripley's {mode_label} function computed for {len(categories)} cell types "
        f"across {adata.n_obs} spots with {args.n_simulations} simulations. "
        f"Results capture spatial clustering/dispersion patterns at {args.n_steps} distance steps. "
        "The annotated h5ad is the input with the result added under uns; its obs columns are unchanged."
        + _mode_note(frame, frame_mode, section_key)
        + _in_tissue_note(n_spots_supplied, n_off_tissue, off_tissue_labels, args.cluster_key)
        + _dropped_note(n_dropped, args.cluster_key)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── Task: centrality_scores ────────────────────────────────────────────


def run_centrality_scores(args):
    """Compute graph centrality scores for each cell type cluster."""
    import pandas as pd
    import squidpy as sq

    adata = _load_adata(args.h5ad_path, _coords_key(args))
    adata, n_spots_supplied, n_off_tissue, off_tissue_labels = _keep_in_tissue(adata, args.cluster_key)
    allow_drop = bool(getattr(args, "drop_unlabeled", False))
    adata, n_dropped = _validate_cluster_key(adata, args.cluster_key, allow_drop_unlabeled=allow_drop)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # squidpy scores every *category* of the column, and a category that labels no cell (common after
    # subsetting: pandas keeps the categories of the full table) hands networkx an empty node group,
    # which raised a bare ZeroDivisionError from average_clustering. There is nothing to score for a
    # label with no cells, so the empty categories are removed first and named in the payload.
    scored = set(_observed_categories(adata, args.cluster_key))
    empty_categories = [c for c in adata.obs[args.cluster_key].cat.categories if c not in scored]
    if empty_categories:
        log(f"Removing {len(empty_categories)} categories of obs['{args.cluster_key}'] that label no cell")
        adata.obs[args.cluster_key] = adata.obs[args.cluster_key].cat.remove_unused_categories()
    frame, frame_mode, section_key = _read_frame(adata, args, "centrality_scores")
    _refuse_grid_in_3d(args.coord_type, frame_mode)

    # Build spatial neighbors
    _build_spatial_neighbors(
        adata,
        args.coord_type,
        args.n_neighs,
        1,
        SOG_UM,
        library_key=section_key if frame_mode == MODE_PER_SECTION else None,
    )

    log("Running centrality_scores")
    sq.gr.centrality_scores(
        adata,
        cluster_key=args.cluster_key,
    )

    centrality_key = f"{args.cluster_key}_centrality_scores"
    centrality_df = adata.uns[centrality_key]
    if not isinstance(centrality_df, pd.DataFrame):
        centrality_df = pd.DataFrame(centrality_df)

    centrality_csv = out_dir / "squidpy_centrality_scores.csv"
    _atomic_to_csv(centrality_df, centrality_csv)
    log(f"Saved centrality scores to {centrality_csv}")

    h5ad_out = out_dir / "squidpy_centrality_scores.h5ad"
    _atomic_write_h5ad(adata, h5ad_out)
    log(f"Saved annotated h5ad to {h5ad_out}")

    categories = list(adata.obs[args.cluster_key].cat.categories)

    # Find most central cell type by closeness centrality
    if "closeness_centrality" in centrality_df.columns:
        most_central_idx = centrality_df["closeness_centrality"].idxmax()
        most_central_score = float(centrality_df.loc[most_central_idx, "closeness_centrality"])
        most_central = str(most_central_idx)
    else:
        most_central = categories[0] if categories else "unknown"
        most_central_score = 0.0

    metrics = list(centrality_df.columns)

    renamed = adata.uns.get("identifier_renames", {})

    out = WorkerOutput("squidpy", task="centrality_scores")
    out.set_data(n_spots=int(adata.n_obs), n_genes=int(adata.n_vars), n_cells_dropped_unlabeled=int(n_dropped))
    out.add_output_files(
        {
            "annotated_h5ad": str(h5ad_out),
            "centrality_csv": str(centrality_csv),
        }
    )
    out.add_params(
        {
            "cluster_key": args.cluster_key,
            "coord_type": args.coord_type,
            "n_neighs": args.n_neighs,
            "drop_unlabeled": allow_drop,
        }
    )
    out.add_params(identifier_rename_params(renamed))
    _report_frame(out, adata, frame, frame_mode, section_key)
    record_in_tissue(out, n_spots_supplied, n_off_tissue, "spots")
    out.set_summary(
        n_celltypes=len(categories),
        celltypes=categories,
        centrality_metrics=metrics,
        most_central_type=most_central,
        most_central_closeness=most_central_score,
        empty_categories_removed=[str(c) for c in empty_categories],
        labels_only_off_tissue=[str(c) for c in off_tissue_labels],
    )
    out.set_analysis(
        f"Graph centrality scores computed for {len(categories)} cell types across {adata.n_obs} spots. "
        f"Metrics: {', '.join(metrics)}. "
        f"Most central cell type by closeness: {most_central} (score={most_central_score:.4f})."
        + _mode_note(frame, frame_mode, section_key)
        + (
            f" Categories of obs['{args.cluster_key}'] that labelled no cell were left out "
            f"({len(empty_categories)}: {', '.join(str(c) for c in empty_categories[:10])}); the annotated "
            "h5ad carries the column without them."
            if empty_categories
            else ""
        )
        + _in_tissue_note(n_spots_supplied, n_off_tissue, off_tissue_labels, args.cluster_key)
        + _dropped_note(n_dropped, args.cluster_key)
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


# ─── CLI and task dispatch ──────────────────────────────────────────────

TASK_DISPATCH = {
    "spatial_neighbors": run_spatial_neighbors,
    "nhood_enrichment": run_nhood_enrichment,
    "co_occurrence": run_co_occurrence,
    "spatial_autocorr": run_spatial_autocorr,
    "ripley": run_ripley,
    "centrality_scores": run_centrality_scores,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Squidpy worker: spatial graph analysis for spatial transcriptomics.")

    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_DISPATCH.keys()),
        help="Which squidpy analysis task to run.",
    )
    parser.add_argument("--h5ad-path", required=True, help="Path to AnnData (.h5ad).")
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--cluster-key", default="cluster", help="obs column with cell type labels.")
    parser.add_argument("--spatial-key", default="spatial", help="obsm key with spatial coordinates.")
    parser.add_argument(
        "--coords-key",
        default=None,
        help="obsm key holding the coordinates (default: --spatial-key, i.e. 'spatial'); an aligned 3D frame "
        "such as 'spatial_3d_aligned' with --dims 3.",
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=(2, 3),
        default=2,
        help="2 or 3. 3 builds the graph in the aligned frame in micrometres; it needs a frame declared in "
        "uns['spatial_3d']['frames'] with units and a measured or registered z.",
    )
    parser.add_argument(
        "--section-key",
        default=None,
        help="obs column naming each cell's section. Required for a 2D run on a multi-section file (the run is "
        "then per section) and for the cross-section edge count of a 3D run.",
    )
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out cells whose obs[cluster_key] is missing (NaN/empty) instead of stopping; the count is "
        "reported. Tasks that consume the key only (nhood_enrichment, co_occurrence, ripley, centrality_scores).",
    )

    # spatial_neighbors / nhood_enrichment / spatial_autocorr / centrality
    parser.add_argument("--coord-type", default="generic", help="Coordinate type: 'generic' or 'grid'.")
    parser.add_argument("--n-neighs", type=int, default=6, help="Number of spatial neighbors.")
    parser.add_argument("--n-rings", type=int, default=1, help="Number of rings for grid data.")
    parser.add_argument("--delaunay", action="store_true", help="Use Delaunay triangulation.")

    # nhood_enrichment / spatial_autocorr
    parser.add_argument(
        "--n-perms",
        type=int,
        default=None,
        help="Number of permutations. nhood_enrichment: defaults to 1000. spatial_autocorr: omitted -> analytic "
        "p-values (pval_norm) drive significance and the SVG filter; given -> permutation p-values (pval_sim).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (nhood_enrichment, spatial_autocorr).")

    # co_occurrence / ripley
    parser.add_argument("--n-steps", type=int, default=50, help="Number of distance steps.")
    parser.add_argument(
        "--max-estimated-s",
        type=float,
        default=None,
        help="co_occurrence: the wall-time budget, in seconds, the run's estimate must fit (else "
        "SOG_SQUIDPY_COOCCURRENCE_MAX_SECONDS, else 1800); 0 or less means no budget.",
    )

    # spatial_autocorr
    parser.add_argument(
        "--mode",
        default=None,
        help="The statistic: spatial_autocorr 'moran' (default) or 'geary'; ripley 'F', 'G' or 'L' (default 'L').",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument(
        "--top-n", type=int, default=200, help="Number of top spatially variable genes to emit in predicted_genes.json."
    )
    parser.add_argument(
        "--pvalue-threshold", type=float, default=0.05, help="P-value cutoff for the SVG list (set <=0 to disable)."
    )
    parser.add_argument(
        "--use-fdr",
        action="store_true",
        help="Apply Benjamini-Hochberg to the chosen p-value column before the significance count and the SVG filter.",
    )
    parser.add_argument(
        "--use-highly-variable",
        action="store_true",
        help="Test only genes flagged in var['highly_variable'] (reported as a reduction). Default: every gene.",
    )

    # ripley
    parser.add_argument("--n-simulations", type=int, default=100, help="Number of simulations for Ripley's.")

    return parser.parse_args()


#: Each task's own statistics, its default first. ``--mode`` was one flag with autocorr's default ('moran'), so a
#: ripley call without it handed 'moran' to squidpy, which died inside RipleyStat (SQPY-1).
_STATISTICS = {"spatial_autocorr": ("moran", "geary"), "ripley": ("L", "F", "G")}


def _statistic_for(task, mode):
    """The statistic ``task`` runs: ``mode`` when it is one of the task's own, its default when omitted."""
    allowed = _STATISTICS.get(task)
    if allowed is None:
        return mode
    if mode is None or str(mode).strip() == "":
        return allowed[0]
    if mode not in allowed:
        if task == "ripley":
            names = "'F', 'G' or 'L'"
        else:
            names = "'moran' or 'geary'"
        raise ValueError(f"--mode {mode!r} is not a {task} statistic; {task} takes {names} (default {allowed[0]!r}).")
    return mode


def main():
    import json

    args = parse_args()

    # Redirect stdout to stderr during processing to keep stdout clean for JSON
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr

    result = None
    error_msg = None
    error_exc = None
    try:
        task_fn = TASK_DISPATCH[args.task]
        args.mode = _statistic_for(args.task, args.mode)
        # Set default for n_perms based on task
        if args.task == "nhood_enrichment" and args.n_perms is None:
            args.n_perms = 1000
        result = task_fn(args)
    except Exception as e:
        log(f"ERROR in task '{args.task}':")
        traceback.print_exc(file=sys.stderr)
        error_msg = str(e)
        error_exc = e
    finally:
        sys.stdout = orig_stdout

    # Print a single JSON line to stdout
    if result is None:
        WorkerOutput.emit_error("squidpy", error_msg, task=args.task, exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
