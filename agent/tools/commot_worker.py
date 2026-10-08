#!/usr/bin/env python

"""
commot_worker.py

CLI worker for COMMOT cell-cell communication on spatial transcriptomics data.

Runs inside /opt/conda/envs/COMMOT and is called by the FastMCP wrapper
(commot_mcp_server.py) via subprocess.

IMPORTANT:
- All logs go to STDERR
- STDOUT prints exactly ONE line containing JSON (status + metadata)

What the run is measured in. ``commot.tl.spatial_communication`` compares ``dis_thr`` with
Euclidean distances computed from ``obsm['spatial']`` as stored, so the threshold is in the
coordinates' own units: full-resolution image pixels for 10x Visium (one 100 um spot pitch is
anywhere from ~30 to ~370 of them depending on the scan), microns for most imaging platforms.
A fixed default of 200 in those units meant "within-spot signalling only" on every Visium slide
whose pitch exceeds 200 pixels, and still returned ``status: ok``. The default is now
``dis_thr=0`` = auto: 200 um, converted with the slide's own scalefactors when it carries them.
Every run reports the effective threshold in both units, the measured spot spacing and how many
spot pairs fall inside it, and a threshold that leaves every spot without a partner is refused.

When the input carries ``obsp['spatial_distance']``, COMMOT uses that matrix instead of distances
from ``obsm['spatial']`` (upstream documents this), so ``dis_thr`` is in the matrix's units. The
worker leaves it in place, measures the spacing on it, and converts microns into it only when it is
the Euclidean distance of ``obsm['spatial']``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from typing import Any

from worker_utils import (
    WorkerOutput,
    _declared_frame,
    available_memory_bytes,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    per_section,
    record_in_tissue,
    record_method,
    spatial_frame,
    unsupported_choice_msg,
)

#: The databases ``commot.pp.ligand_receptor_database`` has a branch for, and the species files each
#: one ships (read from the installed package's ``_data/LRdatabase``). Any other name reaches no
#: branch and dies with ``UnboundLocalError: ... 'df_ligrec'``; "OmniPath" and "FANTOM5", which the
#: docs used to advertise, are among them.
SUPPORTED_LR_DATABASES = {
    "CellChat": ("human", "mouse", "zebrafish"),
    "CellPhoneDB_v4.0": ("human", "mouse"),
}
_LR_DATABASE_ALIASES = {
    "cellchat": "CellChat",
    "cellchatdb": "CellChat",
    "cellphonedb": "CellPhoneDB_v4.0",
    "cellphonedb_v4": "CellPhoneDB_v4.0",
    "cellphonedb_v4.0": "CellPhoneDB_v4.0",
    "cellphonedbv4.0": "CellPhoneDB_v4.0",
}

#: ``ligand_receptor_database``'s own default, which this worker does not override: only the
#: secreted-signalling pairs of either database are loaded. Reported so a pair count is not read as
#: "the whole database".
LR_SIGNALING_TYPE = "Secreted Signaling"
#: ``filter_lr_database``'s own default: a pair is kept when every ligand and receptor subunit is
#: detected (> 0) in at least this fraction of spots.
LR_MIN_CELL_PCT = 0.05

#: 10x's ``spot_diameter_fullres`` is "the number of pixels that span the diameter of a theoretical
#: 65 um spot in the full-resolution image" -- not the 55 um physical spot. Checked against the
#: library: V1_Breast 177.48 px / 65 = 2.7305 px/um, and its measured nearest-neighbour spacing is
#: 273 px = 100.0 um, the Visium pitch; /55 would make it 84.6 um.
VISIUM_SPOT_DIAMETER_FULLRES_UM = 65.0
#: Centre-to-centre distance of neighbouring Visium spots. A scalefactor that makes the measured
#: spacing much SMALLER than this does not describe these coordinates (typically obsm['spatial'] was
#: rescaled to the hires or lowres image), so it is not used. Larger is fine: a subset of spots is
#: sparser than the full array.
VISIUM_PITCH_UM = 100.0
_MIN_PLAUSIBLE_VISIUM_PITCH_UM = 80.0

#: What ``dis_thr <= 0`` (the default) means: this many microns.
AUTO_DIS_THR_UM = 200.0
DIS_THR_UNITS = ("coordinates", "um")

#: Below this share of spots with at least one partner inside the threshold, the run proceeds but
#: says so. At zero it is refused: COMMOT would score within-spot signalling only.
_LOW_PARTNER_FRACTION = 0.5

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{5,}(\.\d+)?$")
#: var columns that hold gene symbols beside an Ensembl-indexed var (CELLxGENE uses feature_name).
SYMBOL_COLUMNS = (
    "feature_name",
    "gene_symbols",
    "gene_symbol",
    "gene_name",
    "gene_names",
    "symbol",
    "symbols",
    "SYMBOL",
    "GeneName",
    "gene",
)

METHOD_NAME = "COMMOT spatial_communication (collective optimal transport)"

#: ``spatial_communication`` measures ``dis_thr`` on this matrix whenever it exists, and on Euclidean
#: distances of ``obsm['spatial']`` only when it does not (``commot/tools/_spatial_communication.py``).
DISTANCE_KEY = "spatial_distance"
OBSM_SPACE = "obsm['spatial']"
OBSP_SPACE = "obsp['spatial_distance']"

#: Rows of an n x n matrix handled at once when it is scanned, so no second n x n array is made.
_BLOCK_ELEMENTS = 4_000_000


def log(msg: str) -> None:
    """Print log messages to STDERR with a consistent prefix."""
    print(f"[commot-worker] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# Argument checks (cheap, before anything is loaded)
# ---------------------------------------------------------------------------------------------


def resolve_lr_database(lr_database: str, species: str) -> tuple:
    """``(database, species)`` as upstream spells them, or a ValueError naming what would work."""
    requested = str(lr_database or "").strip()
    database = requested if requested in SUPPORTED_LR_DATABASES else _LR_DATABASE_ALIASES.get(requested.lower())
    if database is None:
        raise ValueError(
            unsupported_choice_msg(
                "lr_database",
                lr_database,
                list(SUPPORTED_LR_DATABASES),
                extra=(
                    "These are the only databases the installed commot.pp.ligand_receptor_database "
                    "implements; any other name has no branch there."
                ),
            )
        )
    species_arg = str(species or "").strip().lower()
    if species_arg not in SUPPORTED_LR_DATABASES[database]:
        raise ValueError(
            unsupported_choice_msg(
                "species",
                species,
                list(SUPPORTED_LR_DATABASES[database]),
                extra=f"COMMOT ships {database} for these species only.",
            )
        )
    return database, species_arg


def check_database_name(database_name: str) -> str:
    """The label goes into output file names and AnnData keys, so it must be a plain name."""
    name = str(database_name or "")
    if not name.strip():
        raise ValueError("database_name is empty; it labels the output files and AnnData keys, so give it a name.")
    bad = [c for c in ("/", "\\", "\x00") if c in name]
    if bad:
        raise ValueError(
            f"database_name={database_name!r} contains {bad[0]!r}; it becomes part of the output file names "
            "(commot_<database_name>_results.h5ad) and of HDF5 key names, so it must be a plain name."
        )
    return name


def resolve_dis_thr_unit(dis_thr_unit: str) -> str:
    unit = str(dis_thr_unit or "coordinates").strip().lower()
    unit = {"micron": "um", "microns": "um", "\u00b5m": "um", "coordinate": "coordinates", "coords": "coordinates"}.get(
        unit, unit
    )
    if unit not in DIS_THR_UNITS:
        raise ValueError(unsupported_choice_msg("dis_thr_unit", dis_thr_unit, list(DIS_THR_UNITS)))
    return unit


# ---------------------------------------------------------------------------------------------
# Spatial scale and the threshold
# ---------------------------------------------------------------------------------------------


def coordinate_scale(uns: Any, median_nn: float) -> tuple:
    """``(coordinate units per micron, source)``, or ``(None, why not)``.

    Read from ``uns['spatial'][library]['scalefactors']``: ``microns_per_pixel`` when SpaceRanger
    wrote it, otherwise ``spot_diameter_fullres`` / 65 um. The second is accepted only when it puts
    the measured spot spacing at a plausible Visium pitch, because it describes full-resolution
    pixels and ``obsm['spatial']`` is sometimes stored at another scale.
    """
    spatial = None
    try:
        spatial = uns.get("spatial") if hasattr(uns, "get") else None
    except Exception:
        spatial = None
    if not hasattr(spatial, "items") or not len(spatial):
        return None, "the file has no uns['spatial'] scalefactors"
    found = []
    for library, entry in spatial.items():
        factors = entry.get("scalefactors") if hasattr(entry, "get") else None
        if not hasattr(factors, "get"):
            continue
        mpp = _positive_float(factors.get("microns_per_pixel"))
        if mpp is not None:
            found.append((1.0 / mpp, f"uns['spatial'][{library!r}]['scalefactors']['microns_per_pixel']", False))
            continue
        spot = _positive_float(factors.get("spot_diameter_fullres"))
        if spot is not None:
            found.append(
                (
                    spot / VISIUM_SPOT_DIAMETER_FULLRES_UM,
                    f"uns['spatial'][{library!r}]['scalefactors']['spot_diameter_fullres'] / "
                    f"{VISIUM_SPOT_DIAMETER_FULLRES_UM:g} um",
                    True,
                )
            )
    if not found:
        return None, "uns['spatial'] carries no spot_diameter_fullres or microns_per_pixel scalefactor"
    scales = [f[0] for f in found]
    if max(scales) > 1.01 * min(scales):
        return None, (
            f"uns['spatial'] holds {len(found)} libraries with different scalefactors "
            f"({min(scales):.4g}-{max(scales):.4g} units per um), so no single micron scale applies"
        )
    units_per_um, source, is_visium_spot = found[0]
    if is_visium_spot and median_nn is not None and math.isfinite(median_nn):
        pitch_um = median_nn / units_per_um
        if pitch_um < _MIN_PLAUSIBLE_VISIUM_PITCH_UM:
            return None, (
                f"{source} gives {units_per_um:.4g} units per um, which would put neighbouring spots "
                f"{pitch_um:.1f} um apart; a Visium pitch is {VISIUM_PITCH_UM:g} um, so obsm['spatial'] "
                "is not in full-resolution pixels and that scalefactor does not describe it"
            )
    return units_per_um, source


def _positive_float(value: Any):
    try:
        out = float(value)
    except (TypeError, ValueError):
        try:
            out = float(getattr(value, "item", lambda: None)())
        except (TypeError, ValueError):
            return None
    return out if math.isfinite(out) and out > 0 else None


def resolve_dis_thr(dis_thr: float, unit: str, units_per_um, scale_source: str) -> tuple:
    """``(threshold in obsm['spatial'] units, threshold in um or None, basis sentence)``."""
    value = float(dis_thr)
    if math.isnan(value):
        raise ValueError("dis_thr is NaN; pass a distance, or 0 for the automatic 200 um threshold.")
    if value <= 0:
        if units_per_um is not None:
            eff = AUTO_DIS_THR_UM * units_per_um
            return (
                eff,
                AUTO_DIS_THR_UM,
                f"automatic: {AUTO_DIS_THR_UM:g} um x {units_per_um:.4g} coordinate units per um ({scale_source})",
            )
        return (
            AUTO_DIS_THR_UM,
            None,
            f"automatic: {AUTO_DIS_THR_UM:g} obsm['spatial'] units, taking the coordinates to be microns because "
            f"{scale_source}",
        )
    if unit == "um":
        if units_per_um is None:
            raise ValueError(
                f"dis_thr_unit='um' needs a micron scale for obsm['spatial'], and {scale_source}. Pass dis_thr in "
                "the coordinates' own units with dis_thr_unit='coordinates' (the error below, or a run with "
                "dis_thr=0, reports the measured spot spacing in those units)."
            )
        return (
            value * units_per_um,
            value,
            f"{value:g} um x {units_per_um:.4g} coordinate units per um ({scale_source})",
        )
    um = value / units_per_um if units_per_um is not None else None
    return value, um, f"{value:g} obsm['spatial'] units, as passed"


def neighbour_stats(coords: Any, threshold: float) -> dict:
    """Spacing and connectivity of the spots under ``threshold``, without an n x n matrix."""
    import numpy as np
    from scipy.spatial import cKDTree

    pts = np.asarray(coords, dtype=np.float64)
    n = int(pts.shape[0])
    if n < 2:
        raise ValueError(f"COMMOT needs at least 2 spots to score communication between them; the input has {n}.")
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=2)
    nn = dist[:, 1]
    within = int((nn <= threshold).sum())
    # count_neighbors counts ordered pairs including each spot with itself.
    ordered = int(tree.count_neighbors(tree, float(threshold)))
    pairs = max(0, (ordered - n) // 2)
    return {
        "n_spots": n,
        "median_nn": float(np.median(nn)),
        "min_nn": float(np.min(nn)),
        "n_spots_with_partner": within,
        "n_spot_pairs_within": pairs,
        "mean_partners_per_spot": float(2.0 * pairs / n),
    }


def isolated_threshold_message(
    threshold: float, unit_basis: str, stats: dict, units_per_um, space: str = OBSM_SPACE, convertible: bool = True
) -> str:
    """``convertible=False``: COMMOT measures a precomputed matrix of unknown units, so microns cannot be offered."""
    nn = stats["median_nn"]
    scale = ""
    if units_per_um is not None:
        scale = (
            f" On this slide 1 um = {units_per_um:.4g} coordinate units, so the median spacing is "
            f"{nn / units_per_um:.1f} um and {AUTO_DIS_THR_UM:g} um is {AUTO_DIS_THR_UM * units_per_um:.4g} units."
        )
    return (
        f"dis_thr={threshold:.4g} (in {space} units; {unit_basis}) is shorter than the distance from every "
        f"spot to its nearest neighbour (median {nn:.4g}, minimum {stats['min_nn']:.4g} units): no two spots fall "
        f"within it, so COMMOT would score only signalling within each spot and report it as a communication map."
        f"{scale} "
        + (
            "Pass a larger dis_thr (a few times the spacing), dis_thr_unit='um' with a distance in microns, "
            "or dis_thr=0 for the automatic 200 um."
            if convertible
            else f"Pass a larger dis_thr in that matrix's own units (a few times the spacing), or remove {space} from "
            f"the file to have COMMOT measure {OBSM_SPACE}."
        )
    )


def precomputed_distance_matrix(obsp: Any, n_obs: int):
    """The input's ``obsp['spatial_distance']`` as a dense array, or None when it has none.

    Upstream ``spatial_communication`` reads this matrix whenever it exists ("If the spatial distance is
    absent in .obsp['spatial_distance'], Euclidean distance determined from .obsm['spatial'] will be
    used"), so it is COMMOT's distance and ``dis_thr`` is in its units. A sparse one is refused: upstream's
    ``np.where(M <= max_cutoff)`` fails on it ("Calling nonzero on 0d arrays is not allowed"), and its
    absent entries would read as distance 0.
    """
    import numpy as np

    try:
        present = DISTANCE_KEY in obsp
    except Exception:
        present = False
    if not present:
        return None
    dmat = obsp[DISTANCE_KEY]
    if hasattr(dmat, "tocsr") or hasattr(dmat, "nnz"):
        raise ValueError(
            f"{OBSP_SPACE} is a sparse matrix. COMMOT uses that matrix in place of distances from {OBSM_SPACE} "
            "whenever it exists, and needs it dense: upstream's np.where(M <= dis_thr) fails on a sparse one "
            "('Calling nonzero on 0d arrays is not allowed'), and its absent entries would read as distance 0. "
            f"Store it as a dense array, or remove {OBSP_SPACE} from the file to have COMMOT measure {OBSM_SPACE}."
        )
    dmat = np.asarray(dmat)
    if dmat.ndim != 2 or dmat.shape != (int(n_obs), int(n_obs)):
        raise ValueError(
            f"{OBSP_SPACE} has shape {dmat.shape}; COMMOT needs one distance per pair of the {n_obs} spots."
        )
    return dmat


def matrix_neighbour_stats(dmat: Any, threshold: float) -> dict:
    """The ``neighbour_stats`` of a precomputed distance matrix, diagonal masked, scanned in row blocks.

    A NaN entry counts as out of range, as it does in upstream's ``M <= cutoff``.
    """
    import numpy as np

    n = int(dmat.shape[0])
    if n < 2:
        raise ValueError(f"COMMOT needs at least 2 spots to score communication between them; the input has {n}.")
    step = max(1, _BLOCK_ELEMENTS // n)
    nn = np.empty(n, dtype=np.float64)
    with_partner = 0
    ordered = 0
    for start in range(0, n, step):
        stop = min(n, start + step)
        block = np.array(dmat[start:stop], dtype=np.float64)
        local = np.arange(stop - start)
        block[local, local + start] = np.inf
        block[np.isnan(block)] = np.inf
        nn[start:stop] = block.min(axis=1)
        within = block <= threshold
        with_partner += int(within.any(axis=1).sum())
        ordered += int(within.sum())
    pairs = ordered // 2
    return {
        "n_spots": n,
        "median_nn": float(np.median(nn)),
        "min_nn": float(np.min(nn)),
        "n_spots_with_partner": with_partner,
        "n_spot_pairs_within": pairs,
        "mean_partners_per_spot": float(2.0 * pairs / n),
    }


def matrix_matches_coords(dmat: Any, coords: Any) -> tuple:
    """``(True, None)`` when ``dmat`` is the Euclidean distance matrix of ``coords``; else ``(False, where)``.

    Every row is compared, in blocks. Only then is ``dmat`` in ``obsm['spatial']`` units, so that the
    slide's micron scale applies to it.
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    pts = np.asarray(coords, dtype=np.float64)
    n = int(pts.shape[0])
    extent = float(np.ptp(pts, axis=0).max()) if n else 0.0
    atol = 1e-6 * (extent if extent > 0 else 1.0)
    step = max(1, _BLOCK_ELEMENTS // max(n, 1))
    for start in range(0, n, step):
        stop = min(n, start + step)
        block = np.asarray(dmat[start:stop], dtype=np.float64)
        euclid = cdist(pts[start:stop], pts)
        close = np.isclose(block, euclid, rtol=1e-5, atol=atol)
        if not close.all():
            i, j = (int(v) for v in np.argwhere(~close)[0])
            return False, (
                f"{OBSP_SPACE}[{start + i}, {j}] = {block[i, j]:.6g} where the Euclidean distance of {OBSM_SPACE} "
                f"is {euclid[i, j]:.6g}"
            )
    return True, None


def foreign_matrix_message(dis_thr: float, unit: str, difference: str, stats: dict) -> str:
    asked = (
        f"dis_thr={dis_thr:g} (automatic {AUTO_DIS_THR_UM:g} um)"
        if dis_thr <= 0
        else f"dis_thr={dis_thr:g} with dis_thr_unit={unit!r}"
    )
    return (
        f"The input carries {OBSP_SPACE}, and COMMOT measures dis_thr on that matrix instead of on {OBSM_SPACE}. "
        f"It is not the Euclidean distance of {OBSM_SPACE} ({difference}), so its units are unknown here and "
        f"{asked} cannot be converted into them. Pass dis_thr in that matrix's own units with "
        f"dis_thr_unit='coordinates' (its median nearest-neighbour distance is {stats['median_nn']:.4g}), or remove "
        f"{OBSP_SPACE} from the file to have COMMOT measure {OBSM_SPACE}."
    )


# ---------------------------------------------------------------------------------------------
# Genes and expression
# ---------------------------------------------------------------------------------------------


def _looks_ensembl(names: Any) -> bool:
    names = [str(n) for n in list(names)[:500]]
    if not names:
        return False
    hits = sum(1 for n in names if _ENSEMBL_RE.match(n))
    return hits >= 0.5 * len(names)


def use_symbol_var_names(adata: Any) -> dict:
    """Name genes by symbol when ``var_names`` are Ensembl IDs and ``var`` has a symbol column.

    The ligand-receptor databases are keyed by symbol; matched against Ensembl IDs no pair
    survives, and COMMOT then dies with ``IndexError: single positional indexer is
    out-of-bounds``. A gene whose symbol is empty keeps its ID; a symbol already taken by an
    earlier gene also keeps the ID, so no name is invented. The ID each gene had is kept in
    ``var['var_name_in_input']``.
    """
    import numpy as np

    report = {"gene_id_column": None, "n_genes_named_by_symbol": 0, "n_duplicate_symbols_kept_as_ids": 0}
    if not _looks_ensembl(adata.var_names):
        return report
    columns = list(getattr(adata.var, "columns", []))
    for col in SYMBOL_COLUMNS:
        if col not in columns:
            continue
        symbols = np.array([str(s).strip() for s in adata.var[col].astype(object).values], dtype=object)
        valid = np.array([bool(s) and s.lower() not in ("nan", "none", "na") for s in symbols], dtype=bool)
        if not valid.any() or _looks_ensembl(symbols[valid]):
            continue
        original = np.array([str(v) for v in adata.var_names], dtype=object)
        new = original.copy()
        seen = set()
        n_named = 0
        n_dup = 0
        for i in range(len(new)):
            if not valid[i]:
                continue
            if symbols[i] in seen:
                n_dup += 1
                continue
            seen.add(symbols[i])
            new[i] = symbols[i]
            n_named += 1
        adata.var["var_name_in_input"] = list(original)
        adata.var_names = [str(v) for v in new]
        report.update(gene_id_column=col, n_genes_named_by_symbol=n_named, n_duplicate_symbols_kept_as_ids=n_dup)
        log(
            f"var_names are Ensembl IDs; {n_named} genes renamed to their symbol from var['{col}'] "
            f"({n_dup} duplicate symbols kept their ID)."
        )
        return report
    return report


def _first_value(X: Any, predicate) -> Any:
    """The first stored value of ``X`` for which ``predicate`` holds, scanning in blocks, or None."""
    import numpy as np

    if hasattr(X, "tocsr") and hasattr(X, "data"):
        values = np.asarray(X.data)
        blocks = [values[i : i + 1_000_000] for i in range(0, values.size, 1_000_000)]
    else:
        arr = np.asarray(X)
        width = arr.shape[1] if arr.ndim == 2 and arr.shape[1] else 1
        step = max(1, 1_000_000 // width)
        blocks = [arr[i : i + step] for i in range(0, arr.shape[0], step)]
    for block in blocks:
        block = np.asarray(block, dtype=np.float64).ravel()
        hit = predicate(block)
        if hit.any():
            return float(block[np.flatnonzero(hit)[0]])
    return None


def check_expression(X: Any, normalize: bool, uns: Any):
    """Refuse expression COMMOT cannot use; return a warning (or None) for what it can.

    COMMOT transports ligand and receptor *amounts*, so negative values (scaled or centred data)
    are meaningless to it. ``normalize=True`` applies normalize_total + log1p; on a matrix the file
    itself marks as log-transformed (``uns['log1p']``) that normalises twice and logs a log.
    """
    import numpy as np

    negative = _first_value(X, lambda b: b < 0)
    if negative is not None:
        raise ValueError(
            f"X holds negative values (e.g. {negative:g}): it looks scaled or centred. COMMOT transports "
            "non-negative ligand and receptor amounts; supply counts (normalize=True) or normalised, "
            "log-transformed expression (normalize=False)."
        )
    non_finite = _first_value(X, lambda b: ~np.isfinite(b))
    if non_finite is not None:
        raise ValueError(f"X holds a non-finite value ({non_finite}); COMMOT needs finite expression.")
    if not normalize:
        return None
    fractional = _first_value(X, lambda b: b != np.floor(b))
    if fractional is None:
        return None
    try:
        marked_log = "log1p" in uns
    except Exception:
        marked_log = False
    if marked_log:
        raise ValueError(
            f"normalize=True would run normalize_total + log1p on X, but X is not counts (e.g. {fractional:g}) and "
            "uns['log1p'] records that it is already log-transformed; that would normalise twice and log a log. "
            "Pass normalize=False to use X as it is."
        )
    return (
        f"X holds non-integer values (e.g. {fractional:g}); normalize=True treated them as counts and ran "
        "normalize_total + log1p. If X is already normalised, pass normalize=False."
    )


def usable_pair_count(df_ligrec: Any, var_names: Any, heteromeric: bool) -> int:
    """How many rows ``spatial_communication`` keeps, by its own rule (it re-filters on entry)."""
    genes = {str(v) for v in var_names}
    n = 0
    for i in range(df_ligrec.shape[0]):
        lig = str(df_ligrec.iloc[i, 0])
        rec = str(df_ligrec.iloc[i, 1])
        if heteromeric:
            ok = set(lig.split("_")).issubset(genes) and set(rec.split("_")).issubset(genes)
        else:
            ok = lig in genes and rec in genes
        n += int(ok)
    return n


def no_pairs_message(df_db: Any, var_names: Any, database: str, species: str, gene_report: dict) -> str:
    names = [str(v) for v in var_names]
    genes = set(names)
    db_genes = set()
    for i in range(df_db.shape[0]):
        for col in (0, 1):
            db_genes.update(str(df_db.iloc[i, col]).split("_"))
    present = len(db_genes & genes)
    shown = ", ".join(names[:3])
    if present == 0:
        why = (
            f"none of the {len(db_genes)} genes it names are in var_names (e.g. {shown}). The database is keyed by "
            f"{species} gene symbols; set var_names to symbols"
        )
        if not gene_report.get("gene_id_column"):
            why += (
                " (the worker does so itself when var_names are Ensembl IDs and var has one of the columns "
                + ", ".join(SYMBOL_COLUMNS[:4])
                + ")"
            )
        why += ", or pass the species these genes come from"
    else:
        why = (
            f"{present} of its {len(db_genes)} genes are in var_names, but no pair has its ligand and every receptor "
            f"subunit detected in at least {LR_MIN_CELL_PCT:.0%} of spots"
        )
    return (
        f"No ligand-receptor pair of {database} ({species}, {LR_SIGNALING_TYPE}; {df_db.shape[0]} pairs) survived "
        f"commot.pp.filter_lr_database: {why}. COMMOT has nothing to score."
    )


# ---------------------------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------------------------


def dense_distance_bytes(n_spots: int, precomputed: bool = False) -> int:
    """Upstream builds a dense float64 n x n distance matrix and an n x n boolean mask of it.

    With a precomputed ``obsp['spatial_distance']`` (already loaded with the file) it builds only the mask.
    """
    n = int(n_spots)
    return (1 if precomputed else 9) * n * n


def check_dense_memory(n_spots: int, available=None, precomputed: bool = False) -> int:
    """Refuse up front when upstream's dense n x n intermediates cannot fit.

    ``available`` defaults to ``worker_utils.available_memory_bytes()``, which counts a cgroup's page
    cache as reclaimable: ``memory.current`` includes it, and right after ``sc.read_h5ad`` a
    memory-limited container sits near its limit on cache alone, so ``memory.max - memory.current``
    refused runs that fit.
    """
    need = dense_distance_bytes(n_spots, precomputed=precomputed)
    if available is None:
        available = available_memory_bytes()
    if available is not None and need > available:
        gib = float(1 << 30)
        built = (
            f"an n x n boolean mask of the input's {OBSP_SPACE} ({n_spots} x {n_spots})"
            if precomputed
            else f"a dense {n_spots} x {n_spots} distance matrix (scipy distance_matrix, float64) and an equally "
            "large mask"
        )
        raise MemoryError(
            f"COMMOT's spatial_communication builds {built} while it selects the pairs within dis_thr: "
            f"about {need / gib:.1f} GiB for {n_spots} spots, and {available / gib:.1f} GiB is available here. "
            "That dense matrix is how upstream COMMOT works; run on a machine with at least that much free memory."
        )
    return need


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------


def _write_h5ad_atomic(adata: Any, path: str) -> None:
    partial = path + ".partial"
    adata.write_h5ad(partial)
    os.replace(partial, path)


def _write_csv_atomic(df: Any, path: str) -> None:
    partial = path + ".partial"
    df.to_csv(partial)
    os.replace(partial, path)


def signal_summary(matrix: Any) -> tuple:
    """``(share of the finite score between distinct spots, number of non-finite scores)``.

    ``(None, 0)`` when there is no matrix. A slide where only the self-pairs are in range makes every
    cost zero, and COMMOT's cost normalisation then divides 0 by 0: the old default produced a total
    matrix of NaN on the SI_P12 slide, at status ok. Counted here so a run says so.
    """
    import numpy as np

    if matrix is None:
        return None, 0
    if hasattr(matrix, "tocoo"):
        coo = matrix.tocoo()
        values, rows, cols = np.asarray(coo.data, dtype=np.float64), coo.row, coo.col
    else:
        dense = np.asarray(matrix, dtype=np.float64)
        rows, cols = np.nonzero(dense)
        values = dense[rows, cols]
    finite = np.isfinite(values)
    n_bad = int((~finite).sum())
    total = float(values[finite].sum())
    if not total > 0:
        return 0.0, n_bad
    between = float(values[finite & (rows != cols)].sum())
    return between / total, n_bad


#: Measured 2026-10-05 (PERF-1) on the 4,035-spot lymph node Visium slide at the automatic 200 um threshold, on one
#: core (COMMOT's optimal transport is single-threaded, so the thread budget does not change it): 50 pairs 400 s,
#: 100 pairs 633 s, 200 pairs 889 s, all 279 that pass the filter 1,079 s -- the step that ran past ten minutes
#: live. Least-squares line through the four: 299 s fixed + 2.87 s a pair (max error 47 s). Scaled with spots.
COST_FIXED_S = 299.0
COST_PER_PAIR_S = 2.87
COST_SPOTS = 4035
#: What an interactive step aims for, whatever the step budget allows: about eight minutes.
TARGET_S = 480.0
#: The fewest pairs a capped run scores: below this the answer is not a communication analysis.
MIN_PAIRS = 20


def full_minutes(n_pairs: int, n_spots: int) -> int:
    """About how long scoring ``n_pairs`` on ``n_spots`` takes, in whole minutes, at the measured costs."""
    scale = max(0.05, n_spots / COST_SPOTS)
    return max(1, round((COST_FIXED_S + COST_PER_PAIR_S * n_pairs) * scale / 60))


def pairs_that_fit(budget_s: float, n_spots: int) -> int:
    """How many pairs fit an interactive step on ``n_spots`` spots: the smaller of half the step budget and
    :data:`TARGET_S`, less the fixed work, at the measured cost per pair. Never below :data:`MIN_PAIRS`."""
    scale = max(0.05, n_spots / COST_SPOTS)
    usable = max(0.0, min(budget_s * 0.5, TARGET_S) - COST_FIXED_S * scale)
    return max(MIN_PAIRS, int(usable // (COST_PER_PAIR_S * scale)))


def most_expressed_pairs(df_ligrec: Any, adata: Any, keep: int) -> Any:
    """The ``keep`` rows of ``df_ligrec`` whose ligand and receptor are both expressed in the most spots.

    A pair's coverage is the smallest fraction of spots expressing any of its genes (ligand and receptor, each
    split on ``_`` into subunits): a pair is only as present as its scarcest part. Ties keep the database order.
    """
    import numpy as np

    names = list(adata.var_names)
    index = {g: i for i, g in enumerate(names)}
    X = adata.X
    nonzero = np.asarray((X > 0).sum(axis=0)).ravel() / max(1, X.shape[0])

    def cover(gene_field: Any) -> float:
        parts = [g for g in str(gene_field).split("_") if g]
        vals = [float(nonzero[index[g]]) for g in parts if g in index]
        return min(vals) if vals and len(vals) == len(parts) else 0.0

    scores = [min(cover(row.iloc[0]), cover(row.iloc[1])) for _, row in df_ligrec.iterrows()]
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[: int(keep)]
    return df_ligrec.iloc[sorted(order)].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Modes: one 2D slide, per-section 2D, a bounded 3D block
# ---------------------------------------------------------------------------------------------

#: The explorer refuses a direction, matrix or dot-plot view over this many links. A copy of ``MAX_NNZ`` in
#: ``spatialomicsgym/viz/ccc/facets.py`` (this worker runs in the COMMOT env and cannot import the portal);
#: ``SOG_CCC_MAX_NNZ`` overrides it, for a test or a host whose explorer was configured otherwise.
EXPLORER_MAX_NNZ = 50_000_000
#: How many cells a 3D block may hold unless the caller says otherwise.
DEFAULT_BLOCK_MAX_CELLS = 40_000
#: The documented upper bound of the installed commot's collective optimal transport working set, as a multiple
#: of the dense distance matrix (its cost matrices, the masked copy and the transport plans per pair). An estimate.
OT_WORKING_SET_FACTOR = 3
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_GIB = float(1 << 30)


def explorer_link_cap() -> int:
    """``SOG_CCC_MAX_NNZ`` when it is a positive integer, else :data:`EXPLORER_MAX_NNZ`."""
    raw = (os.environ.get("SOG_CCC_MAX_NNZ") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return EXPLORER_MAX_NNZ
    return value if value > 0 else EXPLORER_MAX_NNZ


def _bytes_text(n_bytes) -> str:
    if n_bytes is None:
        return "an unknown amount (MemAvailable could not be read)"
    n_bytes = int(n_bytes)
    return f"{n_bytes:,} bytes ({n_bytes / _GIB:.2f} GiB)"


def expression_bytes(X: Any) -> int:
    """Bytes the expression matrix occupies: a sparse one's data, indices and indptr; a dense one's array."""
    import numpy as np

    if hasattr(X, "data") and hasattr(X, "indices") and hasattr(X, "indptr"):
        return int(np.asarray(X.data).nbytes + np.asarray(X.indices).nbytes + np.asarray(X.indptr).nbytes)
    return int(np.asarray(X).nbytes)


def memory_plan(n_cells: int, X: Any, available=None) -> dict:
    """``{dense_bytes, ot_working_set_bytes, expression_bytes, peak_estimate_bytes, mem_available_bytes}``.

    ``dense_bytes`` is upstream's float64 n x n distance matrix and its boolean mask (9 n^2). The peak is an
    estimate: that matrix, :data:`OT_WORKING_SET_FACTOR` times it for the collective optimal transport, and the
    expression matrix. ``mem_available_bytes`` is ``MemAvailable`` from ``/proc/meminfo``, bounded by this
    process's cgroup limit where one is set (``worker_utils.available_memory_bytes``); None when unreadable.
    """
    dense = dense_distance_bytes(n_cells)
    ot = OT_WORKING_SET_FACTOR * dense
    expr = expression_bytes(X)
    if available is None:
        available = available_memory_bytes()
    return {
        "dense_bytes": int(dense),
        "ot_working_set_bytes": int(ot),
        "expression_bytes": int(expr),
        "peak_estimate_bytes": int(dense + ot + expr),
        "mem_available_bytes": int(available) if available is not None else None,
    }


def check_peak_memory(plan: dict, what: str = "the block") -> None:
    """Refuse before any transport when the estimated peak exceeds the memory available, each number stated."""
    available = plan.get("mem_available_bytes")
    if available is None or plan["peak_estimate_bytes"] <= available:
        return
    raise MemoryError(
        f"commot: {what}'s run would peak at about {_bytes_text(plan['peak_estimate_bytes'])} (an estimate), over "
        f"the {_bytes_text(available)} available here. The three numbers: COMMOT's dense distance matrix needs "
        f"{_bytes_text(plan['dense_bytes'])}; the collective optimal transport's working set, estimated at "
        f"{OT_WORKING_SET_FACTOR}x that, {_bytes_text(plan['ot_working_set_bytes'])}; the expression matrix "
        f"{_bytes_text(plan['expression_bytes'])}. Choose fewer cells: a few adjacent sections and a bounding box."
    )


def stored_nnz_estimate(n_links: int, n_cells: int) -> int:
    """The entries the explorer will read from a result's obsp matrix, at most: COMMOT stores sender x receiver
    for both directions of every pair within the threshold, plus the diagonal -- ``2 * n_links + n_cells``.

    An upper bound: COMMOT drops the links its transport gives zero weight. Measured on the Zhuang fixture
    (2,589 cells, 150 um, 167,672 pairs): ``obsp['commot-CellChat-total-total'].nnz`` = 87,434, 0.26 of the
    estimate of 337,933. So a run under the cap by this estimate is under it in the explorer, and a warning
    can only err towards caution.
    """
    return 2 * int(n_links) + int(n_cells)


def link_cap_message(n_links: int, stored: int, threshold: float, unit_text: str, cap: int) -> str:
    return (
        f"{n_links:,} cell pairs lie within dis_thr = {threshold:.4g} {unit_text}, so the result's obsp matrices may "
        f"store up to {stored:,} entries (2 x pairs + cells: both directions and the diagonal), over the explorer's "
        f"cap: direction/matrix/dotplot will be refused there (cap {cap:,} stored entries; SOG_CCC_MAX_NNZ). A "
        "shorter dis_thr_um or a smaller block keeps the result explorable."
    )


def in_plane_axes(axis_map: Any) -> tuple:
    """``([a, b], stacking column, sentence)`` for a 3-column frame.

    The stacking axis is the column whose ``axis_map`` entry says "stack" (Zhuang: x, "CCF anterior-posterior
    (section stacking axis)", so the in-plane axes are CCF y and z); without such an entry it is column 2 (z),
    the contract's default, and the in-plane axes are the frame's x and y.
    """
    stack = None
    names = {}
    if hasattr(axis_map, "items"):
        for key, text in axis_map.items():
            idx = _AXIS_INDEX.get(str(key).strip().lower())
            if idx is None:
                continue
            names[idx] = str(text)
            if stack is None and "stack" in str(text).lower():
                stack = idx
    source = "the frame's axis_map names it the section stacking axis"
    if stack is None:
        stack, source = 2, "no axis_map entry names a stacking axis, so the third column (z) is taken"
    plane = [i for i in range(3) if i != stack]
    labels = ["xyz"[i] + (f" ({names[i]})" if i in names else "") for i in plane]
    sentence = (
        f"block_bbox_um is [{'xyz'[plane[0]]}0, {'xyz'[plane[1]]}0, {'xyz'[plane[0]]}1, {'xyz'[plane[1]]}1] over "
        f"columns {plane[0]} and {plane[1]} -- {labels[0]} and {labels[1]}; column {stack} is the stacking axis "
        f"({source})"
    )
    return plane, stack, sentence


def check_bbox(bbox: Any):
    """``None`` or four finite floats ``[a0, b0, a1, b1]`` with ``a0 < a1`` and ``b0 < b1``."""
    if bbox is None:
        return None
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError):
        raise ValueError(f"block_bbox_um={bbox!r} is not four numbers [x0, y0, x1, y1] in micrometres.") from None
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"block_bbox_um={bbox!r} must be four finite numbers [x0, y0, x1, y1] in micrometres.")
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError(f"block_bbox_um={values} must have x0 < x1 and y0 < y1 (lower corner first).")
    return values


def _box_that_fits(m: Any, plane_um: Any, cap: int) -> tuple:
    """``(box, cells held)``: an in-plane box around the median of the cells ``m`` that holds at most ``cap`` of them.

    The first guess scales the cells' extent by ``sqrt(0.9 * cap / count)`` (even density); the box then shrinks
    until it really fits, and is rounded inward to 0.1 um so the box printed holds no more than the box counted.
    """
    import numpy as np

    pts = plane_um[m]
    centre = np.median(pts, axis=0)
    half = 0.5 * np.ptp(pts, axis=0) * math.sqrt(min(1.0, 0.9 * cap / max(int(m.sum()), 1)))

    def held_by(box):
        return int(
            (
                m
                & (plane_um[:, 0] >= box[0])
                & (plane_um[:, 0] <= box[2])
                & (plane_um[:, 1] >= box[1])
                & (plane_um[:, 1] <= box[3])
            ).sum()
        )

    while True:
        box = [
            math.ceil(float(centre[0] - half[0]) * 10) / 10,
            math.ceil(float(centre[1] - half[1]) * 10) / 10,
            math.floor(float(centre[0] + half[0]) * 10) / 10,
            math.floor(float(centre[1] + half[1]) * 10) / 10,
        ]
        held = held_by(box)
        if held <= cap or not (half > 0).any():
            return box, held
        half = half * 0.9


#: How many adjacent sections a cropped block offer keeps: enough for a section to have a neighbour on each side.
OFFER_SECTIONS = 3


def offer_block(order: list, labels: Any, plane_um: Any, inside: Any, cap: int) -> str:
    """A block that fits ``cap``, keeping what a 3D block is for -- cells of adjacent sections:

    1. the most adjacent WHOLE sections whose cells fit, when that is two or more sections;
    2. else up to :data:`OFFER_SECTIONS` adjacent sections of ``order`` (the window holding the most cells)
       cropped to an in-plane box that fits, when every one of them keeps a cell (COMMOT-1: a whole single
       section was offered here, a 3D block with no cross-section pair at all);
    3. else a box in the one section with the fewest cells.
    """
    import numpy as np

    counts = [int(((labels == s) & inside).sum()) for s in order]
    best = None
    for i in range(len(order)):
        total = 0
        for j in range(i, len(order)):
            total += counts[j]
            if total > cap:
                break
            cand = (j - i + 1, total, i, j)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is not None and best[0] >= 2:
        _, total, i, j = best
        return f"for example block_sections={list(order[i : j + 1])} ({total:,} cells)"
    if not any(counts):
        return "for example a bounding box holding fewer cells"
    width = min(OFFER_SECTIONS, len(order))
    if width >= 2:
        start = max(range(len(order) - width + 1), key=lambda i: sum(counts[i : i + width]))
        window = list(order[start : start + width])
        m = np.isin(labels, window) & inside
        box, held = _box_that_fits(m, plane_um, cap)
        in_box = (
            m
            & (plane_um[:, 0] >= box[0])
            & (plane_um[:, 0] <= box[2])
            & (plane_um[:, 1] >= box[1])
            & (plane_um[:, 1] <= box[3])
        )
        if held >= 2 and all(bool((in_box & (labels == s)).any()) for s in window):
            return f"for example block_sections={window} with block_bbox_um={box} ({held:,} cells)"
    if best is not None:
        _, total, i, j = best
        return f"for example block_sections={list(order[i : j + 1])} ({total:,} cells)"
    k = int(np.argmin([c if c else np.inf for c in counts]))
    m = (labels == order[k]) & inside
    box, held = _box_that_fits(m, plane_um, cap)
    return f"for example block_sections=[{order[k]!r}] with block_bbox_um={box} ({held:,} cells)"


def resolve_requested_threshold(dis_thr: float, unit: str, dis_thr_um: float) -> tuple:
    """``(dis_thr, unit)`` after ``dis_thr_um``: a positive ``dis_thr_um`` is the threshold in micrometres, and the
    old ``dis_thr``/``dis_thr_unit`` pair is used only when it is 0 (bare 2D data without units)."""
    um = float(dis_thr_um or 0.0)
    if math.isnan(um):
        raise ValueError("dis_thr_um is NaN; pass a distance in micrometres, or 0 for the automatic 200 um.")
    if um <= 0:
        return float(dis_thr), unit
    if float(dis_thr) > 0 and not (unit == "um" and float(dis_thr) == um):
        raise ValueError(
            f"dis_thr_um={um:g} and dis_thr={float(dis_thr):g} ({unit}) were both given; pass one. dis_thr_um is the "
            "threshold in micrometres for a frame with units; dis_thr/dis_thr_unit are for bare 2D data without units."
        )
    return um, "um"


#: The refusal for ``dis_thr_um`` on two columns with no declared units. Never the old ``dis_thr_unit`` message: a
#: threshold in micrometres is converted by the frame's declaration, not guessed from a slide's scalefactors.
UNDECLARED_UNITS_MSG = (
    "dis_thr_um needs the coordinates' units: declare them (uns['spatial_3d']['frames']['spatial'] with xy_units, "
    "e.g. via spatial3d.contract.write_frame) or pass dis_thr in coordinate units."
)


def check_threshold_units(dis_thr_um: float, frame: Any) -> None:
    """Refuse ``dis_thr_um > 0`` when the frame's units are not declared (``units_per_axis`` holds a None)."""
    if float(dis_thr_um or 0.0) <= 0:
        return
    units = list(getattr(frame, "units_per_axis", ()) or ())
    if not units or any(u is None for u in units):
        raise ValueError(UNDECLARED_UNITS_MSG)


def _section_folder(label: str) -> str:
    return "section_" + (re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("._") or "unnamed")


def _coordinate_digest(arr: Any) -> str:
    """``spatialomicsgym.spatial3d.contract.coordinate_digest``, restated (this worker cannot import the portal)."""
    import hashlib

    import numpy as np

    a = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def _declare_spatial_um(adata: Any, declaration: dict) -> None:
    """Declare the micrometre ``obsm['spatial']`` this result carries: ``uns['spatial_3d']['frames']['spatial']``
    = ``declaration`` and ``original_digest`` refreshed to the coordinates written. Without it a rescaled
    millimetre input kept saying "mm" over micrometre numbers."""
    spatial_3d = dict(adata.uns.get("spatial_3d") or {})
    frames = dict(spatial_3d.get("frames") or {})
    frames["spatial"] = dict(declaration)
    spatial_3d["frames"] = frames
    spatial_3d["original_digest"] = _coordinate_digest(adata.obsm["spatial"])
    adata.uns["spatial_3d"] = spatial_3d


#: The declaration of a 2D ``spatial`` the worker rescaled to micrometres.
RAW_UM_DECLARATION = {"role": "raw", "xy_units": "um", "z_source": "unknown", "z_units": "unknown"}


def _frame_scale(frame: Any, coords_key: str):
    """``(1.0, sentence)`` when the frame's units were converted to micrometres, else None (raw units)."""
    units = list(getattr(frame, "units_per_axis", ()) or ())
    if units and all(u is not None for u in units):
        return 1.0, f"obsm['{coords_key}'] declares its units, converted to micrometres"
    return None


def _score(
    adata: Any,
    coords: Any,
    output_dir: str,
    out: Any,
    database: str,
    species_arg: str,
    database_name: str,
    dis_thr: float,
    unit: str,
    heteromeric: bool,
    pathway_sum: bool,
    normalize: bool,
    max_lr_pairs: int,
    time_budget_s: float,
    n_spots_supplied: int,
    n_spots_off_tissue: int,
    frame_scale=None,
    peak_check: bool = False,
    refuse_over_cap: bool = False,
    what: str = "the input",
) -> dict:
    """Score one set of cells whose ``obsm['spatial']`` holds ``coords``; fill ``out``; return
    ``{analysis, dis_thr, dis_thr_um, n_links, stored_nnz_estimate}``.

    ``frame_scale`` = ``(coordinate units per um, source)`` when the caller knows the units (a declared frame,
    converted to micrometres); otherwise the slide's scalefactors are read as before. ``peak_check`` refuses on
    the estimated total peak (:func:`memory_plan`) instead of the dense matrix alone. Before any transport the
    links within the threshold are counted (the number ``cKDTree.query_pairs`` returns, without materialising the
    pairs) and compared with the explorer's cap: a warning, or a refusal with ``refuse_over_cap``.
    """
    import commot as ct
    import numpy as np
    import pandas as pd
    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)
    coords = np.asarray(coords, dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("obsm['spatial'] holds NaN or infinite coordinates; COMMOT needs a position for every spot.")
    # COMMOT's own distances when the file carries them (upstream reads obsp['spatial_distance'] first).
    dmat = precomputed_distance_matrix(adata.obsp, adata.n_obs)
    if peak_check:
        plan = memory_plan(adata.n_obs, adata.X)
        check_peak_memory(plan, what)
        out.set_data(memory=plan)
    else:
        check_dense_memory(adata.n_obs, precomputed=dmat is not None)

    # ---- Genes: symbols, unique ----
    gene_report = use_symbol_var_names(adata)
    renamed = make_names_unique_and_report(adata, axes=("var",))

    # ---- Expression ----
    expression_warning = check_expression(adata.X, normalize, adata.uns)
    if expression_warning:
        out.add_warning(expression_warning)
    if normalize:
        log("Normalizing data with sc.pp.normalize_total + sc.pp.log1p ...")
        sc.pp.normalize_total(adata, inplace=True)
        sc.pp.log1p(adata)
    else:
        log("Skipping normalization (assuming adata is already preprocessed).")

    # ---- Distance threshold, in the units of the distances COMMOT will measure ----
    probe = neighbour_stats(coords, 0.0)
    if frame_scale is not None:
        units_per_um, scale_source = frame_scale
    else:
        units_per_um, scale_source = coordinate_scale(adata.uns, probe["median_nn"])
    space = OBSM_SPACE
    matrix_is_euclidean = None
    threshold_units_per_um = units_per_um
    if dmat is None:
        threshold, threshold_um, basis = resolve_dis_thr(dis_thr, unit, units_per_um, scale_source)
        stats = neighbour_stats(coords, threshold)
    else:
        space = OBSP_SPACE
        matrix_is_euclidean, difference = matrix_matches_coords(dmat, coords)
        log(
            f"The input carries {OBSP_SPACE}; COMMOT measures dis_thr on it "
            + (f"(it equals the Euclidean distances of {OBSM_SPACE})." if matrix_is_euclidean else f"({difference}).")
        )
        if matrix_is_euclidean:
            threshold, threshold_um, basis = resolve_dis_thr(dis_thr, unit, units_per_um, scale_source)
            basis += f"; COMMOT measures it on the input's {OBSP_SPACE}, the Euclidean distances of {OBSM_SPACE}"
        elif dis_thr <= 0 or unit == "um":
            raise ValueError(foreign_matrix_message(dis_thr, unit, difference, matrix_neighbour_stats(dmat, 0.0)))
        else:
            threshold, threshold_um, threshold_units_per_um = float(dis_thr), None, None
            basis = (
                f"{threshold:g} {OBSP_SPACE} units, as passed: COMMOT uses the input's precomputed distance matrix, "
                f"which is not the Euclidean distance of {OBSM_SPACE}, so no micron equivalent is known"
            )
        stats = matrix_neighbour_stats(dmat, threshold)
    log(
        f"dis_thr effective = {threshold:.6g} {space} units ({basis}); median NN spacing "
        f"{stats['median_nn']:.6g}; {stats['n_spots_with_partner']}/{stats['n_spots']} spots have a partner; "
        f"{stats['n_spot_pairs_within']} spot pairs within it."
    )
    if stats["n_spots_with_partner"] == 0:
        raise ValueError(
            isolated_threshold_message(
                threshold,
                basis,
                stats,
                threshold_units_per_um,
                space=space,
                convertible=dmat is None or bool(matrix_is_euclidean),
            )
        )
    # ---- Links at the threshold, against the explorer's cap (the cell cap never implies this one) ----
    n_links = int(stats["n_spot_pairs_within"])
    stored = stored_nnz_estimate(n_links, adata.n_obs)
    cap = explorer_link_cap()
    out.set_data(n_links=n_links, stored_nnz_estimate=stored, explorer_link_cap=cap)
    if stored > cap:
        unit_text = "um" if threshold_units_per_um == 1.0 else f"{space} units"
        cap_text = link_cap_message(n_links, stored, threshold, unit_text, cap)
        if refuse_over_cap:
            raise ValueError(f"commot: {cap_text} refuse_over_cap=True, so the run stops before any transport.")
        out.add_warning(cap_text)
        out.set_data(link_cap_warning=cap_text)
    partner_fraction = stats["n_spots_with_partner"] / float(stats["n_spots"])
    if partner_fraction < _LOW_PARTNER_FRACTION:
        out.add_warning(
            f"Only {stats['n_spots_with_partner']} of {stats['n_spots']} spots have another spot within "
            f"dis_thr={threshold:.4g} (median nearest-neighbour spacing {stats['median_nn']:.4g}); the rest can "
            "only signal to themselves."
        )
    if units_per_um is None and dis_thr <= 0:
        out.add_warning(
            f"No micron scale was found ({scale_source}); the automatic threshold was applied as "
            f"{AUTO_DIS_THR_UM:g} obsm['spatial'] units on the assumption that the coordinates are microns."
        )

    # ---- LR database ----
    log(
        f"Building LR database via commot.pp.ligand_receptor_database(database='{database}', species='{species_arg}') ..."
    )
    df_db = ct.pp.ligand_receptor_database(database=database, species=species_arg)
    n_db = int(df_db.shape[0])
    log(f"Retrieved LR pairs: {n_db} rows ({LR_SIGNALING_TYPE})")

    # Always filtered with heteromeric=True: a plain gene splits into itself, so this rule is right for both
    # modes, and upstream's heteromeric=False branch compares filter_criteria with a misspelt 'min_cell_prc'
    # and would keep nothing. spatial_communication(heteromeric=False) then drops the complexes itself.
    log("Filtering LR database to genes detected in the dataset with commot.pp.filter_lr_database ...")
    try:
        df_ligrec = ct.pp.filter_lr_database(df_db, adata, heteromeric=True, min_cell_pct=LR_MIN_CELL_PCT)
    except Exception as exc:
        raise RuntimeError(
            f"commot.pp.filter_lr_database failed ({type(exc).__name__}: {exc}); the run stops rather than score "
            "the unfiltered database as if it had been filtered."
        ) from exc
    n_filtered = int(df_ligrec.shape[0])
    log(f"LR pairs after filtering: {n_filtered} rows")
    if n_filtered == 0:
        raise ValueError(no_pairs_message(df_db, adata.var_names, database, species_arg, gene_report))
    if time_budget_s > 0:
        fits = pairs_that_fit(time_budget_s, int(adata.n_obs))
        log(f"time budget {time_budget_s:.0f}s for {adata.n_obs} spots: about {fits} pairs fit")
        max_lr_pairs = min(max_lr_pairs, fits) if max_lr_pairs else fits
    if max_lr_pairs and n_filtered > max_lr_pairs:
        # A bounded default for an interactive run (2026-10-05, PERF-1): on the 4,035-spot lymph node slide the full
        # filtered CellChat set kept one step running past ten minutes. The pairs kept are the ones expressed in the
        # most spots (the scarcer of ligand and receptor, subunits included); the run says how many it set aside.
        df_ligrec = most_expressed_pairs(df_ligrec, adata, max_lr_pairs)
        out.add_warning(
            f"To fit the time this step has, COMMOT scored the {max_lr_pairs} ligand-receptor pairs expressed in the "
            f"most spots, of the {n_filtered} that passed the expression filter. Scoring all of them would take about "
            f"{full_minutes(n_filtered, int(adata.n_obs))} minutes, longer than one interactive step allows; a "
            "pathway carried only by the rarer pairs may be missing."
        )
        log(f"LR pairs capped to the {max_lr_pairs} most widely expressed of {n_filtered}")
        n_filtered = int(df_ligrec.shape[0])
    n_usable = usable_pair_count(df_ligrec, adata.var_names, heteromeric)
    if n_usable == 0:
        raise ValueError(
            f"heteromeric=False keeps only pairs whose ligand and receptor are single genes, and none of the "
            f"{n_filtered} pairs that passed the expression filter are; pass heteromeric=True."
        )

    # A caller's obsp['spatial_distance'] stays in place: COMMOT uses it, as upstream documents, and dis_thr
    # was resolved and checked against it above.
    log("Running commot.tl.spatial_communication ...")
    ct.tl.spatial_communication(
        adata,
        database_name=database_name,
        df_ligrec=df_ligrec,
        dis_thr=threshold,
        heteromeric=heteromeric,
        pathway_sum=pathway_sum,
    )
    log("COMMOT spatial_communication completed.")

    info = adata.uns.get(f"commot-{database_name}-info")
    used_table = info.get("df_ligrec") if hasattr(info, "get") else None
    n_lr_pairs = int(used_table.shape[0]) if used_table is not None and hasattr(used_table, "shape") else n_usable
    total_key = f"commot-{database_name}-total-total"
    between, n_nonfinite = signal_summary(adata.obsp[total_key] if total_key in adata.obsp else None)
    if n_nonfinite:
        out.add_warning(
            f"COMMOT returned {n_nonfinite} non-finite (NaN/inf) scores in obsp['{total_key}']; "
            "summary.fraction_of_signal_between_spots counts the finite scores only."
        )
    if between is not None and between == 0.0:
        out.add_warning(
            "COMMOT assigned no communication between distinct spots: every finite score lies on the diagonal "
            "(within-spot) or the total is zero."
        )

    # ---- Save annotated AnnData ----
    annotated_h5ad = os.path.join(output_dir, f"commot_{database_name}_results.h5ad")
    _write_h5ad_atomic(adata, annotated_h5ad)
    log(f"Saved annotated AnnData to: {annotated_h5ad}")

    # ---- Export sender/receiver summaries from obsm ----
    obsm_sender_key = f"commot-{database_name}-sum-sender"
    obsm_receiver_key = f"commot-{database_name}-sum-receiver"

    sender_csv = None
    receiver_csv = None

    if obsm_sender_key in adata.obsm:
        log(f"Found sender summary in adata.obsm['{obsm_sender_key}']")
        df_sender = adata.obsm[obsm_sender_key]
        if not isinstance(df_sender, pd.DataFrame):
            df_sender = pd.DataFrame(df_sender, index=adata.obs_names)
        sender_csv = os.path.join(output_dir, f"commot_{database_name}_sum_sender.csv")
        _write_csv_atomic(df_sender, sender_csv)
        log(f"Saved sender summary CSV to: {sender_csv}")
    else:
        log(f"Sender key '{obsm_sender_key}' not found in adata.obsm")

    if obsm_receiver_key in adata.obsm:
        log(f"Found receiver summary in adata.obsm['{obsm_receiver_key}']")
        df_receiver = adata.obsm[obsm_receiver_key]
        if not isinstance(df_receiver, pd.DataFrame):
            df_receiver = pd.DataFrame(df_receiver, index=adata.obs_names)
        receiver_csv = os.path.join(output_dir, f"commot_{database_name}_sum_receiver.csv")
        _write_csv_atomic(df_receiver, receiver_csv)
        log(f"Saved receiver summary CSV to: {receiver_csv}")
    else:
        log(f"Receiver key '{obsm_receiver_key}' not found in adata.obsm")

    # ---- Build JSON result using WorkerOutput ----
    median_um = stats["median_nn"] / threshold_units_per_um if threshold_units_per_um is not None else None
    out.set_data(
        n_spots=int(adata.n_obs),
        n_genes=int(adata.n_vars),
        n_spots_supplied=int(n_spots_supplied),
        n_spots_off_tissue_dropped=int(n_spots_off_tissue),
    )
    out.add_output_files(
        {
            "annotated_h5ad": annotated_h5ad,
            "sender_summary_csv": sender_csv,
            "receiver_summary_csv": receiver_csv,
        }
    )
    out.add_params(
        {
            "lr_database": database,
            "species": species_arg,
            "database_name": database_name,
            "dis_thr": threshold,
            "dis_thr_unit": unit,
            "heteromeric": heteromeric,
            "pathway_sum": pathway_sum,
            "normalize": normalize,
        }
    )
    out.add_params(
        {
            "dis_thr_requested": float(dis_thr),
            "dis_thr_um": threshold_um,
            "dis_thr_basis": basis,
            "coordinate_units_per_um": units_per_um,
            "coordinate_scale_source": scale_source,
            "distance_source": space,
            "distance_matrix_is_obsm_euclidean": matrix_is_euclidean,
            "lr_signaling_type": LR_SIGNALING_TYPE,
            "lr_min_cell_pct": LR_MIN_CELL_PCT,
            # How many pairs the run was cut to when it had to fit a step (PERF-1); None when every pair was scored.
            "lr_pairs_scored_cap": max_lr_pairs or None,
            "gene_id_column": gene_report["gene_id_column"],
        }
    )
    out.add_params(identifier_rename_params(renamed))
    record_method(out, METHOD_NAME)
    out.set_summary(
        n_lr_pairs=n_lr_pairs,
        n_lr_pairs_in_database=n_db,
        n_lr_pairs_after_expression_filter=n_filtered,
        dis_thr_effective=threshold,
        dis_thr_um=threshold_um,
        median_nn_spacing=stats["median_nn"],
        median_nn_spacing_um=median_um,
        n_spots_with_partner=stats["n_spots_with_partner"],
        n_spot_pairs_within_dis_thr=stats["n_spot_pairs_within"],
        mean_partners_per_spot=stats["mean_partners_per_spot"],
        fraction_of_signal_between_spots=between,
        n_nonfinite_scores=n_nonfinite,
        n_genes_named_by_symbol=gene_report["n_genes_named_by_symbol"],
        n_duplicate_symbols_kept_as_ids=gene_report["n_duplicate_symbols_kept_as_ids"],
    )
    labelled_as = (
        database_name if database_name in SUPPORTED_LR_DATABASES else _LR_DATABASE_ALIASES.get(database_name.lower())
    )
    if labelled_as is not None and labelled_as != database:
        out.add_warning(
            f"database_name={database_name!r} labels results computed from lr_database={database!r}: output files "
            f"and AnnData keys say {database_name} while the pairs come from {database}. Pass "
            f"database_name={database!r} to label them truthfully."
        )

    um_text = f" = {threshold_um:.4g} um" if threshold_um is not None else ""
    spacing_um = f" ({median_um:.1f} um)" if median_um is not None else ""
    between_text = (
        f" {between:.1%} of the total communication score is between distinct spots." if between is not None else ""
    )
    symbol_text = ""
    if gene_report["gene_id_column"]:
        symbol_text = (
            f" var_names were Ensembl IDs; {gene_report['n_genes_named_by_symbol']} genes were named by their symbol "
            f"from var['{gene_report['gene_id_column']}'] to match the database."
        )
    matrix_text = ""
    if dmat is not None:
        matrix_text = (
            f" Distances are the input's {OBSP_SPACE}, which COMMOT uses in place of {OBSM_SPACE} when present; it "
            "is kept unchanged in the annotated h5ad."
        )
    tissue_text = ""
    if n_spots_off_tissue:
        tissue_text = (
            f" {n_spots_off_tissue} of the {n_spots_supplied} spots supplied have obs['in_tissue'] == 0 (background "
            "outside the tissue) and were left out before normalisation and the ligand-receptor filter, so the "
            f"annotated h5ad and the sender/receiver tables hold the {adata.n_obs} in-tissue spots only."
        )
    analysis = (
        f"COMMOT (collective optimal transport) scored cell-cell communication across {adata.n_obs} spots through "
        f"{n_lr_pairs} ligand-receptor pairs of the {database} database ({species_arg}, {LR_SIGNALING_TYPE} only: "
        f"{n_db} pairs, {n_filtered} with ligand and receptor detected in >= {LR_MIN_CELL_PCT:.0%} of spots, "
        f"{n_lr_pairs} analysed{'' if heteromeric else ' as single genes, heteromeric=False'}).{matrix_text} Distance "
        f"threshold {threshold:.4g} {space} units{um_text} ({basis}); the median nearest-neighbour spacing is "
        f"{stats['median_nn']:.4g} units{spacing_um}, and {stats['n_spots_with_partner']} of {stats['n_spots']} "
        f"spots have at least one other spot within the threshold ({stats['mean_partners_per_spot']:.1f} on "
        f"average).{between_text}{tissue_text}{symbol_text}{identifier_rename_note(renamed)}"
    )
    out.set_analysis(analysis)
    return {
        "analysis": analysis,
        "dis_thr": threshold,
        "dis_thr_um": threshold_um,
        "n_links": n_links,
        "stored_nnz_estimate": stored,
    }


def run_spatial_communication(
    st_h5ad: str,
    output_dir: str,
    lr_database: str,
    species: str,
    database_name: str,
    dis_thr: float,
    heteromeric: bool,
    pathway_sum: bool,
    normalize: bool,
    dis_thr_unit: str = "coordinates",
    max_lr_pairs: int = 0,
    time_budget_s: float = 0.0,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key=None,
    dis_thr_um: float = 0.0,
    block_sections: Any = "all",
    block_bbox_um: Any = None,
    block_max_cells: int = DEFAULT_BLOCK_MAX_CELLS,
    refuse_over_cap: bool = False,
) -> dict[str, Any]:
    """
    Run COMMOT spatial_communication on a spatial AnnData (.h5ad), in one of three modes.

    The coordinates are read once, by ``worker_utils.spatial_frame(adata, coords_key, dims, section_key)``, which
    refuses a 2D run that would overlay sections, a 3-column ``spatial``, and a 3D frame without units or with a
    rank z, each naming both ways out.

    * ``dims=2`` without ``section_key`` -- one 2D slide (``params.mode = "2d"``), as before: ``dis_thr`` is in
      obsm['spatial'] units (``dis_thr_unit='coordinates'``) or microns (``dis_thr_unit='um'``, converted with the
      slide's scalefactors); ``dis_thr <= 0`` means 200 um. When the input carries ``obsp['spatial_distance']``,
      COMMOT measures on that matrix, so an explicit ``dis_thr`` is in its units, and microns are converted only
      when it is the Euclidean distance of ``obsm['spatial']``.
    * ``dims=2`` with ``section_key`` -- per-section 2D (``"per-section-2d"``): one ``section_<label>/`` folder per
      section with the usual three files and its own provenance.
    * ``dims=3`` -- a bounded 3D block (``"3d"``) in the frame's micrometres: the cells of ``block_sections``
      inside ``block_bbox_um`` (over the frame's two in-plane axes, see :func:`in_plane_axes`), at most
      ``block_max_cells`` of them, refused over that with the count, the memory and a block that fits. The result
      h5ad's ``obsm['spatial']`` holds the block's three micrometre columns and ``uns['spatial_3d']['frames']
      ['spatial']`` declares them (role aligned) -- the block result's own convention, an exception to the
      contract's two-column ``spatial``, so the explorer reads it as 3D. ``commot_block.json`` says what was
      scored.

    ``dis_thr_um`` > 0 is the threshold in micrometres (0 = automatic 200 um) for a frame with units; the old
    ``dis_thr``/``dis_thr_unit`` pair remains for bare 2D data without units. Before any transport the run states
    the dense matrix, the estimated peak and the memory available (refusing when the peak does not fit; 3D and
    per-section modes), and the number of links within the threshold against the explorer's cap
    (``SOG_CCC_MAX_NNZ``, default 50,000,000): a warning, or a refusal with ``refuse_over_cap``. **The cell cap
    never implies the link cap**: 40,000 cells at a generous threshold can still exceed it.

    Spots with ``obs['in_tissue'] == 0`` (the background a CELLxGENE Visium export carries) are left out
    right after loading, before the memory check, normalisation and the ligand-receptor filter, and the
    payload reports how many (``params.in_tissue_filter``, ``data.n_spots_off_tissue_dropped``).
    """
    import numpy as np

    # ---- Arguments that can be checked before anything is loaded ----
    database, species_arg = resolve_lr_database(lr_database, species)
    database_name = check_database_name(database_name)
    unit = resolve_dis_thr_unit(dis_thr_unit)
    if math.isnan(float(dis_thr)):
        raise ValueError("dis_thr is NaN; pass a distance, or 0 for the automatic 200 um threshold.")
    dis_thr, unit = resolve_requested_threshold(dis_thr, unit, dis_thr_um)
    dims = int(dims)
    if dims not in (2, 3):
        raise ValueError(f"commot: dims must be 2 or 3, not {dims}.")
    bbox = check_bbox(block_bbox_um)
    block_max_cells = int(block_max_cells)
    if block_max_cells < 2:
        raise ValueError(f"block_max_cells={block_max_cells} must be at least 2.")
    section_key = section_key or None

    import scanpy as sc

    os.makedirs(output_dir, exist_ok=True)

    log("Task = spatial_communication")
    log(f"st_h5ad      = {st_h5ad}")
    log(f"output_dir   = {output_dir}")
    log(f"lr_database  = {database} (requested {lr_database!r})")
    log(f"species      = {species_arg}")
    log(f"database_name= {database_name}")
    log(f"dis_thr      = {dis_thr} ({unit}; <= 0 means automatic {AUTO_DIS_THR_UM:g} um)")
    log(f"coords_key   = {coords_key}, dims = {dims}, section_key = {section_key}")
    log(f"heteromeric  = {heteromeric}")
    log(f"pathway_sum  = {pathway_sum}")
    log(f"normalize    = {normalize}")

    if not os.path.exists(st_h5ad):
        raise FileNotFoundError(f"Spatial h5ad not found: {st_h5ad}")

    log("Loading spatial AnnData...")
    adata = sc.read_h5ad(st_h5ad)
    log(f"Loaded ST data: n_spots={adata.n_obs}, n_genes={adata.n_vars}")

    out = WorkerOutput("commot", task="spatial_communication")

    # ---- Background spots: left out before anything is measured or normalised ----
    # A CELLxGENE Visium export carries every array spot with obs['in_tissue'] 0/1. Kept, the glass
    # outside the tissue (ambient reads, normalised up to tissue depth) sent and received signal, and
    # filter_lr_database's 5%-of-spots rule counted it in the denominator: on Heart Fetal12W 104 of
    # CellChat's 1199 pairs passed on all 4992 spots against 149 on the 1983 in-tissue ones.
    adata, n_spots_supplied, n_spots_off_tissue = keep_in_tissue(adata, "spots")
    if n_spots_off_tissue:
        log(
            f"Left out {n_spots_off_tissue} of {n_spots_supplied} spots with obs['in_tissue'] == 0; "
            f"{adata.n_obs} in-tissue spots are analysed."
        )
    record_in_tissue(out, n_spots_supplied, n_spots_off_tissue)

    # ---- Coordinates: one frame, refused rather than misread ----
    coords, frame = spatial_frame(adata, coords_key, dims, section_key, "commot")
    check_threshold_units(dis_thr_um, frame)
    frame_info = frame.to_dict()
    common = {
        "database": database,
        "species_arg": species_arg,
        "database_name": database_name,
        "dis_thr": dis_thr,
        "unit": unit,
        "heteromeric": heteromeric,
        "pathway_sum": pathway_sum,
        "normalize": normalize,
        "max_lr_pairs": max_lr_pairs,
        "time_budget_s": time_budget_s,
    }
    out.add_params({"coords_key": coords_key, "dims": dims, "section_key": section_key, "frame": frame_info})

    if dims == 3:
        return _run_block(
            adata,
            coords,
            frame,
            output_dir,
            out,
            common,
            n_spots_supplied,
            n_spots_off_tissue,
            block_sections,
            bbox,
            block_max_cells,
            refuse_over_cap,
        )

    scale = _frame_scale(frame, coords_key)
    if section_key is not None:
        return _run_per_section(
            adata,
            coords,
            frame,
            scale,
            output_dir,
            out,
            common,
            n_spots_supplied,
            n_spots_off_tissue,
            refuse_over_cap,
        )

    # ---- One 2D slide, as COMMOT reads it (obsm['spatial']) ----
    if coords_key != "spatial" or scale is not None:
        adata.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    if scale is not None:
        _declare_spatial_um(adata, RAW_UM_DECLARATION)
    out.add_params({"mode": "2d"})
    _score(
        adata,
        coords,
        output_dir,
        out,
        n_spots_supplied=n_spots_supplied,
        n_spots_off_tissue=n_spots_off_tissue,
        frame_scale=scale,
        refuse_over_cap=refuse_over_cap,
        **common,
    )
    return out.to_dict()


def _run_per_section(
    adata, coords, frame, scale, output_dir, out, common, n_spots_supplied, n_spots_off_tissue, refuse_over_cap
):
    """Per-section 2D: one ``section_<label>/`` folder per section, each with its own provenance."""
    import numpy as np

    section_key = frame.section_key
    adata.obsm["_sog_coords"] = np.asarray(coords, dtype=np.float64)
    sections = {}

    def run_one(sub, label):
        import pandas as pd

        folder = os.path.join(output_dir, _section_folder(label))
        sub_coords = np.asarray(sub.obsm["_sog_coords"], dtype=np.float64)
        del sub.obsm["_sog_coords"]
        sub.obsm["spatial"] = sub_coords
        if scale is not None:
            _declare_spatial_um(sub, RAW_UM_DECLARATION)
        sub_out = WorkerOutput("commot", task="spatial_communication")
        sub_out.add_params(
            {"mode": "per-section-2d", "section_key": section_key, "section": label, "frame": frame.to_dict()}
        )
        log(f"Section {label!r}: {sub.n_obs} cells -> {folder}")
        _score(
            sub,
            sub_coords,
            folder,
            sub_out,
            n_spots_supplied=int(sub.n_obs),
            n_spots_off_tissue=0,
            frame_scale=scale,
            peak_check=True,
            refuse_over_cap=refuse_over_cap,
            what=f"section {label!r}",
            **common,
        )
        result = sub_out.to_dict()
        sections[label] = result
        s = result.get("summary", {})
        return pd.DataFrame(
            [
                {
                    "n_cells": int(sub.n_obs),
                    "n_links": result.get("data", {}).get("n_links"),
                    "stored_nnz_estimate": result.get("data", {}).get("stored_nnz_estimate"),
                    "dis_thr_effective": s.get("dis_thr_effective"),
                    "dis_thr_um": s.get("dis_thr_um"),
                    "n_lr_pairs": s.get("n_lr_pairs"),
                    "fraction_of_signal_between_spots": s.get("fraction_of_signal_between_spots"),
                    "folder": _section_folder(label),
                }
            ]
        )

    table = per_section(adata, section_key, run_one, "commot")
    table_csv = os.path.join(output_dir, "commot_sections.csv")
    _write_csv_atomic(table.set_index("section"), table_csv)

    thr_um = [v for v in table["dis_thr_um"].tolist() if v is not None and v == v]
    thr_text = f"dis_thr {thr_um[0]:g} µm" if thr_um else f"dis_thr {table['dis_thr_effective'].iloc[0]:.4g} units"
    title = f"per-section 2D ({len(table)} sections, {int(table['n_cells'].sum()):,} cells, {thr_text})"
    out.add_params({"mode": "per-section-2d", "sections": [str(s) for s in table["section"]]})
    out.set_data(
        n_spots=int(table["n_cells"].sum()),
        n_spots_supplied=int(n_spots_supplied),
        n_spots_off_tissue_dropped=int(n_spots_off_tissue),
        n_links={str(r.section): int(r.n_links) for r in table.itertuples()},
        n_links_total=int(table["n_links"].sum()),
    )
    out.add_output_files(
        {
            "output_dir": output_dir,
            "sections_csv": table_csv,
        }
    )
    for label, res in sections.items():
        for key, path in (res.get("output_files") or {}).items():
            out.add_output_file(f"{_section_folder(label)}/{key}", path)
    for label, res in sections.items():
        for w in res.get("warnings", []) or []:
            out.add_warning(f"section {label}: {w}")
    out.set_summary(
        title=title,
        n_sections=int(len(table)),
        per_section=json.loads(table.to_json(orient="records")),
    )
    out.set_analysis(
        f"{title}: each section was scored on its own, so no signal crosses a section boundary. "
        + " ".join(f"[{label}] {res.get('analysis', '')}" for label, res in sections.items())
    )
    return out.to_dict()


def cross_section_counts(total: Any, labels: Any, coords_um: Any, thr_um: float) -> dict:
    """What crosses a section boundary in a scored 3D block, counted from the section labels (COMMOT-2):

    ``n_cross_section_links`` / ``n_links_scored`` -- the off-diagonal entries of COMMOT's ``total-total`` that
    join cells of two different sections, of all of them (what the explorer draws and counts);
    ``n_cross_section_pairs_within`` -- the cell pairs within the threshold whose cells lie in two sections
    (the candidates); ``closest_cross_section_um`` -- the distance of the closest such pair, when none lies
    within the threshold (what a threshold must reach for a cross-section link to be possible).
    """
    import numpy as np
    from scipy.spatial import cKDTree

    labels = np.asarray(labels).astype(str)
    out = {"n_cross_section_links": 0, "n_links_scored": 0, "n_cross_section_pairs_within": 0}
    if total is not None:
        m = total.tocoo() if hasattr(total, "tocoo") else None
        if m is not None:
            off = m.row != m.col
            out["n_links_scored"] = int(off.sum())
            out["n_cross_section_links"] = int((labels[m.row[off]] != labels[m.col[off]]).sum())
    coords_um = np.asarray(coords_um, dtype=np.float64)
    if len(set(labels.tolist())) < 2:
        return out
    pairs = cKDTree(coords_um).query_pairs(float(thr_um), output_type="ndarray")
    if len(pairs):
        out["n_cross_section_pairs_within"] = int((labels[pairs[:, 0]] != labels[pairs[:, 1]]).sum())
    if out["n_cross_section_pairs_within"] == 0:
        closest = np.inf
        for s in np.unique(labels):
            own = labels == s
            dist, _ = cKDTree(coords_um[~own]).query(coords_um[own], k=1)
            closest = min(closest, float(np.min(dist)))
        out["closest_cross_section_um"] = round(closest, 1)
    return out


def cross_section_sentence(cross: dict, thr_um: float, n_sections: int) -> str:
    """The analysis' sentence on cross-section communication: said only when links cross, else that none do."""
    if n_sections < 2:
        return "The block holds one section, so no link can cross a section boundary."
    links, scored = cross["n_cross_section_links"], cross["n_links_scored"]
    if links > 0:
        share = links / scored if scored else 0.0
        return (
            "Inferred cross-section communication: a sender in one section may reach a receiver in an adjacent "
            f"one within the threshold -- {links:,} of the {scored:,} scored links ({share:.1%}) join two sections."
        )
    if cross["n_cross_section_pairs_within"] > 0:
        return (
            f"{cross['n_cross_section_pairs_within']:,} cell pairs in two different sections lie within the "
            "threshold, but COMMOT assigned none of them a signal: no cross-section link at this threshold."
        )
    closest = cross.get("closest_cross_section_um")
    reach = f"; the closest pair of cells in two sections is {closest:g} µm apart" if closest is not None else ""
    return (
        f"No pair of cells in two different sections lies within {thr_um:g} µm{reach}, so there is no "
        "cross-section link at this threshold: each section's links stay within it."
    )


def _run_block(
    adata,
    coords_um,
    frame,
    output_dir,
    out,
    common,
    n_spots_supplied,
    n_spots_off_tissue,
    block_sections,
    bbox,
    block_max_cells,
    refuse_over_cap,
):
    """A bounded 3D block in the frame's micrometres (see :func:`run_spatial_communication`)."""
    import numpy as np

    n_total = int(adata.n_obs)
    coords_um = np.asarray(coords_um, dtype=np.float64)
    column = frame.section_key
    order = list(frame.sections or [])
    labels = adata.obs[column].astype(str).to_numpy() if column else np.array([""] * n_total)

    # ---- Sections ----
    wanted = block_sections
    if wanted is None or (isinstance(wanted, str) and wanted.strip().lower() == "all"):
        chosen = list(order)
    else:
        if isinstance(wanted, str):
            wanted = [wanted]
        wanted = [str(s) for s in wanted]
        if len(wanted) == 1 and wanted[0].strip().lower() == "all":
            chosen = list(order)
        else:
            if not column:
                raise ValueError(
                    f"commot: block_sections={wanted} needs a section column; pass section_key=<column> naming the "
                    "obs column of section labels."
                )
            unknown = [s for s in wanted if s not in order]
            if unknown:
                raise ValueError(
                    f"commot: block_sections {unknown} are not labels of obs['{column}']; its sections, in stacking "
                    f"order, are {order}."
                )
            chosen = [s for s in order if s in set(wanted)]
    in_sections = np.isin(labels, chosen) if column else np.ones(n_total, dtype=bool)

    # ---- In-plane bounding box ----
    declared = _declared_frame(adata, frame.key) or {}
    plane, stack_col, axes_sentence = in_plane_axes(declared.get("axis_map"))
    plane_um = coords_um[:, plane]
    if bbox is None:
        inside = np.ones(n_total, dtype=bool)
    else:
        inside = (
            (plane_um[:, 0] >= bbox[0])
            & (plane_um[:, 0] <= bbox[2])
            & (plane_um[:, 1] >= bbox[1])
            & (plane_um[:, 1] <= bbox[3])
        )
    mask = in_sections & inside
    n = int(mask.sum())
    log(f"3D block: {n} of {n_total} cells (sections {chosen}, bbox {bbox}; {axes_sentence})")

    # ---- The cell cap, with the memory it stands for and a block that fits ----
    plan = memory_plan(n, adata.X[mask] if n else adata.X[:0])
    if n > block_max_cells:
        offer = (
            offer_block(chosen or order, labels, plane_um, inside, block_max_cells)
            if column
            else ("for example a bounding box holding fewer cells")
        )
        raise ValueError(
            f"commot: {n:,} cells; COMMOT's dense distance matrix needs {_bytes_text(plan['dense_bytes'])} and the "
            f"run's total peak about {_bytes_text(plan['peak_estimate_bytes'])} (an estimate: the matrix, "
            f"{OT_WORKING_SET_FACTOR}x it for the collective optimal transport, and the expression matrix), with "
            f"{_bytes_text(plan['mem_available_bytes'])} available here; choose ≤ {block_max_cells:,} cells: a few "
            f"adjacent sections and a bounding box ({offer}; {axes_sentence})."
        )
    if n < 2:
        raise ValueError(
            f"commot: the block holds {n} cell(s) (sections {chosen}, block_bbox_um={bbox}; {axes_sentence}); "
            "COMMOT needs at least 2. Widen the bounding box or name other sections."
        )

    # ---- The block copy: its micrometre coordinates are what COMMOT reads ----
    block = adata[mask].copy()
    block.obsm["spatial"] = coords_um[mask]
    if DISTANCE_KEY in block.obsp:
        del block.obsp[DISTANCE_KEY]
        out.add_warning(
            f"The input's {OBSP_SPACE} was left out of the block: COMMOT would measure on it instead of on the "
            "block's micrometre coordinates."
        )
    _declare_spatial_um(block, {"role": "aligned", "z_source": frame.z_source, "xy_units": "um", "z_units": "um"})
    block.obs["commot_block"] = True

    scored = _score(
        block,
        coords_um[mask],
        output_dir,
        out,
        n_spots_supplied=n_spots_supplied,
        n_spots_off_tissue=n_spots_off_tissue,
        frame_scale=(1.0, f"obsm['{frame.key}'] in its declared units, converted to micrometres"),
        peak_check=True,
        refuse_over_cap=refuse_over_cap,
        what="the block",
        **common,
    )

    # ---- What was scored ----
    thr_um = float(scored["dis_thr_um"])
    n_links = int(scored["n_links"])
    per_section_counts = {
        str(s): {"in_block": int((mask & (labels == s)).sum()), "total": int((labels == s).sum())} for s in order
    }
    # The sections the block scored (COMMOT-3: the frame named the input's sections, every one of them).
    scored_sections = [str(s) for s in chosen if per_section_counts[str(s)]["in_block"] > 0]
    block_frame = frame.to_dict()
    block_frame["sections"] = scored_sections
    cross = cross_section_counts(
        block.obsp.get(f"commot-{common['database_name']}-total-total"),
        labels[mask],
        coords_um[mask],
        thr_um,
    )
    record = {
        "sections": scored_sections,
        "bbox_um": bbox,
        "bbox_axes": axes_sentence,
        "in_plane_columns": plane,
        "stacking_column": stack_col,
        "n_cells": n,
        "n_cells_total": n_total,
        "n_cells_outside_block": n_total - n,
        "cells_per_section": per_section_counts,
        "dis_thr_um": thr_um,
        "frame": block_frame,
        "n_links": n_links,
        "stored_nnz_estimate": int(scored["stored_nnz_estimate"]),
        **cross,
        "mode": "3d",
        "memory": {k: plan[k] for k in ("dense_bytes", "peak_estimate_bytes", "mem_available_bytes")},
    }
    block_json = os.path.join(output_dir, "commot_block.json")
    partial = block_json + ".partial"
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(partial, block_json)

    title = f"3D block ({n:,} cells, {len(chosen)} sections, dis_thr {thr_um:g} µm)"
    out.add_params(
        {
            "mode": "3d",
            "frame": block_frame,
            "block_sections": record["sections"],
            "block_bbox_um": bbox,
            "block_bbox_axes": axes_sentence,
            "block_max_cells": block_max_cells,
            "refuse_over_cap": bool(refuse_over_cap),
        }
    )
    out.set_data(
        memory=record["memory"],
        n_cells_block=n,
        n_cells_total=n_total,
        n_cells_outside_block=n_total - n,
        **cross,
    )
    out.add_output_files({"block_json": block_json})
    out.set_summary(
        title=title, n_cells_block=n, n_links=n_links, stored_nnz_estimate=int(scored["stored_nnz_estimate"])
    )
    out.set_analysis(
        f"{title}: "
        + ("inferred cross-section communication -- " if cross["n_cross_section_links"] > 0 else "")
        + f"COMMOT scored the cells of sections {record['sections']}"
        + (f" inside block_bbox_um={bbox}" if bbox else "")
        + f" in obsm['{frame.key}'] converted to micrometres (z from {frame.z_source}); {n_links:,} cell pairs lie "
        f"within the threshold. {cross_section_sentence(cross, thr_um, len(scored_sections))} "
        f"{n_total - n:,} of the {n_total:,} cells are outside the block and were not scored. {scored['analysis']}"
    )
    return out.to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="COMMOT worker for spatial cell-cell communication.")
    parser.add_argument(
        "--task",
        type=str,
        default="spatial_communication",
        choices=["spatial_communication"],
        help="Task to run (currently only 'spatial_communication').",
    )
    parser.add_argument(
        "--st-h5ad",
        dest="st_h5ad",
        required=True,
        help=(
            "Path to spatial AnnData (.h5ad). Must contain obsm['spatial']. Spots with obs['in_tissue'] == 0 are "
            "left out and counted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        help="Output directory for COMMOT results.",
    )
    parser.add_argument(
        "--lr-database",
        dest="lr_database",
        default="CellChat",
        help="Ligand-receptor database: 'CellChat' (human, mouse, zebrafish) or 'CellPhoneDB_v4.0' (human, mouse).",
    )
    parser.add_argument(
        "--species",
        dest="species",
        default="human",
        help="Species for LR database: 'human', 'mouse', or 'zebrafish' (CellChat only).",
    )
    parser.add_argument(
        "--database-name",
        dest="database_name",
        default="CellChat",
        help="Name used to tag COMMOT outputs in AnnData (keys like 'commot-<database_name>-sum-sender').",
    )
    parser.add_argument(
        "--dis-thr",
        dest="dis_thr",
        type=float,
        default=0.0,
        help=(
            "Maximum signalling distance, in the units of obsm['spatial'] (full-resolution pixels for 10x Visium) "
            "or in microns with --dis-thr-unit um. 0 or less (default) = automatic: 200 um, converted with the "
            "slide's scalefactors when it has them, else 200 coordinate units. When the input carries "
            "obsp['spatial_distance'], COMMOT measures on that matrix, so the value is in its units (microns are "
            "converted only when it is the Euclidean distance of obsm['spatial'])."
        ),
    )
    parser.add_argument(
        "--time-budget-s",
        dest="time_budget_s",
        type=float,
        default=0.0,
        help=(
            "The time this step has, in seconds (the portal passes its step budget). When set, the pairs scored are "
            "capped to what fits it at the measured cost per pair; 0 (default) = no cap from time."
        ),
    )
    parser.add_argument(
        "--max-lr-pairs",
        dest="max_lr_pairs",
        type=int,
        default=0,
        help=(
            "Score at most this many ligand-receptor pairs: the ones expressed in the most spots. 0 (default) = all "
            "that pass the expression filter. A run that caps the set says so in its warnings."
        ),
    )
    parser.add_argument(
        "--dis-thr-unit",
        dest="dis_thr_unit",
        default="coordinates",
        help="Unit of --dis-thr: 'coordinates' (obsm['spatial'] units, default) or 'um' (needs Visium scalefactors).",
    )
    parser.add_argument(
        "--coords-key",
        dest="coords_key",
        default="spatial",
        help="obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as 'spatial_3d_aligned').",
    )
    parser.add_argument(
        "--dims",
        dest="dims",
        type=int,
        default=2,
        choices=[2, 3],
        help="2 (one slide, or per section with --section-key) or 3 (a bounded block in the frame's micrometres).",
    )
    parser.add_argument(
        "--section-key",
        dest="section_key",
        default=None,
        help="obs column of section labels: a 2D run on a multi-section file is per section; a 3D block selects by it.",
    )
    parser.add_argument(
        "--dis-thr-um",
        dest="dis_thr_um",
        type=float,
        default=0.0,
        help=(
            "Maximum signalling distance in micrometres, for a frame with units (a 2D spatial declared in "
            "uns['spatial_3d']['frames']['spatial'], else refused); 0 (default) = automatic 200 um, or "
            "--dis-thr/--dis-thr-unit for bare 2D data without units."
        ),
    )
    parser.add_argument(
        "--block-sections",
        dest="block_sections",
        nargs="+",
        default=["all"],
        help="3D block: the section labels to score, or 'all' (default).",
    )
    parser.add_argument(
        "--block-bbox-um",
        dest="block_bbox_um",
        nargs=4,
        type=float,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help=(
            "3D block: an in-plane bounding box in micrometres, [a0, b0, a1, b1] = lower corner then upper corner, "
            "over the frame's two in-plane axes in column order (the axis_map's stacking axis left out; without one, "
            "x and y). On the Zhuang CCF frame (stacked along CCF x) a is CCF y (dorsal-ventral) and b is CCF z "
            "(medial-lateral): --block-bbox-um 4900 1700 5600 2400 keeps CCF y 4900-5600 um and z 1700-2400 um."
        ),
    )
    parser.add_argument(
        "--block-max-cells",
        dest="block_max_cells",
        type=int,
        default=DEFAULT_BLOCK_MAX_CELLS,
        help="3D block: refuse a block of more cells than this (default 40000).",
    )
    parser.add_argument(
        "--refuse-over-cap",
        dest="refuse_over_cap",
        action="store_true",
        help="Refuse, instead of warning, when the links within the threshold exceed the explorer's cap.",
    )

    # Boolean flags with sensible defaults
    parser.set_defaults(heteromeric=True, pathway_sum=True, normalize=True)
    parser.add_argument(
        "--no-heteromeric",
        dest="heteromeric",
        action="store_false",
        help="Disable heteromeric receptor handling.",
    )
    parser.add_argument(
        "--no-pathway-sum",
        dest="pathway_sum",
        action="store_false",
        help="Disable pathway-level sum of communications.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Skip normalize_total/log1p and assume preprocessed data.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.task == "spatial_communication":
            result = run_spatial_communication(
                st_h5ad=args.st_h5ad,
                output_dir=args.output_dir,
                lr_database=args.lr_database,
                species=args.species,
                database_name=args.database_name,
                dis_thr=args.dis_thr,
                heteromeric=args.heteromeric,
                pathway_sum=args.pathway_sum,
                normalize=args.normalize,
                dis_thr_unit=args.dis_thr_unit,
                max_lr_pairs=max(0, int(args.max_lr_pairs or 0)),
                time_budget_s=max(0.0, float(args.time_budget_s or 0.0)),
                coords_key=args.coords_key,
                dims=args.dims,
                section_key=args.section_key,
                dis_thr_um=args.dis_thr_um,
                block_sections=args.block_sections,
                block_bbox_um=args.block_bbox_um,
                block_max_cells=args.block_max_cells,
                refuse_over_cap=args.refuse_over_cap,
            )
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        # SUCCESS: print JSON to STDOUT
        print(json.dumps(result, default=str))
    except Exception as e:
        log("ERROR:")
        traceback.print_exc(file=sys.stderr)
        WorkerOutput.emit_error("commot", str(e), task="spatial_communication")
        sys.exit(1)


if __name__ == "__main__":
    main()
