#!/usr/bin/env python3
"""mistyR multi-view spatial modeling MCP wrapper for SpatialOmicsLab.

The R worker (``tools/mistyr_worker.R``) reads ONE coordinates CSV and models one plane. MISTy's paraview
is a two-dimensional kernel, so a serial-section stack is run per section, and that loop lives here: with
``section_key`` this server splits the coordinates and expression CSVs by section, calls the worker once per
section with its own ``output_dir`` (``section_<label>/``), and writes the worker's four tables at the top
level with a leading ``section`` column. Without ``section_key`` a coordinates CSV holding two or more
sections is refused (a 2D run would overlay them), and ``dims=3`` is refused outright.
"""

import json
import os
import re
import sys
import tempfile
import time
from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "mistyr"
WORKER_RSCRIPT, WORKER_SCRIPT = get_worker_paths(
    "MISTYR",
    "/opt/conda/envs/mistyr/bin/Rscript",
    "/workspace/epic-fermat/agent/tools/mistyr_worker.R",
)

mcp = create_mcp(TOOL_NAME)

#: The refusal for ``dims=3``: MISTy's paraview is a two-dimensional kernel over a coordinates CSV.
TWO_D_ONLY = "MISTy builds its neighbourhood in two dimensions; run per section with `dims=2, section_key=<column>`."

#: The worker's tables, concatenated with a leading ``section`` column in a per-section run.
TABLES = (
    ("importances_csv", "mistyr_importances.csv"),
    ("top_interactions", "mistyr_top_interactions.csv"),
    ("performance_csv", "mistyr_performance.csv"),
    ("contributions_csv", "mistyr_contributions.csv"),
)

#: Copies of worker_utils' names (the server imports worker_utils lazily, so a broken worker env can never
#: stop this module from registering its tool).
_PROVENANCE_FILENAME = "sog_run_provenance.json"


#: A coordinates CSV declares no units, so a bandwidth in micrometres cannot be honoured.
L_UM_REFUSED = "l_um needs the coordinates' units; a CSV carries none — pass `l` in the CSV's own units."


def _error(message: str) -> dict[str, Any]:
    return {"status": "error", "tool": TOOL_NAME, "task": "spatial_modeling", "error": message}


def section_dir_name(label) -> str:
    """``section_<label>`` with every character a path cannot safely carry replaced by ``_``."""
    return "section_" + re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))


def _read_table(path: str):
    import pandas as pd
    from worker_utils import sniff_tabular_sep

    return pd.read_csv(path, index_col=0, sep=sniff_tabular_sep(path))


def _coords_sections(coords_csv: str, section_key: str | None):
    """``(coords frame or None, section column or None)``, or raises ValueError with the refusal.

    Only a header naming its columns can name a section column, so a headerless positions file (which the
    R worker recognises on its own) is passed through untouched when no ``section_key`` is given.
    """
    from worker_utils import MAX_SECTION_LEVELS, STACK_COLUMN_NAMES

    try:
        frame = _read_table(coords_csv)
    except Exception as exc:  # unreadable here: the R worker has its own reader and its own messages
        if section_key:
            raise ValueError(f"MISTy: coords_csv {coords_csv} could not be read to split it by section: {exc}") from exc
        return None, None
    columns = [str(c) for c in frame.columns]
    if section_key:
        if section_key not in columns:
            raise ValueError(
                f"MISTy: section_key '{section_key}' is not a column of coords_csv {coords_csv}. Columns: {columns}. "
                "Name the column that holds each spot's section label."
            )
        return frame, section_key
    by_lower = {c.lower(): c for c in columns}
    candidates = [c for c in STACK_COLUMN_NAMES if c in columns]
    if "z" in by_lower:
        candidates.append(by_lower["z"])
    for column in candidates:
        n = int(frame[column].astype(str).nunique())
        if 2 <= n <= MAX_SECTION_LEVELS or (column.lower() == "z" and n >= 2):
            raise ValueError(
                f"MISTy: this file holds {n} sections in coords_csv column '{column}'; a 2D run would overlay "
                f"them. Choose per-section 2D with `dims=2, section_key='{column}'`; MISTy is two-dimensional, "
                "so a 3D run is not offered."
            )
    return frame, None


#: MISTy's cost, estimated before any worker runs (MISTY-1). The paraview weighs every pair of spots in a plane,
#: so a section costs about ``FIXED + LINEAR * n * f + QUADRATIC * n^2 * f`` seconds for n spots and f features.
#: Calibrated 2026-10-06 on this host (mistyR 1.18.0, single-threaded) on one Zhuang section subsampled to
#: 500/1,000/2,000/4,000/8,000 spots x 6 features: 5.4/7.7/14.3/37.0/118.1 s measured; the model gives
#: 5.5/7.8/14.5/37.0/118.0 s. A 30,532-spot section then costs about 25 min (the real case had not finished at
#: 1,236 s), and the 10-section object about 4 h.
MISTY_FIXED_S = 4.0
MISTY_LINEAR_S = 3.75e-4
MISTY_QUADRATIC_S = 2.5e-7
#: The budget a run's estimate must fit unless ``max_estimated_s`` or ``SOG_MISTY_MAX_SECONDS`` says otherwise.
DEFAULT_MAX_ESTIMATED_S = 1800.0


def misty_section_seconds(n_spots: int, n_features: int) -> float:
    """The estimated wall time of one MISTy run on ``n_spots`` spots and ``n_features`` features, in seconds."""
    n, f = float(max(0, n_spots)), float(max(1, n_features))
    return MISTY_FIXED_S + MISTY_LINEAR_S * n * f + MISTY_QUADRATIC_S * n * n * f


def _budget(max_estimated_s) -> tuple[float, str]:
    """``(seconds, source)``: the argument, else ``SOG_MISTY_MAX_SECONDS``, else the default. <= 0 means no budget."""
    if max_estimated_s is not None:
        return float(max_estimated_s), "max_estimated_s"
    raw = (os.environ.get("SOG_MISTY_MAX_SECONDS") or "").strip()
    if raw:
        try:
            return float(raw), "SOG_MISTY_MAX_SECONDS"
        except ValueError:
            pass
    return DEFAULT_MAX_ESTIMATED_S, "default"


def _duration(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _cost_plan(expression_csv: str, coords, section_key, max_estimated_s) -> dict[str, Any]:
    """The run's cost estimate, before any worker runs: spots per plane from coords, features from the expression
    table (oriented as the worker orients it), the per-plane and total estimates, and the budget they meet."""
    expr = _read_table(expression_csv)
    coord_ids = set(coords.index.astype(str))
    rows = expr.index.astype(str)
    spots_in_rows = bool(coord_ids & set(rows)) or not ({str(c) for c in expr.columns} & coord_ids)
    n_features = int(expr.shape[1] if spots_in_rows else expr.shape[0])
    if section_key:
        sizes = {str(k): int(v) for k, v in coords[section_key].astype(str).value_counts(sort=False).items()}
    else:
        sizes = {"": int(len(coords))}
    per = {label: misty_section_seconds(n, n_features) for label, n in sizes.items()}
    largest = max(sizes, key=lambda k: sizes[k])
    budget, source = _budget(max_estimated_s)
    return {
        "n_features": n_features,
        "n_planes": len(sizes),
        "largest_section": largest or None,
        "largest_section_spots": sizes[largest],
        "estimated_s": round(sum(per.values()), 1),
        "estimated_s_largest_section": round(per[largest], 1),
        "max_estimated_s": budget,
        "budget_source": source,
        "model": "seconds ~ 4 + 3.75e-4*spots*features + 2.5e-7*spots^2*features per plane (calibrated on one host)",
    }


def _over_budget(plan: dict[str, Any]) -> str | None:
    """The refusal when the estimate exceeds a positive budget, every number stated; else None."""
    budget = plan["max_estimated_s"]
    if budget <= 0 or plan["estimated_s"] <= budget:
        return None
    if plan["largest_section"] is not None:
        where = (
            f"{plan['n_planes']} sections; the largest, '{plan['largest_section']}', has "
            f"{plan['largest_section_spots']:,} spots x {plan['n_features']} features, about "
            f"{_duration(plan['estimated_s_largest_section'])} on its own"
        )
    else:
        where = f"one plane of {plan['largest_section_spots']:,} spots x {plan['n_features']} features"
    return (
        f"MISTy: this run is estimated at about {_duration(plan['estimated_s'])} ({where}). The paraview weighs "
        "every pair of spots in a section, so the cost grows with spots^2 x features; the estimate is calibrated "
        f"on one host and approximate. It is over the budget of {_duration(budget)} (max_estimated_s, or "
        "SOG_MISTY_MAX_SECONDS for the host), so nothing was run. To run it: restrict each section to a region "
        "(fewer spots per section), model fewer features or sections, or raise max_estimated_s."
    )


def _frame_record(section_key, sections) -> dict[str, Any]:
    """The same shape as ``worker_utils.Frame.to_dict``, for a coordinates CSV (which declares no units)."""
    return {
        "coords_key": "coords_csv",
        "dims": 2,
        "units_per_axis_um": [None, None],
        "z_source": None,
        "section_key": section_key,
        "sections": sections,
    }


def _write_provenance(output_dir: str, payload: dict[str, Any]) -> None:
    """``sog_run_provenance.json`` beside the outputs (the R worker writes none). Never raises."""
    if (os.environ.get("SOG_WRITE_PROVENANCE") or "").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        record = {
            "schema": "sog.run_provenance/1",
            "tool": TOOL_NAME,
            "task": "spatial_modeling",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": payload.get("params", {}),
            "input_data": payload.get("data", {}),
            "output_files": payload.get("output_files", {}),
            "executable": WORKER_RSCRIPT,
            "worker_script": WORKER_SCRIPT,
            "driven_by": os.path.abspath(__file__),
        }
        path = os.path.join(output_dir, _PROVENANCE_FILENAME)
        with open(path + ".partial", "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False, default=str)
            fh.write("\n")
        os.replace(path + ".partial", path)
    except Exception:
        pass


def _worker_args(expression_csv, coords_csv, output_dir, l_value, n_top, bypass_intra, seed) -> list[str]:
    args = [
        "--expression-csv",
        expression_csv,
        "--coords-csv",
        coords_csv,
        "--output-dir",
        output_dir,
        "--n-top",
        str(n_top),
        "--seed",
        str(seed),
    ]
    if l_value is not None:
        args += ["--l", str(l_value)]
    if bypass_intra:
        args.append("--bypass-intra")
    return args


def _run_per_section(
    expression_csv, coords, section_key, output_dir, l_value, n_top, bypass_intra, seed, shared
) -> dict[str, Any]:
    """One worker run per section, then the four tables concatenated with a leading ``section`` column."""
    import pandas as pd

    labels = [str(v) for v in pd.unique(coords[section_key].astype(str))]
    if len(labels) < 2:
        return _error(
            f"MISTy: coords_csv column '{section_key}' has {len(labels)} level(s), so there is nothing to run per "
            "section. Run once on the whole file with `dims=2` and no section_key."
        )
    dirs: dict[str, str] = {}
    for label in labels:
        name = section_dir_name(label)
        if name in dirs.values():
            return _error(f"MISTy: two sections would both be written to the folder {name}/; rename one of them.")
        dirs[label] = name

    expr = _read_table(expression_csv)
    coords = coords.copy()
    coords.index = coords.index.astype(str)
    expr.index = expr.index.astype(str)
    expr.columns = [str(c) for c in expr.columns]
    # Same orientation rule as the worker: spots in rows, unless no row name is a spot.
    spots_in_rows = len(expr.index.intersection(coords.index)) > 0 or not set(expr.columns) & set(coords.index)
    section_of = coords[section_key].astype(str)
    plane_coords = coords.drop(columns=[section_key])

    per: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="sog_mistyr_sections_") as scratch:
        for label in labels:
            ids = section_of.index[section_of.to_numpy() == label]
            c_path = os.path.join(scratch, f"{dirs[label]}_coords.csv")
            e_path = os.path.join(scratch, f"{dirs[label]}_expression.csv")
            plane_coords.loc[ids].to_csv(c_path)
            if spots_in_rows:
                expr.loc[expr.index.intersection(ids)].to_csv(e_path)
            else:
                expr[[c for c in expr.columns if c in set(ids)]].to_csv(e_path)
            folder = os.path.join(output_dir, dirs[label])
            os.makedirs(folder, exist_ok=True)
            res = run_worker_cli(
                TOOL_NAME,
                WORKER_RSCRIPT,
                WORKER_SCRIPT,
                _worker_args(e_path, c_path, folder, l_value, n_top, bypass_intra, seed),
            )
            if not isinstance(res, dict) or res.get("status") != "ok":
                error = res.get("error") if isinstance(res, dict) else res
                return _error(f"MISTy: section '{label}' ({section_key}) failed: {error}")
            per[label] = res

    output_files: dict[str, Any] = {}
    for key, name in TABLES:
        parts = []
        for label in labels:
            table = pd.read_csv(os.path.join(output_dir, dirs[label], name))
            table.insert(0, "section", label)
            parts.append(table)
        path = os.path.join(output_dir, name)
        pd.concat(parts, ignore_index=True).to_csv(path + ".partial", index=False)
        os.replace(path + ".partial", path)
        output_files[key] = path
    output_files["section_dirs"] = [os.path.join(output_dir, dirs[label]) for label in labels]

    warnings = [f"section '{label}': {w}" for label in labels for w in (per[label].get("warnings") or [])]
    gains = {label: (per[label].get("summary") or {}).get("mean_r2_gain") for label in labels}
    data = {
        "n_sections": len(labels),
        "n_spots": int(sum(int((per[label].get("data") or {}).get("n_spots", 0)) for label in labels)),
        "n_spots_per_section": {label: (per[label].get("data") or {}).get("n_spots") for label in labels},
        "n_features_per_section": {label: (per[label].get("data") or {}).get("n_features") for label in labels},
    }
    params = dict(shared)
    params.update(
        {
            "method": (per[labels[0]].get("params") or {}).get("method"),
            "mode": "per-section-2d",
            "sections": labels,
            "section_key": section_key,
            "frame": _frame_record(section_key, labels),
            "l_param_per_section": {label: (per[label].get("params") or {}).get("l_param") for label in labels},
            "per_section_params": {label: per[label].get("params") for label in labels},
        }
    )
    gain_text = "; ".join(
        f"{label}: {data['n_spots_per_section'][label]} spots, mean R2 gain "
        + (f"{gains[label]:.4f}" if isinstance(gains[label], (int, float)) else "not readable")
        for label in labels
    )
    payload: dict[str, Any] = {
        "status": "ok",
        "tool": TOOL_NAME,
        "task": "spatial_modeling",
        "data": data,
        "output_files": output_files,
        "params": params,
        "summary": {"mean_r2_gain_per_section": gains},
        "analysis": (
            f"MISTy spatial modeling run per section in 2D ({len(labels)} sections of {section_key}; each section "
            f"has its own views and models, and no paraview spans two sections): {gain_text}. The four tables at "
            "the top level carry a leading section column; each section's own run is in section_<label>/."
        ),
    }
    if warnings:
        payload["warnings"] = warnings
    return payload


@mcp.tool()
def mistyr_spatial_modeling(
    expression_csv: str,
    coords_csv: str,
    output_dir: str,
    l: float | None = None,
    n_top: int = 20,
    bypass_intra: bool = False,
    seed: int = 0,
    coords_key: str = "spatial",
    dims: int = 2,
    section_key: str | None = None,
    l_um: float | None = None,
    max_estimated_s: float | None = None,
) -> dict[str, Any]:
    """
    Run mistyR multi-view spatial modeling to identify intra- and inter-cellular
    interactions from spatial expression data.

    mistyR builds multiple "views" (intraview for cell-intrinsic, paraview for
    neighborhood-level) and uses machine learning to estimate feature importances
    and spatial interaction strengths.

    Two-dimensional by design: the paraview is built in two dimensions, so dims=3 is refused, and a
    coordinates CSV holding two or more sections (a section-like column such as `section`, or a `z`
    column) is refused unless section_key names the column. With section_key the run is per section:
    one MISTy run per section in output_dir/section_<label>/, and mistyr_importances.csv,
    mistyr_top_interactions.csv, mistyr_performance.csv and mistyr_contributions.csv at the top level
    with a leading `section` column. params.mode is "per-section-2d" (or "2d" for one plane),
    params.sections lists the sections and params.frame the coordinates read; the same is written to
    sog_run_provenance.json.

    Parameters
    ----------
    expression_csv:
        Path to expression matrix CSV (spots/cells as rows, features/genes as
        columns, with row names; genes x spots, as convert_h5ad_to_csv writes it, is
        transposed when no row name matches a spot). MISTy models every feature as a
        target and cannot model a constant one, so features that are constant across the
        spots analysed (all-zero genes are common in a whole-transcriptome export) are left
        out before any view is built and counted in data.n_features_dropped_zero_variance
        (of data.n_features_supplied; the first 20 names in
        params.zero_variance_features_dropped). A non-numeric column or a missing value is
        refused by name. MISTy fits a random forest per feature, so the run time grows with
        the number of features.
    coords_csv:
        Path to spatial coordinates CSV (spots as rows, spot names in the first column). Axis columns are
        matched by name -- imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col,
        row/col or x/y -- so Space Ranger's tissue_positions.csv can be passed as it is. The headerless
        tissue_positions_list.csv of Space Ranger before 2.0 is recognised from its first line and read
        with 10x's column names (params.coords_header says how the file was read; the columns used are
        params.coord_columns_used). When the file has an in_tissue column, spots with in_tissue == 0
        (background) are left out and counted in data.n_spots_off_tissue_dropped.
    output_dir:
        Directory for mistyR output files.
    l:
        Bandwidth of the paraview's Gaussian kernel, exp(-d^2 / l^2), in the units of the
        coordinates (on Visium, full-resolution pixels: Space Ranger's pxl_*_in_fullres, and the
        converter's x/y copied from obsm['spatial']). Unset
        (default) means 10x the median distance from a spot to its nearest other spot -- mistyR's own
        example uses l = 10 on a grid of unit spacing -- and the value chosen is reported as
        params.l_param. The old fixed default of 10 was 10 pixels: below one spot spacing on Visium,
        where every paraview weight was exactly 0. An explicit l so small that even the closest pair of
        spots gets weight 0 is refused; one below the spot spacing is run and warned about. In a
        per-section run an unset l is measured per section (params.l_param_per_section).
    n_top:
        Number of top interactions to report in the summary file (an integer >= 0; a
        negative value is refused).
    bypass_intra:
        If True, skip the intraview and only model inter-cellular (paraview)
        interactions.
    seed:
        Random seed passed to mistyR::run_misty, which draws the cross-validation folds and seeds
        ranger with it. 0 (the default) runs as mistyR's own default seed 42, because ranger reads a
        seed of 0 as "no seed" and would give different forests on every run; the seed used is
        reported as params.seed_used beside the one asked for (params.seed). It must fit
        R's 32-bit integer range; a larger value is refused, not turned into NA.
    coords_key, dims, section_key:
        coords_key: the obsm key holding the coordinates (default 'spatial'; an aligned 3D frame such as
        'spatial_3d_aligned'). dims: 2 or 3; 3 builds the graph in the aligned frame in micrometres and
        needs a frame with recorded units and a measured or registered z. section_key: the obs column
        naming sections; required for a 2D run on a multi-section file (the run is per section) and for
        the cross-section edge count of a 3D run.
        For MISTy: the coordinates come from coords_csv, so coords_key stays 'spatial' (any other key is
        refused -- write that frame's two columns into coords_csv instead); dims=3 is refused; and
        section_key names a column of coords_csv.
    l_um:
        Refused. A bandwidth in micrometres needs the coordinates' units, and coords_csv declares none, so
        any l_um is refused: "l_um needs the coordinates' units; a CSV carries none — pass `l` in the CSV's
        own units." Give the bandwidth as l.
    max_estimated_s:
        The wall-time budget, in seconds, the run's estimate must fit. MISTy's paraview weighs every pair of
        spots in a section, so a run costs about spots^2 x features per section; the cost is estimated before
        any worker runs and reported as params.cost_estimate, and a run estimated over the budget is refused
        with every number (for example, one 30,532-spot section x 6 features is about 25 min). Unset: the
        host's SOG_MISTY_MAX_SECONDS, else 1800 s. 0 or less means no budget.
    """
    if int(dims) == 3:
        return _error(TWO_D_ONLY)
    if int(dims) != 2:
        return _error(f"MISTy: dims must be 2 (or 3, which is refused), not {dims}.")
    if coords_key != "spatial":
        return _error(
            f"MISTy reads its coordinates from coords_csv, not from an obsm key, so coords_key={coords_key!r} "
            "cannot be used. Write that frame's two columns into coords_csv (per section, with section_key) and "
            "leave coords_key at 'spatial'."
        )
    if l_um is not None:
        return _error(L_UM_REFUSED)
    l_value = l
    section_key = section_key or None

    os.makedirs(output_dir, exist_ok=True)
    try:
        coords, column = _coords_sections(coords_csv, section_key)
    except ValueError as exc:
        return _error(str(exc))

    shared = {
        "coords_key": "coords_csv",
        "dims": 2,
        "l_units": "coords_csv's own units (a CSV declares none)",
    }
    if coords is not None:
        try:
            plan = _cost_plan(expression_csv, coords, column, max_estimated_s)
        except Exception as exc:  # the worker reads the table itself and refuses what it cannot read, by name
            print(f"[mistyr] cost estimate skipped: {exc}", file=sys.stderr, flush=True)
            plan = None
        if plan is not None:
            refusal = _over_budget(plan)
            if refusal:
                return _error(refusal)
            shared["cost_estimate"] = plan
    if column:
        try:
            payload = _run_per_section(
                expression_csv, coords, column, output_dir, l_value, n_top, bypass_intra, seed, shared
            )
        except Exception as exc:
            print(f"[mistyr] per-section run failed: {exc}", file=sys.stderr, flush=True)
            return _error(f"MISTy: the per-section run failed: {exc}")
    else:
        payload = run_worker_cli(
            TOOL_NAME,
            WORKER_RSCRIPT,
            WORKER_SCRIPT,
            _worker_args(expression_csv, coords_csv, output_dir, l_value, n_top, bypass_intra, seed),
        )
        if isinstance(payload, dict) and payload.get("status") == "ok":
            params = payload.setdefault("params", {})
            params.update(shared)
            params.update({"mode": "2d", "sections": None, "section_key": None, "frame": _frame_record(None, None)})
    if isinstance(payload, dict) and payload.get("status") == "ok":
        _write_provenance(output_dir, payload)
    return payload


if __name__ == "__main__":
    mcp.run()
