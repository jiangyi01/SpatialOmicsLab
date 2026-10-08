"""One function per portal call, tying the diagnosis modules together.

Each returns a JSON-shaped payload: a status, the numbers, the sentences that carry them, and --
where something could not be established -- the question whose answer would establish it. Nothing
here raises for an ordinary bad outcome. A stack with no section order, a pair with no overlap, a
tool whose output cannot be located: each is a normal result with a reason attached, because a
refusal the model can read is an answer and a traceback is not.

**Phase 1 writes nothing.** ``diagnose_3d_stack`` opens the file, measures it, and writes only its
own report and tables into the output directory. ``test/test_spatial3d_the_original_coordinates_survive.py``
holds that promise by digesting obsm, uns and obs either side of a full run.
"""

from __future__ import annotations

import json
import os
from typing import Any

from . import classify as cls
from . import contract, profile
from . import thresholds as T

SCHEMA = "sog.spatial3d_report/1"

#: How many adjacent pairs a report prints in full before it summarises. A 147-section atlas has
#: 146 of them and a payload that printed every one would be unreadable; the table on disk always
#: holds all of them.
MAX_PAIRS_IN_PAYLOAD = 12

#: Expression values read per row block when the library sizes are summed: 2**25 values, 256 MB as
#: float64. The whole matrix was densified to float64 and copied twice more by nansum; on the
#: 4.17M x 1,122 atlas the thresholds were calibrated on that peaked near 100 GB
#: (hunt 2026-09-30, u21-3d-12).
ROW_BLOCK_VALUES = 1 << 25


def _err(tool: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "tool": tool, "error": message, **extra}


def _read_for_diagnosis(data_path: str):
    """The object, with a dense X left on disk and read a gene column or a row block at a time.

    A sparse X is brought into memory as it is stored -- its nonzeros, never densified -- because
    anndata's backed sparse matrix cannot be indexed by column, which is how the genes are read.
    Close the returned object's file with :func:`_close` (hunt 2026-09-30, u21-3d-12).
    """
    import anndata as ad
    import h5py

    adata = ad.read_h5ad(data_path, backed="r")
    handed_on = False
    try:
        if isinstance(adata.X, h5py.Dataset):
            handed_on = True  # the caller reads X from disk and closes it with _close
            return adata
        return adata.to_memory()
    finally:  # a failed to_memory still releases the HDF5 lock on the input
        if not handed_on and getattr(adata, "isbacked", False):
            adata.file.close()


def _close(adata: Any) -> None:
    try:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    except Exception:
        pass


def _read_annotations(data_path: str):
    """``obs``, ``obsm`` and ``uns`` only. X, layers and raw stay on disk.

    The inspector lists keys; it decompressed the whole expression matrix to do it
    (hunt 2026-09-30, u21-3d-12).
    """
    import anndata as ad
    import h5py

    try:
        from anndata.io import read_elem
    except ImportError:  # anndata < 0.11
        from anndata.experimental import read_elem

    try:
        with h5py.File(data_path, "r") as f:
            obs = read_elem(f["obs"])
            obsm = {str(k): read_elem(f["obsm"][k]) for k in f["obsm"]} if "obsm" in f else {}
            uns = read_elem(f["uns"]) if "uns" in f else {}
        return ad.AnnData(obs=obs, obsm=obsm, uns=uns)
    except Exception:
        # An encoding this reader does not know: the same three tables through anndata's own
        # backed read, which also leaves X on disk.
        adata = ad.read_h5ad(data_path, backed="r")
        try:
            return ad.AnnData(obs=adata.obs.copy(), obsm=dict(adata.obsm), uns=dict(adata.uns))
        finally:
            if getattr(adata, "isbacked", False):
                adata.file.close()


def _in_tissue_rows(obs: Any) -> tuple[Any, int, str]:
    """``(mask or None, n_left_out, refusal)`` for ``obs['in_tissue']``.

    The same rule as ``tools/worker_utils.keep_in_tissue``: CELLxGENE Visium exports carry every
    array spot with in_tissue 0/1 and 56-70% of them are glass, so the outline, the centroid and the
    containment described the capture area rather than the tissue (hunt 2026-09-30, u21-3d-3).
    True/"1"/1 count as in tissue; no column, or one that is 1 everywhere, changes nothing.
    """
    import numpy as np
    import pandas as pd

    if "in_tissue" not in obs.columns:
        return None, 0, ""
    raw = obs["in_tissue"]
    flag = pd.to_numeric(raw.astype(str).str.strip().str.lower().replace({"true": "1", "false": "0"}), errors="coerce")
    keep = np.asarray(flag == 1)
    if keep.all():
        return None, 0, ""
    if not keep.any():
        seen = sorted({str(v) for v in raw.unique()})[:8]
        return (
            None,
            0,
            f"obs['in_tissue'] marks none of the {len(keep)} cells as in tissue (values seen: {seen}); "
            "fix the column so in-tissue spots are 1, or remove it if every spot is tissue",
        )
    return keep, int((~keep).sum()), ""


def _library_sizes(X: Any, rows: Any = None) -> tuple[Any, Any, int, int]:
    """Per-cell totals and detected-gene counts, nan-safe, read one row block at a time.

    Returns ``(totals, detected, n_nonfinite, n_values)`` over the cells ``rows`` keeps (every cell
    when None). Never a whole-matrix dense copy.
    """
    import numpy as np

    n, n_vars = int(X.shape[0]), int(X.shape[1])
    totals: list[Any] = []
    detected: list[Any] = []
    n_nonfinite = n_values = 0
    step = max(1, ROW_BLOCK_VALUES // max(n_vars, 1))
    for start in range(0, n, step):
        stop = min(start + step, n)
        block = X[start:stop]
        matrix = np.asarray(block.toarray() if hasattr(block, "toarray") else block, dtype=float)
        if rows is not None:
            matrix = matrix[rows[start:stop]]
        n_nonfinite += int((~np.isfinite(matrix)).sum())
        n_values += int(matrix.size)
        totals.append(np.nansum(matrix, axis=1).ravel())
        detected.append(np.nansum(matrix > 0, axis=1).ravel())
    if not totals:
        return np.zeros(0), np.zeros(0), 0, 0
    return np.concatenate(totals), np.concatenate(detected), n_nonfinite, n_values


class _RowsOf:
    """A gene source restricted to the rows being diagnosed."""

    def __init__(self, src: Any, rows: Any):
        self._src, self._rows = src, rows
        self.genes = src.genes
        self.source = src.source

    def column(self, gene: str) -> Any:
        return self._src.column(gene)[self._rows]


def _resolve_sections(adata: Any, slice_key: str) -> tuple[str, Any] | tuple[str, None]:
    """The obs column naming each section, or a refusal listing what was tried.

    Never guessed from an ordering of rows: a merged object's row order is the order the files
    happened to be concatenated in, which is not a fact about the tissue.
    """
    import numpy as np

    candidates = (
        [slice_key]
        if slice_key
        else ["slice_id", "section", "section_id", "library_id", "brain_section_label", "Bregma", "batch"]
    )
    for name in candidates:
        if name and name in adata.obs.columns:
            values = adata.obs[name].astype(str).to_numpy()
            if len(np.unique(values)) >= 2:
                return name, values
    return "", None


def diagnose_3d_stack(
    data_path: str,
    output_dir: str,
    *,
    slice_key: str = "",
    z_key: str = "",
    slice_order: str = "",
    coords_key: str = "spatial",
    n_genes: int = 8,
    skip_expression: bool = False,
) -> dict[str, Any]:
    """Phase 1: profile the sections, measure every adjacent pair, and classify the stack.

    Reads the data and writes only its report. The verdict is A, B, C or unknown, and it arrives
    with the value and threshold behind every criterion it was decided on.
    """
    tool = "diagnose_3d_stack"
    if not os.path.exists(data_path):
        return _err(tool, f"no such file: {data_path}")
    os.makedirs(output_dir, exist_ok=True)

    adata = _read_for_diagnosis(data_path)
    try:
        return _diagnose(
            tool,
            adata,
            data_path,
            output_dir,
            slice_key=slice_key,
            z_key=z_key,
            slice_order=slice_order,
            coords_key=coords_key,
            n_genes=n_genes,
            skip_expression=skip_expression,
        )
    finally:
        _close(adata)


def _diagnose(
    tool: str,
    adata: Any,
    data_path: str,
    output_dir: str,
    *,
    slice_key: str,
    z_key: str,
    slice_order: str,
    coords_key: str,
    n_genes: int,
    skip_expression: bool,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from . import batch as batch_mod
    from . import biology, geometry

    if coords_key not in adata.obsm:
        return _err(
            tool,
            f"obsm[{coords_key!r}] is absent, so there are no coordinates to diagnose",
            obsm_keys=sorted(map(str, adata.obsm)),
        )
    xy = np.asarray(adata.obsm[coords_key], dtype=float)[:, :2]

    key, sections = _resolve_sections(adata, slice_key)
    if sections is None:
        return _err(
            tool,
            "no obs column names the section each cell belongs to, so the object cannot be read "
            "as a stack of serial sections. Name the column with slice_key.",
            obs_columns=sorted(map(str, adata.obs.columns))[:40],
        )

    # An explicit order is stripped and checked against the sections. 's1, s2, s3' kept ' s2' and
    # ' s3', every pair came out empty and the verdict asked whether the data were complete; an
    # order that left a section out dropped it silently (hunt 2026-09-30, u21-3d-8).
    order = [s.strip() for s in slice_order.split(",") if s.strip()] if slice_order else []
    if order:
        present = sorted(set(map(str, sections)))
        unknown = [s for s in order if s not in present]
        missing = [s for s in present if s not in order]
        repeated = sorted({s for s in order if order.count(s) > 1})
        if unknown or missing or repeated:
            problems = []
            if unknown:
                problems.append(f"not a section of obs[{key!r}]: {unknown}")
            if missing:
                problems.append(f"sections the order leaves out: {missing}")
            if repeated:
                problems.append(f"named more than once: {repeated}")
            return _err(
                tool,
                f"slice_order does not match the sections in obs[{key!r}] -- {'; '.join(problems)}. Give "
                f"every section exactly once, in physical order.",
                sections=present[:60],
            )

    z = None
    z_notes: list[str] = []
    zk = z_key or next((c for c in ("slice_z", "z", "Bregma") if c in adata.obs.columns), "")
    if zk:
        # An explicit z_key that is absent or not numeric was ignored without a word, and the
        # verdict then asked for the z column the caller had just named (hunt 2026-09-30, u21-3d-8).
        if zk not in adata.obs.columns:
            return _err(
                tool,
                f"z_key {zk!r} is not a column of obs, so no section z can be read from it",
                obs_columns=sorted(map(str, adata.obs.columns))[:40],
            )
        try:
            z = adata.obs[zk].to_numpy(dtype=float)
        except Exception:
            seen = sorted({str(v) for v in adata.obs[zk].unique()})[:5]
            if z_key:
                return _err(
                    tool,
                    f"obs[{zk!r}] cannot be read as numbers (values such as {seen}), so it is not a z",
                    obs_columns=sorted(map(str, adata.obs.columns))[:40],
                )
            z_notes.append(
                f"obs[{zk!r}] looks like a z column but cannot be read as numbers (values such as {seen}), "
                f"so no z was taken from it"
            )
            zk = ""

    tissue, n_off, refusal = _in_tissue_rows(adata.obs)
    if refusal:
        return _err(tool, refusal)
    if tissue is not None:
        xy, sections = xy[tissue], sections[tissue]
        z = z[tissue] if z is not None else None

    prof = profile.profile_stack(
        xy,
        sections,
        z=z,
        slice_order=order or None,
        slice_key=key,
        z_key=zk,
        obsm_keys={str(k): list(np.asarray(v).shape) for k, v in adata.obsm.items()},
    )
    prof.notes.extend(z_notes)
    if n_off:
        prof.notes.append(
            f"{n_off} of {n_off + len(xy)} cells have obs['in_tissue'] == 0 (background outside the tissue) "
            f"and were left out; {len(xy)} in-tissue cells were diagnosed."
        )

    pairs = prof.adjacent_pairs
    # pair_geometry leaves out and counts any non-finite coordinate itself.
    geoms = [geometry.pair_geometry(a, xy[sections == a], b, xy[sections == b]) for a, b in pairs]
    placed = np.isfinite(xy).all(axis=1)

    bios: list[Any] = [None] * len(geoms)
    selection = None
    batch_report = None
    if not skip_expression and pairs:
        try:
            src = biology.AnnDataGenes(adata)
            if tissue is not None:
                src = _RowsOf(src, tissue)
            # Nan-safe, and the count is reported rather than absorbed. A single all-NaN gene
            # column -- the Moffitt hypothalamus atlas has exactly one, Fos -- makes a plain
            # sum(axis=1) NaN for EVERY cell, which cascades into every per-section statistic and
            # ends with the report claiming that every gene carries a batch effect.
            totals, detected, n_nonfinite, n_values = _library_sizes(adata.X, tissue)
            if n_nonfinite:
                prof.notes.append(
                    f"{n_nonfinite:,} of {n_values:,} expression values are not finite "
                    f"({n_nonfinite / n_values:.3%}). Library sizes are computed ignoring them, "
                    f"and any gene with a non-finite value is excluded from the consistency metric "
                    f"and listed separately -- it is unreadable, not batch-affected."
                )
            selection = biology.select_genes(src, sections, totals, n_genes=n_genes)
            if selection.chosen:
                smoothed: dict[str, dict[str, Any]] = {}
                raw = {g: src.column(g) for g in selection.chosen}
                norm = {g: biology._cpm_log1p_within(raw[g], sections, totals) for g in selection.chosen}
                for s in prof.slice_order:
                    m = (sections == s) & placed
                    smoothed[s] = {g: biology.smooth_within_section(norm[g][m], xy[m]) for g in selection.chosen}
                bios = [
                    biology.pair_biology(
                        a, xy[(sections == a) & placed], smoothed[a], b, xy[(sections == b) & placed], smoothed[b]
                    )
                    for a, b in pairs
                ]
                profiles = {
                    s: np.array([norm[g][sections == s].mean() for g in selection.chosen]) for s in prof.slice_order
                }
                batch_report = batch_mod.batch_report(
                    sections,
                    totals,
                    detected,
                    prof.slice_order,
                    profiles=profiles,
                    rejected_batchy=selection.rejected_batchy,
                )
        except Exception as exc:  # a diagnosis still has its geometry without its biology
            bios = [None] * len(geoms)
            prof.notes.append(f"the expression metrics could not be computed: {type(exc).__name__}: {exc}")

    verdict = cls.classify_stack(
        geoms,
        bios,
        slice_order_known=prof.slice_order_known,
        units_known=prof.xy_units != "unknown",
        coincident_z_groups=prof.coincident_z_groups,
    )

    rows = [g.as_row() for g in geoms]
    for row, b in zip(rows, bios, strict=True):
        if b is not None:
            row.update(
                bio_consistency=b.bio_consistency,
                bio_null=b.bio_null,
                bio_gap=b.bio_gap,
                bio_matched_fraction=b.matched_fraction,
            )
    table = os.path.join(output_dir, "adjacent_pair_metrics.csv")
    if rows:
        pd.DataFrame(rows).to_csv(table, index=False)

    report = {
        "schema": SCHEMA,
        "tool": tool,
        "data_path": os.path.abspath(data_path),
        "class": verdict.label,
        "class_description": cls.DESCRIPTIONS[verdict.label],
        "summary": verdict.summary(),
        "n_sections": len(prof.slices),
        "n_adjacent_pairs": len(pairs),
        "slice_key": key,
        "z_key": zk,
        "slice_order": prof.slice_order,
        "slice_order_known": prof.slice_order_known,
        "slice_order_from": prof.slice_order_from,
        "xy_units": prof.xy_units,
        "units_evidence": prof.units_from,
        "z_spacings": prof.z_spacings,
        "z_spacing_uniform": prof.z_spacing_uniform,
        "coincident_z_groups": prof.coincident_z_groups,
        "coordinate_coverage": prof.coordinate_coverage,
        "in_tissue_filter": (
            {"n_cells_supplied": n_off + len(xy), "n_cells_off_tissue_dropped": n_off, "n_cells_used": len(xy)}
            if n_off
            else None
        ),
        "counts": verdict.counts,
        "pair_verdicts": [v.sentence() for v in verdict.pairs[:MAX_PAIRS_IN_PAYLOAD]],
        "reasons": verdict.reasons,
        "questions": prof.questions + ([verdict.question] if verdict.question else []),
        "notes": prof.notes,
        "thresholds": {t.name: {"value": t.value, "units": t.units, "from": t.calibrated_on} for t in T.ALL},
        "calibration_table": T.CALIBRATION_TABLE,
    }
    if selection is not None:
        report["genes_used"] = selection.chosen
        report["genes_rejected_as_batchy"] = selection.rejected_batchy[:20]
        report["genes_rejected_as_unreadable"] = selection.rejected_nonfinite[:20]
        report["genes_scanned"] = selection.scanned
    if batch_report is not None:
        report["batch"] = {
            "summary": batch_report.summary(),
            "depth_fold_range": batch_report.depth_fold_range,
            "median_total_counts": batch_report.median_total_counts,
            "notes": batch_report.notes,
            "not_computed": batch_report.not_computed,
        }

    report_path = os.path.join(output_dir, "alignment_diagnosis.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    return {
        "status": "ok",
        "tool": tool,
        "data": report,
        "output_files": {"report": report_path, **({"pair_metrics": table} if rows else {})},
        "summary": verdict.summary(),
        "next_step": _next_step(verdict.label),
    }


def _next_step(label: str) -> str:
    # assemble_3d_stack exists nowhere, moscot_run's keyword is problem_type rather than problem,
    # and class C named only a point-CSV tool (hunt 2026-09-30, u21-3d-9).
    if label == "A":
        return (
            "The sections already share a frame. Write the stack as a raw frame with "
            "spatialomicsgym.spatial3d.contract.write_frame (Frame key 'spatial_3d_raw', role 'raw', "
            "with the declared z) and go straight to Phase 3; running an aligner here would move "
            "coordinates that are already right."
        )
    if label == "B":
        return (
            "A rigid misalignment. paste_pairwise_align for full-overlap serial sections, or "
            "moscot_run(problem_type='alignment', batch_key=<the section column>) for a large or "
            "complex stack. Present this classification and its evidence to the user before "
            "moving any coordinate."
        )
    if label == "C":
        return (
            "A non-rigid deformation. A similarity transform cannot undo it: cast_align_slices "
            "fits a free-form deformation to single-cell sections, paste2_partial_align handles "
            "sections that only partly overlap, and stalign_align_points fits a diffeomorphism to "
            "point CSVs. Present this classification and its evidence to the user before moving "
            "any coordinate."
        )
    return (
        "The class was not determined. The report's `questions` list says exactly what would "
        "determine it; put those to the user rather than picking a class."
    )


def inspect_3d_coordinates(data_path: str, *, coords_key: str = "spatial") -> dict[str, Any]:
    """Read-only: which coordinate keys this object has, what shape, and what the contract says."""
    tool = "inspect_3d_coordinates"
    import numpy as np

    if not os.path.exists(data_path):
        return _err(tool, f"no such file: {data_path}")
    adata = _read_annotations(data_path)
    frames = contract.find_frames(adata)
    problems = contract.validate(adata)
    obsm_keys = {str(k): list(np.asarray(v).shape) for k, v in adata.obsm.items()}
    # coords_key was accepted, documented and never read (hunt 2026-09-30, u21-3d-22).
    coords = {"key": coords_key, "present": coords_key in obsm_keys, "shape": obsm_keys.get(coords_key)}
    summary = (
        f"{len(frames)} three-column frame(s); {len(problems)} contract problem(s)"
        if frames
        else f"no 3D frame; {len(problems)} contract problem(s)"
    )
    if coords_key and not coords["present"]:
        summary += f"; obsm[{coords_key!r}] is absent (present: {sorted(obsm_keys) or 'none'})"
    return {
        "status": "ok",
        "tool": tool,
        "data": {
            "obsm_keys": obsm_keys,
            "coords_key": coords,
            "three_column_keys": contract.three_column_keys(adata),
            "frames": {
                k: {"role": f.role, "z_source": f.z_source, "z_units": f.z_units, "aligner": f.aligner}
                for k, f in frames.items()
            },
            "has_provenance_block": "spatial_3d" in adata.uns,
            "contract_problems": problems,
            "obs_columns": sorted(map(str, adata.obs.columns))[:60],
        },
        "summary": summary,
    }


def explain_3d_contract() -> dict[str, Any]:
    """Read-only: the keys, the roles and the rules, so a caller need not guess them."""
    return {
        "status": "ok",
        "tool": "explain_3d_contract",
        "data": {
            "schema": contract.SCHEMA,
            "original_key": contract.ORIGINAL_KEY,
            "original_rule": (
                "obsm['spatial'] stays two columns and is never written by this package. PASTE "
                "asserts X.shape[1] == 2; ST-GEARS indexes obsm['spatial'][:, 2], so its worker "
                "hands it a working copy whose z is the slice ordinal and writes the two-column "
                "original back. Both take the same two-column file."
            ),
            "frames": {
                contract.RAW_FRAME: "the stack before alignment",
                contract.ALIGNED_FRAME: "the stack after alignment",
                f"{contract.EXTERNAL_PREFIX}<source>": "somebody else's registration, e.g. spatial_3d_ccf",
            },
            "roles": list(contract.ROLES),
            "units": list(contract.UNITS),
            "z_sources": list(contract.Z_SOURCES),
            "z_spacing_rule": "there is no default anywhere in this package; an unknown spacing is asked about",
        },
        "summary": "the 3D coordinate contract: keys, roles, units and the rules that hold them together",
    }


def list_aligner_adapters() -> dict[str, Any]:
    """Read-only: where each shipped aligner leaves its answer, or why it cannot be read."""
    from . import adapters

    rows = adapters.ADAPTERS
    rows = list(rows.values()) if isinstance(rows, dict) else list(rows)
    return {
        "status": "ok",
        "tool": "list_aligner_adapters",
        "data": {
            "adapters": [
                {
                    "function": r.function,
                    "server": r.server,
                    "status": r.status,
                    "writes_key": r.writes_key,
                    "original_recoverable": r.original_recoverable,
                    "seed_exposed": r.seed_exposed,
                    "reason": getattr(r, "refusal_reason", "") or "",
                }
                for r in rows
            ],
            "aligned_obsm_keys": sorted(adapters.ALIGNED_OBSM_KEYS),
        },
        "summary": f"{len(rows)} alignment functions, "
        f"{sum(1 for r in rows if r.status in ('adapted', 'derived'))} readable",
    }
