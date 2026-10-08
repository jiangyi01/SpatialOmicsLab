"""One function per tool call: inspect, check, read, draw, record, declare.

Every producing pipeline here follows the same six steps in the same order, and the order is the
point. The prerequisite check happens **before** anything is read, so a request that cannot be
answered costs one backed open rather than a half-drawn figure. The manifest is written **last**,
because a figure that is declared but absent is worse than one that is present and undeclared.

Each returns a plain dictionary the worker turns into its payload: what was drawn, what was
refused and why, and the warnings a reader needs. Nothing here raises for an expected condition
-- a missing column, an absent image, a differential-expression result without p-values are all
outcomes, not accidents, and each comes back as a refusal that names the next call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import capabilities as _caps
from . import layers as _layers
from . import manifest_io as _manifest_io
from . import palette as _palette
from . import profile as _profile
from . import spec as _spec
from .render import base as _base
from .render import points as _points
from .render import spatial as _spatial


def _refusal(
    kind: str,
    why: str,
    *,
    plot_id: str = "",
    missing: list[str] | None = None,
    instead: list[str] | None = None,
    fix: str = "",
) -> dict[str, Any]:
    """The one refusal shape. A refusal that does not say what to do next is a dead end."""
    return {
        "status": "refused",
        "error_kind": kind,
        "plot_id": plot_id,
        "why": why,
        "missing": missing or [],
        "instead": instead or [],
        "fix": fix,
    }


#: The colour semantic for a signed quantity centred on zero. The producers asked palette for
#: "signed", which it does not know, so continuous() returned viridis and limits() did not centre
#: on zero: depletion and enrichment on one sequential ramp whose midpoint was not zero, which is the
#: defect palette's own docstring names (hunt 2026-09-30, u20a-viz-pipelines-16).
_SIGNED = "zscore"


def _layer_errors_refuse(plot_id: str):
    """Turn an escaped ``LayerError`` into the refusal shape this module promises.

    A ``LayerError`` is by construction an expected condition with a sentence saying what to do --
    an absent obs column, an obsm key that is not there, an object too large to open. Several reads
    sat outside any ``try`` (``open_source`` itself, ``obs_values`` on a named groupby, ``obsm_values``
    on a basis), so the condition escaped as an exception and the worker reported a bare error with
    the fix dropped (hunt 2026-09-30, u20a-viz-pipelines-31).
    """
    import functools

    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> dict[str, Any]:
            try:
                return fn(*args, **kwargs)
            except _layers.LayerError as exc:
                return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)

        return inner

    return wrap


def _with_selectors(fingerprint: str, **selectors: Any) -> str:
    """The stem's source identity, extended by selectors ``spec.DATA_PARAMS`` does not list.

    ``spec.stem_for`` hashes only the keys in ``DATA_PARAMS``, so a selector outside that set --
    ``pseudotime_key``, ``split_by``, ``standardize``, ``pathways``, a result table's content -- never
    reached the filename, and two different figures overwrote each other with both calls returning
    ok (hunt 2026-09-30, u20a-viz-pipelines-4, -5). An empty selector leaves the identity unchanged,
    so a request that names none of them keeps the stem it always had.
    """
    import json

    chosen = {k: v for k, v in sorted(selectors.items()) if v not in (None, "", [], False)}
    if not chosen:
        return fingerprint
    return f"{fingerprint}|{json.dumps(chosen, sort_keys=True, default=str)}"


def _file_fingerprint(path: str) -> str:
    """Sixteen hex characters of the file's content, or "" when it cannot be read.

    A result table carries no dataset fingerprint, and its PATH is not a selector the stem hashes,
    so every table of one family drew to one filename (hunt 2026-09-30, u20a-viz-pipelines-4). The
    content, not the name: the same table under two names is the same figure.
    """
    import hashlib

    if not path:
        return ""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return ""
    return digest.hexdigest()[:16]


def _coords_and_mask(adata: Any, library_id: str = "") -> tuple[Any, str, Any]:
    """``(coordinates of every observation, where they came from, section mask or None)``.

    ``layers.spatial_coords(adata, library_id)`` returns the coordinates of the named section only,
    while every producer kept painting values for all observations, so each 2D map crashed on a
    length mismatch the moment ``library_id`` was used (hunt 2026-09-30, u20a-viz-pipelines-1). The
    mask is ``layers.library_mask`` itself -- the rule ``spatial_coords`` applies -- so the
    coordinates and the values cannot disagree about which observations are drawn. A re-derived
    mask read only obs section columns, so naming the one ``uns['spatial']`` library of a plain
    Visium object, which selects every spot, was refused as "selects 0 observations".
    """
    import numpy as np

    coords, where = _layers.spatial_coords(adata)
    coords = np.asarray(coords, dtype=float)
    if not library_id:
        return coords, where, None
    mask, source = _layers.library_mask(adata, library_id)
    if mask is None:
        return coords, f"{where} ({source})", None
    return coords, f"{where} restricted to {source} == {library_id!r}", np.asarray(mask, dtype=bool)


def _slot_integral(slot: dict[str, Any], prof: dict[str, Any], arrays: Any) -> bool | None:
    """Whether the matrix the values were READ from holds integer counts.

    The profile describes ``X``. A counts-shaped request resolves to ``layers['counts']`` or
    ``.raw`` on a processed object, and handing ``normalize_for_display`` the integrality of a
    log-normalised ``X`` drew raw counts on a linear scale captioned "already transformed" (hunt
    2026-09-30, u20a-viz-pipelines-19). Any other slot is judged from the values actually drawn.
    """
    import numpy as np

    if slot.get("slot") == "X":
        return prof["matrix"]["integral"]
    parts = [np.asarray(a, dtype=float).ravel() for a in arrays]
    if not parts:
        return None
    flat = np.concatenate(parts)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return None
    return bool((flat >= 0).all() and (np.mod(flat, 1.0) == 0).all())


def _natural_key(text: Any) -> list[Any]:
    """Sort ``S2`` before ``S10``: string order put S10 between S1 and S2 (hunt 2026-09-30, -17, -29)."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"(\d+)", str(text))]


def _align_rows_to_obs(
    frame: Any, obs_names: Any, what: str, plot_id: str, warnings: list[str]
) -> tuple[Any, Any, Any]:
    """``(frame reindexed to every observation, which observations it covers, refusal)``.

    A table another tool wrote was painted by row POSITION: its barcode index was read and then
    ignored, so a table in another row order drew every value on another spot with status ok, and a
    tool that dropped spots crashed on the length (hunt 2026-09-30, u20a-viz-pipelines-12, and the
    same defect in the activity map). Rows are matched by name, a transposed table is recognised,
    and too little overlap is refused rather than drawn.
    """
    import numpy as np

    names = np.asarray(obs_names).astype(str)
    wanted = set(names.tolist())

    def overlap(index: Any) -> int:
        return sum(1 for v in np.asarray(index).astype(str).tolist() if v in wanted)

    if overlap(frame.index) == 0 and overlap(frame.columns) > 0:
        frame = frame.T
    rows = np.asarray(frame.index).astype(str)
    matched = overlap(rows)
    if matched < 0.5 * min(len(rows), len(names)):
        return (
            None,
            None,
            _refusal(
                "invalid_request",
                f"the {what} names {matched} of the object's {len(names)} observations among its "
                f"{len(rows)} rows, so its rows are not this object's spots",
                plot_id=plot_id,
                fix="pass the table written for this object; its first column must hold the observation names",
            ),
        )
    frame = frame.copy()
    frame.index = rows
    duplicated = int(frame.index.duplicated(keep="first").sum())
    frame = frame[~frame.index.duplicated(keep="first")]
    present = np.isin(names, frame.index.to_numpy())
    if duplicated:
        warnings.append(f"{duplicated} row name(s) appear more than once in the {what}; the first of each is used")
    if len(rows) - duplicated > int(present.sum()):
        warnings.append(
            f"{len(rows) - duplicated - int(present.sum())} row(s) of the {what} name no observation in this object and are ignored"
        )
    if not present.all():
        warnings.append(
            f"{int((~present).sum())} of {len(names)} observations have no row in the {what} and are not drawn"
        )
    return frame.reindex(names), present, None


def inspect_dataset(
    data_path: str,
    *,
    obs_keys: str = "",
    var_names: str = "",
    max_categories: int = 200,
    sample_spots: int = 5000,
    include_capabilities: bool = True,
) -> dict[str, Any]:
    """Everything a plot needs to know about a dataset. Writes nothing."""
    prof = _profile.profile_for_viz(
        data_path,
        obs_keys=[s.strip() for s in obs_keys.split(",") if s.strip()],
        var_names=[s.strip() for s in var_names.split(",") if s.strip()],
        max_categories=int(max_categories),
        sample_spots=int(sample_spots),
    )
    out: dict[str, Any] = {"status": "ok" if prof["dataset"]["readable"] else "error", "profile": prof}
    if not prof["dataset"]["readable"]:
        out["error"] = f"could not open {Path(data_path).name}: {prof['dataset']['read_error']}"
        return out
    if include_capabilities:
        out["capabilities"] = _caps.evaluate(prof)
        out["recommended"] = _caps.recommend(prof)
    return out


def list_visualization_capabilities(data_path: str = "", include_unsupported: bool = True) -> dict[str, Any]:
    """The catalogue, evaluated against a dataset when one is named and in the abstract when not."""
    if not data_path:
        # The rule ``capabilities.evaluate`` applies: the abstract listing named
        # plot_spatial_segmentation and plot_spatialdata_scene, which no portal registers, with no
        # flag saying so (hunt 2026-09-30, u20b-viz-rest-25).
        registered = _caps.registered_functions()
        rows = [
            {
                "plot_id": c.plot_id,
                "function": c.function,
                "title": c.title,
                "purpose": c.purpose,
                "kind": c.kind,
                "tier": c.tier,
                "reads_result": c.reads_result,
                "unsupported_reason": c.unsupported_reason,
                "implemented": (c.function in registered) if registered else True,
            }
            for c in _caps.CAPABILITIES.values()
            if include_unsupported or c.tier != _caps.TIER_UNSUPPORTED
        ]
        return {"status": "ok", "evaluated_against": "", "capabilities": rows}
    prof = _profile.profile_for_viz(data_path)
    if not prof["dataset"]["readable"]:
        return {"status": "error", "error": prof["dataset"]["read_error"]}
    return {
        "status": "ok",
        "evaluated_against": Path(data_path).name,
        "capabilities": _caps.evaluate(prof, include_unsupported=include_unsupported),
    }


def recommend_visualizations(data_path: str, question: str = "", limit: int = 8) -> dict[str, Any]:
    """A short ordered list of what is worth drawing for this dataset."""
    prof = _profile.profile_for_viz(data_path)
    if not prof["dataset"]["readable"]:
        return {"status": "error", "error": prof["dataset"]["read_error"]}
    # The description says the question orders the suggestions, and it was only echoed back, so a
    # model that phrased the user's goal got the same list for any goal (hunt 2026-09-30,
    # u20b-viz-rest-24). The same keyword relevance run_visualization_pipeline applies, over a wider
    # pool so a relevant plot just past the limit can come forward. No question: the list is as before.
    if question.strip():
        rows = _order_by_question(_caps.recommend(prof, limit=int(limit) * 3), question)[: int(limit)]
    else:
        rows = _caps.recommend(prof, limit=int(limit))
    return {
        "status": "ok",
        "question": question,
        "dataset": Path(data_path).name,
        "modality": prof["derived"]["modality"],
        "platform": prof["derived"]["platform"]["name"],
        "recommended": rows,
        "why_these": (
            "Ordered by what this dataset can answer on its own, then by what suits its modality, "
            "then by how much of its structure the plot uses. Plots that read another tool's "
            "output are listed last because nothing in the dataset says whether that tool has run."
            + (" Plots whose name or purpose shares words with the question come first." if question.strip() else "")
        ),
    }


def _order_by_question(rows: list[dict[str, Any]], question: str) -> list[dict[str, Any]]:
    """Stable sort by how many of the question's words (longer than three letters) a row mentions."""
    wanted = str(question or "").lower().strip()
    if not wanted:
        return list(rows)

    def relevance(row: dict[str, Any]) -> int:
        text = f"{row['plot_id']} {row.get('title', '')} {row.get('purpose', '')}".lower()
        return -sum(1 for word in wanted.split() if len(word) > 3 and word in text)

    return sorted(rows, key=relevance)


def validate_plot_request(data_path: str, plot_id: str, params_json: str = "") -> dict[str, Any]:
    """Can this exact request be drawn, and if not, what is missing and what would produce it."""
    import json

    try:
        params = json.loads(params_json) if params_json.strip() else {}
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"params_json is not valid JSON: {exc}"}
    prof = _profile.profile_for_viz(data_path)
    if not prof["dataset"]["readable"]:
        return {"status": "error", "error": prof["dataset"]["read_error"]}
    verdict = _caps.validate(plot_id, params, prof)
    verdict["status"] = "ok"
    verdict["dataset"] = Path(data_path).name
    if not verdict.get("ok") and verdict.get("error_kind") == "prerequisite_not_met":
        verdict["produced_by"] = _produced_by(verdict.get("missing") or [])
    return verdict


#: What would produce each missing prerequisite. Named calls, not advice.
_PRODUCED_BY: dict[str, dict[str, str]] = {
    "has_embedding": {
        "kind": "in_process",
        "call": "sc.pp.neighbors(adata); sc.tl.umap(adata)",
        "note": "needs a PCA first on a raw object: sc.pp.normalize_total, sc.pp.log1p, sc.pp.pca",
    },
    "has_de_effect_and_p": {
        "kind": "in_process",
        "call": "sc.tl.rank_genes_groups(adata, groupby=<column>, method='wilcoxon')",
        "note": (
            "scanpy stores fold changes and adjusted p-values together; a ranking that holds only "
            "names and scores was written by an older call or a different method"
        ),
    },
    "has_de_result": {
        "kind": "in_process",
        "call": "sc.tl.rank_genes_groups(adata, groupby=<column>, method='wilcoxon')",
        "note": "",
    },
    "has_coords": {
        "kind": "mcp_tool",
        "call": "convert the platform output with the data converter, which writes obsm['spatial']",
        "note": "",
    },
    "has_categorical_obs": {
        "kind": "mcp_tool",
        "call": "run a clustering or domain tool, which writes a categorical column such as spatial_domain",
        "note": "",
    },
    "has_proportions": {
        "kind": "mcp_tool",
        "call": "run a deconvolution tool, which writes per-spot proportions",
        "note": "",
    },
    "has_pseudotime": {
        "kind": "in_process",
        "call": "sc.tl.diffmap(adata); adata.uns['iroot'] = <a root you choose>; sc.tl.dpt(adata)",
        "note": "the root is a modelling decision and the figure will say who chose it",
    },
    "has_neighbors": {"kind": "in_process", "call": "sc.pp.neighbors(adata)", "note": ""},
    "has_qc_metrics": {
        "kind": "in_process",
        "call": "sc.pp.calculate_qc_metrics(adata, inplace=True)",
        "note": "",
    },
}


def _produced_by(missing: list[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for key in missing:
        entry = _PRODUCED_BY.get(key)
        sentence = _caps.PREDICATES.get(key, ("", None))[0]
        out.append(
            {
                "field": key,
                "why": sentence,
                **(entry or {"kind": "user_action", "call": "", "note": ""}),
            }
        )
    return out


@_layer_errors_refuse("spatial.expression")
def plot_spatial_expression(
    data_path: str,
    *,
    genes: str = "",
    obs_key: str = "",
    obsm_key: str = "",
    output_dir: str = "",
    layer: str = "",
    use_raw: bool = False,
    library_id: str = "",
    image_key: str = "",
    image_alpha: float = 1.0,
    spot_alpha: float = 1.0,
    point_size: float = 0.0,
    color_map: str = "",
    vmin: str = "",
    vmax: str = "",
    normalize: str = "auto",
    share_scale: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Paint continuous values on the tissue, optionally over the histology image."""
    asked = [v for v in (genes, obs_key, obsm_key) if v]
    if len(asked) != 1:
        return _refusal(
            "invalid_request",
            "name exactly one of genes, obs_key or obsm_key: they are three different sources and "
            "a figure drawn from two of them at once would have no single colour meaning",
            plot_id="spatial.expression",
        )

    prof = _profile.profile_for_viz(data_path, var_names=[g.strip() for g in genes.split(",") if g.strip()])
    if not prof["dataset"]["readable"]:
        return _refusal("unreadable", prof["dataset"]["read_error"], plot_id="spatial.expression")

    plot_id = "spatial.histology_overlay" if image_key else "spatial.expression"
    verdict = _caps.validate(plot_id, {}, prof)
    if not verdict.get("ok"):
        return _refusal(
            verdict["error_kind"],
            verdict["why"],
            plot_id=plot_id,
            missing=verdict.get("missing"),
            instead=verdict.get("instead"),
        )

    out_dir = Path(output_dir or ".").resolve()
    figures_dir = _manifest_io.figures_dir_for(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = list(prof.get("warnings") or [])
    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        try:
            coords, coords_from, section = _coords_and_mask(adata, library_id)
        except _layers.LayerError as exc:
            return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)

        image = None
        scalef = 1.0
        if image_key:
            try:
                image, scalef, _lib = _layers.histology(adata, library_id, image_key)
            except _layers.LayerError as exc:
                return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)
            if image is None:
                # An overlay was asked for and there is no image for this section: drawing the
                # spots alone under the overlay's id would answer a different question with ok.
                return _refusal(
                    "prerequisite_not_met",
                    f"no {image_key!r} image is stored"
                    + (f" for section {library_id!r}" if library_id else "")
                    + " in uns['spatial'], so there is nothing to draw under the spots",
                    plot_id=plot_id,
                    instead=["spatial.expression"],
                    fix="leave image_key empty to draw the spots alone, or name a section that has an image",
                )

        panels: dict[str, Any] = {}
        expression_record: dict[str, Any] | None = None
        semantic = "expression"
        if genes:
            wanted = [g.strip() for g in genes.split(",") if g.strip()]
            slot = _layers.resolve_expression_slot(
                adata,
                layer=layer,
                use_raw=use_raw,
                need_counts=True,
                x_has_negative=bool(prof["matrix"]["has_negative"]),
            )
            found, missing, ambiguous = _layers.gene_values(adata, wanted, slot)
            if not found:
                return _refusal(
                    "prerequisite_not_met",
                    f"none of {wanted} is a gene in this object"
                    + (f"; {ambiguous} appear more than once" if ambiguous else ""),
                    plot_id=plot_id,
                    fix="check the spelling, or use the identifiers the dataset inspection listed",
                )
            if missing:
                warnings.append(f"not drawn, absent from this object: {', '.join(missing)}")
            if ambiguous:
                warnings.append(
                    f"not drawn, the symbol appears more than once so it does not identify one row: {', '.join(ambiguous)}"
                )
            transform = ""
            integral = _slot_integral(slot, prof, found.values())
            for name, values in found.items():
                shown, transform = _layers.normalize_for_display(values, mode=normalize, integral=integral)
                panels[name] = shown
            expression_record = {
                "slot": slot["slot"],
                "key": slot["key"],
                "transform": transform,
                "integral": integral,
                "why": slot["why"],
            }
        elif obs_key:
            values, is_categorical = _layers.obs_values(adata, obs_key)
            if is_categorical:
                return _refusal(
                    "invalid_request",
                    f"obs['{obs_key}'] is categorical; this plot paints continuous values",
                    plot_id=plot_id,
                    instead=["spatial.annotation"],
                )
            panels[obs_key] = values
            semantic = "score"
        else:
            values, label = _layers.obsm_values(adata, obsm_key)
            import numpy as np

            array = np.asarray(values)
            if array.ndim == 2:
                names = [f"{label}[{i}]" for i in range(array.shape[1])]
                for i, name in enumerate(names):
                    panels[name] = array[:, i]
            else:
                panels[label] = array
            semantic = "proportion" if "proportion" in obsm_key.lower() else "score"

    import numpy as np

    section_note = ""
    if section is not None:
        panels = {name: np.asarray(values)[section] for name, values in panels.items()}
        coords = coords[section]
        section_note = f"only section {library_id!r} is drawn ({int(section.sum()):,} of {len(section):,} observations)"

    scales = _palette.PanelScales.build(panels, share=bool(share_scale), vmin=vmin, vmax=vmax, semantic=semantic)
    cmap = _palette.continuous(semantic, color_map)
    fmt = (figure_format or "png").lower()

    if len(panels) == 1:
        fig, axes = _base.canvas(1, 1, panel_w=5.0, panel_h=4.6, dpi=dpi or None)
        note = _spatial.tissue_scatter(
            axes[0],
            coords,
            next(iter(panels.values())),
            categorical=False,
            title=title or next(iter(panels)),
            value_label=next(iter(panels)),
            image=image,
            image_scalef=scalef,
            image_alpha=image_alpha,
            point_alpha=spot_alpha,
            point_size=point_size or None,
            cmap=cmap,
            vlim=scales.for_panel(0),
            fmt=fmt,
        )
        notes = {"panels": [note], "dropped": 0}
        fig.tight_layout()
    else:
        fig, notes = _spatial.tissue_panels(
            coords,
            panels,
            categorical=False,
            scales=scales,
            ncols=int(ncols),
            title=title,
            value_label="",
            image=image,
            image_scalef=scalef,
            image_alpha=image_alpha,
            point_alpha=spot_alpha,
            point_size=point_size or None,
            cmap=cmap,
            fmt=fmt,
        )

    data_params = {
        "data_path": data_path,
        "genes": genes,
        "obs_key": obs_key,
        "obsm_key": obsm_key,
        "layer": layer,
        "use_raw": use_raw,
        "normalize": normalize,
        "library_id": library_id,
    }
    stem = _spec.stem_for(plot_id, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    rel = _base.save(fig, figures_dir, stem, fmt=fmt, dpi=dpi or None)

    if notes.get("dropped"):
        warnings.append(f"{notes['dropped']} further panels were not drawn; the grid is capped")
    limitations: list[str] = [section_note] if section_note else []
    if image is not None:
        limitations.append(
            f"The image is the {image_key} resolution, registered with its own scalefactor ({scalef:.5f})"
        )
    caption = _spec.compose_caption(
        subject=title or f"{', '.join(list(panels)[:4])} on the tissue",
        expression=expression_record,
        scale_note=scales.caption(),
        limitations=limitations,
        claim="descriptive",
    )

    cache_name, cache_why = _spec.write_cache(
        figures_dir, stem, {"coords": coords, **{f"panel_{i}": v for i, v in enumerate(panels.values())}}
    )
    if cache_why:
        warnings.append(cache_why)

    record = _spec.new_spec(
        plot_id=plot_id,
        function="plot_spatial_expression",
        kind="spatial_map",
        stem=stem,
        revision=1,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "n_vars": prof["dataset"]["n_vars"],
            "coords_from": coords_from,
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "image_key": image_key,
            "color_map": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "share_scale": share_scale,
            "ncols": ncols,
            "figure_format": fmt,
        },
        param_origin={"color_map": "default" if not color_map else "caller"},
        expression=expression_record,
        caption=caption,
        limitations=limitations,
        outputs={"figure": rel, "cache": cache_name},
        warnings=warnings,
    )
    _spec.save(record, figures_dir)

    published = _manifest_io.publish(
        out_dir,
        tool_name="plot_spatial_expression",
        figures=[{"path": rel, "title": title or stem, "caption": caption, "kind": "spatial_map"}],
        findings=[("n_panels", len(panels), "panels drawn")],
        warnings=warnings,
    )
    if not published["written"] and published["why"]:
        warnings.append(published["why"])

    return {
        "status": "ok",
        "plot_id": plot_id,
        "figure_id": record["figure_id"],
        "figure": rel,
        "caption": caption,
        "panels": list(panels),
        "warnings": warnings,
        "output_dir": str(out_dir),
        "manifest": published["path"],
    }


# --------------------------------------------------------------------------------------------
# The shared tail. Every producer ends the same way, and factoring it is what keeps the spec,
# the cache and the manifest from drifting apart between families.
# --------------------------------------------------------------------------------------------


def _finalize(
    *,
    fig: Any,
    out_dir: Path,
    figures_dir: Path,
    plot_id: str,
    function: str,
    kind: str,
    stem: str,
    fmt: str,
    dpi: int,
    title: str,
    caption: str,
    limitations: list[str],
    warnings: list[str],
    source: dict[str, Any],
    params: dict[str, Any],
    param_origin: dict[str, Any],
    expression: dict[str, Any] | None,
    cache: dict[str, Any] | None,
    findings: list[tuple[str, Any, str]] | None = None,
    tables: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rel = _base.save(fig, figures_dir, stem, fmt=fmt, dpi=dpi or None)
    params = _names_only(params)
    cache_name = ""
    if cache:
        cache_name, cache_why = _spec.write_cache(figures_dir, stem, cache)
        if cache_why:
            warnings.append(cache_why)
    record = _spec.new_spec(
        plot_id=plot_id,
        function=function,
        kind=kind,
        stem=stem,
        revision=1,
        source=source,
        params=params,
        param_origin=param_origin,
        expression=expression,
        caption=caption,
        limitations=limitations,
        outputs={"figure": rel, "cache": cache_name},
        warnings=warnings,
    )
    _spec.save(record, figures_dir)
    published = _manifest_io.publish(
        out_dir,
        tool_name=function,
        figures=[{"path": rel, "title": title or stem, "caption": caption, "kind": kind}],
        tables=tables or [],
        findings=findings or [],
        warnings=warnings,
    )
    if not published["written"] and published["why"]:
        warnings.append(published["why"])
    return {
        "status": "ok",
        "plot_id": plot_id,
        "figure_id": record["figure_id"],
        "figure": rel,
        "caption": caption,
        "warnings": warnings,
        "output_dir": str(out_dir),
        "manifest": published["path"],
    }


#: Parameters that name an input file. The sidecar is served raw through the results route, and
#: ``spec.save`` forbids an absolute path in it, but seven families recorded the full data_path,
#: results_path or proportions table (hunt 2026-09-30, u20a-viz-pipelines-26).
_PATH_PARAMS = ("data_path", "results_path", "proportions_csv", "result_path", "pvalues_path")


def _names_only(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    for key in _PATH_PARAMS:
        value = out.get(key)
        if isinstance(value, str) and value:
            out[key] = Path(value).name
    return out


def _prepare(
    data_path: str,
    plot_id: str,
    output_dir: str,
    *,
    satisfied: tuple[str, ...] = (),
    selectors: dict[str, Any] | None = None,
    **profile_kw: Any,
):
    """Profile, validate, and make the output directory. Returns ``(profile, dirs)`` or a refusal.

    ``satisfied`` names prerequisites the request itself supplies -- a proportions table passed
    beside the object stands in for ``has_proportions``, which looks only in obsm. ``selectors`` are
    the call's own arguments that name something in the dataset (a section column, a pseudotime
    column, an obsm matrix, a spacing); one that names something real satisfies the prerequisite
    it selects for, by ``capabilities.supplied_by``.
    """
    prof = _profile.profile_for_viz(data_path, **profile_kw)
    if not prof["dataset"]["readable"]:
        return None, _refusal("unreadable", prof["dataset"]["read_error"], plot_id=plot_id)
    verdict = _caps.validate(plot_id, {}, prof)
    # A selector the caller named satisfies the prerequisite it selects for; the gate read only the
    # profile's fixed names, so plot_section_grid(section_key='sample_id'), plot_trajectory(
    # pseudotime_key='latent_time'), plot_spatial_3d(z_spacing=...) and plot_deconvolution(
    # obsm_key=<an abundance matrix>) were refused before the producer that honours them (hunt
    # 2026-09-30, u20b-viz-rest-16, -7, -4).
    satisfied = (*satisfied, *_caps.supplied_by(plot_id, selectors or {}, prof))
    if (
        not verdict.get("ok")
        and satisfied
        and verdict.get("error_kind") == "prerequisite_not_met"
        and set(verdict.get("missing") or []) <= set(satisfied)
    ):
        verdict = {"ok": True}
    if not verdict.get("ok"):
        refusal = _refusal(
            verdict["error_kind"],
            verdict["why"],
            plot_id=plot_id,
            missing=verdict.get("missing"),
            instead=verdict.get("instead"),
        )
        if verdict.get("error_kind") == "prerequisite_not_met":
            refusal["produced_by"] = _produced_by(verdict.get("missing") or [])
        return None, refusal
    out_dir = Path(output_dir or ".").resolve()
    figures_dir = _manifest_io.figures_dir_for(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return (prof, out_dir, figures_dir), None


@_layer_errors_refuse("spatial.annotation")
def plot_spatial_annotation(
    data_path: str,
    *,
    obs_key: str = "",
    output_dir: str = "",
    library_id: str = "",
    image_key: str = "",
    image_alpha: float = 1.0,
    spot_alpha: float = 1.0,
    point_size: float = 0.0,
    split_panels: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Paint a categorical annotation -- a domain, a cell type, a cluster -- on the tissue."""
    prepared, refusal = _prepare(data_path, "spatial.annotation", output_dir)
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared

    key = obs_key or prof["obs_roles"]["cluster"] or prof["obs_roles"]["cell_type"]
    if not key:
        return _refusal(
            "prerequisite_not_met",
            "no obs column was named and none looks like a domain, cluster or cell type",
            plot_id="spatial.annotation",
            missing=["has_cluster_or_cell_type"],
            instead=["spatial.expression"],
        )
    warnings: list[str] = list(prof.get("warnings") or [])
    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        try:
            coords, coords_from, section = _coords_and_mask(adata, library_id)
            values, is_categorical = _layers.obs_values(adata, key)
        except _layers.LayerError as exc:
            return _refusal("prerequisite_not_met", str(exc), plot_id="spatial.annotation", fix=exc.fix)
        if not is_categorical:
            return _refusal(
                "invalid_request",
                f"obs['{key}'] is continuous; this plot paints categories",
                plot_id="spatial.annotation",
                instead=["spatial.expression"],
            )
        image, scalef = (None, 1.0)
        if image_key:
            try:
                image, scalef, _lib = _layers.histology(adata, library_id, image_key)
            except _layers.LayerError as exc:
                return _refusal("prerequisite_not_met", str(exc), plot_id="spatial.annotation", fix=exc.fix)
            if image is None:
                return _refusal(
                    "prerequisite_not_met",
                    f"no {image_key!r} image is stored"
                    + (f" for section {library_id!r}" if library_id else "")
                    + " in uns['spatial'], so there is nothing to draw under the spots",
                    plot_id="spatial.annotation",
                    fix="leave image_key empty to draw the spots alone, or name a section that has an image",
                )

    import numpy as np

    section_note = ""
    if section is not None:
        # (hunt 2026-09-30, u20a-viz-pipelines-1) the labels follow the coordinates to one section.
        values = np.asarray(values)[section]
        coords = coords[section]
        section_note = f"only section {library_id!r} is drawn ({int(section.sum()):,} of {len(section):,} observations)"
    fmt = (figure_format or "png").lower()
    levels = list(dict.fromkeys(np.asarray(values).astype(object).tolist()))
    if split_panels:
        panels = {str(lvl): (np.asarray(values).astype(object) == lvl).astype(float) for lvl in levels}
        scales = _palette.PanelScales.build(panels, share=True, semantic="proportion")
        fig, notes = _spatial.tissue_panels(
            coords,
            panels,
            categorical=False,
            scales=scales,
            ncols=int(ncols),
            title=title or f"{key}, one panel per level",
            value_label="",
            image=image,
            image_scalef=scalef,
            image_alpha=image_alpha,
            point_alpha=spot_alpha,
            point_size=point_size or None,
            cmap="cividis",
            fmt=fmt,
        )
        scale_note = scales.caption()
    else:
        fig, axes = _base.canvas(1, 1, panel_w=5.6, panel_h=4.8, dpi=dpi or None)
        note = _spatial.tissue_scatter(
            axes[0],
            coords,
            values,
            categorical=True,
            title=title or key,
            value_label=key,
            image=image,
            image_scalef=scalef,
            image_alpha=image_alpha,
            point_alpha=spot_alpha,
            point_size=point_size or None,
            fmt=fmt,
        )
        notes = {"panels": [note], "dropped": 0}
        scale_note = ""
        fig.tight_layout()

    pooled = notes["panels"][0].get("pooled") if notes.get("panels") else []
    limitations: list[str] = [section_note] if section_note else []
    if pooled:
        limitations.append(
            f"{len(pooled)} rare levels share one grey because the palette holds "
            f"{_palette.max_categories()} distinct colours"
        )
    if len(levels) == 1:
        limitations.append("every observation has the same label, so this map draws no distinction")
    if image is not None:
        limitations.append(
            f"the image is the {image_key} resolution, registered with its own scalefactor ({scalef:.5f})"
        )

    caption = _spec.compose_caption(
        subject=title or f"{key} on the tissue ({len(levels)} levels)",
        expression=None,
        scale_note=scale_note,
        limitations=limitations,
        claim="descriptive",
    )
    data_params = {"data_path": data_path, "obs_key": key, "library_id": library_id}
    stem = _spec.stem_for("spatial.annotation", prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id="spatial.annotation",
        function="plot_spatial_annotation",
        kind="spatial_map",
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=title or key,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "coords_from": coords_from,
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "image_key": image_key,
            "split_panels": split_panels,
            "ncols": ncols,
            "figure_format": fmt,
        },
        param_origin={"obs_key": "caller" if obs_key else "inferred from the dataset"},
        expression=None,
        cache={"coords": coords, "labels": np.asarray(values).astype(str)},
        findings=[("n_levels", len(levels), f"levels of {key}")],
    )


@_layer_errors_refuse("embedding.scatter")
def plot_embedding(
    data_path: str,
    *,
    color: str = "",
    basis: str = "auto",
    output_dir: str = "",
    layer: str = "",
    use_raw: bool = False,
    groupby: str = "",
    legend_loc: str = "right margin",
    point_size: float = 0.0,
    color_map: str = "",
    vmin: str = "",
    vmax: str = "",
    normalize: str = "auto",
    share_scale: bool = False,
    ncols: int = 3,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Scatter a stored embedding, coloured by genes or metadata, optionally split by a grouping."""
    prepared, refusal = _prepare(
        data_path,
        "embedding.scatter",
        output_dir,
        var_names=[g.strip() for g in color.split(",") if g.strip()],
    )
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared

    embeddings = [o["key"] for o in prof["obsm"] if o["family"] == "embedding"]
    wanted_basis = {"auto": "", "umap": "X_umap", "tsne": "X_tsne", "pca": "X_pca"}.get(basis, basis)
    if wanted_basis:
        chosen = wanted_basis if wanted_basis in embeddings else ""
    else:
        chosen = next((k for k in ("X_umap", "X_tsne", "X_pca") if k in embeddings), "")
    if not chosen:
        return _refusal(
            "prerequisite_not_met",
            f"this object has no {basis!r} embedding; it has {embeddings or 'none at all'}",
            plot_id="embedding.scatter",
            missing=["has_embedding"],
            instead=["spatial.annotation"] if prof["spatial"]["has_obsm_spatial"] else [],
        )

    warnings: list[str] = list(prof.get("warnings") or [])
    import numpy as np

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        xy = np.asarray(adata.obsm[chosen], dtype=float)[:, :2]
        requested = [c.strip() for c in color.split(",") if c.strip()]
        if not requested:
            fallback = prof["obs_roles"]["cluster"] or prof["obs_roles"]["cell_type"]
            requested = [fallback] if fallback else []
        panels: dict[str, Any] = {}
        # Per panel, not per figure: one gene used to make the whole figure continuous, so a named
        # cell type beside it crashed the float cast and numeric cluster ids were painted on a ramp
        # (hunt 2026-09-30, u20a-viz-pipelines-18).
        panel_is_cat: dict[str, bool] = {}
        expression_record: dict[str, Any] | None = None
        genes_wanted: list[str] = []
        found_genes: set[str] = set()
        for name in requested:
            if name in adata.obs.columns:
                values, is_cat = _layers.obs_values(adata, name)
                panels[name] = values
                panel_is_cat[name] = bool(is_cat)
            else:
                genes_wanted.append(name)
        if genes_wanted:
            slot = _layers.resolve_expression_slot(
                adata,
                layer=layer,
                use_raw=use_raw,
                need_counts=True,
                x_has_negative=bool(prof["matrix"]["has_negative"]),
            )
            found, missing, ambiguous = _layers.gene_values(adata, genes_wanted, slot)
            transform = ""
            integral = _slot_integral(slot, prof, found.values())
            for name, values in found.items():
                shown, transform = _layers.normalize_for_display(values, mode=normalize, integral=integral)
                panels[name] = shown
                panel_is_cat[name] = False
                found_genes.add(name)
            if found:
                expression_record = {
                    "slot": slot["slot"],
                    "key": slot["key"],
                    "transform": transform,
                    "integral": integral,
                    "why": slot["why"],
                }
            if missing:
                warnings.append(f"not drawn, absent from this object: {', '.join(missing)}")
            if ambiguous:
                warnings.append(f"not drawn, the symbol is not unique: {', '.join(ambiguous)}")
        if not panels:
            return _refusal(
                "invalid_request",
                "nothing to colour by: none of the names given is an obs column or a gene in this object",
                plot_id="embedding.scatter",
            )
        group_values = None
        if groupby:
            group_values, group_is_cat = _layers.obs_values(adata, groupby)
            if not group_is_cat:
                return _refusal(
                    "invalid_request",
                    f"obs['{groupby}'] is continuous; a facet needs categories",
                    plot_id="embedding.facet",
                )

    fmt = (figure_format or "png").lower()
    basis_label = chosen.replace("X_", "").upper()
    categorical = all(panel_is_cat.values())
    if group_values is not None:
        # A facet draws ONE colour per panel grid. Every other requested colour used to be dropped
        # silently while the caption and n_panels still named it, and vmin/vmax were ignored (hunt
        # 2026-09-30, u20a-viz-pipelines-30). Draw the first, say so, and record only what was drawn.
        first_name = next(iter(panels))
        if len(panels) > 1:
            warnings.append(
                f"a split by {groupby} draws one colour; {', '.join(list(panels)[1:])} were not drawn -- "
                "call again with each of them as color"
            )
        panels = {first_name: panels[first_name]}
        categorical = panel_is_cat[first_name]
        if first_name not in found_genes:
            expression_record = None
        first = panels[first_name]
        scales = (
            _palette.PanelScales.build({"all": first}, share=True, vmin=vmin, vmax=vmax) if not categorical else None
        )
        fig, notes = _points.facet_by(
            xy,
            group_values,
            colour_by=first,
            categorical=categorical,
            ncols=int(ncols),
            title=title or f"{next(iter(panels))} split by {groupby}",
            basis_label=basis_label,
            point_size=point_size or None,
            cmap=_palette.continuous("expression", color_map),
            vlim=scales.for_panel(0) if scales else (None, None),
            fmt=fmt,
        )
        scale_note = (
            "All panels share axes and one colour scale, so they are comparable."
            if not categorical
            else "All panels share axes."
        )
        plot_id = "embedding.facet"
    else:
        # Scales over the continuous panels only: a categorical panel has no colour limits, and
        # casting its labels to float is what crashed (hunt 2026-09-30, u20a-viz-pipelines-18).
        continuous_panels = {n: v for n, v in panels.items() if not panel_is_cat[n]}
        scales = (
            _palette.PanelScales.build(continuous_panels, share=bool(share_scale), vmin=vmin, vmax=vmax)
            if continuous_panels
            else None
        )
        if len(panels) == 1:
            fig, axes = _base.canvas(1, 1, panel_w=5.2, panel_h=4.6, dpi=dpi or None)
            note = _points.embedding_scatter(
                axes[0],
                xy,
                next(iter(panels.values())),
                categorical=categorical,
                title=title or next(iter(panels)),
                value_label=next(iter(panels)),
                basis_label=basis_label,
                point_size=point_size or None,
                cmap=_palette.continuous("expression", color_map),
                vlim=scales.for_panel(0) if scales else (None, None),
                legend=legend_loc,
                fmt=fmt,
            )
            notes = {"panels": [note], "dropped": 0}
            fig.tight_layout()
        elif len(set(panel_is_cat.values())) > 1:
            fig, notes = _mixed_embedding_panels(
                xy,
                panels,
                panel_is_cat,
                scales,
                ncols=int(ncols),
                title=title,
                basis_label=basis_label,
                point_size=point_size or None,
                cmap=_palette.continuous("expression", color_map),
                legend=legend_loc,
                fmt=fmt,
            )
        else:
            fig, notes = _points.embedding_panels(
                xy,
                panels,
                categorical=categorical,
                scales=scales,
                ncols=int(ncols),
                title=title,
                basis_label=basis_label,
                point_size=point_size or None,
                cmap=_palette.continuous("expression", color_map),
                legend=legend_loc,
                fmt=fmt,
            )
        scale_note = scales.caption() if scales else ""
        plot_id = "embedding.scatter"

    if notes.get("dropped"):
        warnings.append(f"{notes['dropped']} further panels were not drawn; the grid is capped")
    limitations = [f"positions are {basis_label} coordinates, which preserve neighbourhoods and not distances"]
    caption = _spec.compose_caption(
        subject=title or f"{', '.join(list(panels)[:4])} on {basis_label}",
        expression=expression_record,
        scale_note=scale_note,
        limitations=limitations,
        claim="descriptive",
    )
    data_params = {
        "data_path": data_path,
        "color": color,
        "basis": chosen,
        "layer": layer,
        "use_raw": use_raw,
        "groupby": groupby,
        "normalize": normalize,
    }
    stem = _spec.stem_for(plot_id, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_embedding",
        kind="grid" if len(panels) > 1 or groupby else "scatter",
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=title or basis_label,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "basis": chosen,
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "ncols": ncols,
            "legend_loc": legend_loc,
            "share_scale": share_scale,
            "figure_format": fmt,
        },
        param_origin={"basis": "caller" if basis != "auto" else "the first embedding present"},
        expression=expression_record,
        cache={"xy": xy, **{f"panel_{i}": np.asarray(v) for i, v in enumerate(panels.values())}},
        findings=[("n_panels", len(panels), "panels drawn")],
    )


def _mixed_embedding_panels(
    xy: Any,
    panels: dict[str, Any],
    is_categorical: dict[str, bool],
    scales: Any,
    *,
    ncols: int,
    title: str,
    basis_label: str,
    point_size: float | None,
    cmap: str,
    legend: str,
    fmt: str,
) -> tuple[Any, dict[str, Any]]:
    """``render.points.embedding_panels`` with the categorical decision made per panel.

    That renderer takes one flag for the whole grid, which is right when every panel is the same
    kind of value and wrong for ``color='cell_type,CD3E'`` -- the case the tool description names.
    """
    names = list(panels)
    rows, cols, shown = _base.panel_grid(len(names), ncols)
    fig, axes = _base.canvas(rows, cols, panel_w=4.4, panel_h=4.0)
    notes: dict[str, Any] = {"panels": [], "dropped": max(0, len(names) - shown)}
    continuous = [n for n in names if not is_categorical[n]]
    for i in range(shown):
        name = names[i]
        categorical = is_categorical[name]
        vlim = scales.for_panel(continuous.index(name)) if (scales is not None and not categorical) else (None, None)
        note = _points.embedding_scatter(
            axes[i],
            xy,
            panels[name],
            categorical=categorical,
            title=name,
            value_label=name,
            basis_label=basis_label if i == 0 else "",
            point_size=point_size,
            cmap=cmap,
            vlim=vlim,
            legend=legend,
            fmt=fmt,
            show_colorbar=not categorical,
        )
        note["name"] = name
        notes["panels"].append(note)
    for j in range(shown, len(axes)):
        _base.blank(axes[j])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig, notes


@_layer_errors_refuse("qc.overview")
def generate_qc_report(
    data_path: str,
    *,
    output_dir: str = "",
    groupby: str = "",
    compute_if_missing: bool = True,
    on_tissue: bool = True,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Counts, detected genes and mitochondrial fraction, as distributions and on the tissue.

    One multi-panel figure rather than six single ones: the chat card shows four figures, and a
    quality-control overview is one thing a reader looks at, not six.
    """
    prepared, refusal = _prepare(data_path, "qc.overview", output_dir)
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])
    computed: list[str] = []

    import numpy as np

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        metrics = [c for c in ("total_counts", "n_genes_by_counts", "pct_counts_mt") if c in adata.obs.columns]
        if len(metrics) < 2 and compute_if_missing:
            import scanpy as sc

            # Counts, not whatever X holds. On the common processed shape -- log-normalised X with
            # the raw counts kept in layers['counts'] -- the metrics were computed on X, so
            # "total_counts" was a sum of log1p values, orders of magnitude below the library size,
            # under a quality-control caption (hunt 2026-09-30, u20a-viz-pipelines-20).
            counts_layer = "counts" if "counts" in (getattr(adata, "layers", {}) or {}) else None
            if counts_layer is None and prof["matrix"]["integral"] is False:
                return _refusal(
                    "prerequisite_not_met",
                    "this object stores no quality-control metrics, its X holds transformed values "
                    "rather than counts, and there is no layers['counts'] to compute them from; a "
                    "library size summed over transformed values is not a library size",
                    plot_id="qc.overview",
                    missing=["has_qc_metrics"],
                    fix="keep the raw counts in layers['counts'] (or compute the metrics before normalising) and draw again",
                )
            adata.var["mt"] = adata.var_names.str.upper().str.startswith(("MT-", "MT."))
            sc.pp.calculate_qc_metrics(
                adata, qc_vars=["mt"], inplace=True, log1p=False, percent_top=None, layer=counts_layer
            )
            computed.append("sc.pp.calculate_qc_metrics on " + ("layers['counts']" if counts_layer else "X"))
            metrics = [c for c in ("total_counts", "n_genes_by_counts", "pct_counts_mt") if c in adata.obs.columns]
        if not metrics:
            return _refusal(
                "prerequisite_not_met",
                "this object carries no quality-control metrics and they were not computed",
                plot_id="qc.overview",
                missing=["has_qc_metrics"],
            )
        values = {m: adata.obs[m].to_numpy(dtype=float) for m in metrics}
        groups = None
        if groupby:
            groups, is_cat = _layers.obs_values(adata, groupby)
            if not is_cat:
                return _refusal(
                    "invalid_request",
                    f"obs['{groupby}'] is continuous; a grouping needs categories",
                    plot_id="qc.overview",
                )
        coords = None
        if on_tissue and prof["spatial"]["has_obsm_spatial"]:
            coords, _from = _layers.spatial_coords(adata)

    fmt = (figure_format or "png").lower()
    n_panels = len(metrics) + (len(metrics) if coords is not None else 0)
    rows, cols, shown = _base.panel_grid(n_panels, 3)
    fig, axes = _base.canvas(rows, cols, panel_w=4.0, panel_h=3.2, dpi=dpi or None)
    i = 0
    for metric in metrics:
        ax = axes[i]
        if groups is not None:
            levels = list(dict.fromkeys(np.asarray(groups).astype(object).tolist()))
            ax.violinplot(
                [values[metric][np.asarray(groups).astype(object) == lvl] for lvl in levels], showextrema=False
            )
            ax.set_xticks(range(1, len(levels) + 1))
            ax.set_xticklabels([str(lvl) for lvl in levels], rotation=45, ha="right", fontsize=7)
        else:
            ax.hist(values[metric][np.isfinite(values[metric])], bins=60, color="#0f7d76")
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
        i += 1
    if coords is not None:
        for metric in metrics:
            if i >= shown:
                break
            _spatial.tissue_scatter(
                axes[i],
                coords,
                values[metric],
                categorical=False,
                title=f"{metric} on the tissue",
                value_label="",
                fmt=fmt,
            )
            i += 1
    for j in range(i, len(axes)):
        _base.blank(axes[j])
    fig.suptitle(title or f"Quality control: {prof['dataset']['n_obs']:,} observations", fontsize=12)
    fig.tight_layout()

    limitations = []
    if computed:
        limitations.append(f"the metrics were computed here ({', '.join(computed)}) and are not stored in the object")
    if "pct_counts_mt" in metrics and prof["var"]["n_mito"] == 0:
        limitations.append(
            "no mitochondrial genes were found by prefix, so the mitochondrial fraction is zero by construction rather than by measurement"
        )
    caption = _spec.compose_caption(
        subject=title or "Quality-control overview",
        expression=None,
        limitations=limitations,
        claim="descriptive",
    )
    data_params = {"data_path": data_path, "groupby": groupby}
    stem = _spec.stem_for("qc.overview", prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id="qc.overview",
        function="generate_qc_report",
        kind="grid",
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=title or "Quality control",
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "on_tissue": coords is not None,
            "figure_format": fmt,
        },
        param_origin={"metrics": "computed here" if computed else "stored in the object"},
        expression=None,
        cache=dict(values),
        findings=[("n_metrics", len(metrics), "quality-control metrics shown")],
    )


@_layer_errors_refuse("de.volcano")
def plot_differential_expression(
    data_path: str,
    *,
    group: str = "",
    output_dir: str = "",
    kind: str = "volcano",
    top_n: int = 15,
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a stored differential-expression result. Refuses rather than inventing an axis.

    A volcano needs an effect size and a corrected p-value, and a stored ranking that carries
    only names and a test statistic has neither. The statistic is not an effect size and no
    p-value can be recovered from it without the null it was computed against, so this refuses
    and names the call that would produce a complete result.
    """
    if kind not in ("volcano", "ranked"):
        # Any other word used to become the ranked bar chart, so a run asking for "de.heatmap" got
        # a volcano or a bar chart under that id (hunt 2026-09-30, u20a-viz-pipelines-28).
        return _refusal(
            "invalid_request",
            f"kind must be 'volcano' or 'ranked', not {kind!r}; this function draws no heatmap",
            plot_id="de.volcano",
            instead=["markers.heatmap"],
            fix="for a heatmap of the stored ranking's top genes, call plot_marker_expression with kind='heatmap'",
        )
    want = "de.volcano" if kind == "volcano" else "de.ranked"
    prepared, refusal = _prepare(data_path, want, output_dir)
    if refusal:
        refusal["never_do"] = (
            "do not plot the stored test statistic on either axis and call it a volcano, and do "
            "not derive a p-value from a statistic whose null was not recorded"
        )
        return refusal
    prof, out_dir, figures_dir = prepared
    detail = prof["uns_analyses"]["rank_genes_groups_detail"]
    groups = detail.get("groups") or []
    chosen = group or (groups[0] if groups else "")
    if chosen and groups and chosen not in groups:
        return _refusal(
            "invalid_request",
            f"{chosen!r} is not one of the groups in the stored result; it has {groups}",
            plot_id=want,
        )

    warnings: list[str] = list(prof.get("warnings") or [])
    import numpy as np
    import pandas as pd

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        block = adata.uns["rank_genes_groups"]
        frame = pd.DataFrame({f: np.asarray(block[f][chosen]) for f in detail["fields"] if f in block})

    fmt = (figure_format or "png").lower()
    fig, axes = _base.canvas(1, 1, panel_w=5.4, panel_h=4.6, dpi=dpi or None)
    if kind == "volcano":
        note = _points.volcano(
            axes[0],
            frame["logfoldchanges"],
            frame["pvals_adj"],
            frame["names"],
            title=title or f"{chosen} versus the rest",
            fc_threshold=fc_threshold,
            p_threshold=p_threshold,
            label_top=top_n,
        )
        plot_kind = "scatter"
        if note["p_floor_clamped"]:
            warnings.append(
                f"{note['p_floor_clamped']} genes had an adjusted p-value of exactly zero, which is the "
                "correction's floor rather than a measurement; they are drawn at the top of the axis"
            )
        limitations = [
            f"selection is |log2 fold change| >= {fc_threshold} and adjusted p < {p_threshold}; "
            "these are thresholds, not a statement about biological importance"
        ]
        subject = title or f"Differential expression, {chosen} versus the rest"
        claim = "tested"
    else:
        statistic = "scores" if "scores" in frame else detail["fields"][0]
        top = frame.reindex(frame[statistic].abs().sort_values(ascending=False).index)[: int(top_n)]
        axes[0].barh(range(len(top))[::-1], top[statistic].to_numpy(), color="#0f7d76")
        axes[0].set_yticks(range(len(top))[::-1])
        axes[0].set_yticklabels([str(v) for v in top["names"]], fontsize=8)
        axes[0].set_xlabel(statistic, fontsize=9)
        axes[0].set_title(title or f"Top {len(top)} for {chosen}", fontsize=10)
        plot_kind = "bar"
        limitations = [f"ranked by {statistic}, which is the stored test statistic and not an effect size"]
        subject = title or f"Top markers for {chosen}"
        claim = "descriptive"
        note = {"n_points": int(len(frame)), "n_selected": int(len(top))}
    fig.tight_layout()

    method = (detail.get("params") or {}).get("method", "")
    groupby_col = (detail.get("params") or {}).get("groupby", "")
    if method:
        limitations.append(f"the ranking was computed with {method} over obs['{groupby_col}'], as stored in the object")
    limitations.append(
        "each observation is one cell or spot, not a biological replicate; this compares groups "
        "within one sample and is not a condition-level result"
    )
    caption = _spec.compose_caption(
        subject=subject,
        expression=None,
        limitations=limitations,
        claim=claim,
    )
    data_params = {"data_path": data_path, "group": chosen, "top_n": top_n}
    stem = _spec.stem_for(want, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=want,
        function="plot_differential_expression",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "read_from": "uns['rank_genes_groups']",
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "kind": kind,
            "fc_threshold": fc_threshold,
            "p_threshold": p_threshold,
            "figure_format": fmt,
        },
        param_origin={"group": "caller" if group else "the first group in the stored result"},
        expression=None,
        cache={c: frame[c].to_numpy() for c in frame.columns if frame[c].dtype.kind in "fiu"},
        findings=[("n_selected", int(note.get("n_selected", 0)), "genes past both thresholds")],
    )


@_layer_errors_refuse("deconv.proportions")
def plot_deconvolution(
    data_path: str,
    *,
    obsm_key: str = "",
    proportions_csv: str = "",
    output_dir: str = "",
    kind: str = "maps",
    library_id: str = "",
    ncols: int = 3,
    top_n: int = 9,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Cell-type proportions from a deconvolution: per-type maps, the dominant type, or a summary.

    Reads what a deconvolution tool already wrote -- a proportions matrix in ``obsm`` or the
    barcode-by-cell-type table those tools emit -- and never re-infers proportions itself.
    """
    plot_id = {"maps": "deconv.proportions", "dominant": "deconv.dominant", "summary": "deconv.composition"}.get(
        kind, ""
    )
    if not plot_id:
        return _refusal(
            "invalid_request", f"kind must be maps, dominant or summary, not {kind!r}", plot_id="deconv.proportions"
        )
    # A proportions table passed beside the object IS the proportions source, and the capability's
    # has_proportions looks only in obsm -- so the documented input, a spatial object plus the CSV a
    # deconvolution tool wrote, was refused before the CSV was read (hunt 2026-09-30,
    # u20a-viz-pipelines-11).
    # An obsm_key the caller names is the source too -- a cell2location abundance matrix whose rows
    # do not sum to one was refused "an obsm matrix holds per-observation proportions that sum to
    # one" (hunt 2026-09-30, u20b-viz-rest-4).
    prepared, refusal = _prepare(
        data_path,
        plot_id,
        output_dir,
        satisfied=("has_proportions",) if proportions_csv else (),
        selectors={"obsm_key": obsm_key},
    )
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    import numpy as np
    import pandas as pd

    frame: pd.DataFrame | None = None
    source_note = ""
    if proportions_csv:
        try:
            frame = pd.read_csv(proportions_csv, index_col=0)
            source_note = Path(proportions_csv).name
        except Exception as exc:
            return _refusal("invalid_request", f"the proportions table could not be read: {exc}", plot_id=plot_id)
    coords = None
    section_note = ""
    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        obs_names = np.asarray(adata.obs_names).astype(str)
        # Coordinates through the same resolver every other family uses, which falls back to an
        # obs x/y pair. Reading them only when obsm['spatial'] existed turned a 'maps' request on
        # such an object into a bar summary recorded under the map's id (hunt 2026-09-30,
        # u20a-viz-pipelines-34); a map without coordinates is now refused rather than substituted.
        try:
            coords_all, _from, section = _coords_and_mask(adata, library_id)
        except _layers.LayerError as exc:
            if kind != "summary" or library_id:
                return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)
            coords_all, section = None, None
        selected = ""
        if frame is None:
            candidates = [o["key"] for o in prof["obsm"] if o.get("family") == "proportions"]
            # 'matrix:column' and 'matrix:*', as the tool description documents: the bare
            # adata.obsm[obsm_key] lookup raised KeyError on 'proportions:*' (hunt 2026-09-30,
            # u20a-viz-pipelines-24). The whole matrix is read either way, so the row sums below are
            # the matrix's, not the one column's.
            key, _, selected = (obsm_key or (candidates[0] if candidates else "")).partition(":")
            selected = "" if selected.strip() == "*" else selected.strip()
            if not key:
                return _refusal(
                    "prerequisite_not_met",
                    "no proportions matrix was named and none in obsm has rows summing to one",
                    plot_id=plot_id,
                    missing=["has_proportions"],
                    instead=["spatial.annotation"],
                    fix="run a deconvolution tool first, or pass its proportions table as proportions_csv",
                )
            if key not in adata.obsm:
                return _refusal(
                    "prerequisite_not_met",
                    f"this object has no obsm matrix named {key!r}; it has {sorted(adata.obsm)}",
                    plot_id=plot_id,
                    fix="name one of the matrices above, or pass the proportions table as proportions_csv",
                )
            matrix = adata.obsm[key]
            try:
                labels = [str(c) for c in matrix.columns]
            except Exception:
                labels = [f"{key}[{i}]" for i in range(np.asarray(matrix).shape[1])]
            frame = pd.DataFrame(np.asarray(matrix), columns=labels, index=obs_names)
            present = np.ones(len(obs_names), dtype=bool)
            source_note = f"obsm['{key}']"
            if selected and selected not in frame.columns:
                try:
                    selected = str(frame.columns[int(selected)])
                except (ValueError, IndexError):
                    return _refusal(
                        "invalid_request",
                        f"obsm[{key!r}] has no column named {selected!r}; it has {labels[:24]}",
                        plot_id=plot_id,
                        fix=f"name one of the columns above, or obsm_key='{key}' for every cell type",
                    )
            if selected and kind == "dominant":
                return _refusal(
                    "invalid_request",
                    "the dominant type is chosen among every cell type, so it cannot be drawn from one column",
                    plot_id=plot_id,
                    fix=f"pass obsm_key='{key}' for the dominant map, or kind='maps' for {selected!r} alone",
                )
        else:
            # By barcode, never by row position (hunt 2026-09-30, u20a-viz-pipelines-12).
            frame, present, refusal = _align_rows_to_obs(frame, obs_names, "proportions table", plot_id, warnings)
            if refusal:
                return refusal
            # A deconvolution table often carries a label column beside the proportions (a
            # first_type, a dominant call); the float cast below raised on it, so the documented
            # input crashed rather than drawing its proportions (hunt 2026-09-30, found beside
            # u20a-viz-pipelines-11).
            text = [str(c) for c in frame.columns if c not in frame.select_dtypes("number").columns]
            frame = frame.select_dtypes("number")
            if frame.shape[1] == 0:
                return _refusal(
                    "invalid_request",
                    f"{Path(proportions_csv).name} holds no numeric column, so it holds no proportions",
                    plot_id=plot_id,
                    fix="pass the observations-by-cell-type table of proportions the deconvolution tool wrote",
                )
            if text:
                warnings.append(
                    f"{len(text)} column(s) of the proportions table hold text rather than proportions and "
                    f"are not drawn: {', '.join(text[:6])}"
                )

    keep = present if section is None else (present & section)
    if section is not None:
        section_note = f"only section {library_id!r} is drawn ({int(keep.sum()):,} of {len(keep):,} observations)"
    frame = frame[keep]
    if coords_all is not None:
        coords = np.asarray(coords_all, dtype=float)[keep]
    if frame is None or frame.empty:
        return _refusal("prerequisite_not_met", "the proportions table is empty", plot_id=plot_id)
    values = frame.to_numpy(dtype=float)
    row_sums = values.sum(axis=1)
    # Abundances (cell2location's estimated cell counts) do not sum to one and were drawn and
    # captioned as proportions on a 'proportion' colour bar once the gate let them through (hunt
    # 2026-09-30, u20b-viz-rest-4). They are named for what they are and never renormalised here.
    proportions = bool(np.allclose(row_sums, 1.0, atol=1e-2))
    quantity = "proportion" if proportions else "abundance"
    if not proportions:
        warnings.append(
            f"rows do not sum to one (median {float(np.median(row_sums)):.3f}); these are shown as "
            "stored abundances and are not renormalised here"
        )
    if (values < -1e-9).any():
        warnings.append(
            f"the table holds negative values, which a{'' if proportions else 'n'} {quantity} cannot be; they are drawn as stored"
        )
    if selected:
        values = frame[[selected]].to_numpy(dtype=float)
        frame = frame[[selected]]

    fmt = (figure_format or "png").lower()
    limitations: list[str] = [
        f"{quantity}s are an estimate from a deconvolution model, not a measured count of cells",
        section_note,
    ]
    if coords is not None:
        limitations.append("each mark is one spot, which on a sequencing-based platform is several cells")

    if kind == "dominant" and coords is not None:
        dominant = np.asarray(frame.columns)[values.argmax(axis=1)]
        fig, axes = _base.canvas(1, 1, panel_w=5.6, panel_h=4.8, dpi=dpi or None)
        _spatial.tissue_scatter(
            axes[0],
            coords,
            dominant,
            categorical=True,
            title=title or "Dominant cell type per spot",
            value_label="cell type",
            fmt=fmt,
        )
        fig.tight_layout()
        if len(set(dominant.tolist())) == 1:
            limitations.append("every spot has the same dominant type, so this map draws no distinction")
        share = float(np.max(values, axis=1).mean())
        if not proportions:
            # The share of the spot's own total; the raw maximum of an abundance is a count of cells.
            totals = np.where(row_sums > 0, row_sums, np.nan)
            share = float(np.nanmean(np.max(values, axis=1) / totals))
        limitations.append(
            f"the dominant type holds {share:.0%} of a spot on average, so most spots are mixtures rather than one type"
        )
        kind_out, subject = "spatial_map", title or "Dominant cell type"
        cache = {"coords": coords, "dominant": dominant.astype(str)}
    elif kind == "maps" and coords is not None:
        order = np.argsort(-values.mean(axis=0))[: max(1, int(top_n))]
        panels = {str(frame.columns[i]): values[:, i] for i in order}
        scales = _palette.PanelScales.build(panels, share=True, semantic="proportion")
        fig, notes = _spatial.tissue_panels(
            coords,
            panels,
            categorical=False,
            scales=scales,
            ncols=int(ncols),
            title=title or f"Cell-type {quantity}s",
            value_label=quantity,
            cmap=_palette.continuous("proportion"),
            fmt=fmt,
        )
        if notes.get("dropped"):
            warnings.append(f"{notes['dropped']} further cell types were not drawn; the grid is capped")
        limitations.append(scales.caption())
        kind_out, subject = "grid", title or f"Cell-type {quantity}s"
        cache = {"coords": coords, **{f"panel_{i}": values[:, i] for i in order}}
    else:
        means = frame.mean(axis=0).sort_values(ascending=False)[: max(1, int(top_n))]
        fig, axes = _base.canvas(1, 1, panel_w=5.6, panel_h=4.0, dpi=dpi or None)
        axes[0].barh(range(len(means))[::-1], means.to_numpy(), color="#0f7d76")
        axes[0].set_yticks(range(len(means))[::-1])
        axes[0].set_yticklabels([str(v) for v in means.index], fontsize=8)
        axes[0].set_xlabel(f"mean {quantity} across observations", fontsize=9)
        axes[0].set_title(title or "Mixture summary", fontsize=10)
        fig.tight_layout()
        kind_out, subject = "bar", title or "Mixture summary"
        cache = {"means": means.to_numpy()}

    caption = _spec.compose_caption(
        subject=subject + f" ({source_note})",
        expression=None,
        limitations=[t for t in limitations if t],
        claim="descriptive",
    )
    # The table is recorded under its own parameter name, so a redraw passes it back rather than
    # dropping an unknown 'result_path' and falling back to obsm under the same lineage, and by name
    # only, because the sidecar is served raw (hunt 2026-09-30, u20a-viz-pipelines-10, -26). Its
    # content reaches the stem, so two tables are two files.
    data_params = {"data_path": data_path, "obsm_key": obsm_key, "kind": kind, "library_id": library_id}
    stem = _spec.stem_for(
        plot_id,
        _with_selectors(prof["dataset"]["fingerprint"], proportions_csv=_file_fingerprint(proportions_csv)),
        data_params,
        slug=figure_id,
    )
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_deconvolution",
        kind=kind_out,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=[t for t in limitations if t],
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "read_from": source_note,
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "proportions_csv": Path(proportions_csv).name if proportions_csv else "",
            "top_n": top_n,
            "ncols": ncols,
            "figure_format": fmt,
        },
        param_origin={"obsm_key": "caller" if obsm_key else "the first proportions matrix present"},
        expression=None,
        cache=cache,
        findings=[("n_cell_types", int(frame.shape[1]), "cell types in the deconvolution")],
    )


@_layer_errors_refuse("markers.dotplot")
def plot_marker_expression(
    data_path: str,
    *,
    genes: str = "",
    groupby: str = "",
    output_dir: str = "",
    kind: str = "dotplot",
    layer: str = "",
    use_raw: bool = False,
    normalize: str = "auto",
    standardize: bool = False,
    top_n: int = 5,
    split_by: str = "",
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Marker expression across groups: a dot plot, violins, a heatmap, or group composition.

    The genes may be named, or taken from a stored ranking -- and when they come from a ranking
    the caption says so, because "the top five markers of each cluster" is a statement about a
    computation somebody already ran, not about this figure.

    ``composition`` draws no expression at all: it is the make-up of one categorical column across
    another, and it is captioned descriptive with no vocabulary of significance, because a stacked
    bar of proportions is not a test.
    """
    plot_ids = {
        "dotplot": "markers.dotplot",
        "violin": "markers.violin",
        "heatmap": "markers.heatmap",
        "composition": "composition.stacked",
    }
    want = plot_ids.get(kind)
    if want is None:
        return _refusal(
            "invalid_request",
            f"{kind!r} is not a marker view; choose one of {sorted(plot_ids)}",
            plot_id="markers.dotplot",
        )
    prepared, refusal = _prepare(data_path, want, output_dir)
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    roles = prof["obs_roles"]
    group_key = groupby or roles.get("cell_type") or roles.get("cluster") or ""
    if not group_key:
        candidates = roles.get("categorical") or []
        group_key = candidates[0] if candidates else ""
    if not group_key:
        return _refusal(
            "prerequisite_not_met",
            "this dataset has no categorical column to group by",
            plot_id=want,
            missing=["has_categorical_obs"],
        )
    if group_key not in (roles.get("categorical") or []):
        return _refusal(
            "invalid_request",
            f"obs['{group_key}'] is not a categorical column; the categorical ones are {roles.get('categorical')}",
            plot_id=want,
        )

    import numpy as np
    import pandas as pd

    fmt = (figure_format or "png").lower()
    gene_origin = "caller"
    wanted: list[str] = [g.strip() for g in (genes or "").split(",") if g.strip()]

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        groups_raw, _ = _layers.obs_values(adata, group_key, max_levels=_palette.max_categories())
        groups = pd.Series(np.asarray(groups_raw).astype(object)).astype(str)

        if kind == "composition":
            second = split_by or ""
            if not second:
                others = [c for c in (roles.get("categorical") or []) if c != group_key]
                second = others[0] if others else ""
            if not second:
                return _refusal(
                    "prerequisite_not_met",
                    "a composition needs a second categorical column to break the groups down by",
                    plot_id=want,
                    missing=["has_categorical_obs"],
                )
            inner_raw, _ = _layers.obs_values(adata, second, max_levels=_palette.max_categories())
            inner = pd.Series(np.asarray(inner_raw).astype(object)).astype(str)
            table = pd.crosstab(groups, inner)
            frame = table.div(table.sum(axis=1), axis=0)
            slot = None
        else:
            if not wanted:
                detail = prof["uns_analyses"]["rank_genes_groups_detail"]
                stored = detail.get("groups") or []
                if not stored:
                    # Second source, and it is a READ, not a computation: scanpy writes
                    # var['highly_variable'] when somebody selected genes, and re-using that
                    # selection runs nothing new. Computing a variance here to pick genes would be
                    # this family running an analysis it never declared.
                    var_frame = getattr(adata, "var", None)
                    flagged = []
                    if var_frame is not None and "highly_variable" in var_frame:
                        subset = var_frame[var_frame["highly_variable"].astype(bool)]
                        if "highly_variable_rank" in subset:
                            subset = subset.sort_values("highly_variable_rank")
                        flagged = [str(n) for n in subset.index[: max(1, int(top_n) * 3)]]
                    if not flagged:
                        return _refusal(
                            "prerequisite_not_met",
                            "no genes were named, this object holds no stored ranking to take them "
                            "from, and no highly-variable selection is flagged in var",
                            plot_id=want,
                            missing=["has_de_result"],
                            instead=["spatial.expression", "embedding.scatter"],
                        )
                    wanted = flagged
                    gene_origin = (
                        "the object's own var['highly_variable'] flag, which is a selection somebody "
                        "made earlier and not a ranking of markers for these groups"
                    )
                block = adata.uns["rank_genes_groups"] if stored else None
                picked: list[str] = []
                for level in stored if block is not None else []:
                    for name in np.asarray(block["names"][level])[: int(top_n)]:
                        if str(name) not in picked:
                            picked.append(str(name))
                if picked:
                    wanted = picked
                    gene_origin = (
                        f"the top {int(top_n)} of each group in the stored uns['rank_genes_groups'] "
                        f"({(detail.get('params') or {}).get('method', 'method not recorded')})"
                    )
            slot = _layers.resolve_expression_slot(
                adata,
                layer=layer,
                use_raw=use_raw,
                x_has_negative=bool(prof["matrix"].get("has_negative")),
            )
            found, missing, ambiguous = _layers.gene_values(adata, wanted, slot)
            if ambiguous:
                warnings.append(
                    f"{len(ambiguous)} symbol(s) match more than one row and were skipped: {', '.join(ambiguous[:6])}"
                )
            if missing:
                warnings.append(f"{len(missing)} gene(s) are not in this object: {', '.join(missing[:8])}")
            if not found:
                return _refusal(
                    "invalid_request",
                    f"none of the requested genes are in this object; {len(missing)} were not found",
                    plot_id=want,
                    instead=["qc.overview"],
                )
            raw = pd.DataFrame({g: np.asarray(v, dtype=float).ravel() for g, v in found.items()})
            integral = _slot_integral(slot, prof, found.values())
            shown, transform = _layers.normalize_for_display(raw.to_numpy(), mode=normalize, integral=integral)
            frame = pd.DataFrame(np.asarray(shown), columns=list(raw.columns))
            frame["__group__"] = groups.to_numpy()

    limitations: list[str] = []
    scale_note = ""
    cmap = _palette.continuous("expression", color_map)

    if kind == "composition":
        fig, axes = _base.canvas(1, 1, panel_w=max(5.0, 0.5 * len(frame) + 3.0), panel_h=4.4, dpi=dpi or None)
        ax = axes[0]
        levels = list(frame.columns)
        colours, pooled = _palette.categorical(np.asarray(levels, dtype=object), uniq=levels)
        bottom = np.zeros(len(frame))
        for level in levels:
            values = frame[level].to_numpy()
            ax.bar(range(len(frame)), values, bottom=bottom, label=str(level), color=colours.get(level))
            bottom = bottom + values
        ax.set_xticks(range(len(frame)))
        ax.set_xticklabels([str(i) for i in frame.index], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("fraction of the group", fontsize=9)
        ax.set_ylim(0, 1)
        if len(levels) <= 30:
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        if pooled:
            warnings.append(f"{len(pooled)} rare level(s) were pooled into one colour")
        subject = title or f"Composition of {group_key} by {second}"
        plot_kind = "bar"
        limitations.append(
            "proportions within each group, drawn from counts of cells or spots; no test was run "
            "and no difference between groups is claimed"
        )
        cache = {str(c): frame[c].to_numpy() for c in frame.columns}
        expression = None
    else:
        grouped = frame.groupby("__group__", observed=True)
        means = grouped.mean()
        order = list(means.index)
        gene_order = [g for g in wanted if g in means.columns]
        means = means.reindex(index=order, columns=gene_order)
        if standardize:
            spread = means.std(axis=0).replace(0.0, np.nan)
            means = (means - means.mean(axis=0)).div(spread).fillna(0.0)
            scale_note = "each gene is standardised across groups, so colour is relative and not a level"
            cmap = _palette.continuous(_SIGNED, color_map)

        if kind == "dotplot":
            fraction = grouped.apply(lambda d: (d[gene_order] > 0).mean(), include_groups=False).reindex(order)
            fig, axes = _base.canvas(
                1,
                1,
                panel_w=max(5.0, 0.42 * len(gene_order) + 3.0),
                panel_h=max(3.2, 0.34 * len(order) + 1.8),
                dpi=dpi or None,
            )
            ax = axes[0]
            xs, ys, sizes, colours_v = [], [], [], []
            for yi, level in enumerate(order):
                for xi, gene in enumerate(gene_order):
                    xs.append(xi)
                    ys.append(yi)
                    sizes.append(float(fraction.loc[level, gene]) * 180.0 + 4.0)
                    colours_v.append(float(means.loc[level, gene]))
            vmin, vmax, clip_note = _palette.limits(
                np.asarray(colours_v), semantic=_SIGNED if standardize else "expression"
            )
            art = ax.scatter(xs, ys, s=sizes, c=colours_v, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
            bar = fig.colorbar(art, ax=ax, shrink=0.7, pad=0.02)
            bar.set_label("mean expression in group", fontsize=9)
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels(gene_order, rotation=90, fontsize=8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([str(o) for o in order], fontsize=8)
            ax.set_xlim(-0.6, len(gene_order) - 0.4)
            ax.set_ylim(-0.6, len(order) - 0.4)
            plot_kind = "heatmap"
            limitations.append("dot area is the fraction of the group with a non-zero value; colour is the group mean")
            if clip_note:
                limitations.append(clip_note)
        elif kind == "heatmap":
            fig, axes = _base.canvas(
                1,
                1,
                panel_w=max(5.0, 0.36 * len(gene_order) + 3.0),
                panel_h=max(3.0, 0.32 * len(order) + 1.6),
                dpi=dpi or None,
            )
            ax = axes[0]
            matrix = means.to_numpy(dtype=float)
            vmin, vmax, clip_note = _palette.limits(matrix, semantic=_SIGNED if standardize else "expression")
            art = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            bar = fig.colorbar(art, ax=ax, shrink=0.7, pad=0.02)
            bar.set_label("mean expression in group", fontsize=9)
            ax.set_xticks(range(len(gene_order)))
            ax.set_xticklabels(gene_order, rotation=90, fontsize=8)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels([str(o) for o in order], fontsize=8)
            plot_kind = "heatmap"
            if clip_note:
                limitations.append(clip_note)
        else:  # violin
            shown_genes = gene_order[: _base.max_panels()]
            rows_n, cols_n, shown_n = _base.panel_grid(len(shown_genes), 3)
            fig, axes = _base.canvas(rows_n, cols_n, panel_w=4.2, panel_h=3.0, dpi=dpi or None)
            for i in range(shown_n):
                gene = shown_genes[i]
                series = [frame.loc[frame["__group__"] == level, gene].to_numpy() for level in order]
                series = [s if s.size else np.zeros(1) for s in series]
                axes[i].violinplot(series, showmeans=False, showextrema=False, widths=0.85)
                axes[i].set_xticks(range(1, len(order) + 1))
                axes[i].set_xticklabels([str(o) for o in order], rotation=90, fontsize=7)
                axes[i].set_title(gene, fontsize=10)
                axes[i].set_ylabel("expression", fontsize=8)
            for j in range(shown_n, len(axes)):
                _base.blank(axes[j])
            if len(gene_order) > shown_n:
                warnings.append(
                    f"{len(gene_order) - shown_n} further gene(s) were not drawn; the panel cap is {shown_n}"
                )
            plot_kind = "boxplot"
            limitations.append("each violin is the distribution over cells or spots in that group, not over replicates")

        subject = title or f"{len(gene_order)} gene(s) across {group_key}"
        expression = {
            "slot": slot["slot"],
            "key": slot["key"],
            "transform": transform,
            "integral": integral,
            "why": slot["why"],
        }
        cache = {"means": means.to_numpy(dtype=float)}
        limitations.append(
            "each observation is one cell or spot, not a biological replicate; this describes groups within one sample"
        )
        if gene_origin != "caller":
            limitations.append(f"the genes shown were taken from {gene_origin}")

    fig.tight_layout()
    caption = _spec.compose_caption(
        subject=subject,
        expression=expression,
        scale_note=scale_note,
        limitations=limitations,
        claim="descriptive",
    )
    data_params = {
        "data_path": data_path,
        "groupby": group_key,
        "kind": kind,
        "genes": ",".join(wanted),
        "layer": layer,
        "use_raw": use_raw,
        "normalize": normalize,
    }
    # split_by and standardize change the values drawn and are in no DATA_PARAMS set, so the
    # composition by sample and by condition shared one file (hunt 2026-09-30, u20a-viz-pipelines-5).
    stem = _spec.stem_for(
        want,
        _with_selectors(
            prof["dataset"]["fingerprint"],
            split_by=second if kind == "composition" else "",
            standardize=bool(standardize) and kind != "composition",
        ),
        data_params,
        slug=figure_id,
    )
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=want,
        function="plot_marker_expression",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "read_from": (
                f"obs['{group_key}']"
                if kind == "composition"
                else f"obs['{group_key}'] and "
                + (f"layers['{slot['key']}']" if slot["slot"] == "layers" else slot["slot"])
            ),
        },
        params={
            **data_params,
            "data_path": Path(data_path).name,
            "standardize": standardize,
            "split_by": second if kind == "composition" else split_by,
            "top_n": top_n,
            "figure_format": fmt,
        },
        param_origin={
            "groupby": "caller" if groupby else "the dataset's cell-type or cluster column",
            "genes": gene_origin,
        },
        expression=expression,
        cache=cache,
    )


def _read_table(path: str, what: str) -> tuple[Any, dict[str, Any] | None]:
    """Read a result table another tool wrote, sniffing the delimiter rather than assuming one.

    The tools in this platform write comma and tab separated files both, and a wrong guess does not
    raise -- it produces a single-column frame whose one column name is the whole header line, and
    every later lookup then reports the column as missing.
    """
    import pandas as pd

    p = Path(path)
    if not p.is_file():
        return None, _refusal(
            "invalid_request",
            f"no {what} at {p.name!r}; pass the path to the table the analysis tool wrote",
            plot_id="pathway.enrichment",
        )
    try:
        # The delimiter is chosen from the header among the ones tools write. sep=None (csv.Sniffer) split a
        # one-column header on one of its own letters -- 'TGFb' became 'TGF' plus an empty column -- so a
        # one-pathway table was drawn under a corrupted name or refused (review of u20a-viz-pipelines-12,
        # 2026-10-01).
        with p.open(encoding="utf-8", errors="replace") as fh:
            header = fh.readline()
        sep = max(("\t", ",", ";"), key=header.count) if any(d in header for d in ("\t", ",", ";")) else ","
        frame = pd.read_csv(p, sep=sep)
    except Exception as exc:
        return None, _refusal("unreadable", f"{p.name} could not be parsed: {exc}", plot_id="pathway.enrichment")
    if frame.empty:
        return None, _refusal("invalid_request", f"{p.name} has no rows", plot_id="pathway.enrichment")
    return frame, None


def _numeric_column(frame: Any, column: str) -> bool:
    """Whether a column holds at least one number -- a header names a score; only its cells say it is one."""
    import pandas as pd

    return bool(pd.to_numeric(frame[column], errors="coerce").notna().any())


def _melt_square(frame: Any, value: str = "score") -> Any:
    """An index-by-columns matrix as one ``(source, target, value)`` row per cell, empty cells skipped.

    deeplinc and squidpy write the sender on the rows and the receiver on the columns, the row labels in a first
    column whose header is empty (pandas reads it as ``Unnamed: 0``). spaotsc's signaling matrix has no row labels
    at all, so its rows are named by position, as its numbered header names its columns.
    """
    import pandas as pd

    first = str(frame.columns[0]).strip().lower()
    if first in ("", "unnamed: 0"):
        frame = frame.set_index(frame.columns[0])
    rows = []
    for source, row in frame.iterrows():
        for target, cell in row.items():
            number = pd.to_numeric(cell, errors="coerce")
            if pd.notna(number):
                rows.append((str(source), str(target), float(number)))
    return pd.DataFrame(rows, columns=["source", "target", value])


@_layer_errors_refuse("pathway.enrichment")
def plot_pathway_results(
    results_path: str = "",
    *,
    data_path: str = "",
    output_dir: str = "",
    kind: str = "enrichment",
    collection: str = "",
    method: str = "",
    pathways: str = "",
    obsm_key: str = "",
    top_n: int = 15,
    fdr_threshold: float = 0.05,
    library_id: str = "",
    ncols: int = 3,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a pathway result another tool computed: enriched terms, or per-spot activity on tissue.

    Nothing here runs an enrichment. The terms, their effect sizes and their corrected p-values are
    read from the table the analysis tool wrote, and a table missing the column that carries
    significance is refused rather than ranked by whatever number is left.
    """
    if kind not in ("enrichment", "activity"):
        return _refusal(
            "invalid_request",
            f"{kind!r} is not a pathway view; choose 'enrichment' or 'activity'",
            plot_id="pathway.enrichment",
        )

    import numpy as np

    fmt = (figure_format or "png").lower()
    warnings: list[str] = []

    if kind == "enrichment":
        frame, refusal = _read_table(results_path, "enrichment table")
        if refusal:
            return refusal
        columns = {c.lower().strip(): c for c in frame.columns}
        term_col = next((columns[c] for c in ("term", "name", "pathway", "description") if c in columns), "")
        # FDR first, and in THIS order: a raw p-value ranked as though corrected overstates every
        # term on the list. A table with only a raw p-value is drawn, and the caption says which
        # it is rather than calling it an FDR.
        sig_col, sig_label = "", ""
        for candidate, label in (
            ("fdr", "FDR"),
            ("fdr p-value", "FDR"),
            ("padj", "adjusted p"),
            ("adj_pval", "adjusted p"),
            ("qvalue", "q-value"),
            ("fdr_bh", "FDR"),
            ("pvalue", "raw p"),
            ("pval", "raw p"),
            ("p_value", "raw p"),
        ):
            if candidate in columns:
                sig_col, sig_label = columns[candidate], label
                break
        if not term_col or not sig_col:
            missing = [n for n, ok in (("a term column", term_col), ("a significance column", sig_col)) if not ok]
            return _refusal(
                "prerequisite_not_met",
                f"{Path(results_path).name} has no {' and no '.join(missing)}; its columns are {list(frame.columns)}",
                plot_id="pathway.enrichment",
                missing=["has_enrichment_table"],
                instead=["markers.dotplot"],
            )
        effect_col = next(
            (
                columns[c]
                for c in ("effect", "nes", "odds ratio", "odds_ratio", "combined score", "score")
                if c in columns
            ),
            "",
        )
        size_col = next(
            (columns[c] for c in ("n_overlap", "overlap", "intersection_size", "count") if c in columns), ""
        )

        if collection and "collection" in columns:
            frame = frame[frame[columns["collection"]].astype(str) == collection]
        if method and "method" in columns:
            frame = frame[frame[columns["method"]].astype(str) == method]
        if frame.empty:
            return _refusal(
                "invalid_request",
                "no rows are left after the collection and method filters",
                plot_id="pathway.enrichment",
            )

        significance = frame[sig_col].astype(float)
        kept = frame.assign(__sig__=significance).sort_values("__sig__").head(int(top_n))
        n_pass = int((significance < fdr_threshold).sum())

        floor = float(np.nextafter(0, 1))
        y = -np.log10(np.clip(kept["__sig__"].to_numpy(dtype=float), floor, None))
        clamped = int((kept["__sig__"].to_numpy(dtype=float) <= 0).sum())
        if clamped:
            warnings.append(
                f"{clamped} term(s) report a {sig_label} of exactly zero, which is the floor of the "
                "correction rather than a measurement; they are drawn at the extreme of the axis"
            )
        x = kept[effect_col].astype(float).to_numpy() if effect_col else y
        sizes = (
            (
                kept[size_col].astype(float).to_numpy() / max(1.0, float(kept[size_col].astype(float).max())) * 170.0
                + 12.0
            )
            if size_col
            else np.full(len(kept), 60.0)
        )
        labels = [str(v) for v in kept[term_col]]

        fig, axes = _base.canvas(1, 1, panel_w=7.0, panel_h=max(3.0, 0.32 * len(kept) + 1.6), dpi=dpi or None)
        ax = axes[0]
        art = ax.scatter(
            x, range(len(kept))[::-1], s=sizes, c=y, cmap=_palette.continuous("expression", color_map), linewidths=0
        )
        bar = fig.colorbar(art, ax=ax, shrink=0.7, pad=0.02)
        bar.set_label(f"-log10 {sig_label}", fontsize=9)
        ax.set_yticks(range(len(kept))[::-1])
        ax.set_yticklabels([lbl[:58] for lbl in labels], fontsize=8)
        ax.set_xlabel(effect_col if effect_col else f"-log10 {sig_label}", fontsize=9)
        plot_kind = "scatter"
        subject = title or f"Top {len(kept)} enriched terms"
        limitations = [
            f"ranked by {sig_label} from {Path(results_path).name}; this figure runs no enrichment of its own",
            f"{n_pass} of {len(frame)} term(s) in the table are below {fdr_threshold}",
        ]
        if not effect_col:
            limitations.append("the table carries no effect size, so the horizontal axis repeats the significance")
        if size_col:
            limitations.append(f"dot area is {size_col}, the number of genes shared with the set")
        claim = "tested" if sig_label != "raw p" else "descriptive"
        if sig_label == "raw p":
            limitations.append(
                "the table carries only an uncorrected p-value, so these terms are not multiple-testing corrected"
            )
        source = {"name": Path(results_path).name, "read_from": f"{term_col}, {sig_col}"}
        cache = {"significance": kept["__sig__"].to_numpy(dtype=float), "effect": x}
        expression = None
        # The table's content and the filters reach the stem: with an empty fingerprint the up- and
        # down-regulated tables drew to one file (hunt 2026-09-30, u20a-viz-pipelines-4).
        fingerprint = _with_selectors(_file_fingerprint(results_path), kind=kind, collection=collection, method=method)
        data_params = {"results_path": results_path, "kind": kind, "collection": collection, "top_n": top_n}
        findings = [("n_terms_below_threshold", n_pass, f"terms below {fdr_threshold}")]
    else:
        if not data_path:
            return _refusal(
                "invalid_request",
                "an activity map needs the spatial object the scores belong to; pass data_path",
                plot_id="pathway.activity_map",
            )
        prepared, refusal = _prepare(data_path, "pathway.activity_map", output_dir)
        if refusal:
            return refusal
        prof, out_dir_p, figures_dir_p = prepared
        warnings.extend(prof.get("warnings") or [])
        wanted = [p.strip() for p in (pathways or "").split(",") if p.strip()]

        import pandas as pd

        with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
            coords, coord_note, section = _coords_and_mask(adata, library_id)
            present = None
            positional_note = ""
            if results_path:
                frame, refusal = _read_table(results_path, "activity table")
                if refusal:
                    return refusal
                first = frame.columns[0]
                # The first column is the index when it is unnamed, holds text, or names more of the
                # observations than the row labels do. A table whose integer cell ids sit in a NAMED
                # column ('cell' = 1000, 1001, ...) was refused as naming none of them, by a refusal
                # whose own fix asked for exactly that table (hunt 2026-09-30, the review of the fix
                # beside u20a-viz-pipelines-12).
                known = set(np.asarray(adata.obs_names).astype(str).tolist())

                def _naming(values: Any) -> int:
                    return sum(1 for v in np.asarray(values).astype(str).tolist() if v in known)

                # _read_table sets no index, so frame.index is always the row numbers 0..m-1. On an
                # object whose obs_names are '0'..'n-1' those numbers name every observation and won
                # the tie against a 'cell' column holding the same ids in another order: every score
                # was placed by row position and 'cell' was drawn as a pathway, with status ok and no
                # warning (hunt 2026-09-30, u20a-viz-pipelines-12 residual). A named integer column
                # with one distinct value per row is an id column -- a score is a float -- so it wins
                # a tie, it is never drawn, and placement by row position is said, not silent.
                n_by_row_number = _naming(frame.index)
                n_by_first = _naming(frame[first])
                first_is_id = bool(pd.api.types.is_integer_dtype(frame[first]) and frame[first].is_unique)
                by_row_number = not (
                    str(first).strip() == ""
                    or str(first).startswith("Unnamed")
                    or frame[first].dtype == object
                    or n_by_first > n_by_row_number
                    or (first_is_id and n_by_first >= n_by_row_number)
                )
                if not by_row_number:
                    if first_is_id and n_by_row_number and frame[first].tolist() != list(range(len(frame))):
                        warnings.append(
                            f"the activity table's rows are matched by its {str(first)!r} column, read as "
                            "observation ids; its row numbers name these observations too, in another order"
                        )
                    frame = frame.set_index(first)
                elif first_is_id:
                    frame = frame.drop(columns=[first])
                    warnings.append(
                        f"the activity table's {str(first)!r} column holds one distinct integer per row, so it is "
                        f"read as an id column and not drawn; it names {n_by_first} of the object's "
                        f"{len(known)} observations and the row numbers name {n_by_row_number}"
                    )
                if by_row_number:
                    # No column names the observations, so the rows can only be placed by POSITION: row i
                    # on the object's i-th observation, which needs the same row count. Matching the row
                    # numbers to obs_names by name put a table written in the object's own order on the
                    # wrong spots whenever obs_names looked like integers but were not '0'..'n-1' in order
                    # (a filtered, R-style or reordered object), and refused it on a barcode-named one
                    # (review of the u20a-viz-pipelines-12 repair, 2026-10-01).
                    if len(frame) != adata.n_obs:
                        return _refusal(
                            "invalid_request",
                            f"the activity table has no column of observation names and {len(frame)} rows for "
                            f"the object's {adata.n_obs} observations, so its rows cannot be placed",
                            plot_id="pathway.activity_map",
                            fix="pass the table written for this object, with the observation names as its first column",
                        )
                    frame = frame.copy()
                    frame.index = np.asarray(adata.obs_names).astype(str)
                # Matched by observation name; only the row count was checked, so a table in another
                # row order painted every score on another spot (hunt 2026-09-30, same defect as
                # u20a-viz-pipelines-12).
                frame, present, refusal = _align_rows_to_obs(
                    frame, adata.obs_names, "activity table", "pathway.activity_map", warnings
                )
                if refusal:
                    return refusal
                if by_row_number:
                    # Said every time: right only if the tool kept the object's row order (hunt 2026-09-30,
                    # u20a-viz-pipelines-12 residual).
                    positional_note = (
                        "the activity table has no column of observation names, so its scores are placed "
                        "by row position (row i on the object's i-th observation); this is right only if the "
                        "tool kept the object's row order"
                    )
                    warnings.append(positional_note)
                scores = frame.select_dtypes("number")
            else:
                key = obsm_key or next(
                    (e["key"] for e in prof["obsm"] if "estimate" in e["key"] or "activity" in e["key"]), ""
                )
                if not key:
                    return _refusal(
                        "prerequisite_not_met",
                        "no activity matrix was named and none is stored in obsm",
                        plot_id="pathway.activity_map",
                        missing=["has_activity_scores"],
                        instead=["spatial.expression"],
                    )
                matrix, label = _layers.obsm_values(adata, key)
                array = np.asarray(matrix, dtype=float)
                # The matrix's own column names, so a pathway can be asked for by name and the
                # stem can hash which were drawn; the bare array numbered them, and joining those
                # numbers into the stem raised (found while fixing u20a-viz-pipelines-1).
                try:
                    names = [str(c) for c in adata.obsm[str(key).partition(":")[0]].columns]
                except Exception:
                    names = []
                if array.ndim == 1:
                    scores = pd.DataFrame({str(label): array})
                else:
                    if len(names) != array.shape[1]:
                        names = [f"{label}[{i}]" for i in range(array.shape[1])]
                    scores = pd.DataFrame(array, columns=names)

        keep = np.ones(len(scores), dtype=bool) if present is None else np.asarray(present, dtype=bool)
        if section is not None:
            keep = keep & section
        scores = scores[keep]
        coords = np.asarray(coords, dtype=float)[keep]
        if section is not None:
            coord_note = f"only section {library_id!r} is drawn ({int(keep.sum()):,} of {len(keep):,} observations)"
        available = list(scores.columns)
        chosen = [p for p in wanted if p in available] or available[: int(ncols) * 2]
        unknown = [p for p in wanted if p not in available]
        if unknown:
            warnings.append(f"{len(unknown)} pathway(s) are not in the score matrix: {', '.join(unknown[:6])}")
        chosen = chosen[: _base.max_panels()]

        panels = {str(c): scores[c].to_numpy(dtype=float) for c in chosen}
        scales = _palette.PanelScales.build(panels, share=False, semantic=_SIGNED)
        fig, notes = _spatial.tissue_panels(
            coords,
            panels,
            categorical=False,
            scales=scales,
            ncols=ncols,
            title=title or "Pathway activity",
            cmap=_palette.continuous(_SIGNED, color_map),
            fmt=fmt,
        )
        plot_kind = "spatial_map"
        subject = title or f"{len(panels)} pathway(s) on the tissue"
        limitations = [
            "activity is an inferred score from a gene set, not a measurement of pathway flux",
            coord_note,
            positional_note,
        ]
        limitations = [item for item in limitations if item]
        claim = "descriptive"
        source = {
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "read_from": Path(results_path).name if results_path else f"obsm['{obsm_key}']",
        }
        cache = dict(panels)
        expression = None
        # pathways is in no DATA_PARAMS set and the score source was not hashed at all, so two
        # activity maps of different pathways or tables shared one file (hunt 2026-09-30,
        # u20a-viz-pipelines-5).
        fingerprint = _with_selectors(
            prof["dataset"]["fingerprint"],
            pathways=",".join(str(c) for c in chosen),
            scores=_file_fingerprint(results_path) if results_path else "",
        )
        out_dir, figures_dir = out_dir_p, figures_dir_p
        data_params = {
            "data_path": data_path,
            "kind": kind,
            "pathways": ",".join(str(c) for c in chosen),
            "results_path": results_path,
            "obsm_key": obsm_key,
            "library_id": library_id,
        }
        findings = [("n_pathways_drawn", len(panels), "pathways mapped")]
        fig.tight_layout()

    if kind == "enrichment":
        out_dir = Path(output_dir or ".").resolve()
        figures_dir = _manifest_io.figures_dir_for(out_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()

    caption = _spec.compose_caption(
        subject=subject,
        expression=expression,
        scale_note="" if kind == "enrichment" else scales.caption(),
        limitations=limitations,
        claim=claim,
    )
    plot_id = "pathway.enrichment" if kind == "enrichment" else "pathway.activity_map"
    stem = _spec.stem_for(plot_id, fingerprint, data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_pathway_results",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source=source,
        params={
            **data_params,
            "fdr_threshold": fdr_threshold,
            "method": method,
            "figure_format": fmt,
        },
        param_origin={"collection": "caller" if collection else "every collection in the table"},
        expression=expression,
        cache=cache,
        findings=findings,
    )


@_layer_errors_refuse("organisation.graph")
def plot_spatial_statistics(
    data_path: str = "",
    *,
    results_path: str = "",
    output_dir: str = "",
    kind: str = "graph",
    obs_key: str = "",
    n_neighbors: int = 6,
    top_n: int = 20,
    library_id: str = "",
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw how the tissue is organised: the neighbour graph, or a statistic another tool computed.

    Only the neighbour graph is built here, from the coordinates, and it is a drawing of adjacency
    rather than a claim about it. Autocorrelation, neighbourhood enrichment and co-occurrence are
    read from what the spatial-statistics tool wrote -- this module does not recompute a statistic
    and then present it beside the number the analysis reported.
    """
    plot_ids = {
        "graph": "organisation.graph",
        "autocorrelation": "organisation.autocorrelation",
        "neighborhood": "organisation.neighborhood",
        "cooccurrence": "organisation.cooccurrence",
    }
    want = plot_ids.get(kind)
    if want is None:
        return _refusal(
            "invalid_request",
            f"{kind!r} is not an organisation view; choose one of {sorted(plot_ids)}",
            plot_id="organisation.graph",
        )

    import numpy as np

    fmt = (figure_format or "png").lower()
    warnings: list[str] = []
    fingerprint = ""
    cache: dict[str, Any] = {}
    findings: list[tuple[str, Any, str]] = []

    if kind in ("autocorrelation", "cooccurrence"):
        frame, refusal = _read_table(results_path, "spatial-statistics table")
        if refusal:
            refusal["plot_id"] = want
            return refusal
        columns = {c.lower().strip(): c for c in frame.columns}
        out_dir = Path(output_dir or ".").resolve()
        figures_dir = _manifest_io.figures_dir_for(out_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        if kind == "autocorrelation":
            stat_col = next(
                (
                    columns[c]
                    for c in ("i", "morani", "moran_i", "moransi", "c", "gearyc", "statistic", "score")
                    if c in columns
                ),
                "",
            )
            name_col = next((columns[c] for c in ("gene", "name", "feature", "term") if c in columns), frame.columns[0])
            if not stat_col:
                return _refusal(
                    "prerequisite_not_met",
                    f"{Path(results_path).name} carries no autocorrelation statistic; its columns are {list(frame.columns)}",
                    plot_id=want,
                    missing=["has_autocorrelation"],
                    instead=["spatial.expression"],
                )
            # Geary's C runs the other way: below one is positive autocorrelation, so ranking it
            # highest first showed the LEAST spatially structured genes as the top (hunt
            # 2026-09-30, u20a-viz-pipelines-15). The squidpy worker itself sorts geary ascending.
            is_geary = str(stat_col).lower().strip() in ("c", "gearyc", "geary_c", "gearys_c")
            top = frame.reindex(frame[stat_col].astype(float).sort_values(ascending=is_geary).index).head(int(top_n))
            fig, axes = _base.canvas(1, 1, panel_w=6.2, panel_h=max(3.0, 0.3 * len(top) + 1.6), dpi=dpi or None)
            axes[0].barh(range(len(top))[::-1], top[stat_col].astype(float).to_numpy(), color="#0f7d76")
            axes[0].set_yticks(range(len(top))[::-1])
            axes[0].set_yticklabels([str(v)[:40] for v in top[name_col]], fontsize=8)
            axes[0].set_xlabel(stat_col, fontsize=9)
            plot_kind = "bar"
            subject = title or (
                f"Top {len(top)} by {stat_col} (lowest first)" if is_geary else f"Top {len(top)} by {stat_col}"
            )
            limitations = [
                f"{stat_col} as stored in {Path(results_path).name}; no statistic was recomputed here",
                (
                    "Geary's C is ranked lowest first, because a C below one is positive spatial "
                    "autocorrelation and a C above one is dispersion"
                    if is_geary
                    else "a high autocorrelation says values are spatially structured"
                )
                + ", not that the gene is biologically important",
            ]
            cache = {stat_col: top[stat_col].astype(float).to_numpy()}
            findings = [("n_features_ranked", int(len(frame)), "features in the table")]
        else:
            distance_col = next(
                (columns[c] for c in ("distance", "interval", "radius", "bin") if c in columns), frame.columns[0]
            )
            value_cols = [c for c in frame.columns if c != distance_col and frame[c].dtype.kind in "fiu"][: int(top_n)]
            if not value_cols:
                return _refusal(
                    "prerequisite_not_met",
                    f"{Path(results_path).name} has no numeric co-occurrence columns beside {distance_col!r}",
                    plot_id=want,
                    missing=["has_cooccurrence"],
                )
            fig, axes = _base.canvas(1, 1, panel_w=6.4, panel_h=4.2, dpi=dpi or None)
            for column in value_cols:
                axes[0].plot(frame[distance_col].to_numpy(), frame[column].to_numpy(), linewidth=1.2, label=str(column))
            axes[0].set_xlabel(str(distance_col), fontsize=9)
            axes[0].set_ylabel("co-occurrence ratio", fontsize=9)
            if len(value_cols) <= 20:
                axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
            plot_kind = "line"
            subject = title or "Co-occurrence with distance"
            limitations = [
                f"read from {Path(results_path).name}; the ratio is relative to the overall frequency of each label",
                "a ratio above one at short distance means the pair is found together more often than chance, which is a description of arrangement and not of interaction",
            ]
            cache = {str(c): frame[c].to_numpy(dtype=float) for c in [distance_col, *value_cols]}
            findings = [("n_pairs_drawn", len(value_cols), "label pairs")]
        source = {"name": Path(results_path).name, "read_from": ", ".join(list(frame.columns)[:4])}
        expression = None
        data_params = {"results_path": results_path, "kind": kind, "top_n": top_n}
        # The table's content reaches the stem: with an empty fingerprint every autocorrelation
        # table drew to one file (hunt 2026-09-30, u20a-viz-pipelines-4).
        fingerprint = _file_fingerprint(results_path)
    else:
        prepared, refusal = _prepare(data_path, want, output_dir)
        if refusal:
            return refusal
        prof, out_dir, figures_dir = prepared
        warnings.extend(prof.get("warnings") or [])
        fingerprint = prof["dataset"]["fingerprint"]

        if kind == "neighborhood":
            with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
                block = None
                for candidate in (f"{obs_key}_nhood_enrichment" if obs_key else "", "nhood_enrichment"):
                    if candidate and candidate in adata.uns:
                        block = adata.uns[candidate]
                        break
                if block is None:
                    block = next(
                        (adata.uns[k] for k in adata.uns if "nhood" in str(k).lower() or "enrich" in str(k).lower()),
                        None,
                    )
                if block is None:
                    return _refusal(
                        "prerequisite_not_met",
                        "this object holds no neighbourhood-enrichment result; the spatial-statistics tool writes one",
                        plot_id=want,
                        missing=["has_nhood_enrichment"],
                        instead=["spatial.annotation", "organisation.graph"],
                    )
                matrix = np.asarray(
                    block["zscore"] if isinstance(block, dict) and "zscore" in block else block, dtype=float
                )
                key = obs_key or (prof["obs_roles"].get("cell_type") or prof["obs_roles"].get("cluster") or "")
                labels = []
                if obs_key and obs_key not in adata.obs.columns:
                    return _refusal(
                        "invalid_request",
                        f"this object has no obs column named {obs_key!r} to label the matrix with",
                        plot_id=want,
                        fix="name the column the neighbourhood enrichment was computed over",
                    )
                if key and key in adata.obs.columns:
                    # squidpy orders the matrix by the column's categories (a leiden column
                    # natsorts: 0, 1, 2, ..., 11), and a string sort put '10' at index 2, so every
                    # row from there on was misnamed with no warning (hunt 2026-09-30,
                    # u20a-viz-pipelines-14). A non-categorical column is ordered the way pandas
                    # orders it on conversion, which is what the squidpy worker does to it.
                    import pandas as pd

                    series = adata.obs[key]
                    if isinstance(series.dtype, pd.CategoricalDtype):
                        labels = [str(c) for c in series.cat.categories]
                    else:
                        labels = [str(c) for c in pd.Categorical(series.dropna()).categories]
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                return _refusal(
                    "invalid_request",
                    f"the stored enrichment is {matrix.shape}, which is not a square label-by-label matrix",
                    plot_id=want,
                )
            if len(labels) != matrix.shape[0]:
                labels = [str(i) for i in range(matrix.shape[0])]
                warnings.append("the stored result records no label order, so axes are numbered rather than named")
            vmin, vmax, clip_note = _palette.limits(matrix, semantic=_SIGNED)
            fig, axes = _base.canvas(
                1,
                1,
                panel_w=max(4.6, 0.32 * len(labels) + 3.0),
                panel_h=max(4.0, 0.3 * len(labels) + 2.0),
                dpi=dpi or None,
            )
            art = axes[0].imshow(
                matrix, cmap=_palette.continuous(_SIGNED, color_map), vmin=vmin, vmax=vmax, interpolation="nearest"
            )
            fig.colorbar(art, ax=axes[0], shrink=0.75, pad=0.02).set_label("enrichment z-score", fontsize=9)
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=90, fontsize=7)
            axes[0].set_yticks(range(len(labels)))
            axes[0].set_yticklabels(labels, fontsize=7)
            plot_kind = "heatmap"
            subject = title or "Neighbourhood enrichment"
            limitations = [
                "the z-score is against a permutation null of the same labels on the same coordinates, as computed by the analysis tool",
                "adjacency is not interaction: two labels being neighbours says where they sit, not that they signal",
            ]
            if clip_note:
                limitations.append(clip_note)
            cache = {"zscore": matrix}
            findings = [("n_labels", len(labels), "labels in the matrix")]
            source = {
                "name": Path(data_path).name,
                "fingerprint": fingerprint,
                "n_obs": prof["dataset"]["n_obs"],
                "read_from": "uns neighbourhood enrichment",
            }
            data_params = {"data_path": data_path, "kind": kind, "obs_key": obs_key}
        else:  # graph
            with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
                coords, coord_note = _layers.spatial_coords(adata, library_id)
            coords = np.asarray(coords, dtype=float)[:, :2]
            k = max(1, min(int(n_neighbors), max(1, coords.shape[0] - 1)))
            drawn, sampling = _base.subsample(coords.shape[0], limit=_profile.SUBSAMPLE_ABOVE)
            points = coords[drawn] if sampling.sampled else coords
            from scipy.spatial import cKDTree

            tree = cKDTree(points)
            _, idx = tree.query(points, k=k + 1)
            edges = [(i, j) for i, row in enumerate(np.atleast_2d(idx)) for j in row[1:]]
            fig, axes = _base.canvas(1, 1, panel_w=5.6, panel_h=5.2, dpi=dpi or None)
            note = _spatial.neighbor_graph(axes[0], points, edges, title=title or f"{k}-nearest-neighbour graph")
            plot_kind = "spatial_map"
            subject = title or f"Spatial neighbour graph, k={k}"
            limitations = [
                f"edges are the {k} nearest spots by Euclidean distance on the stored coordinates; "
                "this is a drawing of adjacency, not a statistic about it",
                coord_note,
            ]
            limitations = [item for item in limitations if item]
            cache = {"n_edges": np.asarray([len(edges)])}
            findings = [("n_edges", len(edges), "edges drawn")]
            if sampling.sampled:
                warnings.append(sampling.caption())
            source = {
                "name": Path(data_path).name,
                "fingerprint": fingerprint,
                "n_obs": prof["dataset"]["n_obs"],
                "read_from": "obsm['spatial']",
            }
            data_params = {"data_path": data_path, "kind": kind, "n_neighbors": k}
            note = note or {}
        expression = None

    fig.tight_layout()
    caption = _spec.compose_caption(subject=subject, expression=None, limitations=limitations, claim="descriptive")
    stem = _spec.stem_for(want, fingerprint, data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=want,
        function="plot_spatial_statistics",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source=source,
        params={**data_params, "figure_format": fmt},
        param_origin={"kind": "caller"},
        expression=expression,
        cache=cache,
        findings=findings,
    )


@_layer_errors_refuse("trajectory.pseudotime")
def plot_trajectory(
    data_path: str,
    *,
    output_dir: str = "",
    kind: str = "pseudotime",
    pseudotime_key: str = "",
    basis: str = "auto",
    genes: str = "",
    groupby: str = "",
    layer: str = "",
    use_raw: bool = False,
    normalize: str = "auto",
    n_bins: int = 20,
    library_id: str = "",
    point_size: float = 0.0,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a pseudotime another tool computed: on an embedding, on the tissue, or against genes.

    Pseudotime is an ordering inferred from expression similarity. It is not time, it has no units,
    and its direction is set by whichever cell was chosen as the root -- so every caption here says
    so, and the root is named when the object recorded one.

    There is no spatial-trajectory family. A pseudotime painted on the tissue is exactly that: the
    same ordering, drawn at each spot's position. Distance across a section is not elapsed time, and
    this refuses to word it as though it were.
    """
    plot_ids = {
        "pseudotime": "trajectory.pseudotime",
        "spatial": "trajectory.spatial_pseudotime",
        "gene_trend": "trajectory.gene_trend",
    }
    want = plot_ids.get(kind)
    if want is None:
        return _refusal(
            "invalid_request",
            f"{kind!r} is not a trajectory view; choose one of {sorted(plot_ids)}",
            plot_id="trajectory.pseudotime",
        )
    prepared, refusal = _prepare(data_path, want, output_dir, selectors={"pseudotime_key": pseudotime_key})
    if refusal:
        refusal["never_do"] = (
            "do not describe a spatial gradient as a trajectory, and do not present pseudotime as "
            "elapsed time or in any unit"
        )
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    import numpy as np

    numeric = prof["obs_roles"]["numeric"]
    key = pseudotime_key or next(
        (c for c in numeric if "pseudotime" in c.lower() or c.lower() == "dpt" or c.lower() == "dpt_pseudotime"), ""
    )
    if not key:
        return _refusal(
            "prerequisite_not_met",
            "this object stores no pseudotime column",
            plot_id=want,
            missing=["has_pseudotime"],
            instead=["embedding.scatter"],
        )
    if key not in numeric:
        return _refusal(
            "invalid_request",
            f"obs['{key}'] is not a numeric column; the numeric ones are {numeric[:12]}",
            plot_id=want,
        )

    fmt = (figure_format or "png").lower()
    wanted = [g.strip() for g in (genes or "").split(",") if g.strip()]
    cmap = _palette.continuous("ordering", color_map)
    expression = None
    transform = ""

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        times = np.asarray(_layers.obs_values(adata, key, max_levels=10**9)[0], dtype=float)
        root_note = ""
        for slot in ("iroot", "root", "dpt_root"):
            if slot in getattr(adata, "uns", {}):
                root_note = f"the ordering starts from the root recorded in uns['{slot}']"
                break
        if kind == "spatial":
            coords, coord_note, section = _coords_and_mask(adata, library_id)
            if section is not None:
                # (hunt 2026-09-30, u20a-viz-pipelines-1) the ordering follows the coordinates.
                coords = coords[section]
                times = times[section]
                coord_note = (
                    f"only section {library_id!r} is drawn ({int(section.sum()):,} of {len(section):,} observations)"
                )
            values = times
        elif kind == "pseudotime":
            # The same words plot_embedding accepts: 'umap' worked there and raised here (hunt
            # 2026-09-30, u20a-viz-pipelines-31).
            chosen_basis = {"umap": "X_umap", "tsne": "X_tsne", "pca": "X_pca"}.get(basis, basis)
            if basis in ("", "auto"):
                embeddings = [o["key"] for o in prof["obsm"] if o["family"] == "embedding"]
                preferred = (
                    [e for e in embeddings if "umap" in e.lower()]
                    or [e for e in embeddings if "tsne" in e.lower()]
                    or embeddings
                )
                chosen_basis = preferred[0] if preferred else ""
            if not chosen_basis:
                return _refusal(
                    "prerequisite_not_met",
                    "no embedding is stored, so there is nothing to draw the ordering on",
                    plot_id=want,
                    missing=["has_embedding"],
                    instead=["trajectory.spatial_pseudotime"],
                )
            xy, _ = _layers.obsm_values(adata, chosen_basis)
            xy = np.asarray(xy, dtype=float)
        else:
            if not wanted:
                return _refusal(
                    "invalid_request",
                    "name the genes whose trend along the ordering should be drawn",
                    plot_id=want,
                )
            slot = _layers.resolve_expression_slot(
                adata, layer=layer, use_raw=use_raw, x_has_negative=bool(prof["matrix"].get("has_negative"))
            )
            found, missing, ambiguous = _layers.gene_values(adata, wanted, slot)
            if ambiguous:
                warnings.append(f"{len(ambiguous)} symbol(s) match more than one row and were skipped")
            if missing:
                warnings.append(f"{len(missing)} gene(s) are not in this object: {', '.join(missing[:8])}")
            if not found:
                return _refusal(
                    "invalid_request",
                    "none of the requested genes are in this object",
                    plot_id=want,
                    instead=["trajectory.pseudotime"],
                )
            raw = np.column_stack([np.asarray(v, dtype=float).ravel() for v in found.values()])
            integral = _slot_integral(slot, prof, found.values())
            shown, transform = _layers.normalize_for_display(raw, mode=normalize, integral=integral)
            trend_values = np.asarray(shown)
            trend_names = list(found)
            expression = {
                "slot": slot["slot"],
                "key": slot["key"],
                "transform": transform,
                "integral": integral,
                "why": slot["why"],
            }

    finite = np.isfinite(times)
    if not finite.any():
        return _refusal("invalid_request", f"obs['{key}'] holds no finite value", plot_id=want)
    if int((~finite).sum()):
        warnings.append(f"{int((~finite).sum())} observation(s) have no pseudotime and are not drawn")

    limitations = [
        f"pseudotime is an ordering inferred from expression similarity and stored in obs['{key}']; "
        "it has no units and is not elapsed time",
    ]
    if root_note:
        limitations.append(root_note)
    else:
        limitations.append("the object records no root, so the direction of the ordering is not established here")

    if kind == "spatial":
        panels = {f"pseudotime ({key})": times[finite]}
        scales = _palette.PanelScales.build(panels, share=True, semantic="ordering")
        fig, _notes = _spatial.tissue_panels(
            coords[finite],
            panels,
            categorical=False,
            scales=scales,
            ncols=1,
            title=title or "Pseudotime across the section",
            cmap=cmap,
            point_size=point_size or None,
            fmt=fmt,
        )
        plot_kind = "spatial_map"
        subject = title or "Pseudotime at each spot's position"
        limitations.append(
            "this is the same ordering drawn at each spot's position; distance across the section is not elapsed time"
        )
        if coord_note:
            limitations.append(coord_note)
        cache = {"pseudotime": times[finite]}
        scale_note = scales.caption()
    elif kind == "pseudotime":
        fig, axes = _base.canvas(1, 1, panel_w=5.4, panel_h=4.8, dpi=dpi or None)
        vmin, vmax, clip_note = _palette.limits(times[finite], semantic="ordering")
        _points.embedding_scatter(
            axes[0],
            xy[finite],
            times[finite],
            categorical=False,
            title=title or f"Pseudotime on {chosen_basis}",
            value_label=key,
            basis_label=chosen_basis.replace("X_", "").upper(),
            point_size=point_size or None,
            cmap=cmap,
            vlim=(vmin, vmax),
            fmt=fmt,
        )
        plot_kind = "scatter"
        subject = title or f"Pseudotime on {chosen_basis}"
        if clip_note:
            limitations.append(clip_note)
        cache = {"pseudotime": times[finite]}
        scale_note = ""
    else:
        order = np.argsort(times[finite])
        t = times[finite][order]
        values = trend_values[finite][order]
        bins = max(4, min(int(n_bins), max(4, t.size // 5)))
        edges = np.linspace(t.min(), t.max(), bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        which = np.clip(np.digitize(t, edges) - 1, 0, bins - 1)
        fig, axes = _base.canvas(1, 1, panel_w=6.4, panel_h=4.2, dpi=dpi or None)
        cache = {"pseudotime_bin": centres}
        for i, name in enumerate(trend_names):
            means = np.array([values[which == b, i].mean() if (which == b).any() else np.nan for b in range(bins)])
            axes[0].plot(centres, means, linewidth=1.4, label=name)
            cache[name] = means
        axes[0].set_xlabel(f"pseudotime ({key})", fontsize=9)
        axes[0].set_ylabel("mean expression in bin", fontsize=9)
        if len(trend_names) <= 20:
            axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
        plot_kind = "line"
        subject = title or f"{len(trend_names)} gene(s) along the ordering"
        limitations.append(
            f"each point is the mean over the cells in one of {bins} equal-width pseudotime bins, "
            "so a bin holding few cells is as prominent as one holding many"
        )
        scale_note = ""

    limitations.append(
        "each observation is one cell or spot, not a biological replicate; no condition-level claim is made"
    )
    fig.tight_layout()
    caption = _spec.compose_caption(
        subject=subject, expression=expression, scale_note=scale_note, limitations=limitations, claim="descriptive"
    )
    data_params = {
        "data_path": data_path,
        "kind": kind,
        "pseudotime_key": key,
        "genes": ",".join(wanted),
        "basis": basis,
        "layer": layer,
        "use_raw": use_raw,
        "normalize": normalize,
        "n_bins": n_bins,
        "library_id": library_id,
    }
    # pseudotime_key is in no DATA_PARAMS set, so dpt and palantir orderings on one embedding shared
    # one file (hunt 2026-09-30, u20a-viz-pipelines-5).
    stem = _spec.stem_for(
        want, _with_selectors(prof["dataset"]["fingerprint"], pseudotime_key=key), data_params, slug=figure_id
    )
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=want,
        function="plot_trajectory",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(data_path).name,
            "fingerprint": prof["dataset"]["fingerprint"],
            "n_obs": prof["dataset"]["n_obs"],
            "read_from": f"obs['{key}']",
        },
        params={**data_params, "figure_format": fmt},
        param_origin={"pseudotime_key": "caller" if pseudotime_key else "the first pseudotime-like numeric column"},
        expression=expression,
        cache=cache,
        findings=[("n_ordered", int(finite.sum()), "observations with a pseudotime")],
    )


@_layer_errors_refuse("communication.interactions")
def plot_cell_communication(
    results_path: str,
    *,
    output_dir: str = "",
    kind: str = "interactions",
    source_column: str = "",
    target_column: str = "",
    score_column: str = "",
    top_n: int = 20,
    color_map: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
    pvalues_path: str = "",
) -> dict[str, Any]:
    """Draw a cell-communication result another tool inferred, as a sender-receiver matrix or a ranking.

    Every caption says *inferred*. A ligand-receptor result is a statement about co-expression of
    an annotated pair in two populations, filtered through a database somebody curated. No signal
    was measured, no cells were watched talking, and a strong score is not evidence that the
    interaction occurs in this tissue.

    The columns are told by ``viz.ccc.detect`` -- the reader the explorer's panels use -- so every
    tool's table draws without naming them: a long table by its sender, receiver and score names,
    a square matrix (deeplinc, squidpy) melted to one row per sender-receiver cell, and a two-column
    vector (ncem's strength) as a ranking only. Columns the caller names always win.
    ``pvalues_path`` names a square p-value matrix beside a square score matrix; the heatmap then
    marks the cells under 0.05.
    """
    if kind not in ("interactions", "ranked"):
        return _refusal(
            "invalid_request",
            f"{kind!r} is not a communication view; choose 'interactions' or 'ranked'",
            plot_id="communication.interactions",
        )
    frame, refusal = _read_table(results_path, "cell-communication table")
    if refusal:
        refusal["plot_id"] = "communication.interactions"
        return refusal

    from .ccc import detect as _ccc_detect

    name = Path(results_path).name
    found = [str(c) for c in frame.columns]
    explicit = bool(source_column or target_column or score_column)
    detected = _ccc_detect.table_columns(found, name)
    shape = detected["shape"] if detected else ""
    warnings: list[str] = []
    # The p-values the stars mark: only ``pvalues_path``'s, merged beside a square matrix. A long table's own ``p``
    # column is not that file, and a ``pvalues_path`` given beside one was not read.
    merged_p = False
    if explicit:
        shape = "long"
    elif shape == "square":
        frame = _melt_square(frame)
        if frame.empty:
            return _refusal(
                "prerequisite_not_met",
                f"{name} reads as a sender-by-receiver matrix but holds no numeric cell",
                plot_id="communication.interactions",
                missing=["has_communication_score"],
            )
        detected = {"shape": "square", "source": "source", "target": "target", "score": "score"}
        if pvalues_path:
            p_frame, refusal = _read_table(pvalues_path, "p-value matrix")
            if refusal:
                refusal["plot_id"] = "communication.interactions"
                return refusal
            frame = frame.merge(_melt_square(p_frame, "p"), on=["source", "target"], how="left")
            merged_p = True
    elif shape == "vector":
        if kind == "interactions":
            return _refusal(
                "prerequisite_not_met",
                f"{name} holds one value per group ({detected['source']!r}, {detected['score']!r}), not a "
                f"sender and a receiver, so there is no matrix to draw; its columns are {found}. Draw "
                "kind='ranked' instead",
                plot_id="communication.interactions",
                missing=["has_communication_table"],
            )
        if not _numeric_column(frame, detected["score"]):
            return _refusal(
                "prerequisite_not_met",
                f"{name} has two columns, {found}, and the second holds no numbers to rank by",
                plot_id="communication.interactions",
                missing=["has_communication_score"],
            )
    if pvalues_path and not merged_p:
        warnings.append("pvalues_path is read only beside a square score matrix; it was not used")

    def pick(explicit_name: str, role: str) -> str:
        if explicit_name:
            return explicit_name if explicit_name in frame.columns else ""
        if detected and detected.get(role):
            return str(detected[role])
        return _ccc_detect.named_column(list(frame.columns), role) or ""

    src = pick(source_column, "source")
    dst = "" if shape == "vector" else pick(target_column, "target")
    score = pick(score_column, "score")
    if not src or (not dst and shape != "vector"):
        return _refusal(
            "prerequisite_not_met",
            f"{name} has no sender and receiver columns; its columns are {found}",
            plot_id="communication.interactions",
            missing=["has_communication_table"],
            instead=["spatial.annotation", "markers.dotplot"],
        )
    if not score:
        return _refusal(
            "prerequisite_not_met",
            f"{name} names sender and receiver but carries no score column, so "
            f"there is nothing to weight the interactions by; its columns are {found}",
            plot_id="communication.interactions",
            missing=["has_communication_score"],
        )

    fmt = (figure_format or "png").lower()
    out_dir = Path(output_dir or ".").resolve()
    figures_dir = _manifest_io.figures_dir_for(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    inference = (
        "Inferred, not measured: this is co-expression of an annotated ligand-receptor pair in two "
        "populations, scored through a curated database."
    )

    if kind == "interactions":
        table = frame.pivot_table(index=src, columns=dst, values=score, aggfunc="sum").fillna(0.0)
        matrix = table.to_numpy(dtype=float)
        # A melted matrix keeps its own sign (squidpy's z-scores): a signed ramp centred on zero. A long table
        # draws as it always has.
        semantic = _SIGNED if shape == "square" and matrix.size and float(matrix.min()) < 0 else "expression"
        vmin, vmax, clip_note = _palette.limits(matrix, semantic=semantic)
        fig, axes = _base.canvas(
            1,
            1,
            panel_w=max(4.6, 0.34 * table.shape[1] + 3.0),
            panel_h=max(4.0, 0.32 * table.shape[0] + 2.0),
            dpi=dpi or None,
        )
        art = axes[0].imshow(
            matrix, cmap=_palette.continuous(semantic, color_map), vmin=vmin, vmax=vmax, interpolation="nearest"
        )
        label = f"{name} value" if shape == "square" else f"summed {score}"
        fig.colorbar(art, ax=axes[0], shrink=0.75, pad=0.02).set_label(label, fontsize=9)
        axes[0].set_xticks(range(table.shape[1]))
        axes[0].set_xticklabels([str(c)[:22] for c in table.columns], rotation=90, fontsize=7)
        axes[0].set_yticks(range(table.shape[0]))
        axes[0].set_yticklabels([str(i)[:22] for i in table.index], fontsize=7)
        axes[0].set_xlabel(f"receiver ({dst})", fontsize=9)
        axes[0].set_ylabel(f"sender ({src})", fontsize=9)
        plot_kind = "heatmap"
        subject = title or "Inferred interactions, sender by receiver"
        if shape == "square":
            limitations = [f"cells hold {name}'s own value for that sender (row) and receiver (column)"]
        else:
            limitations = [f"cells hold the sum of {score} over every pair reported for that sender and receiver"]
        if clip_note:
            limitations.append(clip_note)
        if merged_p:
            import numpy as np

            p_table = frame.pivot_table(index=src, columns=dst, values="p", aggfunc="min").reindex(
                index=table.index, columns=table.columns
            )
            for (i, j), value in np.ndenumerate(p_table.to_numpy(dtype=float)):
                if value < 0.05:
                    axes[0].text(j, i, "*", ha="center", va="center", fontsize=9, color="#111111")
            limitations.append(f"* marks p < 0.05 in {Path(pvalues_path).name}, not corrected here for the many cells")
        cache = {"matrix": matrix}
        findings = [("n_pairs", int(len(frame)), "rows in the source table")]
    else:
        ranked = frame.reindex(frame[score].astype(float).sort_values(ascending=False).index).head(int(top_n))
        if shape == "vector":
            labels = list(ranked[src].astype(str))
        else:
            labels = [f"{a} to {b}" for a, b in zip(ranked[src].astype(str), ranked[dst].astype(str), strict=True)]
        pair_col = _ccc_detect.named_column(list(frame.columns), "pair") or ""
        if pair_col:
            labels = [f"{lbl}: {p}" for lbl, p in zip(labels, ranked[pair_col].astype(str), strict=True)]
        fig, axes = _base.canvas(1, 1, panel_w=7.2, panel_h=max(3.0, 0.3 * len(ranked) + 1.6), dpi=dpi or None)
        axes[0].barh(range(len(ranked))[::-1], ranked[score].astype(float).to_numpy(), color="#0f7d76")
        axes[0].set_yticks(range(len(ranked))[::-1])
        axes[0].set_yticklabels([lbl[:58] for lbl in labels], fontsize=7)
        axes[0].set_xlabel(str(score), fontsize=9)
        plot_kind = "bar"
        subject = title or f"Top {len(ranked)} inferred interactions"
        limitations = [f"ranked by {score} as stored in {Path(results_path).name}"]
        cache = {score: ranked[score].astype(float).to_numpy()}
        findings = [("n_pairs", int(len(frame)), "rows in the source table")]

    limitations.append(
        "a score is not evidence that the interaction occurs in this tissue; it is a ranking of candidates"
    )
    fig.tight_layout()
    caption = _spec.compose_caption(
        subject=subject, expression=None, limitations=limitations, inference=inference, claim="descriptive"
    )
    data_params = {"results_path": results_path, "kind": kind, "top_n": top_n}
    # The table's content, the view and the columns read reach the stem: with an empty fingerprint
    # CellPhoneDB and LIANA results, and both views of one table, drew to one file (hunt
    # 2026-09-30, u20a-viz-pipelines-4).
    stem = _spec.stem_for(
        "communication.interactions",
        _with_selectors(
            _file_fingerprint(results_path),
            kind=kind,
            source_column=src,
            target_column=dst,
            score_column=score,
            pvalues=_file_fingerprint(pvalues_path) if merged_p else "",
        ),
        data_params,
        slug=figure_id,
    )
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id="communication.interactions",
        function="plot_cell_communication",
        kind=plot_kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "name": Path(results_path).name,
            "read_from": (
                "every cell, rows as sender and columns as receiver"
                if shape == "square"
                else ", ".join(c for c in (src, dst, score) if c)
            ),
        },
        params={
            **data_params,
            "source_column": src,
            "target_column": dst,
            "score_column": score,
            "figure_format": fmt,
            **({"pvalues_path": pvalues_path} if merged_p else {}),
        },
        param_origin={
            "score_column": "caller"
            if score_column
            else ("the matrix's cells" if shape == "square" else "the first score-like column in the table")
        },
        expression=None,
        cache=cache,
        findings=findings,
    )


#: Which pipeline draws each ``function`` recorded in a spec. A revision re-invokes the producer
#: rather than re-implementing its drawing, so a figure's second version is made by the same code
#: that made its first and cannot drift from it.

# ---------------------------------------------------------------------------------------------
# Three dimensions
#
# Every 2D producer above reads obsm['spatial'] through `_layers.spatial_coords`, which is why
# writing aligned coordinates to a new key made them invisible to the whole toolkit. These three
# read through `_layers.spatial_coords_3d` instead, which resolves the contract's keys, falls back
# to a z built from the section axis, and refuses rather than inventing a spacing.
# ---------------------------------------------------------------------------------------------


def _one_value(adata, prof, gene: str, obs_key: str, plot_id: str, *, layer: str = "", use_raw: bool = False):
    """``(values, label, categorical, expression_record)`` for the one value these families paint.

    The 3D families paint one value across the whole stack rather than one panel per gene, so they
    read a single column -- but they read it through the same slot resolution as everything else,
    because a figure that does not record which matrix it came from cannot be reproduced.
    """
    if gene:
        slot = _layers.resolve_expression_slot(
            adata, layer=layer, use_raw=use_raw, x_has_negative=bool(prof["matrix"]["has_negative"])
        )
        found, missing, ambiguous = _layers.gene_values(adata, [gene], slot)
        if not found:
            return None, _refusal(
                "prerequisite_not_met",
                f"{gene!r} is not a gene in this object"
                + ("; it appears more than once, so it does not identify one row" if ambiguous else ""),
                plot_id=plot_id,
                fix="check the spelling, or use the identifiers the dataset inspection listed",
            )
        integral = _slot_integral(slot, prof, [found[gene]])
        values, transform = _layers.normalize_for_display(found[gene], mode="auto", integral=integral)
        record = {
            "slot": slot["slot"],
            "key": slot["key"],
            "transform": transform,
            "integral": integral,
            "why": slot["why"],
        }
        return (values, gene, False, record), None
    if obs_key:
        values, categorical = _layers.obs_values(adata, obs_key)
        return (values, obs_key, categorical, None), None
    return (None, "", False, None), None


def _resolve_3d(adata, plot_id, *, coords_key, section_key, z_spacing):
    """``(xyz, coords_from, sections, section_from)`` or a refusal dict."""
    try:
        return _layers.spatial_coords_3d(
            adata, coords_key=coords_key, section_key=section_key, z_spacing=z_spacing
        ), None
    except _layers.LayerError as exc:
        return None, _refusal(
            "prerequisite_not_met",
            str(exc),
            plot_id=plot_id,
            missing=["has_3d_coords"],
            fix=exc.fix,
            instead=["spatial.expression", "spatial.sections"],
        )


#: Mirrors ``sog_portal.interactive.MAX_TRACES``, which REFUSES a spec with more traces rather than
#: truncating it. Duplicated rather than imported for the reason that module gives about its own
#: colour-scale list: the viz package must not import the portal's.
_FIG3D_MAX_TRACES = 64


def _fig3d_spec(xyz, values, *, categorical, value_label, sections, labels, units, title, warnings=None):
    """The data-only spec the portal renders in a frame. Numbers and enumerated strings, nothing else.

    No markup, no script, no colour the caller chose: a trace names a colour scale from a fixed list
    or nothing at all. The portal rebuilds a plotly document from this against its own allowlist and
    reads nothing it was not expecting, so what is produced here only has to survive that rebuild.

    A categorical value becomes one trace per level, because a 3D scatter has no legend for a
    colour bar of strings, and a continuous one becomes a single trace with a scale. A stack with
    neither is drawn per section, so turning it shows which plane a point belongs to -- which is
    the question the rotatable view exists to answer.
    """
    import numpy as np

    arr = np.asarray(xyz, dtype=float)
    # ``subsample`` returns ``None`` for "kept everything", which is not an index. Making it one
    # here keeps every use below a plain fancy-index instead of a branch at each of them.
    index, sampled = _base.subsample(len(arr), _spec.FIG3D_MAX_POINTS)
    kept = np.arange(len(arr)) if index is None else np.asarray(index)
    arr = arr[kept]
    sampling = {
        "n_total": int(sampled.n_total),
        "n_drawn": int(sampled.n_drawn),
        "method": str(sampled.method),
    }
    traces: list[dict[str, Any]] = []

    def _xyz(mask=None):
        block = arr if mask is None else arr[mask]
        return {
            "x": [round(float(v), 4) for v in block[:, 0]],
            "y": [round(float(v), 4) for v in block[:, 1]],
            "z": [round(float(v), 4) for v in block[:, 2]],
        }

    if values is not None and not categorical:
        column = np.asarray(values, dtype=float)[kept]
        traces.append(
            {
                "name": value_label or "value",
                **_xyz(),
                "color_values": [round(float(v), 4) for v in np.nan_to_num(column)],
                "colorscale": "Viridis",
                "colorbar_title": value_label or "value",
            }
        )
    else:
        groups = values if values is not None else sections
        if groups is None:
            traces.append({"name": "cells", **_xyz()})
        else:
            level_of = np.asarray(groups).astype(str)[kept]
            # Every level, in natural order -- or, for sections, in depth order. The 12-panel grid
            # cap and a string sort kept S1, S10..S19, S2 and dropped the rest of a stack from the
            # rotatable view with nothing said (hunt 2026-09-30, u20a-viz-pipelines-29). Past the
            # portal's trace limit the remainder is pooled into one named trace, and said so.
            levels = sorted(set(level_of.tolist()), key=_natural_key)
            if values is None:
                depth = {lvl: float(np.nanmean(arr[level_of == lvl, 2])) for lvl in levels}
                levels = sorted(levels, key=lambda lvl: (depth[lvl], _natural_key(lvl)))
            if len(levels) > _FIG3D_MAX_TRACES:
                head, rest = levels[: _FIG3D_MAX_TRACES - 1], levels[_FIG3D_MAX_TRACES - 1 :]
                if warnings is not None:
                    warnings.append(
                        f"the rotatable view shows {len(head)} of {len(levels)} levels by name; the other "
                        f"{len(rest)} are drawn together as one trace named 'other'"
                    )
            else:
                head, rest = levels, []
            for level in head:
                traces.append({"name": str(level), **_xyz(level_of == level)})
            if rest:
                traces.append({"name": "other", **_xyz(np.isin(level_of, rest))})

    return {
        "title": str(title),
        "axis_labels": [str(a) for a in labels],
        "z_units": str(units or ""),
        "traces": traces,
        "n_points": int(len(arr)),
        "sampling": sampling,
    }


@_layer_errors_refuse("spatial3d.scatter")
def plot_spatial_3d(
    data_path: str,
    *,
    view: str = "scatter",
    genes: str = "",
    obs_key: str = "",
    layer: str = "",
    use_raw: bool = False,
    coords_key: str = "",
    section_key: str = "",
    z_spacing: float = 0.0,
    axis: str = "z",
    n_bins: int = 20,
    output_dir: str = "",
    point_size: float = 0.0,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Draw a reconstructed stack: the volume itself, its depth profile, or an axis gradient.

    ``view`` selects which: ``scatter`` for the three-angle volume, ``depth`` for cells and mean
    value per z plane, ``axis`` for a value binned along a named anatomical axis, and
    ``interactive`` for the volume again plus a data-only spec the portal can render in a frame the
    reader turns.

    ``interactive`` writes the PNG as well, and the PNG is what the manifest declares. A frame that
    fails to load, a reader who exports the conversation, a PDF -- none of those may lose the
    figure, so the rotatable view is an addition and never the only copy.
    """
    plot_id = {
        "scatter": "spatial3d.scatter",
        "depth": "spatial3d.depth_profile",
        "axis": "spatial3d.axis_profile",
        "interactive": "interactive.volume",
    }.get(view, "")
    if not plot_id:
        return _refusal(
            "invalid_request",
            f"view={view!r} is not one of 'scatter', 'depth', 'axis' or 'interactive'",
            plot_id="spatial3d.scatter",
        )
    prepared, refusal = _prepare(
        data_path,
        plot_id,
        output_dir,
        selectors={"coords_key": coords_key, "section_key": section_key, "z_spacing": z_spacing},
    )
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    import numpy as np

    from spatialomicsgym.viz.render import volume as _volume

    gene_list = [g.strip() for g in (genes or "").split(",") if g.strip()]
    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        resolved, refusal = _resolve_3d(
            adata, plot_id, coords_key=coords_key, section_key=section_key, z_spacing=z_spacing
        )
        if refusal:
            return refusal
        xyz, coords_from, sections, section_from = resolved

        try:
            picked, refusal = _one_value(
                adata,
                prof,
                gene_list[0] if gene_list else "",
                obs_key,
                plot_id,
                layer=layer,
                use_raw=use_raw,
            )
        except _layers.LayerError as exc:
            return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)
        if refusal:
            return refusal
        values, value_label, categorical, expression_record = picked

        provenance = adata.uns.get("spatial_3d") if hasattr(adata, "uns") else None
        provenance = dict(provenance) if isinstance(provenance, dict) else {}

    limitations: list[str] = []
    # Cells the 3D frame does not cover carry NaN rows (the contract records coordinate_coverage
    # below one); the interactive spec failed allow_nan with a bare '(ValueError)', and nothing said
    # cells were missing (hunt 2026-09-30, u20b-viz-rest-27). Left out, counted, and said.
    finite_rows = np.isfinite(np.asarray(xyz, dtype=float)).all(axis=1)
    if not finite_rows.all():
        missing_rows = (
            f"{int((~finite_rows).sum()):,} of {len(finite_rows):,} observations have no coordinate in "
            f"{coords_from} and are left out of every view"
        )
        warnings.append(missing_rows)
        limitations.append(missing_rows)
        xyz = np.asarray(xyz, dtype=float)[finite_rows]
        values = None if values is None else np.asarray(values)[finite_rows]
        sections = None if sections is None else np.asarray(sections)[finite_rows]
        if not len(xyz):
            return _refusal(
                "prerequisite_not_met",
                f"no observation has a finite coordinate in {coords_from}",
                plot_id=plot_id,
                missing=["has_3d_coords"],
            )

    fmt = (figure_format or "png").lower()
    units = str(provenance.get("z_units") or "")
    axis_map = provenance.get("axis_map") or {}
    labels = tuple(str(axis_map.get(str(i), name)) for i, name in enumerate(("x", "y", "z")))
    findings: list[tuple[str, Any, str]] = []
    fig3d: dict[str, Any] | None = None

    if view in ("scatter", "interactive"):
        fig, note = _volume.cloud_3d_views(
            xyz,
            values,
            categorical=categorical,
            labels=labels,
            title=title or (f"{value_label} across the stack" if value_label else "the stack"),
        )
        sampling = note.get("sampling") or {}
        if (
            isinstance(sampling, dict)
            and sampling.get("n_kept", 0)
            and sampling.get("n_kept") != sampling.get("n_total")
        ):
            limitations.append(
                f"{sampling.get('how', 'subsampled')}: an Axes3D scatter has no depth buffer, so "
                f"above {_volume.MAX_3D_POINTS:,} points it is both illegible and wrong about what "
                f"is in front"
            )
        findings.append(("n_points_drawn", note.get("n_points"), "points in the 3D view"))
        subject = title or f"{value_label or 'the stack'} in three dimensions, from three angles"
        kind = "scatter"
        if view == "interactive":
            fig3d = _fig3d_spec(
                xyz,
                values,
                categorical=categorical,
                value_label=value_label,
                sections=sections,
                labels=labels,
                units=units,
                title=title or subject,
                warnings=warnings,
            )
    elif view == "depth":
        fig, note = _volume.depth_profile(
            xyz[:, 2],
            values if not categorical else None,
            title=title or "what sits at each depth",
            z_units=units,
        )
        findings.append(("n_planes", note.get("n_planes"), "distinct z planes"))
        # A binned z has no plane gaps to compare, so it was always called "not uniform" (hunt
        # 2026-09-30, u20b-viz-rest-27); one plane has no spacing at all.
        if note.get("binned"):
            limitations.append(
                f"the z is continuous ({note['n_planes']:,} distinct values), so it is summarised in "
                f"{note['n_bins']} equal-width bins rather than plane by plane"
            )
        elif note.get("n_planes", 0) > 1 and not note.get("uniform"):
            limitations.append(
                "the section spacing is not uniform, so any 3D neighbourhood built on a single "
                "radius is wrong wherever the real gap differs"
            )
        if note.get("n_planes", 0) and sections is not None and note["n_planes"] < len(set(sections)):
            limitations.append(
                f"{len(set(sections))} sections occupy only {note['n_planes']} distinct z values, so "
                f"some sections are coincident and adjacency among them is undefined"
            )
        subject = title or "cells and mean value at each depth"
        kind = "line"
    else:
        index = {"x": 0, "y": 1, "z": 2}.get(axis.lower())
        if index is None:
            return _refusal("invalid_request", f"axis={axis!r} is not 'x', 'y' or 'z'", plot_id=plot_id)
        if values is None or categorical:
            return _refusal(
                "invalid_request",
                "an axis profile needs a continuous value: name a gene, or a numeric obs column",
                plot_id=plot_id,
                instead=["spatial3d.scatter"],
            )
        position = np.asarray(xyz[:, index], dtype=float)
        edges = np.linspace(position.min(), position.max(), int(max(2, n_bins)) + 1)
        centres = (edges[:-1] + edges[1:]) / 2.0
        binned = np.digitize(position, edges[1:-1])
        means = np.array(
            [
                float(np.nanmean(np.asarray(values, float)[binned == b])) if (binned == b).any() else np.nan
                for b in range(len(centres))
            ]
        )
        fig, axes = _base.canvas(1, 1, panel_w=5.6, panel_h=3.6, dpi=dpi or None)
        axes[0].plot(centres, means, marker="o")
        _base.finish(
            fig,
            axes[0],
            title or f"{value_label} along {labels[index]}",
            f"{labels[index]} ({units})" if units else labels[index],
            value_label or "value",
        )
        fig.tight_layout()
        note = {"axis": labels[index], "n_bins": int(n_bins)}
        limitations.append(
            "this is a position gradient, not a trajectory: the axis is a place in the tissue, and "
            "the figure carries no pseudotime, no root and no direction of development"
        )
        subject = title or f"{value_label} binned along {labels[index]}"
        kind = "line"

    if "plus a z" in coords_from:
        limitations.append(
            "the z was built from the section axis rather than read from a 3D coordinate key, so "
            "the in-plane coordinates are the ones the sections were measured in and no alignment "
            "has been applied to them"
        )

    caption = _spec.compose_caption(
        subject=subject,
        expression=expression_record,
        scale_note="",
        limitations=limitations,
        claim="descriptive",
    )
    # layer, use_raw and z_spacing select different values and were left out of the dict the stem
    # hashes (hunt 2026-09-30, u20a-viz-pipelines-5).
    data_params = {
        "data_path": data_path,
        "view": view,
        "genes": genes,
        "obs_key": obs_key,
        "layer": layer,
        "use_raw": use_raw,
        "coords_key": coords_key,
        "section_key": section_key,
        "z_spacing": z_spacing,
        "axis": axis,
        "n_bins": n_bins,
    }
    stem = _spec.stem_for(plot_id, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    interactive_name = ""
    if fig3d is not None:
        interactive_name, why_not = _spec.write_fig3d(figures_dir, stem, fig3d)
        if why_not:
            warnings.append(why_not)
            limitations.append(
                "the rotatable view was not written, so this is the static figure only; the reason is in the warnings"
            )
    result = _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_spatial_3d",
        kind=kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={
            "coords_from": coords_from,
            "section_from": section_from,
            "z_units": units,
            "provenance": provenance,
            "n_sections": int(len(set(sections))) if sections is not None else 0,
        },
        params=data_params,
        param_origin={},
        expression=expression_record,
        cache=None,
        findings=findings,
    )
    if interactive_name and result.get("status") == "ok":
        result["interactive"] = interactive_name
    return result


@_layer_errors_refuse("spatial.sections")
def plot_section_grid(
    data_path: str,
    *,
    genes: str = "",
    obs_key: str = "",
    layer: str = "",
    use_raw: bool = False,
    section_key: str = "",
    ncols: int = 4,
    output_dir: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """One panel per section, on one shared colour scale.

    ``spatial.sections`` was in the catalogue from the day the toolkit shipped and nothing drew it:
    the tissue-map functions panel per GENE, so a merged multi-section object came out as every
    section overlaid in a single frame with nothing said about it.
    """
    plot_id = "spatial.sections"
    prepared, refusal = _prepare(data_path, plot_id, output_dir, selectors={"section_key": section_key})
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    from spatialomicsgym.viz.render import volume as _volume

    gene_list = [g.strip() for g in (genes or "").split(",") if g.strip()]
    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        try:
            sections, section_from = _layers.section_labels(adata, section_key)
            if sections is None:
                return _refusal(
                    "prerequisite_not_met",
                    "this object has no section axis, so there are no sections to lay side by side",
                    plot_id=plot_id,
                    missing=["has_section_axis"],
                    instead=["spatial.expression"],
                )
            coords, coords_from = _layers.spatial_coords(adata)
            picked, refusal = _one_value(
                adata,
                prof,
                gene_list[0] if gene_list else "",
                obs_key,
                plot_id,
                layer=layer,
                use_raw=use_raw,
            )
            if refusal:
                return refusal
            values, value_label, categorical, expression_record = picked
        except _layers.LayerError as exc:
            return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)

    fmt = (figure_format or "png").lower()
    fig, note = _volume.section_panels(
        coords,
        sections,
        values,
        categorical=categorical,
        ncols=int(ncols),
        title=title or (f"{value_label} per section" if value_label else "sections"),
    )
    limitations: list[str] = []
    if note["n_not_drawn"]:
        limitations.append(
            f"{note['n_not_drawn']} of {note['n_sections']} sections are not drawn: the panel grid "
            f"is capped, and the ones left out are {', '.join(note['not_drawn'][:6])}"
        )
    if note.get("shared_scale"):
        limitations.append(
            f"every panel shares one colour scale, {note['shared_scale'][0]:.3g} to "
            f"{note['shared_scale'][1]:.3g}, so panels are comparable to each other"
        )
    subject = title or f"{value_label or 'each section'}, one panel per section"
    caption = _spec.compose_caption(
        subject=subject,
        expression=expression_record,
        scale_note="",
        limitations=limitations,
        claim="descriptive",
    )
    data_params = {
        "data_path": data_path,
        "genes": genes,
        "obs_key": obs_key,
        "layer": layer,
        "use_raw": use_raw,
        "section_key": section_key,
    }
    stem = _spec.stem_for(plot_id, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_section_grid",
        kind="grid",
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={"coords_from": coords_from, "section_from": section_from, "n_sections": note["n_sections"]},
        params=data_params,
        param_origin={},
        expression=expression_record,
        cache=None,
        findings=[("n_sections", note["n_sections"], "sections in the object")],
    )


@_layer_errors_refuse("alignment.before_after")
def plot_alignment_qc(
    data_path: str,
    *,
    mode: str = "before_after",
    before_key: str = "",
    after_key: str = "",
    coords_key: str = "",
    section_key: str = "",
    pair: str = "",
    output_dir: str = "",
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """One adjacent pair of sections, side by side across two frames or overlaid in one.

    ``mode='before_after'`` needs BOTH frames and refuses without them. That refusal is
    load-bearing: an aligner that overwrote the coordinates it was given leaves nothing to compare,
    so the figure is unobtainable exactly when the alignment cannot be validated.

    ``mode='overlay'`` draws the same pair in a single frame at full size, and needs only one set of
    coordinates -- which is what makes it the Phase-1 figure: it answers "do these two adjacent
    sections sit on top of each other" before any alignment has been run, and afterwards it answers
    it again about whichever frame is named.
    """
    if mode not in ("before_after", "overlay"):
        return _refusal(
            "invalid_request",
            f"mode={mode!r} is not 'before_after' or 'overlay'",
            plot_id="alignment.before_after",
        )
    plot_id = "alignment.before_after" if mode == "before_after" else "alignment.pair_overlay"
    prepared, refusal = _prepare(
        data_path,
        plot_id,
        output_dir,
        selectors={"section_key": section_key, "before_key": before_key, "after_key": after_key},
    )
    if refusal:
        return refusal
    prof, out_dir, figures_dir = prepared
    warnings: list[str] = list(prof.get("warnings") or [])

    import numpy as np

    with _layers.open_source(data_path, estimated_dense_gb=prof["dataset"]["estimated_dense_gb"]) as adata:
        obsm = getattr(adata, "obsm", {}) or {}
        # The one BEFORE/AFTER table the catalogue counts frames by; a second literal copy here could
        # drift from it and put the catalogue back to offering a pair this refuses (hunt 2026-09-30,
        # u20b-viz-rest-8).
        default_before, default_after = _profile.frame_pair(list(obsm))
        before = before_key or default_before
        after = after_key or default_after
        if mode == "overlay":
            # One frame, named or preferred-aligned-then-original. The aligned frame first because
            # once an alignment has been run that is what a reader means by "the coordinates".
            before = coords_key or after or before
            if not before or before not in obsm:
                return _refusal(
                    "prerequisite_not_met",
                    f"there is no coordinate key to draw; this object has {sorted(obsm)}",
                    plot_id=plot_id,
                    missing=["has_coords"],
                )
            after = before
        elif not before or not after or before == after:
            return _refusal(
                "prerequisite_not_met",
                (
                    f"a before/after needs two distinct coordinate frames; this object has "
                    f"{sorted(obsm)}. An aligner that overwrote obsm['spatial'] leaves nothing to "
                    f"compare, which is the same condition under which its alignment cannot be validated."
                ),
                plot_id=plot_id,
                missing=["has_3d_coords"],
                instead=["spatial.sections"],
            )
        try:
            sections, section_from = _layers.section_labels(adata, section_key)
        except _layers.LayerError as exc:
            return _refusal("prerequisite_not_met", str(exc), plot_id=plot_id, fix=exc.fix)
        if sections is None:
            return _refusal(
                "prerequisite_not_met",
                "this object has no section axis, so there is no adjacent pair to draw",
                plot_id=plot_id,
                missing=["has_section_axis"],
            )
        absent = [k for k in dict.fromkeys((before, after)) if k not in obsm]
        if absent:
            return _refusal(
                "prerequisite_not_met",
                f"there is no obsm[{absent[0]!r}]; this object has {sorted(obsm)}",
                plot_id=plot_id,
                missing=["has_coords"],
                fix="name one of the coordinate keys above",
            )
        b = np.asarray(obsm[before], dtype=float)[:, :2]
        a = np.asarray(obsm[after], dtype=float)[:, :2]

    # Numeric labels by value, others in natural order, so S1..S12 gives S1 and S2 as the first
    # adjacent pair rather than S1 and S10 (hunt 2026-09-30, u20a-viz-pipelines-17).
    order = list(dict.fromkeys(np.asarray(sections).astype(str).tolist()))
    try:
        order = sorted(order, key=float)
    except ValueError:
        order = sorted(order, key=_natural_key)
    if pair:
        wanted = [p.strip() for p in pair.split(",") if p.strip()]
        if len(wanted) != 2 or any(w not in order for w in wanted):
            return _refusal(
                "invalid_request",
                f"pair={pair!r} must name two of {order}",
                plot_id=plot_id,
            )
        first, second = wanted
    else:
        if len(order) < 2:
            return _refusal("prerequisite_not_met", "only one section is present, so there is no pair", plot_id=plot_id)
        first, second = order[0], order[1]

    labels = np.asarray(sections).astype(str)
    m1, m2 = labels == first, labels == second
    fmt = (figure_format or "png").lower()

    def _offset(frame):
        return float(np.linalg.norm(frame[m1].mean(0) - frame[m2].mean(0)))

    def _draw(ax, frame, heading):
        ax.scatter(frame[m1, 0], frame[m1, 1], s=3, linewidths=0, label=first, alpha=0.7)
        ax.scatter(frame[m2, 0], frame[m2, 1], s=3, linewidths=0, label=second, alpha=0.7)
        ax.set_aspect("equal")
        ax.legend(fontsize=8, markerscale=3)
        _base.finish(fig, ax, heading, "x", "y")

    if mode == "overlay":
        fig, axes = _base.canvas(1, 1, panel_w=6.4, panel_h=6.0, dpi=dpi or None)
        _draw(axes[0], a, title or f"{first} and {second} in obsm[{before!r}]")
        fig.tight_layout()
        limitations = [
            f"centroid offset between the two sections: {_offset(a):.4g}, in the units of "
            f"obsm[{before!r}]. Two adjacent sections are different cells, so they never coincide: "
            f"read this as how far apart the tissue outlines sit, not as a registration error."
        ]
        subject = title or f"{first} and {second}, overlaid"
        kind = "scatter"
        findings = [("centroid_offset", round(_offset(a), 5), "centroid offset between the pair")]
    else:
        fig, axes = _base.canvas(1, 2, panel_w=4.6, panel_h=4.2, dpi=dpi or None)
        for ax, frame, when in ((axes[0], b, "before"), (axes[1], a, "after")):
            _draw(ax, frame, f"{when} ({before if when == 'before' else after})")
        fig.suptitle(title or f"{first} and {second}, before and after alignment")
        fig.tight_layout()
        limitations = [
            f"centroid offset between the two sections: {_offset(b):.4g} before, {_offset(a):.4g} "
            f"after. Read the pair, not the stack: an alignment that improves the median while "
            f"collapsing one pair is the failure this figure exists to make visible."
        ]
        subject = title or f"{first} and {second}, before and after"
        kind = "grid"
        findings = [
            ("centroid_offset_before", round(_offset(b), 5), "centroid offset before alignment"),
            ("centroid_offset_after", round(_offset(a), 5), "centroid offset after alignment"),
        ]
    caption = _spec.compose_caption(
        subject=subject, expression=None, scale_note="", limitations=limitations, claim="descriptive"
    )
    data_params = {
        "data_path": data_path,
        "mode": mode,
        "before_key": before,
        "after_key": after,
        "section_key": section_key,
        "pair": f"{first},{second}",
    }
    stem = _spec.stem_for(plot_id, prof["dataset"]["fingerprint"], data_params, slug=figure_id)
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id=plot_id,
        function="plot_alignment_qc",
        kind=kind,
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={"before_key": before, "after_key": after, "section_from": section_from},
        params=data_params,
        param_origin={},
        expression=None,
        cache=None,
        findings=findings,
    )


_PRODUCERS: dict[str, str] = {
    "plot_spatial_expression": "plot_spatial_expression",
    "plot_spatial_annotation": "plot_spatial_annotation",
    "plot_embedding": "plot_embedding",
    "generate_qc_report": "generate_qc_report",
    "plot_marker_expression": "plot_marker_expression",
    "plot_differential_expression": "plot_differential_expression",
    "plot_deconvolution": "plot_deconvolution",
    "plot_pathway_results": "plot_pathway_results",
    "plot_spatial_statistics": "plot_spatial_statistics",
    "plot_trajectory": "plot_trajectory",
    "plot_cell_communication": "plot_cell_communication",
    "plot_spatial_3d": "plot_spatial_3d",
    "plot_section_grid": "plot_section_grid",
    "plot_alignment_qc": "plot_alignment_qc",
}


def update_visualization(
    figure_spec: str,
    *,
    set_params: str = "",
    unset_params: str = "",
    output_dir: str = "",
    data_path: str = "",
    allow_reread: bool = True,
) -> dict[str, Any]:
    """Change a figure that already exists, without redoing the analysis behind it.

    A change of appearance or layout is served from the table of values frozen beside the figure
    when it was drawn, so it costs no read of the dataset and works in a later turn, after a
    restart, and on a box where the source has moved. A change that selects different values
    needs the dataset and says so.
    """
    import json

    try:
        record = _spec.load(figure_spec)
    except FileNotFoundError as exc:
        return _refusal("not_found", str(exc), plot_id="")
    try:
        changes = json.loads(set_params) if set_params.strip().startswith("{") else _kv(set_params)
    except Exception as exc:
        return _refusal("invalid_request", f"set_params could not be read: {exc}", plot_id=record.get("plot_id", ""))
    drop = [s.strip() for s in unset_params.split(",") if s.strip()]
    # A full path handed back for an input the record keeps by NAME is that same input, not a new
    # selection. Classed as a data change, the redraw this function's own refusal asks for -- "pass
    # its full path" -- came back as needs_redraw, and re-running the producer with only a new title
    # landed on version one's file (hunt 2026-09-30, u20a-viz-pipelines-10).
    recorded = record.get("params") or {}
    sources = {
        key: str(changes.pop(key))
        for key in list(changes)
        if key in _PATH_PARAMS
        and isinstance(changes[key], str)
        and recorded.get(key)
        and Path(changes[key]).name == recorded.get(key)
    }
    if not changes and not drop:
        return _refusal(
            "invalid_request",
            "no change was requested" + (f"; {', '.join(sources)} names the input already drawn" if sources else ""),
            plot_id=record.get("plot_id", ""),
        )

    patched, klass, changed = _spec.patch(record, changes, drop)
    producer = _PRODUCERS.get(str(record.get("function") or ""))
    if klass == _spec.CLASS_DATA:
        if not allow_reread:
            return _refusal(
                "needs_the_data",
                f"changing {', '.join(k for k in changed if k in _spec.DATA_PARAMS)} selects different "
                "values, which cannot be done from the frozen table",
                plot_id=record.get("plot_id", ""),
                fix="call the plotting function again with the new selection, or allow a re-read",
            )
        # Only the producer's own parameters, and the source as a path rather than the recorded file
        # name: the merged record carried non-parameters and a bare basename, so following call_with
        # verbatim failed (hunt 2026-09-30, u20a-viz-pipelines-33).
        accepted = _signature_of(producer) if producer else set()
        # What the caller chose, not what the producer resolved: the record keeps the colour map it
        # drew with ('viridis') under param_origin 'default', and handing that back pins a choice
        # the caller never made (hunt 2026-09-30, u20a-viz-pipelines-33, the review of its fix).
        origin = record.get("param_origin") or {}
        resolved = {k for k, how in origin.items() if how not in ("caller", "revision") and k not in changes}
        merged = {k: v for k, v in {**(record.get("params") or {}), **changes}.items() if k not in resolved}
        call_with = {k: v for k, v in merged.items() if not accepted or k in accepted}
        given = _source_role(producer, data_path) if (data_path and producer) else ""
        if given:
            call_with[given] = data_path
        call_with.update({k: v for k, v in sources.items() if not accepted or k in accepted})
        named_only = [
            k for k in _PATH_PARAMS if call_with.get(k) and k not in changes and k != given and k not in sources
        ]
        reply = {
            "status": "needs_redraw",
            "plot_id": record.get("plot_id", ""),
            "function": record.get("function", ""),
            "why": (
                f"{', '.join(k for k in changed if k in _spec.DATA_PARAMS)} changes which values are "
                "plotted, so this is a different figure rather than a new view of this one"
            ),
            "call_with": call_with,
        }
        if named_only:
            reply["note"] = (
                f"{', '.join(named_only)} in call_with {'is' if len(named_only) == 1 else 'are'} the file name the "
                "record kept, not a path; pass the full path"
            )
        return reply

    # The cache is read from beside the record, always. output_dir is where a redraw is WRITTEN --
    # the meaning every producer gives it -- and reading the cache from it refused every edit made
    # with the run's own output_dir as "not cached" (hunt 2026-09-30, u20a-viz-pipelines-8).
    figures_dir = Path(figure_spec).parent
    cached = _spec.read_cache(figures_dir, record["stem"])
    source_path = data_path or ""
    if cached is None and not ((source_path or sources) and allow_reread and producer is not None):
        # Without a cache the view can still be drawn from the source, which is what a redraw
        # does anyway; the 3D families write no cache, and refusing them even with data_path given
        # made every style change on them impossible (hunt 2026-09-30, skeptic note on -7).
        return _refusal(
            "needs_the_data",
            "the values behind this figure were not cached, so its appearance cannot be changed "
            "without reading the dataset again",
            plot_id=record.get("plot_id", ""),
            fix="pass data_path so it can be drawn again with the new appearance",
        )
    # The record describes the new view. Now make it exist: a revision that returns a description
    # and no file is the defect this whole layer is for -- the caller believes the figure changed,
    # the chat still shows version one, and nothing reports a problem.
    #
    # Version two lands BESIDE version one and never over it. Overwriting silently breaks three
    # things at once: the harvested dataset record is idempotent by path and keeps the old title
    # for ever, the manifest entry keeps its old slot, and an earlier turn's card has its picture
    # changed underneath a reader scrolling back.
    if producer is None:
        redraw_note = f"no producer is registered for {record.get('function')!r}, so it cannot be redrawn here"
    elif not (source_path or sources):
        redraw_note = (
            "the figure's record stores the source by name rather than by path, so the file it was "
            "drawn from has to be named again; pass data_path to have this drawn"
        )
    elif not allow_reread:
        return _refusal(
            "needs_the_data",
            "this view has to be drawn from the source again, and a re-read was not allowed",
            plot_id=record.get("plot_id", ""),
            fix="allow a re-read, or call the plotting function directly with the new appearance",
        )
    else:
        call, redraw_note = _redraw_call(producer, patched.get("params") or {}, source_path, sources)
        if call is not None:
            call["output_dir"] = str(output_dir or figures_dir.parent.parent)
            # The next free version, not "revision + 1": two edits of one v1 both became v2, and since
            # the stem hashes only data parameters the second overwrote the first (hunt 2026-09-30,
            # u20a-viz-pipelines-7).
            call["figure_id"], version = _next_version_slug(
                figures_dir, str(record["stem"]), _manifest_io.figures_dir_for(call["output_dir"])
            )
            accepted = _signature_of(producer)
            try:
                drawn = globals()[producer](**{k: v for k, v in call.items() if k in accepted})
            except Exception as exc:
                drawn = {"status": "error", "why": f"the redraw failed: {type(exc).__name__}: {exc}"}
            if drawn.get("status") == "ok":
                saved, lineage_error = _record_lineage(drawn, record, changed, patched)
                reply = {
                    "status": "ok",
                    "change_class": klass,
                    "changed": changed,
                    "figure_id": drawn["figure_id"],
                    "derived_from": record.get("figure_id", ""),
                    "cached_arrays": sorted(cached or {}),
                    "figure": drawn["figure"],
                    "caption": drawn["caption"],
                    "output_dir": drawn["output_dir"],
                    "manifest": drawn.get("manifest", ""),
                    "note": f"version {version} was drawn beside the earlier one; the earlier figure is untouched",
                    "spec": saved or patched,
                }
                if lineage_error:
                    reply["lineage_saved"] = False
                    reply["warnings"] = [
                        lineage_error + "; the figure was drawn, and its record on disk does not name what it "
                        "was derived from or carry this change, so a later edit of it starts from the producer's "
                        "own parameters"
                    ]
                return reply
            redraw_note = str(drawn.get("why") or drawn.get("error") or "the redraw did not succeed")

    return {
        "status": "ok",
        "change_class": klass,
        "changed": changed,
        "figure_id": patched["figure_id"],
        "derived_from": patched["derived_from"],
        "cached_arrays": sorted(cached or {}),
        "note": ("the new view is described by the patched record, but no new image was written: " + redraw_note),
        "drawn": False,
        "spec": patched,
    }


#: A path the caller hands a redraw is a table when it ends like one; anything else is a dataset.
_TABLE_SUFFIXES = (".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")


def _source_role(producer: str, path: str) -> str:
    """Which of the producer's source parameters the caller's one path fills.

    The redraw passed it POSITIONALLY, so for ``plot_spatial_statistics(data_path, *, results_path)``
    an autocorrelation table landed in data_path, and for ``plot_pathway_results(results_path, *,
    data_path)`` a dataset landed in results_path -- update and export could never redraw a
    results-table figure (hunt 2026-09-30, u20a-viz-pipelines-9).
    """
    accepted = _signature_of(producer)
    is_table = str(path).lower().endswith(_TABLE_SUFFIXES)
    if "results_path" in accepted and (is_table or "data_path" not in accepted):
        return "results_path"
    return "data_path"


def _redraw_call(
    producer: str,
    params: dict[str, Any],
    source_path: str,
    sources: dict[str, str] | None = None,
    *,
    how: str = "",
) -> tuple[dict[str, Any] | None, str]:
    """``(keyword arguments for the producer, "")`` or ``(None, why it cannot be redrawn)``.

    The record keeps every input file by NAME (the sidecar is served raw), so a recorded source the
    caller did not hand back is refused by name rather than dropped: dropping proportions_csv made a
    deconvolution redraw fall back to obsm and show another result under the same lineage (hunt
    2026-09-30, u20a-viz-pipelines-10). ``sources`` are full paths the caller handed back by
    parameter name; ``how`` says how a missing one can be handed back to the tool asking.
    """
    accepted = _signature_of(producer)
    call = {k: v for k, v in params.items() if k in accepted and k not in _PATH_PARAMS}
    if source_path:
        call[_source_role(producer, source_path)] = source_path
    call.update({k: v for k, v in (sources or {}).items() if k in accepted})
    for key in _PATH_PARAMS:
        if key in accepted and params.get(key) and not call.get(key):
            return None, (
                f"this figure was drawn from {key}={Path(str(params[key])).name!r} as well, and the record "
                "keeps it by name only; " + (how or f"pass its full path in set_params as {key}=<path> to redraw it")
            )
    return call, ""


def _next_version_slug(figures_dir: Path, stem: str, *also: Path) -> tuple[str, int]:
    """``(slug, version)`` for the next version of *stem* that is on disk in none of the directories.

    *figures_dir* is the source record's; *also* names where the redraw will be written. Looking only
    in the source's directory, and for the untruncated root, let two edits redirected to another
    output_dir -- or two edits of a figure whose slug is 46 characters or more, which ``stem_for``
    cuts to fit the suffix -- land on one ``-v2`` file, the second over the first (hunt 2026-09-30,
    u20a-viz-pipelines-7).
    """
    base = re.sub(r"-[0-9a-f]{8}$", "", stem)
    match = re.search(r"-v(\d+)$", base)
    current = int(match.group(1)) if match else 1
    root = base[: match.start()] if match else base

    def saved_root(version: int) -> str:
        # stem_for keeps 48 characters of a slug; a suffix cut off there would put the new version
        # back on the old file.
        return root[: 48 - len(f"-v{version}")]

    taken = [current]
    pattern = re.compile(r"(.+)-v(\d+)-[0-9a-f]{8}\.figspec\.json$")
    for directory in dict.fromkeys(Path(d) for d in (figures_dir, *also) if d):
        for path in directory.glob("*.figspec.json"):
            found = pattern.match(path.name)
            if found and found.group(1) == saved_root(int(found.group(2))):
                taken.append(int(found.group(2)))
    version = max(taken) + 1
    return f"{saved_root(version)}-v{version}", version


def _record_lineage(
    drawn: dict[str, Any],
    record: dict[str, Any],
    changed: list[str],
    patched: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Write the version's lineage into its own record, which the producer saved as a first draw.

    ``(the saved record, "")``, or ``(None, why it could not be written)``. The payload named
    derived_from and the record on disk did not, so the lineage was never kept (hunt 2026-09-30,
    u20a-viz-pipelines-7). The patched parameters the producer does not record itself -- a point
    size, a title -- are kept too: they were marked 'revision' and left out, so an edit OF the edit
    redrew without them and dropped the earlier change silently; and a failed write was swallowed
    while the payload still claimed the lineage (the review of that fix).
    """
    try:
        target_dir = _manifest_io.figures_dir_for(drawn["output_dir"])
        saved = _spec.load(target_dir / Path(str(drawn["figure"])).name)
        saved["derived_from"] = record.get("figure_id", "")
        params = dict(saved.get("params") or {})
        for key, value in ((patched or {}).get("params") or {}).items():
            params.setdefault(key, value)
        saved["params"] = params
        origin = dict(saved.get("param_origin") or {})
        origin.update(dict.fromkeys(changed, "revision"))
        saved["param_origin"] = origin
        _spec.save(saved, target_dir)
        return saved, ""
    except Exception as exc:
        return None, f"the new version's record could not be updated with its lineage ({type(exc).__name__}: {exc})"


def _signature_of(function_name: str) -> set[str]:
    """The parameter names a producer accepts, so a patched record cannot pass it an unknown one."""
    import inspect

    try:
        return set(inspect.signature(globals()[function_name]).parameters)
    except Exception:
        return set()


#: What a contact sheet can place: the formats PIL reads.
_RASTER_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp")


def compose_figure(
    figure_specs: str,
    *,
    output_dir: str = "",
    ncols: int = 2,
    title: str = "",
    figure_format: str = "png",
    dpi: int = 0,
    figure_id: str = "",
) -> dict[str, Any]:
    """Place several finished figures on one sheet, as an index to them.

    This is a contact sheet, and it is captioned as one. Each panel is the image of a figure that
    was drawn separately, on its own colour scale, from its own selection -- so nothing here may be
    compared panel to panel, and the caption says that rather than letting the shared canvas imply
    otherwise. The originals stay on disk at full resolution and each panel names its own.

    A publication figure whose panels DO share a scale is a different thing, and the honest way to
    get one is to draw it as one figure: ``plot_spatial_expression`` with several genes and
    ``share_scale`` puts them on one canvas with one colour bar, computed together.
    """
    paths = [s.strip() for s in str(figure_specs).split(",") if s.strip()]
    if len(paths) < 2:
        return _refusal(
            "invalid_request",
            "name at least two figure records to compose, comma separated",
            plot_id="",
            fix="pass the .figspec.json paths of the figures to place on the sheet",
        )

    panels: list[dict[str, Any]] = []
    problems: list[str] = []
    for item in paths:
        try:
            record = _spec.load(item)
        except Exception as exc:
            problems.append(f"{Path(item).name}: {exc}")
            continue
        image = Path(item).parent / Path(str(record.get("outputs", {}).get("figure", ""))).name
        if not image.is_file():
            problems.append(f"{Path(item).name}: its figure is not beside its record")
            continue
        if image.suffix.lower() not in _RASTER_SUFFIXES:
            # The sheet places images, and an SVG or PDF is not one PIL can read -- the error escaped
            # as a crash, on everything depth='publication' draws (hunt 2026-09-30,
            # u20a-viz-pipelines-21). Named, with the call that makes a placeable copy.
            problems.append(
                f"{image.name}: a {image.suffix.lstrip('.')} figure cannot be placed on a sheet; "
                "export it with export_visualization(export_format='png') and compose that record"
            )
            continue
        panels.append({"record": record, "image": image})
    if not panels:
        return _refusal(
            "invalid_request",
            "none of the named records could be read with their figures: " + "; ".join(problems[:4]),
            plot_id="",
        )

    import matplotlib.image as mpimg

    images = []
    for entry in panels:
        try:
            images.append((entry, mpimg.imread(entry["image"])))
        except Exception as exc:
            problems.append(f"{entry['image'].name}: could not be read as an image ({type(exc).__name__})")
    if not images:
        return _refusal(
            "invalid_request",
            "none of the named figures could be read as an image: " + "; ".join(problems[:4]),
            plot_id="",
        )
    panels = [entry for entry, _ in images]

    fmt = (figure_format or "png").lower()
    rows, cols, shown = _base.panel_grid(len(panels), ncols)
    fig, axes = _base.canvas(rows, cols, panel_w=5.6, panel_h=4.6, dpi=dpi or None)
    letters = "abcdefghijklmnopqrstuvwxyz"
    named: list[str] = []
    for i in range(shown):
        entry = panels[i]
        axes[i].imshow(images[i][1])
        axes[i].set_axis_off()
        label = str(entry["record"].get("plot_id", "")) or entry["image"].stem
        axes[i].set_title(f"({letters[i]}) {label}", fontsize=10, loc="left")
        named.append(f"({letters[i]}) {entry['image'].name}")
    for j in range(shown, len(axes)):
        _base.blank(axes[j])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    warnings = list(problems)
    if shown < len(panels):
        warnings.append(f"{len(panels) - shown} figure(s) did not fit the panel cap and are not on the sheet")
    limitations = [
        "a contact sheet: each panel is an image of a figure drawn separately, on its own colour "
        "scale and from its own selection, so panels are NOT comparable with one another",
        "the originals are unchanged and remain at full resolution; "
        + ", ".join(named[:4])
        + (f", and {len(named) - 4} more" if len(named) > 4 else ""),
    ]
    if fmt != "png":
        limitations.append(
            "the panels are raster images placed on a vector canvas, so this file is not a vector figure"
        )
    out_dir = Path(output_dir or Path(paths[0]).parent.parent.parent).resolve()
    figures_dir = _manifest_io.figures_dir_for(out_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    subject = title or f"{shown} figures on one sheet"
    caption = _spec.compose_caption(subject=subject, expression=None, limitations=limitations, claim="descriptive")
    data_params = {"figure_specs": ",".join(Path(p).name for p in paths), "ncols": ncols}
    # Which figures, in which order, reach the stem: neither key is in DATA_PARAMS, so every sheet
    # in one output directory was contact_sheet-<one hash> and each new sheet replaced the last
    # (hunt 2026-09-30, u20a-viz-pipelines-6).
    composed = [f"{entry['record'].get('stem', '')}@{entry['record'].get('revision', 1)}" for entry in panels[:shown]]
    stem = _spec.stem_for(
        "composed.sheet", _with_selectors("", panels=composed), data_params, slug=figure_id or "contact_sheet"
    )
    return _finalize(
        fig=fig,
        out_dir=out_dir,
        figures_dir=figures_dir,
        plot_id="composed.sheet",
        function="compose_figure",
        kind="grid",
        stem=stem,
        fmt=fmt,
        dpi=dpi,
        title=subject,
        caption=caption,
        limitations=limitations,
        warnings=warnings,
        source={"name": "several figures", "read_from": ", ".join(Path(p).name for p in paths[:6])},
        params={**data_params, "figure_format": fmt},
        param_origin={"figure_specs": "caller"},
        expression=None,
        cache=None,
        findings=[("n_panels", shown, "figures on the sheet")],
    )


def export_visualization(
    figure_spec: str,
    *,
    output_dir: str = "",
    data_path: str = "",
    export_format: str = "svg",
    dpi: int = 0,
    include_values: bool = True,
    bundle: bool = False,
) -> dict[str, Any]:
    """Re-render a figure for publication, and ship the numbers behind it.

    The vector path is SVG with editable text, which is what a journal wants and what the portal
    can serve inline. PDF is produced on request and offered as a download rather than a preview,
    because there is no viewer for it here. A point layer above the rasterisation threshold is
    rasterised inside the vector file while the axes, text and legend stay vector -- measured, that
    is the difference between a 16.9 MB file and a 3.66 MB one for the same 120,000 points.

    ``include_values`` writes the plotted numbers beside the figure. A figure without them is a
    picture; with them it is a result somebody else can check.
    """
    import csv
    import os
    import zipfile

    try:
        record = _spec.load(figure_spec)
    except FileNotFoundError as exc:
        return _refusal("not_found", str(exc), plot_id="")
    fmt = (export_format or "svg").lower()
    if fmt not in ("svg", "png", "pdf"):
        return _refusal(
            "invalid_request",
            f"{fmt!r} is not an export format here; choose 'svg', 'png' or 'pdf'",
            plot_id=record.get("plot_id", ""),
        )

    figures_dir = Path(figure_spec).parent
    out_dir = Path(output_dir).resolve() if output_dir else figures_dir.parent.parent
    warnings: list[str] = []
    written: dict[str, str] = {}

    producer = _PRODUCERS.get(str(record.get("function") or ""))
    source_path = data_path or ""
    if producer and source_path:
        # By keyword and by role, as update_visualization does (hunt 2026-09-30, u20a-viz-pipelines-9).
        call, why_not = _redraw_call(
            producer,
            record.get("params") or {},
            source_path,
            how=(
                f"export takes one source, so call {record.get('function')} again with both full paths "
                f"and figure_format={fmt!r}"
            ),
        )
        if call is None:
            drawn = {"status": "error", "why": why_not}
        else:
            call.update(
                {
                    "output_dir": str(out_dir),
                    "figure_format": fmt,
                    "dpi": int(dpi) or 0,
                    "figure_id": f"{re.sub(r'-[0-9a-f]{8}$', '', str(record['stem']))}-export",
                }
            )
            try:
                drawn = globals()[producer](**{k: v for k, v in call.items() if k in _signature_of(producer)})
            except Exception as exc:
                drawn = {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
        if drawn.get("status") == "ok":
            written["figure"] = drawn["figure"]
        else:
            warnings.append(f"the figure could not be re-rendered as {fmt}: {drawn.get('why', '')}")
    else:
        existing = figures_dir / Path(str(record.get("outputs", {}).get("figure", ""))).name
        if existing.is_file():
            written["figure"] = f"figures/{existing.name}"
            if existing.suffix.lstrip(".").lower() != fmt:
                warnings.append(
                    f"the figure on disk is {existing.suffix.lstrip('.')}, not {fmt}; re-rendering needs "
                    "the source, so pass data_path to get a true vector export"
                )
        else:
            return _refusal(
                "not_found",
                "the figure named by this record is not beside it, and no source was given to redraw it",
                plot_id=record.get("plot_id", ""),
                fix="pass data_path so the figure can be drawn again",
            )

    if include_values:
        cached = _spec.read_cache(figures_dir, record["stem"])
        if cached:
            tables_dir = _manifest_io.tables_dir_for(out_dir)
            tables_dir.mkdir(parents=True, exist_ok=True)
            # One table per row count, each 2-D array spread into named columns. Ravelling a
            # coordinate matrix into one column beside n-length values paired spot 0's y with spot
            # 1's value, 2n rows for n spots (hunt 2026-09-30, u20a-viz-pipelines-13).
            groups = _value_tables(cached)
            main = max(groups, key=lambda n: (len(groups[n]), n)) if groups else 0
            for n_rows, columns in sorted(groups.items()):
                label = "_".join(sorted({c.split("[")[0] for c in columns}))
                suffix = "" if n_rows == main else "_" + re.sub(r"[^A-Za-z0-9_-]", "_", label)[:40]
                values_path = tables_dir / f"{record['stem']}_values{suffix}.csv"
                partial = values_path.with_name(values_path.name + ".partial")
                with partial.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(list(columns))
                    for i in range(n_rows):
                        writer.writerow([columns[c][i] for c in columns])
                os.replace(partial, values_path)
                written["values" if n_rows == main else f"values{suffix}"] = f"tables/{values_path.name}"
        else:
            warnings.append("this figure cached no values, so none could be exported beside it")

    spec_name = Path(_spec.spec_path(figures_dir, record["stem"])).name
    written["record"] = f"figures/{spec_name}"

    if bundle:
        archive = out_dir / f"{record['stem']}_export.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for label, rel in written.items():
                candidate = _manifest_io.results_dir_for(out_dir) / rel
                if candidate.is_file():
                    zf.write(candidate, arcname=f"{record['stem']}/{Path(rel).name}")
                elif label == "figure" and (figures_dir / Path(rel).name).is_file():
                    zf.write(figures_dir / Path(rel).name, arcname=f"{record['stem']}/{Path(rel).name}")
        written["bundle"] = archive.name

    return {
        "status": "ok",
        "plot_id": record.get("plot_id", ""),
        "figure_id": record.get("figure_id", ""),
        "export_format": fmt,
        "files": written,
        "output_dir": str(out_dir),
        "warnings": warnings,
        "caption": record.get("caption", ""),
        "note": (
            "SVG keeps text editable and is safe to serve inline; PDF is a download because there is "
            "no viewer here. The values beside the figure are the numbers it was drawn from."
        ),
    }


def _value_tables(cached: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """``{row count: {column name: values}}`` -- every cached array as columns of equal length.

    A two-column ``coords``/``xy`` becomes ``coords_x``/``coords_y`` (a third, ``_z``); any other
    matrix becomes one column per column index. Arrays of different lengths -- a matrix of group
    means beside per-spot values -- go to different tables rather than being padded into one.
    """
    import numpy as np

    groups: dict[int, dict[str, Any]] = {}
    for key, value in cached.items():
        array = np.asarray(value)
        if array.size == 0:
            continue
        if array.ndim <= 1:
            groups.setdefault(int(array.size), {})[key] = array.ravel()
            continue
        array = array.reshape(array.shape[0], -1)
        names = (
            [f"{key}_{axis}" for axis in "xyz"[: array.shape[1]]]
            if key in ("coords", "xy", "xyz") and array.shape[1] <= 3
            else [f"{key}[{j}]" for j in range(array.shape[1])]
        )
        block = groups.setdefault(int(array.shape[0]), {})
        for j, name in enumerate(names):
            block[name] = array[:, j]
    return groups


#: How many figures each depth draws. Small on purpose: a report that draws everything it could
#: is not a report, and the four-figure chat card means the rest is scrolled past anyway.
_DEPTHS: dict[str, int] = {"overview": 3, "detailed": 6, "focused": 2, "publication": 4}

#: A plot id is not a function call. Several families share one producer and differ only by `kind`,
#: and dropping that turns "draw the composition" into "draw a dot plot with no genes" -- which
#: fails, and reads as the family being broken rather than as the caller being lost.
_PLOT_CALLS: dict[str, dict[str, Any]] = {
    "markers.dotplot": {"kind": "dotplot"},
    "markers.violin": {"kind": "violin"},
    "markers.heatmap": {"kind": "heatmap"},
    "composition.stacked": {"kind": "composition"},
    "de.volcano": {"kind": "volcano"},
    "de.ranked": {"kind": "ranked"},
    "deconv.proportions": {"kind": "maps"},
    "deconv.dominant": {"kind": "dominant"},
    "deconv.composition": {"kind": "summary"},
    "organisation.graph": {"kind": "graph"},
    "organisation.neighborhood": {"kind": "neighborhood"},
    "organisation.autocorrelation": {"kind": "autocorrelation"},
    "organisation.cooccurrence": {"kind": "cooccurrence"},
    "pathway.enrichment": {"kind": "enrichment"},
    "pathway.activity_map": {"kind": "activity"},
    "trajectory.pseudotime": {"kind": "pseudotime"},
    "trajectory.spatial_pseudotime": {"kind": "spatial"},
    "trajectory.gene_trend": {"kind": "gene_trend"},
    "communication.interactions": {"kind": "interactions"},
    # One producer, three figures. `view` is this family's `kind`: without it an unattended
    # run asking for a depth profile gets the point cloud, which is a different claim.
    "spatial3d.scatter": {"view": "scatter"},
    "spatial3d.depth_profile": {"view": "depth"},
    "spatial3d.axis_profile": {"view": "axis"},
    # Without an entry the producer ran its default view, so a static scatter with no rotatable
    # spec was reported as the interactive volume (hunt 2026-09-30, u20a-viz-pipelines-28).
    "interactive.volume": {"view": "interactive"},
    # No 'de.heatmap' entry: the catalogue declares that row unsupported (its figure is
    # markers.heatmap), so no run reaches it, and an entry here only read as a call that exists
    # (hunt 2026-09-30, the review of u20a-viz-pipelines-28).
    "alignment.before_after": {"mode": "before_after"},
    "alignment.pair_overlay": {"mode": "overlay"},
}

#: Families an unattended run must not guess for. Each needs the caller to name something whose
#: choice changes what the figure MEANS -- which genes, which result file -- and a pipeline that
#: picked one silently would be answering a question nobody asked. They are reported as skipped
#: with the reason, not as failures, so the caller can ask for them by name.
_NEEDS_A_CHOICE: dict[str, str] = {
    "spatial.expression": "name the genes, obs column or score column to paint",
    "spatial.sections": "name what to paint, and the sections to compare",
    "spatial.histology_overlay": "name the genes or annotation to show over the image",
    "embedding.facet": "name the column to split the embedding by",
    "trajectory.gene_trend": "name the genes whose trend along the ordering to draw",
    "pathway.enrichment": "pass the enrichment table another tool wrote",
    "organisation.autocorrelation": "pass the autocorrelation table another tool wrote",
    "organisation.cooccurrence": "pass the co-occurrence table another tool wrote",
    "communication.interactions": "pass the cell-communication table another tool wrote",
    "spatial3d.axis_profile": "name the gene or numeric obs column whose gradient to draw",
}


def run_visualization_pipeline(
    data_path: str,
    *,
    output_dir: str = "",
    depth: str = "overview",
    question: str = "",
    figure_format: str = "",
    dpi: int = 0,
) -> dict[str, Any]:
    """Inspect the dataset, decide what is worth drawing, draw that, and report what was left out.

    It starts from the recommendation, not from the catalogue. The catalogue is what COULD be drawn;
    a run that drew all of it would produce dozens of near-identical panels and bury the two that
    mattered. So the depth is a budget, the budget is small, and everything the budget excluded is
    named in the reply so the caller can ask for it by name.
    """
    budget = _DEPTHS.get(depth)
    if budget is None:
        return _refusal(
            "invalid_request",
            f"{depth!r} is not a depth; choose one of {sorted(_DEPTHS)}",
            plot_id="",
        )
    prof = _profile.profile_for_viz(data_path)
    if not prof["dataset"]["readable"]:
        return _refusal("unreadable", prof["dataset"]["read_error"], plot_id="")

    suggestions = _caps.recommend(prof, limit=budget * 3)
    if not suggestions:
        return {
            "status": "ok",
            "drew": [],
            "skipped": [],
            "why_nothing": (
                "nothing in the catalogue can be drawn from this dataset as it stands; "
                "call inspect_dataset to see what is missing"
            ),
            "output_dir": str(Path(output_dir or ".").resolve()),
        }

    suggestions = _order_by_question(suggestions, question)

    fmt = figure_format or ("svg" if depth == "publication" else "png")
    drew: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    needs_a_choice: list[dict[str, str]] = []
    seen_functions: set[str] = set()
    for row in suggestions:
        if len(drew) >= budget:
            break
        producer = _PRODUCERS.get(row["function"])
        if producer is None:
            continue
        if row["plot_id"] in _NEEDS_A_CHOICE:
            needs_a_choice.append({"plot_id": row["plot_id"], "ask": _NEEDS_A_CHOICE[row["plot_id"]]})
            continue
        # One figure per family, so a depth of three is three different views rather than three
        # spellings of the same one.
        if row["function"] in seen_functions:
            continue
        call = {"output_dir": output_dir, "figure_format": fmt, "dpi": dpi}
        call.update(_PLOT_CALLS.get(row["plot_id"], {}))
        accepted = _signature_of(producer)
        try:
            result = globals()[producer](data_path, **{k: v for k, v in call.items() if k in accepted})
        except Exception as exc:
            result = {"status": "error", "why": f"{type(exc).__name__}: {exc}"}
        if result.get("status") == "ok":
            seen_functions.add(row["function"])
            drew.append(
                {
                    # What was drawn, not what was asked for: the two differ when a producer falls
                    # back to its own default view (hunt 2026-09-30, u20a-viz-pipelines-28).
                    "plot_id": result.get("plot_id") or row["plot_id"],
                    "asked_for": row["plot_id"],
                    "figure": result["figure"],
                    "figure_id": result["figure_id"],
                    "caption": result["caption"],
                }
            )
        else:
            failed.append({"plot_id": row["plot_id"], "why": result.get("why") or result.get("error", "")})

    considered = {row["plot_id"] for row in suggestions}
    drawn_ids = {row["asked_for"] for row in drew}
    return {
        "status": "ok" if drew else "refused",
        "depth": depth,
        "budget": budget,
        "drew": drew,
        "skipped": sorted(considered - drawn_ids - {f["plot_id"] for f in failed}),
        "needs_a_choice": needs_a_choice,
        "failed": failed,
        "output_dir": str(Path(output_dir or ".").resolve()),
        "note": (
            f"{len(drew)} figure(s) drawn against a budget of {budget}; the rest were considered and "
            "not drawn, and are listed under 'skipped' so they can be asked for by name. "
            "This ran no analysis: every figure is drawn from what the dataset already holds."
        ),
    }


def _kv(text: str) -> dict[str, Any]:
    """``key=value,key=value`` into a dict, with numbers and booleans recognised.

    A comma-separated piece with no ``=`` continues the previous value, so ``genes=CD3E,CD8A`` is
    two genes and ``title=A, B`` is one title. Splitting on every comma dropped the second gene and
    half the title silently (hunt 2026-09-30, u20a-viz-pipelines-33).
    """
    pairs: list[list[str]] = []
    for chunk in str(text).split(","):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            pairs.append([key.strip(), value])
        elif pairs:
            pairs[-1][1] += "," + chunk
    out: dict[str, Any] = {}
    for key, raw in pairs:
        value = raw.strip()
        if not key:
            continue
        if value.lower() in ("true", "false"):
            out[key] = value.lower() == "true"
            continue
        try:
            out[key] = int(value) if value.isdigit() else float(value)
        except ValueError:
            out[key] = value
    return out
