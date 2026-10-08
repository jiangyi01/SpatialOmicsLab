"""A figure's own record of how it was drawn, and the thing that makes it revisable.

Every figure this toolkit writes has a ``.figspec.json`` beside it. It carries the dataset's
identity and a change detector for it, every resolved parameter including the ones that were
inferred rather than asked for, which matrix slot the values came from, the seed, the package
versions, and the paths of any upstream result it read. Two things follow from that, and they are
the reason the file exists rather than a nicety.

**A figure can be changed without redoing the analysis.** Parameters fall into three classes. A
*style* or *layout* change -- a palette, a colour limit, a point size, a panel count, an export
format -- is served from the small table of values that were actually plotted, written beside the
figure at draw time. The dataset is never reopened, so the change works in a later turn, after a
restart, in a session that never ran the original analysis, and on a box where the source has
since moved. Only a *data* change -- different genes, a different column, a different layer or
normalisation -- reads the dataset again, and even then it reruns no clustering, no
deconvolution and no differential expression.

**A figure can be reproduced.** The spec is the complete argument list. Handing it back produces
the same picture, and the recorded package versions say what to install to get the same picture
somewhere else.

Two constraints on identifiers that look arbitrary and are not. The figure id is readable --
``<stem>@v<n>`` -- and the dataset fingerprint is sixteen hexadecimal characters, because the
portal's redactor masks any hexadecimal run of thirty-two or more on its way to the reader. A
sha256 id would arrive as ``[redacted]`` and the revision tool would then report that the figure
does not exist.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

SCHEMA = "sog.figure_spec/1"

#: Parameters that change *which values are plotted*. Changing one of these needs the dataset.
#: Everything not listed here is presentation, and presentation is served from the cached table.
DATA_PARAMS: frozenset[str] = frozenset(
    {
        "data_path",
        "genes",
        "obs_key",
        "obsm_key",
        "color",
        "groupby",
        "basis",
        "layer",
        "use_raw",
        "normalize",
        "library_id",
        "crop",
        "top_n",
        "group",
        "result_path",
        "seed",
        "compute_if_missing",
        "n_neighbors",
        "n_pcs",
        # The 3D families' selectors, added 2026-09-22. Each of these changes WHICH VALUES are
        # drawn, so each has to reach the stem: without them `--pair S0,S1` and `--pair S2,S3`
        # hashed identically and the second figure silently overwrote the first, under a filename
        # that named neither. They also classify a change as CLASS_DATA in `update_visualization`,
        # which is correct -- none of them can be served from the frozen table.
        "view",
        "mode",
        "axis",
        "n_bins",
        "coords_key",
        "section_key",
        "before_key",
        "after_key",
        "pair",
        "z_spacing",
        # Every other producer's selectors, added 2026-09-30. Only the 3D families had been
        # audited, so two communication tables, two pathway sets, two pseudotime columns or two
        # kinds of one family hashed to one stem and the second figure overwrote the first; and a
        # change to any of them classified as STYLE, a revision of the same picture (hunt
        # 2026-09-30, u20b-viz-rest-11). A test now fails on any producer parameter in no set.
        "results_path",
        "proportions_csv",
        "kind",
        "source_column",
        "target_column",
        "score_column",
        "fc_threshold",
        "p_threshold",
        "fdr_threshold",
        "standardize",
        "split_by",
        "split_panels",
        "collection",
        "method",
        "pathways",
        "pseudotime_key",
        "on_tissue",
        "figure_specs",
        # The p-value matrix a communication heatmap marks its cells from (2026-10-05).
        "pvalues_path",
    }
)

#: Presentation. A change to one of these is free after the first draw.
STYLE_PARAMS: frozenset[str] = frozenset(
    {
        "title",
        "palette",
        "color_map",
        "vmin",
        "vmax",
        "point_size",
        "spot_alpha",
        "image_alpha",
        "image_key",
        "legend_loc",
        "share_scale",
        "flip_y",
        "annotate",
        "show_labels",
    }
)

#: Canvas. Also free after the first draw.
LAYOUT_PARAMS: frozenset[str] = frozenset({"ncols", "width_in", "height_in", "figure_format", "dpi"})

CLASS_DATA = "data"
CLASS_STYLE = "style"
CLASS_LAYOUT = "layout"

_SPEC_SUFFIX = ".figspec.json"
_CACHE_SUFFIX = ".figdata.npz"

#: The cached table of plotted values is what makes a later cosmetic edit free. Above this it is
#: not written, and the spec says so, so a later edit knows it will need the source rather than
#: discovering it.
CACHE_MAX_BYTES = 32 * 1024 * 1024


def figure_id(stem: str, revision: int = 1) -> str:
    """``<stem>@v<n>`` -- readable on purpose. See this module's docstring."""
    return f"{stem}@v{int(revision)}"


def parse_figure_id(value: str) -> tuple[str, int]:
    """The inverse. A bare stem is revision one."""
    text = str(value or "").strip()
    if "@v" in text:
        stem, _, rev = text.rpartition("@v")
        try:
            return stem, int(rev)
        except ValueError:
            return text, 1
    return text, 1


def stem_for(plot_id: str, source_fingerprint: str, data_params: dict[str, Any], slug: str = "") -> str:
    """A deterministic stem, so re-running the same request overwrites rather than accumulates.

    Two identical requests against an unchanged dataset produce one file, not a directory of
    near-duplicates that a reader has to date-sort. A change to anything that selects values
    produces a different stem, because it is a different figure and both should be able to exist.
    """
    import hashlib

    if slug:
        base = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in slug)[:48]
    else:
        base = plot_id.replace(".", "_")
    payload = json.dumps(
        {"plot": plot_id, "source": source_fingerprint, "params": _canonical(data_params)},
        sort_keys=True,
        default=str,
    )
    return f"{base}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]}"


def _canonical(params: dict[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in sorted(params) if k in DATA_PARAMS and params[k] not in (None, "", [])}


def classify(changed: Iterable[str]) -> str:
    """The strongest class among the changed keys: data beats layout beats style."""
    keys = {str(k) for k in changed}
    if keys & DATA_PARAMS:
        return CLASS_DATA
    if keys & LAYOUT_PARAMS:
        return CLASS_LAYOUT
    return CLASS_STYLE


def new_spec(
    *,
    plot_id: str,
    function: str,
    kind: str,
    stem: str,
    revision: int,
    source: dict[str, Any],
    params: dict[str, Any],
    param_origin: dict[str, str],
    expression: dict[str, Any] | None,
    caption: str,
    limitations: list[str],
    outputs: dict[str, str],
    warnings: list[str] | None = None,
    derived_from: str = "",
) -> dict[str, Any]:
    """Assemble the record. Every field is filled; a reader never needs ``.get``."""
    return {
        "schema": SCHEMA,
        "figure_id": figure_id(stem, revision),
        "stem": stem,
        "revision": int(revision),
        "derived_from": derived_from,
        "plot_id": plot_id,
        "function": function,
        "kind": kind,
        "source": source,
        "params": dict(params),
        "param_origin": dict(param_origin),
        "expression": expression,
        "caption": caption,
        "limitations": list(limitations),
        "outputs": dict(outputs),
        "warnings": list(warnings or []),
        "environment": _environment(),
    }


def _environment() -> dict[str, Any]:
    """What drew it. Recorded so the same picture can be produced somewhere else."""
    import sys

    versions: dict[str, str] = {}
    for name in ("matplotlib", "numpy", "pandas", "scipy", "anndata", "scanpy", "seaborn"):
        try:
            import importlib.metadata as md

            versions[name] = md.version(name)
        except Exception:
            continue
    return {"python": ".".join(str(v) for v in sys.version_info[:3]), "packages": versions}


def spec_path(figures_dir: str | Path, stem: str, revision: int = 1) -> Path:
    name = stem if revision <= 1 else f"{stem}.r{revision}"
    return Path(figures_dir) / f"{name}{_SPEC_SUFFIX}"


def cache_path(figures_dir: str | Path, stem: str) -> Path:
    """The cache is per stem, not per revision: every revision of one figure plots the same values."""
    return Path(figures_dir) / f"{stem}{_CACHE_SUFFIX}"


def _names_only(value: Any) -> Any:
    """An absolute path becomes its file name; a comma-joined list of them, a list of names."""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        if any(os.path.isabs(p) for p in parts if p):
            return ",".join(Path(p).name if os.path.isabs(p) else p for p in parts)
        return value
    if isinstance(value, (list, tuple)):
        return [_names_only(v) for v in value]
    return value


def save(spec: dict[str, Any], figures_dir: str | Path) -> Path:
    """Write the sidecar atomically beside the figure.

    No absolute path and no long hexadecimal string may go in here: the sidecar is downloadable
    through the results route, which serves raw bytes with no redaction, so an absolute path in
    it would be a served byte that names this machine's directory layout.

    The rule is enforced here, not trusted to each family: seven producers recorded the absolute
    data_path, results_path or result_path, and a shipped smoke sidecar carried the repository's
    own path (hunt 2026-09-30, u20b-viz-rest-28).
    """
    spec = dict(spec)
    for section in ("params", "source"):
        if isinstance(spec.get(section), dict):
            spec[section] = {k: _names_only(v) for k, v in spec[section].items()}
    target = spec_path(figures_dir, spec["stem"], int(spec.get("revision", 1)))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    tmp.write_text(json.dumps(spec, indent=2, sort_keys=False, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load(reference: str | Path) -> dict[str, Any]:
    """Load a spec from its own path, or from the path of the figure it describes.

    Raises ``FileNotFoundError`` with a sentence rather than a bare path, because this is reached
    by an agent following a figure id from an earlier turn and "no such file" is not actionable.
    """
    ref = Path(reference)
    if ref.suffix == ".json" and ref.name.endswith(_SPEC_SUFFIX):
        candidate = ref
    else:
        candidate = ref.with_suffix("")
        candidate = ref.parent / f"{ref.stem}{_SPEC_SUFFIX}"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"no figure record beside {ref.name!r}. Every figure this toolkit draws writes one; "
            "a figure from another tool has none, and cannot be revised without redrawing it."
        )
    return json.loads(candidate.read_text(encoding="utf-8"))


def patch(
    spec: dict[str, Any], set_params: dict[str, Any], unset: Iterable[str] = ()
) -> tuple[dict[str, Any], str, list[str]]:
    """Apply changes to a loaded spec. Returns the new spec, its change class, and what changed.

    The revision is bumped for a presentation change, because it is the same figure seen
    differently. A data change is a *different* figure and gets a new stem, which the caller
    computes -- this function reports the class and leaves that decision where the dataset is.
    """
    changed: list[str] = []
    params = dict(spec.get("params") or {})
    origin = dict(spec.get("param_origin") or {})
    for key, value in (set_params or {}).items():
        if params.get(key) != value:
            params[key] = value
            origin[key] = "revision"
            changed.append(key)
    for key in unset or ():
        if key in params:
            params.pop(key)
            origin.pop(key, None)
            changed.append(key)
    klass = classify(changed)
    out = dict(spec)
    out["params"] = params
    out["param_origin"] = origin
    if klass in (CLASS_STYLE, CLASS_LAYOUT):
        out["revision"] = int(spec.get("revision", 1)) + 1
        out["derived_from"] = spec.get("figure_id", "")
        out["figure_id"] = figure_id(spec["stem"], out["revision"])
    return out, klass, changed


def write_cache(figures_dir: str | Path, stem: str, arrays: dict[str, Any]) -> tuple[str, str]:
    """Freeze the plotted values. Returns ``(relative path, why not)`` -- one of them is empty.

    This is what makes a later cosmetic edit free, and what lets a figure still be recoloured
    after its dataset has moved on. Above the cap it is skipped and the reason is recorded, so a
    later edit is told it will need the source rather than finding out.
    """
    import numpy as np

    target = cache_path(figures_dir, stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        estimated = sum(int(getattr(np.asarray(v), "nbytes", 0)) for v in arrays.values())
    except Exception:
        estimated = 0
    if estimated > CACHE_MAX_BYTES:
        return (
            "",
            f"the plotted values are {estimated / 1e6:.0f} MB, above the {CACHE_MAX_BYTES / 1e6:.0f} MB cache cap; a later change of appearance will reopen the dataset",
        )
    try:
        np.savez_compressed(target, **arrays)
    except Exception as exc:
        return ("", f"the plotted values could not be cached ({type(exc).__name__})")
    return (target.name, "")


#: The interactive spec's schema tag. It is in the file and the reader checks it, so a JSON that
#: happens to have a ``traces`` key is not mistaken for one of ours.
FIG3D_SCHEMA = "spatialomicsgym.fig3d/1"

#: Above this the spec is not written, and the reason is recorded. A browser given twelve million
#: numbers does not draw a slow figure, it stops responding -- and the PNG beside it is still there,
#: which is the whole reason the interactive view is an addition rather than a replacement.
FIG3D_MAX_POINTS = 60_000

#: The only colour scales a spec may name. A free string here would reach plotly's own parser, and
#: the point of rebuilding from an allowlist is that nothing in the file is handed onward unread.
FIG3D_COLORSCALES = ("Viridis", "Cividis", "Plasma", "Magma", "Greys", "Blues", "Reds", "Turbo")


def fig3d_path(figures_dir: str | Path, stem: str) -> Path:
    return Path(figures_dir) / f"{stem}.fig3d.json"


def write_fig3d(figures_dir: str | Path, stem: str, spec: dict[str, Any]) -> tuple[str, str]:
    """Write the data-only 3D spec beside the figure. ``(relative name, why not)``.

    A tool writes **numbers and a handful of enumerated strings**, never markup and never script.
    The portal rebuilds a plotly document from them against its own allowlist; this side's job is
    only to produce a file that survives that rebuild, and to refuse rather than write one that
    would make a browser stop responding.
    """
    import json

    total = 0
    for trace in spec.get("traces") or []:
        total += len(trace.get("x") or [])
    if total > FIG3D_MAX_POINTS:
        return ("", f"the interactive view was not written: {total:,} points is above the {FIG3D_MAX_POINTS:,} cap")
    target = fig3d_path(figures_dir, stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(spec)
    payload["schema"] = FIG3D_SCHEMA
    try:
        target.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    except Exception as exc:
        return ("", f"the interactive view could not be written ({type(exc).__name__})")
    return (target.name, "")


def read_cache(figures_dir: str | Path, stem: str) -> dict[str, Any] | None:
    import numpy as np

    target = cache_path(figures_dir, stem)
    if not target.is_file():
        return None
    try:
        with np.load(target, allow_pickle=False) as handle:
            return {k: handle[k] for k in handle.files}
    except Exception:
        return None


def compose_caption(
    *,
    subject: str,
    expression: dict[str, Any] | None,
    scale_note: str = "",
    sampling_note: str = "",
    limitations: Iterable[str] = (),
    inference: str = "",
    claim: str = "descriptive",
) -> str:
    """Build the caption from the record, rather than letting each family write its own.

    One function, so a lens can assert that no drawing family composes a caption string itself.
    That matters because the renderer's honesty is only as good as the sentence under the
    picture, and a family that writes its own will eventually leave out the one clause that
    mattered -- which slot the values came from, or that the panels are not comparable.

    The clauses are ordered by how badly a reader is misled without them: what is being shown,
    then what would make the picture mean something other than it appears to, then the caveats.
    """
    parts: list[str] = [subject.strip().rstrip(".") + "."] if subject else []
    if inference:
        parts.append(inference.strip())
    if scale_note:
        parts.append(scale_note.strip().rstrip(".") + ".")
    if sampling_note:
        parts.append(sampling_note.strip())
    if expression:
        slot = expression.get("slot", "")
        key = expression.get("key", "")
        where = f"layers['{key}']" if slot == "layers" else (slot or "X")
        transform = expression.get("transform", "")
        shown = f"Values: {where}"
        if transform:
            shown += f", {transform}"
        if expression.get("integral") is None:
            shown += (
                "; the matrix type could not be verified from the sample, so it is treated as the caller described it"
            )
        parts.append(shown.rstrip(".") + ".")
    if claim == "descriptive":
        parts.append("Descriptive: no statistical test was run for this figure.")
    for item in limitations:
        text = str(item).strip()
        if text:
            parts.append(text.rstrip(".") + ".")
    return " ".join(parts)
