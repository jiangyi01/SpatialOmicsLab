"""
Read, validate and normalise a post-analysis ``manifest.json`` -- the contract in
``docs/design/post_analysis_contract.md`` (schema v1).

This is the **consumer** side of the contract, shared by the two L3 surfaces (the standalone HTML
report and the web portal). It is deliberately paranoid, because the same manifest is rendered into
a page a browser opens and read from a directory a web server serves:

* :func:`safe_subpath` is the ONE containment primitive. Every ``path`` in a manifest, and every
  path a browser asks for, goes through it. ``../``, absolute paths, drive letters, UNC prefixes,
  NUL bytes and symlinks that leave the tree are all refused. There is no second implementation.
* :func:`normalize` guarantees every contract key exists with the contract's type, so a consumer
  never needs ``.get()`` with a default and a truncated / hand-edited manifest degrades into an
  honest empty section rather than a traceback.
* Unknown keys are dropped, so a v2 manifest still renders here.

**The schema vocabulary is not restated here.** The contract names
``spatialomicsgym/postanalysis/manifest.py`` as the one schema module; a second copy of
``TASK_TYPES``/``FIGURE_KINDS``/``SCHEMA_VERSION`` would let the reader drift from the writer, so
they are imported. What stays in this file is the reader's own business: containment, tolerant
coercion of an on-disk file, and the errors a consumer has to handle.

Stdlib only -- ``import spatialomicsgym.report`` must succeed in the 1.6 GB minimal agent env
(``postanalysis.manifest`` is stdlib-only too, by contract non-negotiable 3).
"""

from __future__ import annotations

import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
from typing import Any

from spatialomicsgym.postanalysis import manifest as _schema
from spatialomicsgym.postanalysis.manifest import (
    DEFAULT_RESULTS_DIRNAME,
    FIGURE_KINDS,
    SCHEMA_VERSION,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    TASK_TYPES,
    declined_for_no_handler,
)

#: The contract puts ``normalize()`` in the schema module. It is not written there yet; when it is,
#: :func:`normalize` below delegates to it instead of running its own copy.
_upstream_normalize = getattr(_schema, "normalize", None)

#: The manifest filename. Written LAST and atomically by L1, so its presence marks a complete run.
MANIFEST_NAME = "manifest.json"

#: ``status`` values the contract defines. Anything else renders with a neutral badge.
STATUSES = frozenset({STATUS_OK, STATUS_PARTIAL, STATUS_FAILED})

#: ``review.verdict`` values the contract defines. The contract assigns these to the schema module;
#: they actually live in L2's ``review.py``, so read them from there and fall back to the contract
#: text if that module is mid-edit -- a missing L2 must not stop a report from rendering.
try:  # pragma: no cover - the fallback only fires on a broken/partial tree
    from spatialomicsgym.postanalysis.review import VERDICTS as _VERDICTS
except Exception:
    _VERDICTS = ("ok", "suspicious", "unusable")
VERDICTS = frozenset(_VERDICTS)

__all__ = [
    "DEFAULT_RESULTS_DIRNAME",
    "FIGURE_KINDS",
    "MANIFEST_NAME",
    "MAX_MANIFEST_BYTES",
    "SCHEMA_VERSION",
    "STATUSES",
    "TASK_TYPES",
    "VERDICTS",
    "ManifestError",
    # Imported for the same reason as ``TASK_TYPES``: L1 writes the "no handler for this task type"
    # warning and L3 has to recognise it, and a second spelling in the reader would drift the first
    # time either is re-worded -- silently, because the reader would simply stop recognising the
    # degrade and go back to calling it a contract violation.
    "declined_for_no_handler",
    "is_run_dir",
    "load",
    "normalize",
    "safe_subpath",
]

# Refuse to read a manifest larger than this. A manifest is metadata; a multi-hundred-MB one is a
# bug or an attack, and json.loads() on it would pin the whole thing in memory on a web server.
MAX_MANIFEST_BYTES = 8 * 1024 * 1024

# Cap on any single normalised string. A manifest field is written by a tool, not by us, and
# MAX_MANIFEST_BYTES leaves room for one 8 MB ``task_type``. Every consumer of this module puts
# those strings somewhere that assumes they are short -- a pill, a table cell, a PDF line, a JSON
# error body -- and no one of them can defend itself against the others' mistakes. 4096 is chosen
# so it can never truncate a real path: Linux PATH_MAX is 4096 including the NUL, so a path that
# hits this cap could not have been opened anyway. A truncated value keeps the ellipsis, so it
# reads as truncated rather than as a shorter value that happens to be wrong.
MAX_TEXT_CHARS = 4096


class ManifestError(RuntimeError):
    """A results directory has no readable, well-formed ``manifest.json``."""


# --------------------------------------------------------------------------- #
# containment -- the single security primitive
# --------------------------------------------------------------------------- #
def safe_subpath(root: Path | str, rel: Any) -> Path | None:
    """Resolve ``rel`` under ``root`` and return it **only** if it truly lives there, else ``None``.

    Refused, in order: a non-string / empty value; a NUL byte (truncates in the C layer, so a
    ``"a.png\\x00.h5ad"`` check-then-open mismatch is possible); a Windows drive letter or UNC
    prefix; a POSIX-absolute path; any ``..`` component; and -- after ``resolve()``, which follows
    symlinks -- anything whose real location is not ``root`` or a descendant of it. That last step
    is what stops a symlink planted inside a results directory from reading the rest of the disk.

    ``root`` itself is a legal answer (``rel == "."``), so a caller can validate "this directory".
    """
    if isinstance(rel, Path):
        rel = str(rel)
    if not isinstance(rel, str):
        return None
    s = rel.strip()
    if not s or "\x00" in s:
        return None
    # A backslash is a separator on Windows and an ordinary filename character on POSIX; treating it
    # as a separator here is the conservative reading (it can only ever reject more).
    s = s.replace("\\", "/")
    if s.startswith("/"):
        return None
    if ntpath.splitdrive(s)[0] or os.path.splitdrive(s)[0]:
        return None
    parts = [p for p in PurePosixPath(s).parts if p not in (".",)]
    if any(p == ".." for p in parts):
        return None
    try:
        root_resolved = Path(root).resolve()
        target = root_resolved.joinpath(*parts) if parts else root_resolved
        resolved = target.resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    if resolved != root_resolved and root_resolved not in resolved.parents:
        return None
    return resolved


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _clip(text: str) -> str:
    """Bound one normalised string. See :data:`MAX_TEXT_CHARS` for why the cap lives here."""
    if len(text) <= MAX_TEXT_CHARS:
        return text
    return text[: MAX_TEXT_CHARS - 1] + "\u2026"


def _text(value: Any, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return default


def _str_list(value: Any) -> list[str]:
    if isinstance(value, str):  # a lone string is a common hand-edit; treat it as one entry
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        text = _text(item)
        if text:
            out.append(text)
    return out


#: Words a JSON producer writes when it means a bool but its serializer did not emit one. Both
#: directions, because a producer that wrote ``"true"`` stated a result and dropping it would lose
#: one; anything outside this vocabulary is declined rather than guessed at.
_TEXTUAL_TRUE = frozenset({"true", "yes", "on", "1"})
_TEXTUAL_FALSE = frozenset({"false", "no", "off", "0"})


def _tristate(value: Any) -> bool | None:
    """``True``/``False``/``None`` -- where ``None`` means "this layer could not read it".

    ``bool(value)`` cannot be used here: ``bool("false")`` and ``bool("no")`` are both ``True``, and
    the renderer turns that into a green ``PASS`` beside the check's own detail text explaining why
    it failed. Python truthiness answers "is this a non-empty object", which is not the question.

    The renderer already carries a third state (``--``) for a result it does not have, so declining
    is available and PASS never has to be guessed -- and PASS is the guess that reads as "checked
    and fine". Real bools and JSON's own ``0``/``1`` keep exactly the meaning they have today.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TEXTUAL_TRUE:
            return True
        if word in _TEXTUAL_FALSE:
            return False
    return None


def _mappings(value: Any) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _figure(raw: dict) -> dict:
    return {
        "path": _text(raw.get("path")),
        "title": _text(raw.get("title")) or _text(raw.get("path")) or "Figure",
        "caption": _text(raw.get("caption")),
        "kind": _text(raw.get("kind")),
    }


def _table(raw: dict) -> dict:
    rows = raw.get("rows")
    return {
        "path": _text(raw.get("path")),
        "title": _text(raw.get("title")) or _text(raw.get("path")) or "Table",
        "rows": rows if isinstance(rows, int) and not isinstance(rows, bool) else None,
    }


def _json_scalar(value: Any) -> Any:
    """A scalar the JSON *encoder* will accept, not merely one the decoder produced.

    Those are different sets, and the gap is where the portal broke. Python's ``json`` writes bare
    ``NaN``/``Infinity`` and reads them straight back, so a metric that came out non-finite -- a
    correlation against a constant vector, a mean over an empty selection -- round-trips through a
    manifest as a perfectly ordinary ``float``. Starlette then encodes the response with
    ``allow_nan=False`` and raises *after* the handler returned: ``/api/results/run`` 500s and the
    whole run card goes with it, over one metric, on a run that finished fine.

    Spelled out rather than dropped. ``nan`` is a result -- usually the interesting one -- and a
    finding that silently loses its value reads as a finding that was never computed.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _finding(raw: dict) -> dict:
    key = _text(raw.get("key"))
    value = raw.get("value")
    return {
        "key": key,
        "value": _json_scalar(value) if isinstance(value, (str, int, float, bool)) or value is None else _text(value),
        "label": _text(raw.get("label")) or key or "Finding",
    }


def _next_step(raw: dict) -> dict:
    priority = raw.get("priority")
    return {
        "action": _text(raw.get("action")),
        "why": _text(raw.get("why")),
        "priority": priority if isinstance(priority, int) and not isinstance(priority, bool) else None,
    }


def _review(raw: Any) -> dict | None:
    """The L2 block, or ``None``. L1 always writes ``null``; that is not an error, it is 'not yet'."""
    if not isinstance(raw, dict):
        return None
    checks = []
    for c in _mappings(raw.get("checks")):
        checks.append(
            {
                "name": _text(c.get("name")) or "check",
                "passed": _tristate(c.get("passed")),
                "detail": _text(c.get("detail")),
            }
        )
    return {
        "verdict": _text(raw.get("verdict")) or "unknown",
        "reasons": _str_list(raw.get("reasons")),
        "checks": checks,
    }


def normalize(raw: Any) -> dict:
    """Coerce any JSON value into the full contract shape. Never raises.

    Every contract key is present afterwards, with the contract's type. Unknown keys are dropped
    (the contract says consumers ignore them, so a v2 manifest renders here unchanged).

    The contract assigns ``normalize()`` to the schema module. It is not there yet, so the
    implementation lives below -- but the schema module's version wins the moment it exists, which
    is what keeps this from becoming the second implementation the contract forbids.
    """
    upstream = globals().get("_upstream_normalize")
    if upstream is not None:
        return upstream(raw)
    src = raw if isinstance(raw, dict) else {}
    version = src.get("schema_version")
    return {
        "schema_version": version if isinstance(version, int) and not isinstance(version, bool) else SCHEMA_VERSION,
        # No substitute name here. ``tool_name`` is optional on ``run_post_analysis`` and all 92
        # recorded runs carry ``null``, so a placeholder invented at this depth is not a rare
        # fallback -- it is what every consumer reads, printed as a recorded fact ("Tool: unknown
        # tool") beside a card that knows the directory the tool actually wrote. The empty string
        # keeps the contract's type and says the honest thing: nothing was recorded. Each display
        # surface then picks its own better answer, because only it knows what it has to work with.
        "tool_name": _text(src.get("tool_name")),
        "task_type": _text(src.get("task_type")) or "unknown",
        "source_outputs": _str_list(src.get("source_outputs")),
        "status": _text(src.get("status")) or "unknown",
        "figures": [_figure(f) for f in _mappings(src.get("figures"))],
        "tables": [_table(t) for t in _mappings(src.get("tables"))],
        "findings": [_finding(f) for f in _mappings(src.get("findings"))],
        "warnings": _str_list(src.get("warnings")),
        "review": _review(src.get("review")),
        "next_steps": [_next_step(s) for s in _mappings(src.get("next_steps"))],
    }


def load(results_dir: Path | str, *, with_specs: bool = False) -> dict:
    """Read ``<results_dir>/manifest.json`` and return it normalised.

    Raises :class:`ManifestError` -- and only that -- when there is nothing usable to read, so a
    caller (report writer, web handler) has exactly one exception type to handle.

    ``with_specs`` folds each figure's ``.figspec.json`` sidecar into its entry, and is **off by
    default because it costs one ``stat`` per declared figure**. Discovery walks every run on the
    box and caps its own stat budget deliberately (``discover.MAX_DECLARED_TO_STAT``); a manifest
    declaring a hundred thousand figures must not turn into a hundred thousand extra stats on a
    listing that shows a card. Only the surfaces that actually display a figure's record ask for
    it -- today that is the chat attachment alone -- and it reads ONE run, not all of them, and
    folds at most :data:`MAX_FIGURE_SPECS` sidecars within that run.
    """
    # Contained like every other file this module reads: a manifest.json that is a symlink out of the
    # run directory is refused, not followed. The portal reads it as root, and a worker could link
    # another account's manifest into its own run to have the root portal show it (hunt 2026-09-30,
    # u17-cli-report-1) -- which the module docstring already promised could not happen.
    path = safe_subpath(results_dir, MANIFEST_NAME)
    if path is None:
        raise ManifestError(f"{MANIFEST_NAME} in {results_dir} leads outside the run directory; refusing to read it")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ManifestError(f"no {MANIFEST_NAME} in {results_dir}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{MANIFEST_NAME} is implausibly large ({size} bytes); refusing to read it")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path} is not a JSON object")
    normalised = normalize(raw)
    return _with_figure_specs(normalised, Path(results_dir)) if with_specs else normalised


#: How much of a figure's record is worth carrying to a reader. The spec sidecar holds the whole
#: provenance -- every resolved parameter, the package versions, the seeds -- and most of it is for
#: reproducing the figure, not for reading beside it.
_SPEC_MAX_PARAMS = 12
_SPEC_MAX_LIMITATIONS = 6


def _figure_spec(results_dir: Path, figure: dict) -> dict:
    """Fold a figure's ``.figspec.json`` sidecar into its manifest entry, when one is beside it.

    The contract's figure entry is four keys -- path, title, caption, kind -- and
    :func:`Manifest.add_figure` writes exactly those. Everything a reader would want to know about
    HOW the figure was drawn (which matrix slot, which transform, what was sampled, what it must
    not be read as) lives in a sidecar the visualization toolkit writes beside the image.

    It is folded in HERE rather than in :func:`normalize`, and that split is deliberate: normalize
    takes a raw mapping and nothing else, it is pure, and the contract assigns it to the schema
    module the moment that module grows it. Only ``load`` knows the directory, and only a caller
    with the directory can find the sidecar. Carrying the spec inside the manifest entry instead
    would mean changing ``add_figure`` and the schema version every other consumer reads, for data
    that is already on disk in a file of its own.

    A figure with no sidecar -- which is every figure every other tool in this platform writes --
    comes back exactly as it went in. Never raises: a sidecar that is missing, unreadable or not
    JSON leaves the entry untouched, because a figure that displays without its provenance is worth
    more than a run page that 500s.
    """
    rel = _text(figure.get("path"))
    if not rel:
        return figure
    image = results_dir / rel
    sidecar = safe_subpath(results_dir, (Path(rel).parent / f"{image.stem}.figspec.json").as_posix())
    if sidecar is None:
        return figure  # a sidecar that is a link out of the run is not this run's provenance
    try:
        if not sidecar.is_file() or sidecar.stat().st_size > MAX_MANIFEST_BYTES:
            return figure
        record = json.loads(sidecar.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError, RecursionError):
        return figure
    if not isinstance(record, dict):
        return figure

    params = record.get("params") if isinstance(record.get("params"), dict) else {}
    limitations = [str(x) for x in (record.get("limitations") or []) if isinstance(x, (str, int, float))]
    expression = record.get("expression") if isinstance(record.get("expression"), dict) else None
    shown = dict(list(params.items())[:_SPEC_MAX_PARAMS])
    spec = {
        "figure_id": _text(record.get("figure_id")),
        "plot_id": _text(record.get("plot_id")),
        "plot_type": _text(record.get("plot_id")).replace(".", " ") or _text(figure.get("kind")),
        "function": _text(record.get("function")),
        "revision": record.get("revision") if isinstance(record.get("revision"), int) else 1,
        "derived_from": _text(record.get("derived_from")),
        "source": _text((record.get("source") or {}).get("name")) if isinstance(record.get("source"), dict) else "",
        # Which matrix the values came from, spelled the way the caption spells it. This is the one
        # field whose absence turns a figure into a picture nobody can check.
        "layer": (
            (
                f"layers['{expression.get('key')}']"
                if expression.get("slot") == "layers"
                else _text(expression.get("slot"))
            )
            if expression
            else ""
        ),
        "transform": _text(expression.get("transform")) if expression else "",
        "parameters": shown,
        "n_parameters": len(params),
        "limitations": limitations[:_SPEC_MAX_LIMITATIONS],
        "n_limitations": len(limitations),
    }
    return {**figure, "spec": spec}


#: How many sidecars one read may look for. Making the fold opt-in fixed *discovery*, which walks
#: every run on the box -- but the caller that opts in renders at most four figures and would still
#: have paid one ``stat`` per DECLARED figure, so a manifest claiming a hundred thousand of them
#: cost a hundred thousand stats on the chat path instead of the listing one. Well above anything
#: any surface shows (the chat card caps at four), and bounded, which is the point.
MAX_FIGURE_SPECS = 32


def _with_figure_specs(manifest: dict, results_dir: Path) -> dict:
    """Every figure, with its sidecar folded in where there is one, up to :data:`MAX_FIGURE_SPECS`.

    Figures past the cap come back untouched rather than absent: a reader that shows figure fifty
    still gets its path, title, caption and kind, and loses only the record. Dropping the figure
    would be the worse failure, and silently folding a hundred thousand sidecars is the one this
    cap exists to stop.
    """
    figures = manifest.get("figures")
    if not isinstance(figures, list) or not figures:
        return manifest
    folded = [_figure_spec(results_dir, f) for f in figures[:MAX_FIGURE_SPECS]]
    return {**manifest, "figures": folded + list(figures[MAX_FIGURE_SPECS:])}


def is_run_dir(candidate: Path | str) -> bool:
    """True when ``candidate`` holds a *file* named ``manifest.json``. Never raises.

    Not "a readable one", which is what this used to claim: ``is_file()`` does not open anything, so
    a ``manifest.json`` the process cannot read passes here. That is the intended behaviour and not
    a gap -- this is the gate for "does an already-issued identifier still point at something", one
    ``stat`` and no more. :func:`~spatialomicsgym.report.discover.run_dir_kind` is where a walk
    decides what the file actually *is*, and it sends an unreadable one to ``RUN_DIR_OPAQUE``, which
    renders as a card saying so, rather than dropping the run out of the listing.
    """
    try:
        return (Path(candidate) / MANIFEST_NAME).is_file()
    except OSError:
        return False
