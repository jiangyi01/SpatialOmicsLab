"""
Render one demo / real-run agent probe to a durable, scrollable transcript — a self-contained styled
HTML file and a clean ANSI plaintext for an in-terminal ``less`` pager. Built from ``probe.log`` (the
full per-step strings) + the extracted answer. Stdlib only; every value is ``redact``'d + escaped.

The terminal shows a terse cleaned timeline (see ``demo`` / ``progress`` compact mode); THIS module is
where the full detail lives, so the two together satisfy "beautiful terminal + scrollable full detail".
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import constants
from ._agent_probe import _summarize_step
from .session_log import redact

# Output artifacts the report surfaces. Figures are embedded inline as base64 ``data:`` URIs (still
# fully self-contained — no external asset); data files are listed by name + size. A staged INPUT
# lives under a ``data_lake`` dir and is deliberately excluded so inputs aren't shown as results.
_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
_DATA_EXT = {".csv", ".tsv", ".h5ad", ".pdf", ".xlsx", ".json", ".txt", ".parquet", ".loom", ".h5", ".zarr"}
_INPUT_DIR_PART = "data_lake"  # anything under here is a staged input, not a result


def _human_size(n: int) -> str:
    """A compact human-readable byte size (``742 B`` / ``12.3 KB`` / ``4.1 MB``).

    Coerces a missing / ``None`` / non-numeric size to ``0`` rather than raising: the callers pass
    ``item.get("size", 0)``, which yields ``None`` for a present-but-null ``size`` (the default only
    applies when the key is ABSENT), and this feeds the end-of-run HTML/text report
    (``_artifact_html`` / ``render_text``) that runs OUTSIDE ``write_html``'s try and is contractually
    "never raises" — a ``float(None)`` here would otherwise lose the user's whole transcript at the end
    of a long provisioning run. Mirrors the boundary ``try/except``-coerce pattern used elsewhere. (R22)
    """
    try:
        size = float(n)
    except (TypeError, ValueError):
        size = 0.0
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{int(size)} B"  # unreachable; keeps type-checkers happy


def scan_artifacts(
    root: Path | str | None,
    *,
    max_images: int = 12,
    max_image_bytes: int = 3_000_000,
    max_total_bytes: int = 15_000_000,
    max_data: int = 60,
    since: float | None = None,
) -> dict:
    """Scan a run's data root for OUTPUT artifacts to surface in the report. Best-effort — never raises.

    Returns ``{"images": [...], "data": [...], "images_omitted": int, "data_omitted": int}`` where each
    image is ``{name, rel, size, uri}`` (a base64 ``data:`` URI, so the report stays self-contained) and
    each data entry is ``{name, rel, size}``. Figures over ``max_image_bytes``, or past the count /
    ``max_total_bytes`` embed budget, are NOT base64-embedded — they are listed as data files instead
    (surfaced, never silently dropped) and counted in ``images_omitted``. Files under a ``data_lake``
    directory are staged INPUTS and are skipped entirely.

    ``since`` (epoch seconds) skips files last modified before it: the demo/real-run data root is
    durable and shared across launches, so without it a report embedded earlier runs' figures as this
    run's outputs (hunt 2026-09-30, u37-setup-checks-12)."""
    empty = {"images": [], "data": [], "images_omitted": 0, "data_omitted": 0}
    if root is None:
        return empty
    base = Path(root)
    try:
        if not base.is_dir():
            return empty
        files = sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    except OSError:
        return empty

    images: list[dict] = []
    data: list[dict] = []
    images_omitted = 0
    data_omitted = 0
    total = 0
    for p in files:
        try:
            parts = {part.lower() for part in p.relative_to(base).parts}
            if _INPUT_DIR_PART in parts:
                continue  # a staged input, not a result
            suffix = p.suffix.lower()
            rel = str(p.relative_to(base))
            stat = p.stat()
            size = stat.st_size
        except (OSError, ValueError):
            continue
        if since is not None and stat.st_mtime < since:
            continue  # an earlier run's output in the shared root, not this run's
        if suffix in _IMAGE_MIME:
            fits = len(images) < max_images and size <= max_image_bytes and (total + size) <= max_total_bytes
            uri = _image_data_uri(p, _IMAGE_MIME[suffix]) if fits else None
            if uri is not None:
                images.append({"name": p.name, "rel": rel, "size": size, "uri": uri})
                total += size
            else:
                images_omitted += 1  # too big / over budget / unreadable → fall through to the data list
                if len(data) < max_data:
                    data.append({"name": p.name, "rel": rel, "size": size})
                else:
                    data_omitted += 1
        elif suffix in _DATA_EXT:
            if len(data) < max_data:
                data.append({"name": p.name, "rel": rel, "size": size})
            else:
                data_omitted += 1
    return {"images": images, "data": data, "images_omitted": images_omitted, "data_omitted": data_omitted}


def _image_data_uri(path: Path, mime: str) -> str | None:
    """Read an image and return a base64 ``data:`` URI, or ``None`` if it can't be read."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


# Self-contained styling for the HTML transcript. Restrained technical palette (cool-biased neutrals +
# a single blue accent, green reserved for the answer) — deliberately NOT the AI-generic cream/terracotta.
# Inline only: the Artifact/headless-box constraint forbids any external asset. Monospace for code, real
# type hierarchy, <details> for progressive disclosure.
_CSS = """
:root{--ink:#1f2328;--dim:#6b7280;--line:#e5e7eb;--accent:#2563a8;--code-bg:#f6f8fa;--ok:#1a7f4b}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:#fbfcfd;
font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:60rem;margin:0 auto;padding:2.2rem 1.4rem 4rem}
.eyebrow{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.12em;
text-transform:uppercase;color:var(--accent)}
h1{font-size:1.5rem;margin:.3rem 0 .2rem;text-wrap:balance}
.meta{color:var(--dim);font-size:.85rem;margin-bottom:1.6rem}
.answer{border:1px solid var(--line);border-left:3px solid var(--ok);border-radius:8px;
background:#fff;padding:1rem 1.2rem;margin:0 0 2rem;white-space:pre-wrap}
.answer h2{font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ok);margin:0 0 .5rem}
details{border:1px solid var(--line);border-radius:8px;background:#fff;margin:.5rem 0;overflow:hidden}
summary{cursor:pointer;padding:.6rem .9rem;list-style:none;display:flex;gap:.6rem;align-items:baseline}
summary::-webkit-details-marker{display:none}
summary .lab{font-weight:600}
.step-body{border-top:1px solid var(--line);padding:.7rem .9rem;background:var(--code-bg)}
pre{margin:.3rem 0 0;overflow-x:auto;white-space:pre-wrap;word-break:break-word;
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(15rem,1fr));gap:1rem;margin:.4rem 0 2rem}
.figs figure{margin:0;border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden}
.figs img{display:block;width:100%;height:auto;background:var(--code-bg)}
.figs figcaption{padding:.5rem .7rem;border-top:1px solid var(--line);word-break:break-word;
font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim)}
.files{list-style:none;padding:0;margin:.4rem 0 2rem;border:1px solid var(--line);border-radius:8px;background:#fff}
.files li{display:flex;justify-content:space-between;gap:1rem;padding:.5rem .9rem;border-top:1px solid var(--line);
font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.files li:first-child{border-top:none}
.files .sz{color:var(--dim);font-variant-numeric:tabular-nums;white-space:nowrap}
.note{color:var(--dim);font-size:.8rem;margin:.2rem 0 1.4rem}
.foot{color:var(--dim);font-size:.8rem;margin-top:2rem;border-top:1px solid var(--line);padding-top:1rem}
"""


def build_steps(probe_log: list[str]) -> list[dict]:
    """Turn the raw per-step log strings into ``{label, mode, thought, action, raw}`` display steps.

    Reuses the probe's own ``_summarize_step`` (pure) for the label/mode; keeps the full raw string so
    the HTML/text can show everything on expand. Never raises on a malformed entry."""
    steps: list[dict] = []
    for entry in probe_log or []:
        raw = str(entry)
        try:
            s = _summarize_step(raw)
        except Exception:
            s = {"mode": "thinking", "label": "Step", "thought": raw, "action": ""}
        steps.append(
            {
                "label": s.get("label") or (s.get("mode") or "step").capitalize(),
                "mode": s.get("mode", ""),
                "thought": s.get("thought", ""),
                "action": s.get("action", ""),
                "raw": raw,
            }
        )
    return steps


def _artifact_lines(artifacts: dict | None) -> list[str]:
    """Plaintext OUTPUTS block for the pager — figure + data filenames with sizes, never base64 bytes."""
    if not artifacts:
        return []
    images = artifacts.get("images") or []
    data = artifacts.get("data") or []
    if not images and not data:
        return []
    lines = ["OUTPUTS", "-" * 60]
    for item in images:
        lines.append(
            f"    [figure] {redact(str(item.get('rel') or item.get('name')))}  ({_human_size(item.get('size', 0))})"
        )
    for item in data:
        lines.append(
            f"    [data]   {redact(str(item.get('rel') or item.get('name')))}  ({_human_size(item.get('size', 0))})"
        )
    omitted = int(artifacts.get("images_omitted") or 0)
    if omitted:
        lines.append(f"    (+{omitted} figure(s) too large to embed — listed above as data)")
    lines.append("")
    return lines


def render_text(*, title: str, route: str, answer: str, steps: list[dict], artifacts: dict | None = None) -> str:
    """A clean plaintext transcript for the ``less`` pager: header, answer, outputs, then each full step."""
    out: list[str] = []
    out.append(redact(title or ""))
    if route:
        out.append(f"routed to: {redact(route)}")
    out.append("=" * 60)
    out.append("")
    out.append("ANSWER")
    out.append(redact(answer or "(empty)"))
    out.append("")
    out.extend(_artifact_lines(artifacts))
    out.append("STEPS")
    out.append("-" * 60)
    for i, s in enumerate(steps or [], 1):
        out.append(f"{i}. {redact(str(s.get('label') or 'Step'))}")
        body = redact(str(s.get("raw") or s.get("thought") or ""))
        for ln in body.splitlines():
            out.append("    " + ln)
        out.append("")
    return "\n".join(out)


def _esc(text: object) -> str:
    return html.escape(redact(str(text or "")))


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(token: object, fallback: str) -> str:
    """A filename-safe path component: keep ``[A-Za-z0-9._-]``, collapse every other run to ``_``, and
    strip leading/trailing ``. _ -`` so a value like ``..`` or ``a/b/c`` can neither traverse out of the
    transcripts dir nor nest into a missing subdir (which would silently drop the file). Falls back when
    nothing printable survives. A normal conda env name (``zzsog`` / ``sog``) is returned unchanged."""
    cleaned = _UNSAFE_NAME.sub("_", str(token or "")).strip("._-")
    return cleaned or fallback


def _artifact_html(artifacts: dict | None) -> str:
    """The results block: an embedded figure grid + an output-file list. Empty string when there is
    nothing to show, so a run with no outputs renders exactly like the old transcript. Every figure is
    a base64 ``data:`` URI (self-contained — the ``src`` never points at an external host)."""
    if not artifacts:
        return ""
    images = artifacts.get("images") or []
    data = list(artifacts.get("data") or [])
    if not images and not data:
        return ""
    parts: list[str] = []
    if images:
        cells = "".join(
            f"<figure><img alt='{_esc(im.get('name'))}' src='{im.get('uri', '')}'>"
            f"<figcaption>{_esc(im.get('rel') or im.get('name'))}</figcaption></figure>"
            for im in images
            if im.get("uri")
        )
        parts.append(f"<div class='eyebrow'>Figures</div><div class='figs'>{cells}</div>")
    if data:
        items = "".join(
            f"<li><span class='nm'>{_esc(d.get('rel') or d.get('name'))}</span>"
            f"<span class='sz'>{_esc(_human_size(d.get('size', 0)))}</span></li>"
            for d in data
        )
        omitted = int(artifacts.get("images_omitted") or 0)
        note = (
            f"<div class='note'>+{omitted} figure(s) too large to embed — listed above as output files.</div>"
            if omitted
            else ""
        )
        parts.append(f"<div class='eyebrow'>Output files</div><ul class='files'>{items}</ul>{note}")
    return "".join(parts)


def write_html(
    *, basic: str, kind: str, title: str, route: str, answer: str, steps: list[dict], artifacts: dict | None = None
) -> Path | None:
    """Write a self-contained styled HTML **analysis report** to ``transcripts_dir()/<basic>_<kind>.html``.

    Beyond the answer + reasoning steps, it embeds the run's output FIGURES (base64 ``data:`` URIs — still
    fully self-contained, works offline) and lists its output DATA files. ``artifacts`` is the mapping from
    ``scan_artifacts``; omit it (or pass ``None``) and the report is byte-for-byte the old transcript.

    Inline CSS only — no external asset ever. Every value is ``redact``'d then ``html.escape``'d.
    Best-effort: returns ``None`` on an OSError, never raises."""
    steps = steps or []
    rows: list[str] = []
    for i, s in enumerate(steps, 1):
        detail = s.get("raw") or s.get("thought") or ""
        action = s.get("action") or ""
        body = _esc(detail)
        if action:
            body = f"{_esc(action)}\n\n{body}" if detail else _esc(action)
        rows.append(
            f"<details><summary><span class='lab'>{i}. {_esc(s.get('label') or 'Step')}</span></summary>"
            f"<div class='step-body'><pre>{body}</pre></div></details>"
        )
    doc = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)} — Analysis Report</title><style>{_CSS}</style></head><body><div class='wrap'>"
        f"<div class='eyebrow'>SpatialOmicsLab · {_esc(kind)} run · Analysis Report</div>"
        f"<h1>{_esc(title)}</h1>"
        f"<div class='meta'>routed to <code>{_esc(route or 'n/a')}</code> · {len(steps)} steps</div>"
        f"<section class='answer'><h2>Answer</h2>{_esc(answer or '(empty)')}</section>"
        f"{_artifact_html(artifacts)}"
        f"<div class='eyebrow'>Reasoning steps</div>{''.join(rows)}"
        "<div class='foot'>ST-Coscientist analysis report. Generated by sog-setup — safe to share.</div>"
        "</div></body></html>"
    )
    try:
        out_dir = constants.transcripts_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{_safe_component(basic, 'sog')}_{_safe_component(kind, 'run')}.html"
        path.write_text(doc, encoding="utf-8")
        return path
    except OSError:
        return None


def page(text: str, *, interactive: bool) -> bool:
    """Show ``text`` in ``less -R`` when it is safe to do so; return whether we spawned the pager.

    A no-op (returns ``False``) unless ``interactive`` AND ``less`` is on PATH — so scripted /
    ``--answers`` / ``--dry-run`` / non-TTY / no-``less`` never blocks. Writes the text to a temp file
    (avoids a broken-pipe on stdin) and never raises: a pager error just returns ``False``."""
    if not interactive or not shutil.which("less"):
        return False
    tmp = None
    try:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
                tmp = fh.name  # bind BEFORE the write so a failed/interrupted write still cleans up (no leak)
                fh.write(redact(text or ""))
        except OSError:
            return False
        subprocess.run(["less", "-R", tmp], check=False)
        return True
    except (OSError, KeyboardInterrupt):
        return False
    finally:
        if tmp:
            try:
                Path(tmp).unlink()
            except OSError:
                pass
