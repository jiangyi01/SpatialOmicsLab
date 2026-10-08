"""
Render a post-analysis ``manifest.json`` into a human-readable HTML report (L3).

Two output modes, one renderer -- so the file a user emails and the page the web portal shows can
never drift apart:

* **Self-contained** (default, what :func:`write_report` writes to ``<results_dir>/report.html``):
  every figure is a base64 ``data:`` URI and the CSS is inlined. No network, no CDN, no sibling
  files -- the single ``.html`` opens off a USB stick.
* **Served** (``figure_src=`` / ``table_href=`` callables): figures stay on disk and the page points
  at them, which is what the web portal wants for a directory of 200-dpi PNGs. Either callback may
  answer ``None`` for "I will not hand that file over" -- what is servable is the *caller's* rule,
  not this module's, and a caller that has a rule is the only one that can apply it. Say so in the
  reader's terms rather than emitting a link that dies.

Follows ``install/sog_install/transcript.py``'s house pattern rather than inventing a second HTML
stack: stdlib only, inline ``<style>``, base64 ``data:`` URIs, and every value ``redact``-ed then
``html.escape``-d on the way in. There is no JavaScript in the output at all, so the report is inert.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import html
import io
import os
import re as _re_actions
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spatialomicsgym import redaction
from spatialomicsgym.paths import shown_path
from spatialomicsgym.postanalysis.actions import humanize_action
from spatialomicsgym.postanalysis.manifest import discard_staging, open_staging, staging_name
from spatialomicsgym.postanalysis.review import failed_steps
from spatialomicsgym.postanalysis.sources import leads_under_benchmark_results

from .discover import encodable_text, run_label
from .manifest import MANIFEST_NAME, STATUSES, TASK_TYPES, VERDICTS, declined_for_no_handler, load, safe_subpath

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

#: The image types a report will point an ``<img>`` at, whether it embeds them or links them.
#: Public because the list stopped being only this module's business: ``research.report`` stages a
#: run's figures into a folder beside its own document and has to copy exactly the types this
#: renderer is willing to link -- and a second spelling of the list is a second thing to keep in
#: step with this one.
IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}
#: The spelling the rest of this module has always used.
_IMAGE_MIME = IMAGE_MIME

# Do not inline a figure bigger than this; say so instead of producing a 400 MB "email me" file.
MAX_EMBED_BYTES = 6 * 1024 * 1024
# ...nor more total embedded bytes than this across the whole report.
MAX_EMBED_TOTAL_BYTES = 40 * 1024 * 1024
# CSV preview budget. A table can be a million rows; the report shows the shape, the file has the data.
PREVIEW_ROWS = 8
PREVIEW_COLS = 12
PREVIEW_BYTES = 512 * 1024

# This was a hand-kept copy of ``sog_portal/server.py::_SECRET_RE``, imported from nowhere because the web
# layer imports this package and importing it back would cycle. The shared list now lives in
# :mod:`spatialomicsgym.redaction`, a leaf that imports only ``re`` -- so there is no cycle to avoid
# and nothing left to keep in sync by hand.
_SECRET_RE = redaction.SECRET_RE


def redact_text(value: Any) -> str:
    """Mask secrets in any string bound for the report or the portal. Never raises.

    Also the one place that guarantees the result can *leave the process*. A path is bytes, not
    text: one that is not valid UTF-8 arrives here carrying lone surrogates (PEP 383) and every
    encoder downstream raises on them -- Starlette rendering the page body, ``JSONResponse``
    encoding the run list. Both the HTML and the JSON already funnel every printed string through
    this function, so the guarantee costs one encode per string and is made in one place instead of
    at each of the dozens of f-strings that would otherwise each have to remember.
    """
    return encodable_text(redaction.redact(value))


#: What every :func:`_esc` runs before escaping. ``redact_text`` -- masking credentials and nothing
#: else -- is the default because that is the right answer for the two callers that write a report
#: to a file the user opens on their own machine: ``chat_cli`` and ``python -m spatialomicsgym.report``
#: print real directories on purpose, and a terminal that renamed them would be telling the person
#: sitting at the machine a half-truth about their own disk.
#:
#: The web portal is the caller with the other requirement, and it passes its own filter
#: (``sog_portal.api.routers.results._shown_text`` -- redact, *then* rename this installation's roots) through
#: :func:`render_report_body`. A :class:`~contextvars.ContextVar` rather than a parameter threaded
#: through the ten section helpers: every one of them would have to carry and forward it, and a
#: helper that forgot would leak silently -- which is the exact failure this is here to close.
#: ContextVar and not a module global because it must not leak out of the call that set it, across
#: a thread, or across an ``await``.
_TEXT_FILTER: ContextVar[Callable[[Any], str]] = ContextVar("_TEXT_FILTER", default=redact_text)


def _esc(value: Any) -> str:
    """The only way text reaches the output: filter (redact, and for the portal rename), then escape.

    Escaping **last** is load-bearing. Scrubbing an already-escaped document would splice an
    unescaped replacement into it -- and the replacement is built from directory names, which on a
    box where the agent names its own output directories is not a string this package controls.

    Through ``encodable_text`` as well: a manifest path carrying a PEP 383 surrogate (a filename that
    is not UTF-8) printed into the body made the whole page fail to encode (u03-results-3).
    """
    return html.escape(encodable_text(_TEXT_FILTER.get()(value)), quote=True)


#: One format for the "generated" line, so a caller-supplied stamp and the default one match.
STAMP_FORMAT = "%Y-%m-%d %H:%M %Z"


def analysed_at(results_dir: Path | str) -> str | None:
    """When this run was analysed, formatted for the report's ``generated`` line.

    :func:`render_report_body` stamps ``datetime.now()`` when the caller names no time, which is
    right for :func:`write_report` -- that file IS being generated now. It is wrong for the web
    portal, which re-renders an existing run from its manifest every time a reader opens it: all 92
    recorded runs were stamped with the moment of the click, so a report of an analysis from last
    week read as generated today, and two readers of the same unchanged run saw two different times.

    The manifest's mtime is when L1 finished writing the run. ``None`` if it cannot be read, and the
    caller then falls back to ``now`` -- no worse than today's behaviour.
    """
    try:
        mtime = (Path(results_dir) / MANIFEST_NAME).stat().st_mtime
    except OSError:
        return None
    return _dt.datetime.fromtimestamp(mtime).astimezone().strftime(STAMP_FORMAT)


def esc_url(value: Any) -> str:
    """The only way a URL reaches an ``href``/``src``: escape, without redacting.

    :func:`_esc` is redact-then-escape and is right for every character a reader is shown. It is
    wrong for a URL attribute, because the value there is not prose: it is an identifier this
    package built and the server validates on the way back in. ``redact_text`` masks any run of 32+
    hex characters, so a single figure named after a content hash rendered as
    ``src='.../[redacted].png'`` -- a broken image and a download link that 404s, with nothing on
    the page explaining why.

    This does not change what a reader is *shown*: the same path is still redacted everywhere it is
    printed (the ``<code>`` beside the download link, the card title above it). It changes only what
    the link points at. The portal's JSON already draws this line -- ``_ID_KEYS`` in
    ``sog_portal/api/routers/results.py`` exempts ``dir`` from redaction for exactly this reason -- and the HTML
    pages now draw it in the same place rather than a different one.
    """
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# styling -- inline, print-friendly, no external asset
# --------------------------------------------------------------------------- #
#: The three words a check row can show. They live here, beside the stylesheet that has to fit
#: them, because the pair is the invariant: ``.check .mark`` was written for a one-character glyph
#: and the emitter has always written four-character words, so ``PASS`` painted on top of the check
#: name in every report. A test reads this constant and ``REPORT_CSS`` from this one module and
#: fails if a longer word is added without widening the column.
CHECK_MARKS = {True: "PASS", False: "FAIL", None: "--"}

#: Shared by the standalone report and the web portal so the two look like one product.
REPORT_CSS = """
:root{--ink:#16202b;--dim:#4f5b67;--faint:#8a95a1;--line:#d0d8df;--bg:#f0f3f5;--card:#fafbfc;
--accent:#0f7d76;--accent-text:#0a5d58;--accent-wash:#e6f4f2;--ok:#1a7f4b;--warn:#9a6b06;--warn-wash:#fdf5e3;
--err:#b3402f;--err-wash:#fbeeec;--code:#ebeff2}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);background:var(--bg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:62rem;margin:0 auto;padding:2.4rem 1.4rem 5rem}
/* `--accent-text`, not `--accent`, wherever the accent is INK: the accent measures 4.47:1 on the
   page ground since the palette darkened (C1), below AA; the text variant is the same hue held at
   4.5+, the way the SPA's wordmark and links already do it (globals.css --accent-text). */
a{color:var(--accent-text)}
.eyebrow{font:600 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.14em;
text-transform:uppercase;color:var(--accent-text)}
h1{font-size:1.7rem;margin:.45rem 0 .3rem;letter-spacing:-.01em;overflow-wrap:anywhere}
h2{font-size:1.02rem;margin:2.2rem 0 .7rem;padding-bottom:.35rem;border-bottom:1px solid var(--line)}
.meta{color:var(--dim);font-size:.86rem;display:flex;flex-wrap:wrap;gap:.45rem .8rem;align-items:center;
margin-bottom:1.4rem}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.86em;overflow-wrap:anywhere}
.badge{display:inline-block;padding:.14rem .55rem;border-radius:999px;border:1px solid var(--line);
background:var(--card);font:600 11px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.04em;
text-transform:uppercase;white-space:nowrap;
max-width:100%;overflow:hidden;text-overflow:ellipsis;vertical-align:bottom}
.badge.ok{color:var(--ok);border-color:#bde0cb;background:#eef8f2}
.badge.partial{color:var(--warn);border-color:#ecd7a2;background:var(--warn-wash)}
.badge.failed,.badge.unusable{color:var(--err);border-color:#eec4bc;background:var(--err-wash)}
.badge.suspicious{color:var(--warn);border-color:#ecd7a2;background:var(--warn-wash)}
.card{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:1rem 1.15rem;margin:.6rem 0}
.callout{border-left:3px solid var(--accent);border-radius:8px;background:var(--accent-wash);
border-top:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line);
padding:.85rem 1.1rem;margin:.9rem 0}
.callout.warn{border-left-color:#d9a521;background:var(--warn-wash)}
.callout.err{border-left-color:var(--err);background:var(--err-wash)}
.callout p{margin:.2rem 0}
/* Review reasons and next steps are prose the *agent* wrote, so they routinely carry an unbroken
   filename (`scanpy_spatial_domains.h5ad` is 26 characters). Every other text box in this sheet
   already says `anywhere`; these two lists were simply missed, and at 320px the reason list
   overflowed its callout by 29px. */
.callout li,ol.steps li{overflow-wrap:anywhere}
.quiet{color:var(--dim);font-size:.88rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(11rem,100%),1fr));gap:.7rem;margin:.6rem 0 0}
.tile{border:1px solid var(--line);border-radius:10px;background:var(--card);padding:.7rem .85rem}
.tile .k{font:600 10px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:.09em;
text-transform:uppercase;color:var(--faint);overflow-wrap:anywhere}
.tile .v{font-size:1.35rem;font-weight:640;letter-spacing:-.02em;margin-top:.15rem;overflow-wrap:anywhere}
.figs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(19rem,100%),1fr));gap:1rem;margin:.6rem 0 0}
figure{margin:0;border:1px solid var(--line);border-radius:10px;background:var(--card);overflow:hidden;
display:flex;flex-direction:column}
figure .ftitle{padding:.6rem .85rem .1rem;font-weight:620;font-size:.95rem;overflow-wrap:anywhere}
figure img{display:block;width:100%;height:auto;background:#fff;padding:.75rem;border-radius:10px;box-shadow:0 1px 2px rgba(0,0,0,.18),0 6px 18px rgba(0,0,0,.14)}
figcaption{padding:.5rem .85rem .75rem;color:var(--dim);font-size:.84rem;overflow-wrap:anywhere}
figure .absent{padding:1.6rem .85rem;text-align:center;color:var(--faint);background:var(--code);
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{text-align:left;padding:.4rem .7rem;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:var(--code);font:600 11px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;
letter-spacing:.05em;text-transform:uppercase;color:var(--dim)}
tr:last-child td{border-bottom:none}
ul.plain{list-style:none;padding:0;margin:.4rem 0 0}
ul.plain li{border:1px solid var(--line);border-radius:8px;background:var(--card);padding:.6rem .9rem;
margin:.4rem 0;overflow-wrap:anywhere}
ol.steps{padding-left:1.2rem;margin:.4rem 0 0}
ol.steps li{margin:.45rem 0}
ol.steps .why{color:var(--dim);font-size:.9rem}
.check{display:flex;gap:.6rem;align-items:baseline;padding:.35rem 0;border-bottom:1px dashed var(--line)}
.check:last-child{border-bottom:none}
.check .mark{font-weight:700;flex:0 0 auto;min-width:3rem}
.check > span + span{min-width:0;overflow-wrap:anywhere}
.check .mark.y{color:var(--ok)} .check .mark.n{color:var(--err)}
.foot{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--faint);font-size:.8rem}
@media print{body{background:#fff}.wrap{max-width:none;padding:0}.card,figure,.tblwrap{break-inside:avoid}}
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _human_size(n: Any) -> str:
    try:
        size = float(n)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{int(size)} B"  # pragma: no cover - unreachable


def _pretty(token: str) -> str:
    return str(token or "").replace("_", " ").strip() or "unknown"


def _run_heading(m: dict, results_dir: Path | str, run_name: str = "") -> str:
    """What to call this run at the top of its own page.

    ``tool_name`` is optional on ``run_post_analysis`` and every recorded run omits it, so "no tool
    name" is the ordinary case rather than the broken one. The directory the tool wrote is then the
    most specific true thing this page knows, and it is the label the portal card beside it already
    carries -- so the two front doors name the same run the same way instead of one saying
    ``Unknown Tool`` while the other says ``run_spagcn_visium/post_analysis``.

    Returned verbatim, which is the point of taking it from ``run_label`` at all. It used to go
    through :func:`_pretty` first, and that turns ``bass/post_analysis`` into ``bass/post analysis``
    -- reintroducing exactly the disagreement described above, on all 92 recorded runs, since none of
    them carry a ``tool_name``. ``sog_portal.api.routers.results`` prints ``run_label`` unaltered everywhere it names a
    run, and a name is for finding the thing again: in the portal's list, in a directory listing, in
    a file manager. A recorded ``tool_name`` is likewise what the registry, the logs and the MCP
    layer all call it.

    THIS IS THE ANSWER FOR A PAGE WITH NO ROOT
    ..........................................
    ``run_label`` is called here with no ``rel``, because a rendered page is a file that can be
    copied anywhere and has no root to be relative to (R80/S also forbids it printing the absolute
    path it does have). That answer names a run by two trailing path components, which is ambiguous
    once a layout nests: ``sample1/bass/post_analysis`` and ``sample2/bass/post_analysis`` both
    reduce to ``bass/post_analysis``.

    A caller that *does* know the root resolves that itself and passes it as ``run_name``; the portal
    does, from :func:`sog_portal.api.routers.results._named`, which is why its card and the page it opens agree at any
    depth. The no-root answer remains the right default for the standalone renderer, and it is what
    all 92 recorded runs get either way -- at their depth the two rules produce the same characters.

    ``run_name`` loses to a recorded ``tool_name`` here and wins outright in the provenance table,
    and that is not an inconsistency: this is the page's *heading* and that row is headed "Results
    directory". A tool name is the better headline and is not a directory at all.

    It is not the better ``<title>``, though, and :func:`render_report` no longer reaches for this
    when it needs one -- on either door. A heading sits above a page that also prints the directory;
    a tab sits beside the other tabs with nothing else on it, and one tool name across a sweep makes
    them indistinguishable. The tab therefore takes the last two branches of this function and skips
    the first.

    ``_pretty`` was also supplying the fallback for a run with no derivable name, so that stays.
    """
    recorded = str(m.get("tool_name") or "").strip()
    return recorded or run_name or run_label(Path(results_dir)) or "unknown"


def badge_class(value: str, allowed) -> str:
    """The CSS class a badge may carry for ``value`` -- ``""`` when this version cannot read it.

    Public because the portal's run cards need the same rule and had written their own: they
    interpolated the word straight into ``class='badge {value}'``, and ``class`` is a
    space-separated list, so a verdict of ``"ok but only 2 of 12 checks ran"`` matched ``.badge.ok``
    and the card went green next to a red ``failed`` status. Colour is a claim about which of the
    declared outcomes this is; a word the vocabulary does not contain has made no such claim.

    Case and surrounding space are normalized because a producer writing ``"OK"`` did name a
    declared outcome. Anything else is declined rather than guessed at -- an uncoloured badge still
    prints its text, so nothing is hidden, only un-asserted.
    """
    v = (value or "").strip().lower()
    return v if v in allowed else ""


class _Escapes(list):
    """Collects 'this manifest path leaves the results dir' notes so they are surfaced, not dropped."""


def _resolve(results_dir: Path, rel: str, escapes: _Escapes, what: str) -> Path | None:
    """Contain one manifest path. On refusal, record a warning and return ``None``."""
    if not rel:
        return None
    target = safe_subpath(results_dir, rel)
    if target is None:
        escapes.append(f"{what} {rel!r} points outside the results directory and was not read.")
        return None
    return target


def _data_uri(path: Path, budget: list[int]) -> tuple[str | None, str]:
    """``(uri, note)``. ``uri`` is ``None`` when the file is absent / unreadable / over budget.

    ``budget`` is ``[bytes remaining, figures embedded so far]`` -- the count exists so the second
    refusal below can say what it is actually about.
    """
    mime = _IMAGE_MIME.get(path.suffix.lower())
    if mime is None:
        return None, f"unsupported image type {path.suffix or '(none)'}"
    try:
        size = path.stat().st_size
    except OSError:
        return None, "file is missing"
    # Two different refusals shared one sentence, "too large to embed (N)". Only the first is about
    # this file's size; the second is the report-wide budget, which earlier figures spent -- so a
    # 200 KB panel was reported as too large to embed and the reader went looking for a big file
    # that does not exist. Worse, the fix differs: one figure needs shrinking, the other means the
    # report is full and nothing is wrong with the panel at all.
    if size > MAX_EMBED_BYTES:
        return None, (
            f"is {_human_size(size)}, over the {_human_size(MAX_EMBED_BYTES)} per-figure embed "
            "limit - open it in the results directory"
        )
    if budget[0] - size < 0:
        return None, (
            f"was not embedded: the report's {_human_size(MAX_EMBED_TOTAL_BYTES)} total size limit "
            f"was reached after {budget[1]} figure(s) - open it in the results directory"
        )
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "file could not be read"
    budget[0] -= size
    budget[1] += 1
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", ""


def _fit(row: list[str], width: int) -> list[str]:
    """``row`` squared off to exactly ``width`` cells: cropped if long, blank-filled if short."""
    return row[:width] + [""] * (width - len(row))


def _csv_preview(path: Path) -> tuple[list[str], list[list[str]], str, int, bool]:
    """``(header, rows, note, n_columns, more_rows)`` -- a bounded peek at a CSV/TSV. Never raises.

    ``n_columns`` is the width of the file, which is not the width of ``header``: the preview keeps
    :data:`PREVIEW_COLS` of them. The caller needs both, because a table silently cropped to its
    first 12 columns looks complete -- ``colocalization.csv`` for 20 cell types was drawn with eight
    of them missing under a caption about how many *rows* were shown.

    The width is the widest row seen and not the width of the header line, and ``header`` and every
    row come back squared off to the same number of cells. Both follow from the crop being applied
    per row while the caller decides whether to mention it with ``n_columns > len(header)``: read off
    ``parsed[0]``, that guard compares the header against itself on any file whose first line is
    narrower than its widest, and the caption never fires. Ragged is ordinary here -- R's
    ``write.table`` omits the row-name column from the header line and writes it in every row after
    it, and a read stopped by the byte budget can leave a short header over wide rows -- so a
    3-column header over 20-column rows was drawn as twelve values apiece, under three names, with
    no caption and more ``<td>`` than ``<th>``.

    Which name belongs to which value is left alone. A header one field short of its rows is
    conventionally an index column and pandas reads it that way, but a CSV cannot prove it and
    guessing would relabel real columns. The preview draws what is there and says how much there is.

    ``more_rows`` says whether anything was left below the preview, and it is read off the file for
    the same reason ``n_columns`` is. The caller used to answer that question from the manifest's
    declared row count, which is a different fact written at a different time: a table declared as
    eight rows but 40 rows long was drawn as its first eight with no notice at all, and one declared
    as 500 but only three rows long carried a notice about rows that do not exist.

    Two things end a preview and both mean "there is more": one row past the display budget, and the
    byte budget, which for a file of very long rows can stop the read before even that. A read cut
    by the byte budget almost certainly stopped mid-row, so the partial tail is dropped rather than
    drawn -- half a value in a cell is a wrong number on the page, and ``more_rows`` reports the cut
    either way.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            # One past the budget purely to learn whether the budget was reached.
            blob = fh.read(PREVIEW_BYTES + 1)
    except OSError:
        return [], [], "file could not be read", 0, False
    cut = len(blob) > PREVIEW_BYTES
    if cut:
        end = blob.rfind("\n", 0, PREVIEW_BYTES)
        if end < 0:
            # No line ending in half a megabyte: nothing here is known to be a whole row, so there
            # is no header to name columns with and no cell that can be shown as complete.
            return [], [], "no complete row within the preview budget", 0, True
        blob = blob[: end + 1]
    if not blob.strip():
        return [], [], "file is empty", 0, cut
    # Imported here rather than at module scope: `spatialomicsgym/utils/__init__.py` re-exports the
    # whole package, so binding anything from `utils.file_io` initialises the parent package and
    # pulls langchain, numpy and pandas in with it. This module is on the web portal's import path,
    # which answers the page and the settings routes before any key is configured, and
    # test/test_webui_stays_out_of_the_heavy_stack.py holds that path to importing none of them.
    from spatialomicsgym.utils.file_io import BENCHMARK_TRUSTED_EXTENSIONS, WHITESPACE_SEP, sniff_tabular_sep

    # `.csv` and `.tsv` keep being taken at their word, which is BENCHMARK_TRUSTED_EXTENSIONS and
    # what every already-rendered run was previewed with. Any other suffix -- `.tab`, `.txt`, a
    # tool's own invention -- has no such convention behind it, so the header decides and the
    # suffix supplies only the fallback. Before this, a tab-delimited `.txt` result table previewed
    # as a single column whose name was the whole header line, under a truthful-looking count.
    delimiter = sniff_tabular_sep(
        path,
        trust_extension=BENCHMARK_TRUSTED_EXTENSIONS,
        default="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",",
    )
    try:
        if delimiter == WHITESPACE_SEP or len(delimiter) != 1:
            # A run of spaces or tabs (numpy.savetxt's default, which file_io names for spaotsc): a
            # regex the sniffer returns and csv.reader cannot take -- it raised TypeError, which this
            # "never raises" function let through, and the run page 500'd (u17-cli-report-16).
            lines = (line.split() for line in io.StringIO(blob) if line.strip())
            parsed = [row for _, row in zip(range(PREVIEW_ROWS + 2), lines, strict=False)]
        else:
            reader = csv.reader(io.StringIO(blob), delimiter=delimiter)
            parsed = [row for _, row in zip(range(PREVIEW_ROWS + 2), reader, strict=False)]
    except (csv.Error, ValueError, TypeError):
        return [], [], "file is not readable as delimited text", 0, cut
    if not parsed:
        return [], [], "file is empty", 0, cut
    n_columns = max(len(row) for row in parsed)
    # A first line with no fields stays width zero: it names nothing, the caller renders that as an
    # unavailable preview, and padding it would put a row of blank headings over real values.
    width = min(n_columns, PREVIEW_COLS) if parsed[0] else 0
    header = _fit(parsed[0], width)
    body = parsed[1:]
    rows = [_fit(row, width) for row in body[:PREVIEW_ROWS]]
    return header, rows, "", n_columns, cut or len(body) > PREVIEW_ROWS


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def _section_findings(m: dict) -> str:
    if not m["findings"]:
        return ""
    tiles = "".join(
        f"<div class='tile'><div class='k'>{_esc(f['label'])}</div><div class='v'>{_esc(f['value'])}</div></div>"
        for f in m["findings"]
    )
    return f"<h2>Findings</h2><div class='tiles'>{tiles}</div>"


def _section_review(m: dict) -> str:
    review = m["review"]
    if review is None:
        return (
            "<h2>Review</h2><div class='callout'><p class='quiet'>These results have "
            "not been reviewed yet -- no automated verdict was recorded for this run.</p></div>"
        )
    verdict = review["verdict"]
    cls = badge_class(verdict, VERDICTS)
    reasons = "".join(f"<li>{_esc(r)}</li>" for r in review["reasons"])
    checks = "".join(
        "<div class='check'><span class='mark {mk}'>{glyph}</span><span><b>{name}</b>{detail}</span></div>".format(
            mk="y" if c["passed"] else ("n" if c["passed"] is False else ""),
            glyph=CHECK_MARKS[True]
            if c["passed"]
            else (CHECK_MARKS[False] if c["passed"] is False else CHECK_MARKS[None]),
            name=_esc(c["name"]),
            detail=f" &mdash; <span class='quiet'>{_esc(c['detail'])}</span>" if c["detail"] else "",
        )
        for c in review["checks"]
    )
    tone = "err" if verdict == "unusable" else ("warn" if verdict == "suspicious" else "")
    body = [f"<h2>Review</h2><div class='callout {tone}'>"]
    body.append(f"<p><span class='badge {cls}'>{_esc(verdict)}</span></p>")
    if reasons:
        body.append(f"<ul>{reasons}</ul>")
    body.append("</div>")
    if checks:
        body.append(f"<div class='card'>{checks}</div>")
    return "".join(body)


#: A next-step ``action`` that is safe to de-snake: a bare snake_case token, nothing else. Anything
#: with a separator, a space or an extension is prose or a path and must reach ``_esc`` unaltered.
_ACTION_TOKEN = _re_actions.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _action_label(action: Any) -> str:
    """The reader's label for a next step, without defeating the redactor to get it."""
    text = str(action)
    return humanize_action(text) if _ACTION_TOKEN.match(text) else text


def _section_next_steps(m: dict) -> str:
    steps = sorted(
        m["next_steps"],
        key=lambda s: (s["priority"] is None, s["priority"] if s["priority"] is not None else 0),
    )
    if not steps:
        return ""
    # The biologist's label, not the engine's token: ``<code>run_celltype_colocalization</code>``
    # read as plumbing on a page whose whole audience is the person deciding what to do next.
    # ``humanize_action`` de-snakes tokens the shared table does not know, so old manifests render.
    #
    # BUT ONLY FOR THINGS THAT ARE TOKENS. ``_esc`` is filter-then-escape, so de-snaking first ran
    # ``.replace("_", " ")`` over the value BEFORE the redactor saw it. That shreds every underscore
    # in a path, ``paths.ABS_PATH_RE`` then fails to match the whole path, and ``scrub`` cannot
    # resolve it against an anchor -- so an ``action`` carrying prose put a literal absolute path on
    # the page, against a standing rule that there are none. Measured on a generated report:
    # ``Re-run on /tmp/.../run a/data/visium/filtered.h5ad``.
    #
    # A token has no separators and no spaces, so this keeps de-snaking for every real action --
    # including the unknown ones the comment above exists for -- and sends anything else through
    # ``_esc`` untouched, where the redactor gets the original bytes and can do its job.
    items = "".join(
        f"<li>{_esc(_action_label(s['action']))}"
        + (f"<div class='why'>{_esc(s['why'])}</div>" if s["why"] else "")
        + "</li>"
        for s in steps
    )
    return f"<h2>Suggested next steps</h2><ol class='steps'>{items}</ol>"


def _counted(present: int, declared: int, noun: str) -> str:
    """``"3 figure(s)"`` when they all made it, ``"1 of 4 figure(s) on disk"`` when they did not.

    The header used to print ``declared`` alone, which is the manifest's promise rather than this
    page's contents: rendered against a results directory whose ``figures/`` had not been copied,
    it read ``4 figure(s)`` above four boxes of which three said "figure file is missing on disk".
    None of the 92 recorded runs diverge at the moment L1 writes them -- the gap opens between then
    and read time, and the portal re-renders on view.

    Both numbers come from the pass that rendered the boxes, so the header cannot disagree with the
    body it sits above.
    """
    if present == declared:
        return f"{declared} {noun}(s)"
    return f"{present} of {declared} {noun}(s) on disk"


def _on_disk(target: Path | None) -> bool:
    """Whether the reader can be promised this artefact exists.

    Deliberately *not* "did the renderer show it": a figure refused for size is still a figure in
    the results directory, and counting it as absent sends the reader looking for a file that is
    right where the manifest says it is. Same two steps as ``discover._n_present``, so the header,
    the portal card and the chat attachment cannot arrive at three different counts.
    """
    if target is None:
        return False
    try:
        return target.is_file()
    except OSError:  # as everywhere in this module: unreadable is "not shown", not fatal
        return False


def _absent_note(rel: str) -> str:
    """Why :func:`_resolve` returned nothing, in the reader's terms.

    It returns ``None`` for two unrelated reasons and is careful to keep them apart: a path that
    leaves the results directory is recorded in ``escapes``, an entry with no path at all returns
    early and records nothing. Both callers used to print the containment sentence for both, which
    accuses a manifest that pointed nowhere of pointing somewhere it should not have -- a specific
    and false claim, and the kind that sends a reader auditing paths that do not exist.

    L1's writer cannot produce an empty path (``postanalysis.manifest.check_artifact_path`` rejects
    one), and none of the 92 recorded runs carry one. ``report.manifest.normalize`` is what makes it
    reachable: it keeps an entry whose ``path`` is missing or null, normalising it to ``""``. That is
    the portal's read path for every run directory it discovers, and :func:`render_report` is public
    and renders the dict it is handed.
    """
    return "path outside the results directory" if rel else "no file path in the manifest"


def _served_note(rel: str) -> str:
    """The parenthetical after "cannot be served" -- the file type, since that is what decided it.

    Kept vague on purpose about *whose* rule it is: this module does not know, and must not learn.
    The caller that answered ``None`` knows, and the reader mainly needs to know the file is there
    and this page is not the way to it -- the path is printed alongside either way.
    """
    suffix = Path(rel).suffix
    return f" ({_esc(suffix)} files are not served)" if suffix else ""


def _section_figures(m: dict, results_dir: Path, figure_src, escapes: _Escapes, budget: list[int]) -> tuple[str, int]:
    if not m["figures"]:
        return "", 0
    cells = []
    present = 0
    for fig in m["figures"]:
        target = _resolve(results_dir, fig["path"], escapes, "figure")
        present += _on_disk(target)
        if target is None:
            media = f"<div class='absent'>figure not available ({_absent_note(fig['path'])})</div>"
        elif figure_src is not None:
            src = figure_src(fig["path"]) if target.is_file() else ""
            if not target.is_file():
                media = "<div class='absent'>figure file is missing on disk</div>"
            elif src:
                media = f"<img alt='{_esc(fig['title'])}' src='{esc_url(src)}'>"
            else:
                # The caller declined to serve this one. Before, an ``<img>`` went out regardless and
                # the reader got a broken image with nothing to read: the portal's file route is an
                # allowlist of media types, so a manifest declaring ``figures/panel.pdf`` -- which
                # nothing stops a manifest from declaring -- produced exactly that.
                media = f"<div class='absent'>figure cannot be served by this page{_served_note(fig['path'])}</div>"
        else:
            uri, note = _data_uri(target, budget)
            media = (
                f"<img alt='{_esc(fig['title'])}' src='{uri}'>"
                if uri
                else f"<div class='absent'>figure {_esc(note)}</div>"
            )
        caption_bits = [c for c in (fig["caption"], f"kind: {fig['kind']}" if fig["kind"] else "") if c]
        caption = f"<figcaption>{' &middot; '.join(_esc(c) for c in caption_bits)}</figcaption>" if caption_bits else ""
        cells.append(f"<figure><div class='ftitle'>{_esc(fig['title'])}</div>{media}{caption}</figure>")
    return f"<h2>Figures</h2><div class='figs'>{''.join(cells)}</div>", present


def _section_tables(m: dict, results_dir: Path, table_href, escapes: _Escapes) -> tuple[str, int]:
    if not m["tables"]:
        return "", 0
    blocks = []
    present = 0
    for tbl in m["tables"]:
        target = _resolve(results_dir, tbl["path"], escapes, "table")
        on_disk = _on_disk(target)
        present += on_disk
        head = f"<b>{_esc(tbl['title'])}</b>"
        bits = []
        # ``rows`` is the manifest's declaration, written when the analysis produced the CSV. In the
        # subtitle it reads as a statement about the table on this page, so it is only made where
        # there is one: it used to print unconditionally, putting "40 rows" directly above "file is
        # missing on disk". All 221 tables declared across the 92 recorded runs carry a count, so
        # every one of them read that way the moment its file did not resolve. The number still says
        # something about what went missing -- it moves into that sentence, in the past tense.
        if tbl["rows"] is not None and on_disk:
            bits.append(f"{tbl['rows']} rows")
        if tbl["path"]:
            bits.append(f"<code>{_esc(tbl['path'])}</code>")
        if on_disk and table_href is not None:
            href = table_href(tbl["path"])
            # Same rule as the figures above: a link the caller will refuse is worse than no link,
            # because "download" reads like a promise. The preview below is unaffected -- it is read
            # off this filesystem, not fetched.
            bits.append(
                f"<a href='{esc_url(href)}'>download</a>"
                if href
                else f"cannot be served for download{_served_note(tbl['path'])}"
            )
        sub = f"<div class='quiet'>{' &middot; '.join(bits)}</div>" if bits else ""
        declared = f" (the manifest declared {tbl['rows']} rows)" if tbl["rows"] is not None else ""
        if target is None:
            body = f"<div class='quiet'>not available ({_absent_note(tbl['path'])})</div>"
        elif not on_disk:
            body = f"<div class='quiet'>file is missing on disk{declared}</div>"
        else:
            header, rows, note, n_columns, more_rows = _csv_preview(target)
            if note or not header:
                body = f"<div class='quiet'>preview unavailable: {_esc(note or 'no columns')}</div>"
            else:
                th = "".join(f"<th>{_esc(c)}</th>" for c in header)
                tr = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
                # Both axes get cropped, so both have to be declared. Saying only how many rows were
                # shown, of a table also missing eight of its columns, is worse than saying nothing:
                # it reads as a complete statement of what was left out.
                # Both from the file, never from ``tbl["rows"]``. The declaration is written when
                # the manifest is serialized and the CSV when the analysis produced it, so the two
                # can disagree -- and the notice is a claim about the table on this page.
                shown = []
                if more_rows:
                    shown.append(f"the first {len(rows)} rows")
                if n_columns > len(header):
                    shown.append(f"the first {len(header)} of {n_columns} columns")
                more = f"<div class='quiet'>showing {' and '.join(shown)}</div>" if shown else ""
                body = (
                    f"<div class='tblwrap'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>{more}"
                )
        blocks.append(f"<div class='card'>{head}{sub}{body}</div>")
    return f"<h2>Tables</h2>{''.join(blocks)}", present


def _section_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    items = "".join(f"<li>{_esc(w)}</li>" for w in warnings)
    return f"<h2>Warnings</h2><ul class='plain'>{items}</ul>"


def _shown_path(src: Any, results_dir: Path) -> str:
    """An analysed source with the writing machine's directory layout taken off it.

    The manifest stores ``source_outputs`` absolute, which is right for a machine record. The report
    is the shareable rendering of that record -- it travels off the cluster by design -- so printing
    the path verbatim tells the recipient the sender's home directory and project layout, which is
    nothing the report is for. Figures and tables already go through ``safe_subpath``; this is the
    same convention applied to the one section that skipped it.

    A source that does not live under the results directory is still named, and still marked as
    coming from elsewhere: suppressing the path must not suppress the fact. The marker deliberately
    avoids the words :func:`_resolve` uses for a *containment violation* -- an analysed input
    legitimately sitting in another directory is not the same event as a manifest entry pointing out
    of the tree and being refused, and the report must not spell the two the same way (R79/L).

    The branch logic itself now lives in :func:`spatialomicsgym.paths.shown_path`, which generalised
    it so the browser pages could apply the same rule (R80/S is a report rule only because the
    report was the first surface to travel). This is the thin caller that supplies the two things
    the shared helper cannot know: that the frame of reference is *this* results directory, and the
    wording above. On a CLI run no anchors are registered, so the shared helper reduces to exactly
    the two branches this function used to have and the output is unchanged.
    """
    return shown_path(src, base=results_dir, outside="read from another directory")


def _section_provenance(m: dict, results_dir: Path, run_name: str = "") -> str:
    # An absent tool name is stated as absent. This table is the page's provenance record, so a
    # placeholder here reads as "the tool was called 'unknown tool'", which is a claim about the
    # run that nothing in the manifest supports.
    tool = str(m["tool_name"] or "").strip() or "not recorded"
    # The run is named the way the h1, the page title, the portal card and the download filename
    # already name it, so a reader comparing two surfaces sees one run, not two (R79/M). Computing
    # it here instead of taking the caller's ``run_name`` is how this row came to disagree with the
    # h1 four lines above it once layouts nested: ``run_label`` with no root can only answer with a
    # two-component suffix, so the heading said ``sample1/bass/post_analysis`` and this row said
    # ``bass/post_analysis`` -- while the comment above claimed the two agreed.
    #
    # Never the ``heading``, though, which is a recorded ``tool_name`` when there is one: this row
    # is headed "Results directory", and a tool name is not one.
    rows = [("Tool", tool), ("Task type", m["task_type"]), ("Results directory", run_name or run_label(results_dir))]
    for i, src in enumerate(m["source_outputs"]):
        rows.append(("Analysed" if i == 0 else "", _shown_path(src, results_dir)))
    rows.append(("Manifest", f"{MANIFEST_NAME} (schema v{m['schema_version']})"))
    body = "".join(
        f"<tr><th>{_esc(k)}</th><td class='mono'>{_esc(v)}</td></tr>" for k, v in rows if str(v).strip() or k
    )
    return f"<h2>Provenance</h2><div class='tblwrap'><table>{body}</table></div>"


def _task_type_note(m: dict) -> str:
    """Flag the task type only when no layer can account for it.

    The bare membership test this replaced fired on 21 of the 92 recorded runs and was wrong on all
    21, because a ``task_type`` outside :data:`TASK_TYPES` has two legitimate causes and neither is
    a contract violation:

    *No task type at all.* Detection could not settle it and L1 wrote an explicit null to say so.
    :func:`_label` renders that null as "unknown" a few lines below, so the note was attaching an
    accusation to a word this module invented -- 20 of the 21.

    *A registry task type with no handler.* The design contract: "``resolution`` is a valid registry
    task type with no post-analysis handler. Treat it, and any other unlisted string, as ``status:
    "partial"`` with one warning -- not ``"failed"``." L1 does exactly that and says so in a
    warning; :func:`declined_for_no_handler` is L1's own recogniser for that sentence, imported
    rather than respelled, and it is anchored per task type -- a warning about ``resolution`` does
    not excuse a manifest that says ``super_resolution``.

    What is left is the note's real job, unchanged: a manifest holding a string nothing produced.
    """
    task_type = m["task_type"]
    if not task_type or task_type in TASK_TYPES or declined_for_no_handler(m, task_type):
        return ""
    return " <span class='quiet'>(not a contract task type)</span>"


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def render_report_body(
    manifest: dict,
    results_dir: Path | str,
    *,
    figure_src: Callable[[str], str | None] | None = None,
    table_href: Callable[[str], str | None] | None = None,
    heading: str | None = None,
    run_name: str = "",
    generated_at: str | None = None,
    text_filter: Callable[[Any], str] | None = None,
) -> str:
    """The report's ``<body>`` content, with ``text_filter`` in force for the whole render.

    See :func:`_compose_report_body` for what is rendered. This wrapper exists only to bind
    :data:`_TEXT_FILTER` around it, so the ten section helpers below need no new parameter and none
    of them can forget to forward one. ``text_filter`` defaults to :func:`redact_text` -- masking
    credentials and printing real directories -- which is what a report written to a file on the
    user's own machine wants. The web portal passes ``sog_portal.api.routers.results._shown_text`` to get this
    installation's roots renamed as well.

    Reset in a ``finally``: a render that raises must not leave the portal's filter bound for
    whatever this thread does next.
    """
    token = _TEXT_FILTER.set(text_filter or redact_text)
    try:
        return _compose_report_body(
            manifest,
            results_dir,
            figure_src=figure_src,
            table_href=table_href,
            heading=heading,
            run_name=run_name,
            generated_at=generated_at,
        )
    finally:
        _TEXT_FILTER.reset(token)


def _compose_report_body(
    manifest: dict,
    results_dir: Path | str,
    *,
    figure_src: Callable[[str], str | None] | None = None,
    table_href: Callable[[str], str | None] | None = None,
    heading: str | None = None,
    run_name: str = "",
    generated_at: str | None = None,
) -> str:
    """The report's ``<body>`` content (no ``<html>``/``<style>`` wrapper).

    The web portal composes this into its own chrome; :func:`render_report` wraps it into a
    standalone document. Same sections either way -- there is one renderer, not two.

    ``run_name`` is what a caller that knows the run's root calls it -- the portal computes it in
    :func:`sog_portal.api.routers.results._named` and every one of its surfaces shows the same string. Passing it is
    what keeps the two places on this page that name the run from disagreeing with each other and
    with the card that opened it. Omitted, both fall back to what a page with no root can work out
    alone, which is what the standalone renderer has always printed.
    """
    m = dict(manifest)
    d = Path(results_dir)
    escapes = _Escapes()
    budget = [MAX_EMBED_TOTAL_BYTES, 0]

    status = m["status"]
    status_cls = badge_class(status, STATUSES)
    task_note = _task_type_note(m)
    stamp = generated_at or _dt.datetime.now().astimezone().strftime(STAMP_FORMAT)

    figures, n_figures = _section_figures(m, d, figure_src, escapes, budget)
    tables, n_tables = _section_tables(m, d, table_href, escapes)

    parts = [
        "<div class='eyebrow'>SpatialOmicsLab &middot; post-analysis report</div>",
        f"<h1>{_esc(heading or _run_heading(m, d, run_name))}</h1>",
        "<div class='meta'>",
        f"<span class='badge {status_cls}'>{_esc(status)}</span>",
        f"<span>{_esc(_pretty(m['task_type']))}{task_note}</span>",
        f"<span>{_counted(n_figures, len(m['figures']), 'figure')} &middot; "
        f"{_counted(n_tables, len(m['tables']), 'table')} "
        f"&middot; {len(m['findings'])} finding(s)</span>",
        f"<span class='quiet'>generated {_esc(stamp)}</span>",
        "</div>",
    ]
    if status == "partial":
        # "Some analyses did not complete, but the output below is usable" is two claims, and only
        # the first is L1's to make. The second belongs to L2, whose verdict this same page prints a
        # few lines down: of the 92 recorded runs 16 are ``partial``, and 13 of those are reviewed
        # ``unusable`` (10) or ``suspicious`` (3). On every one of them the banner told the reader
        # the output was usable and the Review section then said it was not -- and the banner is
        # what a reader who skims sees first. It is read from the review rather than restated, so
        # the top of the page cannot disagree with the middle of it.
        review = m["review"]
        verdict = review["verdict"] if review else None
        # ...and "Some analyses did not complete" is the other claim the banner was making without
        # the standing to make it. ``degrade()`` means two things: a step did not run, and every
        # step ran but the answer is worth less than it looks (the columns are positional labels;
        # almost every gene passes the threshold; the task type came from a filename tie-break).
        # Of the 92 recorded runs 16 are ``partial``, 11 of them skipped nothing at all, and not
        # one records a step that raised -- so the sentence was wrong on every recorded partial run
        # and it sent the reader to a Warnings section that lists caveats, not skips. ``review.py``
        # separated the two for L2 and for the planner, whose comment says counting warnings
        # instead "sends the planner after a step that never ran"; the banner asks the same
        # question through the same helper.
        n_failed = len(failed_steps(m))
        lead = (
            f"{n_failed} analysis step(s) did not complete"
            if n_failed
            else "Post-analysis recorded a caveat about this result"
        )
        warnings_pointer = "The Warnings section says what happened."
        if verdict in {"unusable", "suspicious"}:
            parts.append(
                f"<div class='callout warn'><p><b>Partial run.</b> {lead}, and "
                f"the automated Review below rates this result <b>{_esc(verdict)}</b> &mdash; read the "
                f"Review section before using it. {warnings_pointer}</p></div>"
            )
        elif review is None:
            # No review is not a clearance. L1 always writes ``review: null`` and L2 fills it in
            # afterwards, so this is the state EVERY partial run is in at the moment it is written
            # -- and the page was announcing "the output below is usable" thirty lines above a
            # Review section that said nothing had looked at it. Deferring to a verdict that does
            # not exist yet is the same mistake as contradicting one that does.
            parts.append(
                f"<div class='callout warn'><p><b>Partial run.</b> {lead}, and "
                "this result has <b>not been reviewed</b> yet &mdash; nothing has judged whether what "
                f"did complete can be relied on. {warnings_pointer}</p></div>"
            )
        elif verdict == "ok":
            parts.append(
                f"<div class='callout warn'><p><b>Partial run.</b> {lead}, but the "
                f"output below is usable. {warnings_pointer}</p></div>"
            )
        else:
            # A verdict outside :data:`VERDICTS`. ``_review`` normalizes a missing key, a JSON
            # ``null`` and any word this version does not know to ``"unknown"``, none of which is
            # the fatal pair and none of which arrives as ``None`` -- so all of them used to reach
            # the clearance above and print "the output below is usable" over a Review badge that
            # ``badge_class`` had already declined to colour. Reading a verdict means the three
            # the contract declares; anything else has not been read, and saying so is the only
            # claim available.
            parts.append(
                f"<div class='callout warn'><p><b>Partial run.</b> {lead}, and the "
                f"Review below carries a verdict this version does not recognise (<b>{_esc(verdict or '')}</b>) "
                f"&mdash; nothing here has judged whether what did complete can be relied on. {warnings_pointer}"
                "</p></div>"
            )
    elif status == "failed":
        parts.append(
            "<div class='callout err'><p><b>This run failed.</b> Anything shown below is what "
            "survived; treat it as diagnostic, not as a result.</p></div>"
        )
    else:
        # ``status`` and ``verdict`` answer different questions -- "did every step run" and "is the
        # result believable" -- and this branch used to print nothing, so the second was never asked
        # unless the first had already gone wrong. A run over degenerate values completes every step
        # it is asked to: of the 92 recorded runs, ``svca`` is ``ok`` and reviewed ``suspicious``
        # because its 'pvalue' column takes 2 distinct values over 100 genes, and it reached the
        # reader as a green badge with no banner and the verdict twelve sections down.
        #
        # Only the two adverse verdicts speak here. L1 always writes ``review: null`` and L2 fills
        # it in afterwards, so an unreviewed ok run is the state EVERY ok run passes through: a
        # "not reviewed yet" banner would fire on 47 of the 92 and say nothing about any of them.
        # That is the opposite of the ``partial`` case, where "some analyses did not complete" is
        # already true and the missing review is the reader's second reason for caution.
        review = m["review"]
        verdict = review["verdict"] if review else None
        if verdict in {"unusable", "suspicious"}:
            parts.append(
                f"<div class='callout warn'><p><b>Reviewed {_esc(verdict)}.</b> Every analysis step "
                "completed &mdash; which is what the status badge reports &mdash; but the automated "
                "Review below does not consider the output reliable. Read the Review section before "
                "using it.</p></div>"
            )
    parts.append(_section_findings(m))
    parts.append(_section_review(m))
    parts.append(_section_next_steps(m))
    parts.append(figures)
    parts.append(tables)
    # Containment refusals are appended to the manifest's own warnings so one section holds
    # everything that went wrong, and an escaping path can never be silently dropped.
    parts.append(_section_warnings(list(m["warnings"]) + list(escapes)))
    parts.append(_section_provenance(m, d, run_name))
    return "".join(p for p in parts if p)


def _tool_figures_note(results_dir: Path | str, root: Path | str | None = None) -> str:
    """The figures the *tool* drew: named, counted, and deliberately not embedded.

    A manifest describes only what post-analysis itself produced. The tool has usually already
    written its own plots beside the run, and on the two runs this section exists for -- one that
    ``failed``, and a task type the engine has no handler for -- they are the only figures there
    are. The portal shows them as pictures and the CLI run card counts them; this file said nothing,
    so the copy a reader downloads was the one surface asserting an empty Figures section over a
    directory holding five plots.

    Not embedded, which is :func:`~sog_portal.api.routers.results._section_tool_figures`'s
    reason for being portal-only and a good one: they live one level above the results directory,
    so inlining them would break this document's self-containment or bloat it by tens of megabytes.
    That is an argument against the images, not against the fact -- the CLI reaches the same split
    for the same reason ("the terminal cannot show a PNG ... so the reader is told they exist and
    where"). Names cost bytes proportional to the names.

    No path is printed. The directory is the sender's home layout, which is what ``_shown_path``
    keeps out of a report that travels off the cluster by design (R80/S), and "the directory this
    results folder sits in" locates it exactly without carrying it.

    ``root`` is how far the caller may read, and it is the caller's to state.
    :func:`~spatialomicsgym.report.discover.tool_figure_dir` refuses a parent outside it, which is
    the whole of its containment check -- and passing ``results_dir.parent``, as this did, satisfies
    that check by construction and so never exercises it. That was defended here as "the layout's
    own guarantee rather than a widening", on the grounds that a caller-chosen ``results_dir=``
    returns ``None``. It does, but from the *other* gate: the run must be named ``post_analysis``.
    The containment check was contributing nothing, and the sentence described a protection that
    was not running.

    Latent in a renderer whose callers have no bound; live in one that does. ``/api/results/report``
    serves this document, and had resolved a root before reaching it -- so on a results root that is
    itself a ``post_analysis`` directory, the run page showed no such section (its own two readers of
    ``tool_figure_dir`` pass the real root) while the printable report beside it named the files one
    level above everything the server serves. R81/D; pinned in
    ``test/test_the_report_does_not_name_files_the_portal_refuses_to_show.py``.

    Omitted, the bound stays ``results_dir.parent``, which is right for the caller that has none:
    :func:`write_report` runs inside L1, in the process that produced the run, writing a file for
    the person who ran it. There is no allowed region there to enforce, and R80/CC put this section
    in that file deliberately. The widening is now one caller's stated choice rather than every
    caller's silent default. Every layout in the recorded work trees puts the root above the tool's
    directory, where the two bounds admit exactly the same directory.

    Never raises: both callers of :func:`render_report` treat a report that did not get written as
    a warning rather than a failed run, and a decoration must not be what spends that.
    """
    try:
        from spatialomicsgym.report import discover

        run = Path(results_dir)
        shown, total, capped = discover.tool_figures(run, run.parent if root is None else root)
    except Exception:
        return ""
    if not shown and not capped:
        return ""
    lead = (
        "<h2>Figures the tool produced</h2>"
        "<p class='quiet'>Written by the analysis tool itself, in the directory this results folder "
        "sits in. Post-analysis did not generate or check these, and they are listed by name rather "
        "than embedded so this file stays small enough to send.</p>"
    )
    if not shown:
        # Nothing found and nothing read are different answers. A 0700 output directory or one
        # holding 20,000 per-gene logs comes back empty with ``capped``, and dropping the heading
        # there tells the reader of a failed run that the tool drew nothing -- the claim R79/P
        # closed on the portal side.
        return lead + (
            "<p class='quiet'>This report did not finish reading that directory, and reached no "
            "images before the scan stopped. Open it to see what the tool wrote.</p>"
        )
    # ``N+`` and not ``N`` once the scan hit a ceiling: the count stopped there, so an exact figure
    # asserts a number nobody established.
    counted = f"{total}+" if capped else str(total)
    items = "".join(
        f"<li class='mono'>{_esc(item['name'])} <span class='quiet'>{_esc(_human_size(item['size']))}</span></li>"
        for item in shown
    )
    more = (
        f"<p class='quiet'>Showing {len(shown)} of {counted}; the rest are in the same directory.</p>"
        if capped or total > len(shown)
        else ""
    )
    return f"{lead}<p>{counted} image(s):</p><ul class='plain'>{items}</ul>{more}"


def render_report(
    manifest: dict,
    results_dir: Path | str,
    *,
    figure_src: Callable[[str], str | None] | None = None,
    table_href: Callable[[str], str | None] | None = None,
    heading: str | None = None,
    run_name: str = "",
    generated_at: str | None = None,
    tool_figures_root: Path | str | None = None,
    text_filter: Callable[[Any], str] | None = None,
) -> str:
    """A complete, standalone HTML document for one analysis run.

    ``text_filter`` is applied to every piece of recorded text on the page -- the ``<title>`` and the
    tool-figures note included, not only the body -- exactly as :func:`render_report_body` applies it.
    The portal passes the same filter its run page uses (MED-8, hunt 2026-09-30 u17-cli-report-15:
    the printable report was the one surface that still printed this machine's directories).

    With no ``figure_src`` the document is fully self-contained (base64 figures, inline CSS, zero
    external requests, zero JavaScript) -- safe to email or copy to a USB stick.

    See :func:`render_report_body` for ``run_name``: it is the run's name as a caller that knows the
    root computes it, and it is what keeps this document agreeing with the portal card that served it.

    ``tool_figures_root`` is how far this document may look outside the run for the tool's own
    plots. A caller serving the report over HTTP has an allowed region and should name it; a caller
    writing the file locally has none, and omitting it keeps the run's own parent. See
    :func:`_tool_figures_note`, which is the only thing that reads it.
    """
    m = dict(manifest)
    # The tab names the *run*, and never the tool -- which is why this is not :func:`_run_heading`
    # with its arguments in a different order. A browser tab is an identifier among tabs, and a
    # recorded ``tool_name`` is shared by every run of that tool: two samples through one tool
    # opened two tabs both reading ``run_spagcn``, with nothing else in a tab to tell them apart.
    # The heading below can afford the tool name because the page prints the directory in its
    # Provenance row a few lines down. R80/V ruled the same way for the downloaded filename, on the
    # same grounds: the tool name "is the true answer to 'what wrote this' and the wrong answer to
    # 'what is this file called'".
    #
    # Both doors take this rule, not just the portal. ``write_report`` leaves one of these in each
    # run's own directory and two of them open the same two tabs; it has no root, so it names the
    # run the way :func:`_run_heading` does without one, which is what the title already read on
    # every run carrying no ``tool_name`` -- all 92 recorded ones. Keeping the rule shared is also
    # what holds R81/B: at L1's own depth the two ways of naming a run agree character for
    # character, so the served page and the written file stay byte-identical there.
    #
    # An explicit ``heading=`` still wins over the derived name, and still loses to ``run_name``:
    # it is a caller naming this one page, where ``tool_name`` names a whole sweep of them.
    title = run_name or heading or run_label(Path(results_dir)) or "unknown"
    body = render_report_body(
        m,
        results_dir,
        figure_src=figure_src,
        table_href=table_href,
        heading=heading,
        run_name=run_name,
        generated_at=generated_at,
        text_filter=text_filter,
    )
    token = _TEXT_FILTER.set(text_filter or redact_text)
    try:
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(title)} &mdash; SpatialOmicsLab analysis report</title>"
            f"<style>{REPORT_CSS}</style></head><body><div class='wrap'>"
            f"{body}{_tool_figures_note(results_dir, tool_figures_root)}"
            "<div class='foot'>Generated by SpatialOmicsLab (ST-Coscientist) from this run&rsquo;s "
            "manifest. Self-contained: figures are embedded, nothing is loaded from the network.</div>"
            "</div></body></html>"
        )
    finally:
        _TEXT_FILTER.reset(token)


def write_report(
    results_dir: Path | str,
    *,
    manifest: dict | None = None,
    filename: str = "report.html",
    generated_at: str | None = None,
) -> Path:
    """Render ``<results_dir>/report.html`` from that directory's manifest and return its path.

    Writes nothing outside ``results_dir``, and writes atomically (a staging file in the same
    directory, then ``os.replace``) so a reader never sees a half-written report. The staging name is
    unique per writer -- see :func:`~spatialomicsgym.postanalysis.manifest.staging_name` for why the
    destination's own name was not, and why it is shared with the two manifest writers. Raises
    :class:`~spatialomicsgym.report.manifest.ManifestError` if there is no readable manifest, and
    ``OSError`` if the directory cannot be written -- both are the caller's to handle, because L1
    must be free to treat a failed report as a warning rather than a failed run.

    With no ``generated_at`` the stamp is the run's own analysis time (:func:`analysed_at`), not the
    moment of the call. The two are the same thing when L1 writes the report immediately after the
    run -- but this is also the by-hand door, and ``python -m spatialomicsgym.report`` exists, in its
    own words, for "a run that finished before the report generator existed, a directory copied off
    a cluster". Stamping those with today re-dates work nobody redid. It also makes re-rendering
    idempotent, which is what lets the portal's served page and this file be compared at all: one
    renderer over one manifest with one stamp, differing only in how they name the run, because a
    page knows the root it resolved under and a file that can be copied anywhere does not (see
    :func:`~sog_portal.api.routers.results.api_results_report`). ``None`` from
    :func:`analysed_at` (no readable manifest mtime) falls through to ``now``.

    Refuses to write under ``benchmarks/results/`` -- contract non-negotiable 1a, which L1 has
    enforced since it was written and L3 did not. The gap was reachable: this is the by-hand door,
    ``python -m spatialomicsgym.report benchmarks/`` walks whatever it is pointed at, and two
    symlinks in ``benchmarks/manuscript/`` lead straight into the 669 GB of recorded output. The
    same resolving predicate L1 uses decides it, so a symlinked spelling does not step over the
    check. The refusal is a ``PermissionError``: both callers already treat ``OSError`` as a report
    that did not get written rather than a run that failed, which is exactly the right outcome.
    """
    d = Path(results_dir)
    if leads_under_benchmark_results(d):
        raise PermissionError(
            f"Refusing to write a report into {d}: benchmarks/results/ holds recorded experiment "
            "outputs and is read-only. Copy the run elsewhere first if you want a report of it."
        )
    m = manifest if manifest is not None else load(d)
    doc = render_report(m, d, generated_at=generated_at or analysed_at(d))
    out = d / filename
    tmp = d / staging_name(filename)
    try:
        with open_staging(tmp) as fh:
            fh.write(doc)
        os.replace(tmp, out)
    except BaseException:
        # ``BaseException``, not ``OSError``: a Ctrl-C landing between the write and the replace is
        # the case that strands a staging file, and it is not an ``OSError``. It costs nothing when
        # the name is shared, since the next writer truncates it, but the name is now unique per
        # writer, so an orphan is permanent. See :func:`~...postanalysis.manifest.discard_staging`.
        discard_staging(tmp)
        raise
    return out
