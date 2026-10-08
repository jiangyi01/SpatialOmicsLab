"""
The append-only record of what each research round did, and the rule that reads it.

Why append-only, and why a second file beside ``research.json``. The loop already persists a
journal, and it persists it by rewriting the whole file after every round. On a real run that died
inside round 1 (``research_20260917_114717``) that left a ``round_1/`` directory and **no
``research.json`` at all** -- while ``loop.py`` states the opposite as a guarantee: "persists
``research.json`` after every round, so a run that dies at round 3 still renders 1-2". A whole-file
rewrite cannot make that promise about the round it is inside of.

So: one line appended when a round STARTS, one appended when it ends. A run killed halfway leaves
a row saying a round began and never closed, which is a true and useful thing to read. This is the
argument ``sog_portal/history.py`` made for the run journal while it was a file (the run history is the ``runs`` table
of ``portal.db`` since 2026-10-07) -- "a history file is the one
artifact that must never be rewritten to be extended" -- applied to the one other file here with
the same shape.

The other half of this module is :func:`verdict`, which is why a run stops. A real run on this box
used **1 round of 4 in 5.9 seconds of a 3600-second budget** and reported ``steps_exhausted``.
Whatever internal budget that named, the sentence a person reads said the run used up its
allowance when it had barely started. A stop reason should say what the NUMBERS did, and that
needs numbers, which is the other reason this file exists.

The reading rule is adapted from ``uditgoenka/autoresearch`` (MIT, commit 050e30dc, itself after
Karpathy's autoresearch); see ``THIRD_PARTY_LICENSES/autoresearch-MIT.txt``. One detail is taken
deliberately and is easy to get wrong: a round whose metric could not be computed is **excluded**
from the plateau window rather than read as zero progress, because a crashed round is not evidence
that the analysis stopped improving. A run of nothing but uncomputable rounds is ``blocked``, which
is a different problem with a different fix.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Beside ``research.json`` in the run directory.
LEDGER_NAME = "rounds.tsv"

#: The columns, in order. Adding one at the END is backward-compatible -- :func:`read_rounds` maps
#: by header name, not by position, so an old file still reads.
COLUMNS: tuple[str, ...] = (
    "round",
    "at",
    "event",
    "metric",
    "guard",
    "status",
    "results_dir",
    "note",
)

#: ``metric`` when the round produced no number. Not ``0``: a zero is a measurement and this is the
#: absence of one, and the difference is the whole of :func:`verdict`'s correctness.
UNKNOWN = "unknown"

#: How many rounds that produced a number the plateau rule looks at.
WINDOW = 5

VERDICT_PROGRESSING = "progressing"
VERDICT_PLATEAU = "plateau"
VERDICT_BLOCKED = "blocked"
VERDICT_UNDECIDED = "undecided"

HIGHER = "higher_is_better"
LOWER = "lower_is_better"


def ledger_path(directory: str | Path) -> Path:
    return Path(directory) / LEDGER_NAME


def open_ledger(directory: str | Path, *, direction: str = "") -> Path:
    """Create the file with its two header lines, or leave an existing one alone.

    The direction goes in a comment on line 1 rather than in a column, because it is a property of
    the RUN and repeating it on every row invites two rows to disagree about which way is better.
    A reader that finds no direction says so instead of guessing: guessing it is how a run reports
    every round as an improvement.
    """
    path = ledger_path(directory)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    head = f"# metric_direction: {_direction(direction) or 'unknown'}\n" + "\t".join(COLUMNS) + "\n"
    # Exclusive create, so two writers racing to start a run cannot both write a header.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # pragma: no cover - the race this guard exists for
        return path
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(head)
    return path


def begin_round(directory: str | Path, number: int, *, change: str = "", now: float | None = None) -> None:
    """Record that a round STARTED, before it runs.

    This is the line that makes a killed run legible. It is written first and deliberately carries
    no metric: there is not one yet, and a row that claimed one would be the file lying about a
    round that has not happened.
    """
    _append(directory, {"round": number, "event": "begin", "metric": UNKNOWN, "status": "running", "note": change}, now)


def close_round(
    directory: str | Path,
    number: int,
    *,
    metric: float | None = None,
    guard: str = "",
    status: str = "",
    results_dir: str = "",
    note: str = "",
    now: float | None = None,
) -> None:
    """Record how a round ended. Appended, never written over the ``begin`` row."""
    _append(
        directory,
        {
            "round": number,
            "event": "end",
            "metric": UNKNOWN if metric is None else _number(metric),
            "guard": guard,
            "status": status or ("keep" if metric is not None else UNKNOWN),
            "results_dir": results_dir,
            "note": note,
        },
        now,
    )


def read_rounds(directory: str | Path) -> list[dict[str, str]]:
    """Every row, in order, mapped by header name.

    Tolerant by construction, because this file is appended to by a process that can be killed
    mid-write: a torn last line, a short row, an unknown column and a missing file all read as
    "that much is known" rather than as an exception. A ledger that raises is worse than a ledger
    that is one row short -- the caller is usually trying to explain a crash.
    """
    path = ledger_path(directory)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    header: list[str] = []
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if not header:
            header = parts
            continue
        if len(parts) < 2:
            # A torn tail: the process died between the write and the newline.
            continue
        rows.append({name: (parts[i] if i < len(parts) else "") for i, name in enumerate(header)})
    return rows


def declare_direction(directory: str | Path, direction: str, *, now: float | None = None) -> bool:
    """Record which way is better, once the run knows.

    Appended as a comment rather than written over line 1, because the file is append-only and a
    rewrite is how a concurrent append is lost. The ledger has to be opened BEFORE round 1 -- that
    is what makes a killed run legible -- and the direction is something round 1 establishes, so
    the two cannot both happen at the header.

    Refuses to change a direction already declared. A loop that flips its own direction mid-run
    reports every round as an improvement, and the flip would be invisible in the numbers.
    """
    wanted = _direction(direction)
    if not wanted:
        return False
    if direction_of(directory):
        return False
    path = ledger_path(directory)
    stamp = time.time() if now is None else float(now)
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"# metric_direction: {wanted} (declared at {stamp:.3f})\n")
    except OSError:
        return False
    return True


def direction_of(directory: str | Path) -> str:
    """The direction the run declared, or ``""`` when it declared none.

    Reads every comment, not only the first line: the header is written before round 1 and the
    direction is declared by round 1, so it arrives later in the file. FIRST declaration wins --
    :func:`declare_direction` already refuses a second, and if one is there anyway (a file edited
    by hand, two processes racing) the honest reading is the one the earlier rounds were scored
    under, not the one that appeared last.
    """
    path = ledger_path(directory)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if not line.startswith("#") or "metric_direction:" not in line:
            continue
        _, _, value = line.partition("metric_direction:")
        found = _direction(value.split("(")[0])
        if found:
            return found
    return ""


def metrics(rows: Sequence[dict[str, str]]) -> list[float | None]:
    """One entry per CLOSED round, in order: the number, or ``None`` when there was not one.

    ``begin`` rows are skipped -- a round that started is not a round that measured -- and a round
    closed twice keeps its last close, which is what a retry means.
    """
    seen: dict[str, float | None] = {}
    order: list[str] = []
    for row in rows:
        if str(row.get("event", "")).strip() != "end":
            continue
        key = str(row.get("round", ""))
        if key not in seen:
            order.append(key)
        raw = str(row.get("metric", "")).strip()
        try:
            seen[key] = None if raw in ("", UNKNOWN) else float(raw)
        except ValueError:
            seen[key] = None
    return [seen[key] for key in order]


def verdict(values: Sequence[float | None], direction: str, *, window: int = WINDOW) -> str:
    """Why a run should stop, from what its numbers did.

    ``progressing`` -- the metric is still net-improving across the window, so keep going.
    ``plateau``     -- it is not. Oscillation that nets flat-or-worse is a plateau, which is why
                       this compares the ends of the window rather than counting consecutive
                       equal values.
    ``blocked``     -- nothing can be computed: no round produced a number, or the last ``window``
                       closed rounds all failed to. A crashed runner must not read as convergence.
    ``undecided``   -- fewer than ``window`` computed values. Not a plateau, and saying so is the
                       point: two flat rounds are not a trend. Also the answer when the run
                       declared no direction, because "better" is not defined without one.

    An uncomputable round is excluded from the window rather than counted as zero progress. That
    exclusion is the one rule here that is easy to get wrong and expensive when it is: a run whose
    every other round crashes would otherwise report a plateau it never had.
    """
    entries = list(values)
    if not entries:
        return VERDICT_BLOCKED
    trailing = 0
    for value in reversed(entries):
        if value is None:
            trailing += 1
        else:
            break
    computed = [v for v in entries if v is not None]
    if not computed:
        return VERDICT_BLOCKED
    if trailing >= max(1, int(window)):
        return VERDICT_BLOCKED
    span = computed[-max(2, int(window)) :]
    if len(span) < max(2, int(window)):
        return VERDICT_UNDECIDED
    declared = _direction(direction)
    if not declared:
        # No direction, no trend. This used to fall through to higher-is-better, so a loss the run
        # drove from 0.9 to 0.1 read "levelled off" and one that rose read "still improving" --
        # the guess :func:`open_ledger` says a reader must not make (hunt 2026-09-30,
        # u19-pa-tasks-research-2).
        return VERDICT_UNDECIDED
    better = span[-1] > span[0] if declared == HIGHER else span[-1] < span[0]
    return VERDICT_PROGRESSING if better else VERDICT_PLATEAU


def summary(directory: str | Path) -> dict[str, Any]:
    """What the ledger says, for a report or a stop sentence."""
    rows = read_rounds(directory)
    values = metrics(rows)
    computed = [v for v in values if v is not None]
    started = {str(r.get("round", "")) for r in rows if str(r.get("event", "")) == "begin"}
    ended = {str(r.get("round", "")) for r in rows if str(r.get("event", "")) == "end"}
    return {
        "direction": direction_of(directory),
        # The series, not only its ends. A chart of a run needs every round -- including the
        # ones that produced nothing, which travel as `null` and must stay distinguishable from
        # a measured zero all the way to the axis.
        "values": values,
        "rounds_started": len(started),
        "rounds_closed": len(ended),
        # A round that began and never ended is a run that was killed inside it, and the whole
        # reason the begin row is written first.
        "unfinished": sorted(started - ended),
        "metric_first": computed[0] if computed else None,
        "metric_last": computed[-1] if computed else None,
        "uncomputable": sum(1 for v in values if v is None),
        "verdict": verdict(values, direction_of(directory)),
    }


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _direction(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in (HIGHER, LOWER) else ""


def _number(value: float) -> str:
    """Enough digits to compare two rounds, and no scientific notation to parse back."""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return UNKNOWN


def _clean(value: object) -> str:
    """A field that cannot break the format it is written into.

    A tab or a newline from a tool's own error text would silently become a column boundary or a
    new row, and the row it corrupted would be the one explaining the failure.
    """
    return " ".join(str(value if value is not None else "").split())[:400]


def _append(directory: str | Path, fields: dict[str, Any], now: float | None) -> None:
    """One ``O_APPEND`` write of one complete line.

    A single write of a line that ends in a newline is what makes a concurrent append safe: two
    writers interleave whole lines rather than halves of them. Never raises -- a research run must
    not die because its bookkeeping could not be written.
    """
    path = ledger_path(directory)
    stamp = time.time() if now is None else float(now)
    values = {"at": f"{stamp:.3f}", **{k: v for k, v in fields.items() if v is not None}}
    line = "\t".join(_clean(values.get(name, "")) for name in COLUMNS) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except OSError:
        return
