"""
Friendly, stdlib-only install progress indicators.

A conda ``env create`` or ``pip install`` can run for minutes while printing nothing to
the user (its own output is captured for the log). Two indicators cover that gap:

* :class:`InstallProgress` — a background daemon thread rewriting ONE terminal line
  (ASCII spinner + elapsed + task label) while a single subprocess runs, then clearing
  it. So a lone build now *looks* alive ("  / building env sog_spatialde   42s") instead
  of appearing hung.
* :class:`StepBox` — a small bordered **checklist box** for a multi-step tool build. Done
  steps show ``✓``, the active step spins with its elapsed time, pending steps are a dim
  ``·``; on completion the whole box is erased so the caller's single ``✅ … ready`` line
  is the collapse target. Inside a :meth:`StepBox.build` frame, :meth:`StepBox.task`
  advances the active step instead of drawing its own line; with no frame open it falls
  back to the one-line spinner, so base-env creates are unchanged.

Both animate only on a real TTY. On a pipe, a CI run, a scripted (``--answers``) run, a
``--dry-run``, or when ``SOG_SETUP_NO_PROGRESS`` asks for quiet (see :func:`progress_muted`
for which spellings do and do not), they are a complete no-op — so
logs stay clean and the unit suite (which writes to ``StringIO``) is never touched. The
box auto-degrades to ASCII markers/borders when the stream can't encode the Unicode glyphs,
so it can't wedge a dumb terminal.

Stdlib only.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING

from .session_log import redact

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO

_FRAMES = "|/-\\"  # ascii only — safe on any terminal
_MIN_WIDTH = 20
_MAX_WIDTH = 100
_THOUGHT_LINES = 5  # default constant height of the thinking-box live body (env-overridable, see _thought_lines)


def _thought_lines() -> int:
    """Height of the thinking box's live thought body (top-padded so a redraw never orphans rows).

    ``SOG_SETUP_THOUGHT_LINES`` overrides the default so a user who wants more of the *current*
    step on screen can grow the live window. Completeness never depends on this: the FULL,
    unclamped text of every finished step is preserved in the durable scrollback transcript
    (see :class:`ThoughtBox`) regardless of the live height."""
    raw = os.environ.get("SOG_SETUP_THOUGHT_LINES")
    if raw:
        try:
            return max(3, min(40, int(raw.strip())))
        except (ValueError, AttributeError):
            pass
    return _THOUGHT_LINES


def _transcript_enabled() -> bool:
    """Whether completed ReAct steps are committed to the durable scrollback transcript (default ON).

    ``SOG_SETUP_THOUGHT_TRANSCRIPT=0`` (or ``false``/``no``/``off``) reverts to the pre-transcript
    ephemeral box — the live screenshot only, nothing left behind — for anyone who finds the
    growing scrollback noisy."""
    return os.environ.get("SOG_SETUP_THOUGHT_TRANSCRIPT", "1").strip().lower() not in ("0", "false", "no", "off")


#: Spellings of ``SOG_SETUP_NO_PROGRESS`` that do NOT mute. ``""`` is in the list because a blank
#: value reads as unset package-wide (the rule ``config._env_raw`` applies), so a conditional
#: ``SOG_SETUP_NO_PROGRESS=`` in a CI file keeps its progress display.
_NOT_MUTED = ("", "0", "false", "no", "off")


def progress_muted() -> bool:
    """Whether the user has asked for a silent install (default: no).

    This is a *presence* switch, not a boolean feature toggle — it sits beside "a pipe, a CI run,
    a scripted run" in the module docstring, and ``NO_COLOR`` next to it in ``prompts.py`` works the
    same way. So any non-blank value mutes... except the four spellings this project documents as
    *no*. Reading ``SOG_SETUP_NO_PROGRESS=0`` as "yes, be quiet" was the reverse of what it says,
    and it disagreed with ``SOG_SETUP_THOUGHT_TRANSCRIPT`` six lines above, which understands ``0``.

    Both gates that mute the display call this, so the two cannot drift into two dialects again:
    the spinner/step-box here, and the tool picker's ANSI redraw in ``prompts.PromptIO``.
    """
    return os.environ.get("SOG_SETUP_NO_PROGRESS", "").strip().lower() not in _NOT_MUTED


def _auto_enabled(stream: TextIO) -> bool:
    """Animate only on a real interactive terminal, and never when explicitly muted."""
    if progress_muted():
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def _term_width() -> int:
    try:
        return max(_MIN_WIDTH, min(_MAX_WIDTH, shutil.get_terminal_size((80, 24)).columns))
    except (ValueError, OSError):
        return 80


class InstallProgress:
    """Live single-line spinner for long, output-captured subprocesses.

    Usage::

        prog = InstallProgress(sys.stdout)
        with prog.task("building env sog_spatialde"):
            subprocess.run(...)  # captured; the spinner shows liveness meanwhile
    """

    def __init__(self, stream: TextIO | None = None, *, enabled: bool | None = None, interval: float = 0.15) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.enabled = _auto_enabled(self.stream) if enabled is None else enabled
        self.interval = max(0.02, interval)

    @contextlib.contextmanager
    def task(self, label: str) -> Iterator[None]:
        """Show a spinner for ``label`` until the block exits (or a no-op if disabled)."""
        if not self.enabled:
            yield
            return
        stop = threading.Event()
        thread = threading.Thread(target=self._animate, args=(label, stop), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1.0)
            self._clear()

    # -- internals ------------------------------------------------------------
    def _animate(self, label: str, stop: threading.Event) -> None:
        start = time.monotonic()
        for frame in itertools.cycle(_FRAMES):
            if stop.is_set():
                break
            elapsed = int(time.monotonic() - start)
            line = f"  {frame} {label}   {elapsed}s"
            width = _term_width()
            self._write("\r" + line[:width].ljust(width))
            stop.wait(self.interval)

    def _write(self, text: str) -> None:
        try:
            self.stream.write(self._encodable(text))
            self.stream.flush()
        except (ValueError, OSError):  # stream closed mid-run — give up silently
            self.enabled = False

    def _encodable(self, text: str) -> str:
        """Coerce ``text`` to what the stream can actually encode, so agent content outside the terminal's
        charset (a Greek label, a µ, an emoji, a stray … from a truncator) can't raise ``UnicodeEncodeError``
        (a ``ValueError``) and wedge the whole box via the handler above. Width-preserving — each un-encodable
        code point becomes one ``?`` so box geometry stays intact — and a NO-OP on a UTF-8 stream (the common
        case), so the default self-heal render stays byte-for-byte identical."""
        enc = getattr(self.stream, "encoding", None)
        if not enc:
            return text
        try:
            text.encode(enc)
        except (UnicodeError, LookupError):
            text = text.encode(enc, "replace").decode(enc, "replace")
        return text

    def _clear(self) -> None:
        width = _term_width()
        self._write("\r" + " " * width + "\r")


class _BoxState:
    """Mutable render state for one open :meth:`StepBox.build` frame.

    Steps start ``pending``; :meth:`begin` flips the matching step ``active`` (and the
    previously-active one ``done``), :meth:`complete` flips it ``done``. An unrecognized
    label (a build that fell back to a different strategy mid-frame) is *appended* as a new
    active step, so the box grows to show the fallback happening rather than silently
    stalling.
    """

    def __init__(self, title: str, steps: list[str]) -> None:
        self.title = title
        self.steps = list(steps)
        self.state = ["pending"] * len(steps)  # pending | active | done
        self.start: dict[int, float] = {}  # step idx -> monotonic start (for elapsed)
        self.lines = 0  # lines drawn last render (cursor-up count)
        self.last_payload = ""  # skip a redraw when nothing visibly changed
        self.lock = threading.Lock()

    def begin(self, label: str) -> int:
        with self.lock:
            idx = next(
                (i for i, s in enumerate(self.steps) if s == label and self.state[i] != "done"),
                None,
            )
            if idx is None:  # a fallback strategy's label — surface it as a new step
                self.steps.append(label)
                self.state.append("pending")
                idx = len(self.steps) - 1
            for i, st in enumerate(self.state):
                if st == "active":
                    self.state[i] = "done"
            self.state[idx] = "active"
            self.start[idx] = time.monotonic()
            return idx

    def complete(self, idx: int) -> None:
        with self.lock:
            if 0 <= idx < len(self.state):
                self.state[idx] = "done"


class StepBox(InstallProgress):
    """A bordered, multi-step checklist that redraws in place, then erases itself.

    Open a frame around a tool build; each :meth:`task` call (from the conda wrapper)
    advances the active step::

        box = StepBox(sys.stdout)
        with box.build("tangram · pip · 3/4", ["creating env sog_tangram", "pip installing into sog_tangram"]):
            conda.create_named("sog_tangram")  # -> task("creating env sog_tangram")
            conda.pip_install("sog_tangram", ...)  # -> task("pip installing into sog_tangram")
        io.ok("sog_tangram ready")  # prints where the (now-erased) box was

    Muted / non-TTY / ``StringIO`` → :meth:`build` and :meth:`task` are complete no-ops,
    identical to :class:`NullProgress`.
    """

    def __init__(self, stream: TextIO | None = None, *, enabled: bool | None = None, interval: float = 0.15) -> None:
        super().__init__(stream, enabled=enabled, interval=interval)
        self._box: _BoxState | None = None
        self._pick_glyphs()

    # -- public API -----------------------------------------------------------
    @contextlib.contextmanager
    def build(self, title: str, steps: list[str]) -> Iterator[None]:
        """Draw a checklist box for ``steps`` and animate it until the block exits.

        A no-op (plain ``yield``) when disabled or when ``steps`` is empty, so callers can
        wrap unconditionally. Only one frame is open at a time; nested calls are not used.
        """
        steps = [s for s in steps if s]
        if not self.enabled or not steps:
            yield
            return
        box = _BoxState(title, steps)
        self._box = box
        stop = threading.Event()
        self._render_box(box, force=True)
        thread = threading.Thread(target=self._animate_box, args=(box, stop), daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=1.0)
            self._erase_box(box)
            self._box = None

    @contextlib.contextmanager
    def task(self, label: str) -> Iterator[None]:
        """Inside a :meth:`build` frame, advance the active step; else the one-line spinner."""
        box = self._box
        if box is None or not self.enabled:
            with super().task(label):
                yield
            return
        idx = box.begin(label)
        try:
            yield
        finally:
            box.complete(idx)

    # -- glyphs (Unicode with an ASCII fallback) ------------------------------
    def _pick_glyphs(self) -> None:
        uni = "✓·◜◝◞◟─│╭╮╰╯"
        enc = getattr(self.stream, "encoding", None) or "ascii"
        try:
            uni.encode(enc)
            ok = True
        except (UnicodeEncodeError, LookupError):
            ok = False
        if ok:
            self._done, self._pending = "✓", "·"
            self._spin = "◜◝◞◟"  # a rotating marker — the active step's "spins ⟳"
            self._b = {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"}
            self._sep = "·"
        else:
            self._done, self._pending = "v", "."
            self._spin = _FRAMES
            self._b = {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"}
            self._sep = "-"

    # -- rendering ------------------------------------------------------------
    def _animate_box(self, box: _BoxState, stop: threading.Event) -> None:
        for _ in itertools.count():
            if stop.is_set():
                break
            self._render_box(box)
            stop.wait(self.interval)

    def _render_box(self, box: _BoxState, *, force: bool = False) -> None:
        width = _term_width()
        with box.lock:
            commits = self._drain_commit_lines_locked(box, width)  # durable lines to leave ABOVE the box
            lines = self._compose_locked(box, width)
            payload = "\n".join(lines)
            if not force and not commits and payload == box.last_payload:
                return
            up = box.lines
            box.last_payload = payload
            box.lines = len(lines)
            # Emit UNDER the lock (B1). The write must be atomic with the `box.lines`/`up` mutation just
            # above: think()/build() join the animator with timeout=1.0, so a write blocked on terminal
            # back-pressure (paused tty, slow SSH, full pipe) can leave the daemon ALIVE past the join —
            # then the teardown thread's own _render_box/_erase_box write races the daemon's, interleaving
            # ANSI with a stale cursor-up count (the half-drawn-border corruption R2 set out to kill; the
            # "sole writer" invariant that join implied was false). Serialising here also means the just-
            # drained `commits` can't be lost if the write fails. Move to the box top, emit durable
            # committed lines (they scroll into history, never repainted), then redraw the constant-height
            # box below. (Mirrors envtools._tee_line's single lock-guarded sink.)
            committed = "".join(f"{c}\x1b[K\n" for c in commits)
            body = "\n".join(f"{ln}\x1b[K" for ln in lines) + "\n"
            self._write((f"\x1b[{up}A" if up else "") + committed + body)

    def _drain_commit_lines_locked(self, box: _BoxState, width: int) -> list[str]:
        """Hook: durable transcript lines to flush ABOVE the box on this redraw, then forget.

        The checklist box has none (its steps are the box); :class:`ThoughtBox` overrides this to
        pop finished ReAct steps off ``box.pending_commits``. Caller MUST hold ``box.lock``."""
        return []

    def _compose_locked(self, box: _BoxState, width: int) -> list[str]:
        """Build the box's text lines. Caller MUST hold ``box.lock``."""
        frame = self._spin[int(time.monotonic() * 6) % len(self._spin)]
        rows = [self._top(box.title, width)]
        for i, label in enumerate(box.steps):
            st = box.state[i]
            if st == "done":
                text = f"{self._done} {label}"
            elif st == "active":
                elapsed = int(time.monotonic() - box.start.get(i, time.monotonic()))
                text = f"{frame} {label}   {elapsed}s"
            else:
                text = f"{self._pending} {label}"
            rows.append(self._row(text, width))
        rows.append(self._bottom(width))
        return rows

    def _top(self, title: str, width: int) -> str:
        b = self._b
        title = title.replace("·", self._sep)
        label = f" {title} "
        head = b["tl"] + b["h"]
        fill = max(0, width - len(head) - len(label) - 1)
        line = head + label + b["h"] * fill + b["tr"]
        return line[:width]

    def _row(self, text: str, width: int) -> str:
        b = self._b
        inner = max(1, width - 4)
        text = text[:inner].ljust(inner)
        return f"{b['v']} {text} {b['v']}"

    def _bottom(self, width: int) -> str:
        b = self._b
        return b["bl"] + b["h"] * max(0, width - 2) + b["br"]

    def _erase_box(self, box: _BoxState) -> None:
        """Erase the whole box so the next printed line lands where the box was."""
        with box.lock:  # serialise with a still-live animator's _render_box write (B1)
            n = box.lines
            if not n:
                return
            self._write(f"\x1b[{n}A" + "\x1b[2K\n" * n + f"\x1b[{n}A")
            box.lines = 0


class _ThoughtState(_BoxState):
    """Render model for one open :meth:`ThoughtBox.think` frame — a live "thinking screenshot".

    Reuses ``_BoxState``'s lock + redraw bookkeeping (``lines`` / ``last_payload``) with no steps;
    the changing content is the ``mode`` / ``status`` / ``thought`` / ``action`` / ``footer`` a feeder
    pushes each turn. ``title`` + ``footer`` are ``redact``'d here so nothing secret can reach the
    terminal even if a feeder forgets (the handle redacts again on every update — belt and braces).
    """

    def __init__(self, title: str, *, footer: str | None = None, compact: bool = False) -> None:
        super().__init__(redact(title), [])
        self.mode = "thinking"  # thinking | planning | acting | surface
        self.status = ""
        self.label = ""  # human step name (from _agent_probe._step_label), shown in compact mode
        self.thought = ""
        self.action = ""
        self.footer = redact(footer) if footer else ""
        self.t0 = time.monotonic()
        self.compact = compact  # opt-in one-liner timeline (demo / real-run); default keeps full detail
        self.pending_commits: list[dict[str, object]] = []  # finished steps queued for the durable transcript

    def queue_commit(self) -> None:
        """Snapshot the CURRENT (about-to-be-replaced) step for the durable scrollback transcript.

        Called at a ReAct-step boundary and once more on frame close, so every step's full context is
        preserved above the live box. A no-op when the transcript is disabled or the step carries no
        content yet (an empty opening frame). Caller MUST hold ``self.lock`` (the render/update lock)."""
        if not _transcript_enabled():
            return
        if not (self.thought or self.action):  # nothing meaningful to preserve
            return
        self.pending_commits.append(
            {
                "mode": self.mode,
                "status": self.status,
                "label": self.label,
                "thought": self.thought,
                "action": self.action,
                "elapsed": int(time.monotonic() - self.t0),
            }
        )


class ThoughtBox(StepBox):
    """A bordered box that redraws the agent's *current* thinking in place, then erases itself.

    One shared renderer for both wizard agents: the in-process env self-heal agent
    (:mod:`installer_scientist`) and the cross-process real-run agent (streamed from
    ``_agent_probe`` through :mod:`testing`). A feeder opens a frame and pushes updates::

        with progress.think(conda.progress, "ST-Coscientist", footer="reasoning live") as box:
            box.update(
                mode="acting",
                status="scanpy_spatial",
                thought="4035 spots — normalize…",
                action="sc.pp.normalize_total(adata)",
            )

    The animator thread is the SOLE writer (R2); :meth:`update` only mutates state under the lock.
    Disabled / non-TTY / an already-open checklist frame ⇒ a complete no-op, exactly like the
    :meth:`StepBox.build` path it mirrors.
    """

    def _pick_glyphs(self) -> None:
        super()._pick_glyphs()
        self._arrow = "▸" if self._done == "✓" else ">"  # the action-row marker
        self._ellipsis = "…" if self._done == "✓" else "..."  # truncation marker; ASCII-safe on a dumb term

    @contextlib.contextmanager
    def think(
        self, title: str, *, footer: str | None = None, compact: bool = False
    ) -> Iterator[_ThoughtHandle | _NullThoughtHandle]:
        """Draw a live thinking box and animate it until the block exits (inert if disabled).

        Inert (yields an inert handle, spawns no thread, writes zero bytes) when disabled or when a
        checklist ``build`` frame is already open on this object — the single-frame invariant that
        keeps two boxes from fighting the same terminal region.

        ``compact`` (opt-in, demo / real-run) renders a one-liner timeline: the live body shows only
        the current step and each finished step commits as a single ``✓ <label> (Ns)`` line. The
        default keeps the full multi-row body + full-thought commits the self-heal agent relies on.
        """
        if not self.enabled or self._box is not None:
            yield _NullThoughtHandle()
            return
        box = _ThoughtState(title, footer=footer, compact=compact)
        self._box = box
        stop = threading.Event()
        self._render_box(box, force=True)
        thread = threading.Thread(target=self._animate_box, args=(box, stop), daemon=True)
        thread.start()
        try:
            yield _ThoughtHandle(box)
        finally:
            stop.set()
            thread.join(timeout=1.0)  # best-effort stop; box.lock (B1) serialises writes even if it lingers
            with box.lock:
                box.queue_commit()  # preserve the final step in the durable transcript…
            self._render_box(box, force=True)  # …flush it ABOVE the box, then
            self._erase_box(box)  # …erase only the (now-redundant) live box, leaving the transcript
            self._box = None

    def _compose_locked(self, box: _BoxState, width: int) -> list[str]:  # type: ignore[override]
        """Render the thinking box. Caller MUST hold ``box.lock``. ``box`` is a ``_ThoughtState``.

        R1 (constant height): the thought body is ALWAYS exactly ``_THOUGHT_LINES`` rows (top-padded)
        and the ``▸`` action row is ALWAYS present, so the inherited cursor-up redraw can never leave
        an orphan row below a frame that shrank.
        """
        if not isinstance(box, _ThoughtState):  # a checklist build() frame opened on this object
            return super()._compose_locked(box, width)
        if getattr(box, "compact", False):  # opt-in one-liner timeline: current step only
            frame = self._spin[int(time.monotonic() * 6) % len(self._spin)]
            elapsed = int(time.monotonic() - getattr(box, "t0", time.monotonic()))
            label = getattr(box, "label", "") or getattr(box, "mode", "thinking").capitalize()
            inner = max(1, width - 4)
            head = f"{frame} {label}".ljust(max(1, inner - 8)) + f"{elapsed}s"
            summary = self._summary_line(getattr(box, "thought", ""), width)
            rows = [self._top(f"{box.title} · working", width), self._row(head, width)]
            rows.append(self._row(f"{self._arrow} {summary}" if summary else self._arrow, width))
            rows.append(self._footer(getattr(box, "footer", ""), width))
            return rows
        inner = max(1, width - 4)
        frame = self._spin[int(time.monotonic() * 6) % len(self._spin)]
        elapsed = int(time.monotonic() - getattr(box, "t0", time.monotonic()))
        mode = getattr(box, "mode", "thinking") or "thinking"
        status = getattr(box, "status", "")
        meta = f"{elapsed}s   {status}" if status else f"{elapsed}s"
        mode_line = f"{frame} {mode.capitalize()}".ljust(12) + meta
        rows = [self._top(f"{box.title} · working", width)]
        rows.append(self._row(mode_line, width))
        rows.append(self._row("", width))
        for wrapped in self._wrap_clamp(getattr(box, "thought", ""), inner, _thought_lines()):
            rows.append(self._row(wrapped, width))
        action = getattr(box, "action", "")
        rows.append(self._row(f"{self._arrow} {action}" if action else self._arrow, width))
        rows.append(self._footer(getattr(box, "footer", ""), width))
        return rows

    def _drain_commit_lines_locked(self, box: _BoxState, width: int) -> list[str]:  # type: ignore[override]
        """Pop every finished ReAct step off ``box.pending_commits`` and format each as durable
        transcript lines to leave ABOVE the live box. Caller MUST hold ``box.lock`` (see the base
        ``_render_box``). The checklist box keeps the inherited no-op; only a thinking frame commits."""
        pend = getattr(box, "pending_commits", None)
        if not pend:
            return []
        compact = bool(getattr(box, "compact", False))
        out: list[str] = []
        for snap in pend:
            out.extend(self._format_commit(snap, width, compact=compact))
        pend.clear()
        return out

    def _format_commit(self, snap: dict, width: int, *, compact: bool = False) -> list[str]:
        """One finished step → permanent scrollback lines.

        Default (self-heal): header + FULL thought + action, NOT clamped — the whole reasoning is
        preserved so the user can scroll back and read every step in full. ``compact`` (demo /
        real-run): a single ``✓ <label> (Ns)`` line plus at most one indented ``→ <summary>`` — the
        terse timeline; the full detail lives in the written HTML/text transcript instead.

        Echoes the box's own vocabulary (``✓`` done marker, ``▸`` action arrow). Redacts defensively
        even though the handle already redacted each field on the way in."""
        if compact:
            label = redact(str(snap.get("label") or snap.get("mode") or "thinking"))
            elapsed = snap.get("elapsed")
            head = "  " + self._done + " " + label + (f"   ({elapsed}s)" if elapsed is not None else "")
            rows = [head[:width]]
            # the commit line prefixes an 8-col indent ("      ▸ ") before the summary, vs the live row's
            # 6 — so budget width-2 here, else _summary_line returns width-6 chars and [:width] silently
            # chops the last 2 (with no ellipsis to signal it). See progress.py compact-commit review.
            summary = self._summary_line(redact(str(snap.get("thought", ""))), width - 2)
            if summary:
                rows.append((f"      {self._arrow} {summary}")[:width])
            return rows
        mode = redact(str(snap.get("mode", "") or "thinking"))
        status = redact(str(snap.get("status", "")))
        head = "  " + self._done + " " + (f"{mode} {self._sep} {status}" if status else mode)
        elapsed = snap.get("elapsed")
        if elapsed is not None:
            head += f"   ({elapsed}s)"
        rows = [head[:width]]
        for ln in self._wrap(redact(str(snap.get("thought", ""))), max(1, width - 4)):
            rows.append(("    " + ln)[:width])
        action = redact(str(snap.get("action", "")))
        if action:
            # Wrap the action too — a long command (e.g. a full `conda run … pip install …@git+https…`
            # line) must be preserved in FULL, not truncated at the terminal edge. The first line
            # carries the ▸ arrow; continuations hang-indent under the text so the command reads as one.
            prefix = f"    {self._arrow} "
            cont = " " * len(prefix)
            wrapped = self._wrap(action, max(1, width - len(prefix)))
            rows.append((prefix + (wrapped[0] if wrapped else ""))[:width])
            for ln in wrapped[1:]:
                rows.append((cont + ln)[:width])
        return rows

    def _wrap(self, text: str, width: int) -> list[str]:
        """Greedy word-wrap ``text`` to ``width``, hard-breaking any token wider than the whole line.
        No clamp/pad — the complete text is returned (the durable transcript keeps everything)."""
        width = max(1, width)
        lines: list[str] = []
        cur = ""
        for word in (text or "").split():
            token = word
            while len(token) > width:  # a token wider than the whole line — hard-break it
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(token[:width])
                token = token[width:]
            if not token:
                continue
            if not cur:
                cur = token
            elif len(cur) + 1 + len(token) <= width:
                cur += " " + token
            else:
                lines.append(cur)
                cur = token
        if cur:
            lines.append(cur)
        return lines

    def _wrap_clamp(self, text: str, width: int, n: int) -> list[str]:
        """The live body: word-wrap ``text``, keep the LAST ``n`` lines, top-pad to EXACTLY ``n`` — so
        the newest words are always visible at a constant body height (see R1). The words that scroll
        off here are never lost: the full thought is preserved in the durable transcript on commit."""
        lines = self._wrap(text, width)[-n:]  # newest n lines
        return [""] * (n - len(lines)) + lines  # top-pad to constant height

    def _summary_line(self, text: str, width: int) -> str:
        """First clause of a (already-cleaned) thought, clipped to one box row — the compact ``→`` line."""
        flat = " ".join((text or "").split())
        for sep in (". ", "; ", " — "):
            if sep in flat:
                flat = flat.split(sep, 1)[0]
                break
        inner = max(1, width - 6)
        if len(flat) <= inner:
            return flat
        ell = getattr(self, "_ellipsis", "…")  # "…" on a UTF-8 term (identical to before), "..." on a dumb one
        return flat[: max(0, inner - len(ell))].rstrip() + ell

    def _footer(self, label: str, width: int) -> str:
        """A bottom border carrying a short label (mirrors :meth:`_top`); plain border if empty."""
        if not label:
            return self._bottom(width)
        b = self._b
        label = redact(label).replace("·", self._sep)
        text = f" {label} "
        head = b["bl"] + b["h"]
        fill = max(0, width - len(head) - len(text) - 1)
        return (head + text + b["h"] * fill + b["br"])[:width]


class _ThoughtHandle:
    """The live handle a feeder pushes updates through. Mutates ``_ThoughtState`` under its lock;
    the animator thread remains the sole terminal writer (R2). Every displayed field is ``redact``'d.
    ``.live`` is ``True`` so a feeder can route its own in-loop messaging into the box."""

    live = True

    def __init__(self, state: _ThoughtState) -> None:
        self._state = state

    def update(
        self,
        *,
        mode: str | None = None,
        status: str | None = None,
        label: str | None = None,
        thought: str | None = None,
        action: str | None = None,
        footer: str | None = None,
    ) -> None:
        st = self._state
        with st.lock:
            if thought is not None:
                new = redact(str(thought))
                # A ReAct-step boundary: a genuinely fresh thought means the previous step finished, so
                # commit it to the durable transcript before we overwrite it — nothing the agent reasoned
                # should scroll away unrecoverably. The ONLY non-boundary is token-streaming the SAME
                # thought: an identical re-emit, or a growing prefix-extension / shrinking truncation
                # BETWEEN TWO NON-EMPTY thoughts. An EMPTY thought on either side is its own distinct step
                # — a bare native-tool-use turn (the documented default model) carries no preamble prose,
                # so `_summarize_step` emits thought="" with a non-empty action; the earlier
                # `st.thought and new` guard treated every such step as a non-boundary and silently
                # dropped it, collapsing a whole run's transcript to one entry. `queue_commit`'s content
                # guard drops the genuinely-empty opening frame, so committing here is safe. Feeder-
                # agnostic: real-run markers and the self-heal turn loop each set a fresh thought per step.
                # (Residual, rare: two consecutive empty-thought steps with no thought-bearing step
                # between them read as identical and don't commit — ReAct interleaves an observation.)
                streaming = new == st.thought or (
                    bool(new) and bool(st.thought) and (new.startswith(st.thought) or st.thought.startswith(new))
                )
                if not streaming:
                    st.queue_commit()
                st.thought = new
            # label/mode/status/action are set AFTER the thought-boundary commit, so a commit snapshots
            # the FINISHING step's fields — not the incoming step's. (Assigning label before the commit
            # paired the next step's label with the finished step's thought; only compact mode renders
            # the label, so the default self-heal transcript never surfaced it.)
            if label is not None:
                st.label = redact(str(label))
            if mode is not None:
                st.mode = redact(str(mode))
            if status is not None:
                st.status = redact(str(status))
            if action is not None:
                st.action = redact(str(action))
            if footer is not None:
                st.footer = redact(str(footer))


class _NullThoughtHandle:
    """Inert handle (disabled box / a checklist frame already open). ``update`` is a no-op and
    ``.live`` is ``False`` so feeders keep printing their own lines exactly as they do today."""

    live = False

    def update(self, **_kwargs: str) -> None:
        return None


@contextlib.contextmanager
def think(
    source: InstallProgress | None,
    title: str,
    *,
    footer: str | None = None,
    fallback_stream: TextIO | None = None,
    compact: bool = False,
) -> Iterator[_ThoughtHandle | _NullThoughtHandle]:
    """Yield a live thinking-box handle backed by a fresh :class:`ThoughtBox` on ``source``'s stream.

    ``source`` is the progress object the wizard already threads through (typically ``conda.progress``)
    so muting stays consistent: the box animates only when ``source`` would, and never when a checklist
    ``build`` frame is already open on it. When ``source`` is ``None`` we fall back to ``fallback_stream``
    (e.g. ``io.out``) and auto-detect the TTY there. Either way, a non-TTY / muted stream ⇒ an inert
    handle, no thread, zero bytes — callers wrap unconditionally.
    """
    if source is not None:
        stream = getattr(source, "stream", None) or fallback_stream or sys.stdout
        # Mirror source's muting; force off if a checklist frame is already drawing on this stream.
        enabled: bool | None = bool(getattr(source, "enabled", False)) and getattr(source, "_box", None) is None
    else:
        stream = fallback_stream if fallback_stream is not None else sys.stdout
        enabled = None  # no progress object — auto-detect the fallback stream's TTY
    box = ThoughtBox(stream, enabled=enabled)
    # Serialize terminal writers. ``source`` (typically ``conda.progress``) drives its OWN single-line
    # spinner from ``Conda._exec`` whenever a build / pip install runs. During an in-process self-heal
    # that spinner fires *inside* this frame, so two daemon animators would drive the same stdout at once
    # — the box's ``\x1b[{n}A`` multi-line redraw and the spinner's ``\r`` fight for the cursor, orphaning
    # the frame into hundreds of half-drawn borders with the thought rows clobbered (the exact live
    # repro). While the box owns the terminal it IS the liveness indicator, so mute ``source`` for the
    # frame's duration and restore it on exit. Only when the box actually animates (a live TTY) and
    # ``source`` was itself live — otherwise there is nothing to serialize.
    muted = source is not None and bool(getattr(box, "enabled", False)) and bool(getattr(source, "enabled", False))
    prior_enabled = getattr(source, "enabled", None)
    if muted:
        source.enabled = False
    try:
        with box.think(title, footer=footer, compact=compact) as handle:
            yield handle
    finally:
        if muted:
            source.enabled = prior_enabled


# A shared do-nothing sentinel for call sites that want an always-off progress object
# without importing ``contextlib`` themselves. Subclasses :class:`StepBox` so a ``.build``
# frame is available (and a no-op) wherever a progress object is expected.
class NullProgress(StepBox):
    def __init__(self) -> None:
        super().__init__(stream=None, enabled=False)
