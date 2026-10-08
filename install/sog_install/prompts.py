"""
Terminal I/O for the wizard, in the agent's own voice.

:class:`PromptIO` provides the input primitives (text / secret / yes-no /
single-select / multi-select) and the presentation helpers (emoji banner +
``[ ]``/``[✓]`` roadmap checklist) that mirror the agent's config banner
(``stcoscientist.py:138``) and plan-first checklist (``prompt_builder.py:212``),
so the setup flow feels like the same product.

Three operating modes:

* **interactive** (a TTY): real ``input``/``getpass`` prompts.
* **scripted** (``input_lines`` provided): answers are popped from an iterator —
  drives the gated interactive-guide smoke test without a human.
* **non-interactive** (``non_interactive=True``, no scripted lines): any prompt
  that lacks a default raises instead of hanging — the ``--answers``/CI guard.

When a :class:`~sog_install.session_log.SessionLog` is attached, every
prompt and (masked) answer is recorded, so the transcript is complete and safe.

Stdlib only.
"""

from __future__ import annotations

import getpass
import os
import shutil
import sys
from typing import TYPE_CHECKING

from . import constants
from .session_log import SessionLog, mask_secret, redact, register_secret

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# Display names for the internal skill/category keys, so a biologist reads
# "Spatial clustering", not "spatial_clustering". Anything not listed here falls
# back to a generic humanizer (``svg_detection`` → "SVG detection"). The raw key is
# always still accepted at the prompt (matching is separator-insensitive), so these
# are purely cosmetic.
_CATEGORY_DISPLAY: dict[str, str] = {
    "spatial_clustering": "Spatial clustering",
    "deconvolution": "Deconvolution",
    "svg_detection": "SVG detection",
    "cell_segmentation": "Cell segmentation",
    "spatial_alignment": "Spatial alignment & 3D",
    "spatial_communication": "Cell–cell communication",
    "spatial_analysis": "Spatial analysis",
    "data_conversion": "Data conversion",
}
_ACRONYMS = {"svg": "SVG", "qc": "QC", "3d": "3D", "rna": "RNA", "st": "ST"}


def _humanize_category(name: str) -> str:
    """``spatial_clustering`` → "Spatial clustering"; ``svg_detection`` → "SVG detection"."""
    if name in _CATEGORY_DISPLAY:
        return _CATEGORY_DISPLAY[name]
    words = name.replace("_", " ").replace("-", " ").split()
    if not words:
        return name
    out = [_ACRONYMS.get(w.lower(), w) for w in words]
    if out[0].islower():  # capitalize the first word unless it's a preserved acronym
        out[0] = out[0].capitalize()
    return " ".join(out)


def _norm_category(token: str) -> str:
    """Fold a category token to a separator-insensitive key so ``svg detection``,
    ``svg_detection`` and ``SVG detection`` all match the same group."""
    return "".join(ch for ch in token.lower() if ch.isalnum())


def _term_width(default: int = 100) -> int:
    """Terminal width, clamped to a readable band (never assume a giant screen)."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except (ValueError, OSError):  # pragma: no cover - defensive
        cols = default
    return max(80, min(cols, 120))


def _encodable(stream: object, text: str) -> str:
    """Coerce ``text`` to what ``stream`` can actually encode, so answer/agent content outside the
    terminal's charset — a µ or ° in a real ST answer, an em-dash / … the truncator injects, a Greek
    label — can't raise ``UnicodeEncodeError`` from ``print`` and crash a whole phase on a ``LANG=C``
    box. Width-preserving (each un-encodable code point becomes one ``?``) and a NO-OP on a UTF-8
    stream or a StringIO with no ``.encoding`` (the common cases), so normal output stays byte-for-byte
    identical. Mirrors ``progress.InstallProgress._encodable`` — the same guard at that write chokepoint."""
    enc = getattr(stream, "encoding", None)
    if not enc:
        return text
    try:
        text.encode(enc)
    except (UnicodeError, LookupError):
        text = text.encode(enc, "replace").decode(enc, "replace")
    return text


class PromptError(RuntimeError):
    """Raised when input is required but unavailable (non-interactive, no default)."""


class PromptIO:
    def __init__(
        self,
        *,
        input_lines: Iterator[str] | None = None,
        non_interactive: bool = False,
        log: SessionLog | None = None,
        stream=None,
    ) -> None:
        self.input_lines = input_lines
        self.non_interactive = non_interactive
        self.log = log
        self.out = stream or sys.stdout

    # -- low-level I/O --------------------------------------------------------
    def _emit(self, text: str = "") -> None:
        # Redact on the way out — the single terminal-write chokepoint for every displayed line
        # (banner / section / say / note / warn / err / ok / the roadmap checklist all route here).
        # Belt-and-braces, mirroring ``progress.py``'s ThoughtBox and ``session_log.notice``: even if
        # a caller interpolates a registered secret into a message, it can never reach the console.
        # ``redact`` early-returns on empty text / no registered secrets, so the hot path is free.
        # ``_encodable`` then keeps a non-ASCII answer/label from raising ``UnicodeEncodeError`` here on
        # a ``LANG=C`` terminal (no-op on UTF-8 / StringIO) — same wedge guard as ``progress._write``.
        # ``flush=True`` because stdout is block-buffered (~8 KB) the moment it is NOT a terminal.
        # Measured on a real install: `sog-setup > real.txt` left real.txt at **0 bytes for 25
        # minutes** while four processes worked — banner, preflight report and all — so `tail -f`
        # showed nothing and a killed run would have lost the whole log. Both animated paths
        # (``_render`` below, ``progress._write``) already flush because an animation cannot work
        # otherwise; this is the *durable* narration path, where an unflushed line is the
        # difference between a usable support log and an empty file.
        #
        # Guarded like ``_render``: a detached or closed stream — `sog-setup | head -5`, whose EPIPE
        # only surfaces once we actually flush — must not abort an installer mid-build.
        try:
            print(_encodable(self.out, redact(text)), file=self.out, flush=True)
        except (ValueError, OSError):  # pragma: no cover - closed/detached stream
            pass

    def _log(self, kind: str, **fields) -> None:
        if self.log is not None:
            self.log.event(kind, **fields)

    def _readline(self, promptline: str, *, secret: bool = False, default: str | None = None) -> str:
        # scripted mode: pop the next canned answer
        if self.input_lines is not None:
            try:
                return next(self.input_lines).rstrip("\n")
            except StopIteration as exc:
                raise PromptError("scripted input exhausted") from exc
        # non-interactive with nothing scripted: honor the caller's default (the string
        # a user would type to accept it) instead of raising — so a prompt WITH a default
        # is answerable headlessly. Only a prompt with NO default (default is None) is
        # genuinely unanswerable and still raises.
        if self.non_interactive:
            if default is not None:
                return default
            raise PromptError(f"input required but running non-interactively: {promptline!r}")
        # Interactive read. A closed stdin (Ctrl-D / piped-and-drained) raises EOFError;
        # treat it like an empty answer — accept the caller's default when there is one
        # (e.g. the pre-selected recommendation), else abort cleanly with the typed
        # PromptError the CLI already maps to a friendly exit instead of a raw traceback.
        #
        # Coerce the prompt to the terminal's charset FIRST — the same guard ``_emit``/``_paint_frame``
        # apply to every *displayed* line, but the interactive prompt handed to ``input()``/``getpass``
        # skipped it. CPython encodes the prompt with ``self.out``'s codec (``errors="strict"``) BEFORE
        # reading, so a non-ASCII prompt — the picker's ``➤`` (U+27A4), or a µ/° interpolated into
        # ``{prompt}``/``{default}`` — raises ``UnicodeEncodeError`` on a ``LANG=C`` / cp1252 / piped-to-C
        # stdout, which the EOFError-only guard below does not catch: a raw traceback kills the one
        # screen the user cannot skip. Coerced to ``?`` here it stays byte-identical on a UTF-8 stream.
        promptline = _encodable(self.out, promptline)
        try:
            if secret:
                try:
                    return getpass.getpass(promptline)
                except (EOFError, getpass.GetPassWarning):
                    # No TTY for a hidden prompt — fall back to visible read.
                    return input(promptline)
            return input(promptline)
        except EOFError as exc:
            if default is not None:
                return default
            raise PromptError(f"input stream closed (EOF) and no default for {promptline!r}") from exc

    def _ansi_capable(self) -> bool:
        """True only when an in-place ANSI redraw of the tool picker is safe: a real interactive
        terminal on BOTH stdin and stdout, not a scripted / non-interactive / redirected run, and
        not explicitly disabled. Everything else falls back to the plain reprint — which keeps every
        scripted / headless path (and its tests) byte-for-byte identical, since ``StringIO.isatty()``
        is ``False``. Mirrors the TTY gate ``progress._auto_enabled`` uses for the install box and
        honors the same ``SOG_SETUP_NO_PROGRESS`` kill-switch, plus the ``NO_COLOR`` / ``TERM=dumb``
        conventions so a user who has opted out of fancy output anywhere gets the plain picker too."""
        if self.input_lines is not None or self.non_interactive:
            return False  # scripted or headless → deterministic plain emit, never cursor tricks
        from .progress import progress_muted  # deferred: keeps the prompts -> progress edge lazy

        env = os.environ
        # NO_COLOR follows the no-color.org convention: PRESENT (any value, even empty) disables.
        # The kill-switch goes through progress_muted() rather than a second copy of the test, so
        # "honors the same SOG_SETUP_NO_PROGRESS kill-switch" above is true by construction.
        if progress_muted() or "NO_COLOR" in env or env.get("TERM", "") == "dumb":
            return False
        try:  # getattr on BOTH ends: a minimal/wrapped stream with no isatty degrades, never raises
            out_tty = getattr(self.out, "isatty", lambda: False)()
            in_tty = getattr(sys.stdin, "isatty", lambda: False)()
            return bool(out_tty) and bool(in_tty)
        except (ValueError, OSError):  # pragma: no cover - detached/closed stream
            return False

    def _paint_frame(self, lines: list[str], *, move_up: int, tty: bool) -> int:
        """Render the tool-picker frame, returning how many screen lines it occupies so the caller
        can move the cursor back to its top on the next repaint.

        On a non-ANSI stream (``tty`` is ``False`` — scripted / StringIO / redirected), or when the
        frame is too tall or too narrow to redraw in place without corrupting the screen, it emits
        each line through the normal :meth:`_emit` chokepoint, exactly like the historical reprint,
        and returns 0 (the plain path needs no cursor bookkeeping). On a real terminal that fits it
        moves the cursor up ``move_up`` lines, clears to the end of the screen, and repaints — so the
        menu updates *in place* instead of scrolling a fresh copy on every keystroke. Each line is
        redacted + charset-coerced + width-clipped (the same guards :meth:`_emit` applies, minus the
        newline it owns), and a closed/detached stream is swallowed rather than crashing the prompt
        (mirrors ``progress.InstallProgress._write``)."""
        # In-place ANSI redraw is only safe when the whole frame fits the viewport. The layout is
        # built for a >=80-column band, so a narrower terminal wraps every row; and a frame taller
        # than the screen scrolls its own top into the scrollback, where the cursor-up rewind can no
        # longer reach it — the next repaint would clamp at row 1 and duplicate rows on every
        # keystroke. When either check fails — or on any non-ANSI stream — reprint plainly (the same
        # bytes the historical reprint produced) and return 0, so the caller starts the next paint
        # fresh instead of rewinding into content it can't account for. The +1 is the input line the
        # terminal echoes just below the frame; both dimensions come from the live terminal, not a guess.
        term = shutil.get_terminal_size((100, 24))
        if not tty or term.columns < 80 or len(lines) + 1 >= term.lines:
            for ln in lines:
                self._emit(ln)
            return 0
        width = min(_term_width(), term.columns)  # never let a clipped line exceed the real width -> no wrap
        chunks: list[str] = []
        if move_up > 0:
            chunks.append(f"\x1b[{move_up}A")  # cursor up to the previous frame's first line
        chunks.append("\x1b[J")  # erase everything from there to the end of the screen
        for ln in lines:
            safe = _encodable(self.out, redact(ln))
            if len(safe) > width:
                safe = safe[:width]
            chunks.append(safe + "\x1b[K\n")  # line + clear-to-EOL (defensive) + newline
        try:
            self.out.write("".join(chunks))
            self.out.flush()
        except (ValueError, OSError):  # pragma: no cover - closed/detached stream
            pass
        return len(lines)

    # -- presentation helpers (the agent voice) -------------------------------
    def banner(self, title: str, subtitle: str | None = None) -> None:
        self._emit()
        self._emit(constants.BANNER_RULE)
        self._emit(f"🔧 {title}")
        self._emit(constants.BANNER_RULE)
        if subtitle:
            self._emit(subtitle)
        self._log("banner", title=title, subtitle=subtitle or "")

    def section(self, title: str) -> None:
        self._emit()
        self._emit("-" * len(constants.BANNER_RULE))
        self._emit(title)

    def roadmap(self, steps: Sequence[str] | None = None, done: Sequence[int] = ()) -> None:
        """Print the ``[ ]``/``[✓]`` roadmap checklist."""
        steps = steps or constants.ROADMAP_STEPS
        done_set = set(done)
        for i, step in enumerate(steps):
            mark = constants.CHECK_DONE if i in done_set else constants.CHECK_TODO
            self._emit(f"  {mark} {i}. {step}")
        self._log("roadmap", done=list(done_set))

    def say(self, message: str) -> None:
        self._emit(message)
        self._log("say", message=message)

    def ok(self, message: str) -> None:
        self._emit(f"✅ {message}")
        self._log("ok", message=message)

    def warn(self, message: str) -> None:
        self._emit(f"⚠️  {message}")
        self._log("warn", message=message)

    def err(self, message: str) -> None:
        self._emit(f"❌ {message}")
        self._log("error", message=message)

    def handoff(self, message: str) -> None:
        """The Stage-A → Stage-B handoff line."""
        self._emit(f"  {constants.CHECK_DONE} {message} 👉")
        self._log("handoff", message=message)

    def note(self, message: str) -> None:
        self._emit(f"  {message}")

    # -- input primitives -----------------------------------------------------
    def ask_text(self, prompt: str, default: str | None = None) -> str:
        suffix = f" [{default}]" if default not in (None, "") else ""
        while True:
            raw = self._readline(f"{prompt}{suffix}: ", default=default).strip()
            if raw:
                self._log("answer", prompt=prompt, value=raw)
                return raw
            if default is not None:
                self._log("answer", prompt=prompt, value=default, defaulted=True)
                return default
            if self.non_interactive:
                raise PromptError(f"no value and no default for {prompt!r}")
            self._emit("  (a value is required)")

    def ask_secret(self, prompt: str, *, allow_empty: bool = False) -> str:
        while True:
            raw = self._readline(f"{prompt}: ", secret=True, default=("" if allow_empty else None)).strip()
            if raw:
                register_secret(raw)
                self._log("answer_secret", prompt=prompt, value=mask_secret(raw))
                return raw
            if allow_empty:
                self._log("answer_secret", prompt=prompt, value="(empty)")
                return ""
            if self.non_interactive:
                raise PromptError(f"no secret provided for {prompt!r}")
            self._emit("  (a value is required)")

    def ask_yesno(self, prompt: str, default: bool = True) -> bool:
        hint = "[Y/n]" if default else "[y/N]"
        while True:
            raw = self._readline(f"{prompt} {hint} ", default="").strip().lower()
            if not raw:
                self._log("answer", prompt=prompt, value=default, defaulted=True)
                return default
            if raw in ("y", "yes"):
                self._log("answer", prompt=prompt, value=True)
                return True
            if raw in ("n", "no"):
                self._log("answer", prompt=prompt, value=False)
                return False
            self._emit("  (please answer y or n)")

    def select(
        self,
        prompt: str,
        options: Sequence[tuple[str, str] | tuple[str, str, str]],
        default: str | None = None,
    ) -> str:
        """Single-choice numbered menu. ``options`` is ``(id, label[, hint])``.
        Accepts the number, the id, or a case-insensitive label prefix."""
        self._emit(prompt)
        ids = [o[0] for o in options]
        for i, opt in enumerate(options, 1):
            oid, label = opt[0], opt[1]
            hint = opt[2] if len(opt) > 2 else ""
            star = " (default)" if oid == default else ""
            self._emit(f"  [{i}] {label}{star}")
            if hint:
                self._emit(f"      {hint}")
        while True:
            suffix = f" [{default}]" if default else ""
            raw = self._readline(f"  Choose{suffix}: ", default=default).strip()
            if not raw and default:
                self._log("answer", prompt=prompt, value=default, defaulted=True)
                return default
            chosen = self._match_option(raw, ids, options)
            if chosen is not None:
                self._log("answer", prompt=prompt, value=chosen)
                return chosen
            if self.non_interactive:
                raise PromptError(f"no valid selection for {prompt!r}: {raw!r}")
            self._emit("  (not a valid choice)")

    def multiselect(
        self,
        prompt: str,
        options: Sequence[tuple[str, str] | tuple[str, str, str]],
        *,
        minimum: int = 1,
    ) -> list[str]:
        """Multi-choice menu. Accepts ``1,3``, ``1 3``, ``all``, or (if
        ``minimum==0``) ``none``. Returns the chosen ids in menu order."""
        self._emit(prompt)
        ids = [o[0] for o in options]
        for i, opt in enumerate(options, 1):
            label = opt[1]
            hint = opt[2] if len(opt) > 2 else ""
            self._emit(f"  [{i}] {label}")
            if hint:
                self._emit(f"      {hint}")
        while True:
            raw = self._readline("  Choose (comma-separated, or 'all'): ").strip().lower()
            if raw == "all":
                self._log("answer", prompt=prompt, value=ids)
                return list(ids)
            if raw in ("none", "") and minimum == 0:
                self._log("answer", prompt=prompt, value=[])
                return []
            tokens = [t for t in raw.replace(",", " ").split() if t]
            chosen: list[str] = []
            ok = True
            for tok in tokens:
                m = self._match_option(tok, ids, options)
                if m is None:
                    ok = False
                    break
                if m not in chosen:
                    chosen.append(m)
            if ok and len(chosen) >= minimum:
                ordered = [i for i in ids if i in chosen]
                self._log("answer", prompt=prompt, value=ordered)
                return ordered
            if self.non_interactive:
                raise PromptError(f"invalid multiselect for {prompt!r}: {raw!r}")
            self._emit(f"  (pick at least {minimum}; use the numbers shown)")

    def grouped_multiselect(
        self,
        prompt: str,
        groups: Sequence[tuple[str, str, Sequence[tuple[str, str] | tuple[str, str, str] | tuple[str, str, str, str]]]],
        *,
        minimum: int = 1,
        preselected: Sequence[str] | None = None,
        collapsible: bool = False,
    ) -> list[str]:
        """An **accordion + live basket** multi-select: pick freely across categories.

        The old flat list dumped every tool on one screen and committed the moment a
        valid selection was typed. This instead shows *one* category's tool rows at a
        time (an accordion), keeps every other category one letter away in a compact
        index, and accumulates picks in a persistent, on-screen **basket** you edit
        until you type ``done`` (or press Enter) — so a biologist can browse
        deconvolution, add two tools, jump to SVG, add one more, and only then finish.

        ``groups`` is ``(name, description, items)`` where ``items`` is
        ``(id, label[, hint[, detail]])`` — plain 2-/3-tuples and the richer 4-field
        :class:`~sog_install.categories.ToolOption` are both accepted
        (``item[0]`` is always the id). Each visible row renders on one aligned line —
        ``<n> <★?><✓?> <Human name>   <(hint) description>`` — with the noisy internal
        key dropped. ``★`` marks a recommendation (``preselected``); ``✓`` marks a
        basket member. The recommendation seeds the basket and auto-opens its category.

        Grammar (one line at a time):

        * a **letter** ``a``–``z`` (or a category **name / prefix**) OPENS that
          category — it no longer bulk-selects it;
        * **numbers / ranges** (``2``, ``1-3``, ``2,4``) are LOCAL to the open category
          and TOGGLE those tools in/out of the basket;
        * a bare **id / label-prefix** toggles that specific tool from any category;
        * ``*`` adds every tool in the open category; ``all`` adds the whole catalog;
        * ``clear`` empties the basket; ``done`` (or Enter) finishes.

        Results come back in global menu order, de-duped. The non-interactive contract
        is unchanged: with a recommendation, an empty answer accepts it; without one and
        an unmet ``minimum``, it raises. ``collapsible`` is accepted for call-site
        compatibility and no longer changes behavior (the accordion is always on).
        """
        # Flatten once: global id order, per-category id lists, humanized headers, and
        # an id -> (label, hint, detail) row map used only for rendering.
        ids: list[str] = []
        rows: dict[str, tuple[str, str, str]] = {}
        group_ids: dict[str, list[str]] = {}
        group_order: list[tuple[str, str]] = []  # (raw key, humanized header) in display order
        for name, _desc, items in groups:
            key = name.lower()
            gids: list[str] = []
            for item in items:
                oid = item[0]
                label = item[1] if len(item) > 1 else oid
                hint = item[2] if len(item) > 2 else ""
                detail = item[3] if len(item) > 3 else ""
                ids.append(oid)
                rows[oid] = (label, hint, detail)
                gids.append(oid)
            group_ids[key] = gids
            group_order.append((key, _humanize_category(name)))

        _ = collapsible  # accepted for call-site compatibility; the accordion is always on now
        wanted = set(preselected or [])
        pre = [i for i in ids if i in wanted]  # ordered, de-duped, junk filtered
        pre_set = set(pre)
        human_of = dict(group_order)  # key -> humanized header
        all_options = [(oid, rows[oid][0]) for oid in ids]  # global (id, label) for name/label matching

        # Letters a..z address categories in display order; extra categories (>26) keep
        # their name as the only handle. The basket accumulates picks across categories.
        letters = "abcdefghijklmnopqrstuvwxyz"
        keys_in_order = [key for key, _h in group_order]
        letter_of = {key: letters[i] for i, key in enumerate(keys_in_order) if i < len(letters)}
        letter_map = {letters[i]: key for i, key in enumerate(keys_in_order) if i < len(letters)}
        basket: list[str] = list(pre)  # ordered set; seeded with the recommendation (★ pre-checked)

        def category_of(oid: str) -> str:
            for key, gids in group_ids.items():
                if oid in gids:
                    return key
            return keys_in_order[0] if keys_in_order else ""

        # Auto-open the recommendation's category (else the first), so the ★ tools are on screen.
        open_key = category_of(pre[0]) if pre else (keys_in_order[0] if keys_in_order else "")

        def resolve_category(tok: str) -> str | None:
            """A separator-insensitive category match (raw key or humanized name),
            exact first then an unambiguous prefix; ``None`` if it isn't a category."""
            nt = _norm_category(tok)
            if not nt:
                return None
            exact = {key for key, human in group_order if nt in (_norm_category(key), _norm_category(human))}
            if len(exact) == 1:
                return next(iter(exact))
            pref = {
                key
                for key, human in group_order
                if _norm_category(key).startswith(nt) or _norm_category(human).startswith(nt)
            }
            return next(iter(pref)) if len(pref) == 1 else None

        def is_numeric(tok: str) -> bool:
            """A plain index (``2``) or a range (``1-3``) — the tokens that are LOCAL to the open category."""
            # isascii-gate keeps this routing predicate truthful: an exotic ``isdigit``-but-not-``int``-parseable
            # glyph (``²``) is NOT a usable local index, so it must not claim the numeric-local lane (where the
            # int() sites live). Behaviour-neutral for such glyphs (both lanes end in "not recognized"), but it
            # keeps the predicate consistent with the now-isascii-gated _match_option/_expand_token int() calls.
            if tok.isascii() and tok.isdigit():
                return True
            if "-" in tok:
                a, b = tok.split("-", 1)
                return a.isascii() and a.isdigit() and b.isascii() and b.isdigit()
            return False

        def resolve_open(tok: str) -> str | None:
            """A single letter, or a category name/prefix, that OPENS a category — never a numeric token."""
            if len(tok) == 1 and tok in letter_map:
                return letter_map[tok]
            if any(is_numeric(t) for t in tok.split()):
                return None
            return resolve_category(tok)

        def basket_ordered() -> list[str]:
            bset = set(basket)
            return [i for i in ids if i in bset]

        def _compose_frame() -> list[str]:
            """Build the picker frame — the letter index, the one open category's rows, and the live
            basket — as a list of lines. Emitting them (plain reprint) vs. repainting them in place
            (TTY) is the caller's job: the plain path prints each line, byte-identical to the historical
            behavior every scripted test asserts against; the TTY path redraws over the previous copy."""
            lines: list[str] = []
            width = _term_width()
            bset = set(basket)
            lines.append(prompt)
            lines.append("  ★ recommended   ✓ selected")
            # compact letter index of every category, packed onto width-aware lines
            lines.append("  Categories:")
            cells: list[str] = []
            for key, human in group_order:
                gids = group_ids[key]
                sel = sum(1 for o in gids if o in bset)
                cell = f"{letter_of.get(key, '-')} {human} ({len(gids)})"
                if sel:
                    cell += f" ✓{sel}"
                cells.append(cell)
            line = "   "
            for cell in cells:
                add = ("   " if line.strip() else "") + cell
                if line.strip() and len(line) + len(add) > width:
                    lines.append(line)
                    line = "   " + cell
                else:
                    line += add
            if line.strip():
                lines.append(line)
            # the open category's tool rows, with LOCAL numbers and ★/✓ marks
            gids = group_ids.get(open_key, [])
            lines.append(f"  ▸ {letter_of.get(open_key, '-')}  {human_of.get(open_key, open_key)}")
            label_w = min(max((len(rows[o][0]) for o in gids), default=0), 32)
            for n, oid in enumerate(gids, 1):
                label, hint, detail = rows[oid]
                mark = ("★" if oid in pre_set else " ") + ("✓" if oid in bset else " ")
                lab = label if len(label) <= label_w else label[: label_w - 1].rstrip() + "…"
                desc = detail
                if hint:
                    desc = f"({hint}) {desc}".rstrip() if desc else f"({hint})"
                row = f"      {n:>2} {mark} {lab.ljust(label_w)}"
                if desc:
                    budget = width - len(row) - 2
                    if budget > 1 and len(desc) > budget:
                        clip = desc[: budget - 1]
                        sp = clip.rfind(" ")
                        desc = (clip[:sp] if sp >= budget // 2 else clip).rstrip() + "…"
                    lines.append(f"{row}  {desc}".rstrip())
                else:
                    lines.append(row.rstrip())
            # the live basket — what will be installed if you finish now
            if basket:
                names = ", ".join(rows[o][0] for o in basket_ordered())
                cap = width - 20
                if cap > 1 and len(names) > cap:
                    names = names[: cap - 1].rstrip() + "…"
                lines.append(f"  🧺 Selected ({len(basket)}): {names}")
            else:
                lines.append("  🧺 Selected (0): nothing yet — toggle a tool by its number")
            lines.append("  numbers=toggle · letter=open category · *=all here · all=everything · clear=empty · done⏎")
            return lines

        default = "" if pre else None
        tty = self._ansi_capable()  # in-place redraw only on a real terminal; scripted/headless stays plain
        painted = 0  # TTY: screen lines the previous paint left above the cursor (frame + input echo + notice)
        notice = ""  # TTY: a one-line message folded into the bottom of the NEXT frame (kept out of scrollback)

        def _hold(msg: str) -> None:
            # TTY: fold the message into the next frame so it updates in place instead of spamming
            # the scrollback. Plain path: emit it immediately — byte-identical to the historical
            # reprint that every scripted/non-interactive test asserts against.
            nonlocal notice
            if tty:
                notice = msg
            else:
                self._emit(msg)

        while True:
            frame = _compose_frame()
            if tty and notice:
                frame = [*frame, notice]
            painted = self._paint_frame(frame, move_up=painted, tty=tty)
            notice = ""  # folded above (if any); a fresh message may be set again below
            raw = self._readline("  ➤ ", default=default).strip()
            if tty and painted:
                # Count the echoed "  ➤ <answer>" line only when we actually painted IN PLACE
                # (painted>0). A frame that fell back to a plain reprint — too tall/narrow for the
                # viewport — returns 0; the next paint must then start fresh (move_up=0), not rewind
                # one line up into the plainly-printed frame and clobber its last row.
                painted += 1
            low = raw.lower()

            # ---- finish (done / Enter) ----
            if low in ("done", ""):
                result = basket_ordered()
                if len(result) >= minimum:
                    self._log("answer", prompt=prompt, value=result)
                    return result
                if self.non_interactive:
                    raise PromptError(f"invalid grouped multiselect for {prompt!r}: {raw!r}")
                _hold(f"  (please pick at least {minimum} — toggle a tool by its number, then press Enter or 'done')")
                continue
            # ---- bulk commands ----
            if low == "all":
                for oid in ids:
                    if oid not in basket:
                        basket.append(oid)
                continue
            if low == "clear":
                basket.clear()
                continue
            if low == "*":
                for oid in group_ids.get(open_key, []):
                    if oid not in basket:
                        basket.append(oid)
                continue
            # ---- open a category (letter or name/prefix) ----
            opened = resolve_open(low)
            if opened is not None:
                open_key = opened
                continue
            # ---- otherwise: toggle tokens (numbers/ranges local; ids/labels global) ----
            gids = group_ids.get(open_key, [])
            local_options = [(oid, rows[oid][0]) for oid in gids]
            targets: list[str] = []
            ok = True
            for tok in low.replace(",", " ").split():
                if is_numeric(tok):  # local index/range within the open category
                    sel = self._expand_token(tok, gids, local_options, {})
                else:  # a bare id / label-prefix, resolved across the whole catalog
                    one = self._match_option(tok, ids, all_options)
                    sel = [one] if one is not None else None
                if sel is None:
                    ok = False
                    break
                for s in sel:
                    if s not in targets:
                        targets.append(s)
            if ok and targets:
                for oid in targets:
                    if oid in basket:
                        basket.remove(oid)
                    else:
                        basket.append(oid)
                continue
            if self.non_interactive:
                raise PromptError(f"invalid grouped multiselect for {prompt!r}: {raw!r}")
            _hold(f'  (I didn\'t recognize "{raw}" — type a number to toggle, a letter to open a category, or "done")')

    # -- matching -------------------------------------------------------------
    @staticmethod
    def _match_option(raw: str, ids: Sequence[str], options: Sequence[tuple]) -> str | None:
        raw = raw.strip()
        if not raw:
            return None
        # ``str.isdigit()`` is True for exotic code points ``int()`` REFUSES — superscripts/subscripts
        # (``²``/``₃``: isdigit yet isdecimal False → ``int('²')`` raises ValueError). On the one screen the
        # user cannot skip (the tool picker), a pasted such glyph would crash the prompt. Gate on ``isascii``
        # first so an exotic numeral simply falls through to the label/id match ("not a valid choice"), never
        # ``int()`` — mirroring this file's ``_encodable`` "no exotic input can crash a prompt" invariant.
        if raw.isascii() and raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(ids):
                return ids[idx]
            return None
        low = raw.lower()
        if low in [i.lower() for i in ids]:
            return next(i for i in ids if i.lower() == low)
        # label prefix match (unambiguous only)
        matches = [o[0] for o in options if o[1].lower().startswith(low)]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _expand_token(
        tok: str,
        ids: Sequence[str],
        options: Sequence[tuple],
        group_ids: dict[str, list[str]],
    ) -> list[str] | None:
        """Resolve one grouped-multiselect token to a list of ids, or ``None``.

        Order: numeric range (``a-b``) → an exact category name (whole group) → a
        single option (number / exact id / unambiguous label-prefix via
        :meth:`_match_option`) → an unambiguous category-name prefix (whole group).
        """
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            # isascii-gate before int() — an exotic-numeral endpoint (``²-³``) is isdigit-True but int()-raises;
            # let it fall through to the non-range paths instead of crashing the picker (see _match_option).
            if lo_s.isascii() and lo_s.isdigit() and hi_s.isascii() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    lo, hi = hi, lo
                # Clamp the ENDPOINTS to the valid 1..len(ids) window *before* building the range —
                # a fat-fingered ``1-999999999`` must not iterate a billion ints (a real CPU hang) just
                # to filter them all out. With clamped bounds the per-element ``1 <= i <= len(ids)`` guard
                # is redundant, so the range yields only in-bounds indices directly. (Round-6 B)
                picked = [ids[i - 1] for i in range(max(1, lo), min(hi, len(ids)) + 1)]
                return picked or None
        if tok in group_ids:  # exact category name -> the whole group
            return list(group_ids[tok])
        one = PromptIO._match_option(tok, ids, options)
        if one is not None:
            return [one]
        cat_pref = [g for g in group_ids if g.startswith(tok)]
        if len(cat_pref) == 1:  # unambiguous category-name prefix -> whole group
            return list(group_ids[cat_pref[0]])
        return None
