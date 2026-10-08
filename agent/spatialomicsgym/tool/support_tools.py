import base64
import io
import logging
import re
import sys
import sysconfig
import threading
import traceback
from io import StringIO

from spatialomicsgym.action import WARNINGS_HEADER
from spatialomicsgym.utils.anndata_compat import ensure_backed_sparse_indexing
from spatialomicsgym.utils.execution import OBSERVATION_CHARS, publish_partial_output
from spatialomicsgym.utils.pandas_display import apply_readable_display


class _BoundedStringIO(StringIO):
    """A StringIO that stops accepting characters past a cap, so runaway LLM-written output
    (``print("x" * 10**9)`` / ``while True: print(...)``) is dropped AT THE SOURCE instead of
    materializing gigabytes in memory before the downstream ~10 KB observation cap truncates it.

    It counts what it drops, and that counting is the whole point of :attr:`dropped`.
    ``clip_observation`` downstream exists because "long tool output puts its decision-relevant
    lines at the END ... the tail is the part a model deciding 'did this succeed' needs" -- so it
    shows a head and a tail and says how many characters it elided between them. Past this cap
    there is no tail left to show: the end of the output was never captured at all. The header
    still promised one, and its elision count understated the real loss by the entire overflow.

    Measured: a cell printing 3,000,000 characters ending in a distinctive last line produced an
    observation that did not contain that line, claimed 1,990,000 characters were elided when
    1,000,037 more had already been thrown away upstream, and told the model "The code already ran
    to completion; this is only a display limit. Do not re-run the analysis to see more." A model
    asked "did it succeed" read a tail that was not the tail and was told not to look further.
    """

    _MAX_CHARS = 2_000_000  # ~2 MB, far above any legitimate observation (display-capped to ~10 KB)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #: Characters this buffer refused, i.e. how much of the END of the output does not exist.
        self.dropped = 0

    def write(self, s):
        remaining = self._MAX_CHARS - self.tell()
        if remaining <= 0:
            self.dropped += len(s)
            return len(s)  # already at the cap; drop
        if len(s) > remaining:
            super().write(s[:remaining])  # a single huge write is truncated AT the cap, then stops
            self.dropped += len(s) - remaining
            return len(s)  # report the full length so the caller doesn't error/retry
        return super().write(s)


#: Appended to a cell's output when the 2 MB source cap actually threw characters away.
#:
#: Phrased for the reader that acts on it. "Elided" is the word ``clip_observation`` uses for text
#: it still holds and chose not to show; this text was never held, so the sentence says so and says
#: what to do instead -- because the observation's own header tells the model not to re-run.
#: The substring ``clip_observation`` looks for to know the tail it is showing is not the tail.
#: Matched rather than re-derived, because the notice's own wording is what a reader sees and
#: those two must not be able to drift apart.
OUTPUT_CAP_MARKER = "characters were dropped at the"

OUTPUT_CAP_NOTICE = (
    "\n... [{dropped:,} characters were dropped at the {cap:,}-character output cap. "
    "The END of this output was never captured -- what follows the cut does not exist anywhere, "
    "and re-reading the observation cannot recover it. Print less (summarise, slice, or write the "
    "full result to a file and print its path) and run again if you need the end.] ..."
)


def _cap_notice(buffer: _BoundedStringIO) -> str:
    """The notice for this buffer, or ``""`` when nothing was dropped."""
    dropped = getattr(buffer, "dropped", 0)
    if not dropped:
        return ""
    return OUTPUT_CAP_NOTICE.format(dropped=dropped, cap=_BoundedStringIO._MAX_CHARS)


#: How much of a timed-out cell's output its observation keeps: the END, because the last lines
#: are where the cell had got to. At most this much, and less when the room is tight: the whole
#: observation is held to ``clip_observation``'s limit (``utils.execution.partial_room``), because
#: past it the clip's header says "the code already ran to completion", which for a timeout is
#: false. The notices, the marker and the cap notice are bounded, and the tail gives way to them; the
#: warnings give way to the tail, down to their one line (``_WarningLog.one_line``).
PARTIAL_TAIL_LINES = 30
PARTIAL_TAIL_CHARS = 4000

#: The platform's own one-line notices printed into a cell while it runs (the budget notice, the
#: env notice, the duplicate-call guard). A timed-out cell's observation keeps them even when they
#: fall in the part the tail drops: the budget notice is printed BEFORE the call that runs out of
#: budget, which is exactly the call whose output pushes it out of the tail. The worker's boundary
#: and figures notices are appended after the cell returns, so they are never in what it printed.
_NOTICE_LINE_RE = re.compile(r"^\[(?:[a-z-]+ notice|duplicate-call guard)\] .*$", re.MULTILINE)
_MAX_KEPT_NOTICES = 3
_MAX_NOTICE_CHARS = 800


def _output_tail(buffer: _BoundedStringIO, room: int | None = None) -> str:
    """What a cell has printed so far, bounded to its tail, saying how much it left out -- in at
    most ``room`` characters when one is given, the tail giving up lines to make it fit.

    Runs every half second in the worker while a cell prints (``repl_host._PartialReporter``), so
    it only splits the last few kilobytes into lines and finds the notices with one regex scan per
    pass. A pass that does not fit shrinks the tail by the excess, and the rest only grows when a
    line leaves the tail (a notice it held is then kept above the marker; a count gains a digit),
    so a second pass almost always fits. The cap on passes is a guard: ``timeout_observation``
    holds the limit whatever this returns.
    """
    text = buffer.getvalue()
    cap = _cap_notice(buffer)
    limit = PARTIAL_TAIL_CHARS
    for _ in range(8):
        shown, tail = _shown_with_tail(text, cap, limit)
        if room is None or len(shown) <= room or not tail:
            break
        limit = len(tail) - (len(shown) - room)
    return shown


def _shown_with_tail(text: str, cap: str, limit: int) -> tuple[str, str]:
    """``text`` as :func:`_output_tail` shows it with a tail of at most ``limit`` characters, and that tail."""
    tail = "".join(text[max(0, len(text) - limit) :].splitlines(keepends=True)[-PARTIAL_TAIL_LINES:])
    dropped = len(text) - len(tail)
    if dropped <= 0:
        return text, tail  # all of it; and nothing was dropped at the source, since it is this short
    head = text[:dropped]
    # The LAST three distinct notices, in the order they were printed: the budget notice for the call
    # that ran out of budget is printed just before it, so it is the latest one, and keeping the
    # first three dropped exactly that one. A repeat moves its line to the end rather than taking a
    # second slot. Held to three as it goes, so a head full of notices cannot grow this.
    kept: dict[str, None] = {}
    for match in _NOTICE_LINE_RE.finditer(head):
        line = match.group(0)
        line = line if len(line) <= _MAX_NOTICE_CHARS else line[: _MAX_NOTICE_CHARS - 3] + "..."
        kept.pop(line, None)
        kept[line] = None
        if len(kept) > _MAX_KEPT_NOTICES:
            del kept[next(iter(kept))]
    notices = list(kept)
    apart = ", apart from the notice line(s) above" if notices else ""
    lines = head.count("\n")
    marker = (
        f"[... {dropped:,} characters ({lines:,} lines) of this cell's earlier output are not "
        f"shown{apart}; its last {len(tail):,} characters follow ...]"
    )
    return "\n".join([*notices, marker, tail]) + cap, tail


def _partial_output(buffer: _BoundedStringIO, warned: "_WarningLog", room: int | None = None) -> str:
    """A running cell's output so far and its warnings so far: what ``run_with_timeout`` reads at
    the deadline, and what the worker reports while the cell runs. The output takes ``room`` first,
    less the one line the warnings need at the least; the warnings fit what it leaves."""
    if room is None:
        return _then(_output_tail(buffer), warned.block())
    floor = len(warned.one_line()) + 1 if warned.seen else 0  # the one line, on a line of its own
    return _with_warnings(_output_tail(buffer, room - floor), warned, room)


def _with_warnings(text: str, warned: "_WarningLog", limit: int = OBSERVATION_CHARS) -> str:
    """``text``, then its warnings on a line of their own, the two held to ``limit`` by shortening the
    warnings and never ``text``: what the model needs to judge the step is what the cell printed.

    ``limit`` is what ``clip_observation`` lets through whole. The block is appended before that
    clip, so a block that did not fit pushed the output's own lines into the elided middle -- in
    E-06 behind 2,000 characters of near-duplicate warnings. Now it lists fewer kinds, then says
    only how many there were (:meth:`_WarningLog.one_line`), then goes: a text that fits whole is
    never clipped for its warnings. A text already past ``limit`` is clipped whatever follows it,
    and costs its tail only that one line.
    """
    if len(text) > limit:
        return _then(text, warned.one_line())
    return _then(text, warned.block(limit - len(text) - (1 if text and not text.endswith("\n") else 0)))


class _WarningLog:
    """The Python warnings shown, and the log messages logged, while one cell ran, for its observation.

    The REPL captures ``sys.stdout`` only. A warning goes to stderr, so it never reached the
    observation -- in E-03 (curio_ovary_batch_driven_clustering r3) that hid the one line
    locating a NaN source. This wraps ``warnings.showwarning`` for the life of the cell: every
    warning is recorded AND handed to whatever was there before, so what the real stderr shows,
    and which warnings the filters let through, do not change. It records what Python shows,
    no more: a warning the filters suppress (the default shows each location once per process)
    is not listed, because it was not shown either. The rest of stderr -- progress bars, text a
    cell writes to ``sys.stderr``, C libraries -- is not touched, and the header says so.

    **And what libraries log.** scanpy says "It seems you use rank_genes_groups on the raw count
    data" through ``logging``, and harmonypy says whether its fit "Converged after N iterations" or
    "Stopped before convergence" -- and neither ever reached the model, while the header read as if
    the block were all of stderr. So ``logging.Logger.callHandlers`` is wrapped too, the same way:
    every record is recorded and then handled exactly as before, so what the handlers -- and
    ``logging.lastResort`` -- write does not change, and a logger that does not propagate (scanpy
    logs through a root logger of its own) is heard all the same. Listed: records at WARNING and
    above, and the few INFO records :data:`_INFO_WORTH_LISTING` names -- and of those only the ones
    a handler writes (:func:`_written`): a record that reaches only a ``NullHandler``, or handlers
    set above its level, is shown nowhere, so it is not listed either, as a warning the filters
    suppress is not. Only from the thread the cell runs on, unlike warnings: ``logging`` is also
    what the platform's own threads talk on.

    **Short, and one line per warning.** A warning is one kind per (category, message), however
    many places raised it: in E-06 (curio_ovary_cumulus_gc_count_immature r1) pandas'
    fragmentation warning, raised from five lines of ``_rank_genes_groups.py``, was listed five
    times at 240 characters each and filled 2,138 characters of the observation. Now it is one
    line naming its places and its total, the message is cut at 160 characters, and the whole
    block is held to :attr:`MAX_BLOCK_CHARS` -- the kinds that do not fit are counted on its last
    line -- and to less when the output leaves less room (:func:`_with_warnings`).

    Process-wide, like ``sys.stdout``: a warning raised on any thread while the cell runs is the
    cell's. Removed only if it is still the hook in place -- the same guard, for the same reason,
    as the stdout restore in :meth:`PythonREPL.run` -- and otherwise left in the chain as a pass-
    through, so an orphan finishing late cannot unhook the cell that is running now.

    **Located at the cell.** Python reports a warning where the library's ``stacklevel`` points,
    and that is often not the model's code: in E-06 ``sc.pp.scale``'s densifying warning landed in
    the stdlib's ``functools`` dispatch wrapper and was printed as
    ``/opt/conda/envs/.../lib/python3.11/functools.py:909`` -- a line nobody wrote, in a path that
    told the model nothing. So the stack that is showing the warning is walked outward to the
    innermost frame running THIS cell's own code -- its code objects, by identity (:meth:`watch`),
    not the ``<string>`` filename, which a function an earlier cell defined and code a library
    ``exec``-ed carry too -- and that line is given first, the library's location second.
    """

    MAX_LISTED = 10
    MAX_BLOCK_CHARS = 800
    _MAX_MESSAGE_CHARS = 160
    #: Places kept per kind, and library line numbers kept per place; the rest are counted.
    _MAX_PLACES = 3
    _MAX_LINENOS = 6

    def __init__(self) -> None:
        #: (category, message) -> where and how often it was shown, in first-shown order; the first
        #: MAX_LISTED kinds only.
        self.seen: dict[tuple[str, str], _Kind] = {}
        #: Warnings shown that are none of those -- counted, not kept, so a loop cannot grow this.
        self.more = 0
        self._active = False
        self._hook = None
        self._previous = None
        self._log_hook = None
        self._log_previous = None
        self._thread = None
        #: Set on a thread while this log hands a warning on: logging.captureWarnings would log it as
        #: a "py.warnings" record, and it is already listed.
        self._relaying = threading.local()
        #: id -> code object, for this cell's code and every function, class and comprehension in it.
        self._cell_codes: dict[int, object] = {}

    def watch(self, code) -> None:
        """Know this cell's code: ``code`` as compiled, and every code object nested in it."""
        found, stack = {}, [code]
        while stack:
            current = stack.pop()
            if id(current) in found:
                continue
            found[id(current)] = current
            stack.extend(c for c in getattr(current, "co_consts", ()) if hasattr(c, "co_code"))
        self._cell_codes = found

    def _cell_lines(self, filename, lineno) -> tuple[list[int], bool]:
        """The lines of this cell on the calling thread's stack, innermost first -- empty on a
        thread the cell's code is not on, or before :meth:`watch` -- and whether the frame Python
        reported, the innermost at ``filename:lineno``, is one of this cell's own. By the frame and
        not by the line number: a function an earlier cell defined is ``<string>`` too, and its line
        can be a line number this cell's stack also has."""
        codes, lines, reported = self._cell_codes, [], None
        frame = sys._getframe(1) if codes else None
        while frame is not None:
            mine = codes.get(id(frame.f_code)) is frame.f_code
            if mine:
                lines.append(frame.f_lineno)
            if reported is None and frame.f_lineno == lineno and frame.f_code.co_filename == filename:
                reported = mine
            frame = frame.f_back
        return lines, bool(reported)

    def install(self) -> None:
        import warnings

        previous = warnings.showwarning
        log = self

        def showwarning(message, category, filename, lineno, file=None, line=None):
            if not log._active:
                return previous(message, category, filename, lineno, file, line)
            log._record(message, category, filename, lineno)
            log._relaying.on = True
            try:
                return previous(message, category, filename, lineno, file, line)
            finally:
                log._relaying.on = False

        previous_call = logging.Logger.callHandlers

        def callHandlers(logger, record):
            if log._active:
                log._record_log(logger, record)
            return previous_call(logger, record)

        self._hook, self._previous = showwarning, previous
        self._log_hook, self._log_previous = callHandlers, previous_call
        self._thread, self._active = threading.get_ident(), True
        warnings.showwarning = showwarning
        logging.Logger.callHandlers = callHandlers

    def uninstall(self) -> None:
        import warnings

        self._active = False
        if self._hook is not None and warnings.showwarning is self._hook:
            warnings.showwarning = self._previous
        if self._log_hook is not None and logging.Logger.callHandlers is self._log_hook:
            logging.Logger.callHandlers = self._log_previous

    def _record(self, message, category, filename, lineno) -> None:
        try:
            name = _NOT_A_KIND.sub("_", getattr(category, "__name__", None) or type(message).__name__)
            self._add(name, str(message), filename, lineno)
        except Exception:
            pass

    def _record_log(self, logger, record) -> None:
        try:
            if threading.get_ident() != self._thread or record.levelno < logging.INFO:
                return
            if record.name == "py.warnings" and getattr(self._relaying, "on", False):
                return  # the warning just listed, handed on to logging.captureWarnings
            name = record.name
            if isinstance(logger, logging.RootLogger) and logger is not logging.root:
                name = type(logger).__module__.split(".")[0]  # scanpy's own root logger is named "root" too
            message = record.getMessage()
            if record.levelno < logging.WARNING:
                allowed = _INFO_WORTH_LISTING.get(name.split(".")[0])
                if allowed is None or not allowed.match(message):
                    return
            if not _written(logger, record):
                return
            kind = f"{_NOT_A_KIND.sub('_', record.levelname)}:{_NOT_A_LOGGER.sub('_', name) or '_'}"
            self._add(kind, message, *_logged_from(logger, record))
        except Exception:
            pass

    def _add(self, name: str, message: str, filename, lineno) -> None:
        """One showing of the warning or log message ``name: message``, raised at ``filename:lineno``."""
        text = " ".join(message.split())
        if len(text) > self._MAX_MESSAGE_CHARS:
            text = text[: self._MAX_MESSAGE_CHARS - 3] + "..."
        cell, at_the_cell = self._cell_lines(filename, lineno)
        if at_the_cell or (filename == "<string>" and not self._cell_codes):
            place, at = (lineno, None), None  # the library pointed at the cell itself
        else:
            place, at = (cell[0] if cell else None, _short_path(filename).translate(_PLAIN_PATH)), lineno
        key = (name, text)
        kind = self.seen.get(key)
        if kind is None:
            if len(self.seen) >= self.MAX_LISTED:
                self.more += 1
                return
            kind = self.seen[key] = _Kind()
        kind.count += 1
        linenos = kind.places.get(place)
        if linenos is None:
            if len(kind.places) >= self._MAX_PLACES:
                kind.elsewhere += 1
                return
            linenos = kind.places[place] = []
        if at is not None and at not in linenos:
            if len(linenos) < self._MAX_LINENOS:
                linenos.append(at)
            elif linenos[-1] is not None:
                linenos.append(None)  # and more lines than are kept

    # What reads the log -- one_line, _line, block -- reads a snapshot of each dict it walks: it also
    # runs on the partial reader's thread (the in-process deadline read, the worker's reporter every
    # half second) while the cell's thread may still add a kind or a place, and a dict that grows
    # under its iterator raises -- which read_partial_output turns into an empty partial output.

    @staticmethod
    def _line(name: str, text: str, kind: "_Kind") -> str:
        places = []
        for (cell, path), linenos in list(kind.places.items()):
            where = f"cell line {cell}" if cell is not None else ""
            if path is not None:
                lib = f"{path}:" + ",".join("..." if n is None else str(n) for n in linenos)
                where = f"{where}, in {lib}" if where else lib
            places.append(where)
        if kind.elsewhere:
            places.append(f"and {kind.elsewhere} elsewhere")
        return f"  {name}: {text} ({'; '.join(places)})" + (f" x{kind.count}" if kind.count > 1 else "")

    def one_line(self) -> str:
        """The block at its shortest: how many warnings there were, on the header line itself."""
        if not self.seen:
            return ""
        total = self.more + sum(kind.count for kind in list(self.seen.values()))
        return (
            f"{WARNINGS_HEADER} {total} warning(s) or log message(s), not listed: "
            "the output above fills the observation.\n"
        )

    def block(self, room: int | None = None) -> str:
        """The lines for the observation, in at most ``room`` characters and never more than
        :attr:`MAX_BLOCK_CHARS`: as many kinds as fit, the rest counted on the last line; then
        :meth:`one_line`; then ``""``. Also ``""`` when no warning was shown."""
        if not self.seen:
            return ""
        limit = self.MAX_BLOCK_CHARS if room is None else min(room, self.MAX_BLOCK_CHARS)
        # Short, because it is on every block and the block's 800 characters are shared with it: at
        # 143 characters it pushed the E-06 curio_ovary_cumulus r1 warnings to 815, and the 800 cap
        # then dropped the kind that case is about (the fragmentation warning, x370).
        header = f"{WARNINGS_HEADER} warnings and log messages (not in the output above; other stderr not shown):"
        kinds = list(self.seen.items())
        lines = [self._line(name, text, kind) for (name, text), kind in kinds]
        for listed in range(len(lines), 0, -1):
            rest = self.more + sum(kind.count for _, kind in kinds[listed:])
            more = [f"  ... and {rest} more warning(s) or log message(s) of other kinds, not listed."] if rest else []
            block = "\n".join([header, *lines[:listed], *more]) + "\n"
            if len(block) <= limit:
                return block
        one = self.one_line()
        return one if len(one) <= limit else ""


class _Kind:
    """One warning -- a category and a message -- and where and how often it was shown.

    ``places`` maps (cell line or None, library path or None) to the library line numbers seen
    there, a trailing None standing for more than were kept; ``elsewhere`` counts showings at
    places past the first few. Bounded, so a warning raised in a loop cannot grow it."""

    __slots__ = ("count", "places", "elsewhere")

    def __init__(self) -> None:
        self.count = 0
        self.places: dict[tuple[int | None, str | None], list[int | None]] = {}
        self.elsewhere = 0


#: What a category name may not contain on a warning line, and what a path shows in place of the
#: characters that would end the line's ``(places)``: the line must stay in the exact form
#: ``action._WARNING_LINE`` takes out of an error's signature.
_NOT_A_KIND = re.compile(r"[^\w.]|^(?=[^A-Za-z_])")
_NOT_A_LOGGER = re.compile(r"[^\w.-]")
_PLAIN_PATH = str.maketrans({"(": "[", ")": "]", "\n": " ", "\r": " "})

#: The INFO records worth a line, by the package that logs them: the ones that say whether a fit
#: converged, which is otherwise said nowhere. harmonypy logs "Converged after N iterations" or
#: "Stopped before convergence" at INFO; its progress lines around them ("Iteration 3 of 10",
#: the k-means start) stay out.
_INFO_WORTH_LISTING = {"harmonypy": re.compile(r"Converged after \d+ iterations?\b|Stopped before convergence")}


def _written(logger, record) -> bool:
    """Whether a handler writes ``record``: ``Logger.callHandlers``' own walk, read and not acted on.

    Up the logger's parents while each propagates, a handler at or below the record's level writes
    it -- except a ``NullHandler``, which a library adds so that nothing is written unless the
    application says so. ``logging.lastResort`` writes a record only when the walk found no handler
    at all, a ``NullHandler`` included. Handlers' filters are not asked: asking one is not free of
    effects, and it is asked again when the record is handled."""
    found, current = False, logger
    while current:
        for handler in current.handlers:
            found = True
            if record.levelno >= handler.level and not isinstance(handler, logging.NullHandler):
                return True
        current = current.parent if current.propagate else None
    last = logging.lastResort
    return not found and last is not None and record.levelno >= last.level


def _logged_from(logger, record) -> tuple[str, int]:
    """Where ``record`` was logged from: the innermost frame outside ``logging``, this file and the
    files that define the logger's class. Python's own ``record.pathname`` stops at the first frame
    outside ``logging``, which for every scanpy record is scanpy's ``_RootLogger`` in
    ``scanpy/logging.py`` -- not the scanpy function that had something to say."""
    skip = {logging.Logger.handle.__code__.co_filename, _logged_from.__code__.co_filename}
    for klass in type(logger).__mro__:
        skip.update(f.__code__.co_filename for f in vars(klass).values() if hasattr(f, "__code__"))
    skip.discard("<string>")  # a logger class the cell defined: its frames are the cell's
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_filename not in skip:
            return frame.f_code.co_filename, frame.f_lineno
        frame = frame.f_back
    return record.pathname, record.lineno


#: Where the running interpreter's standard library lives, longest first: a path under one is shown
#: from there down, as one under ``site-packages/`` is.
_STDLIB_PREFIXES = sorted(
    {p.rstrip("/") + "/" for p in (sysconfig.get_paths().get(k) for k in ("stdlib", "platstdlib")) if p},
    key=len,
    reverse=True,
)


def _short_path(filename) -> str:
    """``filename`` from its package down: ``scanpy/tools/_rank_genes_groups.py``, ``functools.py``."""
    path = str(filename)
    if "site-packages/" in path:
        return path.split("site-packages/")[-1]
    for prefix in _STDLIB_PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _then(text: str, block: str) -> str:
    """``block`` after ``text``, on a line of its own."""
    if not block:
        return text
    if text and not text.endswith("\n"):
        text += "\n"
    return text + block


def _error_label(exc: BaseException) -> str:
    """``KeyError: 'leiden'``, not ``'leiden'``: without the class the message is often unreadable."""
    message = str(exc)
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _where_it_raised(exc: BaseException, command: str) -> str:
    """Which line of the cell raised, and the innermost frame outside the cell -- or ``""``.

    The REPL used to report ``Error: {str(e)}`` and nothing else: no class, no line. In a 60-line
    cell the model could not tell which statement failed, and spent a step finding out. Measured in
    the defect hunt over the E-01/E-02 trials: a diagnostic step per error, and in one trial two
    full-dataset re-runs. A Python started from a bash cell gets a full traceback, so the main path
    was the one with worse diagnostics. Two lines, never a traceback dump, and never raises.
    """
    try:
        lines = command.splitlines()
        cell_lineno, inner = None, None
        if isinstance(exc, SyntaxError):
            if exc.filename in (None, "<string>"):
                cell_lineno = exc.lineno
        else:
            frames = traceback.extract_tb(exc.__traceback__)
            in_cell = [f for f in frames if f.filename == "<string>"]
            if in_cell:
                cell_lineno = in_cell[-1].lineno
            if frames and frames[-1].filename != "<string>" and in_cell:
                inner = frames[-1]
        out = []
        if cell_lineno and 1 <= cell_lineno <= len(lines):
            out.append(f"  at cell line {cell_lineno}: {lines[cell_lineno - 1].strip()[:200]}")
        if inner is not None:
            where = inner.filename.split("site-packages/")[-1]
            out.append(f"  raised in {where}:{inner.lineno} ({inner.name})")
        return ("\n" + "\n".join(out)) if out else ""
    except Exception:
        return ""


def _non_ascii_syntax_hint(exc: BaseException, command: str) -> str:
    """An actionable addendum for a SyntaxError caused by non-ASCII punctuation — else ``""``.

    ``prompt_builder.py`` forbids non-ASCII inside ``<execute>``; models write prose there anyway,
    and CPython's own message (``invalid character '—' (U+2014) (<string>, line 3)``) is a dead end
    for an LLM: it names one character out of however many, cites a line number in a string the
    model never sees, and suggests no remedy. Observed live — the identical observation came back
    twice in one run because the model had nothing to act on and re-emitted the same prose.

    So: quote the offending source line, list EVERY offending character (fixing them one per retry
    is the loop we are breaking), and say what to do. Scoped tightly to this error class — an
    ordinary typo or runtime error must not be sent chasing a Unicode fix."""
    if not isinstance(exc, SyntaxError):
        return ""
    message = str(getattr(exc, "msg", "") or "")
    if "invalid character" not in message and "invalid non-printable character" not in message:
        return ""
    lines = [f"  offending line: {exc.text.strip()}"] if getattr(exc, "text", None) else []
    seen: dict[str, None] = {}
    for ch in command:
        if ord(ch) > 127:
            seen.setdefault(ch, None)
    if seen:
        listed = ", ".join(f"{ch!r} (U+{ord(ch):04X})" for ch in list(seen)[:12])
        more = "" if len(seen) <= 12 else f", and {len(seen) - 12} more"
        lines.append(f"  non-ASCII characters in this block: {listed}{more}")
    lines.append(
        "  Hint: non-ASCII punctuation is not valid Python outside a string literal. Replace ALL of "
        "them with plain ASCII (- for dashes, \" ' for quotes, ... for ellipsis) and re-send. If "
        "this block is prose rather than code, it belongs outside <execute>."
    )
    return "\n" + "\n".join(lines)


#: The capture buffer of the cell that most recently took over ``sys.stdout``; ``None`` once it has
#: given it back. A one-slot list so the cells' closures share it. See ``PythonREPL.run``.
_live_capture: list = [None]
#: The thread that installed ``_live_capture[0]``, so ``run_with_timeout`` can disown exactly the cell it
#: gave up on (see :func:`abandon_capture_of`).
_live_owner: list = [None]


def abandon_capture_of(thread_ident) -> None:
    """``run_with_timeout`` stopped waiting for the cell running on ``thread_ident``.

    That cell is an orphan from here on: whatever it finds in ``sys.stdout`` when it finally ends belongs
    to someone else, so it must restore nothing beyond its own buffer. Clearing the token is what tells it
    so. Without this, an orphan under a non-StringIO capture (pytest's, a terminal harness's) "restored"
    the capture of a turn or test that had already finished as the process's stdout, and a later test's
    output went into a dead buffer (found by the 2026-10-01 gate; the residual the rp-u23 review named).
    """
    if thread_ident is not None and _live_owner[0] == thread_ident:
        _live_capture[0] = None
        _live_owner[0] = None


def _cell_displaced_stdout(beneath) -> bool:
    """Whether the stream now holding ``sys.stdout`` is one the ending cell put there itself.

    Asked only when no later cell has started a capture, and only because ``sys.stdout`` is no
    longer the cell's buffer. Two things can have moved it: the cell's own code, whose stream is
    owed a restore, or -- when the cell is an orphan that ``run_with_timeout`` stopped waiting
    for -- whoever wrapped the finished turn putting its own stream back. Nothing here can see
    which thread made the assignment, so each answer below names the second case's shape.

    * ``beneath`` (what this cell displaced) is a StringIO: a harness captures the turn --
      ``redirect_stdout`` around ``go()`` in the benchmark runners and ``chat_cli --json``. It
      restores its own stream when the turn ends, so a displacement is undone without us; and an
      orphan that outlived the turn "restored" that finished buffer as the process's stdout for
      good (hunt 2026-09-30, rp-u23 review of u23-transcriptomics-skills-17).
    * the stream in place is a StringIO: a capture someone installed over the orphan, which is
      being read (``test_a_timed_out_step_does_not_erase_the_next_ones_output``). A cell that
      rebinds stdout to a StringIO of its own and never puts it back looks the same and is
      therefore left alone.
    * the stream in place is the interpreter's own stdout: what the setup probe puts back after
      diverting stdout to stderr, what ``mcp_integration`` installs around an MCP launch, and what
      a harness over the real terminal restores. A cell that installs it itself leaves the
      stream a turn without a harness writes to anyway.
    """
    current = sys.stdout
    return not isinstance(beneath, StringIO) and not isinstance(current, StringIO) and current is not sys.__stdout__


_FENCED_BLOCK_RE = re.compile(r"\A\s*```([\w+.#-]*)[ \t]*\n(.*?)\n?[ \t]*```\s*\Z", re.S)

#: Fence tags naming a language this REPL does not run, and the marker that runs it instead. The
#: same tags ``action._FENCE_BASH`` / ``_FENCE_R`` run as bash and R: "console" was missing, so a
#: ```console block that reached the REPL still ran as Python (hunt 2026-09-30, rp-u23 review of
#: u23-transcriptomics-skills-15).
_OTHER_LANGUAGE_FENCES = {
    "bash": "#!BASH",
    "sh": "#!BASH",
    "shell": "#!BASH",
    "zsh": "#!BASH",
    "console": "#!BASH",
    "r": "#!R",
}


def _unfenced(command: str) -> tuple[str, str]:
    """(language tag, code) of one markdown fence; ("", the old treatment) for anything else.

    ``command.strip("```")`` removed the backticks and kept the tag, so ``<execute>```python ...```
    </execute>`` ran ``python`` as its first statement and failed with ``NameError: name 'python'
    is not defined`` (hunt 2026-09-30, u23-transcriptomics-skills-15).
    """
    m = _FENCED_BLOCK_RE.match(command)
    if m:
        return m.group(1).lower(), m.group(2).strip()
    return "", command.strip("```").strip()


class PythonREPL:
    """A Python REPL with persistent namespace and matplotlib plot capture."""

    def __init__(self):
        self._namespace = {}
        self._captured_plots = []

    def run(self, command: str) -> str:
        """Executes the provided Python command in a persistent environment and returns the output.
        Variables defined in one execution will be available in subsequent executions.
        """
        return _with_warnings(*self.run_cell(command))

    def run_cell(self, command: str) -> tuple[str, "_WarningLog"]:
        """What :meth:`run` returns, with the warnings not yet put after it: the cell's text -- its
        output, its error, the cap notice -- and the warnings it raised. For a caller that puts lines
        of its own after both (the worker's notices, ``repl_host.Host.op_exec``): it must size the
        warnings with those lines' room taken first, or a text that fits with them is clipped for
        its warnings' sake (:func:`_with_warnings`)."""

        def execute_in_repl(command: str) -> tuple[str, _WarningLog]:
            """Helper function to execute the command in the persistent environment."""
            old_stdout = sys.stdout
            sys.stdout = mystdout = _BoundedStringIO()
            # This cell is now the one whose capture is live; see the restore in ``finally``.
            _live_capture[0] = mystdout
            _live_owner[0] = threading.get_ident()
            warned = _WarningLog()
            warned.install()
            # What ``run_with_timeout`` reads if this cell outlives its budget (and what the worker
            # reports while it runs): the tail of the output so far, and the warnings so far.
            outer_reader = publish_partial_output(lambda room=None: _partial_output(mystdout, warned, room))

            try:
                # Apply matplotlib monkey patches before execution
                self._apply_matplotlib_patches()
                # A printed DataFrame shows every column: this stdout is no terminal, and pandas would
                # otherwise fit frames to an 80-column one by dropping the middle columns (N5).
                apply_readable_display()
                # Backed .h5ad indexing under scipy >= 1.17 (FM-03); a no-op where it already works.
                ensure_backed_sparse_indexing()

                # Execute the command in the persistent namespace. Compiled first, exactly as exec()
                # would, so the warnings log knows which frames are this cell's (``_WarningLog.watch``).
                code = compile(command, "<string>", "exec")
                warned.watch(code)
                exec(code, self._namespace)
                output = mystdout.getvalue()

            except Exception as e:
                # Preserve anything already printed before the error. A cell that logs progress
                # ("loaded 5000 cells") then raises must still hand that context to the agent as the
                # observation — the bare exception alone loses *where* the failure happened. mystdout
                # is in scope here (assigned before the try).
                partial = mystdout.getvalue()
                hint = _non_ascii_syntax_hint(e, command)  # quotes the line itself when it fires
                err = f"Error: {_error_label(e)}" + (hint or _where_it_raised(e, command))
                output = (partial + "\n" + err) if partial else err
            except SystemExit as e:
                # ``sys.exit()`` / ``exit()`` are BaseException, so the branch above never caught
                # them and the exception escaped the REPL: the caller reported the opaque
                # ``Error in execution: 0`` (that is just ``str(SystemExit(0))``) and every line the
                # cell had printed was lost. Generated code reaches for sys.exit() as an ordinary
                # "stop here" — it is not a failure signal, so a 0/None status returns the output
                # as a normal success and only a non-zero status is reported, by name.
                partial = mystdout.getvalue()
                status = e.code
                if status is None or status == 0:
                    output = partial
                else:
                    err = f"Error: SystemExit: {status}"
                    output = f"{partial}\n{err}" if partial else err
            finally:
                # Restore ONLY if the stream we installed is still the one in place -- or, the last
                # paragraph below, if this cell put a stream of its own there.
                #
                # `run_with_timeout` cannot actually kill a blocked step -- it is a daemon-thread
                # join, and `PyThreadState_SetAsyncExc` does not interrupt a thread sitting in a
                # C call such as `subprocess.run`. The agent moves on at the deadline and the
                # orphan keeps running; when it eventually reaches this line, a LATER step has
                # already installed its own capture. An unconditional restore hands `sys.stdout`
                # back to the real stream and the later step's observation comes back EMPTY -- the
                # timeout of one tool call silently erasing the output of the next.
                #
                # Leaving it alone loses the orphan's own tail, which is correct: nobody is
                # reading it, the step it belonged to was reported as timed out minutes ago, and
                # the alternative destroys output somebody IS reading.
                #
                # What this does not fix, stated rather than implied: `print` resolves
                # `sys.stdout` at call time, so an orphan still running WHILE a later step holds
                # the capture writes into that step's buffer. Fixing that needs a thread-routed
                # stdout installed process-wide, which would reach notebooks and library users --
                # the thing the memo comment in `agent/execution.py` is careful about -- so it is
                # a separate decision and not a line to slip in here.
                #
                # A cell that rebound sys.stdout itself (``sys.stdout = open("log.txt", "w")``) is the
                # other case where the stream is not ``mystdout`` -- and there the restore is owed:
                # skipping it left the cell's file as the server's stdout for the life of the
                # process, so later cells "restored" to it and every print outside a cell, other
                # accounts' turns included, went into a file the cell chose (hunt 2026-09-30,
                # u23-transcriptomics-skills-17). See ``_cell_displaced_stdout`` for how that is
                # told apart from an orphan whose turn has ended.
                if sys.stdout is mystdout:
                    sys.stdout = old_stdout
                elif _live_capture[0] is mystdout and _cell_displaced_stdout(old_stdout):
                    sys.stdout = old_stdout
                if _live_capture[0] is mystdout:
                    _live_capture[0] = None
                    _live_owner[0] = None
                warned.uninstall()
                if outer_reader is not None:
                    publish_partial_output(outer_reader)
            # After the finally, so it lands on the error and SystemExit paths too: a cell that
            # printed 2 MB and *then* raised has lost its tail exactly the same way. The warnings
            # likewise: on success, on an error and on exit alike, shortened before the output is.
            return output + _cap_notice(mystdout), warned

        lang, code = _unfenced(command)
        if lang in _OTHER_LANGUAGE_FENCES:
            # Said, not run: the body of a ```bash block executed as Python only fails with a
            # SyntaxError about code that was never Python.
            marker = _OTHER_LANGUAGE_FENCES[lang]
            # A pair, like every other return here (``run`` unpacks it; fix/q4's contract): nothing ran,
            # so the warning log is an empty one (merge of 2026-10-02).
            return (
                f"Error: this cell is a ```{lang} block and this REPL runs Python. Send the code without "
                f"the fence, starting with {marker} on its own first line, to run it as {lang}."
            ), _WarningLog()
        return execute_in_repl(code)

    def _capture_matplotlib_plots(self, only=None, close=True):
        """Capture any matplotlib plots that might have been generated during execution.

        ``only`` restricts the snapshot to one figure; ``None`` means every open one. ``close``
        frees the figures afterwards.

        Snapshotting and closing used to be one indivisible act, and that was wrong for one of the
        two callers. Closing is destructive, and only the caller knows whether the figure is
        finished: ``show()`` ends a figure's life under a non-interactive backend, but ``savefig()``
        does not -- matplotlib leaves the figure open and cells routinely keep drawing into it.
        Closing from ``savefig`` meant the next ``plt.savefig`` resolved ``gcf()`` to a fresh empty
        figure, so every cell that saved twice wrote a blank second PNG and was told it had saved.
        """
        try:
            import matplotlib.pyplot as plt

            # Check if there are any active figures
            figures = [only] if only is not None else [plt.figure(n) for n in plt.get_fignums()]
            if figures:
                for fig in figures:
                    # Save figure to base64. Figure.savefig is the unpatched method, so this does
                    # not re-enter savefig_with_capture below.
                    buffer = io.BytesIO()
                    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                    buffer.seek(0)

                    # Convert to base64
                    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    plot_data = f"data:image/png;base64,{image_data}"

                    # Add to captured plots if not already there
                    if plot_data not in self._captured_plots:
                        self._captured_plots.append(plot_data)

                    # Close the figure to free memory -- only when the caller says it is finished
                    if close:
                        plt.close(fig)

        except ImportError:
            # matplotlib not available
            pass
        except Exception as e:
            print(f"Warning: Could not capture matplotlib plots: {e}")

    def _apply_matplotlib_patches(self):
        """Apply simple monkey patches to matplotlib functions to automatically capture plots."""
        try:
            import matplotlib.pyplot as plt

            # Only patch if matplotlib is available and not already patched
            if hasattr(plt, "_spatialomicsgym_patched"):
                return

            # Store original functions
            original_show = plt.show
            original_savefig = plt.savefig

            # Capture self for use in closures
            repl_instance = self

            def show_with_capture(*args, **kwargs):
                """Enhanced show function that captures plots before displaying them."""
                if _live_capture[0] is None:
                    return original_show(*args, **kwargs)  # not a cell's call; see savefig_with_capture
                # Capture any plots before showing, and close them: show() is terminal under a
                # non-interactive backend, so this is the last chance to snapshot the figure and
                # nothing draws into it afterwards.
                repl_instance._capture_matplotlib_plots()
                # Print a message to indicate plot was generated
                print("Plot generated and displayed")
                # Call the original show function
                return original_show(*args, **kwargs)

            def savefig_with_capture(*args, **kwargs):
                """Enhanced savefig function that captures plots after saving them."""
                if _live_capture[0] is None:
                    # No cell holds stdout, so this call is not a cell's. These wrappers stay on the
                    # pyplot MODULE for the rest of the process, so after the first cell every other
                    # caller -- a tool worker imported into the same process, a report renderer, a
                    # later test in the same pytest run -- had "Plot saved to:" printed into ITS
                    # stdout and its figure filed under this REPL's plots. somde_worker's stdout is
                    # its JSON channel; the full gates of 2026-10-01 and -02 failed on that line.
                    return original_savefig(*args, **kwargs)
                # Get the filename from args if provided
                filename = args[0] if args else kwargs.get("fname", "unknown")
                # Call the original savefig function
                result = original_savefig(*args, **kwargs)
                # Capture the one figure that was just written -- plt.savefig() saves gcf() -- and
                # leave it open, because plt.savefig() does. Closing here blanked every PNG after
                # the first in any cell that saved more than once (a two-panel subplot filled in one
                # axes at a time is the common shape) while still printing "Plot saved to:" for the
                # blank one.
                repl_instance._capture_matplotlib_plots(only=plt.gcf(), close=False)
                # Print a message to indicate plot was saved
                print(f"Plot saved to: {filename}")
                return result

            # Replace functions with enhanced versions
            plt.show = show_with_capture
            plt.savefig = savefig_with_capture

            # Mark as patched to avoid double-patching
            plt._spatialomicsgym_patched = True

        except ImportError:
            # matplotlib not available
            pass
        except Exception as e:
            print(f"Warning: Could not apply matplotlib patches: {e}")

    def get_captured_plots(self):
        """Get all captured matplotlib plots."""
        return list(self._captured_plots)

    def clear_captured_plots(self):
        """Clear all captured matplotlib plots."""
        self._captured_plots.clear()


# Default instance for backward compatibility
_default_repl = PythonREPL()


def process_isolation() -> bool:
    """True only when model-written code must run in the separate REPL worker (``repl_host``).

    False whenever a scored run is in progress -- benchmarking wins, the same shape as
    ``env_fallback.recovery_active()`` and ``know_how.enrolment.enrolment_mode()`` -- and False on
    any failure to read the configuration: the failure direction is "behave as before", never
    "quietly move the model's code somewhere else".
    """
    try:
        from spatialomicsgym.config import default_config
    except Exception:
        return False
    if getattr(default_config, "benchmarking_enabled", False):
        return False
    return str(getattr(default_config, "repl_isolation", "inprocess") or "").strip().lower() == "process"


def forget_repl_name(name: str) -> bool:
    """Unbind one name from the shared REPL namespace. True if it was there.

    ``remove_custom_tool`` cleared ``_custom_functions``, ``_custom_tools``, the registry and
    ``builtins._spatialomicsgym_custom_functions`` -- but not this namespace, which is where
    ``tool_conversion`` had also written the wrapper. So a tool the user deleted, or one the
    config merge refuses to serve, stayed callable from ``<execute>`` for the life of the
    process: the model could invoke a wrapper every catalog said did not exist, while
    ``resync_user_tools``' own docstring promised "a trashed tool stops being callable".
    """
    backend = _backend()
    if backend is not None:
        return bool(backend.forget(name))
    return _default_repl._namespace.pop(name, _MISSING) is not _MISSING


_MISSING = object()


# --------------------------------------------------------------------------------------------- #
# The process backend (repl_isolation = "process"): every function below that touches
# ``_default_repl`` asks ``_backend()`` first, so no caller changes when the cells move.
# --------------------------------------------------------------------------------------------- #
_process_backend: list = []
#: The figures the LAST cell drew in the worker, as data URLs -- what ``get_captured_plots``
#: hands the execute node in process mode.
_process_plots: list[str] = []
#: What the next cell needs from the server: the agent (for the env-failure ledger), the
#: budget, and the per-turn environment to forward. Set by ``bind_process_cell``.
_process_cell: dict = {}


def install_process_backend(client) -> None:
    """Make ``client`` (a ``repl_client.ReplProcess``) the worker every cell goes to in process mode."""
    _process_backend[:] = [client] if client is not None else []


def process_backend():
    """The installed worker client, or ``None``. Does not consult the mode."""
    return _process_backend[0] if _process_backend else None


class BrokenBackend:
    """What the portal installs when its worker could not be set up: every cell fails in words.

    The alternative -- falling back to an in-process REPL or to a same-user worker -- would run
    model code as the server with the server's environment on exactly the box where the boundary
    was asked for and could not be built (the audit's S4). A cell that says why it could not run
    is the honest floor.
    """

    alive = False
    pid = None
    uid = None
    account = ""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def exec(self, code, timeout, *, env_failures=None, general_env_calls=0, env=None):
        return {
            "output": f"Error in execution: the REPL worker could not be started, so this cell did not run ({self.reason}). "
            "An operator must fix the portal; see the server log.",
            "figures": [],
            "env_failures": dict(env_failures or {}),
            "general_env_calls": general_env_calls,
        }

    def boundary(self):
        return "process-only"

    def exec_shell(self, kind, code, timeout, *, env=None):
        """A shell cell fails in words too.

        Without this the portal would fall back to running ``#!BASH`` in the server process as root
        -- on exactly the box where the boundary was asked for and could not be built, which is the
        failure this class exists to refuse for python.
        """
        return {
            "output": f"Error in execution: the REPL worker could not be started, so this {kind} cell did not run "
            f"({self.reason}). An operator must fix the portal; see the server log.",
            "figures": [],
        }

    def summary(self, limit=20):
        return ""

    def forget(self, name):
        return False

    def reset(self):
        return None

    def set_env(self, mapping):
        return None

    def set_upcalls(self, upcalls):
        return None

    def configure(self, payload):
        return {"replayed": False, "error": self.reason}

    def set_account(self, account):
        self.account = str(account or "")

    def ensure(self):
        raise RuntimeError(self.reason)

    def kill(self):
        return None

    def shutdown(self):
        return None


def _backend():
    """The worker client, only when the mode says cells run there.

    Nothing installed means one of two things. The CLI or a notebook set
    ``SOG_REPL_ISOLATION=process`` in the environment itself: then a same-user worker is built on
    first use (crash isolation and a real kill, no privilege boundary, no provider key in its
    environment). Anything else -- a portal whose worker set-up failed before it could install a
    client -- gets NO fallback: the seam must not fail open into the server's process or the
    server's environment, so the cell fails in words instead.
    """
    if not process_isolation():
        return None
    if not _process_backend:
        import os as _os

        if (_os.environ.get("SOG_REPL_ISOLATION") or "").strip().lower() != "process":
            _process_backend[:] = [BrokenBackend("no worker was installed for this process")]
        else:
            try:
                from spatialomicsgym.tool.repl_client import ReplProcess

                _process_backend[:] = [ReplProcess()]
            except Exception as exc:
                _process_backend[:] = [BrokenBackend(f"the worker client could not be built: {exc}")]
    return _process_backend[0]


_cell_seq = [0]


def bind_process_cell(agent, *, timeout, env: dict) -> None:
    """Remember, for the next ``run_python_repl``, whose cell this is and what to forward."""
    _cell_seq[0] += 1
    _process_cell.clear()
    _process_cell.update({"agent": agent, "timeout": timeout, "env": dict(env or {}), "id": _cell_seq[0]})


def unbind_process_cell() -> None:
    """Forget the binding: a cell with no fresh binding must fail in words, never inherit the
    previous turn's agent, budget and environment (the audit's S8)."""
    _process_cell.clear()


def _run_in_process_backend(backend, command: str) -> str:
    from spatialomicsgym.tool import general_env

    if not _process_cell:
        return (
            "Error in execution: this cell was not bound to a turn (the worker could not be configured), "
            "so it did not run."
        )
    cell_id = _process_cell.get("id")
    agent = _process_cell.get("agent")
    timeout = _process_cell.get("timeout") or getattr(agent, "timeout_seconds", None) or 600
    ledger = dict(getattr(agent, "_env_failures", None) or {}) if agent is not None else {}
    result = backend.exec(
        command,
        float(timeout),
        env_failures=ledger,
        general_env_calls=general_env.calls_this_turn(),
        env=_process_cell.get("env") or {},
    )
    # Only the cell that is still current may leave figures behind: an abandoned cell's thread
    # finishing late must not credit a later step with its figures (the audit's S11).
    if _process_cell.get("id") == cell_id:
        _process_plots[:] = list(result.get("figures") or [])
    # The ledger the worker's notices wrote, merged onto the real agent: rescue.py and
    # env_fallback.py keep reading ``agent._env_failures`` and never learn where the cell ran.
    if agent is not None:
        try:
            merged = dict(getattr(agent, "_env_failures", None) or {})
            merged.update(result.get("env_failures") or {})
            agent._env_failures = merged
        except Exception:
            pass
    try:
        general_env.set_calls_this_turn(int(result.get("general_env_calls") or 0))
    except Exception:
        pass
    return str(result.get("output") or "")


def repl_namespace_summary(limit: int = 20) -> str:
    """One line naming what cells have left bound in the shared REPL: ``adata: AnnData 3000x2000, ...``.

    For the rescue prompt (``agent/rescue.py``): a second round is cheap precisely because the
    namespace survives the first, and this is how the model learns what it can reuse instead of
    reloading. Skips modules, callables and private names; never raises -- an unreadable value
    is described by its type alone.
    """
    backend = _backend()
    if backend is not None:
        try:
            return str(backend.summary(limit) or "")
        except Exception:
            return ""
    return _summarize_namespace(_default_repl._namespace, limit)


def _summarize_namespace(namespace: dict, limit: int = 20) -> str:
    """The body of :func:`repl_namespace_summary`, over any namespace -- the worker runs it on its own."""
    import inspect

    parts: list[str] = []
    try:
        for name, value in list(namespace.items()):
            if not isinstance(name, str) or name.startswith("_") or name in ("In", "Out"):
                continue
            if inspect.ismodule(value) or callable(value):
                continue
            desc = type(value).__name__
            try:
                shape = getattr(value, "shape", None)
                if shape is not None:
                    desc += " " + "x".join(str(int(s)) for s in shape)
                elif isinstance(value, (int, float, bool)) and len(repr(value)) <= 60:
                    desc += f" = {value!r}"
                elif isinstance(value, (str, list, tuple, set, dict)):
                    # A string's LENGTH, never its value: the summary goes into the rescue prompt,
                    # and a cell that bound an API key, a patient id or a password had them quoted
                    # into the next model call (driven 2026-09-20).
                    desc += f" (len {len(value)})"
            except Exception:
                pass
            parts.append(f"{name}: {desc}")
            if len(parts) >= limit:
                parts.append("...")
                break
    except Exception:
        return ""
    return ", ".join(parts)


def reset_repl_namespace() -> None:
    """Forget every name bound in the shared REPL.

    THE LEAK THIS EXISTS FOR. The namespace is one process-global object, and in the portal one
    agent serves every signed-in account. Every name a cell binds -- a loaded ``AnnData``, a file
    path, an API response -- was readable by the next person's cell. Driven: ``patient_secret``
    set by one turn came back as ``MRN-00042`` in a later, unrelated one.
    ``_bind_conversation_thread`` separates the recap for exactly this reason; the REPL is the
    half that was not separated.

    The portal calls this when the account driving the agent changes -- see
    ``AgentHandle.events``. It is deliberately NOT called between turns or between conversations
    of the same person: "variables defined in one execution will be available in subsequent
    executions" is the documented behaviour of this REPL and the thing a researcher relies on
    across a session. The account boundary is the one place that promise should not hold.

    Per-thread REPL instances would be the fuller answer. They are not this change: keying the
    namespace would mean threading that key through ``run_python_repl`` and every caller of it,
    including the benchmark execution path, and the leak is fixed at its actual boundary here.

    In process mode the worker is killed and respawned lazily instead: a fresh process is the
    only reset that also drops imported modules, monkeypatches and whatever a cell did to the
    interpreter -- and it is what makes the account boundary a process boundary.
    """
    backend = _backend()
    if backend is not None:
        try:
            backend.reset()
        except Exception:
            pass
        _process_plots.clear()
        return
    _default_repl._namespace.clear()
    # pyplot's figure manager is the other thing a cell leaves behind in this process: a figure the
    # last account drew and saved (Figure.savefig leaves it open) was snapshotted by the next
    # account's first plt.show() and landed in that account's step (hunt 2026-09-30,
    # u23-transcriptomics-skills-16). Only when pyplot is already loaded -- importing it here
    # would pick a backend for a process that never drew anything.
    plt = sys.modules.get("matplotlib.pyplot")
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass
    _default_repl.clear_captured_plots()


def run_python_repl(command: str) -> str:
    """Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.
    """
    backend = _backend()
    if backend is not None:
        return _run_in_process_backend(backend, command)
    return _default_repl.run(command)


def get_captured_plots():
    """Get all captured matplotlib plots."""
    if _backend() is not None:
        return list(_process_plots)
    return _default_repl.get_captured_plots()


def clear_captured_plots():
    """Clear all captured matplotlib plots."""
    if _backend() is not None:
        _process_plots.clear()  # the worker clears its own at the start of every cell
        return
    _default_repl.clear_captured_plots()


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Parameters
    ----------
        function_name (str): Fully qualified function name
            (e.g., 'spatialomicsgym.utils.tool_conversion.write_python_code')

    Returns
    -------
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = str(function_name).split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]
    if not module_path or not func_name:
        return (
            f"Error: Could not find function '{function_name}'. Details: expected a fully qualified "
            "name such as 'package.module.function'."
        )

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    # ValueError/TypeError/OSError too: a builtin or ufunc has no Python source (TypeError), a
    # compiled or frozen module has none on disk (OSError), and each of them raised out of a tool
    # documented to return this string (hunt 2026-09-30, u23-transcriptomics-skills-18).
    except (ImportError, AttributeError, ValueError, TypeError, OSError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"


# def request_human_feedback(question, context, reason_for_uncertainty):
#     """
#     Request human feedback on a question.

#     Parameters:
#         question (str): The question that needs human feedback.
#         context (str): Context or details that help the human understand the situation.
#         reason_for_uncertainty (str): Explanation for why the LLM is uncertain about its answer.

#     Returns:
#         str: The feedback provided by the human.
#     """
#     print("Requesting human feedback...")
#     print(f"Question: {question}")
#     print(f"Context: {context}")
#     print(f"Reason for Uncertainty: {reason_for_uncertainty}")

#     # Capture human feedback
#     human_response = input("Please provide your feedback: ")

#     return human_response


def download_synapse_data(
    entity_ids: str | list[str],
    download_location: str = ".",
    follow_link: bool = False,
    recursive: bool = False,
    timeout: int = 300,
    entity_type: str = "dataset",
):
    """Download data from Synapse using entity IDs.

    Uses the synapse CLI to download files, folders, or projects from Synapse.
    Requires SYNAPSE_AUTH_TOKEN environment variable for authentication, and the synapse CLI
    (pip install synapseclient) on PATH; returns an explanatory error if either is missing.

    CRITICAL: Always check entity type from query_synapse() search results or user hints and pass the correct entity_type!
    The default entity_type="dataset" may not be appropriate for your entity.

    IMPORTANT: Multiple entity IDs are only supported for entity_type="file".
    For datasets, folders, and projects, only a single entity_id is supported.

    Parameters
    ----------
    entity_ids : str or list of str
        Synapse entity ID(s) to download.
        - For files: Can be a single ID string or list of ID strings
        - For datasets/folders/projects: Must be a single ID string only
    download_location : str, default "."
        Directory where files will be downloaded (current directory by default)
    follow_link : bool, default False
        Whether to follow links to download the linked entity
    recursive : bool, default False
        Whether to recursively download folders and their contents
        ONLY valid for entity_type="folder" - ignored for other types
    timeout : int, default 300
        Timeout in seconds for each download operation
    entity_type : str, default "dataset"
        Type of Synapse entity ("dataset", "file", "folder", "project")
        MUST match the actual entity type from search results or user hints!
        The default "dataset" should only be used for actual datasets.
        Check the 'node_type' field in search results to determine correct type.

    Returns
    -------
    dict
        Dictionary containing download results and any errors

    Notes
    -----
    Requires SYNAPSE_AUTH_TOKEN environment variable with your Synapse personal
    access token for authentication.

    AGENT USAGE GUIDANCE:
    1. Always check the 'node_type' field from query_synapse() search results or user hints
    2. Pass the correct entity_type parameter matching the node_type
    3. Do NOT rely on the default entity_type="dataset" unless confirmed
    4. For multiple downloads, ensure all entities are of type "file"
    5. Only use recursive=True with entity_type="folder"

    Examples
    --------
    # After searching with query_synapse(), check node_type and use appropriate entity_type:

    # If search result shows 'node_type': 'dataset'
    download_synapse_data("syn123456", entity_type="dataset")

    # If search result shows 'node_type': 'file'
    download_synapse_data("syn654321", entity_type="file")

    # If search result shows 'node_type': 'folder'
    download_synapse_data("syn789012", entity_type="folder", recursive=True)

    # Multiple files (only if all are 'node_type': 'file')
    download_synapse_data(["syn111", "syn222"], entity_type="file")
    """
    import os
    import subprocess

    # Check for required authentication token
    synapse_token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not synapse_token:
        return {
            "success": False,
            "error": "SYNAPSE_AUTH_TOKEN environment variable is required for downloading",
            "suggestion": "Set SYNAPSE_AUTH_TOKEN with your Synapse personal access token",
        }

    # Check if synapse CLI is available. Report rather than install: this runs in whichever env the
    # agent was started in, and a bare `pip` there resolves against whatever is first on PATH, which
    # under `conda run` or an MCP worker need not even be the interpreter that will import it.
    try:
        subprocess.run(["synapse", "--version"], capture_output=True, check=True, timeout=60)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {
            "success": False,
            "error": f"The synapse CLI is not available in this environment: {e}",
            "suggestion": "Install it where the agent runs: pip install synapseclient",
        }

    # Ensure entity_ids is a list
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]

    # Validate that multiple IDs are only used with file entity type
    if len(entity_ids) > 1 and entity_type != "file":
        return {
            "success": False,
            "error": f"Multiple entity IDs are only supported for entity_type='file'. "
            f"For entity_type='{entity_type}', only a single entity_id is supported.",
            "suggestion": "Use a single entity_id string instead of a list, or change entity_type to 'file'",
        }

    # Validate that recursive is only used with folder entity type
    if recursive and entity_type != "folder":
        return {
            "success": False,
            "error": f"recursive=True is only valid for entity_type='folder'. "
            f"For entity_type='{entity_type}', recursive should be False.",
            "suggestion": "Set recursive=False, or change entity_type to 'folder' if appropriate",
        }

    # Create download directory if it doesn't exist
    os.makedirs(download_location, exist_ok=True)

    results = []
    errors = []

    # The token travels in the environment, never on the command line. ``-p <token>`` put it in
    # argv for the whole download -- readable by every local user through ps and /proc/<pid>/cmdline
    # -- and a failure with empty stderr reported str(CalledProcessError), which quotes argv, into
    # the observation the model reads and the provider receives (hunt 2026-09-30,
    # u23-transcriptomics-skills-14). synapseclient reads SYNAPSE_AUTH_TOKEN itself.
    child_env = {**os.environ, "SYNAPSE_AUTH_TOKEN": synapse_token}

    def _redacted(text) -> str:
        return str(text or "").replace(synapse_token, "[SYNAPSE_AUTH_TOKEN]")

    for entity_id in entity_ids:
        try:
            # Build the synapse download command; authentication comes from child_env
            if entity_type == "dataset":
                # For datasets, use query syntax to download the actual files
                cmd = [
                    "synapse",
                    "get",
                    "-q",
                    f"select * from {entity_id}",
                    "--downloadLocation",
                    download_location,
                ]
            else:
                # For files, folders, projects, use direct ID
                cmd = ["synapse", "get", entity_id, "--downloadLocation", download_location]

            # Add recursive flag only for folders (validation above ensures recursive is only True for folders)
            if entity_type == "folder" and recursive:
                cmd.append("-r")

            if follow_link:
                cmd.append("--followLink")

            # Execute download
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout, env=child_env)

            results.append(
                {
                    "entity_id": entity_id,
                    "success": True,
                    "stdout": _redacted(result.stdout),
                    "download_location": download_location,
                }
            )

        except subprocess.CalledProcessError as e:
            # The exit status and synapse's own stderr, never str(e): that string is the command.
            detail = _redacted(e.stderr).strip() or "no error output"
            error_msg = f"Failed to download {entity_id}: synapse exited with status {e.returncode}: {detail}"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})
        except subprocess.TimeoutExpired:
            error_msg = f"Download timeout for {entity_id} (>{timeout} seconds)"
            errors.append(error_msg)
            results.append({"entity_id": entity_id, "success": False, "error": error_msg})

    # Summary
    successful_downloads = [r for r in results if r["success"]]
    failed_downloads = [r for r in results if not r["success"]]

    return {
        "success": len(failed_downloads) == 0,
        "total_requested": len(entity_ids),
        "successful": len(successful_downloads),
        "failed": len(failed_downloads),
        "download_location": download_location,
        "results": results,
        "errors": errors if errors else None,
    }
