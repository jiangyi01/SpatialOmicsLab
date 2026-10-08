"""The frame-event vocabulary of the portal's SSE wire -- the one place the ten names are written.

``STREAM_PROTOCOL.md`` section 2 is the contract; this module is the machine-readable half of it.

Before it existed the names were literals in several hand-maintained places -- the ``_sse`` call
sites in :mod:`sog_portal.server`, the frame dicts in
:mod:`sog_portal.research`, a private ``_FINAL``/``_ERROR``/``_END`` block in
:mod:`spatialomicsgym.research.loop`, and the client's own comparisons in two readers -- and
nothing failed when one of them gained a name the others did not have. It had already happened:
``phase`` reached the server and one of the two client readers, and the contract document went on
calling the vocabulary "the eight event names" over a table of nine.

**Why this lives in ``contracts/``, outside both the portal and the research loop.**
``sog_portal.__init__`` imports ``server``, which imports
:mod:`spatialomicsgym.research.loop`. A loop importing anything under ``webui`` would close that
circle -- and the loop is headless on purpose: it must run with no server, no browser and no
FastAPI installed. ``spatialomicsgym/__init__`` is stdlib-only and ``contracts/__init__`` imports
nothing, so this module costs a headless caller nothing.

**Three vocabularies share this wire and this is only one of them.** These are frame EVENT names.
``step.kind`` (``observation`` / ``toolerror`` / ``tool`` / ``reasoning`` / ``step`` / ``notice``)
and ``error.kind`` (``timeout`` / ``config`` / ``busy`` / ``gone`` / ``research`` / ...) are
different lists that merely happen to be strings on the same wire. ``research`` is a member of two
of them and means a different thing in each: the multi-round investigation's phase frame, and the
``error.kind`` a research run's synthesised terminal frame carries. Merging them would be a silent
regression, so they are kept apart deliberately; ``step.kind`` has its own single source in the
client's ``lib/stepTone.ts``.

**The TypeScript mirror is ``frontend/src/lib/streamEvents.ts``**, and it is a separate,
hand-kept-equal file rather than something generated. Generating it would put ``npm`` in front of
``pytest``: the Python suite has to collect on a box with no Node toolchain, and a gate that
cannot be run is worse than the duplication it cures.
``test/test_the_stream_vocabulary_has_one_source.py`` is the coupling -- it fails the moment the
two lists differ, the moment a producer emits a name that is in neither, the moment a client
branches on one, and the moment the contract document tabulates a different set.
"""

from __future__ import annotations

#: First frame of every turn. ``{message}`` -- what the USER typed, not what the agent received.
EVENT_START = "start"

#: The only content frame: ``{i, text, kind, note?, lang?, ok?}``. Whole messages, not deltas.
EVENT_STEP = "step"

#: Success only: ``{text, raw, drop, run?}``. ``drop`` names the step row that echoed the answer.
EVENT_FINAL = "final"

#: Mid-stream failure: ``{message, kind?, run?}``. Never an HTTP status, and always followed by
#: ``end``, because the client unlocks its composer on ``end`` rather than on ``final``.
EVENT_ERROR = "error"

#: Terminal: ``{steps}``. The raw index, including the row ``final.drop`` removes.
EVENT_END = "end"

#: Once, when the server minted a conversation for this turn: ``{id, title, new}``.
EVENT_CONVERSATION = "conversation"

#: First frame on reattach: ``{conversation, turn_id, prompt, started, dropped, running, from, gap}``.
EVENT_ATTACH = "attach"

#: ``POST /api/research`` only: ``{phase, name, said?, citations?, metric?, ...}``.
EVENT_RESEARCH = "research"

#: A heartbeat, not a row: ``{phase: "running", lang, seconds}`` while a cell runs, or -- BEFORE ``start`` --
#: ``{phase: "queued", seconds, ahead}`` while a portal turn waits in line for the agent. Carries no ``i``, never
#: enters the trace, and ``final.drop`` can never address one -- see STREAM_PROTOCOL.md section 2.2.
EVENT_PHASE = "phase"

#: Once per turn, when the agent's ONE rescue round begins: ``{reason}`` -- why the first attempt
#: stopped (``agent/rescue.py``). Not a row and carries no ``i``. The round's own ``step`` frames
#: follow it, and its solution is the turn's answer -- see STREAM_PROTOCOL.md section 2.4.
EVENT_RESCUE = "rescue"
#: After a turn's ``final`` / ``error`` and before ``end``: the explorer the chat recommends for what the turn used
#: or produced (``sog_portal/viz_suggest.py``). Display-only -- a client that ignores it loses nothing. STREAM_PROTOCOL.md
#: section 2.
EVENT_VIZ = "viz"
#: Once, after ``start`` on a turn the portal began: ``{turn_id}`` -- the turn's own id. What the stored placeholder
#: turn carries (``tid``) and what a follower re-attaches to (``/api/chat/attach?turn=&after=``). STREAM_PROTOCOL.md
#: section 2.5.
EVENT_TURN = "turn"
#: At most once, right after ``start``, on a turn of a chat with no data that asked for built-in data:
#: ``{v, used?, dataset?, candidates?, none?, task}`` -- what the portal attached for the turn, or the entries that fit
#: equally for the user to choose (``sog_portal/builtin_library.py``). STREAM_PROTOCOL.md section 2.6.
EVENT_LIBRARY = "library"

#: Every frame-event name on this wire, in the order STREAM_PROTOCOL.md section 2 tabulates them.
#: The order is part of the contract only in that the document and this tuple must agree; readers
#: dispatch by name.
#: ``error.kind`` value meaning "something else already holds this lock; nothing was started".
#:
#: NOT a member of :data:`STREAM_EVENTS` -- deliberately. That tuple is the frame EVENT names, and
#: this is a value carried *inside* an ``error`` frame. The module docstring above says the two
#: vocabularies are kept apart on purpose, and folding this in would break that and every reader
#: that iterates the tuple.
#:
#: It is here because it was a wire contract written out five times and named nowhere:
#: ``server.py`` emits it in an SSE error frame, in a JSON body on the reset path, and on the
#: already-running-in-this-conversation path; ``research/loop.py`` compares against it to decide
#: whether to wait; and the client branches on it. Renaming the string in one of those places
#: would make the research loop stop recognising a busy answer and treat it as a round that
#: failed -- a different sentence in the report, arrived at silently.
ERROR_KIND_BUSY = "busy"

#: The same refusal for the PROCESS-WIDE agent lock: some other turn -- another chat, another account
#: -- holds the one agent. Kept apart from ``busy`` (this conversation already has a turn) because
#: the client's sentence for ``busy`` -- "This chat already has a turn running ... open the running
#: turn ... or stop it" -- is false here: this chat runs nothing and its user can neither see nor stop
#: the other turn (hunt 2026-09-30, u01-server-a-5 / u02-server-b-4). The research loop waits out both.
ERROR_KIND_AGENT_BUSY = "agent_busy"
BUSY_KINDS: tuple[str, ...] = (ERROR_KIND_BUSY, ERROR_KIND_AGENT_BUSY)

STREAM_EVENTS: tuple[str, ...] = (
    EVENT_START,
    EVENT_STEP,
    EVENT_FINAL,
    EVENT_ERROR,
    EVENT_END,
    EVENT_CONVERSATION,
    EVENT_ATTACH,
    EVENT_RESEARCH,
    EVENT_PHASE,
    EVENT_RESCUE,
    EVENT_VIZ,
    EVENT_TURN,
    EVENT_LIBRARY,
)

#: For "is this a frame this portal speaks?" without paying for a linear scan. The keepalive is
#: deliberately absent: it is the bare comment ``": keepalive\n\n"`` and carries no ``event:`` line
#: at all, so it is not a member of this vocabulary and must not be given one.
STREAM_EVENT_SET: frozenset[str] = frozenset(STREAM_EVENTS)

__all__ = [
    "EVENT_ATTACH",
    "EVENT_CONVERSATION",
    "EVENT_END",
    "EVENT_ERROR",
    "EVENT_FINAL",
    "EVENT_PHASE",
    "EVENT_RESCUE",
    "EVENT_RESEARCH",
    "EVENT_START",
    "EVENT_STEP",
    "EVENT_VIZ",
    "STREAM_EVENTS",
    "STREAM_EVENT_SET",
]
