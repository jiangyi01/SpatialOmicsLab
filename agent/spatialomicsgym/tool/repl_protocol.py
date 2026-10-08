"""The wire between the server and the REPL worker (``repl_host``): length-prefixed JSON frames.

Stdlib only, imported by both sides. Each frame is a 4-byte big-endian length followed by that
many bytes of UTF-8 JSON, over a plain file descriptor -- the same JSON-on-a-pipe family
``tools_user/worker_utils.py`` uses for tool workers, hardened for a process that runs arbitrary
model-written code: the worker moves its protocol descriptors away from 0/1 before running a
single cell, so a stray ``print``, a C library writing to stdout, ``os.system`` or ``input()``
can never corrupt a frame (``repl_host.main`` records how).

Frame shapes (documented here, asserted nowhere else):

* request  ``{"id": int, "op": str, ...}``          server -> worker
* reply    ``{"id": int, "ok": bool, ...}``         worker -> server, same ``id``
* hello    ``{"hello": {...}}``                     worker -> server, once, first
* upcall   ``{"id": int, "upcall": str, "args": {}}`` worker -> server, DURING a cell
* answer   ``{"id": int, "upcall_result": ...}``    server -> worker, same ``id``
* partial  ``{"id": int, "partial": str}``           worker -> server, DURING a cell, the request's ``id``

Upcalls are how the worker asks the server to do what only the server may do (create an
environment as root, write a tool file, call the model): the worker blocks on the answer, the
server services it from the thread waiting on the cell, and the cell continues.

Partials are the worker reporting, every half second while it changes, the tail of what the cell
has printed: the server kills a cell at its deadline and cannot ask it anything then, so the last
report is what a timed-out cell's observation says it had got to. They never move the deadline.
"""

from __future__ import annotations

import json
import os
import select
import struct
import time
from typing import Any

#: A single frame's ceiling. Cell output is bounded at ~2 MB by ``_BoundedStringIO``; figures are
#: base64 PNGs and a cell that saved a dozen at 150 dpi is still far under this.
MAX_FRAME_BYTES = 64 * 2**20

_HEADER = struct.Struct(">I")

#: How often, while a cell runs, the worker reports what it has printed so far -- sent only when it
#: changed. The server kills the worker at the deadline and cannot ask it anything then (the cell
#: holds the request loop), so the last report is all a timed-out cell's observation can carry,
#: and the observation says how stale it can be in these words.
PARTIAL_INTERVAL = 0.5


class FrameTimeout(TimeoutError):
    """No complete frame arrived before the deadline."""


class FrameTooLarge(ValueError):
    """A frame that would exceed :data:`MAX_FRAME_BYTES` -- refused before it is sent."""


def write_frame(fd: int, payload: dict[str, Any]) -> None:
    """Serialise ``payload`` and write one frame to ``fd``, whole.

    ``ensure_ascii=False`` so a figure's base64 and a cell's Unicode output cost their UTF-8
    length and no more. Raises :class:`FrameTooLarge` rather than sending a frame the reader is
    documented to refuse.
    """
    data = json.dumps(payload, ensure_ascii=False, default=_fallback).encode("utf-8", errors="replace")
    if len(data) > MAX_FRAME_BYTES:
        raise FrameTooLarge(f"frame of {len(data)} bytes exceeds the {MAX_FRAME_BYTES}-byte ceiling")
    _write_all(fd, _HEADER.pack(len(data)) + data)


def read_frame(fd: int, deadline: float | None) -> dict[str, Any] | None:
    """Read one frame from ``fd``. ``None`` on EOF (the peer is gone).

    ``deadline`` is an absolute ``time.monotonic()`` value, or ``None`` to wait forever. A deadline
    that passes mid-frame raises :class:`FrameTimeout` -- the caller owns the peer and decides
    what a stalled peer means (the server kills the worker; the worker has no deadline).
    """
    header = _read_exact(fd, _HEADER.size, deadline)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise FrameTooLarge(f"peer announced a {length}-byte frame; the ceiling is {MAX_FRAME_BYTES}")
    body = _read_exact(fd, length, deadline)
    if body is None:
        return None
    loaded = json.loads(body.decode("utf-8", errors="replace"))
    return loaded if isinstance(loaded, dict) else {"malformed": loaded}


def _fallback(value: Any) -> Any:
    """JSON for what the worker hands back that json cannot encode by itself: its repr."""
    try:
        return repr(value)
    except Exception:
        return "<unrepresentable>"


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _read_exact(fd: int, count: int, deadline: float | None) -> bytes | None:
    """``count`` bytes from ``fd``, or ``None`` on EOF before they all arrived."""
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        if deadline is not None:
            wait = deadline - time.monotonic()
            if wait <= 0:
                raise FrameTimeout("no frame before the deadline")
            ready, _, _ = select.select([fd], [], [], wait)
            if not ready:
                continue
        chunk = os.read(fd, min(remaining, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
