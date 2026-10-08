"""The one exception a communication read answers with.

Its ``kind`` is one of the reader child's refusal kinds, so the child can hand it on as it is; ``detail`` is prose
with no path (the child never knows one); ``knob`` names the setting that would lift a ``too_large``.
"""

from __future__ import annotations

KINDS = ("bad_request", "unreadable", "unsupported", "too_large", "no_join")


class CCCRefusal(Exception):
    """A communication request that cannot be answered, and why."""

    def __init__(self, kind: str, detail: str, knob: str = "") -> None:
        super().__init__(detail)
        self.kind = kind if kind in KINDS else "unreadable"
        self.detail = detail
        self.knob = knob


__all__ = ["KINDS", "CCCRefusal"]
