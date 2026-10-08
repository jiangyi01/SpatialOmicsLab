"""Wait out a provider's rate limit inside the turn, instead of ending the turn on it (R6).

The ReAct loop's model call had no per-call recovery: a 429 left ``generate``, the stream driver
turned it into a degradation note, and the turn ended. The client's own ``max_retries`` is 2 with
the SDK's 0.5-8 s backoff (a ``Retry-After`` of up to 60 s when the provider sends one), while an
Azure rate window is a minute. So a burst ended runs outright -- 2 of 360 archived SpatialBench
trials, one of them "110 seconds of a 21,600-second budget, answer None, a scored zero"
(``docs/audit/t4_candidates_recovery.md`` R6), and one of the first eight trials of the 2026-09-25
control arm, 51 seconds in, after its first call.

:func:`invoke_with_backoff` retries the SAME call on a transient provider error -- a rate limit, an
overload, a 5xx, a dropped connection -- waiting the longer of its schedule and the provider's own
``Retry-After``. Anything else (a 400, an authentication failure, a bug) propagates at once, exactly
as before: a retry loop must never mask a permanent error.

The budget is counted in WALL time since the first call, the calls included, not only the loop's
own waits. A failed call has already spent its time inside the client: ``max_retries + 1`` request
timeouts when the provider hangs (3 x 600 s at the defaults), or two ``Retry-After`` waits of up to
60 s each. A budget that counted only the waits let that repeat six times. Measured with the real
SDK on a fake clock, a hung provider then gave up after 11,048 s instead of a bare call's 1,801 s,
and a sustained 429 with ``Retry-After: 60`` after 1,029 s instead of 122 s -- past the portal's
720 s no-progress deadline, which ends the turn as a timeout and discards the agent session. So a
retry starts only while the time since the first call, plus the next wait, fits the budget, and a
call that alone outlasts it (the hung provider) is not repeated at all.

What the budget bounds is when the last retry may START, not when the loop ends. A call in flight
cannot be cut short, and a retried call can hang as long as any other: when the budget is spent
the last error propagates and the existing degradation path takes over, so the worst case is
today's outcome, at most one budget plus the call in flight later. Measured the same way, a
provider that answers 503 three times and then hangs fails a bare call at 3 s; the loop retries
once, 23 s in, that call spends the client's 3 x 600 s, and the loop fails at 1,824 s -- past the
portal's deadline, where the bare call was not.

A rate limit can also arrive with no status at all. A reply the loop reads as a stream
(``responses_stream``) can carry the provider's error as an event in the stream, and the openai SDK
raises that as a bare ``APIError`` -- no status, no headers. Azure does this for its token rate limit:
a portal turn on 2026-10-02 ended on "APIError (Your requests to gpt-6-astra for gpt-6-astra in eastus
have exceeded token rate limit.)" after the backoff had read it as permanent. Such an error is a rate
limit when its code or its message says so (:func:`_a_rate_limit_without_a_status`), and its wait is
the one its message names, if any; every other status-less error event is still not retried.

``SOG_LLM_TRANSIENT_WAIT_SECONDS`` sets the budget (default 300); ``0`` disables the waiting.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

#: HTTP statuses a retry can outlast: request timeout, rate limit, and EVERY 5xx. The openai and
#: anthropic clients themselves retry any status >= 500, and a gateway or CDN in front of a provider
#: answers 520-524 for the same origin failures a 502 or 504 names; listing four 5xx codes classed a
#: 524 (origin timeout) as permanent and ended the turn on the first one. 529 is Anthropic's
#: "overloaded".
TRANSIENT_STATUS = frozenset({408, 429, *range(500, 600)})
#: SDK exception class names for transient errors raised before any status exists: a refused or
#: dropped connection, and a request timeout (``APITimeoutError`` subclasses ``APIConnectionError``
#: in both SDKs). The transient classes the SDKs DO name -- ``RateLimitError``,
#: ``InternalServerError``, anthropic's ``OverloadedError`` -- are ``APIStatusError`` subclasses, so
#: they always carry a ``status_code`` and :data:`TRANSIENT_STATUS` decides them; listed here they
#: would never be consulted. Not every other error has a status: openai raises a status-less
#: ``APIError`` for an error event inside a streamed response, and ``LengthFinishReasonError`` /
#: ``ContentFilterFinishReasonError``. Those are not retried -- except a stream's error event that is a rate
#: limit (:func:`_a_rate_limit_without_a_status`). ``generate()`` DOES stream a client that
#: will not take ``stop`` (``responses_stream``); a transport failure while that stream is read
#: arrives as raw ``httpx``, and the reader maps it to the class the SDK gives the same failure before
#: the body -- ``APIConnectionError`` / ``APITimeoutError`` -- so it is retried here as it was.
TRANSIENT_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})
#: ``code`` / ``type`` values a provider gives a rate limit it reports in an error event, compared
#: lower-cased: openai's ``rate_limit_exceeded``, anthropic's ``rate_limit_error``, and Azure's bare
#: ``429``. Matched exactly -- a code is a name, and ``insufficient_quota`` is not a wait away.
_RATE_LIMIT_CODES = frozenset({"429", "rate_limit_exceeded", "rate_limit_error", "too_many_requests"})
#: The same fact in prose, for an event that carries only a message: "exceeded token rate limit",
#: "Rate limit reached for requests", "Too Many Requests".
_RATE_LIMIT_TEXT = re.compile(r"\brate[ _-]?limit|\btoo many requests\b", re.IGNORECASE)
#: A wait named in the message, where there are no headers to carry one: Azure's "Please retry after 6
#: seconds", openai's "Please try again in 1.5s" / "in 20ms".
_WAIT_IN_TEXT = re.compile(
    r"\b(?:retry after|try again in)\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|secs?|seconds?)?\b", re.IGNORECASE
)
#: Seconds to wait before each retry. A provider's Retry-After wins when it asks for longer.
DEFAULT_WAITS: tuple[float, ...] = (20.0, 40.0, 60.0, 60.0, 60.0)
#: A Retry-After beyond this is not honoured as asked; it is treated as this.
MAX_SINGLE_WAIT = 120.0


def _status(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _message_of(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    said = body.get("message") if isinstance(body, dict) else None
    if not isinstance(said, str) or not said:
        said = getattr(exc, "message", None)
    return said if isinstance(said, str) and said else str(exc)


def _a_rate_limit_without_a_status(exc: BaseException) -> bool:
    """An SDK ``APIError`` with no status that says it is a rate limit -- by its code, or in words.

    Only the SDKs' own error class is read this way, so a status-less error from anywhere else, which
    happens to mention a rate limit, is not retried on its wording.
    """
    if not any(cls.__name__ == "APIError" for cls in type(exc).__mro__):
        return False
    body = getattr(exc, "body", None)
    named = [getattr(exc, "code", None), getattr(exc, "type", None)]
    if isinstance(body, dict):
        named += [body.get("code"), body.get("type")]
    if any(isinstance(n, (str, int)) and str(n).strip().lower() in _RATE_LIMIT_CODES for n in named):
        return True
    return bool(_RATE_LIMIT_TEXT.search(_message_of(exc)))


def is_transient(exc: BaseException) -> bool:
    """A provider error a wait can outlast. Unknown errors are NOT transient -- they propagate."""
    status = _status(exc)
    if status is not None:
        return status in TRANSIENT_STATUS
    if _a_rate_limit_without_a_status(exc):
        return True
    return any(cls.__name__ in TRANSIENT_NAMES for cls in type(exc).__mro__)


def _wait_named_in_the_message(exc: BaseException) -> float | None:
    found = _WAIT_IN_TEXT.search(_message_of(exc))
    if found is None:
        return None
    value = float(found.group(1))
    if (found.group(2) or "s").lower().startswith("m"):
        value /= 1000.0
    return min(value, MAX_SINGLE_WAIT)


def retry_after_seconds(exc: BaseException) -> float | None:
    """The wait the provider asked for, from ``retry-after-ms`` or ``retry-after``, or None.

    A rate limit reported inside a stream has no headers; the wait its message names stands in.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return _wait_named_in_the_message(exc) if _a_rate_limit_without_a_status(exc) else None
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        try:
            raw = headers.get(key)
        except Exception:
            return None
        if raw is None:
            continue
        try:
            value = float(raw) * scale
        except (TypeError, ValueError):
            continue  # an HTTP-date form is not worth parsing here; the schedule covers it
        if value >= 0:
            return min(value, MAX_SINGLE_WAIT)
    return None


def transient_wait_budget() -> float:
    try:
        from spatialomicsgym.config import default_config

        return max(0.0, float(getattr(default_config, "llm_transient_wait_seconds", 300.0)))
    except Exception:
        return 300.0


def invoke_with_backoff(
    llm: Any,
    messages: Any,
    *,
    budget_seconds: float | None = None,
    waits: Sequence[float] = DEFAULT_WAITS,
    sleep: Callable[[float], None] | None = None,
    clock: Callable[[], float] | None = None,
) -> Any:
    """``llm.invoke(messages)``, retried on a transient provider error. A retry starts only while the
    wall time since the first call -- the calls and the waits between them -- plus its wait fits
    ``budget_seconds``; the call it starts is not cut short, so the loop can end one call past it.

    ``sleep`` and ``clock`` default to :func:`time.sleep` and :func:`time.monotonic`; a test passes a
    fake clock and a ``sleep`` that advances it.
    """
    budget = transient_wait_budget() if budget_seconds is None else max(0.0, budget_seconds)
    sleep = sleep or time.sleep  # resolved per call, so a caller (or a test) can replace time.sleep
    clock = clock or time.monotonic
    started = clock()
    for attempt in range(len(waits) + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            if not is_transient(exc) or attempt >= len(waits):
                raise
            wait = max(float(waits[attempt]), retry_after_seconds(exc) or 0.0)
            elapsed = clock() - started  # the failed calls count: a hung one took 3 x 600 s at the defaults
            if elapsed + wait > budget:
                raise
            print(
                f"[provider] {type(exc).__name__} ({_status(exc) or 'no status'}) {elapsed:.0f}s into a "
                f"{budget:.0f}s budget; waiting {wait:.0f}s, then retrying the same call ({attempt + 1}/{len(waits)})."
            )
            sleep(wait)
    raise AssertionError("unreachable")  # the loop either returns or raises
