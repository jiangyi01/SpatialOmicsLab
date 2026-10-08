"""Read a reply from a provider that will not take ``stop`` as a stream, and stop reading at its first stop.

WHY. The ReAct loop is built on a turn ending at its first ``</execute>`` or ``</solution>``. The
OpenAI Responses family (gpt-5.x) rejects ``stop``, so ``llm.py`` removes it from the request and puts
nothing in its place: the model writes on past its first block until it decides to end, and
``cut_at_stop_sequence`` (``agent/execution.py``) throws the rest away. Over the 48 E-06 trials 182
replies were cut and 399,417 characters discarded, the median cut 1,045 and the largest 57,928: the
call on cumulus r2 that spent 22,254 output tokens and about 285 s, where the trial's other calls
took 5-40 s. Nothing that came after the stop was ever used.

WHAT. :func:`read_to_first_stop` streams the reply through the langchain interface (``llm.stream``),
grows the text a piece at a time by the rule a reply read whole is joined by (``llm.ItemJoin``), asks
the loop's own stop rule after every piece, and closes the stream at the first stop it finds. The
rule reads only what comes before a stop, so the text read so far is cut exactly where the whole
reply would have been, and the message ``generate()`` stores is byte-identical to the one the whole
reply, joined by the same rule, gives.

WHAT IT COSTS, AND WHAT IS NOT GUESSED. The provider reports token usage only in the event that ends
a reply, and a stream closed at its stop never gets there. Such a call is recorded with no counts --
``None``, never a ``0`` and never an estimate, the rule ``agent/usage.py`` keeps -- so a turn that
had one is not ``complete``, and a cost comparison against an arm that was not streamed has to hold
that in view. A stream that runs to its end reports usage as a reply read whole does.

WHERE IT WILL NOT TRUST THE STREAM. Only two ends are taken as a reply: a stop sequence, or the
provider's completion event (``status == "completed"``). langchain_openai's stream drops the events
for a reply that failed or was cut short (``response.failed``, ``response.incomplete``, ``error``),
so a stream that simply stops may be half a reply; that one is asked for again, not streamed, and
whatever that returns is what a reply read whole would have been.

THE FAILURES A STREAM ADDS. A read that was one request is now many, and two things change with it.
A transport failure while the body is read (a dropped connection, a read timeout) reaches the caller
as raw ``httpx`` -- the SDK maps those only while it sends the request -- so it is mapped here the way
the SDK maps the same failure, and ``invoke_with_backoff`` retries it as it did. And the request
timeout bounded a reply read whole from start to finish, while on a stream it bounds only the gap
between two events: a reply still being written when that much time has passed ends here with
:class:`StreamedReplyTimeout`, a timeout like the one it replaces, so a reply that never stops cannot
run longer than it could before.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import openai
from langchain_core.messages import AIMessage
from langchain_core.messages.ai import add_usage

from spatialomicsgym.llm import ItemJoin, content_to_text

if TYPE_CHECKING:
    from collections.abc import Callable

#: How langchain_openai's ``v0`` stream names a message item: the chunk that opens one carries the
#: item's id (``AIMessage.id`` is where ``v0`` keeps a message item id). Every other chunk carries
#: the run id langchain gives it. The same prefix the library itself tests for.
_MESSAGE_ITEM_PREFIX = "msg_"


class StreamedReplyTimeout(openai.APITimeoutError):
    """The reply was still being written when the request timeout ran out, and no stop had come."""

    def __init__(self, seconds: float) -> None:
        super().__init__(request=None)  # type: ignore[arg-type]  # no single request: the time is the stream's
        self.message = (
            f"The reply was still being written {seconds:.0f}s after it was requested, with no stop sequence "
            "in it. That is the request timeout (llm_request_timeout_seconds, SOG_LLM_REQUEST_TIMEOUT; 0 "
            "lifts it); a streamed reply is held to it from start to finish, as one read whole is."
        )
        self.args = (self.message,)


def opened_item(chunk: Any) -> str | None:
    """The message output item ``chunk`` opens, or None."""
    item = getattr(chunk, "id", None)
    return item if isinstance(item, str) and item.startswith(_MESSAGE_ITEM_PREFIX) else None


def _request_of(exc: httpx.TransportError) -> httpx.Request | None:
    try:
        return exc.request
    except RuntimeError:  # httpx raises rather than return None when no request was attached
        return None


def read_to_first_stop(
    llm: Any,
    messages: Any,
    first_stop_end: Callable[[str, int], int | None],
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[AIMessage, bool]:
    """``(reply, stopped)``: ``llm``'s reply to ``messages``, read as a stream up to its first stop.

    ``first_stop_end(text, after)`` is the loop's stop rule (``execution.first_stop_end``): where the
    first stop in ``text`` ends, given none ends within its first ``after`` characters. ``stopped``
    says the stream was closed there: the reply is everything read up to and including the piece
    that completed the stop, with only the usage that had arrived by then -- from a Responses
    stream, none.
    """
    timeout = getattr(llm, "request_timeout", None)
    bounded = isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0
    deadline = clock() + float(timeout) if bounded else None

    join = ItemJoin()
    item: str | None = None
    text = ""
    meta: dict[str, Any] = {}
    usage = None
    stream = llm.stream(messages)
    try:
        for chunk in stream:
            item = opened_item(chunk) or item
            got = getattr(chunk, "response_metadata", None)
            if isinstance(got, dict):
                meta.update(got)
            if getattr(chunk, "usage_metadata", None):
                usage = add_usage(usage, chunk.usage_metadata)
            scanned = len(text)
            text += join.piece(content_to_text(chunk.content), item)
            if len(text) > scanned and first_stop_end(text, scanned) is not None:
                return AIMessage(content=text, response_metadata=meta, usage_metadata=usage), True
            if deadline is not None and clock() > deadline:
                raise StreamedReplyTimeout(float(timeout))
    except httpx.TimeoutException as exc:
        raise openai.APITimeoutError(request=_request_of(exc)) from exc  # type: ignore[arg-type]
    except httpx.TransportError as exc:
        message = str(exc) or "Connection error."
        raise openai.APIConnectionError(message=message, request=_request_of(exc)) from exc  # type: ignore[arg-type]
    finally:
        # Explicitly, not left to the collector: an error raised in here keeps this frame alive in its
        # traceback -- through the backoff's wait before a retry -- and with it the generator and the
        # connection it holds open.
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if meta.get("status") == "completed":
        return AIMessage(content=text, response_metadata=meta, usage_metadata=usage), False
    print(
        "[responses_stream] The streamed reply ended without a stop sequence or the provider's completion "
        "event, so it may be half a reply; asking for it again, not streamed."
    )
    return llm.invoke(messages), False
