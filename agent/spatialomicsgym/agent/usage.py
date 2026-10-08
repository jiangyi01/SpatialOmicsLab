"""What a turn cost, recorded per model call, because nothing recorded it before.

WHY THIS EXISTS. Task 4's rule is that a retained harness change must measure at or above the
baseline on **every** metric. Cost is one of them, and `BASELINE.md` had to list it as
unreportable: no per-turn token or cost record existed for this agent anywhere. That makes the rule
unenforceable for the whole family of candidates that trade model calls for accuracy -- a
verification pass costs +20-30% on a median ten-step trial, and without this there is no number to
weigh that against. So this is a **precondition of Phase 4.2, not a candidate within it**.

WHAT IT DELIBERATELY DOES NOT DO. It does not change a prompt, a message list, a tool choice, a
retry, or anything else the model sees. It reads what the provider already returned and appends a
row. That is what makes it safe to land while RL-3 is in force: a change with no behavioural delta
cannot move a benchmark metric, and its acceptance test is that the assembled prompt is
byte-identical, not that a score held.

WHY THE NUMBERS ARE OFTEN ABSENT, AND WHY THAT IS RECORDED RATHER THAN FILLED IN. Providers differ
about where usage lives and whether they send it at all: LangChain normalises some into
``usage_metadata``, others leave it in ``response_metadata["token_usage"]``, streaming responses
frequently omit it entirely, and a local model may never report it. An absent count is stored as
``None`` and never as ``0`` -- a zero is a measurement and this is the absence of one, and the
difference decides whether a mean over a run is honest. The same rule the research ledger already
follows for its metric.

NO PRICING TABLE. This records tokens, not dollars. A hardcoded price list goes stale the week a
provider changes one, and a cost figure computed from a stale table is worse than no cost figure
because it looks authoritative. Callers that want money multiply by a rate they own.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any

#: Where each provider hangs usage off a response. Ordered: the first that yields a usable mapping
#: wins. ``usage_metadata`` is LangChain's normalised home and is preferred for that reason.
_USAGE_ATTRS = ("usage_metadata",)
_USAGE_META_KEYS = ("token_usage", "usage")

#: Input-token spellings, in the order they are looked for. Providers disagree and always have.
_IN_KEYS = ("input_tokens", "prompt_tokens", "promptTokenCount")
_OUT_KEYS = ("output_tokens", "completion_tokens", "candidatesTokenCount")
_TOTAL_KEYS = ("total_tokens", "totalTokenCount")


def _first_int(mapping: Any, keys: tuple[str, ...]) -> int | None:
    """The first key present as a non-negative int, or ``None``. Never raises, never coerces a
    string that is not a number -- a provider sending ``"unknown"`` means unknown."""
    if not isinstance(mapping, dict):
        return None
    for k in keys:
        v = mapping.get(k)
        if isinstance(v, bool):  # bool is an int subclass and would read as 0/1
            continue
        if isinstance(v, int) and v >= 0:
            return v
        if isinstance(v, float) and v >= 0 and float(v).is_integer():
            return int(v)
    return None


def read_usage(response: Any) -> dict[str, int | None]:
    """``{"input": n|None, "output": n|None, "total": n|None}`` for one model response.

    Defensive in the same way ``completion_was_truncated`` is, and for the same reason: this runs
    on every model call on the live path, and a telemetry helper that can raise would take the turn
    with it. Every failure mode returns Nones.
    """
    blank: dict[str, int | None] = {"input": None, "output": None, "total": None}
    if response is None:
        return blank

    def _attr(obj: Any, name: str) -> Any:
        """``getattr`` with a default swallows only ``AttributeError``. A provider object whose
        ``usage_metadata`` is a property that raises anything else would come straight through and
        take the turn down -- from telemetry, which is the one thing here that must never fail a
        turn. Caught by this module's own test before it could."""
        try:
            return getattr(obj, name, None)
        except Exception:
            return None

    source: Any = None
    for attr in _USAGE_ATTRS:
        candidate = _attr(response, attr)
        if isinstance(candidate, dict) and candidate:
            source = candidate
            break
    if source is None:
        meta = _attr(response, "response_metadata")
        if isinstance(meta, dict):
            for key in _USAGE_META_KEYS:
                candidate = meta.get(key)
                if isinstance(candidate, dict) and candidate:
                    source = candidate
                    break
    if source is None:
        return blank

    got = {
        "input": _first_int(source, _IN_KEYS),
        "output": _first_int(source, _OUT_KEYS),
        "total": _first_int(source, _TOTAL_KEYS),
    }
    # A total the provider did not send, but which both halves determine, is arithmetic rather
    # than invention. The reverse -- splitting a total into halves -- is invention, so it is not
    # done.
    if got["total"] is None and got["input"] is not None and got["output"] is not None:
        got["total"] = got["input"] + got["output"]
    return got


#: Where a prompt-cache hit is reported, most normalised first. LangChain maps Azure/OpenAI Chat
#: Completions ``prompt_tokens_details.cached_tokens``, the Responses API's
#: ``input_tokens_details.cached_tokens`` and Anthropic's ``cache_read_input_tokens`` /
#: ``cache_creation_input_tokens`` into ``usage_metadata["input_token_details"]``; the raw spellings
#: are the fallback for a client that does not normalise.
_CACHE_READ_PATHS = (
    ("usage_metadata", "input_token_details", "cache_read"),
    ("response_metadata", "token_usage", "prompt_tokens_details", "cached_tokens"),
    ("response_metadata", "usage", "input_tokens_details", "cached_tokens"),
    ("response_metadata", "usage", "cache_read_input_tokens"),
)
_CACHE_WRITE_PATHS = (
    ("usage_metadata", "input_token_details", "cache_creation"),
    ("response_metadata", "usage", "cache_creation_input_tokens"),
)

#: Prompt assemblies a ledger keeps (CTX-1). A scored trial makes one per turn; a portal agent may live for hundreds.
MAX_CONTEXTS = 32


def read_cache_usage(response: Any) -> dict[str, int | None]:
    """``{"cache_read": n|None, "cache_write": n|None}`` -- the prompt tokens served from cache.

    Kept apart from :func:`read_usage` so its three-key contract cannot move. Same rules: never
    raises, and an absent count is ``None`` -- a provider that does not report caching has not
    reported *zero* cached tokens, and a mean over a run must be able to tell the difference.
    """

    def _walk(path: tuple[str, ...]) -> int | None:
        node: Any = response
        for i, key in enumerate(path):
            try:
                node = getattr(node, key, None) if i == 0 else (node.get(key) if isinstance(node, dict) else None)
            except Exception:
                return None
            if node is None:
                return None
        return _first_int({"v": node}, ("v",))

    got: dict[str, int | None] = {"cache_read": None, "cache_write": None}
    if response is None:
        return got
    for field_name, paths in (("cache_read", _CACHE_READ_PATHS), ("cache_write", _CACHE_WRITE_PATHS)):
        for path in paths:
            value = _walk(path)
            if value is not None:
                got[field_name] = value
                break
    return got


@dataclass
class Call:
    """One model call. ``node`` says which part of the loop spent it."""

    node: str
    input: int | None = None
    output: int | None = None
    total: int | None = None
    cache_read: int | None = None
    cache_write: int | None = None


@dataclass
class TurnUsage:
    """Every model call in one turn, and what can honestly be summed over them."""

    calls: list[Call] = field(default_factory=list)
    #: What the turns were GIVEN, beside what they cost (CTX-1): one entry per prompt assembly --
    #: the tools, know-how documents and prompt size the retriever chose. Until this, no trial
    #: record said which documents reached a scored prompt; the only way to check a claim about the
    #: prompt was to read a trajectory by hand. The ledger lives as long as the agent object (a
    #: whole trial, see ``sog_install/_agent_probe._usage_payload``), so entries append rather than
    #: overwrite, and a long-lived portal agent keeps only the newest ``MAX_CONTEXTS``.
    contexts: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def note_context(self, **facts: Any) -> None:
        """Record what one system-prompt assembly was built from, in the order assemblies happened."""
        with self._lock:
            self.contexts.append(dict(facts))
            del self.contexts[: max(0, len(self.contexts) - MAX_CONTEXTS)]

    def record(self, node: str, response: Any) -> Call:
        got = read_usage(response)
        call = Call(node=node, **got, **read_cache_usage(response))
        with self._lock:
            self.calls.append(call)
        return call

    def summary(self) -> dict[str, Any]:
        """Totals, plus how much of the turn they actually cover.

        ``measured_calls`` is the honesty field. A total of 12,000 tokens means something different
        over 9 calls of 9 than over 9 calls of 2, and a consumer that cannot tell them apart will
        compare a partial figure against a complete one and call the difference an improvement.
        """
        with self._lock:
            calls = list(self.calls)
        measured = [c for c in calls if c.total is not None]

        def _sum(attr: str) -> int | None:
            vals = [getattr(c, attr) for c in calls if getattr(c, attr) is not None]
            return sum(vals) if vals else None

        return {
            "calls": len(calls),
            "measured_calls": len(measured),
            "input": _sum("input"),
            "output": _sum("output"),
            "total": _sum("total"),
            "cache_read": _sum("cache_read"),
            "cache_write": _sum("cache_write"),
            "complete": bool(calls) and len(measured) == len(calls),
            "by_node": {
                node: sum(c.total for c in calls if c.node == node and c.total is not None) or None
                for node in sorted({c.node for c in calls})
            },
        }

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"summary": self.summary(), "calls": [asdict(c) for c in self.calls]}
        if self.contexts:  # absent, not [], until a prompt is assembled: older readers see the old shape
            out["context"] = [dict(c) for c in self.contexts]
        return out
