"""Which know-how documents are already in the prompt, and which are only named there.

``configure()`` used to answer that question with one word -- *all of them*. Measured on this
checkout, "all of them" is **22 documents and 426,763 characters**, of which the seven-part
tool-creation playbook is **261,254** (61 %). Until that playbook was split it was a single
244,978-character retrieval unit, so one pick delivered the whole encyclopaedia; the largest
document is now 56,369 characters. That block is assembled once, at wire-up, and it is genuinely
the prompt for only four situations:

* turn zero, before any retrieval has happened;
* every turn with ``use_tool_retriever`` off -- **the eval path**;
* every turn where retrieval raised;
* every turn where the retrieval fail-safe fired.

On an ordinary portal turn it is overwritten before the first LLM call, because
``update_system_prompt_with_selected_resources`` *replaces* ``agent.system_prompt`` wholesale with
the retriever's picks. So this module does not change what a normal turn sees. It changes what the
*fallback* looks like, and the fallback is the eval path -- which is why the defaults below lean the
way they do.

Three modes:

``all``
    Today's list, byte-identical. **The default, and what a scored run always gets.**
``summaries``
    No document's text; every document's title and one-line description, under a header that says
    plainly that the full text is absent. ~4.1 KB of descriptions in place of ~427 KB of prose.
``demand``
    ``summaries`` plus the full text of the documents in :data:`ON_DEMAND_DOCUMENTS`.

Two properties are worth stating because their failure would be silent:

**Benchmarking wins, inside this function.** :func:`enrolment_mode` returns ``"all"`` whenever
``benchmarking_enabled`` is on, ignoring both the argument and the configured mode -- the same shape
as ``next_step.post_analysis_active()``. A misconfigured box cannot make a scored run see a smaller
prompt than the one the numbers were produced under. If the configuration cannot be read at all, the
answer is *also* ``"all"``: the failure direction is "behave as before", never "quietly enrol less".

**The two lists partition the corpus.** :func:`baseline_documents` returns what is present in full;
:func:`index_documents` returns exactly the rest, as titles and descriptions. No document appears in
both, and none falls out of both -- otherwise a document could vanish from the prompt with nothing
saying it exists.

One thing this module deliberately does *not* claim: that :data:`ON_DEMAND_DOCUMENTS` is derived
from code. Nothing in the package resolves a know-how document by id except the retriever resolving
its own selection, so the choice is prompt design, not a dependency. The criterion used is "a rule
that must hold on a turn where retrieval never ran", which is what ``post_task_analysis`` is: the
post-analysis review fires after tool output on every turn, and its protocol is written down only
here.
"""

from __future__ import annotations

from typing import Any

#: The vocabulary. Anything else -- a typo, an old value, ``None`` -- resolves to :data:`DEFAULT_MODE`
#: rather than to the smallest option, so a misspelling cannot silently strip the prompt.
BASELINE_MODES: tuple[str, ...] = ("all", "summaries", "demand")

DEFAULT_MODE = "all"

#: Documents whose full text ``demand`` keeps. See the module docstring for why this is a judgement
#: and not a lookup. Keep it short: every entry is paid for on every turn that falls back here.
ON_DEMAND_DOCUMENTS: tuple[str, ...] = ("post_task_analysis",)


def _benchmarking_now() -> bool:
    """Is a scored run in progress? Unreadable configuration answers *yes*, which forces ``all``."""
    try:
        from spatialomicsgym.config import default_config

        return bool(getattr(default_config, "benchmarking_enabled", False))
    except Exception:
        return True


def _configured_mode() -> str:
    try:
        from spatialomicsgym.config import default_config

        return str(getattr(default_config, "know_how_enrolment", DEFAULT_MODE) or DEFAULT_MODE)
    except Exception:
        return DEFAULT_MODE


def enrolment_mode(mode: str | None = None, *, benchmarking: bool | None = None) -> str:
    """Resolve the mode actually in force.

    Idempotent, so a caller may resolve once and pass the answer down to both list builders without
    the two disagreeing because the configuration changed between them.
    """
    if benchmarking is None:
        benchmarking = _benchmarking_now()
    if benchmarking:
        return DEFAULT_MODE
    text = str(mode if mode is not None else _configured_mode()).strip().lower()
    return text if text in BASELINE_MODES else DEFAULT_MODE


def _corpus(loader: Any) -> list[dict[str, Any]]:
    """The loader's documents, defensively -- a loader that failed to load must not crash wire-up."""
    documents = getattr(loader, "documents", None) or {}
    try:
        values = list(documents.values())
    except AttributeError:
        return []
    return [doc for doc in values if isinstance(doc, dict)]


def _full_text_ids(mode: str) -> set[str]:
    if mode == "all":
        return set()  # meaning "not a subset" -- callers special-case ``all`` before asking
    return set(ON_DEMAND_DOCUMENTS) if mode == "demand" else set()


def baseline_documents(
    loader: Any, mode: str | None = None, *, benchmarking: bool | None = None
) -> list[dict[str, Any]]:
    """The documents whose **full text** goes into the initial system prompt.

    Shaped exactly as ``configure()`` shaped them by hand: ``id``/``name``/``description``/
    ``content``/``metadata``, with ``content`` being the metadata-stripped body.
    """
    chosen = enrolment_mode(mode, benchmarking=benchmarking)
    keep = _full_text_ids(chosen)
    out: list[dict[str, Any]] = []
    for doc in _corpus(loader):
        doc_id = str(doc.get("id") or "")
        if chosen != "all" and doc_id not in keep:
            continue
        out.append(
            {
                "id": doc_id,
                "name": doc.get("name") or doc_id,
                "description": doc.get("description", ""),
                "content": doc.get("content_without_metadata") or doc.get("content") or "",
                "metadata": doc.get("metadata", {}),
            }
        )
    return out


def index_documents(loader: Any, mode: str | None = None, *, benchmarking: bool | None = None) -> list[dict[str, Any]]:
    """The documents that are **only named** -- title and one-line description, no body.

    Empty under ``all``, where every document is present in full and an index would be noise.
    """
    chosen = enrolment_mode(mode, benchmarking=benchmarking)
    if chosen == "all":
        return []
    keep = _full_text_ids(chosen)
    out: list[dict[str, Any]] = []
    for doc in _corpus(loader):
        doc_id = str(doc.get("id") or "")
        if doc_id in keep:
            continue
        out.append(
            {
                "id": doc_id,
                "name": doc.get("name") or doc_id,
                "description": str(doc.get("description") or "").strip(),
            }
        )
    return out


# ---------------------------------------------------------------------- tier 2: the merged packs
#
# The same fail-safe shape as ``enrolment_mode`` with the opposite default: that function answers
# "all" when it cannot read the configuration because the safe direction for tier 1 is "behave as
# before". For the packs the safe direction is "add nothing" -- a scored run must never see a pack,
# and an unreadable box must not quietly grow its prompt -- so every failure here answers False or 0.

#: The most pack documents one turn's second pass may add, whatever the configuration says.
MAX_PACK_BUDGET = 10

DEFAULT_PACK_BUDGET = 3


def packs_enabled(*, benchmarking: bool | None = None) -> bool:
    """Are the tier-2 packs in play for this turn?

    False under ``benchmarking_enabled`` (the scored prompt is measured without them), False when
    the configuration cannot be read, otherwise ``default_config.know_how_packs_enabled`` -- which
    defaults to False and is set by ``SOG_KNOW_HOW_PACKS``. Read at call time, not cached: a loader
    filled before benchmarking was switched on offers nothing to the scored turn that follows.
    """
    if benchmarking is None:
        benchmarking = _benchmarking_now()
    if benchmarking:
        return False
    try:
        from spatialomicsgym.config import default_config

        return bool(getattr(default_config, "know_how_packs_enabled", False))
    except Exception:
        return False


def pack_budget() -> int:
    """How many pack documents the second pass may add: ``know_how_pack_budget`` clamped into
    ``0..MAX_PACK_BUDGET``; 0 when the configuration cannot be read, and 0 disables the pass."""
    try:
        from spatialomicsgym.config import default_config

        raw = int(getattr(default_config, "know_how_pack_budget", DEFAULT_PACK_BUDGET))
    except Exception:
        return 0
    return max(0, min(MAX_PACK_BUDGET, raw))
