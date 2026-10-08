"""Autonomous multi-round investigation: analyse, review, follow up, conclude, cite.

A *research run* is an outer loop around the agent that already exists. It does not change how a
turn runs -- it decides how many turns to run, what each one should ask next, and what the whole
thing amounts to at the end. Everything in this package is headless and takes its runner by
argument, so the loop can be exercised to completion without an LLM, a network or a portal.

The three modules split along the only line that matters for testing:

* :mod:`~spatialomicsgym.research.citations` -- what the model claimed, and whether the literature
  agrees. Pure text in, verdicts out; the two lookups are injected.
* :mod:`~spatialomicsgym.research.prompts` -- the words. Round one is ours; round two onward is
  `next_step.build_followup_prompt`, which already names the tool output, the findings and the
  caveats, so this module composes rather than re-writes.
* :mod:`~spatialomicsgym.research.loop` -- the bounds, the state and the persistence.
* :mod:`~spatialomicsgym.research.report` -- the document, composed from the post-analysis
  renderer rather than a second one, plus the sentence saying why the run ended.
"""

from spatialomicsgym.research.citations import Citation, CitationLedger
from spatialomicsgym.research.loop import (
    MAX_ROUNDS_CEILING,
    MAX_ROUNDS_DEFAULT,
    MAX_SECONDS_CEILING,
    MAX_SECONDS_DEFAULT,
    max_rounds,
    max_seconds,
    quiet_followups,
    research_allowed,
    research_directory,
    run_research,
)
from spatialomicsgym.research.report import (
    REPORT_NAME,
    STOP_SENTENCES,
    render_research_report,
    stop_sentence,
    write_research_report,
)

__all__ = [
    "MAX_ROUNDS_CEILING",
    "MAX_ROUNDS_DEFAULT",
    "MAX_SECONDS_CEILING",
    "MAX_SECONDS_DEFAULT",
    "REPORT_NAME",
    "STOP_SENTENCES",
    "Citation",
    "CitationLedger",
    "max_rounds",
    "max_seconds",
    "quiet_followups",
    "render_research_report",
    "research_allowed",
    "research_directory",
    "run_research",
    "stop_sentence",
    "write_research_report",
]
