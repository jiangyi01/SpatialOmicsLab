"""Canonical task-type vocabulary: one place where the system's dialects meet.

The platform grew at least six task-type vocabularies -- the skills planner says
``spatial_communication``, the MCP tool descriptions say ``cell_communication``, the post-analysis
contract (:mod:`spatialomicsgym.postanalysis.manifest`) pins seven canonical names, and so on.
Nothing translated between them, so a caller passing the planner's word for a task into the
post-analysis engine sailed past every handler into the generic fallback.

This module does NOT rename anything. The contract's :data:`TASK_TYPES` stays the single source of
truth (re-exported here, never copied), every registry keeps its own strings, and
:func:`canonicalize` is the translation applied at the seams -- engine entry points, follow-up
lookup -- so each dialect keeps working while the handlers finally understand all of them.

Deliberately NOT aliased: ``resolution``, ``super_resolution`` and ``denoising``. The post-analysis
contract pins their partial-degrade behavior (a run of those tools gets a shallow scan, not a deep
analysis, and says so); mapping them onto a deep handler would silently promise an interpretation
the engine does not have for them.

Import cost: this module and its one import are stdlib-only, so every layer -- agent core, webui,
report rendering -- may import it without dragging in the heavy science stack.
"""

from __future__ import annotations

# The contract is the source of truth; re-export, never copy. (postanalysis/__init__ is lazy and
# manifest.py is stdlib-only, so this chain stays importable in the 1.6GB agent-core env.)
from spatialomicsgym.postanalysis.manifest import TASK_TYPES

__all__ = ["ALIASES", "TASK_TYPES", "WORKFLOW_ARC", "arc_stages", "canonicalize", "workflow_arc_sentence"]

#: Other registries' spellings of tasks the contract already handles. Keys are the foreign words,
#: values are contract words. Nothing maps here unless a deep handler genuinely exists for it.
ALIASES: dict[str, str] = {
    "spatial_communication": "cell_communication",  # skills planner / recommender vocabulary
    "spatial_alignment": "alignment",
    "spatial_imputation": "imputation",
    # Program 7's vocabulary. Both land on `alignment`, which is the contract's own word and
    # already has a handler; neither shadows a contract name.
    "three_d_reconstruction": "alignment",
    "slice_registration": "alignment",
}

#: The biological order of a spatial transcriptomics study, used by the workflow guide and the
#: prompt-context block. Stage tokens, not tool names; ``clustering`` here is the study stage whose
#: contract task type is ``spatial_clustering``.
WORKFLOW_ARC: tuple[str, ...] = (
    "qc",
    "clustering",
    "annotation",
    "deconvolution",
    "svg_detection",
    "cell_communication",
)

#: The order a MULTI-SLICE study runs in. Deliberately a separate constant rather than an extension
#: of :data:`WORKFLOW_ARC`, and the reason is mechanical rather than aesthetic: ``WORKFLOW_ARC`` is
#: rendered verbatim into ``know_how/spatial_workflow_guide.md``, which is one of the 22 documents
#: in the SCORED system prompt. Adding a stage there would move ``RECORDED_SCORED_DIGEST`` and open
#: a benchmark epoch for a change that is about live portal turns and has no scored task family to
#: serve. ``test/test_the_workflow_guide_is_loaded_and_stays_small.py`` pins that rendering, and
#: ``test/test_the_task_type_alias_table_speaks_the_contract_language.py`` pins the tuple itself.
#:
#: Diagnosis comes before alignment, and validation after it, because that ordering is the whole
#: claim of the protocol: a stack that is already aligned must not be aligned again, and an
#: alignment nobody measured is not evidence of anything.
MULTISLICE_ARC: tuple[str, ...] = (
    "qc",
    "diagnosis",
    "alignment",
    "validation",
    "clustering",
    "annotation",
)

#: How each arc stage reads in a sentence for a biologist.
_ARC_LABELS: dict[str, str] = {
    "qc": "quality control",
    "clustering": "spatial domain clustering",
    "annotation": "domain annotation",
    "deconvolution": "cell-type deconvolution",
    "svg_detection": "spatially variable gene detection",
    "cell_communication": "cell-cell communication",
}


def canonicalize(task_type: object) -> str:
    """Translate any registry's task-type spelling into the contract's, when a mapping exists.

    Folds case, surrounding whitespace, and ``-``/space separators, then applies :data:`ALIASES`.
    An unmapped value comes back cleaned but otherwise unchanged -- this function never guesses,
    so the contract's own degrade paths for unknown types keep firing exactly as pinned.
    Non-string input returns ``""`` (the same "no task type" the manifest uses).
    """
    if not isinstance(task_type, str):
        return ""
    cleaned = task_type.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(cleaned, cleaned)


def workflow_arc_sentence() -> str:
    """The arc as one plain-ASCII line, e.g. ``quality control -> ... -> cell-cell communication``."""
    return " -> ".join(_ARC_LABELS.get(stage, stage.replace("_", " ")) for stage in WORKFLOW_ARC)


def arc_stages() -> tuple[tuple[str, str], ...]:
    """The arc as ``(stage_token, biologist's wording)`` pairs, in the arc's own order.

    The same pairing :func:`workflow_arc_sentence` joins into a line, handed over whole so a
    surface that renders the stages as separate elements -- the welcome page's strip -- does not
    have to split a sentence back apart or keep a seventh copy of the labels. Reading
    :data:`_ARC_LABELS` through here is what keeps it private: a caller outside this module gets
    the pairs, never the dict it could edit.
    """
    return tuple((stage, _ARC_LABELS.get(stage, stage.replace("_", " "))) for stage in WORKFLOW_ARC)
