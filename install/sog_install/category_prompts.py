"""
One natural-language task prompt per category — the Tier-2 pipeline test.

Tier-2 exercises a whole ``STCoscientist(...).go(prompt)`` round-trip (retrieval →
tool selection → worker dispatch → answer) against a *provisioned* category, so
the prompt has to read like a request a real biologist would type. Deliberately:

* **no tool / MCP function names** — the agent's own retriever must pick the tool;
  naming it would test dispatch plumbing, not the pipeline.
* **no file paths or ``h5ad``/``AnnData`` jargon** — the dataset is handed to the
  agent via ``STCoscientist(path=…)``; the prompt describes the *goal*, not the file.
* **no code** — this is a user turn, not a script.

Keys are the nine :mod:`~sog_install.categories` skill names. Every
prompt is answerable (best-effort) against the local Visium-shaped
``mini_spatial.h5ad`` + ``mini_sc_ref.h5ad``; categories that really want a second
slice or a tissue image (``spatial_alignment``, ``cell_segmentation``) still pose a
sensible request and simply degrade to a graceful "not enough data" answer — Tier-2
is gated/optional, so that never fails the install.

Stdlib only.
"""

from __future__ import annotations

CATEGORY_PROMPTS: dict[str, str] = {
    "spatial_clustering": (
        "I have a Visium spatial transcriptomics slide. Can you find the spatial "
        "domains for me — group the spots into spatially coherent regions and tell "
        "me which region each spot belongs to?"
    ),
    "deconvolution": (
        "I have a Visium slide together with a matching single-cell reference. Each "
        "spot is a mixture of cell types — please estimate the cell-type composition "
        "of every spot."
    ),
    "svg_detection": (
        "I have a Visium slide. Which genes show a real spatial pattern — expression "
        "that varies across the tissue rather than being uniform? Please give me the "
        "top spatially variable genes."
    ),
    "cell_segmentation": (
        "I have imaging from a spatial experiment on this tissue. Can you segment the "
        "individual cells so I know where each cell sits?"
    ),
    "spatial_alignment": (
        "I have spatial transcriptomics sections from the same tissue. Can you align "
        "them so the coordinates live in one common space?"
    ),
    "spatial_communication": (
        "I have a spatial transcriptomics slide. Which regions are signalling to each "
        "other — can you find the ligand–receptor interactions that look active "
        "between neighbouring spots?"
    ),
    "spatial_analysis": (
        "I have a Visium slide. Can you give me a general spatial characterization — "
        "which genes are spatially structured and how the tissue neighbourhoods are "
        "organized?"
    ),
    "data_conversion": (
        "I have a spatial dataset and I'm not sure it's in the right shape to analyse. "
        "Can you check it over and get it into a standard, ready-to-use form?"
    ),
    "omics": (
        "I have a short list of marker genes coming out of my spatial data. Can you "
        "pull together what's known about them from the literature and reference "
        "databases?"
    ),
}

#: How each category reads as a heading, for the surfaces that show the nine as a menu rather
#: than sending their prompts (the welcome page's task cards, the skills catalog). Keys are
#: :data:`CATEGORY_PROMPTS`' keys exactly -- a test pins the two sets equal, because a label table
#: that drifts from the prompt table is how a card ends up offering a category nothing can run.
#: Derived spellings were tried first and rejected: ``svg_detection`` title-cases to "Svg
#: Detection", and no rule that fixes that one leaves "Omics" and "Data conversion" alone.
CATEGORY_TITLES: dict[str, str] = {
    "spatial_clustering": "Spatial domains",
    "deconvolution": "Cell-type deconvolution",
    "svg_detection": "Spatially variable genes",
    "cell_segmentation": "Cell segmentation",
    "spatial_alignment": "Alignment & integration",
    "spatial_communication": "Cell-cell communication",
    "spatial_analysis": "General spatial analysis",
    "data_conversion": "Data conversion & QC",
    "omics": "Literature & annotation",
}


# Used when a category has no bespoke prompt (should not happen for the nine skills,
# but keeps Tier-2 robust if the category vocabulary ever grows).
_GENERIC_PROMPT = (
    "I have a Visium spatial transcriptomics slide with a matching single-cell "
    "reference. Please run a standard analysis for this kind of data and summarize "
    "what you find."
)


def prompt_for(category: str) -> str:
    """The Tier-2 prompt for a category name (falls back to a generic request)."""
    return CATEGORY_PROMPTS.get(category, _GENERIC_PROMPT)


def title_for(category: str) -> str:
    """The heading for a category name, falling back to the name with its underscores opened."""
    return CATEGORY_TITLES.get(category) or category.replace("_", " ").strip() or category
