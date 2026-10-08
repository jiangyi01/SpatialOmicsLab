"""
Verified ``server_key → natural method name`` map for the Part-D demo.

When the demo runs a *specific* installed tool — the one the user picked from the Q2 menu, or the
lone tool that passed testing — it now **names that method** in the biologist-voice prompt, e.g.
"Please use GraphST for this analysis." That single sentence flips the agent into its USER-SPECIFIED
routing mode (``prompt_builder._detect_user_specified_tool``), so it dispatches to the *installed*
worker env instead of its catalog-ranked default.

Why this matters — the bug it fixes. The agent's recommender ranks a fixed knowledge base of ~113
tools, **installed-agnostic**. On a partial install its top pick is often a tool the user did *not*
build (classically CellCharter for spatial clustering). Handed a generic prompt, the agent then
improvises a raw in-process ``import GraphST`` / ``import STAGATE_pyG`` in the *base* env — where the
heavy per-tool packages deliberately do not live — and the user sees the confusing::

    - CellCharter failed: No module named cellcharter
    - GraphST fallback failed: No module named 'GraphST'
    - STAGATE fallback failed: No module named 'STAGATE_pyG'

…even though those tools are installed and healthy in their own ``<basic>_<tool>`` envs. Naming the
picked method routes around the recommender entirely (see ``demo._demo_plan``); the reassurance net
in ``demo._report_probe`` catches any residual quirk for tools that have no resolvable name here.

Every entry is machine-verified by ``test/test_setup/test_demo_tool_naming.py``: the natural name
resolves — via the agent's ``resolve_tool_name`` **exact/alias** tiers (score ≥ 1.0, *no* LLM call) —
to that tool's own MCP function, and ``_detect_user_specified_tool`` picks it out of a full sentence.
The map is therefore reproducible offline and immune to a stale/placeholder LLM key.

Regeneration (after an agent tool/alias change): the test is the guard — a renamed alias makes it
fail. To rebuild the map, for each spec bridge its ``functions[0].name`` (the MCP function) to
``transcriptomics_skills._MCP_TOOLS[*].full_name``, then keep only the names whose
``resolve_tool_name`` verdict is exact/alias at score ≥ 1.0 (that is exactly what the test asserts).
Tools with **no** resolvable name (e.g. ``sedr`` — no knowledge-base entry; ``scanpy_spatial`` — its
parenthesized full name resolves elsewhere, and it already has its own curated lymph-node demo) are
deliberately omitted: the demo falls back to the generic category prompt for them, and the
reassurance net still covers any routing quirk.

Stdlib only — this is static data. It must never import the agent/tool layer, so ``import
sog_install`` stays stdlib+pyyaml (the heavy agent lives only inside the probe subprocess).
"""

from __future__ import annotations

# server_key -> natural, biologist-voice method name. Test-guarded (see module docstring); every
# name is a deterministic exact/alias hit for that tool's own MCP function in the agent resolver.
DEMO_TOOL_NAMES: dict[str, str] = {
    "bayestme": "BayesTME",
    "card": "CARD",
    "cell2location": "Cell2Location",
    "cellcharter": "CellCharter",
    "celldart": "CellDART",
    "commot": "COMMOT",
    "deepst": "DeepST",
    "destvi": "DestVI",
    "graphst": "GraphST",
    "hotspot": "Hotspot",
    # NOT "MIST" — a different registered server (key "mist") is MIST/ReST. See the mistyR row
    # in transcriptomics_skills._MCP_TOOLS for why the shared name had to stop being ambiguous.
    "mistyr": "mistyR",
    "moscot": "MOSCOT",
    "paste": "PASTE",
    "prost": "PROST",
    "spaceflow": "SpaceFlow",
    "spagft": "SpaGFT",
    "spaotsc": "SPAOTSC",
    "spatial_miso": "MISO",
    "spatialde": "SpatialDE",
    "spatialscope": "SpatialScope",
    "spicemix": "SpiceMix",
    "spotgf": "SpotGF",
    "st_gears": "ST-GEARS",
    "stagate": "STAGATE",
    "stage": "STAGE",
    "starfysh": "Starfysh",
    "stlearn": "stLearn",
    "stride": "STRIDE",
    "svgbit": "SVGbit",
    "tacco": "TACCO",
    "tangram": "Tangram",
    "ucdeconvolve": "UCDeconvolve",
}


def display_name(server_key: str) -> str | None:
    """The verified natural method name for ``server_key``, or ``None`` when the tool has no name the
    agent resolves deterministically — the demo then uses the generic category prompt (and the
    reassurance net still covers any 'No module named' routing quirk)."""
    return DEMO_TOOL_NAMES.get(server_key)


def with_tool_named(text: str, name: str) -> str:
    """``text`` plus the one biologist-voice sentence that names the method to run.

    Shared by the demo (``demo._name_tool``) and the Tier-2 install check (``testing._tier2_prompt``)
    so the two ask for a tool in the same words. The "use <name>" phrasing is what the agent's
    ``_detect_user_specified_tool`` keys on; the pin only fires when the prompt also names an existing
    ABSOLUTE ``.h5ad`` path, which is why both callers now name the staged file by its full path."""
    return f"{text} Please use {name} for this analysis."


# Demo inputs beyond one expression slide (hunt 2026-09-30, u37-setup-checks-15). The demo stages a
# single slide, plus a matching single-cell reference only for deconvolution -- so a tool that maps a
# reference onto space was demoed with no reference at all, and the user's first look at a tool that
# passed Tier-1 was a failure. These ask for the reference too (each one's portal takes an sc input).
DEMO_NEEDS_SC_REF: frozenset[str] = frozenset({"spaotsc", "moscot", "stplus"})

# Categories whose task cannot be posed on the one slide the demo stages, with the reason the demo
# menu gives for leaving their tools out. Alignment registers two or more sections onto each other
# (paste_pairwise_align takes ``slice_h5ads``; ST-GEARS stacks a multi-section slide), and staging the
# same slide twice would "align" a section to itself and call it a result.
DEMO_UNSTAGEABLE_CATEGORIES: dict[str, str] = {
    "spatial_alignment": "aligns two or more tissue sections, and the bundled demo data is a single slide",
}
