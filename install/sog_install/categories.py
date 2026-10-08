"""
Category catalog for Stage B — a thin wrapper over the agent's own SkillRegistry.

A "category" is exactly one of the nine skills the agent ships
(``SkillRegistry.create_default()``). The user multi-selects categories and we
expand each to the set of MCP **server keys** that back it, because provisioning
and wiring operate on servers (one interpreter / conda env per server) while a
skill only lists ``mcp_function`` names. The authoritative function→server map
is ``agent/MCP_server/mcp_config.yaml`` (``mcp_servers`` → ``tools`` →
``spatialomicsgym_name``); we never guess a server key from a function name.

Vocabulary note: the skill names are the user-facing menu. A *different*
six-value vocabulary lives in ``agent/data_validation.py`` ``TOOL_CATEGORY_MAP``
(used for data-routing, not menus); the two disagree on two names
(``spatial_communication``↔``cell_communication``,
``cell_segmentation``↔``image_segmentation``) and the skills add three
skill-only categories (``spatial_analysis``/``data_conversion``/``omics``).
:data:`CATEGORY_ALIASES` reconciles them for cross-reference, but the wizard
drives entirely off the skill names.

Design constraints: stdlib + pyyaml only. The ``skills`` import is lightweight
(no torch/scanpy/anndata are pulled — verified), but it is still done lazily so
a bare launcher env falls back to a static catalog instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import yaml

from . import constants
from .session_log import notice


class ToolOption(NamedTuple):
    """One selectable tool row for the picker.

    A ``tuple`` subclass so positional access is unchanged from the old bare
    ``(key, label, hint)`` triples: ``item[0]`` is still the parseable server
    key that drives selection, ``item[1]`` the display label, ``item[2]`` the
    legacy GPU hint. ``item[3]`` (``detail``) is new — a short description line —
    and every consumer reads it defensively (``item[3] if len(item) > 3``), so
    the plain 3-tuples used in tests keep working.
    """

    key: str  # server_key — the parseable id (item[0]; contract unchanged)
    label: str  # human display: full_name, or the key when no skill metadata
    hint: str = ""  # legacy "GPU"/"" flag (item[2]; contract unchanged)
    detail: str = ""  # one-line, pre-truncated description ("" for bare-env tools)


def _short(text: str, width: int = 72) -> str:
    """Collapse whitespace and truncate to ``width`` chars with an ellipsis.

    Truncation snaps back to the last word boundary so a word is never cut
    mid-token (``…graph neur…`` → ``…graph…``); a single over-long first word
    is hard-cut as a last resort.
    """
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    clipped = text[: width - 1]
    sp = clipped.rfind(" ")
    if sp >= width // 2:  # snap to the word boundary unless it would gut the text
        clipped = clipped[:sp]
    return clipped.rstrip() + "…"


# Skill name  ->  data_validation.TOOL_CATEGORY_MAP name (None = skill-only).
# Kept for anyone cross-referencing the routing vocabulary; the menu uses the
# skill names verbatim.
CATEGORY_ALIASES: dict[str, str | None] = {
    "spatial_clustering": "spatial_clustering",
    "deconvolution": "deconvolution",
    "svg_detection": "svg_detection",
    "spatial_alignment": "spatial_alignment",
    "spatial_communication": "cell_communication",
    "cell_segmentation": "image_segmentation",
    "spatial_analysis": None,
    "data_conversion": None,
    "omics": None,
}

# Used only when ``import skills`` fails (bare launcher env). Descriptions are
# short stand-ins; the live registry supplies the real ones when importable.
_FALLBACK_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("spatial_clustering", "Spatial domain identification and clustering."),
    ("deconvolution", "Cell-type deconvolution and spatial mapping."),
    ("svg_detection", "Spatially variable gene detection."),
    ("cell_segmentation", "Cell / nucleus segmentation from images."),
    ("spatial_alignment", "Alignment and integration of spatial slices."),
    ("spatial_communication", "Cell-cell communication in spatial context."),
    ("spatial_analysis", "General spatial analysis and neighborhood statistics."),
    ("data_conversion", "Format conversion and data preparation."),
    ("omics", "Literature / multi-omics and reference lookups."),
)


class CategoryError(RuntimeError):
    """The category catalog could not be built (config unreadable, etc.)."""


@dataclass
class Category:
    """One user-selectable skill and the servers it expands to."""

    name: str
    description: str
    functions: list[str] = field(default_factory=list)  # mcp_function names
    servers: list[str] = field(default_factory=list)  # MCP server keys (deduped, ordered)
    unmapped: list[str] = field(default_factory=list)  # functions with no server in the config
    # server_key -> {"gpu": bool, "priority": int, "functions": [fn, ...],
    #                "full_name": str, "description": str}
    server_meta: dict[str, dict] = field(default_factory=dict)

    @property
    def gpu(self) -> bool:
        return any(m.get("gpu") for m in self.server_meta.values())

    def to_ctx(self) -> dict:
        """Shape consumed by :class:`GuideContext.categories` / answers expansion."""
        return {"name": self.name, "description": self.description, "tools": list(self.servers)}

    def best_server(self) -> str | None:
        """Lowest ``priority`` number wins (priority 1 == preferred); ties keep order."""
        if not self.servers:
            return None
        return min(self.servers, key=lambda s: self.server_meta.get(s, {}).get("priority", 99))


# --------------------------------------------------------------------------- #
# Function -> server index (authoritative, from the shipped config)
# --------------------------------------------------------------------------- #
def function_server_index(config_path: str | Path | None = None) -> dict[str, str]:
    """Map every ``spatialomicsgym_name`` (mcp_function) to its owning server key.

    First occurrence wins if a function is somehow listed under two servers.
    """
    p = Path(config_path) if config_path else constants.original_mcp_config()
    if not p.exists():
        raise CategoryError(f"MCP config not found: {p}")
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        # `read_text` itself raises on a non-UTF-8 (latin-1/binary) config (UnicodeDecodeError) or an
        # unreadable path — perms, or the path being a directory (OSError/IsADirectoryError). Without
        # these arms the exception escapes `load_categories`' `except CategoryError` and crashes
        # interactive category selection at wizard `_cats = load_categories()` (unguarded). Raise
        # CategoryError so it degrades to the static catalog — byte-identical to the arm answers.py
        # already carries for the same file, and to how specs.load_all_specs tolerates a latin-1 spec.
        raise CategoryError(f"could not read {p}: {exc}") from exc
    if not isinstance(cfg, dict):
        # A top-level YAML list/scalar (hand-edited or truncated config). Degrade to the static catalog
        # via CategoryError — exactly like `mcp_servers` being non-dict below (D-F1) — instead of a bare
        # `cfg.get(...)` AttributeError (`list`/`str` has no `.get`) escaping to a generic exit 1.
        raise CategoryError(f"{p}: top level must be a mapping, got {type(cfg).__name__}")
    servers = cfg.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        # A hand-edited config with `mcp_servers:` written as a list/scalar. Raise CategoryError so
        # load_categories degrades to the static catalog (exactly as a bare launcher env does) instead
        # of an AttributeError (`list`/`str` has no `.items()`) escaping to a generic exit 1. (B)
        raise CategoryError(f"{p}: `mcp_servers` must be a mapping of server: {{...}}, got {type(servers).__name__}")
    index: dict[str, str] = {}
    for server_key, meta in servers.items():
        if not isinstance(meta, dict):
            continue  # a None/scalar server body (a common way to "disable" a server) — skip, don't crash on meta.get
        tools = meta.get("tools")
        # a scalar `tools: 5` (hand-edit slip) → [] here, not a `for tool in 5` TypeError (D-F3).
        for tool in tools if isinstance(tools, list) else []:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("spatialomicsgym_name")
            if fn and fn not in index:
                index[fn] = server_key
    return index


# --------------------------------------------------------------------------- #
# Build the catalog
# --------------------------------------------------------------------------- #
def _skill_iter():
    """Yield ``(name, description, mapping_values)`` per skill, or None if unavailable."""
    try:
        from skills.registry import SkillRegistry  # lazy: keep bare envs working

        reg = SkillRegistry.create_default()
        out = []
        for name in reg.list_skills():
            skill = reg.get_skill(name)
            mapping = skill.get_mcp_tool_mapping() or {}
            out.append((skill.name, skill.description or "", list(mapping.values())))
        return out
    except Exception:
        # Not just the import: ANY failure materializing the live skill catalog — create_default()
        # raising, a skill constructor blowing up, or a skill whose get_mcp_tool_mapping() returns a
        # non-dict (so `.values()` is an AttributeError) — must degrade to the static catalog exactly
        # as a bare launcher env does, not escape past load_categories' `except CategoryError` and
        # crash interactive category selection. The docstring promises "or None if unavailable";
        # load_categories turns None into _FALLBACK_CATEGORIES.
        return None


def _as_priority(val: object, default: int = 99) -> int:
    """Coerce a config ``priority`` to ``int``, tolerating a stray string/None in a hand-edited
    MCP config (a raw ``int(...)`` would raise ``ValueError`` and abort the whole menu)."""
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def load_categories(config_path: str | Path | None = None) -> list[Category]:
    """Return the nine categories, each expanded to its backing server keys."""
    try:
        index = function_server_index(config_path)
    except CategoryError as exc:
        # Missing/corrupt MCP catalog: degrade to the static category names (exactly as a bare
        # launcher env does) with a friendly notice, instead of crashing category selection.
        notice(f"could not read the MCP catalog ({exc}); showing categories without a server roster.")
        return [Category(name=n, description=d) for n, d in _FALLBACK_CATEGORIES]
    skills = _skill_iter()

    if skills is None:
        # Bare env: static names/descriptions, but still expand via the config
        # by intersecting on nothing — we have no function list, so servers stay
        # empty and the guide/menu offer categories without a pre-known roster.
        return [Category(name=n, description=d) for n, d in _FALLBACK_CATEGORIES]

    cats: list[Category] = []
    for name, desc, values in skills:
        functions: list[str] = []
        servers: list[str] = []
        unmapped: list[str] = []
        server_meta: dict[str, dict] = {}
        seen_srv: set[str] = set()
        for v in values:
            fn = v.get("mcp_function")
            if not fn:
                continue
            functions.append(fn)
            srv = index.get(fn)
            if srv is None:
                unmapped.append(fn)
                continue
            if srv not in seen_srv:
                seen_srv.add(srv)
                servers.append(srv)
                server_meta[srv] = {"gpu": False, "priority": 99, "functions": [], "full_name": "", "description": ""}
            m = server_meta[srv]
            m["functions"].append(fn)
            m["gpu"] = m["gpu"] or bool(v.get("gpu"))
            p = _as_priority(v.get("priority", 99))
            # Adopt the identity (name/description) of the preferred function:
            # lowest priority number wins, first-seen on a tie (matches best_server()).
            if p < m["priority"] or not m["full_name"]:
                m["full_name"] = v.get("full_name") or m["full_name"]
                m["description"] = v.get("description") or m["description"]
            m["priority"] = min(m["priority"], p)
        cats.append(
            Category(
                name=name,
                description=desc,
                functions=functions,
                servers=servers,
                unmapped=unmapped,
                server_meta=server_meta,
            )
        )
    return cats


def categories_as_ctx(cats: list[Category]) -> list[dict]:
    """List of ``{name, description, tools}`` dicts for the guide / answers context."""
    return [c.to_ctx() for c in cats]


def grouped_tool_options(
    cats: list[Category],
    all_server_keys: set[str],
) -> list[tuple[str, str, list[ToolOption]]]:
    """Group every installable server under a category header for the picker.

    Returns ``[(category_name, description, [ToolOption, ...]), ...]`` in
    canonical (skill) order. Each server appears **once** — a server listed by
    several categories is placed under the first one. Categories that contribute no
    server are dropped (e.g. an ``omics`` skill that maps to nothing installable).
    A trailing ``"Other"`` group collects any server in ``all_server_keys`` that no
    category claims, so the union of all group items is exactly ``all_server_keys`` —
    every tool is offered exactly once. The shipped catalog currently claims every
    server, so ``"Other"`` is normally absent; it catches a tool captured into
    ``install/recipes/tool_specs/`` before it gains a ``skills/*/mcp_tools.py`` entry.

    Each :class:`ToolOption` carries the parseable ``key`` (``item[0]``, unchanged),
    a human ``label`` (the skill's ``full_name``, or the key when no metadata), the
    legacy GPU ``hint``, and a short ``detail`` description. Coverage/placement logic
    is byte-for-byte the same as before — only the row *type* and label/detail
    *content* changed — so ``item[0]`` still drives selection.
    """
    placed: set[str] = set()
    groups: list[tuple[str, str, list[ToolOption]]] = []
    for cat in cats:
        items: list[ToolOption] = []
        for srv in cat.servers:
            # Skip a server the category claims but that has no captured spec (absent from
            # ``all_server_keys``) — e.g. a tool added to ``agent/MCP_server/mcp_config.yaml`` before its
            # ``install/recipes/tool_specs/<srv>.yaml`` is captured. Showing it would put a row in the picker that
            # ``provision`` then silently skips ("no spec — skipping"), and makes the group union a
            # SUPERSET of ``all_server_keys``, breaking the docstring's "offered exactly once" invariant.
            # Mirrors the ``leftover`` filter below (line uses the same ``all_server_keys`` membership). (R22)
            if srv in placed or srv not in all_server_keys:
                continue
            placed.add(srv)
            m = cat.server_meta.get(srv, {})
            items.append(
                ToolOption(
                    key=srv,
                    label=m.get("full_name") or srv,  # graceful fallback to the server key
                    hint="GPU" if m.get("gpu") else "",
                    detail=_short(m.get("description") or ""),
                )
            )
        if items:
            groups.append((cat.name, cat.description, items))
    leftover = sorted(k for k in all_server_keys if k not in placed)
    if leftover:
        groups.append(("Other", "uncategorized tools", [ToolOption(key=k, label=k) for k in leftover]))
    return groups


# --------------------------------------------------------------------------- #
# Deterministic "recommended starter set" (pure, stdlib-only, no LLM)
# --------------------------------------------------------------------------- #
# Curated per-category keyword triggers. Deliberately NOT ``name.split("_")`` —
# splitting on "spatial" would match nearly every goal to the spatial_* skills.
_GOAL_TRIGGERS: dict[str, tuple[str, ...]] = {
    "spatial_clustering": ("cluster", "domain", "niche"),
    "deconvolution": ("deconv", "cell type", "cell-type", "proportion", "composition"),
    "svg_detection": ("variable gene", "svg", "spatially variable", "marker gene"),
    "cell_segmentation": ("segment", "nucle", "cell boundar", "membrane"),
    "spatial_alignment": ("align", "integrat", "registrat", "reconstruct"),
    "spatial_communication": ("communicat", "ligand", "receptor", "interaction", "crosstalk"),
    "spatial_analysis": (
        "neighborhood",
        "neighbourhood",
        "co-occur",
        "autocorrel",
        "squidpy",
        "enrich",
        "pathway",
        "gene set",
        "gsea",
    ),
    "data_conversion": ("convert", "seurat", "h5ad", ".rds", "format"),
    "omics": (),
}


def _priority1(cat: Category) -> list[str]:
    """Preferred (priority == 1) servers of a category, in catalog order."""
    return [s for s in cat.servers if cat.server_meta.get(s, {}).get("priority", 99) == 1]


def recommended_servers(
    cats: list[Category],
    stated_goals: str = "",
    *,
    per_cat_cap: int = 3,
    max_total: int = 12,
) -> list[str]:
    """A small, buildable starter set of server keys to pre-select in the picker.

    Goal-first: if the user's ``stated_goals`` hits a category's keyword triggers,
    recommend that category's preferred (priority-1) tools. Otherwise fall back to a
    breadth floor — one preferred tool per non-empty category — so the result is
    always non-empty, small, and installable. Pure and deterministic (no LLM, no
    randomness); the LLM's own suggestion, when present, takes precedence upstream.
    """
    goal = " ".join((stated_goals or "").lower().split())
    installable = [c for c in cats if c.servers]
    matched = [c for c in installable if any(t in goal for t in _GOAL_TRIGGERS.get(c.name, ()))] if goal else []
    picks: list[str] = []
    if matched:  # focused: the preferred tools of the goal's categories
        for c in matched:
            picks.extend(_priority1(c)[:per_cat_cap] or ([c.best_server()] if c.best_server() else []))
    else:  # breadth floor: one preferred tool per category (~8 tools)
        # Priority 1 first, as the goal branch above does. best_server() takes the lowest number,
        # and the alignment family's priority-0 rows are read-only 3D diagnostics, so the floor
        # pre-selected 'spatial3d' -- a portal that aligns nothing -- as that family's only tool
        # (hunt 2026-09-30, u29b-skills-config-5). best_server() stays the answer for a category
        # with no priority-1 server.
        picks = [(_priority1(c)[:1] or [c.best_server()])[0] for c in installable if c.best_server()]
    out: list[str] = []
    seen: set[str] = set()
    for s in picks:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:max_total]


def find_category(cats: list[Category], name: str) -> Category | None:
    for c in cats:
        if c.name == name:
            return c
    return None


def servers_for(cats: list[Category], selected: list[str]) -> list[str]:
    """Union of server keys across the selected category names (ordered, deduped)."""
    out: list[str] = []
    seen: set[str] = set()
    for name in selected:
        cat = find_category(cats, name)
        if cat is None:
            continue
        for s in cat.servers:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out
