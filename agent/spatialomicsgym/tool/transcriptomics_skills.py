"""Unified transcriptomics analysis skills module.

Provides intelligent MCP tool selection, QC workflows, and analysis orchestration
for all transcriptomics data types:
  - Spatial: Visium, Xenium, MERFISH, CosMx, Slide-seq, Stereo-seq (STARmap/seqFISH pairs: convert_starmap, not detected)
  - Single-cell: 10x Chromium, Smart-seq2, Drop-seq, inDrop, CITE-seq
  - Bulk RNA-seq

Integrates with 30+ MCP spatial analysis tools to build automated, best-practice
workflows: data diagnosis → QC → preprocessing → tool selection → execution → evaluation.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from spatialomicsgym import layout
from spatialomicsgym.mcp_config_path import find_mcp_config
from spatialomicsgym.utils.file_io import read_h5ad_backed, sniff_tabular_sep, uncompressed_suffix

# ---------------------------------------------------------------------------
# Built-in spatial dataset discovery — Python re-export so STCoscientist's natural
# `from spatialomicsgym.tool.transcriptomics_skills import search_spatial_datasets`
# import works the same as the MCP tool of the same name. Backs onto the
# real registry at <repo>/spatial_library/registry.json -- the library moved into the
# checkout on 2026-10-06 from /workspace/spatial_library_backup (the legacy
# /workspace/data/spatial_library/registry.json path is missing on most installs).
# ---------------------------------------------------------------------------

# The repository root (``layout.repo_root()``), not this file's ``parents[2]``: the package sits under
# ``agent/`` since the re-layout, while ``spatial_library/`` stays at the root with the rest of the
# runtime data. ``None`` off a checkout, where only the absolute and in-package rungs apply.
_REPO = layout.repo_root()

_SPATIAL_REGISTRY_CANDIDATES = (
    *([str(_REPO / "spatial_library" / "registry.json")] if _REPO is not None else []),
    "/workspace/data/spatial_library/registry.json",
)

# The catalog snapshot that ships inside the package. Its MCP twin
# (``tools/spatial_library_worker.py``) has always resolved the registry this way; this re-export
# did not, so on every machine except the one that built the library ``_load_spatial_registry()``
# returned ``[]`` and ``search_spatial_datasets`` answered "registry not found" -- while the
# snapshot sat in the installed package the whole time.
_INREPO_SPATIAL_REGISTRY = str(Path(__file__).resolve().parents[1] / "data" / "spatial_library" / "registry.json")


def _spatial_registry_candidates() -> list[str]:
    """Registry locations to try, best first.

    ``SOG_SPATIAL_LIBRARY_REGISTRY`` first (the same override the MCP twin honours), so a
    deployment can point at its own catalog. The two absolute deployment paths keep their existing
    precedence -- the box that built the library has a richer registry there than the snapshot, and
    reordering would silently change which datasets that box sees. The in-repo snapshot is last:
    a fallback that makes the function work everywhere, changing nothing where the others exist.
    """
    override = os.environ.get("SOG_SPATIAL_LIBRARY_REGISTRY", "").strip()
    return ([override] if override else []) + [*_SPATIAL_REGISTRY_CANDIDATES, _INREPO_SPATIAL_REGISTRY]


# The basenames this repo standardises every dataset to. ``benchmarks/`` names
# ``spatial_transcriptomics.h5ad`` in 54 places and never globs for it, so preferring it is the
# existing convention rather than a new rule.
_CANONICAL_H5AD_NAMES = ("spatial_transcriptomics.h5ad", "single_cell_transcriptomics.h5ad")


def _pick_dataset_h5ad(spatial_dir: str) -> tuple[str | None, list[str]]:
    """Resolve a dataset directory to one ``.h5ad``, returning ``(chosen, all_candidates)``.

    This used to be ``glob.glob(dir + "/*.h5ad")[0]``. ``glob`` does not sort -- it yields
    ``os.scandir`` order, which on ext4 is hash order -- so the answer was machine-specific the
    moment a directory held more than one file. The shipped benchmark tree holds five:
    ``spatial_transcriptomics.h5ad`` plus four byte-identical ``_repaired`` / ``_mcp`` /
    ``_pipeline`` / ``_gpt54_pipeline_fixed`` copies that differ from the canonical file. A run on
    another machine would have analysed a derived artifact under the canonical dataset's name,
    with nothing in the output saying so.

    Hence: prefer a canonical basename, fall back to sorted order, and hand back every candidate
    considered so the caller can report what it actually opened. Recursion stays a last resort --
    a file sitting at the top level is the dataset; one buried deeper is a guess.
    """
    import glob as _glob

    for pattern, recursive in (
        (os.path.join(spatial_dir, "*.h5ad"), False),
        (os.path.join(spatial_dir, "**", "*.h5ad"), True),
    ):
        found = sorted(_glob.glob(pattern, recursive=recursive))
        if not found:
            continue
        by_name = {os.path.basename(p): p for p in found}
        for canonical in _CANONICAL_H5AD_NAMES:
            if canonical in by_name:
                return by_name[canonical], found
        return found[0], found
    return None, []


def _relocate_under_library_root(path: str, root: str) -> str:
    """A catalogued path found under the local library root by its trailing components, else *path*.

    The same rule as ``tools/spatial_library_worker._relocate`` (which this package cannot import:
    it lives beside the portals): the most specific suffix first, and the original path back when
    nothing matches, because a wrong guess is worse than an honest "not here".
    """
    if not path or not root or os.path.exists(path):
        return path
    parts = [p for p in path.split(os.sep) if p]
    for k in range(len(parts), 0, -1):
        candidate = os.path.join(root, *parts[-k:])
        if os.path.exists(candidate):
            return candidate
    return path


def _load_spatial_registry() -> list[dict]:
    for p in _spatial_registry_candidates():
        if os.path.exists(p):
            try:
                # encoding pinned: the catalog carries tissue/donor/author metadata, and a box whose
                # locale is C would otherwise decode it as ASCII and drop the whole registry here.
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict) and "datasets" in data:
                    return list(data["datasets"])
            except Exception:
                continue
    return []


def _record_health_flag(rec: dict) -> Any:
    """A record's own healthy/diseased flag, or ``None`` when it does not carry one.

    The MCP twin (``tools/spatial_library_worker.py``) filters on the *top-level* ``is_healthy``.
    This re-export read ``rec["disease_status"]["is_healthy"]`` instead -- a shape the shipped
    catalog does not use at all -- and because ``bool({}.get("is_healthy"))`` is ``False`` rather
    than ``None``, all 63 catalogued samples read as diseased: ``is_healthy=True`` returned
    nothing and ``is_healthy=False`` returned the whole library, the 31 healthy ones included.

    So: top level first, which is both what the catalog carries and what the twin reads, with the
    nested form kept as a fallback because a deployment registry may still be written that way.
    "Neither field present" stays ``None`` and is returned for neither query -- a sample nobody
    labelled is not evidence of disease, and the caller asked a question the record cannot answer.

    The comparison the caller makes is ``==`` against the requested bool, exactly the twin's, so a
    non-boolean value (a ``"true"`` string, say) matches neither query in either copy rather than
    being coerced here into an answer the MCP path would not give.
    """
    top = rec.get("is_healthy")
    if top is not None:
        return top
    nested = rec.get("disease_status")
    return nested.get("is_healthy") if isinstance(nested, dict) else None


# Which registry keys spell each searchable concept.
#
# The MCP twin (``tools/spatial_library_worker.py``) matches a *pair* of keys for each of these;
# this re-export matched one of each, so the two answered the same question differently. It shows
# up worst on ``organ``: the shipped catalog's ``organ`` is a compressed token (``Intestine``,
# ``SpinalCord``, ``Lymph_Node``, ``Cervical``) and its ``anatomical_entity`` the human-readable
# name, and they disagree on 15 of the 63 records. Asking the Python API for "Small Intestine"
# returned nothing where the MCP tool returned eight samples. ``organism``/``species`` hold the
# same value on every shipped record today and are paired here so they cannot drift apart later.
_SEARCH_FIELD_ALIASES = {
    "organism": ("organism", "species"),
    "organ": ("organ", "anatomical_entity"),
    "disease": ("disease", "description"),
}

# The word rules the MCP twin applies since u30-uncovered-mcp-2/-3/-8, kept in step with it so the
# two answer alike (hunt 2026-09-30); test/test_the_python_dataset_search_answers_like_the_mcp_worker.py
# runs both on one registry. The species are the builder's SPECIES_MAP
# (tools/build_spatial_library_registry.py), which the worker imports and this package cannot.
#
# * organism: the catalog spells every organism "Homo sapiens", so organism="human" -- the first
#   example the tool gives -- matched none of its records. Any spelling matches any other.
# * disease: a class word ("cancer", "tumour") matches a diseased record that names a tumour type,
#   because the catalog labels tumours by type ("Glioblastoma", "Invasive Ductal Carcinoma") and
#   disease="cancer" found none of the four brain tumours. Other words must each appear.
# * keyword: every word must appear somewhere in the record, not the whole phrase in one place
#   ("human glioblastoma" found 1 of the 4 glioblastomas).
_LIBRARY_SPECIES = {
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "rat": "Rattus norvegicus",
    "zebrafish": "Danio rerio",
    "drosophila": "Drosophila melanogaster",
}
_ORGANISM_SPELLINGS: dict[str, tuple[str, ...]] = {}
for _common, _binomial in _LIBRARY_SPECIES.items():
    _spellings = (_common, _binomial.lower(), f"{_binomial[0]}. {_binomial.split()[-1]}".lower())
    for _spelling in _spellings:
        _ORGANISM_SPELLINGS[_spelling] = _spellings
_CANCER_WORDS = frozenset(
    {"cancer", "cancers", "tumor", "tumors", "tumour", "tumours", "malignant", "malignancy", "neoplasm"}
)
_CANCER_LABELS = (
    "cancer",
    "carcinoma",
    "glioblastoma",
    "glioma",
    "melanoma",
    "sarcoma",
    "lymphoma",
    "leukemia",
    "leukaemia",
    "myeloma",
    "blastoma",
    "tumor",
    "tumour",
    "neoplasm",
    "metasta",
)


def _search_words(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,;/]+", (text or "").lower()) if t]


def _sample_id_as_words(rec: dict) -> str:
    """``CytAssist_FFPE_Human_Glioblastoma`` read as words, as the worker reads it."""
    return str(rec.get("sample_id") or "").replace("_", " ")


def search_spatial_datasets(
    organism: str = "",
    organ: str = "",
    disease: str = "",
    technology: str = "",
    keyword: str = "",
    is_healthy: bool | None = None,
    max_results: int = 5,
) -> dict:
    """Search the built-in spatial transcriptomics dataset library.

    Use this when a user asks about spatial analysis but has NOT provided
    any data file path. Returns matching datasets with h5ad paths ready
    for direct use with any spatial analysis MCP tool.

    Parameters mirror the MCP tool of the same name, and so do the rules:
    ``organism`` takes the common name or the binomial ("human", "Homo sapiens");
    every word of ``disease`` and of ``keyword`` must appear in the record, and a
    class word such as "cancer" matches a diseased record that names a tumour
    type; ``organ`` and ``technology`` are case-insensitive substring matches.
    Pass empty strings to disable a given filter. Returns a dict with keys
    `count`, `total_matched`, `results` (list of dataset records with `h5ad_path`).
    """
    reg = _load_spatial_registry()
    if not reg:
        return {"count": 0, "total_matched": 0, "results": [], "error": "registry not found"}

    def _matches(rec: dict) -> bool:
        def _ci_in(needle: str, hay: Any) -> bool:
            if not needle:
                return True
            return needle.lower() in str(hay or "").lower()

        def _ci_any(needle: str, keys: tuple[str, ...]) -> bool:
            """True when any of the concept's keys carries the needle -- the twin's own rule."""
            return not needle or any(_ci_in(needle, rec.get(k)) for k in keys)

        if organism:
            asked = organism.lower().strip()
            held = " ".join(str(rec.get(k) or "") for k in _SEARCH_FIELD_ALIASES["organism"]).lower()
            if not any(w in held for w in _ORGANISM_SPELLINGS.get(asked, (asked,))):
                return False
        if not _ci_any(organ, _SEARCH_FIELD_ALIASES["organ"]):
            return False
        if disease:
            d2 = rec.get("disease_status")
            # disease_status may be a {label: ...} dict OR a bare string; read whichever form it takes.
            label = d2.get("label") if isinstance(d2, dict) else d2
            text = " ".join(
                [str(rec.get(k) or "") for k in (*_SEARCH_FIELD_ALIASES["disease"], "sample_id")]
                + [_sample_id_as_words(rec), str(label or "")]
            ).lower()
            for word in _search_words(disease):
                if word in _CANCER_WORDS:
                    if _record_health_flag(rec) is not False or not any(t in text for t in _CANCER_LABELS):
                        return False
                elif word not in text:
                    return False
        if not _ci_in(technology, rec.get("technology")):
            return False
        if is_healthy is not None and _record_health_flag(rec) != is_healthy:
            return False
        if keyword:
            blob = " ".join(
                [str(v) for v in rec.values()]
                + [_sample_id_as_words(rec), *_ORGANISM_SPELLINGS.get(str(rec.get("organism") or "").lower(), ())]
            ).lower()
            if not all(w in blob for w in _search_words(keyword)):
                return False
        return True

    hits = [r for r in reg if _matches(r)]
    if keyword:
        # The worker's order: records whose organ or disease carries more of the words come first.
        words = _search_words(keyword)
        hits.sort(key=lambda r: -sum(w in f"{r.get('organ', '')} {r.get('disease', '')}".lower() for w in words))
    total = len(hits)
    out_results = []
    # The twin relocates catalogued paths under SOG_SPATIAL_LIBRARY_ROOT; this copy did not, so on
    # a box whose library lives elsewhere every hit came back with h5ad_path None while the MCP tool
    # found the files (hunt 2026-09-30, u23-transcriptomics-skills-26).
    library_root = os.environ.get("SOG_SPATIAL_LIBRARY_ROOT", "").strip()
    for rec in hits[:max_results]:
        h5ad = _relocate_under_library_root(rec.get("h5ad_path") or rec.get("path") or "", library_root) or None
        spatial_dir = _relocate_under_library_root(rec.get("spatial_dir") or "", library_root) or None
        if spatial_dir and spatial_dir != rec.get("spatial_dir"):
            rec = {**rec, "spatial_dir": spatial_dir}
        # The shipped registry leaves h5ad_path empty and populates spatial_dir; if that dir exists
        # locally, locate the .h5ad inside it so the returned h5ad_path is directly usable (spatial_dir
        # itself is still surfaced via **rec for the machine-specific / not-present case).
        if not h5ad and spatial_dir and os.path.isdir(spatial_dir):
            h5ad, candidates = _pick_dataset_h5ad(spatial_dir)
            # Only worth saying when there was actually a choice to make; on the common
            # one-file-per-directory layout this key would be pure noise.
            if len(candidates) > 1:
                rec = {**rec, "h5ad_candidates": candidates}
        out_results.append({**rec, "h5ad_path": h5ad})
    return {
        "count": len(out_results),
        "total_matched": total,
        "results": out_results,
    }


# ---------------------------------------------------------------------------
# Registered MCP tools cache (built at import time from mcp_config.yaml)
# ---------------------------------------------------------------------------


def _load_registered_mcp_tools() -> set[str]:
    """Parse MCP_server/mcp_config.yaml and return the set of registered MCP tool names.

    The mcp_config.yaml schema is:
        mcp_servers:
          <server_name>:
            tools:
              - spatialomicsgym_name: <tool_name>
                ...

    Returns the union of all `spatialomicsgym_name` values across all servers.
    """
    import yaml

    # The package-relative MCP_server/ on a clone; the SOG_MCP_CONFIG pointer or the working
    # directory on an install, where MCP_server/ is not in the wheel at all.
    config_path = find_mcp_config()

    if config_path is None:
        # Graceful degradation: empty cache (catalog integrity test will catch issues elsewhere)
        return set()

    # Guarded like the sibling below: this runs at import time, and since find_mcp_config()
    # the parsed file can be whatever SOG_MCP_CONFIG or the cwd supplies -- a hand-edited
    # config with a syntax error or a bare `server:` line must degrade to an empty cache,
    # never make the agent unimportable.
    try:
        with open(config_path) as fh:
            config = yaml.safe_load(fh) or {}
    except Exception:
        return set()
    if not isinstance(config, dict):
        return set()

    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):  # null, a scalar, or a list -- same empty degrade
        return set()
    tools: set[str] = set()
    for server_cfg in servers.values():
        if not isinstance(server_cfg, dict):
            continue
        entries = server_cfg.get("tools")
        for tool_entry in entries if isinstance(entries, list) else []:
            if not isinstance(tool_entry, dict):
                continue
            name = tool_entry.get("spatialomicsgym_name")
            if name:
                tools.add(name)
    return tools


def _load_declared_tool_params() -> dict[str, dict[str, dict]]:
    """Parse the same config and return {tool name: {parameter name: its spec}}, in config order.

    Needed because a parameter name is not guessable. Each portal picked its own convention for the
    input path -- ``st_h5ad`` (21 tools), ``h5ad_path`` (15), ``spatial_h5ad_path`` (12),
    ``data_path``, ``adata_path``, ``counts_h5ad`` -- and a name the portal has no parameter for is
    discarded by FastMCP without an error, so a wrong guess produces a call that "succeeds" with the
    input missing. See test/test_config_declares_only_real_parameters.py.
    """
    import yaml

    config_path = find_mcp_config()
    if config_path is None:
        return {}  # same graceful degradation as _load_registered_mcp_tools
    try:
        with open(config_path) as fh:
            config = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    if not isinstance(config, dict):
        return {}

    servers = config.get("mcp_servers")
    if not isinstance(servers, dict):
        return {}
    out: dict[str, dict[str, dict]] = {}
    for server_cfg in servers.values():
        if not isinstance(server_cfg, dict):
            continue
        entries = server_cfg.get("tools")
        for tool_entry in entries if isinstance(entries, list) else []:
            if not isinstance(tool_entry, dict):
                continue
            name = tool_entry.get("spatialomicsgym_name")
            params = tool_entry.get("parameters")
            if not name:
                continue
            out[name] = (
                {k: (v if isinstance(v, dict) else {}) for k, v in params.items()} if isinstance(params, dict) else {}
            )
    return out


_REGISTERED_MCP_TOOLS: set[str] = _load_registered_mcp_tools()


def _benchmark_hidden_mcp_tools() -> set[str]:
    """Tool names a scored run must not be offered, resolved at CALL time.

    A server block may declare ``benchmark_visible: false``. :func:`add_mcp` already honours it and
    binds no callable for such a server, so a scored run cannot execute one. This is the other half:
    the catalog reads the config directly, so without it the catalog would list eleven tools the
    same run has no way to call -- and it would list them in the very reply that presents itself as
    the set of names that would have worked.

    Call time, not import time, for the reason :data:`_REGISTERED_MCP_TOOLS` is import time: the set
    of registered names does not depend on the run, and whether a run is scored does. Reading
    ``benchmarking_enabled`` at import would freeze whichever value happened to hold when this
    module was first imported, which for the portal is before the first turn exists.

    Never raises. An unreadable config hides nothing, which is the direction that leaves behaviour
    identical to what it was before this function existed.
    """
    try:
        from spatialomicsgym.config import default_config

        if not bool(getattr(default_config, "benchmarking_enabled", False)):
            return set()
    except Exception:
        return set()

    import yaml

    try:
        config_path = find_mcp_config()
        if config_path is None:
            return set()
        with open(config_path) as fh:
            config = yaml.safe_load(fh) or {}
        servers = config.get("mcp_servers")
        if not isinstance(servers, dict):
            return set()
        hidden: set[str] = set()
        for server_cfg in servers.values():
            if not isinstance(server_cfg, dict) or server_cfg.get("benchmark_visible") is not False:
                continue
            entries = server_cfg.get("tools")
            for tool_entry in entries if isinstance(entries, list) else []:
                if isinstance(tool_entry, dict) and tool_entry.get("spatialomicsgym_name"):
                    hidden.add(tool_entry["spatialomicsgym_name"])
        return hidden
    except Exception:
        return set()


def _visible_mcp_tools() -> set[str]:
    """The registered names this run may actually be offered. See the function above."""
    return (_REGISTERED_MCP_TOOLS | _user_mcp_tools()) - _benchmark_hidden_mcp_tools()


_USER_TOOLS_CACHE: dict[str, Any] = {"key": None, "names": frozenset()}


def _user_mcp_tools() -> frozenset[str]:
    """Callable names of the user-created servers ``add_mcp`` serves on top of the shipped config.

    :data:`_REGISTERED_MCP_TOOLS` is the shipped config read once at import, so a tool created with
    tool creation on -- wired by ``add_mcp`` from ``mcp_config_user.yaml`` -- was callable and yet
    absent from the listing that calls itself every tool this agent can call, and not resolvable by
    name (hunt 2026-09-30, u23-transcriptomics-skills-24). Read at call time, cached on the two
    files' mtimes, through the predicate the merger itself uses to decide what it serves.

    Empty when tool creation is off, under benchmarking, or on any failure to read: the direction
    that leaves every answer as it was before this existed.
    """
    try:
        from spatialomicsgym.config import default_config

        if not getattr(default_config, "tool_creation_enabled", False) or getattr(
            default_config, "benchmarking_enabled", False
        ):
            return frozenset()
        from spatialomicsgym.mcp_user_config import (
            disabled_user_servers,
            resolve_user_config_path,
            shipped_identity,
            user_function_names,
            user_server_skip_reason,
        )

        user_path = resolve_user_config_path()
        base_path = find_mcp_config()
        if not os.path.isfile(user_path):
            return frozenset()
        key = (
            user_path,
            os.path.getmtime(user_path),
            str(base_path),
            os.path.getmtime(base_path) if base_path else 0.0,
        )
        if _USER_TOOLS_CACHE["key"] == key:
            return _USER_TOOLS_CACHE["names"]

        import yaml

        with open(user_path, encoding="utf-8") as fh:
            user = yaml.safe_load(fh) or {}
        base: Any = {}
        if base_path:
            with open(base_path, encoding="utf-8") as fh:
                base = yaml.safe_load(fh) or {}
        user_servers = user.get("mcp_servers") if isinstance(user, dict) else None
        base_servers = base.get("mcp_servers") if isinstance(base, dict) else None
        original_names, original_functions = shipped_identity(base_servers if isinstance(base_servers, dict) else {})
        turned_off = disabled_user_servers(base)
        claimed: set[str] = set()
        for name, meta in (user_servers if isinstance(user_servers, dict) else {}).items():
            if name in turned_off:
                continue
            if user_server_skip_reason(name, meta, original_names, original_functions, claimed) is None:
                claimed |= user_function_names(meta)
        names = frozenset(claimed)
        _USER_TOOLS_CACHE.update(key=key, names=names)
        return names
    except Exception:
        return frozenset()


def _callable_mcp_tools() -> set[str]:
    """Shipped plus user-created names: what ``resolve_tool_name`` may resolve to."""
    return _REGISTERED_MCP_TOOLS | _user_mcp_tools()


_DECLARED_TOOL_PARAMS: dict[str, dict[str, dict]] = _load_declared_tool_params()

_SKILLS_ROWS_BY_FUNCTION: dict[str, dict] | None = None


def _skills_rows_by_function() -> dict[str, dict]:
    """{mcp_function: its row} from ``skills/*/mcp_tools.py`` -- the catalog the retriever shows.

    Covers the whole registry, where the curated ``_MCP_TOOLS`` table covers a subset, so it is what
    lets a listing name a registered tool nobody has written a profile for. Read lazily and cached:
    ``skills`` is a sibling top-level package, and importing it at module scope would make this
    module unimportable in an environment that ships only ``spatialomicsgym``. An environment
    without it degrades to names alone, never to a shorter list.
    """
    global _SKILLS_ROWS_BY_FUNCTION
    if _SKILLS_ROWS_BY_FUNCTION is None:
        rows: dict[str, dict] = {}
        try:
            from skills.registry import SkillRegistry  # lazy: keep bare envs working

            for row in SkillRegistry.create_default().get_all_tools().values():
                name = row.get("mcp_function")
                if name:
                    rows.setdefault(name, row)
        except Exception:
            rows = {}
        _SKILLS_ROWS_BY_FUNCTION = rows
    return _SKILLS_ROWS_BY_FUNCTION


# Preference order within whatever a tool declares. Only matters when a tool declares more than one
# of these (e.g. spaceflow declares h5ad_path AND counts_h5ad_path); the first match wins.
_SLIDE_PARAM_NAMES = (
    "st_h5ad",
    "h5ad_path",
    "spatial_h5ad_path",
    "spatial_h5ad",
    "data_path",
    "adata_path",
    "counts_h5ad",
    "counts_h5ad_path",
    "input_path",
    # run_spacel_scube's spatial_h5ad_paths is a COMMA-SEPARATED str, so one path is a valid value
    # for it. The genuinely list-valued plurals are in _SLIDE_LIST_PARAM_NAMES.
    "spatial_h5ad_paths",
)
_SLIDE_LIST_PARAM_NAMES = ("h5ad_paths", "slice_h5ads")
_REFERENCE_PARAM_NAMES = ("sc_h5ad_path", "sc_h5ad", "scrna_h5ad", "ref_h5ad", "adata_sc_path")
# Nearly every portal calls it output_dir; two spell it differently, and each of those two names is
# declared by exactly one tool, both meaning "directory to write results into".
_OUTPUT_DIR_PARAM_NAMES = ("output_dir", "results_dir", "save_path")


def _declared_input_param(mcp_function: str, preference: tuple[str, ...]) -> str | None:
    """The first name in *preference* that *mcp_function* actually declares, else None.

    None means "this tool takes no such input under any name we recognise" -- for a CSV-input tool
    that is the truth, and inventing an h5ad parameter for it would only be discarded.
    """
    declared = _DECLARED_TOOL_PARAMS.get(mcp_function)
    if not declared:
        return None
    return next((p for p in preference if p in declared), None)


# ---------------------------------------------------------------------------
# Tool-name resolver
# ---------------------------------------------------------------------------

RESOLVE_CONFIDENCE_THRESHOLD = 0.7
"""Below this score, callers should confirm with the user before invoking."""

# Note on ToolRetriever lifecycle: a fresh ToolRetriever instance is created on
# every Tier-3 call (not cached at module level). The current ToolRetriever
# constructor is cheap (no embedding index loaded at init time), so this is
# essentially free. Lazy construction at first Tier-3 use also keeps module
# import cheap and lets tests monkeypatch the class before any instance exists.


def _normalize_name(s: str) -> str:
    """Lowercase, strip whitespace, drop hyphens/underscores."""
    return s.strip().lower().replace("-", "").replace("_", "")


# A bare server name is an alias of that server's main function, which is right whenever the request
# is that function's analysis. These servers also register a function for a different analysis, and
# the alias pinned the main one whatever was asked: "Use GraphST to deconvolve the spots" pinned the
# clustering function with "Do NOT substitute a different tool" while graphst_deconvolution sat
# unused (hunt 2026-09-30, u23-transcriptomics-skills-2). Dropping the alias, as was done for
# "mist", would also unpin the common request it gets right ("Use GraphST to identify spatial
# domains"), so the request's own words choose instead.
#
# {bare name: (words naming the main function's analysis, ((sibling, words naming its analysis), ...))}.
# The first analysis named after the server's name decides (failing that, the last one before it);
# context that names none keeps the main function -- which is also every result before this existed.
_SERVER_ALIAS_SIBLINGS: dict[str, tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]] = {
    "graphst": (
        ("domain", "cluster", "region", "niche"),
        (("graphst_deconvolution", ("deconvol", "cell-type proportion", "cell type proportion", "composition")),),
    ),
    "prost": (
        ("variable gene", "svg", "spatially variable", "prost index"),
        (("prost_pnn_domains", ("domain", "cluster", "region", "niche")),),
    ),
    "spatialprompt": (
        ("domain", "cluster", "region", "niche"),
        (("spatialprompt_deconvolution", ("deconvol", "cell-type proportion", "cell type proportion", "composition")),),
    ),
    "spacel": (
        ("domain", "cluster", "region", "niche"),
        (("run_spacel_scube", ("align", "registration", "register", "3d", "stack")),),
    ),
    # "cluster" is not a main-function word here: "find the markers of each cluster" is a marker request.
    "seurat": (
        ("pipeline", "qc", "quality control", "normaliz", "domain"),
        (
            ("seurat_find_markers", ("marker",)),
            ("seurat_spatial_variable_features", ("variable feature", "variable gene", "spatially variable")),
        ),
    ),
}


def _sibling_for_context(normalized_query: str, canonical: str, context: str) -> str:
    """The function of *canonical*'s server that *context* asks for, else *canonical* itself."""
    entry = _SERVER_ALIAS_SIBLINGS.get(normalized_query)
    if not entry or not context:
        return canonical
    own_words, siblings = entry
    # Paths and file names are not the request: a slide stored as deconvolution_pilot/slide.h5ad
    # must not turn "Use GraphST on <it>" into a deconvolution.
    text = re.sub(r"\S*[/\\]\S*|\S+\.[a-z0-9]{2,6}\b", " ", context.lower())
    # Which analysis, by where it is named rather than whether: "Use GraphST to deconvolve the spots
    # in the tumour region" asks for a deconvolution, and its "region" is anatomy. Any main-function
    # word anywhere used to keep the main pin, so that sentence still pinned the clustering function
    # and "Run SPACEL to align the serial sections of the hippocampal region" the clustering one
    # (hunt 2026-09-30, rp-u23 review of u23-transcriptomics-skills-2).
    name = re.search(r"[-_\s]?".join(map(re.escape, normalized_query)), text)
    start, end = (name.start(), name.end()) if name else (0, 0)
    options = [(canonical, own_words)]
    options += [(fn, words) for fn, words in siblings if fn in _REGISTERED_MCP_TOOLS]
    # (position, option index) of each option's nearest word; the main function wins a tie.
    after, before = [], []
    for i, (_fn, words) in enumerate(options):
        later = [at for at in (text.find(w, end) for w in words) if at >= 0]
        if later:
            after.append((min(later), i))
        earlier = [text.rfind(w, 0, start) for w in words]
        if max(earlier) >= 0:
            before.append((-max(earlier), i))
    nearest = min(after or before or [(0, 0)])
    return options[nearest[1]][0]


def resolve_tool_name(query: str, context: str = "") -> dict:
    """Resolve a user-typed tool name (or near-match) to a canonical MCP tool name.

    Tiered resolution (provider-agnostic, no LLM call):
      1. Exact match against registered MCP tool names (case-insensitive, normalized).
      2. Alias map from ``_MCP_TOOLS[*]["aliases"]``. When the alias is the bare name of a server
         that registers functions for different analyses (GraphST, PROST, SpatialPrompt, SPACEL,
         Seurat), ``context`` -- the sentence the name was found in -- picks the function the
         request is about; without it the server's main function is returned, as always.
      2b. A short name whose canonical form only adds a ``run_`` prefix (source ``"run_prefix"``).
      3. Give up; return a candidates list from a substring match against registered names.

    Args:
        query: User-typed tool name or descriptor (e.g. "RCTD", "cell2loc", "the bayesian one").
        context: Optional text around the name, e.g. "Use GraphST to deconvolve the spots".

    Returns:
        dict with keys:
            ``name``: canonical MCP tool name (str) or None
            ``score``: 1.0 for exact/alias/run_prefix, 0.0 for none
            ``source``: "exact" | "alias" | "run_prefix" | "none"
            ``candidates``: list[str] of top-3 alternatives when source is "none"
    """
    if not query or not isinstance(query, str):
        return {"name": None, "score": 0.0, "source": "none", "candidates": []}

    normalized = _normalize_name(query)
    callable_tools = _callable_mcp_tools()

    # Tier 1: exact match against registered MCP tools (case-insensitive, normalized)
    for tool_name in callable_tools:
        if _normalize_name(tool_name) == normalized:
            return {"name": tool_name, "score": 1.0, "source": "exact", "candidates": []}

    # Tier 2: alias map (curated aliases take precedence over the run_-prefix heuristic below)
    for _key, entry in _MCP_TOOLS.items():
        for alias in entry.get("aliases", []):
            if _normalize_name(alias) == normalized:
                canonical = entry.get("mcp_function")
                if canonical:
                    canonical = _sibling_for_context(normalized, canonical, context if isinstance(context, str) else "")
                    return {"name": canonical, "score": 1.0, "source": "alias", "candidates": []}

    # Tier 2b: a short/common name against a tool whose canonical form only adds a `run_` prefix and that
    # has no curated alias (users type "BASS"/"SEDR"/"SpotLight"; the tools register as run_bass/run_sedr/
    # run_spotlight). Accept ONLY an unambiguous single match so this never mis-routes.
    run_matches = [t for t in callable_tools if _normalize_name(t).removeprefix("run") == normalized]
    if len(run_matches) == 1:
        return {"name": run_matches[0], "score": 1.0, "source": "run_prefix", "candidates": []}

    # (A former Tier 3 here made a full LLM retrieval call just to fuzzy-match the tool NAME. It always
    # used the DEFAULT provider (Anthropic/claude) regardless of the running agent's LLM, so it failed on
    # every non-Anthropic agent — e.g. an Azure gpt-5 run logged two "credit balance too low" 400s per
    # tool-recommendation — and its fixed 0.5 score was below the sole caller's exact-match (>=1.0)
    # acceptance anyway, so the round-trip was pure waste. Removed; fall straight through to the lexical
    # Tier 4. This is provider-agnostic, free, and behavior-identical for the caller.)

    # Tier 4: give up; return candidates by simple substring match against registered tools.
    # Case-insensitive (uses normalized form), whitespace-insensitive (normalize strips whitespace).
    # We match if the first 4 normalized chars of the query appear anywhere in the normalized tool name.
    prefix = normalized[:4]
    cands: list[str] = []
    if prefix:
        cands = sorted([t for t in callable_tools if prefix in _normalize_name(t)])[:3]
    return {"name": None, "score": 0.0, "source": "none", "candidates": cands}


# ---------------------------------------------------------------------------
# MCP Tool Knowledge Base
# ---------------------------------------------------------------------------

_MCP_TOOLS: dict[str, dict[str, Any]] = {
    # ── Serial sections to one 3D object ───────────────────────────────────
    # The question that comes before any aligner. Listed first because on a multi-section object
    # it is the first call: aligning a stack that is already aligned moves coordinates that were
    # right, and aligning a deformed stack with a rigid tool produces a confident wrong answer.
    "spatial3d_diagnose": {
        "task": "three_d_reconstruction",
        "mcp_function": "diagnose_3d_stack",
        "full_name": "Serial-section alignment diagnosis",
        "description": (
            "Classify a stack of serial sections as already aligned, rigidly misaligned or "
            "non-rigidly deformed, with the measured value and the calibrated threshold behind "
            "every criterion. Writes a report and modifies nothing."
        ),
        "strengths": [
            "decides whether an alignment is needed at all, and which kind",
            "thresholds calibrated on a real 147-section atlas rather than chosen",
            "every verdict carries the numbers behind it",
            "refuses rather than guessing when the section order is unknown",
            "no GPU needed",
        ],
        "limitations": [
            "diagnoses only; it moves no coordinate",
            "needs two or more sections and an obs column naming them",
            "cannot tell micrometres from pixels, so physical claims stay unavailable",
        ],
        "best_for": [
            "any multi-section object before an aligner is chosen",
            "deciding between a rigid and a non-rigid aligner",
            "checking that a stack said to be aligned actually is",
        ],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"slice_key": "", "z_key": "", "n_genes": 8},
        "data_scale": "any",
        "gpu": False,
        "priority": 1,
        "aliases": ["diagnose_3d", "alignment_diagnosis", "spatial3d", "diagnose_3d_stack"],
    },
    "spatial3d_inspect": {
        "task": "three_d_reconstruction",
        "mcp_function": "inspect_3d_coordinates",
        "full_name": "3D coordinate inspection",
        "description": (
            "Which coordinate keys an object holds, which of them are three-column frames, and "
            "which contract rules it currently breaks. Read-only."
        ),
        "strengths": ["a cheap read before anything is measured or moved", "no GPU needed"],
        "limitations": ["reports only; writes nothing and measures no alignment"],
        "best_for": ["orienting on an unfamiliar multi-section object"],
        "input_requirements": {"h5ad": True, "spatial_coords": False, "images": False, "sc_reference": False},
        "key_params": {"coords_key": "spatial"},
        "data_scale": "any",
        "gpu": False,
        "priority": 2,
        "aliases": ["inspect_3d", "coordinate_inspection"],
    },
    # ── Serial-section alignment: the two branches that had no tool ────────
    "paste2": {
        "task": "spatial_alignment",
        "mcp_function": "paste2_partial_align",
        "full_name": "PASTE2 (partial optimal transport)",
        "description": (
            "Pairwise alignment for sections that overlap only partially. Takes an overlap "
            "fraction and solves a partial transport problem instead of matching every spot."
        ),
        "strengths": [
            "the only tool here for partial overlap",
            "estimates the overlap fraction per pair when not told",
            "reports the fraction it used, which is itself a claim about the tissue",
            "no GPU needed",
        ],
        "limitations": [
            "pairwise, so a long stack is aligned adjacent pair by adjacent pair",
            "the glmpca dissimilarity is an order of magnitude slower than kl or euclidean, and the "
            "s=0.0 overlap estimate always runs glmpca with 20 solves per pair whatever dissimilarity "
            "is chosen",
            "aligns in plane; it cannot supply a z",
        ],
        "best_for": [
            "a section that was torn, trimmed, or runs out along the axis",
            "any pair whose measured overlap is well below 1",
        ],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"s": 0.0, "alpha": 0.1, "dissimilarity": "glmpca"},
        "data_scale": "small to medium",
        "gpu": False,
        "priority": 4,
        "aliases": ["paste2", "partial_paste", "paste_2"],
    },
    "cast": {
        "task": "spatial_alignment",
        "mcp_function": "cast_align_slices",
        "full_name": "CAST (graph embedding + free-form deformation)",
        "description": (
            "Learns a graph-neural embedding per section, then registers with an affine fit "
            "followed by a free-form deformation. Non-rigid, at single-cell resolution."
        ),
        "strengths": [
            "handles deformation a similarity transform cannot undo",
            "single-cell resolution",
            "consumes and produces AnnData, unlike STalign's point clouds",
            "ffd_iterations=0 gives an affine-only result for a class-B stack",
        ],
        "limitations": [
            "uses only genes present in every slice, because the embedding is joint",
            "highly variable genes are chosen once across all slices; background spots are left out",
            "written for CUDA; runs on CPU at about a minute per 1,400 cells",
            "registers in plane; it cannot supply a z",
        ],
        "best_for": [
            "a class-C diagnosis: tearing, stretching, or partial overlap",
            "single-cell platforms where a spot-level aligner is too coarse",
        ],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"epochs": 400, "ffd_iterations": 400, "graph_strategy": "delaunay"},
        "data_scale": "small to medium",
        "gpu": True,
        "priority": 4,
        "aliases": ["cast", "cast_stack", "cast_mark"],
    },
    # ── Spatial Domain Clustering ──────────────────────────────────────────
    "scanpy_spatial": {
        "task": "spatial_clustering",
        "mcp_function": "run_scanpy_spatial_domain",
        "full_name": "Scanpy Spatial (Leiden/Louvain)",
        "description": (
            "PCA + expression kNN graph + Leiden clustering; spatial coordinates are not used in clustering. "
            "Fast non-spatial baseline that works on any spatial h5ad."
        ),
        "strengths": ["fast", "simple", "good baseline", "no GPU needed", "works with any spatial data"],
        "limitations": [
            "does not use spatial coordinates in clustering (expression-only graph)",
            "no image integration",
        ],
        "best_for": ["quick exploration", "baseline comparison", "large datasets", "first-pass analysis"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"resolution": 0.6, "n_neighbors": 15, "n_pcs": 30},
        "data_scale": "any",
        "gpu": False,
        "priority": 5,
        "aliases": ["scanpy", "scanpy_spatial", "leiden", "louvain"],
    },
    "graphst": {
        "task": "spatial_clustering",
        "mcp_function": "graphst_spatial_clustering",
        "full_name": "GraphST",
        "description": (
            "Graph neural network that learns joint gene-expression and spatial embeddings. "
            "Excels at identifying complex tissue architecture."
        ),
        "strengths": ["captures spatial structure", "graph-based", "good for complex tissues"],
        "limitations": ["GPU recommended", "slower than Leiden-based methods"],
        "best_for": ["complex tissue architecture", "spatially coherent domains", "Visium data"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        # n_clusters is a required int with no default in the portal; "auto or user-specified" made
        # the plan omit it and invited n_clusters="auto" (hunt 2026-09-30, u23-transcriptomics-skills-6).
        "key_params": {"n_clusters": "required", "cluster_tool": "leiden", "device": "auto"},
        "data_scale": "medium",
        "gpu": True,
        "priority": 3,
        "aliases": ["graphst"],
    },
    "stagate": {
        "task": "spatial_clustering",
        "mcp_function": "stagate_spatial_domains",
        "full_name": "STAGATE",
        "description": (
            "Attention-based graph neural network. Learns adaptive spatial neighborhoods "
            "with graph attention, producing spatially smooth domain boundaries."
        ),
        "strengths": ["attention mechanism", "adaptive neighborhoods", "smooth boundaries"],
        "limitations": ["GPU recommended", "sensitive to graph construction radius"],
        "best_for": ["tissues with clear domain boundaries", "Visium and Slide-seq"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        # Values are values; what they mean is in param_notes. The prose used to sit in key_params,
        # and the planner copied it into the call as rad_cutoff / k_cutoff -- a sentence for a
        # number (hunt 2026-09-30, u23-transcriptomics-skills-5).
        "key_params": {"n_clusters": "required", "rad_cutoff": 0.0, "k_cutoff": 0},
        "param_notes": {
            "rad_cutoff": "0 (default) = derived from the spot spacing; a fixed value is in obsm units",
            "k_cutoff": "0 = radius graph; >0 = kNN graph with that k",
        },
        "data_scale": "medium",
        "gpu": True,
        "priority": 0,
        "aliases": ["stagate"],
    },
    "cellcharter": {
        "task": "spatial_clustering",
        "mcp_function": "cellcharter_cluster_spatial_domains",
        "full_name": "CellCharter",
        "description": (
            "Multi-scale spatial clustering that captures tissue organization at multiple "
            "resolutions, from fine-grained to coarse compartments."
        ),
        "strengths": ["multi-scale", "handles heterogeneous tissue", "good for hierarchical structures"],
        "limitations": ["slower on very large datasets"],
        "best_for": ["hierarchical tissue organization", "tumor microenvironment", "complex samples"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        # CellCharter sweeps K rather than taking one: ClusterAutoK evaluates every K in
        # [n_clusters_min, n_clusters_max] and picks by stability. There is no `n_clusters`.
        "key_params": {"n_clusters_min": 3, "n_clusters_max": 12},
        "data_scale": "medium",
        "gpu": True,
        "priority": 4,
        "aliases": ["cellcharter", "cell_charter"],
    },
    "deepst": {
        "task": "spatial_clustering",
        "mcp_function": "deepst_identify_domains",
        "full_name": "DeepST",
        "description": (
            "Graph autoencoder over spatially augmented expression (DeepST/deepstkit) for spatial domain "
            "identification; H&E features are used only when the input already carries obsm['image_feat_pca']."
        ),
        "strengths": ["integrates morphology", "deep learning based"],
        "limitations": [
            "dense spot-by-spot matrices (memory grows with spots^2)",
            "an exact n_domains may be unreachable by the Leiden sweep (refused unless allow_resolution_fallback)",
        ],
        "best_for": ["Visium-scale slides"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        # DeepST spells the domain count `n_domains`, not `n_clusters`.
        "key_params": {"n_domains": 7, "pca_n_comps": 200},
        "data_scale": "medium",
        "gpu": True,
        "priority": 1,
        "aliases": ["deepst", "deep_st"],
    },
    "miso": {
        "task": "spatial_clustering",
        "mcp_function": "run_miso",
        "full_name": "MISO",
        "description": (
            "Multi-modal integration of spatial omics: jointly models RNA expression "
            "and histology image features for domain identification."
        ),
        "strengths": ["multi-modal (RNA + image)", "excellent when histology is available"],
        "limitations": [
            "needs H&E image for full benefit; without one MISO uses no spatial coordinates (expression clusters)",
            "default dense affinity is N x N per modality; sparse=True for large inputs",
            "GPU helpful",
        ],
        "best_for": ["Visium with H&E", "multi-modal spatial analysis"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": True, "sc_reference": False},
        "key_params": {"n_clusters": 6},
        "param_notes": {"histology_image_path": "optional H&E image; omit it to cluster without one"},
        "data_scale": "medium",
        "gpu": True,
        "priority": 4,
        "aliases": ["miso"],
    },
    "spaceflow": {
        "task": "spatial_clustering",
        "mcp_function": "spaceflow_spatial_domains",
        "full_name": "SpaceFlow",
        "description": (
            "Models tissue organization as a spatiotemporal flow field, capturing "
            "continuous spatial gradients and pseudo-spatial-time trajectories."
        ),
        "strengths": ["continuous domains", "pseudo-spatial-time", "gradient modeling"],
        "limitations": ["GPU helpful", "interpretability can be challenging"],
        "best_for": ["tissues with continuous gradients", "developmental spatial patterns"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        # SpaceFlow spells the domain count `target_n_clusters` (0 = let the resolution decide);
        # `n_clusters` is not one of its parameters.
        "key_params": {"target_n_clusters": 0, "seg_resolution": 1.0},
        "data_scale": "medium",
        "gpu": True,
        "priority": 5,
        "aliases": ["spaceflow", "space_flow"],
    },
    "stlearn": {
        "task": "spatial_clustering",
        "mcp_function": "stlearn_spatial_clustering",
        "full_name": "stLearn",
        "description": (
            "Extracts morphology features from H&E via a CNN and combines with expression "
            "for spatially-aware clustering, trajectory, and cell-cell interaction analysis."
        ),
        "strengths": ["CNN morphology features", "trajectory analysis", "cell interaction"],
        "limitations": ["needs H&E images", "older method"],
        "best_for": ["Visium with H&E", "trajectory + clustering combined"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": True, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": True,
        "priority": 6,
        "aliases": ["stlearn", "st_learn"],
    },
    # ── Spatially Variable Genes (SVG) ─────────────────────────────────────
    "hotspot": {
        "task": "svg_detection",
        "mcp_function": "hotspot_spatial_modules",
        "full_name": "Hotspot",
        "description": (
            "Identifies spatially variable genes via local autocorrelation statistics. "
            "Fast, well-calibrated p-values, and identifies gene modules."
        ),
        "strengths": ["fast", "well-calibrated p-values", "gene module detection", "memory efficient"],
        "limitations": ["assumes stationary patterns"],
        "best_for": ["first-pass SVG detection", "large datasets", "gene module discovery"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"n_neighbors": 30, "module_fdr_threshold": 0.05},
        "data_scale": "large",
        "gpu": False,
        "priority": 0,
        "aliases": ["hotspot"],
    },
    "somde": {
        "task": "svg_detection",
        "mcp_function": "somde_run",
        "full_name": "SOMDE",
        "description": (
            "Self-organizing map accelerated SVG detection. Uses SOM to reduce spatial "
            "complexity, then tests for spatial variability on the reduced representation."
        ),
        "strengths": ["memory efficient", "scalable to large datasets", "fast"],
        "limitations": ["SOM approximation may miss subtle patterns"],
        "best_for": ["large spatial datasets", "quick SVG screening", "memory-constrained analysis"],
        "input_requirements": {
            "h5ad": True,
            "spatial_coords": True,
            "images": False,
            "sc_reference": False,
            "visium_dir": True,
        },
        "key_params": {"som_dim": 20},
        "data_scale": "large",
        "gpu": False,
        "priority": 1,
        "aliases": ["somde"],
    },
    "spatialde": {
        "task": "svg_detection",
        "mcp_function": "spatialde_run_svg",
        "full_name": "SpatialDE",
        "description": (
            "Gaussian process-based model for spatial gene expression variability. "
            "Gold standard for SVG detection with pattern decomposition."
        ),
        "strengths": ["gold standard", "pattern decomposition", "well-validated"],
        "limitations": ["slow on large datasets", "memory intensive"],
        "best_for": ["small-medium datasets", "detailed pattern analysis", "benchmark reference"],
        "input_requirements": {
            "h5ad": True,
            "spatial_coords": True,
            "images": False,
            "sc_reference": False,
            "visium_dir": True,
        },
        "key_params": {},
        "data_scale": "small",
        "gpu": False,
        "priority": 3,
        "aliases": ["spatialde", "spatial_de"],
    },
    "spagft": {
        "task": "svg_detection",
        "mcp_function": "spagft_identify_svg",
        "full_name": "SpaGFT",
        "description": (
            "Graph Fourier Transform-based SVG identification. Decomposes spatial gene "
            "expression into frequency components to detect spatial patterns."
        ),
        "strengths": ["novel approach", "frequency-domain analysis", "pattern characterization"],
        "limitations": ["newer method, less validated"],
        "best_for": ["spatial pattern characterization", "periodic pattern detection"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 4,
        "aliases": ["spagft"],
    },
    "svgbit": {
        "task": "svg_detection",
        "mcp_function": "svgbit_run",
        "full_name": "SVGbit",
        "description": (
            "Binary identification of spatially variable genes. Simple and efficient "
            "approach using binarized spatial expression patterns."
        ),
        "strengths": ["simple", "fast", "intuitive interpretation"],
        "limitations": [
            "binary approach may miss continuous patterns",
            "density step holds a dense n_spots x n_spots neighbour matrix per worker process (memory grows with the square of the spot count)",
        ],
        "best_for": ["quick SVG screening", "binary spatial patterns"],
        "input_requirements": {
            "h5ad": True,
            "spatial_coords": True,
            "images": False,
            "sc_reference": False,
        },
        "key_params": {},
        "data_scale": "large",
        "gpu": False,
        "priority": 3,
        "aliases": ["svgbit"],
    },
    "spotgf": {
        # spotgf_denoise is a DENOISING tool that operates on GEM-format spatial data (NOT h5ad); it is
        # not an SVG detector. Filed under a non-goal task so it is no longer recommended for SVG.
        "task": "denoising",
        "mcp_function": "spotgf_denoise",
        "full_name": "SpotGF",
        "description": "Optimal-transport gene filtering (DENOISING) of GEM-format spatial data (not SVG detection).",
        "strengths": ["optimal-transport gene filtering"],
        "limitations": ["GEM-format input", "not an SVG detector"],
        "best_for": ["denoising", "signal recovery"],
        "input_requirements": {
            "gem": True,
            "h5ad": False,
            "spatial_coords": True,
            "images": False,
            "sc_reference": False,
        },
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 5,
        "aliases": ["spotgf"],
    },
    "prost": {
        "task": "svg_detection",
        "mcp_function": "prost_index_svg",
        "full_name": "PROST",
        "description": (
            "Ranks spatially variable genes by the PROST Index (PI), an image-based score of how spatially "
            "separable (SEP) and significant (SIG) each gene's expression pattern is; exports PI with its SEP "
            "and SIG components. No pattern-type classification."
        ),
        "strengths": ["image-based spatial pattern score", "PI with separability/significance components"],
        "limitations": [
            "PROST densifies the expression matrix internally",
            "gene images need a Visium array lattice (array_row/array_col) or grid interpolation",
        ],
        "best_for": ["ranking SVGs on Visium-style lattices"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 4,
        "aliases": ["prost"],
    },
    # ── Cell Type Deconvolution ────────────────────────────────────────────
    "cell2location": {
        "task": "deconvolution",
        "mcp_function": "run_cell2location",
        "full_name": "Cell2Location",
        "description": (
            "Bayesian model for spatially-resolved cell type deconvolution. Maps scRNA-seq "
            "reference signatures to spatial locations. Gold standard for Visium."
        ),
        "strengths": ["gold standard for Visium", "probabilistic", "well-validated", "cell abundance estimation"],
        "limitations": ["needs scRNA reference", "slow training", "GPU strongly recommended"],
        "best_for": ["Visium deconvolution", "cell type abundance mapping", "tissue composition"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        # No epoch overrides: 50/2000 under-trained both models against the portal's 250/30000, and
        # a plan passed them as the "best-practice" run (hunt 2026-09-30, u23-transcriptomics-skills-27,
        # user decision: inherit the portal defaults).
        "key_params": {"labels_key": "CellType", "batch_key": "Sample", "n_cells_per_location": 30},
        "data_scale": "medium",
        "gpu": True,
        "priority": 5,
        "aliases": ["cell2location", "cell2loc", "c2l", "c2location"],
    },
    "tangram": {
        "task": "deconvolution",
        "mcp_function": "tangram_map_sc_to_spatial",
        "full_name": "Tangram",
        "description": (
            "Learns a probabilistic mapping of single-cell profiles (or cell-type clusters) to spatial "
            "locations by gradient descent in PyTorch. Can operate at cell-level or cluster-level resolution."
        ),
        "strengths": [
            "flexible resolution",
            "gradient-descent mapping (PyTorch)",
            "fast with GPU",
            "cell-level mapping",
        ],
        "limitations": ["needs scRNA reference", "GPU recommended"],
        "best_for": ["cell-level spatial mapping", "fine-resolution deconvolution"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"mode": "clusters", "num_epochs": 1000},
        "data_scale": "medium",
        "gpu": True,
        "priority": 2,
        "aliases": ["tangram", "tangram_sc"],
    },
    "tacco": {
        "task": "deconvolution",
        "mcp_function": "tacco_annotate",
        "full_name": "TACCO",
        "description": (
            "Transfer cell type annotations from scRNA reference using conditional "
            "correlation optimization. Works without explicit deconvolution."
        ),
        "strengths": ["annotation transfer", "works without explicit deconvolution", "handles noise well"],
        "limitations": ["needs scRNA reference"],
        "best_for": ["annotation transfer", "noisy spatial data"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 3,
        "aliases": ["tacco"],
    },
    "stride": {
        "task": "deconvolution",
        "mcp_function": "stride_deconvolution",
        "full_name": "STRIDE",
        "description": (
            "Topic model-based deconvolution using scRNA reference. Treats cell types "
            "as topics and spatial spots as documents."
        ),
        "strengths": ["topic model approach", "interpretable", "handles rare cell types"],
        "limitations": ["needs scRNA reference", "topic model assumptions"],
        "best_for": ["interpretable deconvolution", "rare cell type detection"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 4,
        "aliases": ["stride"],
    },
    "bayestme": {
        "task": "deconvolution",
        "mcp_function": "bayestme_deconvolution",
        "full_name": "BayesTME",
        "description": (
            "Bayesian spatial topic model for cell type deconvolution. Models spatial "
            "dependencies between neighboring spots."
        ),
        "strengths": ["Bayesian", "models spatial dependencies", "uncertainty quantification"],
        "limitations": ["slow", "complex model"],
        "best_for": ["spatially-aware deconvolution", "uncertainty estimation"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "small",
        "gpu": False,
        "priority": 5,
        "aliases": ["bayestme", "bayesian_tme"],
    },
    "ucdeconvolve": {
        "task": "deconvolution",
        "mcp_function": "ucdeconvolve_base",
        "full_name": "UCDeconvolve",
        "description": (
            "Reference-free deconvolution — does NOT require a scRNA reference. "
            "UCDBase, a model pre-trained on a large single-cell atlas, runs on the UCD cloud API: it needs a "
            "UCD token and uploads the expression matrix."
        ),
        "strengths": ["no reference needed", "pre-trained model"],
        "limitations": [
            "limited to known cell types in pre-trained model",
            "less precise than reference-based",
            "uploads expression to the UCD cloud API (token required)",
        ],
        "best_for": ["no scRNA reference available", "quick cell type estimation", "exploratory analysis"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["ucdeconvolve", "uc_deconvolve"],
    },
    "starfysh": {
        "task": "deconvolution",
        "mcp_function": "starfysh_deconvolution",
        "full_name": "Starfysh",
        "description": (
            "Reference-free deep generative (auxiliary VAE) deconvolution guided by marker signatures: "
            "provided via signature_csv, or derived by Archetypal Analysis (then the factors are arch_<i>, "
            "not named cell types). PoE mode adds the paired H&E image."
        ),
        "strengths": ["reference-free", "uses marker gene signatures", "optional H&E integration (PoE)"],
        "limitations": ["named cell types need a signature_csv", "PoE needs the paired H&E image", "slower"],
        "best_for": ["archetype identification", "signature-based deconvolution"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": True,
        "priority": 5,
        "aliases": ["starfysh"],
    },
    # ── Spatial Communication ──────────────────────────────────────────────
    "commot": {
        "task": "spatial_communication",
        "mcp_function": "commot_spatial_communication",
        "full_name": "COMMOT",
        "description": (
            "Infers cell-cell communication in spatial transcriptomics using "
            "ligand-receptor databases and optimal transport."
        ),
        "strengths": ["optimal transport-based", "CellChat/CellPhoneDB databases", "spatial-aware"],
        "limitations": [
            "computationally intensive for large datasets",
            "dense spots x spots distance matrix: memory grows with the square of the spot count",
            "gene names must be symbols of the chosen species (Ensembl IDs are renamed from a var symbol column)",
            "3D (dims=3) scores one bounded block -- a few adjacent sections and an in-plane bounding box, at most "
            "block_max_cells (default 40000) -- in micrometres; a whole stack over the cap is refused with the count",
        ],
        "best_for": ["ligand-receptor analysis", "spatial cell communication", "signaling pathway analysis"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"lr_database": "CellChat", "dis_thr": 0.0, "dis_thr_unit": "coordinates"},
        "param_notes": {
            "dims": "2 (default; per section with section_key on a multi-section file) or 3 (one bounded block)",
            "dis_thr_um": "threshold in micrometres for a frame with units; 0 = automatic 200 um",
            "block_bbox_um": "[x0, y0, x1, y1] um over the frame's two in-plane axes (CCF y, z on Zhuang)",
        },
        "data_scale": "medium",
        "gpu": False,
        "priority": 1,
        "aliases": ["commot"],
    },
    "spaotsc": {
        "task": "spatial_communication",
        "mcp_function": "spaotsc_run",
        "full_name": "SPAOTSC",
        "description": (
            "Optimal transport-based spatial signaling analysis (upstream SpaOTsc). Maps scRNA-seq cells onto "
            "spots, then scores ligand-receptor signaling between cells over their OT-derived spatial distance. "
            "Needs an scRNA-seq reference and run_signaling=True with ligands and receptors."
        ),
        "strengths": ["optimal transport", "spatial distance weighting"],
        "limitations": [
            "cell-cell distance solves one Sinkhorn problem per cell pair (O(n_sc^2))",
            "dense n_sc x n_sc matrices",
        ],
        "best_for": ["spatial signaling networks", "distance-weighted communication"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["spaotsc"],
    },
    # ── Spatial Alignment & Integration ────────────────────────────────────
    "paste": {
        "task": "spatial_alignment",
        "mcp_function": "paste_pairwise_align",
        "full_name": "PASTE",
        "description": (
            "Pairwise alignment of spatial transcriptomics slices using optimal transport. "
            "Aligns multiple tissue sections for 3D reconstruction."
        ),
        "strengths": ["slice alignment", "3D reconstruction", "optimal transport"],
        "limitations": ["pairwise only", "assumes similar tissue structure"],
        "best_for": ["serial section alignment", "3D tissue reconstruction", "multi-slice integration"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 1,
        "aliases": ["paste"],
    },
    "moscot": {
        "task": "spatial_alignment",
        "mcp_function": "moscot_run",
        "full_name": "MOSCOT",
        "description": (
            "Optimal transport (moscot) for three problems: aligning two or more spatial sections "
            "concatenated in one AnnData onto a reference section, mapping single cells onto spatial "
            "spots (a cell-to-spot table), and coupling consecutive time points."
        ),
        "strengths": ["section alignment", "cell-to-spot mapping", "time-point couplings"],
        "limitations": [
            "alignment needs the sections concatenated with a batch_key column",
            "dense couplings: memory grows with n_spots x n_cells",
        ],
        "best_for": ["serial-section alignment", "scRNA-seq to spatial mapping", "time-series couplings"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["moscot"],
    },
    "st_gears": {
        "task": "spatial_alignment",
        "mcp_function": "st_gears_reconstruct_3d",
        "full_name": "ST-GEARS",
        "description": (
            "Geometric alignment of serial spatial transcriptomics sections: optimal-transport anchors, "
            "rigid stacking (rotation and translation) and an elastic field for nonlinear deformation."
        ),
        "strengths": ["geometric alignment", "handles deformation", "serial sections"],
        "limitations": ["needs multiple sections", "needs an obs cluster/annotation column for anchors"],
        "best_for": ["serial section alignment", "tissue with deformations"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"slice_key": "slice_id", "group_key": "annotation", "bin_step": 2},
        "data_scale": "medium",
        "gpu": False,
        "priority": 3,
        "aliases": ["stgears", "st_gears"],
    },
    "spacel_scube": {
        "task": "spatial_alignment",
        "mcp_function": "run_spacel_scube",
        "full_name": "SPACEL Scube",
        "description": (
            "3D alignment of multiple spatial transcriptomics slices into a shared coordinate "
            "frame. Takes comma-separated h5ad paths; produces aligned 3D coordinates suitable "
            "for downstream 3D-aware clustering or visualization."
        ),
        "strengths": ["3D coordinate alignment", "multi-slice", "serial sections"],
        "limitations": ["needs ≥2 slices", "assumes serial / near-adjacent sections"],
        "best_for": ["3D tissue reconstruction", "serial Visium stacks", "multi-slice alignment"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 4,
        "aliases": ["spacel_scube", "scube"],
    },
    "spiral_integrate": {
        "task": "spatial_alignment",
        "mcp_function": "spiral_integrate",
        "full_name": "SPIRAL (integrate)",
        "description": (
            "Graph-based adversarial domain adaptation that jointly embeds multiple spatial "
            "slices into a shared latent space, correcting batch effects while preserving "
            "biological variation. Output supports downstream multi-slice clustering."
        ),
        "strengths": ["multi-slice integration", "batch correction", "shared latent space"],
        "limitations": ["needs ≥2 slices", "GPU recommended", "training-time cost"],
        "best_for": ["multi-batch spatial integration", "joint embeddings across serial slices"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"n_epochs": 200, "latent_dim": 32, "cluster_method": "leiden", "resolution": 0.8},
        "data_scale": "medium",
        "gpu": True,
        "priority": 5,
        "aliases": ["spiral_integrate", "spiral"],
    },
    "spiral_align": {
        "task": "spatial_alignment",
        "mcp_function": "spiral_align",
        "full_name": "SPIRAL (align)",
        "description": (
            "Maps the second of two spatial slices into the first slice's frame: SPIRAL integration, "
            "then this wrapper's own per-cluster fused Gromov-Wasserstein transport through the "
            "clusters both slices share. SPIRAL's CoordAlignment is not run; spots outside every "
            "shared cluster are left unplaced (NaN) and counted."
        ),
        "strengths": ["pairwise alignment", "GW optimal transport", "expression + spatial trade-off"],
        "limitations": [
            "pairwise only (two slices)",
            "GPU recommended",
            "spots in clusters found in one slice only are not placed",
        ],
        "best_for": ["two-slice coordinate alignment", "common-frame comparison"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"alpha": 0.5, "cluster_method": "leiden", "resolution": 0.8},
        "data_scale": "medium",
        "gpu": True,
        "priority": 5,
        "aliases": ["spiral_align"],
    },
    # ── Super-resolution ───────────────────────────────────────────────────
    "istar": {
        "task": "super_resolution",
        "mcp_function": "istar_full_pipeline",
        "full_name": "iStar",
        "description": (
            "Super-resolution enhancement of spatial transcriptomics via image-guided "
            "imputation. Predicts gene expression at sub-spot resolution."
        ),
        "strengths": ["sub-spot resolution", "image-guided", "enhances Visium resolution"],
        "limitations": ["needs H&E image", "computationally intensive"],
        "best_for": ["Visium super-resolution", "sub-spot gene expression prediction"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": True, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": True,
        "priority": 1,
        "aliases": ["istar"],
    },
    "xfuse": {
        "task": "super_resolution",
        "mcp_function": "xfuse_run",
        "full_name": "XFuse",
        "description": (
            "Image-guided spatial transcriptomics super-resolution using a generative model "
            "that combines histology and expression data."
        ),
        "strengths": ["generative model", "image-guided", "principled probabilistic approach"],
        "limitations": ["slow training", "GPU required", "needs H&E image"],
        "best_for": ["probabilistic super-resolution", "Visium with H&E"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": True, "sc_reference": False},
        "key_params": {},
        "data_scale": "small",
        "gpu": True,
        "priority": 2,
        "aliases": ["xfuse"],
    },
    # ── Comprehensive Pipeline ─────────────────────────────────────────────
    "seurat": {
        "task": "comprehensive_pipeline",
        "mcp_function": "seurat_spatial_qc_cluster",
        "full_name": "Seurat (R)",
        "description": (
            "R-based spatial analysis pipeline: QC filtering, LogNormalize (NormalizeData), HVGs, PCA, "
            "graph clustering and spatially variable features; separate Seurat tools draw spatial feature "
            "plots and find markers."
        ),
        "strengths": [
            "comprehensive",
            "well-validated",
            "LogNormalize + PCA + graph clustering",
            "marker detection (seurat_find_markers)",
        ],
        "limitations": ["R-based (slower interop)", "large memory footprint"],
        "best_for": ["end-to-end spatial analysis", "when R ecosystem is preferred"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": True, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["comprehensive_pipeline", "full_pipeline", "seurat"],
    },
    # ── Other Specialized ──────────────────────────────────────────────────
    "mist": {
        # mistyr_spatial_modeling is MISTy — multi-view intercellular INTERACTION modeling; it produces
        # no spatial-domain cluster labels. It was miscategorised under "spatial_clustering", so it was
        # recommended for domain identification (which it cannot do); it now sits under its true task.
        #
        # It must NOT claim the alias "mist". A separate registered server keyed "mist" is MIST/ReST
        # (mist_regions_impute — region detection and imputation), and an alias hit resolves at score
        # 1.0, the confidence _detect_user_specified_tool requires to pin a tool. While "mist" was
        # listed here, "Run MIST to detect regions" pinned this tool, which detects no regions.
        # "MIST" now falls through to the fuzzy tier, which returns both candidates at score 0.0 —
        # honest ambiguity, below the pin threshold. The row KEY stays "mist" deliberately: it is the
        # tool_key that names this tool's output directory and expected filenames, so renaming it
        # would move files on disk. Resolution never reads the key, only "aliases" and "mcp_function".
        "task": "spatial_communication",
        "mcp_function": "mistyr_spatial_modeling",
        "full_name": "MISTy",
        "description": "Multi-view spatial interaction modeling of marker-marker relationships from an "
        "intraview and a Gaussian-weighted paraview (no juxtaview). Does NOT produce spatial-domain cluster labels.",
        "strengths": ["intercellular interaction modeling", "multi-view"],
        "limitations": ["not a clustering/region tool"],
        "best_for": ["spatial interaction modeling", "marker relationship analysis"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 7,
        "aliases": ["mistyr"],
    },
    "stage": {
        "task": "spatial_clustering",
        "mcp_function": "stage_run",
        "full_name": "STAGE",
        "description": "Autoencoder that maps expression to spot coordinates and decodes expression at new positions: denser Visium/ST maps (generation) or simulated sections between Slide-seq sections (3d_model). It does not cluster.",
        "strengths": [
            "decodes expression between measured spots",
            "autoencoder-based",
            "3D stacks from serial Slide-seq sections",
        ],
        "limitations": [
            "no cluster labels (cluster its output separately)",
            "generation needs array-grid coordinates (obs array_row/array_col)",
            "recovery cannot run in the installed STAGE 1.0.1",
        ],
        "best_for": ["higher-density expression maps from Visium/ST", "interpolating between serial sections"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"data_type": "10x", "experiment": "generation", "hvg_flavor": "seurat_v3"},
        "data_scale": "medium",
        "gpu": True,
        "priority": 7,
        "aliases": ["stage"],
    },
    "spatialprompt": {
        "task": "spatial_clustering",
        "mcp_function": "spatialprompt_cluster",
        "full_name": "SpatialPrompt",
        "description": "SpatialPrompt SpatialCluster: KMeans domains from expression plus spatial-neighbourhood averages.",
        "strengths": ["CPU-only (numpy/scikit-learn)", "no reference needed"],
        "limitations": [
            "the number of domains (n_clust) must be supplied",
            "fixed internal settings (1000 genes, 50 components)",
        ],
        "best_for": ["quick spatial domain segmentation of spot-based slides"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {},
        "data_scale": "medium",
        "gpu": False,
        "priority": 7,
        "aliases": ["spatialprompt", "spatial_prompt"],
    },
    "spacel_splane": {
        "task": "spatial_clustering",
        "mcp_function": "run_spacel_splane",
        "full_name": "SPACEL Splane",
        "description": (
            "Deep-learning spatial domain identification. Designed to share a domain label "
            "space across multiple serial slices when paired with SPACEL Scube for 3D alignment, "
            "but also runs on a single slice."
        ),
        "strengths": ["serial-section friendly", "label-consistent across slices when paired with Scube"],
        "limitations": [
            "GPU recommended",
            "single-slice mode loses 3D advantage",
            "clusters cell-type PROPORTIONS, so the h5ad needs uns['celltypes'] proportion columns or a celltype_key",
        ],
        "best_for": ["multi-slice / 3D spatial domains", "serial Visium stacks", "joint domain labelling"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"n_clusters": 7, "celltype_key": "", "drop_unlabeled": False},
        "data_scale": "medium",
        "gpu": True,
        "priority": 4,
        "aliases": ["spacel_splane", "splane", "spacel"],
    },
    "spicemix": {
        "task": "spatial_clustering",
        "mcp_function": "run_spicemix",
        "full_name": "SpiceMix",
        "description": (
            "Spatial factorization that decomposes gene expression into K latent metagene "
            "factors while accounting for spatial neighborhood structure. The portal fits one "
            "slice per call; SpiceMix's joint multi-replicate fitting is not exposed."
        ),
        "strengths": ["spatially aware factorization", "interpretable metagenes"],
        "limitations": [
            "factor count K must be chosen",
            "interpretation differs from hard clusters",
            "one slice per call (no joint multi-replicate fit)",
        ],
        "best_for": ["spatial program discovery", "soft domain decomposition"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"K": 10, "n_epochs": 100, "device": "cpu", "min_spots_per_gene": 1},
        "data_scale": "medium",
        "gpu": False,
        "priority": 6,
        "aliases": ["spicemix"],
    },
    # ── Empirical-winner additions (from multi-LLM benchmark, 2026-05-11) ──
    "spacexr_rctd": {
        "task": "deconvolution",
        "mcp_function": "spacexr_rctd_deconvolution",
        "full_name": "SpaceXR / RCTD",
        "description": (
            "Robust Cell Type Decomposition via SpaceXR. Reference-based decomposition with "
            "Poisson likelihood; top empirical performer on Visium and slide-seq."
        ),
        "strengths": ["highest empirical Pearson r", "robust to reference imperfections", "fast"],
        "limitations": ["needs R env (spacexr)"],
        "best_for": ["Visium deconvolution", "slide-seq deconvolution", "any task with a clean sc reference"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"UMI_min": 100, "mode": "doublet"},
        "data_scale": "any",
        "gpu": False,
        "priority": 1,
        "aliases": ["rctd", "spacexr", "spacexr_rctd", "rctd_v2"],
    },
    "card": {
        "task": "deconvolution",
        "mcp_function": "run_card",
        "full_name": "CARD",
        "description": ("Conditional Autoregressive Deconvolution. Reference-based; uses spatial smoothing."),
        "strengths": ["spatial smoothing", "strong on Visium"],
        "limitations": ["needs R env", "memory-heavy at 100k+ spots"],
        "best_for": ["Visium with H&E", "reference-based deconvolution"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"min_count_gene": 100, "min_count_spot": 5},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["card"],
    },
    "spatialscope": {
        "task": "deconvolution",
        "mcp_function": "run_spatialscope",
        "full_name": "SpatialScope",
        "description": (
            "SpatialScope Stage-1 Cell-Type Identification (RCTD-style WarmStart on CPU via ray) giving "
            "per-spot cell-type proportions; the GPU diffusion Stage-2 is not run."
        ),
        "strengths": ["RCTD likelihood model", "CPU-only"],
        "limitations": [
            "one ray task per spot: slow on large slides",
            "dense frames: memory scales with shared genes x (spots + reference cells)",
            "no single-cell Stage-2",
        ],
        "best_for": ["Visium deconvolution", "complex tissue"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"UMI_min_sigma": 300, "input_scale": "auto"},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["spatialscope"],
    },
    "celldart": {
        "task": "deconvolution",
        "mcp_function": "run_celldart",
        "full_name": "CellDART",
        "description": "Domain-adversarial network for cell type deconvolution.",
        "strengths": ["DA training", "robust to ref/spatial domain shift"],
        "limitations": ["GPU recommended"],
        "best_for": ["Visium deconvolution", "slide-seq deconvolution"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"n_iterations": 3000},
        "data_scale": "medium",
        "gpu": True,
        "priority": 3,
        "aliases": ["celldart", "cell_dart"],
    },
    "destvi": {
        "task": "deconvolution",
        "mcp_function": "run_destvi",
        "full_name": "DestVI",
        "description": "scvi-tools-based deep generative deconvolution.",
        "strengths": ["scvi-tools integration", "uncertainty quantification"],
        "limitations": ["long training time", "GPU needed"],
        "best_for": ["Visium deconvolution with scvi pipeline"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": True},
        "key_params": {"max_epochs_sc": 250, "max_epochs_st": 2500},
        "data_scale": "medium",
        "gpu": True,
        "priority": 4,
        "aliases": ["destvi", "dest_vi"],
    },
    "bsp": {
        "task": "svg_detection",
        "mcp_function": "bsp_identify_svg",
        "full_name": "BSP (scBSP, single-cell big-small patch)",
        "description": "Granularity-based SVG test (big-small patch local-mean variances, fitted log-normal null) with a p-value per gene.",
        "strengths": ["per-gene p-values from a fitted null", "strong F1 on visium"],
        "limitations": ["slower than Hotspot on very large data"],
        "best_for": ["Visium SVG detection", "MERFISH"],
        "input_requirements": {"h5ad": True, "spatial_coords": True, "images": False, "sc_reference": False},
        "key_params": {"top_k_genes": 200},
        "data_scale": "medium",
        "gpu": False,
        "priority": 2,
        "aliases": ["bsp"],
    },
    "pathway_enrichment": {
        "task": "functional_enrichment",
        "mcp_function": "run_pathway_enrichment",
        "full_name": "Pathway Enrichment (decoupler ORA / GSEA-prerank)",
        "description": (
            "Gene-set over-representation (Fisher, BH) on the top genes of a ranked table and GSEA-prerank on the whole "
            "ranking, against MSigDB hallmark / GO / Reactome / WikiPathways or a local .gmt. Takes the CSVs the SVG tools "
            "write, a rank_genes_groups group from an h5ad, or a short inline list."
        ),
        "strengths": ["works on any gene list", "two methods, two backends", "offline after the first fetch", "no GPU"],
        "limitations": ["gene symbols only (Ensembl IDs refused)", "mouse is matched to the human collections by case"],
        "best_for": ["interpreting SVGs", "annotating domain markers", "what pathways a cluster's genes belong to"],
        "input_requirements": {"h5ad": False, "spatial_coords": False, "images": False, "sc_reference": False},
        "key_params": {"collections": "hallmark,go_bp,reactome", "top_n": 200},
        "param_notes": {
            "gene_table": "a ranked gene CSV, e.g. an SVG tool's output",
            "score_column": "the ranking column; needed for GSEA",
        },
        "data_scale": "any",
        "gpu": False,
        "priority": 1,
        # No ordinary English words: an alias is a pin, read after "with"/"using"/"via" too, so
        # "enrichment", "pathway" and "functional" pinned this tool onto requests that only
        # mentioned them ("compare tumor and stroma with enrichment of immune genes"), past the
        # goal table that deliberately ignores a bare "enrichment" (RL-3) (hunt 2026-09-30,
        # u23-transcriptomics-skills-3). Method and package names only.
        "aliases": ["gsea", "ora", "gene set", "decoupler", "gseapy", "go terms"],
    },
    "pathway_activity": {
        "task": "functional_enrichment",
        "mcp_function": "run_pathway_activity",
        "full_name": "Per-spot Pathway Activity (decoupler PROGENy / hallmark)",
        "description": (
            "Score PROGENy signalling pathways (weighted ULM) or MSigDB hallmark sets in every spot of a spatial h5ad, "
            "paint the most variable pathways on the tissue, and summarise by spatial domain or cell type."
        ),
        "strengths": ["per-spot scores in obsm", "tissue maps", "per-group means and t-tests", "no GPU"],
        "limitations": ["needs gene symbols as var_names", "small targeted panels may cover few sets"],
        "best_for": ["signalling activity across a section", "pathway differences between domains"],
        "input_requirements": {"h5ad": True, "spatial_coords": False, "images": False, "sc_reference": False},
        "key_params": {"collections": "progeny", "method": "ulm", "group_key": "spatial_domain"},
        "data_scale": "any",
        "gpu": False,
        "priority": 1,
        # Not "hallmark", "signaling" or "signalling": "infer cell-cell communication with signaling
        # pathways" pinned this tool over COMMOT (hunt 2026-09-30, u23-transcriptomics-skills-3).
        # "progeny" stays -- it is PROGENy, the resource this tool scores, named as such.
        "aliases": ["pathway activity", "progeny", "ulm"],
    },
}

# Task-type display names and descriptions
_TASK_TYPES: dict[str, dict[str, str]] = {
    "spatial_clustering": {
        "name": "Spatial Domain Identification",
        "description": "Identify spatially coherent tissue domains/regions by clustering spots using both gene expression and spatial information.",
        "evaluation_metric": "ARI, NMI (if ground truth available)",
    },
    "svg_detection": {
        "name": "Spatially Variable Gene Detection",
        "description": "Identify genes whose expression varies significantly across spatial locations — biomarkers of tissue architecture.",
        "evaluation_metric": "Jaccard overlap, Moran's I, F1 score",
    },
    "deconvolution": {
        "name": "Cell Type Deconvolution",
        "description": "Estimate the cell type composition at each spatial location by deconvolving bulk spot expression using scRNA reference profiles.",
        "evaluation_metric": "RMSE, Pearson r, Jensen-Shannon distance",
    },
    "spatial_communication": {
        "name": "Spatial Cell-Cell Communication",
        "description": "Infer ligand-receptor interactions between spatially proximal cell types to map signaling networks in tissue.",
        "evaluation_metric": "Pathway enrichment, communication score significance",
    },
    "functional_enrichment": {
        "name": "Pathway / Functional Enrichment",
        "description": "Interpret a gene list (SVGs, domain markers) or a whole section against pathway and gene-set collections: over-representation, GSEA-prerank, and per-spot pathway activity.",
        "evaluation_metric": "Recovery of known pathway terms; leading-edge overlap",
    },
    "spatial_alignment": {
        "name": "Spatial Slice Alignment / Integration",
        "description": "Align multiple spatial transcriptomics tissue sections for 3D reconstruction or cross-sample integration.",
        "evaluation_metric": "Alignment score, landmark correspondence",
    },
    "three_d_reconstruction": {
        "name": "Serial Sections to One 3D Object",
        "description": (
            "Decide whether a stack of serial sections is already in a shared coordinate system, "
            "rigidly misaligned or non-rigidly deformed, and act on that: align only when it is "
            "needed, into a new coordinate key with an explicit z, and measure the same adjacent-"
            "pair metrics before and after. Distinct from spatial_alignment, which is the act of "
            "aligning; this is the question of whether to, and the evidence either way."
        ),
        "evaluation_metric": (
            "Adjacent-pair centroid offset, outline containment and local-shift dispersion, each "
            "before and after, with no pair permitted to get worse"
        ),
    },
    "super_resolution": {
        "name": "Spatial Super-Resolution",
        "description": "Enhance spatial resolution by predicting gene expression at sub-spot level using histology images.",
        "evaluation_metric": "Imputation accuracy, spatial correlation",
    },
    "denoising": {
        "name": "Spatial Expression Denoising",
        "description": "Recover biological signal from sparse or noisy spatial measurements by filtering technical dropout and background.",
        "evaluation_metric": "Signal-to-noise ratio, correlation with a reference profile",
    },
    "comprehensive_pipeline": {
        "name": "Comprehensive Pipeline",
        "description": "End-to-end spatial analysis covering QC, normalization, clustering, marker detection, and visualization.",
        "evaluation_metric": "Multiple (per-step metrics)",
    },
}

# scRNA-seq detection markers
_SCRNA_CELLRANGER_MARKERS = {
    "filtered_feature_bc_matrix.h5",
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix.h5",
    "raw_feature_bc_matrix",
    "metrics_summary.csv",
}

_SCRNA_FILE_EXTENSIONS = {".h5ad", ".h5", ".loom", ".rds", ".rda"}

# Platform labels diagnose_spatial_data gives a CONTAINER rather than a spatial technology: every
# h5ad is "pre-converted", every .rds is an R object, a bare filtered_feature_bc_matrix.h5 is "10x
# (counts only)". A single-cell file arrives in each of them as often as a slide does, so for these
# the label is no evidence and only coordinates decide. Reading any label but "unknown" as spatial
# sent a scRNA-seq reference h5ad and a Cell Ranger .h5 to the spatial pipeline, and left the
# single-cell arms below unreachable (hunt 2026-09-30, u23-transcriptomics-skills-1).
_CONTAINER_PLATFORMS = frozenset(
    {
        "pre-converted",
        "R (Seurat / SCE / matrix)",
        "10x (counts only)",
        "10x (Matrix Market triplet)",
        "Matrix Market (MTX)",
        "Loom (loompy / anndata)",
    }
)


def _platform_names_a_spatial_technology(platform: Any) -> bool:
    """True for a label like "10x Visium (Space Ranger)" or "MERFISH/Vizgen"; False for a container.

    Only the bare "unknown" and the container labels say nothing. "unknown (image only)" is an
    image, not a container: it keeps the spatial pipeline's image diagnosis ("Need an h5ad with
    spatial coordinates to embed this image"), which a prefix test on "unknown" sent to the
    non-spatial path to come back "Unrecognized file format" (hunt 2026-09-30, rp-u23 review of
    u23-transcriptomics-skills-1).
    """
    label = str(platform or "unknown")
    return label != "unknown" and label not in _CONTAINER_PLATFORMS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_transcriptomics_data(input_path: str) -> str:
    """Diagnose any transcriptomics data: spatial, single-cell, or bulk.

    Extended version of diagnose_spatial_data() that additionally detects:
      - 10x Chromium scRNA-seq (Cell Ranger output without spatial/)
      - Smart-seq2/3 count matrices
      - Drop-seq DGE
      - Bulk RNA-seq count matrices
      - CITE-seq / multi-modal
      - Pre-existing h5ad with or without spatial info

    For spatial data, delegates to the spatial_pipeline module for detailed
    platform detection. For non-spatial data, performs basic format detection
    and recommends standard scanpy/Seurat workflows.

    Args:
        input_path: Path to a file or directory containing transcriptomics data.

    Returns:
        str: JSON diagnostic report with detected format, data type classification
             (spatial / single_cell / bulk / unknown), completeness assessment,
             and recommended analysis workflow.

    """
    p = Path(input_path)

    if not p.exists():
        return json.dumps(
            {"status": "error", "message": f"Path does not exist: {input_path}", "data_type": "unknown"},
            indent=2,
        )

    # Try spatial diagnosis first
    from spatialomicsgym.tool.spatial_pipeline import diagnose_spatial_data

    spatial_diag = json.loads(diagnose_spatial_data(input_path))

    # If spatial data is detected, enrich with MCP tool recommendations. Treat as spatial when the
    # platform was inferred OR the file plainly carries spatial coordinates (obsm['spatial'] / x-y cols):
    # a real spatial h5ad whose platform stayed "unknown" was wrongly sent to the non-spatial path.
    coords = spatial_diag.get("coordinates")
    has_spatial_coords = (
        bool(coords.get("in_obsm") or coords.get("found") is True) if isinstance(coords, dict) else False
    )
    if spatial_diag.get("status") == "diagnosed" and (
        _platform_names_a_spatial_technology(spatial_diag.get("platform")) or has_spatial_coords
    ):
        spatial_diag["data_type"] = "spatial"
        spatial_diag["mcp_tools_available"] = True
        spatial_diag["recommended_first_steps"] = [
            "Run the spatial data pipeline to convert to MCP-compatible h5ad",
            "Use recommend_analysis_tools() to select appropriate MCP tools",
            "Run build_analysis_workflow() for a complete analysis plan",
        ]
        return json.dumps(spatial_diag, indent=2)

    # If not spatial, try scRNA / bulk detection
    if p.is_file():
        return _diagnose_non_spatial_file(p)
    else:
        return _diagnose_non_spatial_directory(p)


def _missing_reference_error(param: str, path: str) -> dict[str, str]:
    return {
        "status": "error",
        "message": (
            f"{param} was given but does not exist: {path}. Pass the reference's real path, or leave "
            f"{param} unset to plan with reference-free tools only."
        ),
    }


def recommend_analysis_tools(
    h5ad_path: str,
    analysis_goals: str = "auto",
    data_type: str = "auto",
    sc_reference_path: str | None = None,
) -> str:
    """Recommend MCP tools and workflows based on data and analysis goals.

    Inspects the h5ad file to determine data characteristics (has spatial coords,
    has images, has scRNA reference, number of spots/cells, etc.) and matches
    against the MCP tool knowledge base to produce ranked recommendations.

    Args:
        h5ad_path: Path to h5ad file (ideally already MCP-compatible).
        analysis_goals: Comma-separated goals, or 'auto' to infer from the data.
            Options: spatial_clustering, svg_detection, deconvolution,
            spatial_communication, spatial_alignment, three_d_reconstruction,
            super_resolution, denoising, functional_enrichment,
            comprehensive_pipeline. 'all' means the same as 'auto', which only
            ever infers the first four; the others have to be named explicitly.
        data_type: 'spatial', 'single_cell', 'bulk', or 'auto' (auto-detect).
        sc_reference_path: Optional path to a single-cell reference h5ad. When
            provided, deconvolution tools that require an SC reference become
            eligible; without it, they are hard-dropped from the recommendations.

    Returns:
        str: JSON with ranked tool recommendations per analysis goal, including
             tool name, priority, strengths, key parameters, and data compatibility.

    """
    p = Path(h5ad_path)
    if not p.exists():
        return json.dumps({"status": "error", "message": f"File not found: {h5ad_path}"}, indent=2)

    # Inspect the h5ad to determine data characteristics
    data_profile = _profile_h5ad(h5ad_path)
    if data_profile.get("status") == "error":
        return json.dumps(data_profile, indent=2)

    # A reference that was named but is not there is the caller's error, not "no reference": read as
    # the latter, a typo'd path returned reference-free tools -- the cloud-upload one first -- under
    # status success, with nothing naming the missing file (hunt 2026-09-30,
    # u23-transcriptomics-skills-19).
    if sc_reference_path and not Path(sc_reference_path).exists():
        return json.dumps(_missing_reference_error("sc_reference_path", sc_reference_path), indent=2)

    # Caller-supplied SC reference enables deconv tools that need one.
    if sc_reference_path and Path(sc_reference_path).exists():
        data_profile["has_sc_reference"] = True
        data_profile["sc_reference_path"] = sc_reference_path
    else:
        data_profile["has_sc_reference"] = False

    # Auto-detect data type
    if data_type == "auto":
        data_type = data_profile.get("data_type", "unknown")

    # Determine analysis goals. A sequence is accepted as well as the documented comma-separated
    # string: "goals" is plural, the know-how spells the call with a list, and a bare `.split()`
    # on one raised AttributeError *before* any tool ran — leaving the agent to guess a tool,
    # which is the exact failure this recommender exists to prevent.
    goals = _resolve_goals(analysis_goals, data_profile, data_type)

    recommendations: dict[str, Any] = {
        "status": "success",
        "h5ad_path": h5ad_path,
        "data_type": data_type,
        "data_profile": {
            "n_spots_cells": data_profile.get("n_obs"),
            "n_genes": data_profile.get("n_vars"),
            "has_spatial_coords": data_profile.get("has_spatial"),
            "has_images": data_profile.get("has_images"),
            "has_sc_reference": data_profile.get("has_sc_reference", False),
            "platform": data_profile.get("platform", "unknown"),
            "raw_counts_available": data_profile.get("raw_counts_available", False),
            "visium_dir_convertible": data_profile.get("visium_dir_convertible", False),
        },
        "analysis_goals": goals,
        "recommendations": {},
    }

    # A goal that yields no tool used to be a bare `continue`: it vanished from "recommendations"
    # while still being echoed under "analysis_goals" at status success, so the caller could not tell
    # an accepted goal from a refused one. Report it instead, with the reason.
    unplannable: dict[str, str] = {}
    for goal in goals:
        goal_recs = _recommend_for_goal(goal, data_profile)
        if goal_recs:
            recommendations["recommendations"][goal] = {
                "task_info": _TASK_TYPES.get(goal, {}),
                "recommended_tools": goal_recs,
            }
        else:
            unplannable[goal] = _why_a_goal_has_no_tools(goal, data_profile)

    if unplannable:
        recommendations["unplannable_goals"] = unplannable

    # Add general guidance. The hint must not promise a plan when nothing was recommended:
    # build_analysis_workflow() would return QC and no analysis step at all.
    # evaluate_and_compare_results() is defined nowhere, and 'unplannable_goals' is absent when no
    # goal was inferred at all (bulk or unknown data), so neither may be named as the next step
    # (hunt 2026-09-30, u23-transcriptomics-skills-4 and -20).
    if recommendations["recommendations"]:
        recommendations["workflow_hint"] = (
            "For a complete analysis plan with ordered steps, call build_analysis_workflow(). "
            "For multi-tool comparison, run 2-3 tools per task and compare their outputs in Python -- "
            "no registered tool makes that comparison for you."
        )
    elif not goals:
        recommendations["workflow_hint"] = (
            f"No analysis goal is inferred for data_type '{data_type}', so nothing was recommended. "
            f"Name the goal you want in analysis_goals; supported goals: {', '.join(sorted(_TASK_TYPES))}."
        )
    else:
        recommendations["workflow_hint"] = (
            "No tool was recommended, so build_analysis_workflow() would return QC and no analysis "
            "step. Fix the goals or the missing inputs named in 'unplannable_goals' first."
        )

    return json.dumps(recommendations, indent=2)


def build_analysis_workflow(
    h5ad_path: str,
    goals: str = "auto",
    reference_h5ad_path: str | None = None,
    data_type: str = "auto",
) -> str:
    """Build a complete, ordered analysis workflow with specific MCP tool calls.

    Produces a step-by-step plan covering: QC → preprocessing → analysis
    (one or more MCP tools per goal) → evaluation. Each step includes the
    exact MCP function name, parameters, and expected outputs.

    Args:
        h5ad_path: Path to h5ad file.
        goals: Analysis goals -- a list, or the documented comma-separated string. 'auto' and
               'all' both mean "infer from the data", so the ``analysis_goals`` list in a
               recommend_analysis_tools reply can be passed straight through.
        reference_h5ad_path: Optional scRNA reference h5ad for deconvolution.
        data_type: 'spatial', 'single_cell', 'bulk', or 'auto'.

    Returns:
        str: JSON workflow with ordered steps, each containing tool/function
             name, parameters, expected outputs, and rationale.

    """
    p = Path(h5ad_path)
    if not p.exists():
        return json.dumps({"status": "error", "message": f"File not found: {h5ad_path}"}, indent=2)

    data_profile = _profile_h5ad(h5ad_path)
    if data_profile.get("status") == "error":
        return json.dumps(data_profile, indent=2)

    if reference_h5ad_path and not Path(reference_h5ad_path).exists():  # u23-transcriptomics-skills-19
        return json.dumps(_missing_reference_error("reference_h5ad_path", reference_h5ad_path), indent=2)

    # Caller-supplied SC reference enables deconvolution tools that need one. Mirror
    # recommend_analysis_tools: without this, _recommend_for_goal hard-drops EVERY
    # reference-based deconvolution tool (cell2location, RCTD, tangram, tacco, stride, card, ...)
    # so a deconvolution workflow silently contained only the reference-free tools.
    if reference_h5ad_path and Path(reference_h5ad_path).exists():
        data_profile["has_sc_reference"] = True
        data_profile["sc_reference_path"] = reference_h5ad_path
    else:
        data_profile["has_sc_reference"] = False

    if data_type == "auto":
        data_type = data_profile.get("data_type", "unknown")

    # Same parser as recommend_analysis_tools: this is the function its reply hands its own
    # analysis_goals list to, so it has to accept what that reply contains.
    goal_list = _resolve_goals(goals, data_profile, data_type)

    # Under the work root, never beside the input. search_spatial_datasets hands out paths inside the
    # shared dataset library, and a plan rooted at the input's parent wrote the _mcp copy, the QC
    # output and every tool's directory into it -- the derived copies _pick_dataset_h5ad documents
    # contaminating a dataset tree (hunt 2026-09-30, u23-transcriptomics-skills-12). The path hash
    # keeps two datasets that share a basename (every library slide is spatial_transcriptomics.h5ad)
    # out of each other's directories.
    from spatialomicsgym.paths import tool_output_root

    src = Path(h5ad_path)
    run_key = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()[:8]
    output_base = tool_output_root(os.path.join("analysis_results", f"{src.stem}_{run_key}"))

    workflow: dict[str, Any] = {
        "status": "success",
        "h5ad_path": h5ad_path,
        "data_type": data_type,
        "goals": goal_list,
        "steps": [],
    }

    step_num = 0

    # Data-flow annotations (id / depends_on / consumes / produces) say which step feeds which.
    # They are descriptions of the hand-off, never paths -- concrete paths live in each step's
    # params, and the goals still plan from the caller's h5ad exactly as before.
    prep_id: str | None = None

    # ── Step 0: Data Preparation (if needed) ──────────────────────────
    if not data_profile.get("mcp_ready", False):
        step_num += 1
        prep_id = "prepare_data"
        workflow["steps"].append(
            {
                "step": step_num,
                "id": prep_id,
                "depends_on": [],
                "phase": "data_preparation",
                "action": "Run spatial data pipeline to prepare MCP-compatible h5ad",
                "function": "run_spatial_pipeline",
                "module": "spatialomicsgym.tool.spatial_pipeline",
                # Built from the stem, not str.replace(".h5ad", ...): that also rewrote a directory
                # named "x.h5ad_files", and left "X.H5AD" unchanged -- output_path == input_path.
                "params": {"input_path": h5ad_path, "output_path": os.path.join(output_base, f"{src.stem}_mcp.h5ad")},
                "rationale": "Data needs conversion/repair for MCP tool compatibility.",
                "consumes": "the input h5ad as given",
                "produces": "an mcp-ready h5ad",
            }
        )

    # ── Step 1: QC ────────────────────────────────────────────────────
    step_num += 1
    qc_output = os.path.join(output_base, "qc")
    workflow["steps"].append(
        {
            "step": step_num,
            "id": "qc",
            "depends_on": [prep_id] if prep_id else [],
            "phase": "quality_control",
            "action": "Run transcriptomics QC",
            "function": "run_transcriptomics_qc",
            "module": "spatialomicsgym.tool.transcriptomics_skills",
            "params": {"h5ad_path": h5ad_path, "output_dir": qc_output, "data_type": data_type},
            "rationale": "Quality control to filter low-quality spots/cells before analysis.",
            "consumes": "an mcp-ready h5ad" if prep_id else "the input h5ad as given",
            "produces": "qc metrics and keep-or-drop verdicts for spots and genes",
        }
    )

    # ── Step 2+: Analysis per goal ────────────────────────────────────
    # A goal with no runnable tool contributes no step. Record why, so a plan that is QC and
    # nothing else cannot read as the "complete, ordered workflow" this function documents.
    unplannable: dict[str, str] = {}
    for goal in goal_list:
        tools = _recommend_for_goal(goal, data_profile)
        if not tools:
            unplannable[goal] = _why_a_goal_has_no_tools(goal, data_profile)
            continue

        # Pick top 2 tools for comparison (or top 1 for niche tasks)
        n_tools = 2 if goal in ("spatial_clustering", "svg_detection", "deconvolution") else 1
        selected = tools[:n_tools]

        needs_reference = goal == "deconvolution" and data_profile.get("has_sc_reference")
        goal_step_ids: list[str] = []
        for tool_info in selected:
            step_num += 1
            tool_key = tool_info["tool_key"]
            mcp_tool = _MCP_TOOLS[tool_key]

            tool_output = os.path.join(output_base, goal, tool_key)
            params = _build_tool_params(tool_key, h5ad_path, tool_output, reference_h5ad_path, data_profile)

            step_id = f"{goal}.{tool_key}"
            goal_step_ids.append(step_id)
            workflow["steps"].append(
                {
                    "step": step_num,
                    "id": step_id,
                    "depends_on": ["qc"],
                    "phase": "analysis",
                    "goal": goal,
                    "action": f"Run {mcp_tool['full_name']} for {_TASK_TYPES.get(goal, {}).get('name', goal)}",
                    "mcp_tool": tool_key,
                    "mcp_function": mcp_tool["mcp_function"],
                    "params": params,
                    **({"param_notes": mcp_tool["param_notes"]} if mcp_tool.get("param_notes") else {}),
                    "rationale": tool_info.get("reason", ""),
                    "expected_outputs": _expected_outputs(goal, tool_key, tool_output),
                    "consumes": "the qc-checked h5ad" + (" plus the single-cell reference" if needs_reference else ""),
                    "produces": f"{goal} output files from {tool_key}",
                }
            )

        # Evaluation step for multi-tool comparison
        if len(selected) > 1:
            step_num += 1
            workflow["steps"].append(
                {
                    "step": step_num,
                    "id": f"{goal}.compare",
                    "depends_on": goal_step_ids,
                    "phase": "evaluation",
                    "goal": goal,
                    # No wired tool does this: the step named generate_benchmark_report on the "eval"
                    # server, which no config wires, so a model following the plan called a function
                    # that is not in scope (hunt 2026-09-30, the same defect as u13k1-knowhow-14).
                    "action": f"Compare {goal} results across tools in Python (no comparison tool is wired)",
                    "function": None,
                    "mcp_tool": None,
                    "params": {"results_dir": os.path.join(output_base, goal), "task_type": goal},
                    "rationale": f"Compare tool performance using {_TASK_TYPES.get(goal, {}).get('evaluation_metric', 'standard metrics')}.",
                    "consumes": "every tool's output files for this goal",
                    "produces": "a comparison report naming the better-performing tool",
                }
            )

    if unplannable:
        workflow["unplannable_goals"] = unplannable

    workflow["total_steps"] = step_num
    workflow["execution_note"] = (
        "Run steps in depends_on order: a step reads what the steps it depends_on produced. "
        "Steps whose depends_on sets are disjoint may run in either order."
    )

    return json.dumps(workflow, indent=2)


def run_transcriptomics_qc(
    h5ad_path: str,
    output_dir: str = "./qc_results",
    data_type: str = "auto",
    min_counts: int | None = None,
    min_genes: int | None = None,
    max_pct_mt: float | None = None,
) -> str:
    """Run standard QC pipeline on any transcriptomics h5ad.

    Computes and filters based on:
      - Total counts per cell/spot
      - Number of genes detected per cell/spot
      - Mitochondrial gene percentage
      - Ribosomal gene percentage
      - For spatial: spatial distribution of QC metrics
      - Doublet scores (scRNA only, via scrublet)

    Produces QC plots and a filtered h5ad file.

    Args:
        h5ad_path: Path to input h5ad file.
        output_dir: Directory to save QC results and filtered h5ad.
        data_type: 'spatial', 'single_cell', 'bulk', or 'auto'.
        min_counts: Keep cells/spots with at least this many counts. None = adaptive (the data's
            1st percentile, with a floor that never exceeds a tenth of the median).
        min_genes: Keep cells/spots with at least this many detected genes. None = adaptive.
        max_pct_mt: Keep cells/spots with at most this mitochondrial %. None = adaptive.

    Returns:
        str: JSON report with QC metrics, filtering thresholds, number of
             cells/spots removed, and paths to outputs.

    """
    import warnings

    # Scoped, not process-wide: a bare filterwarnings("ignore") stayed installed after the call
    # and silenced every later warning in the agent or server process, for every later turn and
    # account (hunt 2026-09-30, u23-transcriptomics-skills-9).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _run_transcriptomics_qc(h5ad_path, output_dir, data_type, min_counts, min_genes, max_pct_mt)


def _run_transcriptomics_qc(
    h5ad_path: str,
    output_dir: str,
    data_type: str,
    min_counts_override: int | None,
    min_genes_override: int | None,
    max_pct_mt_override: float | None,
) -> str:
    """The body of :func:`run_transcriptomics_qc`, run inside its warnings scope."""
    try:
        import anndata as ad
        import numpy as np
        import scanpy as sc
    except ImportError as e:
        return json.dumps({"status": "error", "message": f"Missing dependency: {e}"}, indent=2)

    p = Path(h5ad_path)
    if not p.exists():
        return json.dumps({"status": "error", "message": f"File not found: {h5ad_path}"}, indent=2)

    os.makedirs(output_dir, exist_ok=True)

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to read h5ad: {e}"}, indent=2)

    n_obs_before = adata.n_obs
    n_vars_before = adata.n_vars

    if n_obs_before == 0 or n_vars_before == 0:
        # np.median/np.mean over an empty axis returns NaN, which json.dumps writes as the bare token
        # `NaN` — invalid JSON that breaks any downstream parse. Reject empty input up front.
        return json.dumps(
            {
                "status": "error",
                "message": f"AnnData is empty ({n_obs_before} obs x {n_vars_before} vars); nothing to QC.",
            },
            indent=2,
        )

    # Auto-detect data type
    if data_type == "auto":
        data_type = "spatial" if "spatial" in adata.obsm else "single_cell"

    # ── Compute QC metrics ────────────────────────────────────────────
    # Mitochondrial genes
    # astype(str) first: integer var_names (a valid AnnData state) have no .str accessor and yield
    # all-NaN mt/ribo/hb masks that can break calculate_qc_metrics.
    _vn = adata.var_names.astype(str).str.upper()
    adata.var["mt"] = _vn.str.startswith("MT-")
    # Ribosomal genes
    adata.var["ribo"] = _vn.str.match(r"^RP[SL]\d")
    # Hemoglobin genes
    adata.var["hb"] = _vn.str.match(r"^HB[^(P)]")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], percent_top=None, log1p=True, inplace=True)

    qc_stats: dict[str, Any] = {
        "total_counts": {
            "median": float(np.median(adata.obs["total_counts"])),
            "mean": float(np.mean(adata.obs["total_counts"])),
            "min": float(np.min(adata.obs["total_counts"])),
            "max": float(np.max(adata.obs["total_counts"])),
        },
        "n_genes_by_counts": {
            "median": float(np.median(adata.obs["n_genes_by_counts"])),
            "mean": float(np.mean(adata.obs["n_genes_by_counts"])),
            "min": float(np.min(adata.obs["n_genes_by_counts"])),
            "max": float(np.max(adata.obs["n_genes_by_counts"])),
        },
        "pct_counts_mt": {
            "median": float(np.median(adata.obs["pct_counts_mt"])),
            "mean": float(np.mean(adata.obs["pct_counts_mt"])),
            "max": float(np.max(adata.obs["pct_counts_mt"])),
        },
        "pct_counts_ribo": {
            "median": float(np.median(adata.obs["pct_counts_ribo"])),
            "mean": float(np.mean(adata.obs["pct_counts_ribo"])),
        },
    }

    # ── QC Plots ──────────────────────────────────────────────────────
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        sc.pl.violin(adata, ["total_counts"], ax=axes[0, 0], show=False)
        axes[0, 0].set_title("Total Counts")
        sc.pl.violin(adata, ["n_genes_by_counts"], ax=axes[0, 1], show=False)
        axes[0, 1].set_title("Genes Detected")
        sc.pl.violin(adata, ["pct_counts_mt"], ax=axes[1, 0], show=False)
        axes[1, 0].set_title("% Mitochondrial")
        sc.pl.violin(adata, ["pct_counts_ribo"], ax=axes[1, 1], show=False)
        axes[1, 1].set_title("% Ribosomal")
        fig.suptitle(f"QC Metrics ({data_type}): {Path(h5ad_path).stem}", fontsize=14)
        plt.tight_layout()
        qc_violin_path = os.path.join(output_dir, "qc_violin.png")
        plt.savefig(qc_violin_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Scatter: total_counts vs n_genes, colored by pct_mt
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            adata.obs["total_counts"],
            adata.obs["n_genes_by_counts"],
            c=adata.obs["pct_counts_mt"],
            cmap="RdYlGn_r",
            s=3,
            alpha=0.5,
        )
        plt.colorbar(scatter, label="% MT")
        ax.set_xlabel("Total Counts")
        ax.set_ylabel("Genes Detected")
        ax.set_title("QC Scatter")
        qc_scatter_path = os.path.join(output_dir, "qc_scatter.png")
        plt.savefig(qc_scatter_path, dpi=150, bbox_inches="tight")
        plt.close()

        # Spatial QC if applicable
        qc_spatial_path = None
        if data_type == "spatial" and "spatial" in adata.obsm:
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for idx, metric in enumerate(["total_counts", "n_genes_by_counts", "pct_counts_mt"]):
                    coords = adata.obsm["spatial"]
                    scatter = axes[idx].scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=adata.obs[metric],
                        cmap="viridis" if idx < 2 else "RdYlGn_r",
                        s=5,
                        alpha=0.7,
                    )
                    plt.colorbar(scatter, ax=axes[idx])
                    axes[idx].set_title(metric)
                    axes[idx].set_aspect("equal")
                    axes[idx].invert_yaxis()
                fig.suptitle("Spatial QC Distribution", fontsize=14)
                plt.tight_layout()
                qc_spatial_path = os.path.join(output_dir, "qc_spatial.png")
                plt.savefig(qc_spatial_path, dpi=150, bbox_inches="tight")
                plt.close()
            except Exception:
                pass
    except Exception:
        qc_violin_path = None
        qc_scatter_path = None
        qc_spatial_path = None

    # ── Compute adaptive filtering thresholds ─────────────────────────
    # The floors were fixed scRNA-seq numbers (100 counts, 50 genes) and the "spatial" branch that
    # meant to lower them applied max(), which can only raise: on a 140-gene MERFISH/Xenium panel
    # (median ~85 counts, ~64 genes) 94% of cells were removed, and on a sparser one all of them,
    # under status "success" (hunt 2026-09-30, u23-transcriptomics-skills-8). The spatial floors
    # are now the lower ones the branch intended, and no floor exceeds a tenth of the median --
    # which leaves transcriptome-wide data (medians in the thousands) exactly where it was. A cell
    # with no counts at all is still dropped.
    count_floor, gene_floor, mt_cap = (50, 30, 30.0) if data_type == "spatial" else (100, 50, 25.0)
    count_floor = max(1, min(count_floor, int(0.1 * float(np.median(adata.obs["total_counts"])))))
    gene_floor = max(1, min(gene_floor, int(0.1 * float(np.median(adata.obs["n_genes_by_counts"])))))
    min_counts = max(int(np.percentile(adata.obs["total_counts"], 1)), count_floor)
    min_genes = max(int(np.percentile(adata.obs["n_genes_by_counts"], 1)), gene_floor)
    max_pct_mt = min(float(np.percentile(adata.obs["pct_counts_mt"], 95)), mt_cap)
    # A caller's value replaces the adaptive one outright.
    if min_counts_override is not None:
        min_counts = int(min_counts_override)
    if min_genes_override is not None:
        min_genes = int(min_genes_override)
    if max_pct_mt_override is not None:
        max_pct_mt = float(max_pct_mt_override)

    thresholds = {
        "min_counts": min_counts,
        "min_genes": min_genes,
        "max_pct_mt": round(max_pct_mt, 1),
    }

    # ── Apply filters ─────────────────────────────────────────────────
    adata_filtered = adata[
        (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["pct_counts_mt"] <= max_pct_mt)
    ].copy()

    # Filter genes: require at least 3 cells/spots
    sc.pp.filter_genes(adata_filtered, min_cells=3)

    n_obs_after = adata_filtered.n_obs
    n_vars_after = adata_filtered.n_vars

    # Nothing left is a failed QC, not a result: an empty h5ad written under status "success" was
    # handed to the next step as the filtered dataset (u23-transcriptomics-skills-8).
    if n_obs_after == 0 or n_vars_after == 0:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"QC removed everything ({n_obs_before} -> {n_obs_after} cells/spots, "
                    f"{n_vars_before} -> {n_vars_after} genes) at min_counts={min_counts}, "
                    f"min_genes={min_genes}, max_pct_mt={round(max_pct_mt, 1)}; no filtered h5ad was "
                    "written. Lower min_counts / min_genes or raise max_pct_mt and run again."
                ),
                "data_type": data_type,
                "qc_metrics": qc_stats,
                "thresholds_applied": thresholds,
            },
            indent=2,
        )

    # ── Save filtered h5ad ────────────────────────────────────────────
    # Written beside its final name and moved into place, so a reader never sees a half-written file.
    filtered_path = os.path.join(output_dir, Path(h5ad_path).stem + "_qc_filtered.h5ad")
    partial_path = filtered_path[: -len(".h5ad")] + ".partial.h5ad"
    adata_filtered.write(partial_path)
    os.replace(partial_path, filtered_path)

    # ── Build report ──────────────────────────────────────────────────
    report: dict[str, Any] = {
        "status": "success",
        "data_type": data_type,
        "input": {"n_cells_spots": n_obs_before, "n_genes": n_vars_before},
        "qc_metrics": qc_stats,
        "thresholds_applied": thresholds,
        "filtering_result": {
            "n_cells_spots_after": n_obs_after,
            "n_genes_after": n_vars_after,
            "cells_spots_removed": n_obs_before - n_obs_after,
            "genes_removed": n_vars_before - n_vars_after,
            "pct_cells_spots_removed": round(100 * (n_obs_before - n_obs_after) / max(n_obs_before, 1), 1),
        },
        "output_files": {
            "filtered_h5ad": filtered_path,
            "qc_violin": qc_violin_path,
            "qc_scatter": qc_scatter_path,
            "qc_spatial": qc_spatial_path,
        },
        "recommendations": _qc_recommendations(qc_stats, data_type, n_obs_before, n_obs_after),
    }

    return json.dumps(report, indent=2)


def prepare_reference_data(
    sc_h5ad_path: str,
    output_path: str = "./reference_prepared.h5ad",
    labels_key: str = "CellType",
    batch_key: str = "Sample",
    min_cells_per_type: int = 10,
) -> str:
    """Prepare a scRNA-seq reference h5ad for use with deconvolution MCP tools.

    Validates and prepares the reference by:
      - Checking for required cell type annotations
      - Filtering rare cell types (< min_cells_per_type)
      - Computing QC metrics
      - Ensuring raw counts are available
      - Reporting cell type composition

    The prepared reference can be used with Cell2Location, Tangram, TACCO,
    STRIDE, and other deconvolution MCP tools.

    Args:
        sc_h5ad_path: Path to scRNA-seq reference h5ad file.
        output_path: Path to save prepared reference.
        labels_key: Column in .obs with cell type labels.
        batch_key: Column in .obs with batch/sample info.
        min_cells_per_type: Minimum cells per type to retain.

    Returns:
        str: JSON report with cell type summary, QC, and compatibility.

    """
    try:
        import anndata as ad
        import numpy as np
        import scanpy as sc
    except ImportError as e:
        return json.dumps({"status": "error", "message": f"Missing dependency: {e}"}, indent=2)

    p = Path(sc_h5ad_path)
    if not p.exists():
        return json.dumps({"status": "error", "message": f"File not found: {sc_h5ad_path}"}, indent=2)

    try:
        adata = ad.read_h5ad(sc_h5ad_path)
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Failed to read h5ad: {e}"}, indent=2)

    report: dict[str, Any] = {
        "status": "success",
        "input_path": sc_h5ad_path,
        "n_cells_input": adata.n_obs,
        "n_genes_input": adata.n_vars,
    }

    # Check for labels
    if labels_key not in adata.obs.columns:
        avail = [c for c in adata.obs.columns if "type" in c.lower() or "label" in c.lower() or "cell" in c.lower()]
        return json.dumps(
            {
                "status": "error",
                "message": f"Labels column '{labels_key}' not found in obs. Available candidates: {avail}",
                "all_obs_columns": list(adata.obs.columns),
            },
            indent=2,
        )

    # Cell type composition
    type_counts = adata.obs[labels_key].value_counts()
    report["cell_types_found"] = len(type_counts)
    report["cell_type_composition"] = {str(k): int(v) for k, v in type_counts.items()}

    # Filter rare cell types
    keep_types = type_counts[type_counts >= min_cells_per_type].index
    removed_types = type_counts[type_counts < min_cells_per_type]
    if len(removed_types) > 0:
        report["removed_cell_types"] = {str(k): int(v) for k, v in removed_types.items()}
        report["removal_reason"] = f"Fewer than {min_cells_per_type} cells"
        adata = adata[adata.obs[labels_key].isin(keep_types)].copy()

    # Check batch info
    has_batch = batch_key in adata.obs.columns
    report["has_batch_info"] = has_batch
    if has_batch:
        report["n_batches"] = adata.obs[batch_key].nunique()

    # If the rare-type filter dropped every cell (all types below the threshold), QC medians become
    # NaN -> invalid JSON (and calculate_qc_metrics can error on an empty AnnData). Fail clearly instead.
    if adata.n_obs == 0:
        return json.dumps(
            {
                "status": "error",
                "message": f"No cells remain after dropping cell types with < {min_cells_per_type} cells; "
                "lower min_cells_per_type.",
            },
            indent=2,
        )

    # QC. astype(str) first: integer var_names (a valid AnnData state) have no .str accessor and give an
    # all-NaN 'mt' mask.
    adata.var["mt"] = adata.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    report["qc_summary"] = {
        "median_counts": float(np.median(adata.obs["total_counts"])),
        "median_genes": float(np.median(adata.obs["n_genes_by_counts"])),
        "median_pct_mt": float(np.median(adata.obs["pct_counts_mt"])),
    }

    # Ensure raw counts. Integer-valued, not "large": X.max() > 100 called a CPM/TPM reference raw,
    # and the agent handed a normalised matrix to count-based deconvolution (hunt 2026-09-30,
    # u23-transcriptomics-skills-11). The agent core's probe is the one _profile_h5ad uses.
    if adata.raw is not None:
        report["raw_counts"] = "available in .raw"
    else:
        from spatialomicsgym.agent.data_validation import _has_raw_counts

        if _has_raw_counts(adata):
            report["raw_counts"] = "X holds non-negative integers (raw counts)"
        else:
            report["raw_counts"] = (
                "WARNING: X is not integer counts (normalised, e.g. CPM/TPM or log) — "
                "deconvolution tools need raw counts"
            )

    # Save
    os.makedirs(str(Path(output_path).parent), exist_ok=True)
    adata.write(output_path)
    report["output_path"] = output_path
    report["n_cells_output"] = adata.n_obs
    report["n_genes_output"] = adata.n_vars

    # MCP tool compatibility: callable names and the parameter names each portal declares. This
    # used to pass sc_h5ad_path to tangram/tacco/stride, whose parameter is sc_h5ad, under row keys
    # that are not functions, and left out the label column tacco and stride require (hunt
    # 2026-09-30, u23-transcriptomics-skills-10).
    compatible = []
    for row_key in ("cell2location", "tangram", "tacco", "stride"):
        fn = _MCP_TOOLS[row_key]["mcp_function"]
        ref_param = _declared_input_param(fn, _REFERENCE_PARAM_NAMES) or "sc_h5ad"
        call: dict[str, Any] = {ref_param: output_path}
        declared = _DECLARED_TOOL_PARAMS.get(fn, {})
        label_param = next((p for p in ("labels_key", "annotation_key") if p in declared), None)
        if label_param:
            call[label_param] = labels_key
        if "batch_key" in declared:
            call["batch_key"] = batch_key if has_batch else ""
        compatible.append({"tool": row_key, "mcp_function": fn, "params": call})
    report["compatible_mcp_tools"] = compatible

    return json.dumps(report, indent=2)


def list_available_mcp_tools(task_type: str = "all") -> str:
    """List the MCP spatial analysis tools this agent can call, optionally filtered by task.

    The answer comes from two tables that cover different amounts of the registry, and both are
    returned, because the caller is asking what it can run and a curated profile is a property of
    our documentation rather than of the tool:

    * ``task_types`` -- the hand-curated ``_MCP_TOOLS`` x ``_TASK_TYPES`` join, carrying strengths,
      limitations, best_for and input requirements. This is a minority of the registry.
    * ``also_registered`` -- everything in ``MCP_server/mcp_config.yaml`` (the registration gate,
      and so the definitive set of callable functions) that the section above has not already
      shown, grouped by the task the skills catalog assigns it. A tool the curated table files
      under a *different* task name arrives here with its full profile plus ``curated_under``
      naming that other spelling; the rest carry name, full name and one-line description, under
      a group note saying so.

    ``total_tools`` counts both. Reporting only the curated half under a heading saying "all" is
    what made 70 callable tools invisible to a model probing for what it could run.

    Args:
        task_type: A task to filter by, or 'all'. Every key of ``_TASK_TYPES`` and every task the
                   skills catalog assigns to a registered tool is accepted, and ``known_task_types``
                   in the reply lists them. An unrecognised value sets ``unknown_task_type`` rather
                   than returning an empty listing, so a typo cannot read as "no such tools".

    Returns:
        str: JSON with categorized tool listings.

    """
    skills_rows = _skills_rows_by_function()
    # Only tasks that a *callable* tool carries. The catalog also describes rows whose function was
    # never registered, and offering their task here advertised three names -- the literature ones --
    # that filtered to nothing, in the same reply that points the caller at this list as the set of
    # names that would have worked.
    visible = _visible_mcp_tools()
    known_tasks = set(_TASK_TYPES) | {r["task"] for fn, r in skills_rows.items() if fn in visible and r.get("task")}

    result: dict[str, Any] = {"status": "success", "task_types": {}}

    for task_key, task_info in _TASK_TYPES.items():
        if task_type != "all" and task_key != task_type:
            continue

        tools_in_task = []
        for tool_key, tool_data in sorted(_MCP_TOOLS.items(), key=lambda x: x[1].get("priority", 99)):
            if tool_data["task"] == task_key:
                tools_in_task.append(_curated_listing_row(tool_key, tool_data))

        if tools_in_task:
            result["task_types"][task_key] = {
                "name": task_info["name"],
                "description": task_info["description"],
                "evaluation_metric": task_info.get("evaluation_metric", ""),
                "n_tools": len(tools_in_task),
                "tools": tools_in_task,
            }

    # What this reply has shown under a curated heading so far -- NOT the same question as "what has
    # a curated profile", because the loop above is filtered by task_type. The two coincide only on
    # 'all'; keeping them apart is what stops the branch below denying a profile it is holding.
    shown_curated = {t["mcp_function"] for group in result["task_types"].values() for t in group["tools"]}
    curated_by_function = {data["mcp_function"]: (key, data) for key, data in _MCP_TOOLS.items()}

    # The rest of the registry. Grouped by the skills catalog's task so a filtered query reaches it
    # too: of the tasks that catalog assigns to a registered tool, twelve are not keys of _TASK_TYPES,
    # and asking for one of them used to return the same empty listing as asking for a string that is
    # not a task at all. Twelve tools are curated under one of those other spellings, so a query that
    # arrives here can still be about a tool we have a full profile for -- serve the profile, and say
    # which task name it is filed under, rather than reporting it as undocumented.
    also: dict[str, dict[str, Any]] = {}
    brief: set[str] = set()
    curated_here: set[str] = set()
    for name in sorted(visible - shown_curated):
        row = skills_rows.get(name, {})
        task_of = row.get("task") or "uncategorised"
        if task_type != "all" and task_of != task_type:
            continue
        group = also.setdefault(
            task_of,
            {
                "n_tools": 0,
                "note": "Callable, but no curated profile exists -- name and summary only.",
                "tools": [],
            },
        )
        if name in curated_by_function:
            tool_key, tool_data = curated_by_function[name]
            entry = _curated_listing_row(tool_key, tool_data)
            entry["task"] = task_of
            entry["curated_under"] = tool_data["task"]
            curated_here.add(name)
        else:
            entry = {
                "mcp_function": name,
                "full_name": row.get("full_name", name),
                "description": row.get("description", ""),
                "task": task_of,
            }
            brief.add(name)
        group["tools"].append(entry)
        group["n_tools"] += 1

    # The note is read as a statement about every tool under it, so derive it from what each group
    # ended up holding. Written after the loop rather than per-append to keep the JSON key order.
    # A group that holds a profile must not carry the blanket denial, not even as a trailing clause:
    # a mixed group describes both kinds instead.
    for group in also.values():
        n_curated = sum(1 for t in group["tools"] if "curated_under" in t)
        if n_curated == group["n_tools"]:
            group["note"] = "Curated under a different task name -- the full profile is included, see curated_under."
        elif n_curated:
            group["note"] = (
                f"{n_curated} of these carry a curated profile filed under a different task name (see "
                f"curated_under); the other {group['n_tools'] - n_curated} are listed by name and "
                "one-line description."
            )

    if also:
        result["also_registered"] = also
    result["n_profiled"] = len(shown_curated | curated_here)
    result["n_brief"] = len(brief)
    result["total_tools"] = len(shown_curated | curated_here | brief)
    result["known_task_types"] = sorted(known_tasks)

    if task_type != "all" and task_type not in known_tasks:
        result["unknown_task_type"] = True
        result["message"] = (
            f"{task_type!r} is not a task type this agent uses, so nothing was filtered against it. "
            "An empty listing here means the name was not recognised, not that no such tools exist "
            "-- see known_task_types."
        )
    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _curated_listing_row(tool_key: str, tool_data: dict[str, Any]) -> dict[str, Any]:
    """One tool's curated listing entry, as ``list_available_mcp_tools`` serves it.

    Shared by both branches of that listing so that a filtered reply cannot hand back a thinner
    copy of a profile the process is holding. ``needs_images`` and ``needs_sc_reference`` are the
    fields a caller rules a tool out on, so losing them is not a cosmetic difference.
    """
    return {
        "tool_key": tool_key,
        "full_name": tool_data["full_name"],
        "mcp_function": tool_data["mcp_function"],
        "description": tool_data["description"],
        "strengths": tool_data["strengths"],
        "limitations": tool_data["limitations"],
        "best_for": tool_data["best_for"],
        "needs_gpu": tool_data.get("gpu", False),
        "needs_sc_reference": tool_data["input_requirements"].get("sc_reference", False),
        "needs_images": tool_data["input_requirements"].get("images", False),
    }


#: Spread of the nearest-neighbour spacing, (q90 - q10) / median, below which spots sit on a
#: lattice. Calibrated on this box: every Visium slide in the dataset library and the benchmark
#: measures 0.000-0.046, the two Slide-seqV2 pucks 0.49 and 1.65, MERFISH about 1.0, Xenium 0.43.
_LATTICE_SPREAD = 0.2


def _spot_layout(coords: Any) -> tuple[bool, int] | None:
    """(positions sit on a lattice, equidistant nearest neighbours per spot), or None if unreadable.

    Unit-free, so it reads pixels and microns alike: a Visium slide is a hexagonal lattice (six
    neighbours at one spacing), a binned Visium HD or Stereo-seq slide a square one (four), and
    Slide-seq beads are packed irregularly. Like the raw-counts probe it is a bounded read -- the
    tree holds every spot; at most 20,000 evenly strided spots are asked for their neighbours.
    """
    try:
        import numpy as np
        from scipy.spatial import cKDTree

        xy = np.asarray(coords, dtype=float)
        if xy.ndim != 2 or xy.shape[1] < 2:
            return None
        xy = xy[:, :2]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) < 50:
            return None
        dist, _ = cKDTree(xy).query(xy[:: max(1, len(xy) // 20000)], k=7)
        dist = dist[dist[:, 1] > 0]
        if not len(dist):
            return None
        q10, med, q90 = np.quantile(dist[:, 1], [0.1, 0.5, 0.9])
        if med <= 0:
            return None
        n_equidistant = int(np.median((dist[:, 1:] <= med * 1.1).sum(axis=1)))
        return bool((q90 - q10) / med < _LATTICE_SPREAD), n_equidistant
    except Exception:
        return None


def _profile_h5ad(h5ad_path: str) -> dict[str, Any]:
    """Profile an h5ad file for tool recommendation."""
    try:
        # Availability probe only -- the read itself goes through read_h5ad_backed. Kept as a real
        # import rather than find_spec so that a present-but-broken anndata is caught here too,
        # where it becomes an error dict, instead of raising out of the recommender.
        import anndata  # noqa: F401
    except ImportError:
        return {"status": "error", "message": "anndata not installed"}

    # ExitStack, rather than a plain ``with``, so the read keeps its own ``except`` -- an unreadable
    # file must still return the error dict rather than raise into the recommender. The context
    # manager releases the HDF5 handle on every exit below; dropping the local is not enough,
    # because an input with a .raw slot is cyclic and would stay locked until the GC ran.
    with contextlib.ExitStack() as stack:
        try:
            adata = stack.enter_context(read_h5ad_backed(h5ad_path))
        except Exception as e:
            return {"status": "error", "message": f"Failed to read: {e}"}

        profile: dict[str, Any] = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "has_spatial": "spatial" in adata.obsm,
            "has_images": False,
            "has_qc": "total_counts" in adata.obs.columns,
            "var_names_unique": adata.var_names.is_unique,
            "has_obs_xy": "x" in adata.obs.columns and "y" in adata.obs.columns,
        }

        # Check for images. Agent-written h5ads sometimes store uns['spatial'] as an array/list/string
        # rather than the Visium mapping, so guard the .items() iteration — an unguarded call raised
        # AttributeError out of _profile_h5ad -> recommend_analysis_tools / build_analysis_workflow.
        uns_spatial = adata.uns.get("spatial")
        if isinstance(uns_spatial, dict):
            for _lib_id, lib_data in uns_spatial.items():
                if isinstance(lib_data, dict) and lib_data.get("images"):
                    profile["has_images"] = True
                    break

        # Determine data type
        if profile["has_spatial"]:
            profile["data_type"] = "spatial"
        else:
            # Check for common scRNA markers
            profile["data_type"] = "single_cell" if adata.n_obs > 100 else "bulk"

        # Data scale classification
        if adata.n_obs < 5000:
            profile["scale"] = "small"
        elif adata.n_obs < 50000:
            profile["scale"] = "medium"
        else:
            profile["scale"] = "large"

        # Platform inference — heuristic but explicit
        #   Visium: embedded H&E images in uns['spatial'], or none but a hexagonal spot lattice
        #   MERFISH: panel-size var_names (<2000) and no embedded images
        #   Slide-seqV2: transcriptome-wide var_names (>=5000), no embedded images, irregular beads
        #   else: unknown
        # The gene count alone used to decide Slide-seqV2, though the comment promised a look at the
        # spots: a Visium h5ad whose images were stripped in conversion, a Visium HD or a Stereo-seq
        # bin table was ranked on the Slide-seqV2 leaderboard row and told "Empirically validated on
        # Slide-seqV2" (hunt 2026-09-30, u23-transcriptomics-skills-28). The platform picks that
        # row, so it is now named only on positive evidence and is "unknown" otherwise. No slide in
        # this box's dataset library or benchmark changes label: all carry images or are Slide-seqV2.
        platform = "unknown"
        if profile["has_spatial"]:
            if profile["has_images"]:
                platform = "Visium"
            elif adata.n_vars < 2000:
                platform = "MERFISH"
            elif adata.n_vars >= 5000:
                layout = _spot_layout(adata.obsm["spatial"])
                if layout is not None and not layout[0]:
                    platform = "Slide-seqV2"
                elif layout is not None and layout[1] == 6:
                    platform = "Visium"
        profile["platform"] = platform

        # Raw counts available — needed by all deconv tools and SVG tools that compute on counts.
        # Heuristic: integer dtype in X, OR a 'counts' layer, OR all-integer sample of X.
        raw_counts = False
        try:
            x_dtype = adata.X.dtype
            if x_dtype.kind in ("i", "u"):
                raw_counts = True
            elif "counts" in adata.layers:
                raw_counts = True
            else:
                # This read goes through backed="r", so adata.X is an anndata _CSRDataset/_CSCDataset
                # and slicing it directly (`adata.X[:5, :100]`) raises AttributeError on modern
                # anndata -- swallowed by the except below, leaving raw_counts False. That made
                # visium_dir_convertible False and hard-dropped somde_run / spatialde_run_svg from
                # every recommendation over a *sparse* h5ad, i.e. essentially all real spatial data.
                # Delegate to the agent-core probe, which does the bounded read of the backing
                # indptr/data/indices group. Lazy import: agent imports tool modules at load.
                from spatialomicsgym.agent.data_validation import _has_raw_counts

                raw_counts = _has_raw_counts(adata)
        except Exception:
            pass
        profile["raw_counts_available"] = raw_counts

        # Visium-directory convertibility: the data_converter MCP can repack a Visium h5ad
        # (with embedded scalefactors and raw counts) into the filtered_feature_bc_matrix.h5 +
        # spatial/ directory layout that somde / spatialde / prost / spark require.
        # has_images too: a Visium slide known only by its lattice has no image to repack.
        profile["visium_dir_convertible"] = (
            platform == "Visium" and profile["has_images"] and raw_counts and "spatial" in adata.uns
        )

        # MCP readiness
        profile["mcp_ready"] = (
            profile["has_spatial"] and profile["has_qc"] and profile["var_names_unique"] and adata.X is not None
        )

        return profile


def _resolve_goals(requested: Any, data_profile: dict, data_type: str) -> list[str]:
    """The goal list a planner resolves its argument to -- shared, because the two copies drifted.

    ``recommend_analysis_tools`` ends its reply with the goals it resolved, as a JSON list under
    ``analysis_goals``, next to a hint telling the model to call ``build_analysis_workflow``. So the
    obvious next call passes that list to the parameter actually named ``goals``. Only the
    recommender had learned to take a sequence; the planner still parsed with a bare
    ``str.split(",")``, which raised AttributeError on the list, and read the literal "all" -- a
    value the recommender's own advertised description offers -- as a task name nothing matches,
    producing a "complete, ordered analysis workflow" with no analysis step in it.

    Accepts a sequence or the documented comma-separated string; "auto", "all", and an empty
    request all mean "infer from the data".
    """
    if isinstance(requested, (list, tuple, set)):
        goals = [str(g).strip() for g in requested if str(g).strip()]
    else:
        goals = [g.strip() for g in str(requested).split(",") if g.strip()]

    if not goals or goals == ["auto"] or goals == ["all"]:
        return _infer_analysis_goals(data_profile, data_type)
    return goals


def _infer_analysis_goals(data_profile: dict, data_type: str) -> list[str]:
    """Infer suitable analysis goals from the data profile."""
    goals = []
    if data_type == "spatial":
        goals.append("spatial_clustering")
        goals.append("svg_detection")
        if data_profile.get("n_obs", 0) < 100000:
            goals.append("deconvolution")
        goals.append("spatial_communication")
    elif data_type == "single_cell":
        goals.append("spatial_clustering")  # clustering still applies
    return goals


# A requirement a tool declares in ``input_requirements`` -> the ``_profile_h5ad`` key that satisfies
# it. The hard-drop in ``_recommend_for_goal`` and the explanation in ``_why_a_goal_has_no_tools`` both
# read this one mapping, so a goal can never be dropped for a reason the reply is unable to name.
# ``h5ad`` is deliberately absent: no tool has ever been dropped on it.
_REQUIREMENT_PROFILE_KEY: dict[str, str] = {
    "spatial_coords": "has_spatial",
    "images": "has_images",
    "sc_reference": "has_sc_reference",
    "visium_dir": "visium_dir_convertible",
    "obs_xy": "has_obs_xy",
}


def _why_a_goal_has_no_tools(goal: str, data_profile: dict) -> str:
    """Say why a goal yielded no candidate tool, in terms the caller can act on.

    ``_recommend_for_goal`` returns an empty list for two unrelated reasons -- the goal is not one of
    the curated task types, or it is one and every tool for it needs an input this dataset does not
    have -- and both callers used to express either as a bare ``continue``. The goal then vanished
    from the reply while ``status`` still read ``success`` and the goal was still echoed under
    ``analysis_goals`` as though it had been accepted.
    """
    if goal not in _TASK_TYPES:
        return (
            f"'{goal}' is not a supported analysis goal, so no tool was considered for it. "
            f"Supported goals: {', '.join(sorted(_TASK_TYPES))}. "
            "Pass the one you meant by name -- analysis_goals='auto' infers only the goals this "
            "dataset itself implies, which is never all of them."
        )

    rows = [t for t in _MCP_TOOLS.values() if t["task"] == goal]
    hidden = _benchmark_hidden_mcp_tools()
    if rows and all(t["mcp_function"] in hidden for t in rows):
        return (
            f"Every tool for '{goal}' is withheld from scored runs (its server is benchmark_visible: "
            "false), so none is offered in this run."
        )

    # Only requirements this dataset actually lacks are listed, so the sentence stays true whether
    # one tool or twelve were ruled out.
    missing: list[str] = []
    for tool_data in rows:
        if tool_data["mcp_function"] in hidden:
            continue
        reqs = tool_data["input_requirements"]
        for req, profile_key in _REQUIREMENT_PROFILE_KEY.items():
            if reqs.get(req) and not data_profile.get(profile_key) and req not in missing:
                missing.append(req)

    if missing:
        return (
            f"Every registered tool for '{goal}' needs an input this dataset does not have. "
            f"Missing: {', '.join(missing)}. Supply it and call again, or choose a different goal."
        )
    return f"No tool is registered for the '{goal}' task type."


def _recommend_for_goal(goal: str, data_profile: dict) -> list[dict[str, Any]]:
    """Get ranked tool recommendations for a specific analysis goal.

    Two-stage ranking:
      1. Hard-drop tools whose input_requirements cannot be satisfied by the
         given data_profile (no spatial coords / no images / no sc_reference /
         no visium-dir convertibility). These tools physically cannot run.
      2. Among compatible candidates, prefer those with empirical leaderboard
         scores for (goal, platform). Empirical-validated tools get a score of
         1000 * mean_score so they dominate the heuristic 10-priority scale;
         un-tested tools keep their heuristic base score.
    """
    # Lazy import to avoid circular dependency at module load.
    try:
        from spatialomicsgym.agent.empirical_leaderboard import lookup_score as _lookup_empirical
    except Exception:
        _lookup_empirical = None

    candidates = []
    platform = data_profile.get("platform", "unknown")
    # A scored run binds no callable for a benchmark_visible: false server, so ranking one there
    # sent the prompt to say the tool "is not installed ... add it with sog-setup" -- false for a
    # tool hidden on purpose. The listing already filtered them; the ranking did not (hunt
    # 2026-09-30, u23-transcriptomics-skills-25).
    hidden = _benchmark_hidden_mcp_tools()

    for tool_key, tool_data in _MCP_TOOLS.items():
        if tool_data["task"] != goal or tool_data["mcp_function"] in hidden:
            continue

        reqs = tool_data["input_requirements"]

        # ── Hard-drop: data physically cannot satisfy tool requirements ──
        # Same five checks as before, read from the shared mapping so _why_a_goal_has_no_tools
        # explains exactly the drops that happen here.
        if any(reqs.get(req) and not data_profile.get(key) for req, key in _REQUIREMENT_PROFILE_KEY.items()):
            continue

        # ── Soft scoring: heuristic base, then empirical override ────────
        base_score = 10 - tool_data.get("priority", 5)
        reasons: list[str] = []

        # Data scale fitness
        scale = data_profile.get("scale", "medium")
        tool_scale = tool_data.get("data_scale", "any")
        if tool_scale == "small" and scale == "large":
            base_score -= 2
            reasons.append("may be slow on large data")

        # GPU note (informational)
        if tool_data.get("gpu") and scale == "large":
            reasons.append("GPU recommended for best performance")

        # Empirical override — use mean score from benchmark cells if available.
        empirical_score = None
        empirical_n = None
        if _lookup_empirical is not None:
            info = _lookup_empirical(goal, platform, tool_data["mcp_function"])
            if info is not None:
                empirical_score = info["score"]
                empirical_n = info["n_cells"]

        if empirical_score is not None:
            # 1000× multiplier so any positive empirical signal dominates the
            # heuristic max (10). Negative empirical scores stay negative and
            # fall below un-tested tools — exactly what we want.
            score = 1000.0 * empirical_score
            reason = (
                f"Empirically validated on {platform}: mean score "
                f"{empirical_score:.3f} over n={empirical_n} benchmark cells"
            )
            if reasons:
                reason += f" (caveats: {'; '.join(reasons)})"
        else:
            score = float(base_score)
            if not reasons:
                reason = f"Well-suited: {', '.join(tool_data['strengths'][:3])}"
            else:
                reason = f"Compatible with caveats: {'; '.join(reasons)}"

        candidates.append(
            {
                "tool_key": tool_key,
                "full_name": tool_data["full_name"],
                "mcp_function": tool_data["mcp_function"],
                "score": score,
                "empirical_score": empirical_score,
                "empirical_n_cells": empirical_n,
                "reason": reason,
                "strengths": tool_data["strengths"],
                "key_params": tool_data.get("key_params", {}),
                **({"param_notes": tool_data["param_notes"]} if tool_data.get("param_notes") else {}),
                "needs_gpu": tool_data.get("gpu", False),
                "needs_sc_reference": reqs.get("sc_reference", False),
                "needs_images": reqs.get("images", False),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    for c in candidates:
        del c["score"]
    return candidates


def _surface_required_inputs(mcp_function: str, params: dict[str, Any]) -> None:
    """Fill *params* with a placeholder for each required parameter the caller could not supply.

    Covers tools whose input is not a single .h5ad slide -- a CSV pair (card, spacexr_rctd,
    mistyr), a GEM file (spotgf), a prepared dataset directory (istar), a TOML config (xfuse) -- and
    the required knobs of the ones whose input is (graphst's n_clusters, tacco's annotation_key).
    The plan cannot fill those in from one .h5ad, and naming a phantom ``st_h5ad`` for them would
    just be discarded, so name what the tool really requires and say it has to be produced first.
    """
    for name, spec in _DECLARED_TOOL_PARAMS.get(mcp_function, {}).items():
        if name in params or not spec.get("required"):
            continue
        desc = " ".join(str(spec.get("description") or "").split())[:90].rstrip(" .")
        params[name] = f"<REQUIRED — {desc}>" if desc else "<REQUIRED — set before running>"


def _build_tool_params(
    tool_key: str, h5ad_path: str, output_dir: str, reference_path: str | None, data_profile: dict
) -> dict[str, Any]:
    """Build the parameter dict for one MCP tool call, under the names its portal declares.

    Every key has to be a parameter the portal really has. FastMCP validates a call against the
    schema it derives from the decorated function's signature and *silently discards* an argument
    that signature has no parameter for, so a guessed name never surfaces as an error -- it produces
    a call that "succeeds" with the input missing. Names therefore come from
    ``MCP_server/mcp_config.yaml`` (the same block ``add_mcp()`` turns into the agent-visible
    schema), not from a convention: this used to hard-code four tools and send every other input
    path to ``st_h5ad``, which is the modal name -- 21 of the configured tools really do use it, so
    it looked right -- and was dropped by the other 27, among them bayestme, card, spacexr_rctd,
    starfysh, svgbit and ucdeconvolve. Pinned by test/test_config_declares_only_real_parameters.py.
    """
    mcp_tool = _MCP_TOOLS[tool_key]
    mcp_function = mcp_tool.get("mcp_function", "")
    params: dict[str, Any] = {}

    if not _DECLARED_TOOL_PARAMS.get(mcp_function):
        # No config to read (missing file, no PyYAML) or a tool absent from it: fall back to the
        # historical convention rather than emit a step with no input at all.
        params["output_dir"] = output_dir
        params["st_h5ad"] = h5ad_path
        if reference_path:
            params["sc_h5ad"] = reference_path
    else:
        out_param = _declared_input_param(mcp_function, _OUTPUT_DIR_PARAM_NAMES)
        if out_param:
            params[out_param] = output_dir
        slide_param = _declared_input_param(mcp_function, _SLIDE_PARAM_NAMES)
        list_param = _declared_input_param(mcp_function, _SLIDE_LIST_PARAM_NAMES)
        if slide_param:
            params[slide_param] = h5ad_path
        elif list_param:
            # Multi-slice tool; one slice is the degenerate case. paste_pairwise_align needs two,
            # which its own required-parameter description says.
            params[list_param] = [h5ad_path]
        if reference_path:
            ref_param = _declared_input_param(mcp_function, _REFERENCE_PARAM_NAMES)
            if ref_param:
                params[ref_param] = reference_path
        # The h5ad was handed to the parameter this tool's "h5ad" mode reads, so say that mode: the
        # default (or the absence of one) reads other parameters, and svgbit's "visium_10x" default
        # then ran on counts_h5/spatial_dir = None (hunt 2026-09-30, u23-transcriptomics-skills-6).
        if slide_param and slide_param in _H5AD_MODE_READS.get(mcp_function, ()):
            params.setdefault("input_mode", "h5ad")

    # Every required parameter, not only those of tools that take no .h5ad: an h5ad tool can
    # require something the slide does not supply (tacco's annotation_key, moscot's problem_type,
    # graphst's n_clusters), and the plan used to omit it silently (u23-transcriptomics-skills-6).
    declared = _DECLARED_TOOL_PARAMS.get(mcp_function, {})
    if declared:
        _surface_required_inputs(mcp_function, params)

    # Add tool-specific defaults. A "required" value (e.g. n_clusters for stagate/cellcharter/spaceflow)
    # must be SURFACED, not silently dropped, or the emitted step is missing a mandatory argument.
    # Anything else is copied only when it is a value of the parameter's declared type: a sentence
    # copied as rad_cutoff, or "optional" as a file path, made a call that could not succeed
    # (hunt 2026-09-30, u23-transcriptomics-skills-5).
    key_params = mcp_tool.get("key_params", {})
    for k, v in key_params.items():
        if k in params:
            continue
        if v == "required":
            params[k] = "<REQUIRED — set before running>"
        elif v != "auto or user-specified" and _fits_declared_type(v, declared.get(k)):
            params[k] = v

    return params


#: {tool: the slide parameters its input_mode="h5ad" reads}. Read from each portal's own mode list;
#: a tool absent here takes no input_mode, or has no h5ad mode at all (starfysh).
_H5AD_MODE_READS: dict[str, tuple[str, ...]] = {
    "spatialde_run_svg": ("h5ad_path",),
    "svgbit_run": ("adata_path",),
    "spatialprompt_cluster": ("spatial_h5ad", "st_h5ad", "h5ad_path"),
    "spatialprompt_deconvolution": ("spatial_h5ad", "st_h5ad", "h5ad_path"),
    "bayestme_deconvolution": ("h5ad_path",),
    "ucdeconvolve_base": ("h5ad_path",),
    "somde_run": ("h5ad_path",),
    "spaceflow_spatial_domains": ("h5ad_path",),
}

_TYPE_CHECKS: dict[str, Any] = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "bool": lambda v: isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "string": lambda v: isinstance(v, str),
}


def _fits_declared_type(value: Any, spec: dict | None) -> bool:
    """True when *value* is of the type the config declares; a type the config does not state passes."""
    check = _TYPE_CHECKS.get(str((spec or {}).get("type") or "").strip().lower())
    return True if check is None else bool(check(value))


def _expected_outputs(goal: str, tool_key: str, output_dir: str) -> dict[str, str]:
    """Where a step's outputs land, and where their names are -- never a guessed file name.

    This used to build names from a ``{tool_key}_output.h5ad`` / ``_domains.csv`` template that no
    worker follows (STAGATE writes stagate_domains.h5ad and stagate_domain_assignments.csv, tacco
    tacco_composition.csv), so an agent checking for the planned files concluded a successful run
    had failed (hunt 2026-09-30, u23-transcriptomics-skills-7). Each tool's reply lists what it
    wrote, so that is where the names come from.
    """
    return {
        "output_dir": output_dir,
        "files": "the tool's reply lists what it wrote (output_files in most portals); take the names from there",
    }


def _qc_recommendations(qc_stats: dict, data_type: str, n_before: int, n_after: int) -> list[str]:
    """Generate QC recommendations based on metrics."""
    recs = []
    pct_removed = 100 * (n_before - n_after) / max(n_before, 1)

    if pct_removed > 30:
        recs.append(
            f"WARNING: {pct_removed:.0f}% of cells/spots removed. If this seems excessive for your "
            "tissue type, run again with a lower min_counts / min_genes or a higher max_pct_mt."
        )

    mt_median = qc_stats.get("pct_counts_mt", {}).get("median", 0)
    if mt_median > 10:
        recs.append(
            f"High median mitochondrial content ({mt_median:.1f}%). May indicate stressed/dying cells "
            "or tissue processing artifacts."
        )

    total_median = qc_stats.get("total_counts", {}).get("median", 0)
    if data_type == "spatial" and total_median < 500:
        recs.append(
            f"Low median counts ({total_median:.0f}). Consider using methods robust to low counts "
            "(e.g., Cell2Location for deconvolution, Hotspot for SVGs)."
        )

    genes_median = qc_stats.get("n_genes_by_counts", {}).get("median", 0)
    if genes_median < 200:
        recs.append(
            f"Low gene detection ({genes_median:.0f} median). For spatial clustering, prefer "
            "methods that handle sparse data (GraphST, STAGATE)."
        )

    if not recs:
        recs.append("QC metrics look normal. Data is ready for downstream analysis.")

    return recs


def _diagnose_non_spatial_file(p: Path) -> str:
    """Diagnose a non-spatial transcriptomics file."""
    # Through the compression, not at it. ``Path.suffix`` answers ``.gz`` for ``counts.csv.gz`` as
    # readily as for ``brain.nii.gz``, so every gzipped table missed all the arms below and fell
    # through to "Unrecognized file format" -- under status "diagnosed", so nothing downstream
    # treats it as a failure and the model is simply told a routine count matrix is a file we do
    # not recognise. ``uncompressed_suffix`` strips one recognised compression suffix and no more,
    # which is what keeps the image volume out of the tabular arm.
    suffix = uncompressed_suffix(p)

    result: dict[str, Any] = {
        "status": "diagnosed",
        "input_type": "file",
        "input_path": str(p),
        "file_name": p.name,
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
    }

    if suffix == ".h5ad":
        try:
            # read_h5ad_backed, not a bare backed read: this diagnosis is routinely followed by a
            # repair or standardization step that rewrites the same path, and a retained HDF5 lock
            # makes that fail with errno 11.
            with read_h5ad_backed(str(p)) as adata:
                has_spatial = "spatial" in adata.obsm
                result["data_type"] = "spatial" if has_spatial else "single_cell"
                result["n_obs"] = adata.n_obs
                result["n_vars"] = adata.n_vars
                result["detected_format"] = "h5ad"

                if not has_spatial:
                    result["platform"] = "single-cell RNA-seq (h5ad)"
                    result["summary"] = (
                        f"scRNA-seq h5ad with {adata.n_obs:,} cells and {adata.n_vars:,} genes. "
                        "No spatial coordinates found. Use standard scanpy workflows or as a "
                        "reference for spatial deconvolution tools."
                    )
                    result["recommended_workflow"] = [
                        "run_transcriptomics_qc for quality control",
                        "Standard scanpy preprocessing (normalize, HVG, PCA, neighbors, UMAP, leiden)",
                        "If using as deconvolution reference: prepare_reference_data()",
                    ]
                    result["can_use_as_deconvolution_reference"] = True
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"Failed to read h5ad: {e}"
        return json.dumps(result, indent=2)

    if suffix == ".loom":
        result["data_type"] = "single_cell"
        result["detected_format"] = "loom"
        result["platform"] = "single-cell (Loom format)"
        result["summary"] = "Loom file detected. Convert to h5ad with scanpy.read_loom()."
        result["recommended_workflow"] = ["Convert to h5ad", "run_transcriptomics_qc"]
        return json.dumps(result, indent=2)

    if suffix == ".h5":
        result["data_type"] = "single_cell"
        result["detected_format"] = "10x_h5_counts"
        result["platform"] = "10x Chromium (scRNA-seq)"
        result["summary"] = "10x HDF5 counts file without spatial directory — likely scRNA-seq."
        result["recommended_workflow"] = ["Convert to h5ad with scanpy.read_10x_h5()", "run_transcriptomics_qc"]
        return json.dumps(result, indent=2)

    if suffix in (".csv", ".tsv", ".txt"):
        result["data_type"] = "unknown"
        result["detected_format"] = "tabular"
        try:
            import pandas as pd

            # Read the delimiter out of the file, not out of its name. This is user-supplied
            # input, so the name is a hint and nothing more: Excel's "Text (comma delimited)"
            # writes commas into a .txt, and a .csv holding tabs is ordinary tool output. Either
            # one used to parse to a single column, and the model was told a 300-cell count
            # matrix was a one-column file. No extension is trusted here -- unlike the
            # benchmarking readers, this path feeds no recorded score, so nothing constrains it
            # to keep a historical reading.
            sep = sniff_tabular_sep(p, default="\t" if suffix in (".tsv", ".txt") else ",")
            # 50 rows rather than 5. The row labels are the evidence for which axis is which, and
            # five of them are too few for the label classifier to reach a verdict. Still a bounded
            # head read -- nothing here scales with the file.
            df_head = pd.read_csv(p, nrows=50, sep=sep)
            n_cols = len(df_head.columns)

            # A leading non-numeric column is the row-label column (gene ids, barcodes); the rest is
            # the body. Keeping them apart is what lets the values be checked and the labels be read.
            labels, body = None, df_head
            if n_cols and not pd.api.types.is_numeric_dtype(df_head.iloc[:, 0]):
                labels, body = list(df_head.iloc[:, 0]), df_head.iloc[:, 1:]
            numeric = sum(bool(pd.api.types.is_numeric_dtype(body[c])) for c in body.columns)
            is_matrix = bool(len(body.columns)) and numeric >= 0.9 * len(body.columns)

            if n_cols > 100 and is_matrix:
                # >100 columns means a matrix rather than a handful of bulk sample columns. It does
                # NOT say which axis is which: this used to assert "cells-as-columns" from the count
                # alone and said it for a matrix and its own transpose alike, so a model acting on it
                # transposed correct data. Orientation is decided by the shared resolver, which
                # weighs the label classes and refuses when they identify only one axis -- refusing
                # is the useful answer, because "check before transposing" is actionable and a
                # coin-flip is not.
                from spatialomicsgym.utils.format_probe import resolve_orientation

                result["data_type"] = "single_cell"
                result["platform"] = "count matrix"
                verdict = resolve_orientation(labels, list(body.columns))
                if verdict.refused:
                    where = (
                        "Orientation could not be determined from the labels; check which axis is "
                        "cells before transposing."
                    )
                elif verdict.orientation == "genes_x_cells":
                    where = "Rows are genes, columns are cells (cells-as-columns)."
                else:
                    where = "Rows are cells, columns are genes (genes-as-columns)."
                # Decision-relevant fields only. The full verdict carries a per-signal breakdown of
                # label fractions that is diagnostic noise in a payload the model has to read.
                result["orientation"] = {
                    "orientation": verdict.orientation,
                    "transpose_needed": verdict.transpose_needed,
                    "confidence": round(float(verdict.confidence), 3),
                    "evidence": list(verdict.evidence),
                }
                result["summary"] = f"Tabular count matrix with {n_cols} columns. {where} Convert to h5ad with scanpy."
            elif n_cols > 100:
                # Wide, but the values are text. Calling this a count matrix -- which happened for a
                # 120-column clinical sheet -- reported single-cell RNA-seq without reading a value.
                result["summary"] = (
                    f"Tabular file with {n_cols} columns whose values are not numeric, so it is an "
                    f"annotation/metadata table rather than a count matrix: {list(df_head.columns)[:5]}"
                )
            else:
                result["summary"] = f"Tabular file with {n_cols} columns: {list(df_head.columns)[:5]}"
        except Exception:
            result["summary"] = "Tabular file, format not auto-detected."
        return json.dumps(result, indent=2)

    if suffix in (".rds", ".rda", ".rdata"):
        result["data_type"] = "unknown"
        result["detected_format"] = "r_object"
        result["platform"] = "R (Seurat/SingleCellExperiment)"
        result["summary"] = (
            "R object. A Seurat .rds can be converted with convert_seurat_rds_to_h5ad(); "
            ".rda/.rdata may need manual export first."
        )
        result["recommended_workflow"] = ["convert_seurat_rds_to_h5ad()", "diagnose_transcriptomics_data() on output"]
        return json.dumps(result, indent=2)

    result["data_type"] = "unknown"
    result["detected_format"] = "unknown"
    result["summary"] = f"Unrecognized file format: {p.name}"
    return json.dumps(result, indent=2)


def _diagnose_non_spatial_directory(p: Path) -> str:
    """Diagnose a non-spatial transcriptomics directory."""
    contents = {f.name for f in p.iterdir() if not f.name.startswith(".")}
    subdirs = {f.name for f in p.iterdir() if f.is_dir()}

    result: dict[str, Any] = {
        "status": "diagnosed",
        "input_type": "directory",
        "input_path": str(p),
        "total_files": len(contents),
    }

    # 10x Cell Ranger scRNA-seq output (no spatial/ directory)
    has_counts = bool(contents & {"filtered_feature_bc_matrix.h5", "raw_feature_bc_matrix.h5"})
    has_mtx_dir = bool(subdirs & {"filtered_feature_bc_matrix", "raw_feature_bc_matrix"})
    has_spatial = "spatial" in subdirs

    if (has_counts or has_mtx_dir) and not has_spatial:
        result["data_type"] = "single_cell"
        result["detected_format"] = "cellranger_scrna"
        result["platform"] = "10x Chromium (Cell Ranger scRNA-seq)"
        result["summary"] = (
            "Cell Ranger scRNA-seq output (no spatial/ directory). "
            "Read with scanpy.read_10x_h5() or scanpy.read_10x_mtx()."
        )
        result["recommended_workflow"] = [
            "Read with scanpy.read_10x_h5()",
            "run_transcriptomics_qc()",
            "Standard clustering pipeline",
            "Can be used as deconvolution reference for spatial data",
        ]
        return json.dumps(result, indent=2)

    # Check for FASTQ files (raw sequencing data)
    fastq_files = [f for f in contents if f.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz"))]
    if fastq_files:
        result["data_type"] = "raw_sequencing"
        result["detected_format"] = "fastq"
        result["platform"] = "raw sequencing reads"
        result["summary"] = f"Found {len(fastq_files)} FASTQ files. Run Cell Ranger / STARsolo / STAR first."
        return json.dumps(result, indent=2)

    # Check for h5ad files
    h5ad_files = [f for f in contents if f.endswith(".h5ad")]
    if h5ad_files:
        result["data_type"] = "multiple_samples"
        result["detected_format"] = "h5ad_collection"
        result["summary"] = f"Directory with {len(h5ad_files)} h5ad files. May be multi-sample study."
        result["h5ad_files"] = sorted(h5ad_files)
        return json.dumps(result, indent=2)

    # Fallback. Through `_uncompressed_name` because a `.gz` on a table is an encoding, not a
    # format: a directory of `.csv.gz` files used to be summarised as holding 0 tabular files.
    from spatialomicsgym.tool.spatial_pipeline import _uncompressed_name

    csv_files = [f for f in contents if _uncompressed_name(f).endswith((".csv", ".tsv", ".txt"))]
    result["data_type"] = "unknown"
    result["detected_format"] = "unknown_directory"
    result["summary"] = (
        f"Directory with {len(csv_files)} tabular files, {len(list(subdirs))} subdirectories. "
        "Could not auto-detect platform."
    )
    return json.dumps(result, indent=2)
