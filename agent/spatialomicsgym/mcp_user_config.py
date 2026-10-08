"""
Where ``mcp_config_user.yaml`` is, resolved without importing the agent.

Moved out of ``agent/mcp_config_merger.py`` unchanged. The reason is measured:
``spatialomicsgym.agent.__init__`` imports ``STCoscientist``, so ANY import from that package
pulls in langchain, langgraph, numpy and pandas -- and ``test_webui_stays_out_of_the_heavy_stack``
pins that the portal's settings routes answer with none of that loaded, because the 1.6 GB minimal
agent env has to serve the page. The settings panel now needs this resolver (a created tool was
invisible to every surface without it), so the resolver moved to where a web route can read it.

Not copied -- moved. ``mcp_config_merger`` re-exports both names, so every existing caller and
every test that patches them is untouched, and there is still exactly ONE implementation. Two
copies of a path resolver is the bug this function was written to fix, in a new place.

``SOG_MCP_USER_CONFIG`` is new and is the operator escape hatch the rest of this project has for
every other path: an absolute override, honoured before anything is searched for. It also makes
the test suite hermetic -- without it, a run on a box that has created a tool reads that tool's
config and counts it, so the answer depends on the developer's machine.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Absolute override. Empty or unset means resolve as below.
USER_CONFIG_ENV = "SOG_MCP_USER_CONFIG"

#: The directory holding the agent trees in a checkout -- ``<repo>/agent``, where ``MCP_server/`` and
#: ``tools_user/`` sit beside this package. (The name predates the re-layout, when that was the
#: repository root.)
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_USER_CONFIG = "MCP_server/mcp_config_user.yaml"


def user_config_override() -> Path | None:
    """The ``SOG_MCP_USER_CONFIG`` override as a path, or ``None`` when it is unset or blank.

    One reading of the variable for the readers (:func:`resolve_user_config_path`) and the writers
    (``tools_user/knowledge_manager.mcp_config_user_path``, ``tools_user/trash_manager``'s twin, and
    ``agent/broker.register_tool`` through the first). The readers honoured it and the writers did
    not, so with it set a created tool was reported registered and never wired -- the file the merger
    and the mtime watcher read was not the one written -- and a trashed tool stayed wired from the
    override (hunt 2026-09-30, u14-mcp-wiring-7).
    """
    override = _override_text()
    return Path(override) if override else None


def _override_text() -> str:
    """The variable's value, stripped; ``""`` when unset. Returned verbatim by the reader below."""
    return (os.environ.get(USER_CONFIG_ENV) or "").strip()


def resolve_user_config_path(user_path: str = DEFAULT_USER_CONFIG, repo_root: Path | None = None) -> str:
    """Locate ``mcp_config_user.yaml`` regardless of the process working directory.

    ``tools_user/knowledge_manager.py`` writes this file to an absolute, repo-anchored path, while
    every reader in the agent spelled it as a bare relative path resolved against the CWD. Nothing
    in the package chdirs, so running the agent from a data directory -- the normal case -- meant
    the base MCP tools loaded (the front doors pass a resolved ``config_path``) and every
    user-created tool silently disappeared, with no warning: a missing user config is
    indistinguishable from an empty one.

    Resolution is additive, so no working setup changes behaviour: an absolute path, or a relative
    one that exists in the CWD, is returned untouched; only when neither holds do we fall back to
    the repo-anchored copy, and only if that exists. When the file is genuinely absent everywhere
    the caller gets back exactly the spelling it passed, preserving the existing "no user tools"
    path. ``repo_root`` is injectable for tests.

    Off a checkout (pip-only install) one more rung follows the repo-anchored one: the writable
    instance root's copy — under ``platform_root.platform_dir()``, the SAME resolver
    ``tools_user/knowledge_manager.py`` writes the file through, so a relocated root
    (``SOG_PLATFORM_ROOT``) is read from where the tool-creation layer wrote it, with the
    seeded SOG_HOME as that resolver's own default. The rung is dark on checkouts AND whenever
    ``repo_root`` was injected, so no existing answer (and no test seam) can be moved by this
    machine's home directory.
    """
    override = _override_text()
    if override and repo_root is None and str(user_path) == DEFAULT_USER_CONFIG:
        # Honoured before anything is searched for, and under two conditions.
        #
        # `repo_root is None`: an injected root is a test asking a specific question, and an
        # environment variable must not answer a different one.
        #
        # `user_path == DEFAULT_USER_CONFIG`: the variable names the user CONFIG, not "whatever
        # path you ask about". Without this it also answered for
        # `resolve_user_path("tools_user/install_log.json")` and for a caller naming some other
        # file entirely -- handing back the config's path for a question about something else.
        # Caught by three tests that ask this resolver about a deliberately-absent spelling.
        return override
    candidate = Path(user_path)
    if candidate.is_absolute() or candidate.exists():
        return str(user_path)
    anchored = (repo_root if repo_root is not None else _REPO_ROOT) / user_path
    if anchored.exists():
        return str(anchored)
    if repo_root is None:
        from spatialomicsgym import platform_root

        if not platform_root.running_from_checkout():
            seeded = platform_root.platform_dir() / user_path
            if seeded.exists():
                return str(seeded)
    return str(user_path)


#: Where the tool registry lives, relative to whichever root holds the user-tool layer.
INSTALL_LOG_REL = "tools_user/install_log.json"


def resolve_user_path(relative: str) -> str:
    """Any file in the user-tool layer, found the same way the user config is.

    The ladder is :func:`resolve_user_config_path`'s, reused rather than restated, because the
    layer is one thing: ``mcp_config_user.yaml``, ``install_log.json`` and the workers beside them
    are written by the same code to the same root, and two resolvers for one root is how they
    come to disagree.

    Concretely, this exists because ``stcoscientist._register_user_tool_skill`` asked
    ``Path("tools_user/install_log.json").exists()`` -- CWD-relative. Started from a data
    directory, which is the normal case, that answered "no user tools" and the skill was never
    registered, so every created tool disappeared from the retriever with nothing said. The same
    class of bug as the documented ``.env`` one, and the same class the user config resolver was
    written to fix.
    """
    return resolve_user_config_path(str(relative))


def install_log_path() -> str:
    """``tools_user/install_log.json``, wherever this install keeps it."""
    return resolve_user_path(INSTALL_LOG_REL)


def _server_tools(meta: dict) -> list:
    """A server entry's ``tools`` as a real list, tolerating a null/scalar value.

    A hand-edited base config (every tool commented out leaves ``tools:`` present but null) or
    an LLM-authored ``mcp_config_user.yaml`` can write ``tools:`` as null or a non-list.
    ``dict.get("tools", [])`` only supplies the default when the KEY is ABSENT, so ``tools: null``
    yields ``None`` and ``for tool in None`` raises ``TypeError`` -- which, at the base-server
    site below, is OUTSIDE the try and so breaks this module's documented "Never raises" contract
    and drops ALL 88 MCP servers. Coercing to a list here keeps iteration always safe (and is a
    no-op for the normal list value).
    """
    tools = meta.get("tools")
    return tools if isinstance(tools, list) else []


# Both names live in `spatialomicsgym.mcp_user_config` now and are re-exported here.
# Moved because the portal's settings routes need the resolver and may not import this
# package: `agent/__init__` imports STCoscientist, which pulls langchain, langgraph, numpy
# and pandas -- pinned against by `test_webui_stays_out_of_the_heavy_stack`. Re-exported
# rather than copied, so there is still one implementation and every caller and patch site
# here keeps working.


def shipped_identity(original_servers: dict) -> tuple[set[str], set[str]]:
    """The names the SHIPPED catalogue already owns: server keys, and callable names.

    Both halves are what :func:`user_server_skip_reason` refuses a user server for, so they are
    derived here once rather than re-spelled by each caller -- the resync pruner has no other way
    to ask the question, and a pruner that computes "which names are taken" differently from the
    merger is the disagreement this pair of functions exists to end.
    """
    names = {str(name) for name in original_servers}
    functions: set[str] = set()
    for meta in original_servers.values():
        functions |= user_function_names(meta)
    return names, functions


def user_function_names(meta: object) -> set[str]:
    """The callable names this server declares. ``set()`` for anything malformed.

    Both entry shapes ``add_mcp`` registers: ``spatialomicsgym_name`` and the discovered ``name``.
    Reading only the first let a user entry written as ``name: run_scanpy_spatial_domain`` through
    the "original always wins" check, and ``add_mcp`` then bound it over the shipped wrapper
    (u14-mcp-wiring-10).
    """
    if not isinstance(meta, dict):
        return set()
    names: set[str] = set()
    for tool in _server_tools(meta):
        if not isinstance(tool, dict):
            continue
        name = tool.get("spatialomicsgym_name") if "spatialomicsgym_name" in tool else tool.get("name")
        if isinstance(name, str):
            names.add(name)
    return names


def user_server_skip_reason(
    name: str,
    meta: object,
    original_names: set[str],
    original_function_names: set[str],
    already_claimed: set[str] | None = None,
) -> tuple[str, str] | None:
    """Why the merged config will NOT serve this user server, or ``None`` when it will.

    ONE predicate, for the merger that decides what to wire and for ``tool_management``'s resync
    pruner that decides what is still live. They had two: the pruner counted a server live on
    ``enabled`` alone, while the merge additionally drops it for a shipped-server name clash, a
    shipped ``spatialomicsgym_name`` clash, and a meta that is not a mapping. So a user tool the
    merge refuses read as live to the pruner, which left it registered in the agent's catalogs --
    a callable name the model can pick and nothing behind it.

    ``already_claimed`` is the callable names **earlier user servers in this same merge** have
    taken. Shipped-vs-user precedence was checked and documented; user-vs-user was not checked at
    all. Two user servers could each declare ``collide_between_users`` and both would be merged
    with no warning: ``module2api`` dedups per module, and ``_custom_functions[tool_name]`` is a
    flat last-writer-wins dict. The catalog then showed the model *both* tools with their distinct
    descriptions, and whichever it picked ran the other one's implementation. That is the
    tool-creation layer, where the agent writes these configs itself and a name reuse is plausible.

    Both callers must pass it, and both must accumulate in the same order, or the merge and the
    resync pruner disagree about what is live -- the precise desync this one predicate exists to
    prevent.

    Returns ``(code, message)``. The code is for the caller's own branching -- ``"disabled"`` is a
    configuration choice the merge passes over in silence and does not count as skipped, the rest
    are warnings about a file that is wrong -- and the message is the merger's wording, kept here
    so there is one place it is written.
    """
    if not isinstance(meta, dict):
        return ("invalid", f"User tool '{name}' has invalid config. Skipping.")
    if name in original_names:
        return ("server_clash", f"User server '{name}' conflicts with original server. Skipping.")
    user_func_names = user_function_names(meta)
    conflicting_funcs = user_func_names & original_function_names
    if conflicting_funcs:
        return (
            "function_clash",
            f"User tool '{name}' has function name conflicts with original tools: {conflicting_funcs}. Skipping.",
        )
    # Said separately from the shipped clash above, because the remedy is different: a shipped
    # name is reserved and must be abandoned, while a name another *user* tool took can be freed
    # by renaming either one.
    taken_by_peer = user_func_names & (already_claimed or set())
    if taken_by_peer:
        return (
            "function_clash_user",
            f"User tool '{name}' declares function names an earlier user tool already claimed: "
            f"{taken_by_peer}. Skipping -- rename one of them, or only one will ever be reachable.",
        )
    if not meta.get("enabled", True):
        return ("disabled", f"User server '{name}' is disabled.")
    return None


#: Top-level key of a BASE config (the web portal's session overlay) naming created servers the
#: portal's Settings turned off. The portal cannot edit ``mcp_config_user.yaml`` per session, and an
#: overlay that forced ``enabled: false`` only on shipped entries left a created tool wired after
#: Settings said it was off (hunt 2026-09-30, u01-server-a-7).
DISABLED_USER_SERVERS_KEY = "disabled_user_servers"


def disabled_user_servers(base: object) -> set[str]:
    """The created servers a base config says are turned off. ``set()`` for anything malformed."""
    names = base.get(DISABLED_USER_SERVERS_KEY) if isinstance(base, dict) else None
    return {n for n in names if isinstance(n, str)} if isinstance(names, list) else set()
