#!/usr/bin/env python3
"""Drive memory-assisted creation of the 21 GitHub tools with a simple prompt each.

This operationalizes the runtime self-extension goal: for each tool, it issues a
*simple* request — "create an MCP tool from GitHub source: <url>" — and lets the
agent's user-tool-creation system + procedural memory do the rest. The memory store
is seeded first (winning recipe + gotchas per tool), so STCoscientist's Phase 1.0
memory consult injects the fix and applies it.

Modes:
  - With a valid LLM API key: actually runs agent.go() per tool and records outcomes.
  - Without a key (e.g. CI): PREVIEW mode — prints each simple prompt and the memory
    hint that WOULD be injected, proving the system is primed. No tools are created.

Usage (from the agent/ directory of a checkout, where ``tools_user`` is importable):
    SOG_MEMORY_ENABLED=true python -m tools_user.create_all_21_tools          # auto mode
    SOG_MEMORY_ENABLED=true python -m tools_user.create_all_21_tools --preview
"""

import os
import sys

os.environ.setdefault("SOG_MEMORY_ENABLED", "true")
os.environ.setdefault("SOG_TOOL_CREATION_ENABLED", "true")  # required by the Phase-0 pre-flight gate
os.environ.setdefault("SOG_MAX_USER_ENVS", "30")  # default cap is 20; raise so all 21 tools fit

# Load .env up front so the API key + LLM_SOURCE are visible to _has_api_key()/the agent.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from tools_user.seed_tool_creation_memory import RECIPES

TASK_HINT = {
    "banksy": "spatial clustering",
    "seagal": "spatial gene association",
    "spage": "gene imputation",
    "spavae": "spatial clustering",
    "spadecon": "deconvolution",
    "gaston": "spatial clustering",
    "spatopic": "spatial clustering",
    "semla": "spatial clustering",
    "hdwgcna": "co-expression network",
    "spatialinfercnv": "CNV detection",
    "nichecompass": "cell communication",
    "liana": "cell communication",
    "stereopy": "spatial clustering",
    "sopa": "segmentation",
    "giotto": "spatial analysis",
    "novae": "spatial clustering",
    "voyager": "spatial gene association",
    "ficture": "segmentation",
    "sctm": "dimension reduction",
    "spanorm": "normalization",
    "splisosm": "isoform detection",
}


def _install_log() -> str:
    """The user-tool registry, wherever this install keeps it (not the launch directory's copy)."""
    from spatialomicsgym.mcp_user_config import install_log_path

    return install_log_path()


def _simple_prompt(tool_id: str, url: str) -> str:
    task = TASK_HINT.get(tool_id, "spatial analysis")
    # Deliberately minimal — the "GitHub source: <url>" line is what triggers the
    # agent's memory consult; everything else the agent figures out itself.
    return f"Please add a new MCP tool for {task}.\nGitHub source: {url}"


def _has_api_key() -> bool:
    return any(
        os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "AZURE_OPENAI_API_KEY")
    )


def _active_tool_ids() -> set:
    """Ground-truth registration check: tool_ids active in the install_log."""
    import json

    try:
        d = json.load(open(_install_log()))
        entries = d if isinstance(d, list) else d.get("install_log", [])
        return {e.get("tool_id") for e in entries if e.get("status", "active") != "trashed"}
    except Exception:
        return set()


def _perm_delete_if_exists(tid: str) -> None:
    """Clean-slate re-test: if the tool is already registered, permanently delete it
    (trash -> permanent_delete, the two-stage lifecycle) before re-creating. Also removes
    its conda env so the creation runs from scratch."""
    import json

    try:
        d = json.load(open(_install_log()))
        entries = d if isinstance(d, list) else d.get("install_log", [])
    except Exception:
        entries = []
    entry = next((e for e in entries if e.get("tool_id") == tid), None)
    if not entry:
        # Not registered, but a prior FAILED attempt may have left an orphan conda env
        # dir — that causes `conda create` to fail with "prefix already exists" on retry.
        # Reuse trash_manager's derived root: hardcoding /opt/conda made this cleanup a no-op
        # wherever conda lives elsewhere, so the retry it exists to unblock still died.
        import shutil

        from tools_user import trash_manager as _tm

        orphan = str(_tm.CONDA_ENVS_DIR / f"user_{tid}")
        if os.path.isdir(orphan):
            shutil.rmtree(orphan, ignore_errors=True)
            print(f"  [perm-delete] removed orphan env {orphan} (failed-attempt leftover)")
        return
    try:
        from tools_user import trash_manager as tm

        if entry.get("status", "active") == "active":
            print(f"  [perm-delete] trash {tid}")
            tm.trash_tool(tid)
        print(f"  [perm-delete] permanent_delete {tid}")
        res = tm.permanent_delete(tid)
        print(f"  [perm-delete] {tid}: success={res.get('success')} reclaimed={res.get('disk_reclaimed')}")
    except Exception as e:
        print(f"  [perm-delete warn] {tid}: {type(e).__name__}: {str(e)[:140]}")


def main():
    only = None
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        if i + 1 < len(sys.argv):
            only = set(sys.argv[i + 1].split(","))  # comma-separated list of tool_ids
    preview = "--preview" in sys.argv or not _has_api_key()
    cold = "--cold" in sys.argv

    # Memory is an optional WARM-START cache, not the mechanism. In --cold mode we skip
    # per-tool seeding entirely: the agent must create each tool from its GENERAL know-how
    # (Phase 0-6 playbook + Error Taxonomy) as if every repo were never-seen, and the
    # memory is built ORGANICALLY from its own attempts (the learning loop). This is the
    # truer test of a generative system that handles arbitrary new tools.
    if cold:
        # True never-seen test: CLEAR any prior short-term memory (incl. earlier warm
        # seeds) so the agent starts with zero per-tool knowledge and creates from its
        # general know-how alone; memory is then built organically from its own attempts.
        from tools_user.memory_manager import MemoryManager

        n = MemoryManager.get().delete_all_short_term(backup=True)
        print(
            f"[cold mode] cleared {n} prior memory entries — true never-seen run; "
            "creating from general know-how, memory learned organically."
        )
    else:
        from tools_user.seed_tool_creation_memory import main as seed

        seed()

    tools = [(tid, url) for (tid, url, *_rest) in RECIPES if only is None or tid in only]

    if preview:
        from tools_user.memory_manager import MemoryManager

        mm = MemoryManager.get()
        print(
            f"\n{'=' * 72}\nPREVIEW MODE (no LLM API key) — {len(tools)} tools primed and ready.\n"
            f"With a key, each prompt below is sent to agent.go().\n{'=' * 72}"
        )
        for i, (tid, url) in enumerate(tools, 1):
            st = mm.read_short_term(url)
            a = (st.get("attempts") or [{}])[-1]
            primed = (
                f"{a.get('outcome', '?')} via "
                f"{(a.get('strategy') or {}).get('install_path', '?')} "
                f"({len(a.get('gotchas') or [])} gotcha)"
            )
            print(f"[{i:2d}/21] {tid:16s} {url.split('github.com/')[-1]:32s} primed: {primed}")
        print("\nSet a valid API key (ANTHROPIC_API_KEY/OPENAI_API_KEY) and re-run to create.")
        return

    # 2) live creation — use the configured provider (Azure gpt-5 when LLM_SOURCE=AzureOpenAI)
    from spatialomicsgym.agent.stcoscientist import STCoscientist

    src = os.environ.get("LLM_SOURCE") or None
    llm = os.environ.get("SOG_CREATE_LLM") or ("azure-gpt-5" if (src or "").lower().startswith("azure") else None)
    # expected_data_lake_files=[] skips the data-lake S3 download (not needed for tool
    # creation, and the renamed bucket currently 404s — see rename regression note).
    kw = {"path": "./data", "expected_data_lake_files": []}
    if llm:
        kw.update(llm=llm, source=src)
    agent = STCoscientist(**kw)
    print(f"[driver] using llm={llm or 'default'} source={src or 'default'}")
    results = {"created": [], "failed": [], "known_hang": []}
    for i, (tid, url) in enumerate(tools, 1):
        print(f"\n{'=' * 72}\n[{i}/{len(tools)}] creating {tid} from {url}\n{'=' * 72}")
        _perm_delete_if_exists(tid)  # clean slate: perm-delete any prior copy, then re-create
        try:
            agent.go(_simple_prompt(tid, url))
        except Exception as e:
            print(f"  [error] {tid}: {type(e).__name__}: {str(e)[:160]}")
        # Ground truth: did the tool actually register in the install_log?
        if tid in _active_tool_ids():
            results["created"].append(tid)
        elif tid == "voyager":
            results["known_hang"].append(tid)
        else:
            results["failed"].append(tid)
    print(
        f"\n{'=' * 72}\nDONE: created={len(results['created'])}/21  "
        f"failed={len(results['failed'])}  known_hang={len(results['known_hang'])}"
    )
    print("  created:", results["created"])
    if results["failed"]:
        print("  failed:", results["failed"])
    if results["known_hang"]:
        print("  known_hang:", results["known_hang"])


if __name__ == "__main__":
    main()
