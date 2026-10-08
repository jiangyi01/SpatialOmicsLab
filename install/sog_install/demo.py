"""
Part D — the post-install ``demo`` phase: an automatic, opt-in showcase + real-run.

After provisioning + the two test tiers finish, the wizard offers (Y/N) to:

1. **Demo** one of the tools that *passed* testing, on a bundled dataset — so the very first
   thing the user sees after a long install is their new setup actually *doing something*.
2. **Real-run** their own task, in plain language, on their own data (or the demo dataset).

Both go through the **same out-of-process Tier-2 machinery** as the real-data test tier
(``testing.run_agent_probe`` → ``_agent_probe`` in the base env): a jargon-free biologist prompt
is handed to a real ``STCoscientist(...).go()`` and the agent's *own* recommender routes it to a
tool. Nothing here inspects or hard-codes which tool runs — that is the whole point of the demo
(the user watches the agent choose).

**Key-gate (Q3 "run if key present, else print the command").** With a usable LLM key visible we
launch the agent now and show the route + answer. Without one we never prompt for a key and never
echo a secret — we write a *self-contained runnable script* (``state_dir()/demo_run.py`` /
``real_run.py``) and print the exact ``conda run`` command, so the user adds a key to ``.env`` and
runs it themselves.

**Soft phase.** ``wizard._phase_demo`` calls :func:`run_demo_phase` and *always* returns ``True`` —
a declined or skipped demo never fails the run (see ``state.SOFT_PHASES``). Headless / scripted /
dry-run / no-passed-tools all short-circuit to a logged SKIP.

This module is env-only and import-light: stdlib + the sibling setup modules (``testing`` /
``testdata`` / ``category_prompts`` / ``constants``). It never imports ``STCoscientist`` — the heavy
agent lives only inside the probe subprocess — so ``import sog_install`` stays
stdlib+pyyaml. It never mutates a conda env or agent/tool source; it only *triggers* the agent.
"""

from __future__ import annotations

import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import constants, credentials, key_library, llm_setup, onboarding, progress, testdata, testing, transcript
from ._agent_probe import extract_solution
from .category_prompts import prompt_for
from .demo_tool_names import DEMO_NEEDS_SC_REF, DEMO_UNSTAGEABLE_CATEGORIES, display_name, with_tool_named
from .session_log import redact

if TYPE_CHECKING:
    from pathlib import Path

    from .categories import Category
    from .envtools import Conda
    from .prompts import PromptIO
    from .session_log import SessionLog
    from .specs import ToolSpec
    from .state import SetupState

# ``scanpy_spatial`` gets a curated, richer demo — the Part-B human lymph-node Visium slide (a real
# 243 MB section that resolves to ~10 spatial domains) — *when that dataset is on disk*. Every other
# passed tool demos on the shared mini dataset via its category's plain-language prompt.
_SCANPY_DEMO_SERVER = "scanpy_spatial"
_V1_LYMPH_FILE = "V1_Human_Lymph_Node.h5ad"  # under constants.mini_data_dir() (test/test_data/)
_LYMPH_PROMPT = (
    "I've profiled a human lymph node with 10x Visium spatial transcriptomics. Map out the distinct "
    "spatial domains — the tissue regions with different expression patterns — across the section, "
    "and tell me roughly how many there are and how the spots split between them."
)
_LYMPH_DESC = (
    "10x Visium spatial transcriptomics of a human lymph node section; raw counts in X, spot "
    "coordinates in obsm['spatial']."
)

# Cap on how much of the agent's free-text answer we echo back into the terminal (the full text is
# in the run's outputs + the setup log; here we only want a readable at-a-glance summary).
_ANSWER_ECHO_CHARS = 2000
# The "which tool did it route to?" line reads CALLS, never mentions (hunt 2026-09-30, u37-setup-checks-6,
# uL4-honesty-9). It used to take the first ``run_*`` token anywhere in the log, which is the enriched
# prompt's own ``from spatialomicsgym.postanalysis import run_post_analysis`` -- so every demo reported
# run_post_analysis -- and it could never name the many wired functions that are not called ``run_*``.
# Only code the agent executed and provider ``Tool:`` lines are read, never a Human message; a name
# counts when the wired config declares it (``_configured_tool_names``), else when it is a ``run_*`` call.
_ROUTE_RE = re.compile(r"\brun_[a-z0-9_]+")
_ROUTE_CODE_RE = re.compile(r"<execute>(.*?)(?:</execute>|$)|```[\w-]*\n(.*?)(?:```|$)", re.S | re.I)
_ROUTE_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_ROUTE_TOOL_LINE_RE = re.compile(r"^\s*Tool:\s*([A-Za-z_][\w.\-]*)", re.M)
# The post-task analysis the agent is told to run on every turn: a library call, not a routing choice.
_NOT_A_ROUTE = frozenset({"run_post_analysis"})
# ``No module named <X>`` in the agent log/answer — the improvised in-base-env import miss the
# reassurance net cross-checks against the installed+healthy tool fleet (see ``_reassurance_notes``).
_NO_MODULE_RE = re.compile(r"No module named\s*['\"]?([A-Za-z_][\w.]*)")


@dataclass
class DemoOutcome:
    """What the ``demo`` phase did — consumed by the wizard only to set the phase status.

    ``skipped`` is ``True`` when nothing ran (declined / headless / no passed tools); ``reason`` is
    the short, logged explanation; ``actions`` lists what *did* run (``"demo:<tool>"`` / ``"real"``),
    each either launched (key present) or emitted as a runnable script (key absent)."""

    skipped: bool = True
    reason: str = ""
    actions: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# State readers (work identically on a fresh run and on resume — read persisted state)
# --------------------------------------------------------------------------- #
def passed_tools(state: SetupState) -> list[str]:
    """Server keys whose Tier-1 mini-data test PASSed — the menu of demoable tools (Q2)."""
    return [k for k, rec in state.tools.items() if rec.get("test_status") == testing.PASS]


def needs_attention_items(state: SetupState) -> list[tuple[str, str]]:
    """``(server_key, reason)`` for every tool the installer-scientist surfaced as Lane-3
    *needs-attention* (a token / special-input / data-shape gap — NOT an env problem). Shown at the
    end of the demo so the user knows precisely what to fix rather than reading a bare FAIL."""
    out: list[tuple[str, str]] = []
    for k, rec in state.tools.items():
        reason = (rec.get("needs_attention") or "").strip()
        if reason:
            out.append((k, reason))
    return out


# Foundational / shared modules that live in the BASE agent-core env (scanpy, anndata, the scientific
# stack) or are always importable (``__future__``). A "No module named <x>" for one of these means a
# genuinely broken BASE env — NOT a per-tool routing quirk. Several specs nonetheless declare one as
# their ``import_check`` (scanpy_spatial→scanpy, mist/stage/data_converter→anndata, svca→scipy,
# spotgf→pandas, scdot→ot, xfuse/ucdeconvolve→h5py, squidpy→squidpy, 5 no-op ``__future__`` probes),
# so the reassurance net must exclude them (G-F1): a *missed* reassurance is harmless, but a *false*
# "it's just routing, the tool is installed" for a base module steers the user away from repairing the
# base env. The net therefore only ever fires for a tool-UNIQUE module (GraphST, STAGATE_pyG, …).
_SHARED_BASE_MODULES = frozenset(
    {"scanpy", "anndata", "scipy", "numpy", "pandas", "h5py", "squidpy", "sklearn", "ot", "__future__"}
)


def installed_module_index(state: SetupState, specs: dict[str, ToolSpec]) -> dict[str, str]:
    """Map each *installed & healthy* tool's import module (lowercased) → its server key.

    Feeds the reassurance net (:func:`_reassurance_notes`). A tool whose Tier-1 mini-data test PASSed
    has a *working* import inside its ``<basic>_<tool>`` worker env — so if the agent later reports
    ``No module named <that module>`` it improvised the import in the BASE env: the tool is fine, the
    routing is the quirk. Keyed by ``spec.import_check`` (the exact top-level module the worker imports,
    e.g. graphst → ``GraphST``, stagate → ``STAGATE_pyG``). Only PASSed tools are indexed **and
    foundational/shared base modules (``_SHARED_BASE_MODULES``) are excluded**, so a note only ever
    fires for a tool-UNIQUE module — it can never falsely reassure about a genuinely-broken base env
    (e.g. a base env missing ``scanpy`` while ``scanpy_spatial`` passed in its own worker env)."""
    idx: dict[str, str] = {}
    for key in passed_tools(state):
        spec = specs.get(key)
        module = (getattr(spec, "import_check", "") or "").strip().lower()
        if module and module not in _SHARED_BASE_MODULES:
            idx[module] = key
    return idx


def _server_category(server_key: str, cats: list[Category]) -> str:
    """The skill/category name that owns ``server_key`` (for ``category_prompts.prompt_for``).
    Uses the ``cats`` skill names — authoritative for the prompt keys — not ``spec.categories()``."""
    for c in cats:
        if server_key in getattr(c, "servers", ()):
            return c.name
    return ""


def _base_env_ready(conda: Conda, basic: str) -> bool:
    """True unless we can POSITIVELY confirm the agent's base env is absent.

    Conservative on purpose: a conda without :meth:`env_exists` (a fake in tests) or a lookup that
    raises errs toward *proceeding* — the central probe path still surfaces a clear reason — so this
    demo-phase guard only short-circuits on a definite, checkable absence, never on uncertainty."""
    check = getattr(conda, "env_exists", None)
    if not callable(check):
        return True
    try:
        return bool(check(basic))
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_demo_phase(
    *,
    state: SetupState,
    io: PromptIO,
    conda: Conda,
    specs: dict[str, ToolSpec],
    cats: list[Category],
    log: SessionLog | None = None,
    dry_run: bool = False,
    stated_goals: str = "",
    llm: object | None = None,
) -> DemoOutcome:
    """Offer + run the post-install demo and real-run. Never raises for a user decline; guards
    (dry-run / non-interactive / no passed tools) short-circuit to a logged SKIP. Returns a
    :class:`DemoOutcome` the wizard maps to the soft phase's DONE/SKIPPED status.

    ``llm`` is the in-memory :class:`LLMChoice` from onboarding (``None`` on resume). It feeds the
    shared :class:`_KeyGate`, which lets the user connect a key at launch time — reusing the setup
    key by default — so the run the user asked for actually launches instead of only emitting a
    script (see :meth:`_KeyGate.ensure`)."""
    if dry_run:
        return _skip(io, log, "dry-run — a plan preview launches no agent")
    if io.non_interactive:
        return _skip(io, log, "non-interactive/scripted run — the demo is an interactive offer")

    passed = passed_tools(state)
    attn = needs_attention_items(state)
    if not passed:
        _print_needs_attention(io, attn)
        return _skip(io, log, "no tool passed testing — nothing to demo yet")

    basic = state.basic_env_name or "sog"
    # The demo and real-run both launch the ST-Coscientist agent INSIDE the base env
    # (``conda run -n <basic> … _agent_probe``). If that env was never built (or was removed) every
    # launch dies with an opaque "no probe contract returned" straight from conda — the exact confusing
    # failure the DOCTOR already flags as "Base env '<basic>': ABSENT". Detect the absent base env up
    # front and skip with an actionable message instead of advertising a run that cannot work.
    if not _base_env_ready(conda, basic):
        io.warn(f"the agent's base env '{basic}' isn't built yet — skipping the live demo.")
        io.say("  finish `sog-setup` provisioning so the base env is created, then re-run the demo.")
        _print_needs_attention(io, attn)
        return _skip(io, log, f"base env '{basic}' absent — the agent can't launch")

    io.section("Try out your new setup")
    _log(log, "demo_phase_start", passed=len(passed), needs_attention=len(attn))
    # One gate for the whole phase: it prompts at most once (on the first launch attempt, demo or
    # real-run) and a key synced for the demo carries into the real run without re-asking.
    gate = _KeyGate(state, llm)
    actions: list[str] = []
    # Index the installed+healthy tools by their import module, so the reassurance net can tell the
    # user that a 'No module named X' the agent improvised is actually a built, healthy tool.
    installed_modules = installed_module_index(state, specs)

    # D1 — a quick demo on a tool that passed testing.
    if io.ask_yesno("Run a quick demo on one of your installed tools?", default=False):
        picked = _pick_tool(io, passed, cats)
        if picked:
            done = _demo_on_tool(picked, io, conda, basic, cats, log, gate, installed_modules=installed_modules)
            if done:
                actions.append(done)

    # D2 — the user's own task on the freshly-installed env.
    if io.ask_yesno("Run your own task now (your data, your question)?", default=False):
        done = _real_run(io, conda, basic, log, stated_goals, gate, installed_modules=installed_modules)
        if done:
            actions.append(done)

    _print_needs_attention(io, attn)
    if actions:
        _log(log, "demo_phase_done", actions=actions)
        return DemoOutcome(skipped=False, reason="", actions=actions)
    return _skip(io, log, "demo + real-run both declined")


# --------------------------------------------------------------------------- #
# D1 — demo on a passed tool
# --------------------------------------------------------------------------- #
def _demo_blocker(server_key: str, cats: list[Category]) -> str:
    """Why ``server_key`` cannot be demoed on the bundled data, or ``""`` when it can."""
    return DEMO_UNSTAGEABLE_CATEGORIES.get(_server_category(server_key, cats), "")


def _pick_tool(io: PromptIO, passed: list[str], cats: list[Category]) -> str:
    """Pick which passed tool to demo. A single passed tool is auto-selected; more than one is a
    numbered menu (Q2 "ask me at demo time"), each labelled with its category for orientation.

    A tool whose task the bundled single slide cannot pose is left out of the menu, with its reason,
    rather than demoed into a failure on a tool that passed Tier-1 (hunt 2026-09-30,
    u37-setup-checks-15). ``""`` when nothing is left to demo."""
    blocked = [(k, _demo_blocker(k, cats)) for k in passed]
    for key, why in blocked:
        if why:
            io.note(f"not offering {key} for the demo: it {why}.")
    passed = [k for k, why in blocked if not why]
    if not passed:
        io.note("none of your tools that passed testing can be demoed on the bundled data — skipping the demo.")
        return ""
    if len(passed) == 1:
        io.note(f"demoing your one tool that passed testing: {passed[0]}")
        return passed[0]
    options: list[tuple[str, str, str]] = []
    for k in passed:
        cat = _server_category(k, cats)
        options.append((k, k, f"category: {cat}" if cat else ""))
    return io.select("Which installed tool should I demo?", options, default=passed[0])


def _demo_on_tool(
    picked: str,
    io: PromptIO,
    conda: Conda,
    basic: str,
    cats: list[Category],
    log: SessionLog | None,
    gate: _KeyGate | None = None,
    installed_modules: dict[str, str] | None = None,
) -> str:
    """Prepare (root, prompt, add_data) for ``picked`` and launch-or-emit. Returns ``"demo:<tool>"``
    on a run/emit, or ``""`` if no dataset could be prepared (surfaced, phase continues)."""
    plan = _demo_plan(picked, cats)
    if plan is None:
        why = constants.mini_data_note()
        io.warn(
            f"couldn't prepare a demo for '{picked}' — no bundled dataset available"
            + (f" ({why})" if why else "")
            + "; skipping it."
        )
        _log(log, "demo_no_dataset", tool=picked)
        return ""
    root, prompt, add_data, blurb = plan
    io.say(f"  demo → {picked}: {blurb}")
    _run_or_emit(
        io, conda, basic, root, prompt, add_data, kind="demo", log=log, gate=gate, installed_modules=installed_modules
    )
    return f"demo:{picked}"


def _demo_plan(picked: str, cats: list[Category]) -> tuple[str, str, dict[str, str] | None, str] | None:
    """The concrete demo inputs for ``picked``:

    * ``scanpy_spatial`` **and** the V1 lymph-node slide present → that real 243 MB Visium section,
      registered via ``add_data`` exactly like the Part-B tutorial driver.
    * any other tool → the shared mini dataset staged into a throwaway data lake + the tool's
      category prompt.

    ``None`` when no dataset can be staged (a fresh clone with neither mini nor fallback)."""
    if picked == _SCANPY_DEMO_SERVER:
        lymph = constants.mini_data_dir() / _V1_LYMPH_FILE
        if lymph.is_file():
            root = _data_root("demo")
            blurb = "spatial domains on a real human lymph-node Visium slide"
            return str(root), _LYMPH_PROMPT, {str(lymph): _LYMPH_DESC}, blurb
        # fall through to the mini-dataset path if the big slide isn't on disk
    # Resolve the category first: the staged inputs depend on it. Deconvolution/mapping tools need a
    # matching single-cell reference staged alongside the slide (a demo that staged only the slide
    # made cell2location refuse for lack of a reference); image/segmentation tools need a slide that
    # carries an H&E histology image (the plain expression slide has none to segment).
    category = _server_category(picked, cats)
    try:
        staged = testdata.stage_mini_dataset(
            dest_root=_data_root("demo"),
            # A tool that maps a reference onto space needs one outside deconvolution too (u37-setup-checks-15).
            with_sc_ref=(category == "deconvolution" or picked in DEMO_NEEDS_SC_REF),
            image=(category == "cell_segmentation"),
        )
    except OSError:
        # Not just a missing source file (FileNotFoundError): a full disk (ENOSPC), a read-only or
        # permission-denied data root all surface as OSError from the copy. Any of them means "no
        # dataset could be staged" — honor the documented None contract rather than let the error
        # escape the demo planner (the wizard's _phase_demo soft-wrap would catch it, but returning
        # None keeps the "no demo" decision here where the caller expects it).
        return None
    base = prompt_for(category)
    # Named by ABSOLUTE path: the agent pins a tool the prompt names ("Please use GraphST") only when
    # the prompt also names an existing absolute .h5ad, so with the bare filename the pin _name_tool
    # promises never fired and the agent routed freely (hunt 2026-09-30, u37-setup-checks-7).
    slide = str(staged.h5ad)
    sc_name = str(staged.sc_ref) if staged.sc_ref else None
    # Name the picked method so the agent runs *this installed tool* rather than its catalog-ranked
    # P1 default (often an uninstalled tool → an in-base-env 'No module named' improvisation). Only
    # names that resolve deterministically in the agent are in the map; anything else (e.g. sedr)
    # falls back to the plain category prompt, and the reassurance net covers any routing quirk.
    name = display_name(picked)
    if name:
        prompt = _with_file(_name_tool(base, name), slide, sc_ref=sc_name, image=staged.has_image)
        blurb = f"running {name} on the bundled mini dataset"
    else:
        prompt = _with_file(base, slide, sc_ref=sc_name, image=staged.has_image)
        blurb = "a quick analysis on the bundled mini dataset"
    return str(staged.root), prompt, None, blurb


# --------------------------------------------------------------------------- #
# D2 — the user's own task
# --------------------------------------------------------------------------- #
def _real_run(
    io: PromptIO,
    conda: Conda,
    basic: str,
    log: SessionLog | None,
    stated_goals: str,
    gate: _KeyGate | None = None,
    installed_modules: dict[str, str] | None = None,
) -> str:
    """Collect the user's plain-language task + data path and launch-or-emit. Returns ``"real"`` on a
    run/emit, or ``""`` when the user enters no task or no dataset can be prepared."""
    if stated_goals.strip():
        io.note(f"(earlier you said: {stated_goals.strip()})")
    # default="" — a blank task returns "" (skip), never auto-launches on a scripted run.
    query = io.ask_text("Describe your task in plain language (blank = skip)", default="")
    if not query.strip():
        io.note("no task entered — skipping the real run.")
        return ""

    demo_default = _demo_dataset_hint()
    data_path = io.ask_text("Path to your data (blank = use the bundled demo dataset)", default=demo_default)
    prepared = _real_inputs(query, data_path, demo_default, io)
    if prepared is None:
        return ""
    root, prompt, add_data = prepared
    _run_or_emit(
        io, conda, basic, root, prompt, add_data, kind="real", log=log, gate=gate, installed_modules=installed_modules
    )
    return "real"


def _demo_dataset_hint() -> str:
    """Path string of the bundled demo dataset (mini or tracked fallback), or ``""`` if none is on
    disk — the ``ask_text`` default for the real-run data path (Enter ⇒ use the demo dataset)."""
    src = testdata.resolve_source()
    return str(src) if src else ""


def _real_inputs(
    query: str, data_path: str, demo_default: str, io: PromptIO
) -> tuple[str, str, dict[str, str] | None] | None:
    """Resolve the real-run into (root, prompt, add_data):

    * blank path, or the demo-dataset default → stage the mini dataset (root = its data lake).
    * a real file the user typed → register it via ``add_data`` (root = a throwaway data root).
    * a real directory the user typed (a Space Ranger ``outs/``, a folder of tables) → registered the
      same way and named by its full path. It used to read as "no file" and the user's question ran on
      the bundled demo slide instead (hunt 2026-09-30, u37-setup-checks-16).
    * a non-existent path → warn and fall back to the bundled demo dataset.

    ``None`` when even the demo dataset can't be staged (nothing to run on)."""
    dp = (data_path or "").strip()
    use_demo = (not dp) or (dp == demo_default.strip())
    if not use_demo:
        abspath = os.path.abspath(os.path.expanduser(dp))
        if os.path.isdir(abspath):
            prompt = f"{query}\n\n(For this run, use the data I've provided in the directory '{abspath}'.)"
            return str(_data_root("real")), prompt, {abspath: "user-provided data directory for this task"}
        if not os.path.isfile(abspath):
            io.warn(f"no file at {abspath} — falling back to the bundled demo dataset.")
            use_demo = True
        else:
            prompt = (
                f"{query}\n\n(For this run, use the dataset I've provided as the file '{os.path.basename(abspath)}'.)"
            )
            return str(_data_root("real")), prompt, {abspath: "user-provided dataset for this task"}
    try:
        staged = testdata.stage_mini_dataset(dest_root=_data_root("real"))
    except OSError:
        # Missing source, full disk, or an unwritable data root all mean "nothing to run on" — see
        # the sibling except in _demo_plan. Fall back to the documented None (skip the real run).
        why = constants.mini_data_note()
        io.warn("no bundled dataset available to run on" + (f" ({why})" if why else "") + " — skipping the real run.")
        return None
    return str(staged.root), _with_file(query, staged.filename), None


# --------------------------------------------------------------------------- #
# D3 — key-gate: launch the agent now, or emit a runnable script + the command
# --------------------------------------------------------------------------- #
@dataclass
class _KeyGate:
    """Ensures a launchable LLM key is present when the user actually starts a run.

    Fires lazily on the first launch attempt (demo *or* real-run) and at most once. When the Tier-2
    key-gate is already satisfied it does nothing. Otherwise it offers — at the terminal — to reuse
    the key configured during setup (no re-typing) or to add one now, then syncs it into ``.env`` via
    the same ``assemble_owned_keys`` + ``write_dotenv`` + ``os.environ`` path onboarding uses
    (``onboarding.configure_key``). ``synced`` records that a key was written *this session*, so a
    just-provided key launches live even for a provider whose real key shape the strict Tier-2 gate
    would not recognise (e.g. Azure/custom)."""

    state: SetupState
    llm: object | None = None  # the in-memory LLMChoice from onboarding (None on resume)
    synced: bool = field(default=False, init=False)
    _done: bool = field(default=False, init=False)

    def ensure(self, io: PromptIO, log: SessionLog | None = None) -> None:
        """Idempotent: run the offer at most once, on the first launch attempt."""
        if self._done:
            return
        self._done = True
        if testing.llm_key_present():
            return  # happy path — a usable key is already visible; no prompt
        provider = self._provider()
        if provider is None:
            return  # unknown provider → _run_or_emit emits the runnable script, exactly as before
        io.say("")
        io.note(testing.llm_key_skip_reason())
        model = (self.state.llm or {}).get("model")
        # Prong 0 — pick from the saved-key library (each row labeled by its LLM). Purely additive:
        # when the vault holds anything we offer it first; "Use a different key" (or an empty/unusable
        # library, or a failed reconstruct) falls straight through to the unchanged Prong A/B below,
        # so an unvalidated setup key stays reachable and every existing gate test is byte-compatible.
        if self._pick_from_library(io, log):
            return
        # Prong A — reuse the SAME key from setup (no re-typing) when we hold a usable one.
        reuse = self._setup_values(provider)
        if reuse and io.ask_yesno(f"  Use the same {provider.label} key you set up earlier?", default=True):
            if onboarding.configure_key(io, provider, values=reuse, model=model, validate=False):
                self.synced = True
                _log(log, "demo_key_synced", source=provider.source, via="setup_reuse")
                return
        # Prong B — add a key now; collected via io.ask_secret (masked) and synced the same way.
        if io.ask_yesno(f"  Add your {provider.label} key now to run this live?", default=True):
            if onboarding.configure_key(io, provider, model=model, validate=True):
                self.synced = True
                _log(log, "demo_key_synced", source=provider.source, via="entered")
                return
        io.note("no key connected — saving a runnable script you can run once you add one.")

    def _pick_from_library(self, io: PromptIO, log: SessionLog | None) -> bool:
        """Prong 0 — offer the saved-key library, each row labeled by its LLM.

        Returns ``True`` (and sets ``synced``) when the user picks a saved key and it is reconstructed
        into ``.env``; ``False`` to fall through to Prong A/B — an empty/unusable library, "Use a
        different key", or a failed reconstruct. The chosen entry may be for a *different* provider than
        the one set up this session, which is the point (switch LLMs at launch). Best-effort: any vault
        or prompt hiccup falls through rather than blocking the run."""
        try:
            lib = key_library.load()
            opts = key_library.menu_options(lib)
            if not opts:
                return False
            choice = io.select(
                "  Connect a key to run this live",
                opts + [(key_library.OTHER_ID, "Use a different key", "")],
                default=opts[0][0],
            )
            if choice == key_library.OTHER_ID:
                return False
            entry = key_library.get(lib, choice)
            if entry is None:
                return False
            eprov = key_library._provider_for(entry.source)
            if eprov is None:
                return False
            if not onboarding.configure_key(
                io, eprov, values=entry.field_values, model=entry.model, validate=False, remember=False
            ):
                return False
            key_library.touch(entry.id, key_library._now())
            self.synced = True
            _log(log, "demo_key_synced", source=entry.source, via="library")
            io.ok(f"Reusing {key_library.label_for(entry)}")
            return True
        except Exception:  # vault/prompt is best-effort — fall through to Prong A/B on any hiccup
            return False

    def _provider(self) -> credentials.ProviderSpec | None:
        """Resolve the provider from the persisted ``state.llm`` (survives resume) or the in-memory
        ``LLMChoice``. ``None`` when nothing usable is configured — the gate then no-ops."""
        source = (self.state.llm or {}).get("source") or getattr(self.llm, "source", None)
        if not source:
            return None
        try:
            return credentials.get_provider(source)
        except KeyError:
            return None

    def _setup_values(self, provider: credentials.ProviderSpec) -> dict[str, str]:
        """The non-empty field values onboarding collected *this session*, but only when the secret is
        trustworthy — validated at setup, or at least not an obvious placeholder. An empty dict means
        "nothing to reuse", so the gate falls through to collecting a key (Prong B)."""
        fv = getattr(self.llm, "field_values", None) or {}
        secret_vars = [f.env_var for f in provider.required if f.secret]
        secrets = [fv.get(v, "") for v in secret_vars]
        if not any((s or "").strip() for s in secrets):
            return {}
        validated = bool(getattr(self.llm, "validated", False))
        if not validated and any(llm_setup.value_looks_like_placeholder(s) for s in secrets):
            return {}
        return {f.env_var: fv[f.env_var] for f in provider.all_fields() if fv.get(f.env_var)}


def _run_or_emit(
    io: PromptIO,
    conda: Conda,
    basic: str,
    root: str,
    prompt: str,
    add_data: dict[str, str] | None,
    *,
    kind: str,
    log: SessionLog | None,
    gate: _KeyGate | None = None,
    installed_modules: dict[str, str] | None = None,
) -> None:
    """The shared launch-or-emit gate for D1 + D2. With a usable key: run the agent out-of-process
    and report route + answer. Without: offer to connect one (via ``gate``), else write a
    self-contained runner and print the exact command — never echoing a secret."""
    cfg = constants.generated_mcp_config()
    if not cfg.exists():
        io.warn(f"no generated MCP config at {cfg} — run `sog-setup` through provisioning first; skipping.")
        _log(log, "demo_no_config", run_kind=kind)
        return
    if gate is not None:
        gate.ensure(io, log)
    if testing.llm_key_present() or (gate is not None and gate.synced):
        io.say("  launching the agent (this calls your LLM + the tool)…")
        # The data root is durable and shared across launches: the report embeds only what this run
        # wrote (2 s of slack for a coarse filesystem clock) (hunt 2026-09-30, u37-setup-checks-12).
        started = time.time() - 2
        # A live thinking box streams the agent's per-step reasoning as it runs (title carries no
        # "working" — the box appends its own " · working" status); disabled / non-TTY ⇒ inert, and
        # ``run_agent_probe`` falls back to the captured path byte-for-byte.
        with progress.think(
            getattr(conda, "progress", None),
            "ST-Coscientist",
            footer="reasoning live",
            fallback_stream=io.out,
            compact=True,
        ) as box:
            probe = testing.run_agent_probe(
                conda, basic, root, str(cfg), prompt, add_data=add_data, box=box, log_target=f"{basic}_{kind}"
            )
        _report_probe(
            io,
            probe,
            log,
            kind,
            basic=basic,
            installed_modules=installed_modules,
            data_root=root,
            known_tools=_configured_tool_names(cfg),
            since=started,
        )
    else:
        why = testing.llm_key_skip_reason()
        io.warn(f"not launching the agent now: {why}")
        script = _write_runner(kind, root, str(cfg), prompt, add_data)
        io.say(f"  saved a runnable script → {script}")
        io.say("  add a working LLM key to .env, then run:")
        io.say(f"      {_run_command(conda, basic, script)}")
        _log(log, "demo_script_written", run_kind=kind, script=str(script))


def _report_probe(
    io: PromptIO,
    probe: testing.ProbeResult,
    log: SessionLog | None,
    kind: str,
    *,
    basic: str = "",
    installed_modules: dict[str, str] | None = None,
    data_root: str | None = None,
    known_tools: set[str] | frozenset[str] | None = None,
    since: float | None = None,
) -> None:
    """Show the agent's route + answer (or the failure), and log a no-secret summary. When the agent
    reports a missing module that is in fact an installed+healthy tool, add a reassurance note so a
    routing quirk never reads as a broken install (``installed_modules`` from
    :func:`installed_module_index`). ``data_root`` is the run's data root — its output figures/files are
    embedded into the durable HTML report; ``since`` (epoch seconds, the launch time) keeps that to
    the files this run wrote. ``known_tools`` is the wired config's function names, which
    :func:`_route_hint` matches calls against."""
    if probe.timed_out:
        io.warn(f"the agent run timed out after {constants.AGENT_TEST_TIMEOUT_SEC}s (killed).")
        _log(log, "demo_probe", run_kind=kind, status="timeout")
        return
    if probe.error:
        # redact BEFORE the [:200] clip: probe.error is the raw parsed-contract error (testing.py,
        # un-redacted); a registered secret straddling char 200 would be split into an unmatchable head
        # fragment that io.warn's downstream redact can no longer mask, leaking to console AND the durable
        # JSONL. Redact the whole string first, then clip. Mirrors testing.py's redact(str(...))[:200]. (R27 B1)
        io.warn(f"the agent could not complete: {redact(str(probe.error))[:200]}")
        _reassure_installed(io, probe, basic, installed_modules)
        _log(log, "demo_probe", run_kind=kind, status="error")
        return
    route = _route_hint(probe.log, known_tools)
    if route:
        io.ok(f"the agent routed to: {route}")
    degraded = (getattr(probe, "degraded", "") or "").strip()
    if degraded:
        # A turn that stopped early (a rejected key, the step budget, a give-up) is not an answer, and
        # whatever text it left is shown as what it is (hunt 2026-09-30, uL4-honesty-6).
        io.warn(f"the agent stopped before finishing: {redact(degraded)[:200]}")
    answer = extract_solution(probe.answer or "").strip()
    if answer:
        # redact BEFORE _truncate: _answer_block redacts what it RECEIVES, but a pre-truncated answer
        # hides a secret straddling the _ANSWER_ECHO_CHARS cut — its head fragment survives the cut and
        # is unmatchable by the whole-substring redact, leaking to console + JSONL. Mask the full answer
        # first, then truncate (the transcript/HTML path already redacts the untruncated answer). (R27 B2)
        _answer_block(io, _truncate(redact(answer), _ANSWER_ECHO_CHARS))
    else:
        io.warn("the agent returned an empty answer (see the full transcript below).")
    _reassure_installed(io, probe, basic, installed_modules)
    _emit_transcript(io, probe, route, answer, basic=basic, kind=kind, data_root=data_root, since=since)
    status = "stopped_early" if degraded else "ok"
    _log(log, "demo_probe", run_kind=kind, status=status, answer_chars=len(answer), route=route)


def _route_hint(log_lines: list[str], known_tools: set[str] | frozenset[str] | None = None) -> str:
    """The tool function(s) the agent actually called, in call order — "which tool ran?".

    Reads ``<execute>``/fenced code and ``Tool:`` lines only, and skips every Human message (the
    prompt, carrying the recommender's skeleton and the post-analysis import, and the loop's nudges).
    With ``known_tools`` (the wired config's function names) a call counts when it is one of them;
    without, when it is a ``run_*`` call. Up to three names, comma-joined; ``""`` when none ran."""
    found: list[str] = []
    for entry in log_lines or []:
        text = str(entry)
        head, _, body = text.lstrip().partition("\n") if text.lstrip().startswith("=") else ("", "", text)
        if "human message" in head.lower() and not body.lstrip().startswith("<observation>"):
            continue
        code = "\n".join(a or b for a, b in _ROUTE_CODE_RE.findall(body))
        names = _ROUTE_TOOL_LINE_RE.findall(body) + _ROUTE_CALL_RE.findall(code)
        for name in names:
            if name in _NOT_A_ROUTE or name in found:
                continue
            if (name in known_tools) if known_tools else bool(_ROUTE_RE.fullmatch(name)):
                found.append(name)
    return ", ".join(found[:3])


def _configured_tool_names(cfg_path) -> frozenset[str]:
    """Every function name the wired MCP config declares (``spatialomicsgym_name`` / ``name``); empty
    when the file is unreadable, which makes :func:`_route_hint` fall back to ``run_*`` calls."""
    from pathlib import Path

    import yaml

    try:
        data = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError):  # ValueError covers a non-UTF-8 file
        return frozenset()
    servers = (data.get("mcp_servers") or data.get("mcpServers") or {}) if isinstance(data, dict) else {}
    names: set[str] = set()
    for meta in servers.values() if isinstance(servers, dict) else ():
        for tool in (meta.get("tools") if isinstance(meta, dict) else None) or []:
            if isinstance(tool, dict):
                name = tool.get("spatialomicsgym_name") or tool.get("name")
                if isinstance(name, str) and name:
                    names.add(name)
    return frozenset(names)


def _reassurance_notes(text: str, installed_modules: dict[str, str] | None) -> list[tuple[str, str]]:
    """``(server_key, module)`` for every ``No module named X`` in ``text`` where X is an installed &
    healthy tool. Deduped by server key, order-preserving. An empty index yields no notes, so a note
    can never be a false reassurance — only tools that passed testing are in ``installed_modules``."""
    if not installed_modules:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _NO_MODULE_RE.finditer(text or ""):
        module = m.group(1)
        top = module.split(".")[0].lower()
        key = installed_modules.get(top) or installed_modules.get(module.lower())
        if key and key not in seen:
            seen.add(key)
            out.append((key, module))
    return out


def _reassure_installed(
    io: PromptIO, probe: testing.ProbeResult, basic: str, installed_modules: dict[str, str] | None
) -> None:
    """Tell the user plainly that a tool the agent reported as a missing module is actually installed
    and healthy — the agent simply didn't route to the built worker this run. Scans the answer, the
    hard error, and the streamed log so a 'No module named' anywhere in the transcript is caught (the
    miss can surface as prose, a crash message, or a mid-run log line)."""
    parts = [probe.answer or "", getattr(probe, "error", "") or ""]
    parts.extend(str(line) for line in (probe.log or []))
    text = "\n".join(parts)
    notes = _reassurance_notes(text, installed_modules)
    if not notes:
        return
    io.section("Note on tool routing")
    io.note("the agent mentioned a missing module while improvising — but these tools ARE installed and healthy:")
    for key, module in notes:
        env = constants.tool_env_name(basic, key) if basic else f"the {key} env"
        io.say(
            f"  - {key}: '{module}' is present in its own env ({env}). The agent just didn't route to the "
            "installed worker this run — a routing quirk, not a broken install."
        )
    io.say('  Tip: name the method in your request (e.g. "use GraphST") to run that tool directly.')


# --------------------------------------------------------------------------- #
# Runner-script emission (key-absent path) — a self-contained, editable driver
# --------------------------------------------------------------------------- #
_RUNNER_TEMPLATE = '''#!/usr/bin/env python
"""Auto-generated by sog-setup (Part-D __KIND__ run). Runs ONE STCoscientist(...).go() on the
dataset you chose, through your setup-generated MCP config. Edit the prompt in REQUEST below and
re-run as often as you like.

Run it in the base env sog-setup built the agent into:

    __RUN_COMMAND__

You need a working LLM key in .env (SOG_LLM / SOG_SOURCE + the matching *_API_KEY). NO key is
stored in this script."""
from __future__ import annotations

import sys

REQUEST = __PAYLOAD__


def main() -> int:
    from spatialomicsgym.agent import STCoscientist

    req = REQUEST
    agent = STCoscientist(path=req["root"], expected_data_lake_files=[])
    if req.get("config_path"):
        agent.add_mcp(config_path=req["config_path"])
    if req.get("add_data"):
        agent.add_data(dict(req["add_data"]))
    print("=== prompt ===\\n" + req["prompt"] + "\\n\\n=== agent working... ===\\n", flush=True)
    result = agent.go(req["prompt"])
    answer = result[1] if isinstance(result, tuple) and len(result) == 2 else result
    # go() hands back the RAW final message: content blocks on a tool-use turn, and the ReAct
    # display scaffolding (<solution> tags, a "Classification:" routing label, the "Loop N -"
    # thinking-protocol preamble) that models emit inside their own answer. clean_answer is the
    # package's one display-layer cleanup -- the same one the CLI and web UI use.
    from spatialomicsgym import clean_answer

    print("\\n=== agent answer ===\\n", clean_answer(answer), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _write_runner(kind: str, root: str, cfg_path: str, prompt: str, add_data: dict[str, str] | None) -> Path:
    """Write ``state_dir()/<kind>_run.py`` — a stdlib-only driver that reproduces exactly what the
    probe would have run (same ``STCoscientist(path=root)`` + ``add_mcp`` + ``add_data`` + ``go``).
    Sentinel ``.replace`` (not ``str.format``) so the JSON payload's braces are never re-parsed."""
    import json

    constants.ensure_state_dirs()
    path = constants.state_dir() / f"{kind}_run.py"
    payload = {
        "root": root,
        "config_path": cfg_path,
        "prompt": prompt,
        "add_data": {str(k): str(v) for k, v in (add_data or {}).items()},
    }
    text = (
        _RUNNER_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, indent=4))
        .replace("__KIND__", kind)
        .replace("__RUN_COMMAND__", _run_command(None, "<base-env>", path))
    )
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    return path


def _run_command(conda: Conda | None, basic: str, script: Path) -> str:
    """The exact ``conda run`` line that executes ``script`` in the base env. ``--no-capture-output``
    so the agent's live progress reaches the terminal (omitted for micromamba, which rejects it)."""
    from .envtools import no_capture_output_args

    exe = (getattr(conda, "exe", None) or "conda") if conda is not None else "conda"
    # Shell-quoted: a checkout or state dir under a path with a space (``~/My Projects/...``) printed a
    # command that handed python half the script path (hunt 2026-09-30, u37-setup-checks-19).
    return shlex.join([exe, "run", *no_capture_output_args(exe), "-n", basic, "python", str(script)])


# --------------------------------------------------------------------------- #
# Small shared helpers
# --------------------------------------------------------------------------- #
def _with_file(text: str, filename: str, sc_ref: str | None = None, image: bool = False) -> str:
    """Append the staged-file harness note (mirrors the Tier-2 prompt) so the agent uses the real
    input already sitting in its data lake rather than inventing a dataset.

    When ``sc_ref`` is given (deconvolution/mapping), name BOTH inputs and say which is which — the
    slide and the matching single-cell reference are staged side by side in the data lake, and the
    agent must pair them (a demo used to stage only the slide → cell2location refused for lack of a
    reference). When ``image`` is set (cell/nucleus segmentation), point out that the staged slide
    carries its matching H&E histology image so the agent segments the real tissue image."""
    if sc_ref:
        return (
            f"{text}\n\n(For this run, use the spatial slide already staged in your data lake as the file "
            f"'{filename}', together with the matching single-cell reference '{sc_ref}' staged alongside it.)"
        )
    if image:
        return (
            f"{text}\n\n(For this run, use the Visium slide already staged in your data lake as the file "
            f"'{filename}', which includes its matching H&E histology image.)"
        )
    return f"{text}\n\n(For this run, use the dataset already staged in your data lake as the file '{filename}'.)"


def _name_tool(text: str, name: str) -> str:
    """Weave a natural, biologist-voice directive naming the method to run into the demo prompt.

    The "use <name>" phrasing is what the agent's ``_detect_user_specified_tool`` keys on to route to
    the user-named (installed) tool instead of its catalog-ranked default. The ``name`` is always one
    that resolves deterministically (:mod:`demo_tool_names`), so this reads like a real request *and*
    routes reliably. Ordered before ``_with_file`` so the tool name is the last resolvable token the
    detector sees (the file note that follows names no tool)."""
    return with_tool_named(text, name)


def _data_root(kind: str) -> Path:
    """A DURABLE data/output root under ``state_dir()`` (``.sog_setup/``) for a demo / real run.

    It sits beside the emitted ``<kind>_run.py`` runner and — unlike the per-run artifact dir — is never a
    cleanup target (``constants.assert_deletable_artifact`` rejects ``state_dir()``). So a key-absent script
    the user defers ("add a key and run it later") still finds its data + output root after a finalize
    cleanup or a re-run, instead of pointing at a directory ``test/installation`` cleanup has since deleted.
    """
    constants.ensure_state_dirs()
    root = constants.state_dir() / f"{kind}_run_data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _print_needs_attention(io: PromptIO, attn: list[tuple[str, str]]) -> None:
    """List the Lane-3 needs-attention tools (once, at the end) with their specific, actionable
    reason — so a non-passing tool reads as "here's what to do" rather than a bare FAIL."""
    if not attn:
        return
    io.section("Tools that need your attention")
    io.note("these didn't pass on their own — but the fix is NOT an env rebuild:")
    for key, reason in attn:
        io.say(f"  - {key}: {reason}")


# Neutralize C0 control chars + DEL in echoed agent text: keep newlines (line structure), turn tabs
# into a single space (so column math / ``ljust`` stays exact), drop everything else — a stray CR or
# ANSI escape in the answer must not corrupt the frame or smuggle terminal control sequences out.
_CTRL_MAP = dict.fromkeys(range(0x20))  # every C0 control → deleted (value None)
_CTRL_MAP[0x7F] = None
_CTRL_MAP[0x09] = ord(" ")
del _CTRL_MAP[0x0A]


def _visible(text: str) -> str:
    return (text or "").translate(_CTRL_MAP)


def _answer_block(io: PromptIO, text: str) -> None:
    """A light framed block for the agent's answer (ASCII-safe). Replaces the old ``----`` + raw echo.

    Redacts the WHOLE answer ONCE up front — before it is split into border-width chunks — so a
    registered secret can't slip through by straddling a chunk boundary (``io.say`` redacts each chunk
    in isolation, which would miss a secret split across two of them). Control chars are neutralized so
    a stray escape / carriage-return in the answer can't corrupt the frame."""
    enc = getattr(io.out, "encoding", None) or "ascii"
    try:
        "╭─╮│╰╯".encode(enc)
        tl, tr, bl, br, h, v = "╭", "╮", "╰", "╯", "─", "│"
    except (UnicodeEncodeError, LookupError):
        tl = tr = bl = br = "+"
        h, v = "-", "|"
    width = 72
    inner = width - 2  # content columns, with one space of padding inside each border
    body = _visible(redact(text or "(empty)")) or "(empty)"
    io.say("")
    io.say(f"{tl}{h} Result " + h * (width - 9) + tr)
    for line in body.splitlines() or ["(empty)"]:
        for chunk in (line[i : i + inner] for i in range(0, max(1, len(line)), inner)):
            io.say(f"{v} {chunk.ljust(inner)} {v}")
    io.say(bl + h * width + br)


def _emit_transcript(
    io: PromptIO,
    probe: testing.ProbeResult,
    route: str,
    answer: str,
    *,
    basic: str,
    kind: str,
    data_root: str | None = None,
    since: float | None = None,
) -> None:
    """Write the durable, self-contained HTML analysis report, print its path, and — only on a real
    interactive TTY — offer to scroll the full run in ``less``. Scripted / non-TTY / no-``less`` all
    fall through to a one-line "open it in a browser" note. When ``data_root`` is given, the run's output
    figures (embedded) + data files are surfaced in the report. Best-effort: an unwritable state dir just
    skips (``write_html`` returned ``None``); artifact scanning never raises."""
    steps = transcript.build_steps(probe.log)
    artifacts = transcript.scan_artifacts(data_root, since=since) if data_root else None
    html_path = transcript.write_html(
        basic=basic or "sog",
        kind=kind,
        title="ST-Coscientist",
        route=route,
        answer=answer,
        steps=steps,
        artifacts=artifacts,
    )
    if html_path is None:
        # Earlier lines already promised "the full transcript below" (empty-answer warning / truncation
        # pointer). If we couldn't write it, say so plainly instead of leaving a pointer to nothing.
        io.note("(couldn't write the transcript file — the state dir may be read-only.)")
        return
    io.say(f"  → full analysis report (figures, outputs, steps): {html_path}")
    text = transcript.render_text(title="ST-Coscientist", route=route, answer=answer, steps=steps, artifacts=artifacts)
    interactive = (
        bool(getattr(sys.stdin, "isatty", lambda: False)())
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
        and io.input_lines is None
        and not io.non_interactive
    )
    if not transcript.page(text, interactive=interactive):
        io.note("open that file in a browser for the full, collapsible run.")


def _truncate(text: str, limit: int) -> str:
    """Trim to ``limit`` at a line/word boundary (never mid-word), with a pointer to the full transcript."""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    nl = clipped.rfind("\n")
    sp = clipped.rfind(" ")
    cut = nl if nl >= limit // 2 else (sp if sp >= limit // 2 else limit)
    return clipped[:cut].rstrip() + "\n  … (truncated — open the full transcript below)"


def _skip(io: PromptIO, log: SessionLog | None, reason: str) -> DemoOutcome:
    io.note(f"demo/real-run: {reason}")
    _log(log, "demo_phase_skipped", reason=reason)
    return DemoOutcome(skipped=True, reason=reason)


def _log(log: SessionLog | None, event: str, **fields) -> None:
    # Our positional is ``event`` (not ``kind``) so callers may pass a ``kind=`` field. Belt-and-braces:
    # ``SessionLog.event(self, kind, **fields)`` owns the record keys ``kind`` (its positional) and ``ts`` —
    # a field of the same name would raise ``TypeError`` / overwrite the record. Re-key any such stray field
    # so this never-fail soft phase can't be crashed by a log call.
    if log is None:
        return
    for reserved in ("kind", "ts"):
        if reserved in fields:
            fields[f"{reserved}_"] = fields.pop(reserved)
    log.event(event, **fields)
