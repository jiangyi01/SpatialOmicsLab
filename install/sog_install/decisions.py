"""
The decision contract shared by the scripted and interactive paths.

Stage B has exactly one seam: every decision the wizard needs is produced by a
:class:`DecisionSource`. Two implementations sit behind it —
:class:`~sog_install.answers.ScriptedSource` (reads ``answers.yaml``,
no LLM) and :class:`~sog_install.guide.InteractiveGuide` (LLM-driven
recommend→confirm loop). Because they return the *same* dataclasses, the wizard
driver, engines, state, and resume logic are identical whether a human is being
guided or CI is feeding a file.

Every side-effecting engine call is gated by :meth:`DecisionSource.confirm`,
which receives a concrete :class:`Proposal` — the propose→confirm→execute gate.

Stdlib only (dataclasses + enum + abc).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


# --------------------------------------------------------------------------- #
# Enums (str-valued so they round-trip through YAML/JSON as plain strings)
# --------------------------------------------------------------------------- #
class _StrEnum(StrEnum):
    @classmethod
    def coerce(cls, value: str | _StrEnum) -> _StrEnum:
        if isinstance(value, cls):
            return value
        if value is None:
            # A present-but-null value must NOT silently coerce. ``str(None).strip().lower() ==
            # "none"`` would match a legal member like ``BuildStrategy.NONE`` — so a spec YAML with an
            # empty ``build_strategy:`` key (its value is None, and because the key is *present* the
            # ``d.get(..., "conda_export")`` default never applies) would be read as "nothing to
            # build" and the tool silently skipped, with no error. None is never a valid explicit
            # choice; reject it so the caller's ``except ValueError`` handler surfaces it as a
            # SpecError. (The other enums already raise on None — via the member-miss path below,
            # since none of them define a "none" member — so this only changes the buggy case.)
            raise ValueError(f"None is not a valid {cls.__name__} (expected one of {[m.value for m in cls]})")
        norm = str(value).strip().lower()
        for member in cls:
            if member.value == norm:
                return member
        raise ValueError(f"{value!r} is not a valid {cls.__name__} (expected one of {[m.value for m in cls]})")


class BaseEnvMode(_StrEnum):
    REUSE = "reuse"  # reuse the current conda env as base, install missing core pkgs (opt-in, install-only)
    NEW = "new"  # create a fresh <basic> env (recommended default for "don't touch my setup")


class OnToolFail(_StrEnum):
    CONTINUE = "continue"  # one bad tool never sinks the run (default)
    ABORT = "abort"


class OnValidationFail(_StrEnum):
    RETRY = "retry"
    SWITCH = "switch"
    CONTINUE_UNVALIDATED = "continue_unvalidated"


class BuildStrategy(_StrEnum):
    ENV_YAML = "env_yaml"  # conda env create -f <tools_user/*_env.yaml>
    CONDA_EXPORT = "conda_export"  # conda env create -f <captured spec>
    CONDA_CLONE = "conda_clone"  # conda create --clone <present source env>
    PIP = "pip"  # pip install into a minimal env
    GITHUB_CREATION = "github_creation"  # fall back to the tool-creation flow
    NONE = "none"  # nothing to build (e.g. the `eval` server)


# --------------------------------------------------------------------------- #
# Stage-A decision (produced deterministically by onboarding.py)
# --------------------------------------------------------------------------- #
@dataclass
class LLMChoice:
    """The verified LLM configuration. ``field_values`` carries secrets only in
    memory for the ``.env`` write; it is never persisted to setup state."""

    source: str  # ALLOWED_SOURCES value
    model: str
    field_values: dict[str, str] = field(default_factory=dict)
    validated: bool = False
    on_validation_fail: OnValidationFail = OnValidationFail.RETRY


# --------------------------------------------------------------------------- #
# Stage-B decisions
# --------------------------------------------------------------------------- #
@dataclass
class CategorySelection:
    categories: list[str]  # >=1 of the 9 skill categories
    servers: list[str]  # expanded server keys
    rationale: str = ""


def expand_category_servers(
    categories: list[str],
    explicit: list[str] | None,
    ctx_categories: list[dict],
) -> list[str]:
    """Expand chosen categories to their server keys, order-preserving + de-duped.

    Shared by both decision sources so the scripted and LLM-guided paths always
    resolve the *same* server list. ``explicit`` (when given) is authoritative — an
    explicit pick is an explicit instruction, so EVERY named tool is kept: the ones
    inside the chosen categories first (in expansion order), then any pick that falls
    *outside* those categories appended. This mirrors the interactive tool-picker
    (which trusts ``chosen`` verbatim) and ``guide.plan_provision`` ("install all of
    them, even ones outside the stated goal"); the old intersection silently dropped
    a scripted ``servers:`` entry whose category the user did not also list.
    """
    by_name = {c.get("name"): c for c in ctx_categories}
    servers: list[str] = []
    for cat in categories:
        # `.get("tools") or []`, not `.get("tools", [])`: a category dict carrying a present-but-null
        # `tools:` (hand-edited categories YAML / partially-built ctx dict) yields None with the 2-arg
        # default, and `list.extend(None)` is a TypeError. `or []` coerces both absent and null.
        servers.extend(by_name.get(cat, {}).get("tools") or [])
    if explicit:
        in_cat = [s for s in servers if s in explicit]
        servers = in_cat + [s for s in explicit if s not in in_cat]
    seen: set[str] = set()
    return [s for s in servers if not (s in seen or seen.add(s))]


def categories_for_servers(servers: list[str], ctx_categories: list[dict]) -> list[str]:
    """The category names that own at least one of ``servers`` (order-preserving).

    Inverse of :func:`expand_category_servers`: when the user picks tools directly,
    this recovers which categories they span so ``state.selected_categories`` (and
    the later test-phase default) stays populated. Servers under no category (the
    picker's ``Other`` bucket) simply contribute no category name.
    """
    chosen = set(servers)
    # `.get("tools") or []` (see expand_category_servers): a present-null `tools:` → None →
    # set.intersection(None) is a TypeError; coerce both absent and null to the empty list.
    return [c.get("name") for c in ctx_categories if chosen.intersection(c.get("tools") or [])]


@dataclass
class BaseEnvDecision:
    mode: BaseEnvMode
    basic_env_name: str
    install_editable: bool = True  # pip install -e . into the base env
    source_env_yaml: str | None = None  # recipe for a NEW env
    rationale: str = ""


@dataclass
class ToolPlan:
    """The concrete plan for provisioning ONE tool's env."""

    server_key: str
    target_env: str  # always <basic>_<server>
    strategy: BuildStrategy
    est_gb: float | None = None
    gpu: bool = False
    service_keys: list[str] = field(default_factory=list)  # env vars this tool needs
    source_env: str | None = None  # clone/export source
    recipe: str | None = None  # path to an env yaml, if any


@dataclass
class ProvisionDecision:
    tools: list[ToolPlan]
    on_tool_fail: OnToolFail = OnToolFail.CONTINUE


@dataclass
class TestDecision:
    run_tier1: bool = True  # worker on mini data (no LLM)
    run_tier2: bool = False  # one agent.go per category (needs a key)
    categories_to_test: list[str] = field(default_factory=list)
    tier2_eval: bool = False  # compute ARI/NMI (never fails the install)


@dataclass
class CleanupDecision:
    delete_artifacts: bool = False  # test/installation/* only; never envs


# --------------------------------------------------------------------------- #
# The propose→confirm→execute gate
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    """A concrete, about-to-happen side effect, shown to the user before it runs."""

    action: str  # short verb phrase, e.g. "create env"
    detail: str  # the exact plan, e.g. "wtdemo_rctd via conda_clone"
    est_gb: float | None = None
    reversible: bool = True
    kind: str = "generic"  # create_env | install | delete_artifacts | write_env | ...


# --------------------------------------------------------------------------- #
# Context handed to a DecisionSource at each decision point
# --------------------------------------------------------------------------- #
@dataclass
class GuideContext:
    """Everything a decision source may condition on. Free-form ``extra`` keeps
    the contract stable as engines learn to pass more context."""

    stated_goals: str = ""  # the user's natural-language intent so far
    current_env: str | None = None  # CONDA_DEFAULT_ENV
    preflight: dict = field(default_factory=dict)  # CheckResult summary
    categories: list[dict] = field(default_factory=list)  # [{name, description, tools}]
    # picker groups: [(category_name, description, [ToolOption(key, label, hint, detail), ...]), ...]
    tool_groups: list = field(default_factory=list)
    selected: CategorySelection | None = None
    base_env: BaseEnvDecision | None = None
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# DecisionSource — the single Stage-B seam
# --------------------------------------------------------------------------- #
class DecisionSource(ABC):
    """Produces every Stage-B decision. Implementations: scripted vs LLM-guided.

    The engines call these methods; they never know which implementation is
    live. All side effects are gated through :meth:`confirm`.
    """

    @abstractmethod
    def choose_categories(self, ctx: GuideContext) -> CategorySelection: ...

    @abstractmethod
    def choose_base_env(self, ctx: GuideContext) -> BaseEnvDecision: ...

    @abstractmethod
    def plan_provision(self, ctx: GuideContext) -> ProvisionDecision: ...

    @abstractmethod
    def collect_service_key(self, env_var: str, ctx: GuideContext) -> str | None:
        """Return the value for a contextual tool credential, or ``None`` to skip
        that one tool. Called only for keys a *selected* tool actually needs."""

    @abstractmethod
    def choose_tests(self, ctx: GuideContext) -> TestDecision: ...

    @abstractmethod
    def choose_cleanup(self, ctx: GuideContext) -> CleanupDecision: ...

    @abstractmethod
    def confirm(self, proposal: Proposal, ctx: GuideContext) -> bool:
        """The gate: return True to execute the proposed side effect."""

    def ask(self, question: str, ctx: GuideContext) -> str:
        """Answer an ad-hoc user question. Default: no answer (scripted path)."""
        return ""
