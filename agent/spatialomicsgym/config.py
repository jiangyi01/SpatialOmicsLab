"""
SpatialOmicsLab Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.
"""

import os
import sys
from dataclasses import dataclass, field, fields


def _env_raw(name: str) -> str | None:
    """Return the stripped value of env var `name`, or None if unset or blank."""
    val = os.getenv(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _same_directory(a: str, b: str) -> bool | None:
    """Whether two path spellings name one directory (``~`` expanded, made absolute); ``None`` if unsure."""
    try:
        return os.path.abspath(os.path.expanduser(a)) == os.path.abspath(os.path.expanduser(b))
    except (OSError, ValueError, TypeError):
        return None


# Every warning below goes to stderr: ``stcoscientist --json`` / ``-q`` import this module before
# they divert stdout, and a WARNING line ahead of the JSON object broke ``json.loads(stdout)``
# (hunt 2026-09-30, uL6-parity-13).
_BOOL_TRUE = ("true", "1", "yes", "on")
_BOOL_FALSE = ("false", "0", "no", "off")


def _env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var. Accepts true/1/yes/on and false/0/no/off (case-insensitive).

    One consistent truthy set everywhere: several flags previously compared `== "true"`,
    so `SOG_COMMERCIAL_MODE=1` (or `yes`/`on`) silently stayed False.

    Blank/unset keeps the default, and so does a spelling neither set recognises -- with a
    warning, like `_env_int`/`_env_float` below. There used to be no falsy set: anything that
    was not one of the four truthy spellings returned False, which meant `y` (the spelling
    `answers.confirm` and five setup modules accept) silently switched off the two flags that
    ship on, `SOG_USE_TOOL_RETRIEVER` and `SOG_POST_ANALYSIS_ENABLED`. The web settings
    endpoint writes any non-blank string straight through to `.env`, so that value is one
    `POST /api/options` away.
    """
    val = _env_raw(name)
    if val is None:
        return default
    lowered = val.lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    print(
        f"WARNING: {name}={val!r} is not a valid boolean "
        f"(use one of {'/'.join(_BOOL_TRUE)} or {'/'.join(_BOOL_FALSE)}); keeping default {default}.",
        file=sys.stderr,
    )
    return default


def _env_int(name: str, default: int, lo: float | None = None, hi: float | None = None) -> int:
    """Parse an int env var; a malformed OR out-of-range value warns and keeps the default.

    Guards module import: an unparseable `SOG_TIMEOUT_SECONDS=xyz` used to raise
    ValueError at `default_config = SpatialOmicsGymConfig()` (module scope), bricking
    every `from spatialomicsgym.config import default_config` importer.

    ``lo``/``hi`` close the other half of that hole. The portal already refused an absurd value
    on the WRITE side (`server._NUMERIC_RANGES`) and nothing checked the READ side, so the same
    number arrived unchallenged from a hand-edited `.env` or an exported variable:
    `SOG_TIMEOUT_SECONDS=99999999999999999999` reached `config.timeout_seconds` intact. That is
    the exact value `server.py` says it is guarding against -- a giant timeout `OverflowError`s
    `thread.join` and no code runs again.

    Out of range **warns and keeps the default** rather than clamping. Clamping would silently
    answer a different question than the operator asked; keeping the default is visible in the
    log and leaves a working process.
    """
    val = _env_raw(name)
    if val is None:
        return default
    try:
        parsed = int(val)
    except (ValueError, TypeError):
        print(f"WARNING: {name}={val!r} is not a valid integer; keeping default {default}.", file=sys.stderr)
        return default
    if (lo is not None and parsed < lo) or (hi is not None and parsed > hi):
        print(f"WARNING: {name}={val!r} is outside {lo}..{hi}; keeping default {default}.", file=sys.stderr)
        return default
    return parsed


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    """Parse a float env var; a malformed OR out-of-range value warns and keeps the default.

    See :func:`_env_int` for why the range belongs on the read side and why it keeps the default
    rather than clamping.
    """
    val = _env_raw(name)
    if val is None:
        return default
    try:
        parsed = float(val)
    except (ValueError, TypeError):
        print(f"WARNING: {name}={val!r} is not a valid number; keeping default {default}.", file=sys.stderr)
        return default
    # `float("nan")` and `float("inf")` parse. A NaN temperature passes every comparison below and
    # 400s the provider on every turn; an infinite timeout is the same OverflowError by a different
    # spelling. Both are "not a valid number" in the only sense that matters here.
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        print(f"WARNING: {name}={val!r} is not a finite number; keeping default {default}.", file=sys.stderr)
        return default
    if (lo is not None and parsed < lo) or (hi is not None and parsed > hi):
        print(f"WARNING: {name}={val!r} is outside {lo}..{hi}; keeping default {default}.", file=sys.stderr)
        return default
    return parsed


#: Inclusive bounds for the numeric knobs, shared with the portal's write-side validator so one
#: table governs both. ``server.py`` refused an absurd value when the settings panel wrote it and
#: nothing checked what was READ back, so the same number arrived intact from a hand-edited
#: ``.env``. The names are the portal's (``temperature``, ``timeout``), because that validator is
#: the older caller and renaming its keys would be a bigger change than this fix.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {"temperature": (0.0, 2.0), "timeout": (1, 86400)}


@dataclass
class SpatialOmicsGymConfig:
    """Central configuration for SpatialOmicsLab agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = SpatialOmicsGymConfig()

        # Override specific settings -- an argument passed here outranks its SOG_* variable
        config = SpatialOmicsGymConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"

    Precedence: a constructor argument > its ``SOG_*`` environment variable > the attribute default.
    """

    def __new__(cls, *args, **kwargs):
        # Which fields the caller passed, so __post_init__ does not replace them with the
        # environment. It did: on any install with a .env (sog-setup writes SOG_LLM and
        # SOG_TIMEOUT_SECONDS) the "Override specific settings" example above silently yielded the
        # env values (hunt 2026-09-30, uL6-parity-20). Recorded here because the generated
        # __init__ cannot tell a passed value from a default.
        self = super().__new__(cls)
        names = [f.name for f in fields(cls)]
        object.__setattr__(self, "_explicit_fields", frozenset(names[: len(args)]) | frozenset(kwargs))
        return self

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    # The azure- prefix proves the provider (AzureOpenAI); the rest is the Azure *deployment*
    # name, and gpt-5* deployments auto-route through the Responses API.
    llm: str = "azure-gpt-6-astra"
    temperature: float = 0.7
    # Per-request LLM timeout + retries. Bounds a stalled provider so a single ReAct step can't hang
    # the whole run forever (a hang is unrecoverable — the stream driver only catches exceptions).
    # Generous by default so legitimate long gpt-5 reasoning isn't cut off.
    llm_request_timeout_seconds: float = 600.0
    llm_max_retries: int = 2
    # C6 candidate, OFF until its own arm: run the post-analysis verdict BEFORE a <solution> is accepted.
    pre_answer_verification: bool = False
    # B-5 candidate, OFF until its own arm: retrieval resources as a cacheable system prefix, query last.
    retrieval_query_last: bool = False
    # R6: a model call that failed on a rate limit / overload / 5xx is retried only if the retry starts
    # within this many wall seconds of the first call -- the failed calls' time counted, not only the
    # waits. Not a hard cap: the retry itself is not cut short, so the loop can run this plus one call's
    # time in the client (3 x 600 s at the defaults above, when the provider hangs).
    llm_transient_wait_seconds: float = 300.0
    # MCP handshake budget (session.initialize / list_tools). Distinct from timeout_seconds, which
    # bounds the tool CALL: a dead handshake is an environment problem, not a slow analysis. 30s is
    # ample for the thin portals shipped here; a deployment whose portals cold-start slowly (network
    # filesystem, container pull) raises it via SOG_MCP_HANDSHAKE_TIMEOUT.
    mcp_handshake_timeout_seconds: float = 30.0

    # Tool settings
    use_tool_retriever: bool = True

    # Data licensing settings
    commercial_mode: bool = False  # If True, excludes non-commercial datasets

    # Tuning settings — hot-pluggable, OFF by default
    tuning_enabled: bool = False  # Global on/off switch for hyperparameter tuning
    tuning_mode: str | None = None  # "benchmark_tuning.light", "benchmark_tuning.full", "adaptive_tuning"
    tuning_tools: list | None = None  # Per-tool enablement list; None = all tools when tuning_enabled
    tuning_tasks: list | None = None  # Per-task enablement list; None = all tasks when tuning_enabled
    tuning_strategy: str | None = None  # Force strategy: "grid", "random", "staged", "bayesian"

    # Benchmarking settings — hot-pluggable, OFF by default
    benchmarking_enabled: bool = False  # When ON: output inspection skill is mandatory before eval
    evaluation_enabled: bool = False  # When ON: metric computation is allowed after inspection
    benchmark_user_tools: bool = False  # When ON: include user MCP tools in benchmark runs

    # Post-task analysis — ON by default, auto-disabled during benchmarking
    post_analysis_enabled: bool = True  # When ON: scan/analyze/visualize after tool runs
    # Follow-on analyses the co-scientist may launch off its own review of a result. 0 keeps the
    # self-review and turns the acting off; postanalysis.next_step clamps to a hard ceiling of 3 so
    # one question can never become an unattended research programme.
    post_analysis_max_followup_rounds: int = 1

    # Self-repair -- ON by default, forced off whenever benchmarking_enabled is on (the same
    # "benchmarking wins" gate post_analysis uses; see agent/env_fallback.py and agent/rescue.py).
    # env_fallback: a tool whose OWN environment fails mid-turn is reported to the model once, with
    # run_in_general_env(code) offered for an in-process redo that the answer must disclose.
    # unsolved_rescue: a turn that ends on a give-up gets ONE more bounded ReAct round, sharing the
    # post-analysis follow-on budget (a turn never gets two extra rounds).
    env_fallback_enabled: bool = True
    unsolved_rescue_enabled: bool = True
    # The general env's interpreter when the operator names one; otherwise
    # sog_install.constants.general_python() looks in the conda roots for spatialomicsgym_env_general.
    general_env_python: str | None = None
    general_env_max_calls: int = 12  # per-turn cap on run_in_general_env calls
    # Where model-written Python runs. "inprocess" is today's behaviour -- one PythonREPL in this
    # process, as this user -- and the only value a scored run ever sees: tool.support_tools.
    # process_isolation() answers False whenever benchmarking_enabled is on, whatever this says.
    # "process" runs every cell in a separate worker process (tool/repl_host.py), which the portal
    # spawns as an unprivileged user when the box has one (sog_portal/boundary.py); sog-web selects it.
    repl_isolation: str = "inprocess"

    # Auto-research — a bounded, resumable investigation the user starts deliberately. Unlike the
    # follow-on rounds above, which ride along on an ordinary question, nothing here happens unless
    # someone asks for a research run; these two numbers are what stops one from running forever.
    # `research.loop` clamps both into hard ceilings, and refuses to start at all while
    # `benchmarking_enabled` is on.
    research_max_rounds: int = 4
    research_max_seconds: int = 3600

    # How much of the know-how corpus sits in the system prompt before anything has been
    # retrieved. "all" is today's behaviour and the only value a scored run ever sees --
    # know_how.enrolment forces it back whenever benchmarking_enabled is on, so setting this
    # cannot affect a benchmark. See that module for what "summaries" and "demand" contain.
    know_how_enrolment: str = "all"

    # Tier-2 know-how: the merged external skill packs under know_how/packs/. OFF by default, and
    # a scored run never loads them whatever this says -- know_how.enrolment.packs_enabled() is
    # False whenever benchmarking_enabled is on, so the scored prompt cannot see a pack. The
    # portal turns this on only after the retrieval measurement (D-056). The budget is the most
    # pack documents one turn's second retrieval pass may add; 0 disables that pass.
    know_how_packs_enabled: bool = False
    know_how_pack_budget: int = 3

    # User tool creation — OFF by default, must be explicitly enabled
    tool_creation_enabled: bool = False  # When ON: allow creating MCP tools from GitHub
    max_user_envs: int = 20  # Maximum conda envs for user-created tools
    strict_safety_audit: bool = True  # When ON: Phase 3.5b audits worker code for
    # dangerous patterns (os.system, eval, subprocess
    # shell=True, etc.). Turn OFF only for rapid
    # prototyping by advanced users.

    # Memory system (OFF by default — fully opt-in) — see memory_manager.py
    # When ON: STCoscientist remembers successful / failed tool creation attempts and
    # applies prior-attempt hints to reduce re-work. Memory is ADVISORY only:
    # it never overrides HARD GATE, safety audit, or Phase 5 tests.
    # When OFF: STCoscientist behaves identically to the pre-memory system. The memory
    # module is not even imported when this flag is False.
    # Must be set BEFORE STCoscientist(...) constructor is called; runtime changes
    # require an STCoscientist restart.
    memory_enabled: bool = False
    memory_path: str | None = None  # default: tools_user/.memory
    memory_max_attempts_per_tool: int = 10
    memory_staleness_days: int = 30
    memory_shortcut_enabled: bool = False  # if False, memory is hint-only, never shortcuts phases

    # Self-review system (OFF by default) — see tools_user/self_review.py
    # When ON: Phase 5 test failures are classified + remediated up to 5 times
    # before rolling back. Reduces transient failures (missing deps, schema
    # typos) by ~40-60% based on plan analysis.
    self_review_enabled: bool = False
    self_review_total_cap: int = 5
    self_review_total_budget_sec: int = 1800
    self_review_blocked_classes: list[str] = field(default_factory=list)

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = None

    # Third-party integrations
    protocols_io_access_token: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist (never over a constructor argument)."""
        # Restored at the end: an argument the caller passed outranks its environment variable.
        explicit = {name: getattr(self, name) for name in getattr(self, "_explicit_fields", ())}
        # Check for environment variable overrides (optional)
        # Support both old and new names for backwards compatibility
        # Route string overrides through _env_raw (strip + blank->None) so a whitespace-only value
        # (e.g. SOG_LLM=" ") falls back to the default instead of overriding with a broken blank the
        # way a malformed int already does.
        _path = _env_raw("SOG_PATH") or _env_raw("SOG_DATA_PATH")
        if _path:
            self.path = _path
        _data_path = _env_raw("SOG_DATA_PATH")
        # As directories, not strings -- ``./data`` and ``data/`` are one place -- and only when the
        # variables decide: a ``path=`` argument uses neither (hunt 2026-09-30, uL6-parity-8).
        if (
            "path" not in explicit
            and _env_raw("SOG_PATH")
            and _data_path
            and _same_directory(_data_path, _path) is False
        ):
            # The settings panel and sog-setup write SOG_DATA_PATH; SOG_PATH, which the launchers'
            # --path help names, outranks it here -- so a directory saved there was silently ignored
            # and reverted on restart. Said, not flipped: flipping would let setup's default
            # SOG_DATA_PATH=./data override an operator's exported SOG_PATH (hunt 2026-09-30, uL6-parity-8).
            print(
                f"WARNING: SOG_PATH={_path!r} and SOG_DATA_PATH={_data_path!r} disagree; using SOG_PATH. "
                "Unset one (the settings panel and sog-setup write SOG_DATA_PATH).",
                file=sys.stderr,
            )
        self.timeout_seconds = _env_int("SOG_TIMEOUT_SECONDS", self.timeout_seconds, *NUMERIC_RANGES["timeout"])
        _llm = _env_raw("SOG_LLM") or _env_raw("SOG_LLM_MODEL")
        if _llm:
            self.llm = _llm
        self.use_tool_retriever = _env_bool("SOG_USE_TOOL_RETRIEVER", self.use_tool_retriever)
        self.commercial_mode = _env_bool("SOG_COMMERCIAL_MODE", self.commercial_mode)
        self.temperature = _env_float("SOG_TEMPERATURE", self.temperature, *NUMERIC_RANGES["temperature"])
        self.llm_request_timeout_seconds = _env_float("SOG_LLM_REQUEST_TIMEOUT", self.llm_request_timeout_seconds)
        self.llm_max_retries = _env_int("SOG_LLM_MAX_RETRIES", self.llm_max_retries)
        self.retrieval_query_last = _env_bool("SOG_RETRIEVAL_QUERY_LAST", self.retrieval_query_last)
        self.pre_answer_verification = _env_bool("SOG_PRE_ANSWER_VERIFICATION", self.pre_answer_verification)
        self.llm_transient_wait_seconds = _env_float(
            "SOG_LLM_TRANSIENT_WAIT_SECONDS", self.llm_transient_wait_seconds, 0.0, 3600.0
        )
        self.mcp_handshake_timeout_seconds = _env_float("SOG_MCP_HANDSHAKE_TIMEOUT", self.mcp_handshake_timeout_seconds)
        _base_url = _env_raw("SOG_CUSTOM_BASE_URL")
        if _base_url:
            self.base_url = _base_url
        _api_key = _env_raw("SOG_CUSTOM_API_KEY")
        if _api_key:
            self.api_key = _api_key
        _source = _env_raw("SOG_SOURCE")
        if _source:
            self.source = _source

        # Tuning overrides
        self.tuning_enabled = _env_bool("SOG_TUNING_ENABLED", self.tuning_enabled)
        _tuning_mode = _env_raw("SOG_TUNING_MODE")
        if _tuning_mode:
            self.tuning_mode = _tuning_mode
        _tuning_strategy = _env_raw("SOG_TUNING_STRATEGY")
        if _tuning_strategy:
            self.tuning_strategy = _tuning_strategy

        # Benchmarking overrides
        self.benchmarking_enabled = _env_bool("SOG_BENCHMARKING_ENABLED", self.benchmarking_enabled)
        self.evaluation_enabled = _env_bool("SOG_EVALUATION_ENABLED", self.evaluation_enabled)
        self.post_analysis_enabled = _env_bool("SOG_POST_ANALYSIS_ENABLED", self.post_analysis_enabled)
        self.post_analysis_max_followup_rounds = _env_int(
            "SOG_POST_ANALYSIS_MAX_FOLLOWUP_ROUNDS", self.post_analysis_max_followup_rounds
        )
        self.env_fallback_enabled = _env_bool("SOG_ENV_FALLBACK_ENABLED", self.env_fallback_enabled)
        self.unsolved_rescue_enabled = _env_bool("SOG_UNSOLVED_RESCUE_ENABLED", self.unsolved_rescue_enabled)
        self.general_env_python = _env_raw("SOG_GENERAL_PYTHON") or self.general_env_python
        self.general_env_max_calls = _env_int("SOG_GENERAL_ENV_MAX_CALLS", self.general_env_max_calls, 0, 200)
        self.repl_isolation = _env_raw("SOG_REPL_ISOLATION") or self.repl_isolation
        self.benchmark_user_tools = _env_bool("SOG_BENCHMARK_USER_TOOLS", self.benchmark_user_tools)
        self.research_max_rounds = _env_int("SOG_RESEARCH_MAX_ROUNDS", self.research_max_rounds)
        self.research_max_seconds = _env_int("SOG_RESEARCH_MAX_SECONDS", self.research_max_seconds)
        _know_how_enrolment = _env_raw("SOG_KNOW_HOW_ENROLMENT")
        if _know_how_enrolment:
            self.know_how_enrolment = _know_how_enrolment
        self.know_how_packs_enabled = _env_bool("SOG_KNOW_HOW_PACKS", self.know_how_packs_enabled)
        self.know_how_pack_budget = _env_int("SOG_KNOW_HOW_PACK_BUDGET", self.know_how_pack_budget, 0, 10)

        # User tool creation overrides
        self.tool_creation_enabled = _env_bool("SOG_TOOL_CREATION_ENABLED", self.tool_creation_enabled)
        self.max_user_envs = _env_int("SOG_MAX_USER_ENVS", self.max_user_envs)

        # Memory system overrides (env precedence > attribute default)
        self.memory_enabled = _env_bool("SOG_MEMORY_ENABLED", self.memory_enabled)
        _memory_path = _env_raw("SOG_MEMORY_PATH")
        if _memory_path:
            self.memory_path = _memory_path
        self.memory_max_attempts_per_tool = _env_int("SOG_MEMORY_MAX_ATTEMPTS", self.memory_max_attempts_per_tool)
        self.memory_shortcut_enabled = _env_bool("SOG_MEMORY_SHORTCUT_ENABLED", self.memory_shortcut_enabled)

        # Self-review overrides (env precedence > attribute default).
        # Without these, parent-process attribute edits don't propagate to STCoscientist
        # subprocess invocations of self_review_loop.
        self.self_review_enabled = _env_bool("SOG_SELF_REVIEW_ENABLED", self.self_review_enabled)
        self.self_review_total_cap = _env_int("SOG_SELF_REVIEW_TOTAL_CAP", self.self_review_total_cap)
        self.self_review_total_budget_sec = _env_int(
            "SOG_SELF_REVIEW_TOTAL_BUDGET_SEC", self.self_review_total_budget_sec
        )

        # Protocols.io access token (prefer specific env vars)
        env_token = os.getenv("PROTOCOLS_IO_ACCESS_TOKEN") or os.getenv("SOG_PROTOCOLS_IO_ACCESS_TOKEN")
        if env_token:
            self.protocols_io_access_token = env_token

        for name, value in explicit.items():
            setattr(self, name, value)
        # Folded AFTER the restore, so a passed value is normalised like an environment one: restored
        # last, ``repl_isolation='PROCESS'`` or a typo stayed verbatim, and the worker selector compares
        # it to "process" exactly (hunt 2026-09-30, uL6-parity-20). Anything but the one other value
        # folds to the default: a typo cannot select a mode.
        self.repl_isolation = "process" if str(self.repl_isolation).strip().lower() == "process" else "inprocess"

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "commercial_mode": self.commercial_mode,
            "tuning_enabled": self.tuning_enabled,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


# Global default config instance (optional, for convenience)
default_config = SpatialOmicsGymConfig()
