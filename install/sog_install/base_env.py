"""
Base-env engine — establish the thin agent-core env that Stage B runs against.

Two modes (the decision is made upstream by a :class:`DecisionSource`):

* **NEW** (recommended default): ``conda env create -f spatialomicsgym_env.yml``
  into a fresh ``<basic>`` env, then ``pip install -e . --no-deps`` to register
  the package (the recipe installs deps but not the package itself).
* **REUSE** (opt-in, install-only): keep the current conda env, scan
  ``CORE_MODULES`` for anything missing, and — only if something is missing and
  the user confirms — ``pip install`` the requirements + ``-e . --no-deps``.
  This branch **never removes** anything, so reusing an env you already work in
  can't disturb it.

Every side effect passes through the ``confirm`` callback (the propose→confirm→
execute gate); with ``dry_run`` the underlying :class:`Conda` short-circuits and
nothing is created. Tier-2 validation (``_llm_ping`` inside the env) is optional
and never hard-fails the base-env step — a good env with a bad key is still a
good env.

Stdlib only (+ the sibling setup modules and ``spatialomicsgym.provider_names``, which is
itself stdlib only — that is why the provider rules live there rather than in ``llm``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from spatialomicsgym.provider_names import OVERRIDABLE_ENV_SOURCES, canonical_source, source_from_model_prefix

from . import constants
from .decisions import BaseEnvDecision, BaseEnvMode, Proposal
from .envtools import CondaError
from .session_log import redact

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .envtools import Conda, RunResult
    from .prompts import PromptIO
    from .session_log import SessionLog

# Fallback list if test/test_core_env.py isn't loadable from the launcher env (no test/ tree, no pytest).
_CORE_MODULES_FALLBACK: tuple[str, ...] = (
    "spatialomicsgym.config",
    "spatialomicsgym.llm",
    "spatialomicsgym.agent.stcoscientist",
    "spatialomicsgym.agent.execution",
    "spatialomicsgym.agent.prompt_builder",
    "spatialomicsgym.agent.tool_management",
    "spatialomicsgym.agent.mcp_integration",
    "spatialomicsgym.agent.mcp_config_merger",
    "spatialomicsgym.agent.empirical_leaderboard",
    "spatialomicsgym.agent.data_validation",
    "spatialomicsgym.model.retriever",
    "spatialomicsgym.tool.transcriptomics_skills",
    "spatialomicsgym.tool.tool_registry",
    "spatialomicsgym.benchmarking.output_inspector",
)

# Third-party runtime deps the agent-core imports LAZILY (inside function bodies on
# the add_mcp()/go() path), so a scan of only `spatialomicsgym.*` modules can't see
# them: a stale/REUSE env that predates one imports fine at Tier-1 but breaks at
# Tier-2 (the sole phase that calls add_mcp()). All are pinned in
# spatialomicsgym_env_requirements.txt, so the REUSE branch's install_requirements
# repairs any that are missing. Import names (not pip names): PyYAML → yaml.
REQUIRED_RUNTIME_DEPS: tuple[str, ...] = ("nest_asyncio", "mcp", "yaml")

# The LLM provider package is likewise imported lazily in llm.py per configured
# source. Probe only the provider the user actually configured, and only sources
# whose package the recipe pins — so we never false-block a provider the base
# requirements can't repair (Ollama's langchain_ollama and Bedrock's langchain_aws,
# which users install themselves). Gemini, Groq and Custom are served by llm.py
# through langchain_openai's ChatOpenAI, which the requirements pin, so they are
# probed for it like OpenAI (hunt 2026-09-30, u35b-setup-state-15).
_PROVIDER_IMPORT: dict[str, str] = {
    "OpenAI": "langchain_openai",
    "AzureOpenAI": "langchain_openai",
    "Anthropic": "langchain_anthropic",
    "Gemini": "langchain_openai",
    "Groq": "langchain_openai",
    "Custom": "langchain_openai",
}

BASE_ENV_EST_GB = 2.0  # the minimal agent-core env is ~1.6 GB installed


def core_modules() -> list[str]:
    """The import-health module list, preferring the development tree's own ``test/test_core_env.py``.

    Loaded by path (``test/`` is not a package); a checkout without ``test/`` -- a fresh clone, a
    wheel -- uses the built-in list, which mirrors it.
    """
    try:
        core_modules_list = constants.load_test_module("test_core_env").CORE_MODULES

        if core_modules_list:
            return list(core_modules_list)
    except Exception:
        pass
    return list(_CORE_MODULES_FALLBACK)


def _provider_module(source: str | None = None, model: str | None = None) -> list[str]:
    """The recipe-pinned langchain provider package the agent is going to import.

    Two axes, weighed in the order :func:`llm.resolve_source` weighs them, because the answer has
    to be the same answer: an explicit ``source`` decides outright; failing that, a model name
    that *proves* its provider outranks an overridable environment default; and the environment
    decides everything the name leaves open. ``SOG_SOURCE`` before ``LLM_SOURCE`` and ``SOG_LLM``
    before ``SOG_LLM_MODEL`` -- the precedences ``llm._env_source`` and ``config.py:192`` already
    use -- with the case-folding ``chat_cli._detect_source`` already applies. This reader used to
    invert the source precedence and match exact case only; it then read the environment alone.

    The model axis arrived late, and honestly so: while ``STCoscientist`` pre-filled its ``source``
    from ``config.source``, the agent's provider really *was* the environment value, and reading
    only that was right. Deleting the pre-fill let a named model outrank a stale default, and left
    this the last reader still describing the old behaviour.

    That matters more here than it looks, because **no production caller passes** either argument:
    the three ``scan_core_modules`` sites below pass neither, and both ``launch`` sites pass
    an explicit ``modules=`` list that skips ``probe_modules`` entirely. So the environment
    lane is not a fallback in practice, it is the whole decision -- and getting it wrong means
    we probe a package nothing will import, find the real one "not missing", report the env
    healthy, and skip the requirements install that would have repaired it.
    """
    resolved = canonical_source(source or os.getenv("SOG_SOURCE") or os.getenv("LLM_SOURCE"))
    if source is None:
        # ``resolve_source`` returns a caller-supplied source verbatim, model name unread -- so the
        # model only gets a vote when the caller named no provider. An unrecognised env spelling
        # leaves ``resolved`` None, which is also what ``resolve_source`` does with it.
        named = source_from_model_prefix(model or os.getenv("SOG_LLM") or os.getenv("SOG_LLM_MODEL"))
        if named is not None and (resolved is None or resolved in OVERRIDABLE_ENV_SOURCES):
            resolved = named
    pkg = _PROVIDER_IMPORT.get(resolved or "")
    return [pkg] if pkg else []


def probe_modules(source: str | None = None, model: str | None = None) -> list[str]:
    """Full import-health probe list.

    The agent-core ``spatialomicsgym.*`` modules **plus** the third-party runtime
    deps imported lazily on the tool-use path (``REQUIRED_RUNTIME_DEPS``) **plus**
    the configured provider's (recipe-pinned) package. The latter two are the
    lazy-import blind spot that a spatialomicsgym-only scan misses.

    ``source`` and ``model`` are both optional and both normally left unset: the provider is
    read from the environment the same way the agent reads it. Pass them when the caller already
    knows what the run will be configured with -- see :func:`_provider_module` for how the two
    are weighed against each other and against the environment.
    """
    return [*core_modules(), *REQUIRED_RUNTIME_DEPS, *_provider_module(source, model)]


def _recipe_asset(filename: str) -> Path:
    """A file from the committed ``agent/spatialomicsgym/spatialomicsgym_env`` tree.

    The instance-root copy first -- on a checkout that is the committed file itself, and it keeps
    the ``SOG_SETUP_REPO_ROOT`` test seam authoritative for planted trees. Off-checkout (pip-only
    install, where ``repo_root()`` is a seeded SOG_HOME that carries no package source) the wheel's
    own copy serves: ``include-package-data`` ships this tree into site-packages, and
    ``test/test_the_env_recipes_reach_a_non_editable_install.py`` pins that placement. When
    neither exists the instance-root path is still returned so the caller's error names the place
    an operator would be expected to put the file.
    """
    preferred = constants.agent_path("spatialomicsgym", "spatialomicsgym_env", filename)
    if preferred.exists():
        return preferred
    packaged = constants.PACKAGE_ROOT / "spatialomicsgym_env" / filename
    return packaged if packaged.exists() else preferred


def base_recipe() -> Path:
    """Path to the committed minimal agent-core recipe."""
    return _recipe_asset("spatialomicsgym_env.yml")


def requirements_file() -> Path | None:
    p = _recipe_asset("spatialomicsgym_env_requirements.txt")
    return p if p.exists() else None


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class BaseEnvResult:
    name: str
    mode: str
    ok: bool = False
    created: bool = False
    installed: bool = False
    skipped: bool = False  # confirm declined ⇒ no side effect
    missing_before: list[str] = field(default_factory=list)
    missing_after: list[str] = field(default_factory=list)
    ping: dict | None = None  # _llm_ping JSON, or None if not run
    messages: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
# ``python -c`` puts the working directory on ``sys.path[0]`` (as ""), so run from the repo root the
# probe imported ``spatialomicsgym`` from the checkout and an env without the package scanned healthy
# (hunt 2026-09-30, u35b-setup-state-4). Drop that entry: the question is what the ENV can import.
_PROBE_TEMPLATE = (
    "import importlib, json, sys\n"
    "if sys.path and sys.path[0] == '':\n"
    "    del sys.path[0]\n"
    "missing = []\n"
    "for m in {mods!r}:\n"
    "    try:\n"
    "        importlib.import_module(m)\n"
    "    except Exception:\n"
    "        missing.append(m)\n"
    "print(json.dumps(missing))\n"
)


def scan_core_modules(
    conda: Conda, env_name: str, *, modules: list[str] | None = None, source: str | None = None
) -> list[str]:
    """Return the probe modules that fail to import inside ``env_name``.

    Probes :func:`probe_modules` (agent-core modules + lazily-imported runtime
    deps + the configured provider package) unless the caller passes an explicit
    ``modules`` list. A read-only probe (``conda run -n <env> python -c …``); if
    the env is absent or the probe can't run, treat *all* as missing (caller
    decides).
    """
    mods = list(modules) if modules is not None else probe_modules(source)
    if not conda.env_exists(env_name):
        return list(mods)
    src = _PROBE_TEMPLATE.format(mods=list(mods))
    try:
        res = conda.run(env_name, ["python", "-c", src], timeout=constants.CONDA_RUN_TIMEOUT_SEC, check=False)
    except CondaError:
        # The probe itself couldn't run (timeout / launch OSError → `_exec` raises even with
        # check=False). Per the docstring contract, an unrunnable probe means "treat all as missing"
        # so the caller re-provisions — never let it unwind and brick an otherwise-healthy step.
        return list(mods)
    if not res.ok:
        return list(mods)
    # The probe prints one JSON line last; be tolerant of banner noise before it.
    for line in reversed((res.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("["):
            try:
                return list(json.loads(line))
            except json.JSONDecodeError:
                break
    return list(mods)


# --------------------------------------------------------------------------- #
# Proposals (shown before any side effect)
# --------------------------------------------------------------------------- #
def propose_create(name: str) -> Proposal:
    return Proposal(
        action="create base env",
        detail=f"{name} from spatialomicsgym_env.yml, then pip install -e . --no-deps",
        est_gb=BASE_ENV_EST_GB,
        reversible=True,
        kind="create_env",
    )


def propose_reuse_install(name: str, missing: list[str]) -> Proposal:
    n = len(missing)
    return Proposal(
        action="install missing core packages",
        detail=f"into current env {name} ({n} module{'s' if n != 1 else ''} missing) — install-only, nothing removed",
        est_gb=None,
        reversible=False,  # pip installs aren't cleanly reversible; flagged honestly
        kind="install",
    )


def propose_existing_editable(name: str) -> Proposal:
    """Shown before an editable install into an env the wizard did NOT just create (NEW mode
    whose chosen name already existed) — so a pre-existing env is never mutated unprompted."""
    return Proposal(
        action="install into existing env",
        detail=f"pip install -e . --no-deps into the pre-existing env {name} (install-only, nothing removed)",
        est_gb=None,
        reversible=False,
        kind="install",
    )


# --------------------------------------------------------------------------- #
# Executors (gated upstream by dry_run in Conda)
# --------------------------------------------------------------------------- #
def create_base(conda: Conda, name: str, *, recipe: Path | None = None) -> RunResult:
    r = recipe or base_recipe()
    # N12: check=False so a non-zero solve/pip exit comes back as ``ok=False`` carrying the FULL
    # stderr (``No matching distribution found for …`` etc.), not a truncated 8-line ``CondaError``
    # that would unwind past ``establish`` to the module-level exit-3 catch. ``establish``'s live
    # create-failed branch then persists the whole stderr and returns a friendly phase-fail. (A
    # timeout/OSError still raises ``CondaError`` — the wizard's ``_phase_base_env`` catches that.)
    return conda.create_from_yaml(str(r), name=name, check=False)


def install_requirements(conda: Conda, name: str) -> RunResult | None:
    req = requirements_file()
    if req is None:
        return None
    # check=False (as with create_base, #101): a failed `pip install -r` returns ok=False carrying
    # its stderr so establish()'s "requirements install failed" branch can report it — NOT a raised
    # CondaError that unwinds past _phase_base_env to the module-level exit-3 catch. (A timeout/OSError
    # still raises CondaError, which _phase_base_env catches.)
    return conda.pip_install(name, ["-r", str(req)], check=False)


def installed_from_spec() -> str | None:
    """Where *this* process's ``spatialomicsgym`` was installed from, as a pip-installable spec.

    THE BUG THIS EXISTS FOR. Off a checkout there is no source tree, so ``install_editable`` fell
    back to ``spatialomicsgym==<version>`` from PyPI -- and that package has never been published
    (``pypi.org/pypi/spatialomicsgym/json`` is a 404). README Lane A, the **first** row of the
    deployment table and the one headed "no clone needed", is ``pip install "git+https://..."``
    followed by ``sog-setup``. The wizard builds the ~1.6 GB conda env first, so the failure landed
    after the longest step of the run, with ``stopped at phase 'base_env'``. The escape hatch
    ``SOG_SETUP_PACKAGE_SPEC`` existed only in a code comment and one line of ``docs/PACKAGING.md``.

    pip already records the answer. PEP 610 writes ``direct_url.json`` into the ``.dist-info`` of
    anything installed from a URL or a path, so a Lane A install carries the exact git URL and the
    resolved commit it came from -- which is a better spec than the version number ever was, since
    it reinstalls *this* code rather than whatever a registry would hand back.

    Returns ``None`` when the distribution is absent or was installed from an index (no
    ``direct_url.json``), which is the case the version spec is genuinely right for.
    """
    try:
        from importlib.metadata import distribution

        raw = distribution("spatialomicsgym").read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        info = json.loads(raw)
        url = str(info.get("url") or "")
    except (ValueError, TypeError, AttributeError):
        return None
    if not url:
        return None
    vcs = info.get("vcs_info") if isinstance(info, dict) else None
    if isinstance(vcs, dict) and vcs.get("vcs"):
        # ``commit_id`` is what pip resolved the ref to, so the new env gets the same code and not
        # whatever the branch has moved to since.
        pin = str(vcs.get("commit_id") or vcs.get("requested_revision") or "").strip()
        spec = f"{vcs['vcs']}+{url}"
        return f"{spec}@{pin}" if pin else spec
    if url.startswith("file://"):
        # A wheel, an sdist, or a directory. Only offer it if it is still on this disk.
        from pathlib import Path
        from urllib.parse import unquote, urlsplit

        local = Path(unquote(urlsplit(url).path))
        return str(local) if local.exists() else None
    return url


#: The sentence an operator gets instead of a bare "stopped at phase 'base_env'".
#:
#: Reaching the version-spec fallback means the run is already doomed -- there is no
#: ``spatialomicsgym`` on any index -- and it fails *after* the ~1.6 GB conda build, which is the
#: longest step. Saying why, and naming the one environment variable that fixes it, is the
#: difference between a re-run that works and one that fails the same way.
PIP_ONLY_FALLBACK_HINT = (
    "this install has no source tree and pip recorded no origin for it, so the fallback spec "
    "'spatialomicsgym==<version>' was used -- no index publishes that package. Point "
    "SOG_SETUP_PACKAGE_SPEC at a wheel, an sdist or a git URL and re-run to resume from here."
)


def pip_only_fallback_in_use() -> bool:
    """True when an off-checkout install would reach the unpublished version spec."""
    if (constants.repo_root() / "pyproject.toml").is_file():
        return False
    return not (os.environ.get("SOG_SETUP_PACKAGE_SPEC") or installed_from_spec())


def install_editable(conda: Conda, name: str, *, no_deps: bool = True) -> RunResult:
    root = constants.repo_root()
    if (root / "pyproject.toml").is_file():
        args = ["-e", str(root)]
    else:
        # Pip-only install: ``repo_root()`` is a seeded instance root (SOG_HOME) with no source
        # tree to install editably -- ``pip install -e`` there would fail on a directory that is
        # not a project. Three candidates, most specific first: an operator's explicit override,
        # then wherever pip installed *this* process from (see :func:`installed_from_spec`), then
        # the version spec. The last is a genuine fallback only -- there is no such package on
        # PyPI, so reaching it means the run is about to fail and should say something useful.
        spec = os.environ.get("SOG_SETUP_PACKAGE_SPEC") or installed_from_spec()
        if not spec:
            from spatialomicsgym.version import __version__

            spec = f"spatialomicsgym=={__version__}"
        args = [spec]
    if no_deps:
        args.append("--no-deps")
    # check=False (as with create_base, #101): a failed editable install returns ok=False so
    # establish() records "pip install -e . failed" and returns a structured phase-fail, instead of
    # raising a CondaError that would flip a nearly-complete base env to a scary exit-3.
    return conda.pip_install(name, args, check=False)


def tier2_ping(conda: Conda, name: str) -> dict:
    """Run ``_llm_ping`` inside the env; return its JSON (never raises)."""
    try:
        res = conda.run(
            name,
            ["python", "-m", "sog_install._llm_ping"],
            timeout=constants.LLM_PING_TIMEOUT_SEC + constants.CONDA_RUN_TIMEOUT_SEC,
            check=False,
        )
    except CondaError as exc:
        # A firewalled / very slow LLM endpoint can make ``conda run`` exceed its timeout (or the
        # launch itself OSError), and ``_exec`` raises CondaError even with ``check=False``. A Tier-2
        # ping is *advisory* — a healthy env whose key can't reach the network yet is still a healthy
        # env (establish() only calls this once ``result.ok`` is already True). Degrade to a failed
        # ping so the phase stays green, instead of flipping a complete install to a scary exit 3.
        return {"ok": False, "error": redact(str(exc))[:200]}
    for line in reversed((res.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {"ok": False, "error": redact(res.stderr or res.stdout or "no output").strip()[:200]}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def establish(
    conda: Conda,
    decision: BaseEnvDecision,
    *,
    io: PromptIO,
    confirm: Callable[[Proposal], bool],
    log: SessionLog | None = None,
    do_ping: bool = True,
) -> BaseEnvResult:
    """Bring the ``<basic>`` base env into a usable state per ``decision``.

    ``confirm`` gates every side effect (pass ``lambda p: source.confirm(p, ctx)``).
    Returns a :class:`BaseEnvResult`; ``ok`` means the env exists and imports the
    core modules (Tier-2 ping result is advisory only).
    """
    name = decision.basic_env_name
    result = BaseEnvResult(name=name, mode=decision.mode.value)

    def _log(kind: str, **f):
        if log is not None:
            log.event(kind, **f)

    # M1 — the single mutation chokepoint. A protected env name reaches here from any entry point
    # (interactive typo, LLM-invented suggestion, or an ``--answers`` file — the last with ``--yes``
    # would auto-confirm the install), and BOTH branches below mutate: NEW ``pip install -e .`` into
    # the pre-existing env, REUSE installs the core requirements. Refuse before either side effect so
    # a reserved env (``base``, ``spatialomicsgym_env``, …) is never modified; return a clean
    # phase-fail (``ok`` stays False), not an exception — nothing was created or touched.
    protected_reason = constants.base_env_protection_reason(name)
    if protected_reason:
        result.messages.append(protected_reason)
        io.err(f"  {protected_reason}")
        _log("base_env_protected_refused", name=name, mode=decision.mode.value)
        return result

    if decision.mode is BaseEnvMode.REUSE:
        if not conda.env_exists(name):
            # Reuse means "install into an env that already exists". A missing target would
            # otherwise scan as "everything missing" and then fail cryptically at the first
            # `conda run -n <name>`; say so plainly instead.
            msg = (
                f"cannot reuse env {name!r} — no conda env by that name. Re-run and pick "
                "'Create a fresh env', or activate the env you meant to reuse first."
            )
            result.messages.append(msg)
            io.err(f"  {msg}")
            return result
        result.missing_before = scan_core_modules(conda, name)
        if not result.missing_before:
            io.ok(f"base env {name!r} already has the core packages — nothing to install")
            result.ok = True
        elif confirm(propose_reuse_install(name, result.missing_before)):
            io.say(f"  installing {len(result.missing_before)} missing core package(s) into {name!r}…")
            rq = install_requirements(conda, name)
            if rq is not None and not rq.ok and not rq.dry_run:
                result.messages.append(f"requirements install failed: {redact(rq.stderr or '')[:200]}")
            ed = install_editable(conda, name, no_deps=True)
            result.installed = ed.ok or ed.dry_run
            if not (ed.ok or ed.dry_run) and pip_only_fallback_in_use():
                result.messages.append(PIP_ONLY_FALLBACK_HINT)
            result.missing_after = scan_core_modules(conda, name)
            result.ok = not result.missing_after
        else:
            result.skipped = True
            result.messages.append("user declined the install; base env left unchanged")
            io.note("skipped — base env left exactly as it was")
    else:  # NEW
        exists = conda.env_exists(name)
        if exists:
            io.note(f"env {name!r} already exists — installing into it (not recreated)")
            _log("base_env_exists", name=name)
            result.created = False
        elif confirm(propose_create(name)):
            io.say(f"  creating base env {name!r} from the minimal recipe…")
            cr = create_base(conda, name)
            result.created = cr.ok or cr.dry_run
            if not (cr.ok or cr.dry_run):
                # LIVE since N12 (create_base now runs check=False): a solve/pip failure lands here as
                # a structured result instead of an exception. Persist the FULL stderr to the log for
                # diagnosis; keep the console message short. A clean phase-fail, not an exit-3 traceback.
                _log("base_env_create_failed", name=name, returncode=cr.returncode, stderr=cr.stderr)
                result.messages.append(f"create failed: {redact(cr.stderr or '')[:200]}")
                io.err(f"could not create {name!r} — see the run log for the full solver output")
                return result
        else:
            result.skipped = True
            result.messages.append("user declined base-env creation")
            io.note("skipped base-env creation")
            return result

        if decision.install_editable:
            # If the NEW-mode target already existed (we did not create it here), an editable
            # install would mutate an env the user may already rely on — gate it behind an explicit
            # confirm, mirroring the REUSE path. A freshly-created env needs no extra prompt
            # (propose_create already disclosed the `pip install -e .` step).
            if exists and not confirm(propose_existing_editable(name)):
                result.skipped = True
                result.messages.append("declined editable-install into the pre-existing env; left unchanged")
                io.note("skipped — existing env left exactly as it was")
                return result
            ed = install_editable(conda, name, no_deps=True)
            result.installed = ed.ok or ed.dry_run
            if not (ed.ok or ed.dry_run):
                result.messages.append(f"pip install -e . failed: {redact(ed.stderr or '')[:200]}")
                if pip_only_fallback_in_use():
                    result.messages.append(PIP_ONLY_FALLBACK_HINT)
        result.missing_after = scan_core_modules(conda, name)
        # In dry-run the env doesn't actually exist, so a non-empty scan is expected. A failed editable
        # install is a failed step even when the scan comes back clean: the package the scan imports
        # may be a stale or foreign copy, and the message above was otherwise never shown, under a
        # green "base env ready" (hunt 2026-09-30, u35b-setup-state-4).
        editable_failed = decision.install_editable and not result.installed
        result.ok = conda.dry_run or (not result.missing_after and not editable_failed)

    if result.ok and do_ping:
        result.ping = tier2_ping(conda, name)
        if result.ping.get("ok"):
            io.ok(f"LLM reachable from {name!r} (model {result.ping.get('model', '?')})")
        else:
            io.note(
                f"(LLM not reachable yet from {name!r}: {result.ping.get('error', 'unknown')} — "
                "you can still provision tools; fix the key later)"
            )

    _log(
        "base_env_done",
        name=name,
        mode=result.mode,
        ok=result.ok,
        created=result.created,
        installed=result.installed,
        skipped=result.skipped,
        missing_after=len(result.missing_after),
    )
    if result.ok:
        io.ok(f"base env {name!r} ready")
    return result
