"""
remediation_planner — the *agentic* last resort for a provision-phase build failure.

When the deterministic classifier (:mod:`envdoctor`) cannot name a build failure, and a
validated LLM chat handle plus an explicit opt-in are present, this module asks the wizard's
own :class:`~sog_install.llm_chat.ChatClient` for a **structured, env-only** fix plan
and executes it through the SAME guarded conda primitives every other setup mutation uses.

Safety is enforced in three layers, none of which trust the model:

1. **Schema** — the reply must be a single JSON object of ``{diagnosis, actions[], confidence}``
   where every action's ``kind`` is one of a closed set (``pip_install`` / ``pip_install_git`` /
   ``conda_install`` / ``recreate``). No free-form shell is ever accepted or executed.
2. **Allowlist** — every index / find-links / git host must be on the trusted lists shared with
   the deterministic repair (:data:`envdoctor.TRUSTED_INDEX_HOSTS` / ``TRUSTED_GIT_HOSTS``);
   every package / channel token must match a strict safe-name regex. A single violation
   discards the WHOLE plan (honest FAIL kept) — the planner is fail-closed.
3. **Guarded execution** — actions run only via ``Conda.create_named`` / ``pip_install`` /
   ``conda_install`` / ``remove_env`` (each dry-run-gated), on a ``<basic>_*`` env verified by
   :func:`constants.is_managed_env` / ``assert_deletable_env``. Health is then re-confirmed by
   re-classification; ``ok`` flips only on a genuine healthy re-pass.

Stdlib only. The heavy ``ChatClient`` is passed in (never imported at module top), so
``sog_install`` keeps its stdlib+pyyaml import boundary.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import constants
from .envdoctor import TRUSTED_CONDA_CHANNELS, TRUSTED_GIT_HOSTS, TRUSTED_INDEX_HOSTS, _host_of
from .session_log import redact

if TYPE_CHECKING:
    from .envtools import Conda
    from .prompts import PromptIO
    from .provision import ToolResult
    from .session_log import SessionLog
    from .specs import ToolSpec

_TRUTHY = {"1", "true", "yes", "y", "on"}

# The closed set of executable action kinds. Anything else discards the plan.
_ALLOWED_KINDS = frozenset({"pip_install", "pip_install_git", "conda_install", "recreate"})

# A pip requirement token: name, optional [extras], optional version constraints. No whitespace,
# no shell metacharacters — even though args go to ``subprocess`` as a list (never a shell), we
# stay defensive and fail closed on anything that isn't a clean requirement.
#
# The version-constraint tail uses **possessive** quantifiers (``++`` / ``*+``, Python >=3.11).
# ``!`` lives in BOTH the operator class ``[<>=!~]`` and the version-body class — that overlap makes
# a naive ``(...)*`` catastrophically backtrack (ReDoS) on an adversarial token like ``a!!!!…!@``,
# where a long run of ``!`` can be split between operator and body in exponentially many ways before
# the trailing junk finally forces a reject. Possessive quantifiers forbid that give-back, so each
# iteration consumes a deterministic chunk: the match verdict is identical for every real specifier
# (``torch==2.1.0+cpu``, ``foo!=1.0``, ``pkg[extra]~=1.4``) but a hostile input rejects in linear time.
_SAFE_PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?([<>=!~]=?[0-9A-Za-z._*+!-]++)*+$")
# A conda channel: a plain name or an ``owner/label`` (conda-forge, bioconda, pytorch, nvidia).
_SAFE_CHANNEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
# A github git requirement, with or without the pip ``git+`` prefix and optional ``@ref`` / ``.git``.
_GITHUB_URL_RE = re.compile(r"^(?:git\+)?https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?:\.git)?(?:@[\w./-]+)?$")


# --------------------------------------------------------------------------- #
# Structured plan
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FixAction:
    kind: str
    packages: tuple[str, ...] = ()
    extra_index_url: str = ""
    index_url: str = ""
    find_links: str = ""
    channels: tuple[str, ...] = ()
    git_url: str = ""

    def pip_index_flags(self) -> list[str]:
        """The ``--extra-index-url`` / ``--index-url`` / ``--find-links`` flags for a pip action
        (each already host-validated). Order is deterministic for testability."""
        flags: list[str] = []
        if self.index_url:
            flags += ["--index-url", self.index_url]
        if self.extra_index_url:
            flags += ["--extra-index-url", self.extra_index_url]
        if self.find_links:
            flags += ["--find-links", self.find_links]
        return flags


@dataclass(frozen=True)
class FixPlan:
    diagnosis: str = ""
    actions: tuple[FixAction, ...] = ()
    confidence: float = 0.0
    valid: bool = False
    reject_reason: str = ""

    def action_kinds(self) -> list[str]:
        return [a.kind for a in self.actions]


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are an environment-repair planner for a conda-based bioinformatics tool installer. "
    "A per-tool conda env failed to BUILD. Propose the smallest ENVIRONMENT-ONLY fix as JSON. "
    "You may ONLY install packages / recreate the env — never edit source code, never run shell. "
    "Return a single JSON object:\n"
    '{"diagnosis": "<one sentence>", "confidence": 0.0-1.0, "actions": [ '
    '{"kind": "pip_install"|"pip_install_git"|"conda_install"|"recreate", '
    '"packages": ["name==ver", ...], "extra_index_url": "https://..."|null, '
    '"index_url": "https://..."|null, "find_links": "https://..."|null, '
    '"channels": ["conda-forge", ...], "git_url": "https://github.com/owner/repo"|null} ] }\n'
    "Rules: custom pip indexes may ONLY be on pypi.org, files.pythonhosted.org, "
    "download.pytorch.org, or data.pyg.org. git_url must be an https github.com URL. "
    "For PyTorch/PyTorch-Geometric '+cpu'/'+ptNN' wheels use find_links "
    "'https://data.pyg.org/whl/torch-<ver>+<compute>.html' and extra_index_url "
    "'https://download.pytorch.org/whl/<compute>'. Keep it minimal and safe."
)


@dataclass(frozen=True)
class RemediationTurn:
    """A summary of ONE prior repair turn, fed back into the next turn's prompt so the planner is
    conditioned on what already failed. The :class:`~sog_install.installer_scientist.InstallerScientist`
    builds these from its transcript; the planner only renders them."""

    diagnosis: str = ""
    actions: tuple[str, ...] = ()
    stderr_tail: str = ""
    outcome: str = "still failing"  # e.g. "still failing" / "changed error" / "rejected plan"


def _user_prompt(spec: ToolSpec, target: str, stderr: str, recipe_pips: list[str], *, salient: str = "") -> str:
    # redact (a94#3): the recipe pip pins egress to the same third-party LLM as the build stderr
    # below — a recipe-embedded credential (``--extra-index-url https://user:TOKEN@host``, a
    # ``git+https://<token>@…`` pin) must be masked here too, exactly like its three siblings. This
    # was the one prompt field that skipped ``redact()``, so a registered index secret leaked raw.
    pins = redact(", ".join(recipe_pips[:60])) if recipe_pips else "(none captured)"
    prompt = (
        f"Tool: {spec.server_key}\nEnv to build: {target}\n"
        f"Import that must work afterwards: {spec.import_check or spec.server_key}\n"
        f"Recipe pip pins: {pins}\n\n"
        # redact-THEN-clip (a94#3): the build stderr egresses to a third-party LLM, so any registered
        # secret it echoed (a credential-bearing --extra-index-url, a git+https token, a leaked key) must
        # be masked first. Redacting before the slice also masks a secret straddling the 6000-char cut.
        f"Build error (all attempted strategies):\n{redact(stderr)[:6000]}\n\n"
    )
    if salient.strip():
        # The log monitor's diagnosis-dense lines, pulled from the persisted build log — handed to the
        # planner as a focused hint on top of the raw build error (which may bury these in noise). Only
        # populated on the opt-in installer_scientist path; the provision single-shot leaves it empty, so
        # its prompt stays byte-for-byte what it was.
        prompt += (
            "SALIENT LOG LINES (the most diagnosis-dense lines the log monitor extracted — focus your "
            f"fix on these):\n{redact(salient)[:2000]}\n\n"  # a94#3: same LLM egress, same redact-then-clip
        )
    return prompt + "Give the JSON fix plan."


def _render_transcript(transcript) -> str:
    """Render prior :class:`RemediationTurn`s as a compact 'do not repeat' history block."""
    lines: list[str] = []
    for i, t in enumerate(transcript, 1):
        acts = ", ".join(t.actions) if t.actions else "(no valid action)"
        lines.append(f"- attempt {i}: tried [{acts}] → {t.outcome or 'still failing'}")
        if t.stderr_tail:
            lines.append(f"    resulting error tail: {redact(t.stderr_tail)[:600]}")  # a94#3: LLM egress
    return "\n".join(lines)


def _user_prompt_iter(
    spec: ToolSpec, target: str, stderr: str, recipe_pips: list[str], transcript, *, salient: str = ""
) -> str:
    """The turn-N prompt: the base build-error prompt, plus (only when non-empty) a history of the
    prior turns. With an EMPTY transcript AND no ``salient`` this is byte-identical to
    :func:`_user_prompt`, so the single-shot delegate below reproduces the original single-turn
    behavior exactly (the provision path never threads ``salient``)."""
    base = _user_prompt(spec, target, stderr, recipe_pips, salient=salient)
    if not transcript:
        return base
    return (
        base
        + "\n\nPrevious repair attempts THIS session — do NOT repeat an action that already failed; "
        + "propose a DIFFERENT, minimal env-only fix or return no actions if nothing safe remains:\n"
        + _render_transcript(transcript)
    )


# --------------------------------------------------------------------------- #
# Parse + validate (fail-closed)
# --------------------------------------------------------------------------- #
def _valid_index(url: str) -> bool:
    return bool(url) and url.startswith("https://") and _host_of(url) in TRUSTED_INDEX_HOSTS


def _validate_action(raw: dict) -> tuple[FixAction | None, str]:
    """Return ``(action, "")`` if this raw action is well-formed and allowlisted, else
    ``(None, reason)``. Fail-closed: any unrecognized field value rejects it."""
    if not isinstance(raw, dict):
        return None, "action is not an object"
    kind = str(raw.get("kind", "")).strip()
    if kind not in _ALLOWED_KINDS:
        return None, f"disallowed kind {kind!r}"

    packages = raw.get("packages") or []
    if not isinstance(packages, list) or any(not isinstance(p, str) for p in packages):
        return None, "packages must be a list of strings"
    packages = [p.strip() for p in packages if p.strip()]
    for p in packages:
        # Bound the token before the regex (belt-and-suspenders alongside the possessive
        # quantifiers in _SAFE_PKG_RE): a genuine requirement is short, and clipping here mirrors
        # the module's other input clips (diagnosis [:300], reject_reason [:200], salient [:800]).
        if len(p) > 128:
            return None, f"package token too long ({len(p)} chars)"
        if not _SAFE_PKG_RE.match(p):
            return None, f"unsafe package token {p!r}"

    channels = raw.get("channels") or []
    if not isinstance(channels, list) or any(not isinstance(c, str) for c in channels):
        return None, "channels must be a list of strings"
    channels = [c.strip() for c in channels if c.strip()]
    for c in channels:
        if not _SAFE_CHANNEL_RE.match(c):
            return None, f"unsafe channel token {c!r}"
        # `_SAFE_CHANNEL_RE` permits ``.`` and ``/``, so a token like ``conda-forge/../attacker`` is
        # "safe-looking" AND its base segment (``conda-forge``) is trusted — yet anaconda.org normalizes
        # the ``..`` and serves repodata from a DIFFERENT owner, whose package runs install-time
        # (pre/post-link) scripts → arbitrary code execution. Reject any ``.``/``..``/empty path segment
        # so the legitimate ``owner/label`` form still passes but traversal cannot escape the trusted
        # owner. (The base-segment trust check below then anchors that owner.)
        if any(seg in ("", ".", "..") for seg in c.split("/")):
            return None, f"unsafe channel path {c!r} (no '.', '..', or empty path segments)"
        # The base channel (before any ``/label``) must be one the wizard trusts — a plan may not
        # pull from an arbitrary conda channel even if the token is otherwise safe-looking.
        if c.split("/", 1)[0] not in TRUSTED_CONDA_CHANNELS:
            return None, f"untrusted conda channel {c!r} (allowed: {sorted(TRUSTED_CONDA_CHANNELS)})"

    def _url(key: str) -> str:
        v = raw.get(key)
        return v.strip() if isinstance(v, str) else ""

    extra_index_url, index_url, find_links = _url("extra_index_url"), _url("index_url"), _url("find_links")
    for label, u in (("extra_index_url", extra_index_url), ("index_url", index_url), ("find_links", find_links)):
        if u and not _valid_index(u):
            return None, f"untrusted {label} host: {u!r}"

    git_url = _url("git_url")
    if git_url:
        if not _GITHUB_URL_RE.match(git_url) or _host_of(git_url) not in TRUSTED_GIT_HOSTS:
            return None, f"untrusted git_url: {git_url!r}"

    # Kind-specific minimum requirements.
    if kind == "pip_install" and not packages:
        return None, "pip_install with no packages"
    if kind == "conda_install" and not packages:
        return None, "conda_install with no packages"
    if kind == "pip_install_git":
        if not git_url:
            return None, "pip_install_git with no git_url"
        git_url = git_url if git_url.startswith("git+") else f"git+{git_url}"

    return (
        FixAction(
            kind=kind,
            packages=tuple(packages),
            extra_index_url=extra_index_url,
            index_url=index_url,
            find_links=find_links,
            channels=tuple(channels),
            git_url=git_url,
        ),
        "",
    )


def parse_and_validate(raw: dict) -> FixPlan:
    """Turn the LLM's raw JSON into a validated :class:`FixPlan`. A single bad action rejects the
    whole plan (``valid=False``) — the planner never executes a partially-trusted plan."""
    if not isinstance(raw, dict):
        return FixPlan(reject_reason="reply was not a JSON object")
    diagnosis = str(raw.get("diagnosis", "")).strip()[:300]
    try:
        confidence = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    raw_actions = raw.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return FixPlan(diagnosis=diagnosis, confidence=confidence, reject_reason="no actions")
    actions: list[FixAction] = []
    for ra in raw_actions:
        action, reason = _validate_action(ra)
        if action is None:
            return FixPlan(diagnosis=diagnosis, confidence=confidence, reject_reason=reason)
        actions.append(action)
    return FixPlan(diagnosis=diagnosis, actions=tuple(actions), confidence=confidence, valid=True)


def plan_env_fix_iter(
    chat, spec: ToolSpec, target: str, stderr: str, recipe_pips: list[str], *, transcript=(), salient: str = ""
) -> FixPlan:
    """Ask the LLM for a fix plan, conditioned on the prior turns in ``transcript`` (a sequence of
    :class:`RemediationTurn`), and validate it. Never raises — a transport/parse failure returns an
    invalid plan with the reason. With an empty ``transcript`` and no ``salient`` this is exactly the
    single-shot planner.

    ``salient`` (optional) is the log monitor's diagnosis-dense tail for the CURRENT error, surfaced as
    a focused hint alongside the raw build stderr (deeper evidence, B3). Only the opt-in
    installer_scientist path threads it; the provision single-shot leaves it empty.

    This is the multi-turn seam the :class:`~sog_install.installer_scientist.InstallerScientist`
    drives: each turn it rebuilds, re-classifies the NEW stderr, appends a ``RemediationTurn``, and
    calls this again so turn-N's plan sees what turns 1..N-1 actually did."""
    try:
        raw = chat.chat_json(
            _SYSTEM,
            _user_prompt_iter(spec, target, stderr, recipe_pips, transcript, salient=salient),
            max_tokens=1600,
        )
    except Exception as exc:  # LLMChatError / anything — degrade to "no plan"
        return FixPlan(reject_reason=f"llm error: {exc}"[:200])
    return parse_and_validate(raw)


def plan_env_fix(chat, spec: ToolSpec, target: str, stderr: str, recipe_pips: list[str]) -> FixPlan:
    """Single-shot delegate (empty transcript) — preserves the exact behavior the provision path and
    ``test_remediation_planner.py`` expect."""
    return plan_env_fix_iter(chat, spec, target, stderr, recipe_pips, transcript=())


# --------------------------------------------------------------------------- #
# Guarded execution + health re-confirmation
# --------------------------------------------------------------------------- #
# How much of the actions' combined stderr to carry back into the NEXT turn's prompt. Enough for the
# LLM to see what its last plan actually produced, bounded so a giant traceback can't blow the context.
_STDERR_TAIL_BYTES = 1500


@dataclass
class _ExecReport:
    ran: list[str] = field(default_factory=list)
    ok: bool = True
    stderr_tail: str = ""  # last ~1.5 KB of the actions' combined stderr — fed back to the multi-turn planner

    def _absorb(self, text: str) -> None:
        """Append ``text`` to the rolling stderr tail, keeping only the last ``_STDERR_TAIL_BYTES``."""
        text = (text or "").strip()
        if text:
            # redact-THEN-clip (a94#3 / Round-21): this rolling tail is fed to the multi-turn planner and
            # egresses to a third-party LLM via _render_transcript (redact(t.stderr_tail)[:600]). That egress
            # redact only masks a FULLY-present secret, so a token straddling the -_STDERR_TAIL_BYTES cut here
            # would survive as an unmatchable fragment. Masking the combined text WHOLE before the clip (the
            # already-absorbed prefix is masked, so re-masking is a no-op) closes that straddle.
            combined = f"{self.stderr_tail}\n{text}" if self.stderr_tail else text
            self.stderr_tail = redact(combined)[-_STDERR_TAIL_BYTES:]


def _execute(
    conda: Conda, spec: ToolSpec, basic_env: str, target: str, plan: FixPlan, py_ver: str, *, io
) -> _ExecReport:
    """Run a validated plan's actions in order via the guarded conda primitives. Namespace-checked
    to ``<basic>_*`` before every mutation. A failing step is recorded but does not raise — the
    authoritative health re-check after all steps decides success."""
    rep = _ExecReport()
    # Defense-in-depth: the remediation loop repairs a per-tool ``<basic>_<server>`` env and NOTHING
    # else — never the base env, never a PROTECTED env (even one like ``sog_reproduce`` that falls
    # INSIDE the ``<basic>_*`` prefix; see ``constants.PROTECTED_ENVS``). ``is_managed_env`` alone is
    # too weak as the loop-wide gate: it returns True for ``target == basic_env`` AND for a protected
    # env in-namespace, so a ``pip_install``/``conda_install`` action below — which, unlike
    # ``recreate``, has no per-action re-check — could mutate the base or a protected env.
    # ``assert_deletable_env`` is exactly the right predicate (own it, not base, not protected); gating
    # the whole loop with it closes that hole for the non-destructive installs, mirroring the hard
    # re-check ``recreate`` already does right before removal. Non-raising contract preserved: a
    # refusal degrades to ``ok=False`` (with a loud note, since this should never fire in practice).
    try:
        constants.assert_deletable_env(basic_env, target)
    except PermissionError as exc:
        io.warn(f"  refusing remediation on {target!r}: {exc}")
        rep.ok = False
        return rep
    recreated = False  # a plan gets at most ONE destructive full-env rebuild (see below)
    for action in plan.actions:
        try:
            if action.kind == "recreate":
                # One plan → at most one recreate. A recreate nukes + rebuilds the whole env, so a
                # SECOND recreate in the same plan only destroys what the first built (plus any
                # installs since) — never useful. Worse, honoring N recreates runs N destructive
                # rebuilds while the caller's per-run budget counts the plan once, silently blowing
                # MAX_ENV_RECREATES (the multi-turn loop tallies +1/plan; the single-shot provision
                # path has no cross-plan cap at all). Skip — loudly — every recreate after the first.
                if recreated:
                    io.note(f"  skipping a duplicate env recreate in one plan for {target} (already rebuilt this plan)")
                    rep.ran.append("recreate:skipped-dup")
                    continue
                constants.assert_deletable_env(basic_env, target)  # raises unless <basic>_* & unprotected
                if conda.env_exists(target):
                    conda.remove_env(target)
                res = conda.create_named(target, python=py_ver or "3.11")
                recreated = True
            elif action.kind == "pip_install":
                res = conda.pip_install(target, [*action.pip_index_flags(), *action.packages], check=False)
            elif action.kind == "pip_install_git":
                res = conda.pip_install(target, [action.git_url], check=False)
            elif action.kind == "conda_install":
                res = conda.conda_install(target, list(action.packages), channels=list(action.channels), check=False)
            else:  # unreachable: parse_and_validate already closed the set
                continue
        except Exception as exc:  # a guarded primitive raising must not crash provisioning
            io.warn(f"  remediation step {action.kind} failed: {exc}")
            rep.ran.append(f"{action.kind}:error")
            rep._absorb(str(exc))
            rep.ok = False
            continue
        step_ok = bool(res.ok or res.dry_run)
        rep.ran.append(f"{action.kind}:{'ok' if step_ok else 'rc' + str(res.returncode)}")
        if not step_ok:  # only a FAILED step's stderr is useful to feed the next turn
            rep._absorb(getattr(res, "stderr", "") or "")
        rep.ok = rep.ok and step_ok
    return rep


def run_llm_remediation(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    result: ToolResult,
    stderr: str,
    chat,
    *,
    io: PromptIO,
    log: SessionLog | None = None,
) -> bool:
    """Entry point called by :func:`provision._run_llm_remediation`. Opt-in gated, fail-closed,
    env-only. Returns ``True`` iff the LLM plan rebuilt the env and its import now works.

    ``chat`` is the live ``ChatClient`` handle (``self.source.chat``) threaded from the wizard;
    the handle-present check happens at the call site, the ``SOG_PROVISION_LLM_REMEDIATION``
    opt-in is re-checked HERE so a scripted/CI run can never reach the network even if a handle
    leaks through."""
    if os.environ.get("SOG_PROVISION_LLM_REMEDIATION", "").strip().lower() not in _TRUTHY:
        return False
    if chat is None:
        return False
    from . import provision  # lazy: avoid an import-time cycle (provision imports us lazily too)

    recipe_pips, py_ver = provision.read_recipe_pip_and_python(getattr(spec, "recipe", "") or "")
    if log is not None:
        log.event("provision_remediate_start", server=spec.server_key, target=target, planner="llm")

    plan = plan_env_fix(chat, spec, target, stderr, recipe_pips)
    if log is not None:
        # Audit the plan WITHOUT leaking anything sensitive: diagnosis + action kinds + confidence.
        log.event(
            "provision_remediate_plan",
            server=spec.server_key,
            target=target,
            planner="llm",
            diagnosis=plan.diagnosis,
            actions=plan.action_kinds(),
            confidence=plan.confidence,
            valid=plan.valid,
            reject_reason=plan.reject_reason,
        )
    if not plan.valid:
        # `reject_reason` is always set by parse_and_validate on an invalid plan; the fallback is a
        # defensive guard against a future bare-invalid FixPlan. It reads "no reason given" — NOT
        # "low confidence": confidence is parsed + audited but never gates a plan, so naming it as the
        # rejection cause would fabricate a mode that does not exist (F2-sh).
        io.note(f"  LLM proposed no usable env-only fix ({plan.reject_reason or 'no reason given'})")
        return False

    io.say(f"  applying LLM fix for {target}: {plan.diagnosis or ', '.join(plan.action_kinds())}")
    rep = _execute(conda, spec, basic_env, target, plan, py_ver, io=io)
    result.repairs.append(
        {
            "kind": "off_index_wheel" if any(a.find_links or a.extra_index_url for a in plan.actions) else "llm",
            "action": "llm:" + "+".join(plan.action_kinds()),
            "attempted": True,
            "repaired": False,  # provisional; set True only on the health re-pass below
            "steps": rep.ran,
            "diagnosis": plan.diagnosis,
        }
    )

    healthy = provision.classify(conda, spec, target) is provision.ToolStatus.SKIP
    if healthy:
        result.ok = True
        result.built = True
        result.repairs[-1]["repaired"] = True
        result.strategy = f"{result.strategy or spec.build_strategy.value}+remediation:llm"
        if log is not None:
            log.event(
                "provision_remediated",
                server=spec.server_key,
                target=target,
                repaired=True,
                planner="llm",
                action="+".join(plan.action_kinds()),
            )
        return True
    if log is not None:
        log.event(
            "provision_remediated", server=spec.server_key, target=target, repaired=False, planner="llm", steps=rep.ran
        )
    return False
