"""
installer_scientist — a *separate*, install-specialized ReAct agent for the setup wizard.

This is the case-by-case self-healing loop the wizard runs at every install/test failure
surface. It borrows the shape of the bio agent's ReAct framework (:mod:`spatialomicsgym.agent.react`
— ``call_model`` → ``tool_node`` → ``should_continue``) but is a **different agent**: it reasons
about *environment build/runtime failures* and acts only through the guarded env primitives. It
does **no** biological analysis and it **never imports** :class:`~spatialomicsgym.agent.STCoscientist`
(nor vice-versa) — the two are kept cleanly apart, and this module stays stdlib-only at import time so
``import sog_install`` keeps its stdlib+pyyaml boundary (the heavy ``ChatClient`` is passed
in; :mod:`remediation_planner` / :mod:`provision` are imported lazily only when a plan is executed).

The mapping to :mod:`react`:

===========================  ==================================================================
``react.py``                 ``InstallerScientist``
===========================  ==================================================================
``AgentState.messages``      ``transcript`` = ``[_Obs(turn, lane, classify, action, …)]``
``call_model`` (reason)      :meth:`_reason` — deterministic ``classify_failure`` FIRST, then the
                             advisory ``diagnose`` lanes, then the gated LLM planner
``tool_node`` (act)          :meth:`_act` — guarded ``repair_env`` / ``repair_diagnosis`` /
                             validated planner ``_execute`` / a no-op retry
``should_continue``          the loop guard: stop on healthy / budget / wall-clock / *no progress*
``go() -> (log, answer)``    :meth:`remediate` → :class:`RemediationOutcome`
===========================  ==================================================================

**One turn** is *observe → reason → act → re-observe*: ``observe()`` returns the current failure text
(``""`` once the env is healthy); :meth:`_reason` picks a lane; :meth:`_act` runs the guarded repair
and captures its ``stderr_tail``; ``observe()`` is called again to see what the action actually did.
That fresh error — including the last action's ``stderr_tail`` — is fed into the next turn's LLM plan,
so *the logs participate in the loop*.

Three lanes, mirroring the taxonomy in :mod:`envdoctor`:

* **Lane 1 — deterministic repair** (``classify_failure`` → :func:`envdoctor.repair_env`, or a
  ``diagnose`` ``MISSING_SYSTEM_LIB`` → :func:`envdoctor.repair_diagnosis`): env-only, namespace-guarded,
  bounded by :data:`constants.MAX_ENV_REPAIRS`.
* **Lane 2 — hand-off** (an ambiguous ``diagnose`` HANDOFF, or an unrecognized failure when the LLM is
  enabled): the fail-closed multi-turn planner (:func:`remediation_planner.plan_env_fix_iter`),
  bounded by :data:`constants.MAX_REACT_STEPS`. Opt-in: without a chat handle + an enabled gate this
  lane is unavailable and the failure is surfaced honestly instead.
* **Lane 3 — surface honestly** (a ``diagnose`` SURFACE — token / special input / data shape / disk /
  DNS / GPU / unmapped ``.so`` / benign fallback): no mutation, a specific actionable message, and the
  loop *stops* — no env change would fix it. A **Lane 4** transient-network diagnosis retries the same
  build without an env change.

Every mutation goes through a primitive that is dry-run-gated and namespace-guarded to ``<basic>_*``;
this module adds no new mutation path. It never raises: a crashing closure or primitive degrades to
"not repaired", never taking down the phase.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import constants, envdoctor, log_monitor
from .envdoctor import DiagLane

# ``redact`` is imported from the stdlib-only ``session_log`` (the same import boundary
# ``remediation_planner`` uses) to redact-THEN-clip every RAW build-stderr tail before it is stored for
# the cross-turn planner / audit log — a clip-first would leave a secret fragment the egress redact misses.
from .session_log import redact

# ``classify_failure`` / ``diagnose`` / ``repair_env`` / ``repair_diagnosis`` are called through the
# ``envdoctor`` *module* (never ``from … import``) so a test — or a surface adapter — that
# monkeypatches ``envdoctor.classify_failure`` / ``envdoctor.repair_env`` is honored here too. This is
# the same dynamic-reference seam the provision / testing loops rely on. ``DiagLane`` is an enum used
# only in ``is`` comparisons, never patched, so it is bound directly.

if TYPE_CHECKING:
    from collections.abc import Callable

    from .envdoctor import Diagnosis, EnvIssue
    from .envtools import Conda
    from .prompts import PromptIO
    from .session_log import SessionLog
    from .specs import ToolSpec


# --------------------------------------------------------------------------- #
# Transcript + outcome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Obs:
    """One observed turn — the InstallerScientist's analogue of a ReAct message. ``stderr_tail`` is
    the tail of what the action produced, carried forward so the next LLM turn sees it."""

    turn: int
    lane: str  # "repair_det" | "repair_diag" | "handoff" | "retry"
    classify: str  # the EnvIssueKind / DiagKind acted on
    action: str  # "pip_install" | "system_lib" | "llm:pip_install+recreate" | "retry" | …
    repaired: bool
    stderr_tail: str = ""
    log_tail: str = ""  # salient build-log lines the monitor surfaced for this turn (audit trail)


@dataclass
class RemediationOutcome:
    """The result of a :meth:`InstallerScientist.remediate` run.

    ``repaired`` is the only signal a caller may use to flip a FAIL→(retry): it is True only when a
    real mutation happened AND a fresh re-observation confirmed the env is now healthy. ``needs_attention``
    carries a specific, actionable Lane-3 message (token / special input / data shape / …) when the
    failure is real but *not* an env-build problem, so the caller can tell the user exactly what to do
    instead of a generic FAIL. ``repairs`` are per-action records for ``state.tools[key]["repairs"]``.
    """

    repaired: bool = False
    needs_attention: str = ""
    reason: str = ""
    turns: int = 0
    transcript: tuple[_Obs, ...] = ()
    repairs: list[dict] = field(default_factory=list)

    @property
    def last_action(self) -> str:
        """The action string of the final turn — the fix that healed the env (empty if none ran).
        Callers use it to name the winning strategy, e.g. ``…+remediation:<last_action>``."""
        return self.transcript[-1].action if self.transcript else ""


@dataclass(frozen=True)
class _Decision:
    """The reason step's verdict for one turn."""

    lane: str  # "repair_det" | "repair_diag" | "handoff" | "retry" | "surface" | "unknown"
    classify: str
    issue: EnvIssue | None = None
    diag: Diagnosis | None = None
    message: str = ""


@dataclass(frozen=True)
class _ActOutcome:
    action: str
    repaired: bool
    stderr_tail: str
    record: dict


# --------------------------------------------------------------------------- #
# No-progress signature
# --------------------------------------------------------------------------- #
_SIG_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")
_SIG_WS_RE = re.compile(r"\s+")


def _signature(text: str) -> str:
    """A volatile-detail-stripped fingerprint of a failure, so ``3.6s | rc=1 | …`` and
    ``4.1s | rc=1 | …`` compare equal. Used to detect *no progress*: if a repair leaves the same
    signature, the action did not help and the loop stops rather than retrying the same fix."""
    t = _SIG_NUM_RE.sub("", (text or "").lower())
    return _SIG_WS_RE.sub(" ", t).strip()[:400]


def _safe_call(fn: Callable, default):
    """Call a caller-supplied closure, degrading a crash to ``default`` (the loop never raises)."""
    try:
        return fn()
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# Live thinking-box plumbing (display only — never affects control flow)
# --------------------------------------------------------------------------- #
# Human-readable lane names for the live thinking box.
_LANE_LABEL = {
    "repair_det": "deterministic repair",
    "repair_diag": "system-lib repair",
    "handoff": "planner",
    "retry": "retry",
    "surface": "needs attention",
    "unknown": "unrecognized",
}

# Classify kinds whose deterministic repair destroys + rebuilds the whole env (``remove_env`` +
# ``create*``), so a repeat past the ``MAX_ENV_RECREATES`` budget must be escalated (C2 pre-act gate):
#   • ``env_broken`` / ``env_absent`` / ``missing_r_package`` dispatch straight to
#     ``_repair_via_recreate`` (``action="recreate"``) and ``missing_py_module`` escalates to it when
#     pip cannot resolve the import (``action="pip_install→recreate"``);
#   • ``off_index_wheel`` → ``_repair_off_index_wheel`` and ``build_toolchain`` → ``_repair_toolchain``
#     (a94#2): BOTH ``assert_deletable_env`` + ``remove_env`` + recreate the env from the recipe, so
#     they are recreates in every sense — their action strings just name the *strategy* (``off_index`` /
#     ``toolchain``) rather than the fact that they rebuilt. They were previously (wrongly) documented as
#     "never recreate / always affordable" and so slipped BOTH this gate and the counter below → the
#     env could be nuked + rebuilt every turn, unbounded, silently violating ``MAX_ENV_RECREATES``.
# ``missing_pip_dist`` (one pip install, no removal) and the ``repair_diag`` system-lib lane are the only
# repairs that genuinely never recreate. Once the recreate budget is spent, a repair of one of THESE
# kinds is escalated to the planner / surfaced rather than nuking + rebuilding the env yet again.
_RECREATE_PRONE = frozenset(
    {"env_broken", "env_absent", "missing_r_package", "missing_py_module", "off_index_wheel", "build_toolchain"}
)

# Action strings whose repair destroyed + rebuilt the env but do NOT contain the token "recreate":
# ``_repair_off_index_wheel`` → ``off_index`` and ``_repair_toolchain`` → ``toolchain`` (a94#2). Matched
# by exact string (not substring) so they never collide with an unrelated future action name.
_DESTRUCTIVE_REBUILD_ACTIONS = frozenset({"off_index", "toolchain"})


def _action_recreates(action: str) -> bool:
    """True if an executed action performed a destructive full-env RECREATE — in ANY lane (N9 + a94#2).

    Deterministic recreates spell it ``recreate`` / ``pip_install→recreate``; a planner (handoff)
    action spells it ``llm:recreate`` / ``llm:recreate+pip_install``. ``recreate`` is unambiguous in
    the whole action-kind vocabulary (``pip_install`` / ``pip_install_git`` / ``conda_install`` /
    ``system_lib`` / ``wire`` / ``retry`` / ``none`` — none contains it), so a substring test catches
    every lane's spelling. The two rebuild-from-recipe repairs name their *strategy* instead
    (``off_index`` / ``toolchain``) yet still ``remove_env`` + recreate the env, so they are matched
    explicitly — else they nuke + rebuild the env every turn, uncounted, past ``MAX_ENV_RECREATES``.
    The C2 budget must count a planner recreate too: the det-lane gate escalates a would-be recreate to
    the planner once the budget is spent, so an uncounted recreate (any lane) would defeat the cap."""
    a = action or ""
    return "recreate" in a or a in _DESTRUCTIVE_REBUILD_ACTIONS


class _NullBox:
    """No-op thinking-box handle — the default when :meth:`InstallerScientist.remediate` runs without a
    live box (tests, or a non-TTY / not-yet-wired caller). It lets the loop call ``self._box.update(…)``
    unconditionally, and ``live=False`` routes :meth:`InstallerScientist._relay` back to plain ``io``
    output (byte-for-byte the pre-box behavior). The real live handle is ``progress._ThoughtHandle``,
    passed in by the driver's ``progress.think(…)`` frame; this module never imports ``progress`` —
    keeping the loop stdlib-at-import and the two agents' renderers decoupled."""

    live = False

    def update(self, **_kwargs) -> None:
        pass


_NULL_BOX = _NullBox()


class InstallerScientist:
    """Install-specialized ReAct remediation agent. Construct once per tool env, then drive it with
    the ``observe`` / ``recheck`` closures for that failure surface.

    ``llm_enabled`` + ``chat`` gate Lane 2: the *caller* decides the opt-in (the provision path checks
    ``SOG_PROVISION_LLM_REMEDIATION``, the Tier-1 path ``SOG_TEST_LLM_REMEDIATION``), so this module
    stays surface-agnostic. With the gate off or no handle, ambiguous failures surface honestly.
    """

    def __init__(
        self,
        conda: Conda,
        spec: ToolSpec,
        basic_env: str,
        *,
        io: PromptIO,
        chat=None,
        llm_enabled: bool = False,
        log: SessionLog | None = None,
    ) -> None:
        self.conda = conda
        self.spec = spec
        self.basic_env = basic_env
        self.target = spec.target_env(basic_env)
        self.io = io
        self.chat = chat
        self.llm_enabled = bool(llm_enabled)
        self.log = log
        self._recipe_cache: tuple[list[str], str] | None = None
        self._box = _NULL_BOX  # replaced per-run by remediate(box=…) when a live thinking box is open
        self._log_path = None  # the log this run reads (remediate(log_path=…)); None ⇒ <target>.build.log
        self._cur_salient = ""  # current turn's salient log tail; threaded into the opt-in planner (B3)

    # -- ReAct: reason ----------------------------------------------------- #
    def _reason(self, text: str) -> _Decision:
        """Deterministic classification FIRST (the conservative, negative-pinned classifier), then the
        advisory diagnose lanes. Mirrors ``react.call_model`` — it decides the next action, it does
        not take it.

        Fail-safe (C3): ``classify_failure`` / ``diagnose`` are pure and conservative, but a raising
        classifier (an unexpected input, a future regex bug) must NOT propagate out of the ``while True``
        and violate this module's documented ``never raises`` contract — it degrades to an ``"unknown"``
        decision so :meth:`remediate` still returns everything collected so far."""
        try:
            issue = envdoctor.classify_failure(text, spec=self.spec)
            if issue is not None:
                return _Decision("repair_det", str(issue.kind), issue=issue)
            diag = envdoctor.diagnose(text, spec=self.spec)
        except Exception as exc:
            return _Decision(
                "unknown",
                "unknown",
                message=f"classifier error ({type(exc).__name__}) — no deterministic env fix is known.",
            )
        if diag is None:
            return _Decision("unknown", "unknown", message="unrecognized failure — no deterministic env fix is known.")
        if diag.lane is DiagLane.REPAIR:
            return _Decision("repair_diag", str(diag.kind), diag=diag, message=diag.message)
        if diag.lane is DiagLane.SURFACE:
            return _Decision("surface", str(diag.kind), diag=diag, message=diag.message)
        if diag.lane is DiagLane.RETRY:
            return _Decision("retry", str(diag.kind), diag=diag, message=diag.message)
        return _Decision("handoff", str(diag.kind), diag=diag, message=diag.message)  # DiagLane.HANDOFF

    # -- ReAct: act -------------------------------------------------------- #
    def _act(self, decision: _Decision, text: str, planner_turns: list, recreate_used: int) -> _ActOutcome:
        """Execute the chosen lane through a guarded primitive. Mirrors ``react.tool_node`` — every
        path is dry-run-gated + namespace-guarded downstream; this method adds no new mutation.

        ``recreate_used`` is the destructive-recreate tally so far this run; it lets the planner lane
        refuse a would-be recreate once the ``MAX_ENV_RECREATES`` budget is spent (N9)."""
        lane = decision.lane
        if lane == "repair_det":
            rr = envdoctor.repair_env(self.conda, self.spec, self.basic_env, decision.issue, io=self.io, log=self.log)
            return _ActOutcome(rr.action, rr.repaired, "", rr.to_dict())
        if lane == "repair_diag":
            rr = envdoctor.repair_diagnosis(
                self.conda, self.spec, self.basic_env, decision.diag, io=self.io, log=self.log
            )
            return _ActOutcome(rr.action, rr.repaired, "", rr.to_dict())
        if lane == "retry":
            self._relay(
                f"  transient error ({decision.classify}) — retrying the same build",
                status=f"retrying — {decision.classify}",
                action="retry (same build)",
            )
            return _ActOutcome(
                "retry", False, "", {"kind": decision.classify, "action": "retry", "attempted": True, "repaired": False}
            )
        if lane == "handoff":
            return self._plan_and_execute(text, decision, planner_turns, recreate_used)
        # surface / unknown never reach _act (the loop handles them)
        return _ActOutcome("none", False, "", {"kind": decision.classify, "action": "none", "attempted": False})

    def _plan_and_execute(self, text: str, decision: _Decision, planner_turns: list, recreate_used: int) -> _ActOutcome:
        """Lane 2 — ask the fail-closed multi-turn planner for an env-only fix and run it through the
        SAME guarded ``_execute`` the provision path uses. ``repaired`` stays provisional (the fresh
        re-observation is authoritative)."""
        from . import remediation_planner as RP  # lazy: keeps the import boundary stdlib+pyyaml

        recipe_pips, py_ver = self._recipe()
        plan = RP.plan_env_fix_iter(
            self.chat,
            self.spec,
            self.target,
            text,
            recipe_pips,
            transcript=tuple(planner_turns),
            salient=self._cur_salient,
        )
        self._audit(
            "installer_scientist_plan",
            diagnosis=plan.diagnosis,
            actions=plan.action_kinds(),
            valid=plan.valid,
            reject_reason=plan.reject_reason,
        )
        if not plan.valid:
            self._relay(
                # "no reason given" (not "low confidence"): reject_reason is always set here, and
                # confidence never gates a plan — see F2-sh in remediation_planner.
                f"  planner proposed no usable env-only fix ({plan.reject_reason or 'no reason given'})",
                status="planner: no usable env-only fix",
            )
            return _ActOutcome(
                "llm:none",
                False,
                "",
                {
                    "kind": "llm",
                    "action": "llm:none",
                    "attempted": True,
                    "repaired": False,
                    "reject_reason": plan.reject_reason,
                },
            )
        # N9 — the destructive-recreate budget binds the planner lane too. Once MAX_ENV_RECREATES is
        # spent (a prior recreate, in either lane), refuse a plan that would recreate again rather than
        # nuke+rebuild the env a second time. Non-destructive planner fixes (pip/conda_install) are
        # unaffected; only a recreate-bearing plan is denied. Reported as "llm:none" (no "recreate"
        # token) so it is NOT re-tallied as a recreate and is treated as "no usable fix" downstream.
        if recreate_used >= constants.MAX_ENV_RECREATES and any(a.kind == "recreate" for a in plan.actions):
            self._relay(
                f"  planner wanted another full env recreate for {self.target}, but the recreate budget "
                f"is spent — refusing (won't rebuild the env again)",
                status="planner: recreate budget spent",
            )
            return _ActOutcome(
                "llm:none",
                False,
                "",
                {
                    "kind": "llm",
                    "action": "llm:none",
                    "attempted": True,
                    "repaired": False,
                    "reject_reason": "recreate budget spent",
                },
            )
        self._relay(
            f"  applying planner fix for {self.target}: {plan.diagnosis or ', '.join(plan.action_kinds())}",
            say=True,
            action=f"apply: {plan.diagnosis or ', '.join(plan.action_kinds())}",
        )
        rep = RP._execute(self.conda, self.spec, self.basic_env, self.target, plan, py_ver, io=self.io)
        action = "llm:" + "+".join(plan.action_kinds())
        off_index = any(a.find_links or a.extra_index_url for a in plan.actions)
        return _ActOutcome(
            action,
            False,
            rep.stderr_tail,
            {
                "kind": "off_index_wheel" if off_index else "llm",
                "action": action,
                "attempted": True,
                "repaired": False,  # provisional; confirmed by the next observe()/recheck()
                "steps": rep.ran,
                "diagnosis": plan.diagnosis,
            },
        )

    # -- ReAct: should_continue (affordability of the chosen lane) --------- #
    def _afford(self, decision: _Decision, det_used: int, llm_used: int, retry_used: int) -> _Decision | None:
        """Return the lane to actually run given the per-tool budgets, or ``None`` when nothing is
        affordable (→ surface). A deterministic lane whose budget is spent *escalates* to the planner
        when one is available — a genuinely useful second attempt, not a wrong-fix guess."""
        lane = decision.lane
        if lane in ("repair_det", "repair_diag"):
            if det_used < constants.MAX_ENV_REPAIRS:
                return decision
            if self._planner_available() and llm_used < constants.MAX_REACT_STEPS:
                return _Decision(
                    "handoff",
                    decision.classify,
                    diag=decision.diag,
                    message="deterministic budget spent; handing to the planner",
                )
            return None
        if lane in ("handoff", "unknown"):
            if self._planner_available() and llm_used < constants.MAX_REACT_STEPS:
                return _Decision("handoff", decision.classify, diag=decision.diag, message=decision.message)
            return None
        if lane == "retry":
            # C5: the transient-network retry lane has its OWN dedicated budget, decoupled from the
            # deterministic ``MAX_ENV_REPAIRS`` (tightening one no longer silently tightens the other).
            return decision if retry_used < constants.MAX_TRANSIENT_RETRIES else None
        return None

    # -- go() ------------------------------------------------------------- #
    def remediate(
        self,
        *,
        observe: Callable[[], str],
        recheck: Callable[[], bool] | None = None,
        budget: float | None = None,
        box=None,
        log_path=None,
    ) -> RemediationOutcome:
        """Drive the observe→reason→act→re-observe loop until the env is healthy, a Lane-3 surface
        stops it, a budget is spent, or a repair makes no progress.

        ``observe()`` returns the current failure text (``""`` once healthy); it is the fresh
        build/worker/probe log for each turn. ``recheck()`` is an optional cheap authoritative health
        confirmation consulted only when ``observe()`` comes back clean (defense against a false-empty
        observation). ``budget`` is the wall-clock second bound (default
        :data:`constants.SELF_REVIEW_BUDGET_SEC`). ``box`` is an optional live thinking-box handle
        (``progress._ThoughtHandle``): when omitted the loop's in-line messages print via ``io`` exactly
        as before and nothing is rendered — the box is purely additive. ``log_path`` optionally overrides
        which persisted log :func:`log_monitor.gather` reads on the seed turn: callers driving a
        test-phase repair pass their own tier sink (``<target>.tier1.log`` / ``<target>.tier2.log``) so
        the loop reasons on THIS run's real error instead of a stale provision-era ``<target>.build.log``
        (C6); when omitted it defaults to the provision build log exactly as before.

        The loop keeps both of its contracts: it **never raises** (``_reason`` / ``_act`` / every
        caller closure are fail-safe) and it **always terminates promptly** — a per-lane budget, an
        explicit :data:`constants.MAX_TURNS` ceiling, a pre-act remaining-budget floor, and a run-wide
        oscillation guard each bound it independently."""
        budget_sec = float(budget if budget is not None else constants.SELF_REVIEW_BUDGET_SEC)
        start = time.monotonic()
        self._box = box or _NULL_BOX
        self._log_path = log_path  # what _needs / _think point the user at
        transcript: list[_Obs] = []
        repairs: list[dict] = []
        planner_turns: list = []  # remediation_planner.RemediationTurn, built lazily to avoid a top import
        attempted_det: set[tuple[str, str]] = set()  # (kind, package) of deterministic repairs already tried
        seen_sigs: set[str] = set()  # run-wide oscillation guard (C4): every signature already acted on
        det_used = llm_used = retry_used = recreate_used = 0
        turn = 0

        text = _safe_call(observe, "") or ""
        self._audit("installer_scientist_start", target=self.target, seed_signature=bool(text))

        while True:
            # Healthy? An empty observation, confirmed by recheck() when present.
            if not text:
                if recheck is None or _safe_call(recheck, False):
                    return self._finish(True, "", "env healthy", turn, transcript, repairs)
                # observe() came back clean but the authoritative recheck() still says broken. There is no
                # error text to classify, so do NOT synthesize a failure string and ship it to the planner
                # as a phantom error (C3) — that would burn an llm turn on a non-error. Surface honestly.
                return self._finish(
                    False,
                    self._needs("the env still fails its health re-check after a clean build"),
                    "stopped: health re-check fails after a clean observation",
                    turn,
                    transcript,
                    repairs,
                )

            # Wall-clock is checked BETWEEN turns; the pre-act floor below adds the within-turn guard (C4).
            if time.monotonic() - start > budget_sec:
                return self._finish(
                    False,
                    self._needs("the wall-clock budget was exceeded before a fix"),
                    "stopped: wall-clock budget exceeded",
                    turn,
                    transcript,
                    repairs,
                )
            # Explicit turn ceiling (C4) — a cheap final backstop so ``while True`` can never outrun the
            # sum of the per-lane budgets, even under maxed-out env overrides.
            if turn >= constants.MAX_TURNS:
                return self._finish(
                    False,
                    self._needs("the self-heal loop reached its turn ceiling"),
                    "stopped: turn ceiling reached",
                    turn,
                    transcript,
                    repairs,
                )

            turn += 1
            # Let the ReAct loop *read the log*, not just the caller's short observe() string. On the
            # FIRST turn the persisted ``<target>.build.log`` is still fresh, so a low-signal observe() is
            # enriched with it (recovering the original multi-strategy build error); later turns trust the
            # authoritative re-observe (the persisted log may now be stale). The no-progress signature is
            # ALWAYS taken from the raw observe text (guaranteed by ``log_monitor.gather``), so the guard
            # below stays byte-for-byte the pre-monitor behavior — the extra log evidence only sharpens
            # *reasoning* and *display*, never which turn counts as "no progress".
            ev = log_monitor.gather(
                text, target=self.target, spec=self.spec, use_build_log=(turn == 1), log_path=log_path
            )
            cur_sig = ev.signature
            self._cur_salient = ev.salient  # fed to the opt-in planner prompt this turn (B3)
            decision = self._reason(ev.classify_text)
            self._think(turn, decision, ev)

            # Lane 3 — honest surface: no mutation, a specific message, stop the loop.
            if decision.lane == "surface":
                self._box.update(mode="surface", status=f"turn {turn} · needs attention", thought=decision.message)
                return self._finish(False, decision.message, f"surface: {decision.classify}", turn, transcript, repairs)

            # Run-wide oscillation guard (C4): the two no-progress guards below are consecutive-only
            # (signature unchanged after a repair) and deterministic-only ((kind,package)). A planner
            # flip-flopping between two DISTINCT signatures "makes progress" every turn and would burn all
            # of MAX_REACT_STEPS. So once a signature we ALREADY acted on this run recurs, we are going in
            # circles — stop honestly. The retry lane is exempt: it re-runs the SAME build by design and is
            # bounded by its own MAX_TRANSIENT_RETRIES budget instead.
            if decision.lane != "retry" and cur_sig and cur_sig in seen_sigs:
                return self._finish(
                    False,
                    self._needs(
                        "the self-heal loop is oscillating between the same failures", classify=decision.classify
                    ),
                    "stopped: oscillation (a prior failure signature recurred)",
                    turn,
                    transcript,
                    repairs,
                )
            if cur_sig:
                seen_sigs.add(cur_sig)

            # (kind, package) no-progress guard for the DETERMINISTIC lanes (mirrors the provision /
            # tier loops' ``attempted`` set): if a repair we already ran leaves the classifier naming
            # the *same* issue, that repair is not working — escalate to the planner if one is wired,
            # else stop honestly. (Distinct packages — a cascade like six→tqdm — are not blocked.)
            if decision.lane in ("repair_det", "repair_diag"):
                dsig = self._det_sig(decision)
                if dsig in attempted_det:
                    if self._planner_available() and llm_used < constants.MAX_REACT_STEPS:
                        decision = _Decision(
                            "handoff",
                            decision.classify,
                            diag=decision.diag,
                            message="deterministic repair made no progress; handing to the planner",
                        )
                    else:
                        return self._finish(
                            False,
                            self._needs(
                                "a deterministic repair made no progress on the same issue", classify=decision.classify
                            ),
                            "stopped: no progress (same issue after repair)",
                            turn,
                            transcript,
                            repairs,
                        )
                else:
                    attempted_det.add(dsig)

            # Cap destructive full-env RECREATEs at MAX_ENV_RECREATES per run (C2). The (kind,package) memo
            # above does NOT catch a cascade across DIFFERENT recreate-prone classifications (env_broken →
            # missing_py_module → env_absent), each of which nukes+rebuilds internally. Once the recreate
            # budget is spent, escalate a would-be recreate to the planner (a non-destructive second
            # opinion) or surface it, instead of nuking + rebuilding the env yet again.
            if (
                decision.lane == "repair_det"
                and decision.classify in _RECREATE_PRONE
                and recreate_used >= constants.MAX_ENV_RECREATES
            ):
                if self._planner_available() and llm_used < constants.MAX_REACT_STEPS:
                    decision = _Decision(
                        "handoff",
                        decision.classify,
                        diag=decision.diag,
                        message="recreate budget spent; handing to the planner instead of rebuilding again",
                    )
                else:
                    return self._finish(
                        False,
                        self._needs(
                            "the env needs another full rebuild but the recreate budget is spent",
                            classify=decision.classify,
                        ),
                        "stopped: recreate budget spent",
                        turn,
                        transcript,
                        repairs,
                    )

            eff = self._afford(decision, det_used, llm_used, retry_used)
            if eff is None:
                # Nothing affordable. Either the read was a real requirement (unknown/handoff with no
                # planner) or a per-lane budget is spent — either way hand back an actionable
                # needs_attention (C6): the message / dep plus the persisted build.log path.
                if decision.lane in ("handoff", "unknown"):
                    needs = self._needs(decision.message, classify=decision.classify)
                    reason = "stopped: no automatic repair available"
                else:
                    needs = self._needs("the automatic repair budget is exhausted", classify=decision.classify)
                    reason = "stopped: repair budget exhausted"
                return self._finish(False, needs, reason, turn, transcript, repairs)

            # Pre-act remaining-budget floor (C4): the wall-clock is otherwise checked only BETWEEN turns,
            # but one _act can launch a create+pip that runs for a full CONDA_CREATE / PIP timeout. If less
            # than one turn's worth of budget remains, stop now rather than launch a doomed long op.
            if budget_sec - (time.monotonic() - start) < constants.SELF_HEAL_MIN_TURN_SEC:
                return self._finish(
                    False,
                    self._needs(
                        "too little wall-clock budget remained to attempt another repair", classify=eff.classify
                    ),
                    "stopped: insufficient budget for another turn",
                    turn,
                    transcript,
                    repairs,
                )

            self._acting(turn, eff)
            act = self._act(eff, text, planner_turns, recreate_used)
            self._box.update(action=act.action)  # refine the acting placeholder to the real action string
            transcript.append(
                _Obs(turn, eff.lane, eff.classify, act.action, act.repaired, act.stderr_tail, log_tail=ev.salient)
            )
            repairs.append(act.record)
            if eff.lane in ("repair_det", "repair_diag"):
                det_used += 1
            elif eff.lane == "handoff":
                llm_used += 1
            elif eff.lane == "retry":
                # Back off before the next observe() re-runs the (possibly hour-long) build (C5), clamped
                # to the remaining wall-clock so a retry delay can never itself blow the budget. The
                # exponent is the count of PRIOR retries (first retry → base·2**0).
                self._backoff_sleep(retry_used, budget_sec - (time.monotonic() - start))
                retry_used += 1
            # Count a destructive full-env RECREATE in ANY lane (C2 + N9 + a94#2), not just the det
            # lanes. A planner (handoff) recreate — and the off_index / toolchain rebuild-from-recipe
            # repairs — previously slipped past this tally and the pre-act gate, so remediation could
            # nuke+rebuild the env every turn up to MAX_REACT_STEPS. ``attempted`` guards the off_index /
            # toolchain no-op early-returns (missing recipe / no pip section / underivable plan): those
            # keep the action string but removed nothing, so they must not spend the recreate budget.
            if _action_recreates(act.action) and act.record.get("attempted", True):
                recreate_used += 1
            self._audit(
                "installer_scientist_turn",
                turn=turn,
                lane=eff.lane,
                classify=eff.classify,
                action=act.action,
                repaired=act.repaired,
                log_source=ev.source,
                # C7: the evidence the loop reasoned on. redact-THEN-clip (a94#3 / Round-21): session_log
                # redacts JSONL fields, but only a FULLY-present secret — a token straddling char 800 would be
                # cut first, so its surviving prefix would slip past that backstop into the durable transcript.
                salient=redact(ev.salient)[:800],
            )

            # A DRY-RUN deterministic repair mutated nothing — the env is byte-for-byte what it was, so
            # re-observing the authoritative check is wasted work that cannot surface a new signal. Stop
            # honestly (mirrors the pre-agent loops' skip of the confirming re-run on a dry-run). A repair
            # that genuinely ran but *failed* (``repaired=False``, not dry-run) is NOT short-circuited: it
            # may have partially changed the env, so the signal-changed guard below is the real arbiter —
            # and the LLM lane's provisional ``repaired=False`` (its commands DID run) must re-observe too.
            if eff.lane in ("repair_det", "repair_diag") and act.record.get("dry_run"):
                return self._finish(
                    False,
                    self._needs("the repair ran in --dry-run, so the env was left unchanged", classify=eff.classify),
                    "stopped: dry-run repair (env unchanged)",
                    turn,
                    transcript,
                    repairs,
                )

            # Re-observe: authoritative confirmation of what the action did.
            new_text = _safe_call(observe, "") or ""
            if not new_text and (recheck is None or _safe_call(recheck, False)):
                return self._finish(True, "", "env healthy after repair", turn, transcript, repairs)

            new_sig = _signature(new_text)
            # No progress: a *mutating* lane that left the error unchanged will not fix it on a retry.
            # (A transient-retry lane is expected to see the same error, so it is exempt — it is bounded
            # by its own retry budget instead.)
            if eff.lane != "retry" and new_sig == cur_sig:
                return self._finish(
                    False,
                    self._needs("the error was unchanged after the repair", classify=eff.classify),
                    "stopped: no progress (error unchanged after repair)",
                    turn,
                    transcript,
                    repairs,
                )

            # Feed the fresh error (and what the last action produced) into the next planner turn. Prefer
            # the action's own stderr, then the monitor's curated salient lines, then a raw tail — so the
            # planner's cross-turn "do NOT repeat a failed action" memory always has real error context.
            planner_turns.append(
                self._planner_turn(
                    diagnosis=eff.classify,
                    action=act.action,
                    # redact-THEN-clip the raw-tail fallback (a94#3 / Round-21): this value is stored on the
                    # RemediationTurn and egresses to a third-party LLM via _render_transcript; a secret
                    # straddling the -600 cut would survive that egress redact as an unmatchable fragment.
                    # (act.stderr_tail is already redacted by _ExecReport._absorb; _salient by its own redact.)
                    stderr_tail=act.stderr_tail or self._salient(new_text) or redact(new_text)[-600:],
                    outcome="changed error" if new_sig != cur_sig else "still failing",
                )
            )
            text = new_text

    # -- helpers ----------------------------------------------------------- #
    def _planner_available(self) -> bool:
        return bool(self.llm_enabled and self.chat is not None)

    def _needs(self, what: str, *, classify: str = "") -> str:
        """Compose an actionable needs-attention line for a non-healthy finish (C6): the reason, the
        classify/dep when known, and the persisted ``<target>.build.log`` path the user can inspect —
        so a give-up is never the bare ``failed to build`` line with the dep dropped. ``what`` is
        embedded verbatim (a Lane-3 diagnose message keeps its own wording). Self-contained + never
        raises (the path is pure construction, but guarded anyway so ``_finish`` can always run)."""
        base = (what or "").strip() or "the environment could not be repaired automatically"
        tag = f" [{classify}]" if classify and classify not in ("unknown", "") else ""
        try:
            tail = f"; see {self._watched_log()}"
        except Exception:
            tail = ""
        return f"{base}{tag}{tail}"

    def _watched_log(self):
        """The log this run actually reads: the caller's ``log_path`` (a test-phase self-heal passes its
        ``<target>.tier1.log`` / ``.tier2.log``) or the provision ``<target>.build.log``. Both messages
        named the build log unconditionally, which on a test-phase heal is a provision-era file that may
        be stale or absent (hunt 2026-09-30, u37-setup-checks-20)."""
        if self._log_path:
            return Path(self._log_path)
        return log_monitor.build_log_path(self.target)

    def _backoff_sleep(self, prior_retries: int, remaining: float) -> float:
        """Sleep before the next observe() re-runs a transient-failure build (C5). Exponential backoff
        (``base·2**prior_retries``) clamped BOTH to a ceiling and to the remaining wall-clock, so a
        retry delay can never itself blow the budget. Jitter is a sub-second fraction of ``monotonic()``
        (no ``random`` — keeps the module import-light and the schedule reproducible-enough). Returns
        the delay actually slept (audit/telemetry only); never raises."""
        base = constants.TRANSIENT_RETRY_BACKOFF_SEC
        ceiling = constants.TRANSIENT_RETRY_BACKOFF_MAX_SEC
        jitter = time.monotonic() % 1.0
        want = min(base * (2 ** max(0, prior_retries)) + jitter, ceiling)
        delay = max(0.0, min(want, remaining))
        if delay > 0:
            time.sleep(delay)
        return delay

    # -- live thinking box + planner memory -------------------------------- #
    def _think(self, turn: int, decision: _Decision, ev) -> None:
        """Update the live box to the *reasoning* view: the salient log lines the monitor is reading,
        plus the lane the classifier picked. A no-op on the null box, so it is called unconditionally
        (it never prints — the animator is the sole writer while a frame is open)."""
        label = _LANE_LABEL.get(decision.lane, decision.lane)
        self._box.update(
            mode="thinking",
            status=f"turn {turn} · {label}",
            thought=ev.salient or ev.classify_text or decision.message,
            action="",
            footer=f"watching {self._watched_log().name}",
        )

    def _acting(self, turn: int, eff: _Decision) -> None:
        """Update the live box to the *acting/planning* view just before the (slow) guarded repair runs,
        so the box reflects what is happening during the pip/conda work; the action row is refined to the
        real action string once :meth:`_act` returns."""
        label = _LANE_LABEL.get(eff.lane, eff.lane)
        self._box.update(
            mode="planning" if eff.lane == "handoff" else "acting",
            status=f"turn {turn} · {label}",
            action=f"{label}…",
        )

    def _relay(self, message: str, *, say: bool = False, **box_fields) -> None:
        """Route an in-loop agent message to the live thinking box (the animator is the sole writer while
        a frame is open — R2) or, when no live box is attached, print it via ``io`` exactly as before."""
        if self._box.live:
            self._box.update(**box_fields)
        else:
            (self.io.say if say else self.io.note)(message)

    def _salient(self, text: str) -> str:
        """Joined salient log lines from ``text`` for the planner's cross-turn memory (``""`` if none)."""
        is_r = getattr(self.spec, "worker_kind", "") == "rscript"
        # redact-THEN-clip (a94#3 / Round-21): salient lines are pulled from RAW build stderr and can carry a
        # credentialed --extra-index-url / git+https token; this string is stored for the planner's cross-turn
        # memory and egresses to a third-party LLM, so mask any secret WHOLE before the -800 tail cut (a clip
        # first would leave an unmatchable fragment that the egress redact can no longer catch).
        return redact("\n".join(log_monitor.salient_lines(text, is_r=is_r)))[-800:]

    def _recipe(self) -> tuple[list[str], str]:
        if self._recipe_cache is None:
            try:
                from . import provision

                self._recipe_cache = provision.read_recipe_pip_and_python(getattr(self.spec, "recipe", "") or "")
            except Exception:
                self._recipe_cache = ([], "3.11")
        return self._recipe_cache

    @staticmethod
    def _det_sig(decision: _Decision) -> tuple[str, str]:
        """The ``(kind, package)`` fingerprint of a deterministic decision — the key of the
        already-attempted set. Package comes from the classifier issue or the diagnosis."""
        pkg = ""
        if decision.issue is not None:
            pkg = decision.issue.package
        elif decision.diag is not None:
            pkg = decision.diag.package
        return (decision.classify, pkg)

    @staticmethod
    def _planner_turn(*, diagnosis: str, action: str, stderr_tail: str, outcome: str):
        from . import remediation_planner as RP

        return RP.RemediationTurn(diagnosis=diagnosis, actions=(action,), stderr_tail=stderr_tail, outcome=outcome)

    def _audit(self, kind: str, **fields) -> None:
        """Emit a per-turn audit event — lane / action-kind / repaired / counts only, never secrets.

        Best-effort: ``log.event`` writes+flushes to disk, so it can raise ``OSError`` on a full or
        vanished mount. Auditing is telemetry — a logging failure must NEVER abort an in-progress
        self-heal (this fires every ReAct turn), so a raising sink is swallowed (F1-sh)."""
        if self.log is not None:
            try:
                self.log.event(kind, server=self.spec.server_key, **fields)
            except Exception:
                pass

    def _finish(
        self,
        repaired: bool,
        needs_attention: str,
        reason: str,
        turns: int,
        transcript: list[_Obs],
        repairs: list[dict],
    ) -> RemediationOutcome:
        self._audit(
            "installer_scientist_finish",
            target=self.target,
            repaired=repaired,
            turns=turns,
            needs_attention=bool(needs_attention),
            reason=reason,
        )
        return RemediationOutcome(
            repaired=repaired,
            needs_attention=needs_attention,
            reason=reason,
            turns=turns,
            transcript=tuple(transcript),
            repairs=repairs,
        )
