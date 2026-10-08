"""
The setup driver — a resumable state machine over the seven phases.

One :class:`Wizard` run walks ``state.PHASES`` in order, persisting
:class:`~sog_install.state.SetupState` atomically after every
transition so a crash or Ctrl-C can be resumed cleanly. It owns nothing that the
engines don't already do — it only *sequences* them and records status:

* **Stage A** (``preflight`` + ``onboarding``) always runs on every launch. It is
  deterministic, fast, and read-only except for the ``.env`` write; re-running it
  on a resume simply re-checks the machine and re-establishes the LLM config
  (``onboarding`` detects an existing working ``.env`` and reuses it), which is
  exactly what rebuilds the in-memory :class:`DecisionSource` for Stage B.
* **Stage B** (``category_select`` → ``finalize``) honors the resume cursor: a
  phase already ``done``/``skipped`` in the loaded state is not re-run — its
  result is read back from state (``selected_servers``, ``basic_env_name``, the
  per-tool records) so the next phase has what it needs.

The single Stage-B seam is the :class:`DecisionSource`: ``--answers`` builds a
:class:`ScriptedSource` (no LLM, reproducible), otherwise the verified key from
onboarding seeds a :class:`ChatClient` behind an :class:`InteractiveGuide`. Every
side-effecting engine call is gated by ``source.confirm`` (or forced true by
``--yes``); ``--dry-run`` skips execution entirely and just prints the plan.

Non-interference: the driver only ever calls engines that write the wizard's own
artifacts (``.sog_setup/``, ``test/installation/``, ``install/recipes/mcp_config.setup.yaml``)
or create/repair conda envs inside the ``<basic>_*`` namespace. The one agent file it
rewrites is ``agent/MCP_server/mcp_config.yaml``: ``finalize`` regenerates it for this box
(after a timestamped backup) unless ``--keep-agent-config`` is passed. It never edits
the server files themselves.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import os
import signal
import sys
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import base_env, constants, demo, doctor, llm_setup, mcp_resolver, onboarding, provision, testing, wiring
from .answers import ScriptedSource, load_answers
from .categories import categories_as_ctx, grouped_tool_options, load_categories, recommended_servers
from .decisions import (
    BaseEnvDecision,
    BaseEnvMode,
    CategorySelection,
    GuideContext,
    Proposal,
    ToolPlan,
    expand_category_servers,
)
from .envtools import Conda, CondaError
from .guide import InteractiveGuide
from .llm_chat import ChatClient
from .preflight import has_hard_failure, run_preflight, summarize
from .progress import StepBox
from .prompts import PromptIO
from .session_log import DryRunSessionLog, SessionLog, redact, register_index_url_secrets
from .specs import load_all_specs
from .state import PHASES, SetupState, Status, archive_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from .categories import Category
    from .decisions import DecisionSource, LLMChoice
    from .specs import ToolSpec

_TRUTHY = {"1", "true", "yes", "y", "on"}


def _stream_build_logs_enabled() -> bool:
    """Opt-in live build-log streaming, read from ``SOG_STREAM_BUILD_LOGS`` (default off).

    When set, the three long build calls (``create_from_yaml`` / ``pip_install`` / ``clone``) tee
    their output live to a ``tail -f``-able per-tool ``<target>.build.log`` instead of capturing
    silently — so a minutes-long solve shows progress. The returned RunResult is byte-identical, so
    this changes only visibility, never behavior; unset ⇒ every build keeps the plain captured path."""
    return os.environ.get("SOG_STREAM_BUILD_LOGS", "").strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# Options
# --------------------------------------------------------------------------- #
@dataclass
class WizardOptions:
    answers_path: str | None = None  # --answers file.yaml  → ScriptedSource (no LLM)
    resume: bool = False  # --resume  : continue without asking
    restart: bool = False  # --restart : archive prior state, start clean
    dry_run: bool = False  # --dry-run : print the plan, zero side effects
    assume_yes: bool = False  # --yes     : auto-confirm every proposal
    only: list[str] = field(default_factory=list)  # --only a,b : restrict to these servers
    check_net: bool = True  # --no-net  : skip the network preflight probe
    keep_agent_config: bool = False  # --keep-agent-config : don't rewrite MCP_server/mcp_config.yaml
    skip_install: bool = False  # --skip-install : skip tool-env building, go straight to the chat


# The write-free dry-run log now lives in session_log (DryRunSessionLog) so the wizard (N4) and
# `reset --dry-run` (R20) share one implementation of the "preview writes nothing" contract. Kept as a
# private alias so every existing wizard reference / test name for `_DryRunLog` still resolves.
_DryRunLog = DryRunSessionLog


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
class Wizard:
    """Sequences the seven phases against one :class:`SetupState`."""

    def __init__(self, opts: WizardOptions, *, io: PromptIO | None = None) -> None:
        self.opts = opts
        self.io = io or PromptIO(non_interactive=opts.answers_path is not None)
        self.log: SessionLog | None = None  # opened once the run id is known
        self._conda: Conda | None = None  # built lazily, AFTER preflight clears conda
        self.answers: dict | None = None
        self.source: DecisionSource | None = None
        self.choice: LLMChoice | None = None
        self.state = SetupState()
        self.ctx = GuideContext()
        self._specs: dict[str, ToolSpec] | None = None
        self._cats: list[Category] | None = None
        # Set once at the top of a resumed run (B2): True when a prior run left a selected
        # tool unbuilt, so provision+test must re-run instead of being skipped as "done".
        self._retry_unbuilt = False

    @property
    def conda(self) -> Conda:
        """The conda wrapper, built on first use so :meth:`_phase_preflight` can
        surface a missing-conda problem with a friendly message before we ever try
        to construct one. ``--dry-run`` makes every mutating conda op a no-op."""
        if self._conda is None:
            self._conda = Conda(
                dry_run=self.opts.dry_run,
                log=self.log,
                progress=self._install_progress(),
                stream_build_logs=_stream_build_logs_enabled(),
            )
        return self._conda

    def _install_progress(self) -> StepBox:
        """A live checklist box (and single-line spinner fallback) for long env builds /
        pip installs during interactive runs.

        Explicitly disabled for ``--dry-run`` (nothing is built) and scripted / non-interactive
        runs (deterministic, often piped output); otherwise ``enabled=None`` lets
        :class:`StepBox` auto-detect a real TTY so CI and tests stay silent."""
        muted = self.opts.dry_run or self.opts.answers_path is not None or self.io.non_interactive
        return StepBox(self.io.out, enabled=False if muted else None)

    # -- lazily-loaded, cached heavy data ------------------------------------
    @property
    def specs(self) -> dict[str, ToolSpec]:
        if self._specs is None:
            self._specs = load_all_specs()
        return self._specs

    @property
    def cats(self) -> list[Category]:
        if self._cats is None:
            self._cats = load_categories()
            self.ctx.categories = categories_as_ctx(self._cats)
            # Every installable spec, grouped by category, for the interactive picker.
            # Empty when specs/skills are unavailable (bare env) -> guide degrades to
            # the category-level menu.
            self.ctx.tool_groups = grouped_tool_options(self._cats, set(self.specs))
        return self._cats

    # -- confirm gate --------------------------------------------------------
    def _confirm(self) -> Callable[[Proposal], bool]:
        """The propose→confirm→execute gate, forced open by ``--yes``."""
        if self.opts.assume_yes:
            return lambda _p: True
        assert self.source is not None
        return lambda p: self.source.confirm(p, self.ctx)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _seed_platform_home(self) -> None:
        """Seed/refresh the pip-only instance root (see ``home_seed``). Never raises."""
        try:
            from . import home_seed

            report = home_seed.ensure_seeded()
        except Exception as exc:
            self.io.warn(f"platform home seeding skipped: {exc}")
            return
        status = report.get("status")
        if status in ("seeded", "refreshed"):
            self.io.note(f"{status} platform home {report.get('root')} ({report.get('written', 0)} files)")
        elif status == "external":
            self.io.note(f"using externally-managed platform root {report.get('root')} (left untouched)")
        elif status == "bad-source":
            self.io.warn(
                f"packaged platform payload at {report.get('source')} has an unreadable manifest"
                " -- nothing seeded; reinstall the package"
            )

    def run(self) -> int:
        """Run the wizard; the stop-signal handlers it installs are put back when it returns.

        They used to stay installed after ``run`` returned, so a caller that ran the wizard in its own
        process -- a test, ``sog-setup chat`` -- had every later Ctrl-C turned into this wizard's
        "interrupted" exit 130 (found by the 2026-10-01 gate: a terminal-session test's Ctrl-C).
        """
        try:
            return self._run()
        finally:
            self._restore_signal_handlers()

    def _run(self) -> int:
        # Off a checkout, make the writable instance root real BEFORE anything reads or
        # writes through constants.repo_root() (state, specs, canonical config). WARN-only:
        # a seeding hiccup must not block a wizard that can still do useful work, and on a
        # checkout ensure_seeded() is a guaranteed no-op. Doctor/conncheck stay read-only
        # on purpose -- the wizard is the single seeding trigger.
        self._seed_platform_home()
        if self.opts.answers_path:
            # --skip-install relaxes the file's tool/base_env requirements as `skip_install: true` in the
            # file does (hunt 2026-09-30, u35-setup-ux-15).
            self.answers = load_answers(self.opts.answers_path, skip_install=self.opts.skip_install)
        self._load_or_start_state()
        self._open_log()
        self._install_signal_handlers()

        # Stage A always runs — it (re)establishes the machine check + LLM/source.
        if not self._phase_preflight():
            return 2
        self._phase_onboarding()
        # Skip-install gate — asked once, right after the LLM key is ensured in BOTH paths.
        if self._resolve_skip_install():
            return self._run_app_only()
        self._capture_stated_goals()

        # Decide ONCE — before the loop can mutate phase state — whether a prior run left
        # any *selected* tool unbuilt (build FAILED / no record). If so, re-open provision
        # (and test) so the retry actually happens instead of the resume cursor skipping
        # them as "done"; a re-run rebuilds only the failures (provision_one skips healthy
        # envs). On a fresh run nothing is selected yet, so this is False. (B2)
        self._retry_unbuilt = self._has_unbuilt_selected()

        # Stage B honors the resume cursor.
        stage_b: list[tuple[str, Callable[[], bool]]] = [
            ("category_select", self._phase_category_select),
            ("base_env", self._phase_base_env),
            ("provision", self._phase_provision),
            ("test", self._phase_test),
            ("finalize", self._phase_finalize),
        ]
        for name, fn in stage_b:
            done = self.state.phase_status(name) in (Status.DONE.value, Status.SKIPPED.value)
            if done and self._stale_completion(name):
                # A prior run marked this phase done but its result isn't usable, so re-run it
                # despite the resume cursor. _stale_completion trips for two DIFFERENT reasons, and
                # the message must match the actual one (R24-A2): category_select => the persisted
                # selection had 0 tools; provision/test/finalize => a *selected* tool's build FAILED
                # (B2) and the retry rebuilds only the failures. The old code always said "0 tools",
                # which is flatly wrong (and confusing) on the far more common build-retry path.
                if name == "category_select":
                    self.io.warn("a prior run completed tool selection with 0 tools — re-running that step.")
                else:
                    self.io.warn(f"a prior run left a selected tool unbuilt — re-running '{name}' to retry the build.")
                self.state.set_phase(name, Status.PENDING.value)
                self._persist()
                done = False
            if done:
                self.io.note(f"phase '{name}' already complete — skipping (resume)")
                self._rehydrate_after_skip(name)
                continue
            if not fn():
                self.io.err(f"stopped at phase '{name}'. Re-run `sog-setup` to resume from here.")
                return 1
            if name == "category_select":
                # category_select just (re)populated the selection. On a legacy 0-tool resume it was
                # FORCED to re-run above (stale-completion), but the pre-loop _retry_unbuilt snapshot
                # was taken when the selection was still empty → False. Refresh now so the freshly-picked
                # tools re-open provision/test/finalize instead of being skipped as "done" by the resume
                # cursor. Monotonic OR: never clears a retry already flagged for other reasons. (S4)
                self._retry_unbuilt = self._retry_unbuilt or self._has_unbuilt_selected()
        self._finish()
        self._phase_demo()
        # Hand off to the terminal chat (print the command, offer to start it). Side-effect only — its
        # return is ignored, AND any exception is swallowed, so a declined/headless/Ctrl-C'd launch
        # can never flip the honest install verdict below.
        try:
            self._offer_chat()
        except Exception as exc:  # a soft handoff must never sink a completed install
            if self.log is not None:
                self.log.event(
                    "chat_handoff_error",
                    error=redact(f"{type(exc).__name__}: {exc}")[:300],
                    traceback=redact(traceback.format_exc())[:2000],
                )
            self.io.warn(f"chat handoff skipped (non-fatal): {type(exc).__name__}")
        # B1 — the process exit code must reflect the honest verdict. Every stage-B phase returns
        # True even when tools fail to build (one bad tool never sinks the run), so control reaches
        # here on a degraded/incomplete run too; a hardcoded 0 hid that from CI. A dry-run changes
        # nothing, so it always succeeds.
        return 0 if self.opts.dry_run else _verdict_exit_code(self._final_verdict())

    # ------------------------------------------------------------------ #
    # Skip-install → chat
    # ------------------------------------------------------------------ #
    def _resolve_skip_install(self) -> bool:
        """Decide whether to skip tool installation and go straight to the chat.

        Precedence (non-interactive-safe): explicit ``--skip-install`` → scripted
        ``skip_install:`` answer → never-skip when non-interactive → interactive prompt. On an
        unfinished prior run the prompt becomes a 3-way resume-vs-skip choice."""
        if self.opts.skip_install:
            return True
        if self.answers is not None:
            return bool(self.answers.get("skip_install", False))
        if self.io.non_interactive:
            return False
        # A genuine *unfinished install* means a prior run reached Stage B (tool selection/build).
        # Preflight + onboarding (Stage A) re-run every launch and mark themselves done BEFORE this
        # gate, so `state.has_progress()`/`resume_cursor()` are ALWAYS truthy here — keying on them
        # would tell every fresh first-run "a previous install is unfinished". Check Stage-B phases,
        # which Stage A never touches.
        stage_b = ("category_select", "base_env", "provision", "test", "finalize")
        prior_install = any(self.state.phase_status(p) != Status.PENDING.value for p in stage_b)
        if prior_install and not self.state.is_complete():
            choice = self.io.select(
                "A previous install is unfinished. Resume it, or skip straight to the chat?",
                [
                    ("resume", "Resume the install", "finish building your tools"),
                    ("app", "Skip to the chat", "start ST-Coscientist now"),
                ],
                default="resume",
            )
            return choice == "app"
        return self.io.ask_yesno(
            "Skip installation and go straight to the ST-Coscientist chat?",
            default=False,
        )

    def _run_app_only(self) -> int:
        """The skip path: make sure an agent env exists, then hand off to the chat.

        Skips category_select…finalize + demo (they stay PENDING, so a later `sog-setup` resumes a
        full install — skip is non-destructive). On a fresh machine with no agent-capable env, builds
        the small agent-core base env (the one install skip cannot avoid); otherwise reuses the
        env :func:`launch.resolve_base_env` finds."""
        from . import launch

        self.io.section("Skipping tool installation")
        self.state.meta["skip_install"] = True
        self._persist()

        basic = launch.resolve_base_env(self.conda, self.state)
        if basic is None:
            self.io.note("no agent env yet — building a small agent-core env to host the app (skipping the tool envs).")
            if not self._phase_base_env():
                self.io.err("could not establish a base env — cannot launch. Re-run `sog-setup` for the full setup.")
                return 1
        else:
            # An agent-capable env already exists; adopt it in-memory for the verdict line + handoff.
            self.state.basic_env_name = basic

        self._finish_app_only()
        # The handoff is dry-run-safe on its own: launch.offer_chat treats a missing/not-yet-built
        # env under dry-run as a clean 0 (not the real "no env" exit 2), so a fresh-clone dry-run no
        # longer contradicts itself — while a dry-run with a real env still records the offer.
        return self._offer_chat()

    def _finish_app_only(self) -> None:
        """A lighter 'All done' for the skip path (no install verdict to report)."""
        self.io.section("All done")
        base = self.state.basic_env_name or "(unset)"
        if self.opts.dry_run:
            self.io.ok(f"[dry-run] would skip tool installation and start the chat for base env {base}")
            if self.log is not None:
                self.log.event("run_complete", run_id=self.state.run_id, dry_run=True, skip_install=True)
            return
        self.io.ok(f"skipped tool installation for base env {base} — connecting you to the chat")
        self.io.note(
            "analysis tools run best-effort under the base env until a full `sog-setup` pins each "
            "tool to its own env; `sog-setup doctor` shows how many are runnable."
        )
        self.io.note("install the analysis tools anytime: re-run `sog-setup` and choose the full run.")
        if self.log is not None:
            self.log.event("run_complete", run_id=self.state.run_id, dry_run=False, skip_install=True)

    def _offer_chat(self) -> int:
        """Shared handoff to the terminal chat (both the skip path and the end of a full run)."""
        from . import launch

        basic = launch.resolve_base_env(self.conda, self.state)
        # The wizard's io.non_interactive is answers-based (True only under --answers), so a bare
        # piped `sog-setup [--yes]` run would otherwise reach the launch decision as "interactive"
        # and foreground-launch (HANG) the chat in CI. Gate on the TTY too — mirrors
        # launch.main()'s `PromptIO(non_interactive=not sys.stdin.isatty())`.
        return launch.offer_chat(
            self.conda,
            basic,
            io=self.io,
            dry_run=self.opts.dry_run,
            assume_yes=self.opts.assume_yes,
            non_interactive=self.io.non_interactive or not sys.stdin.isatty(),
        )

    def _final_verdict(self) -> dict:
        """Aggregate what the run *actually* accomplished into an honest status.

        Reads only persisted state, so it is correct on a fresh run and on resume
        (where phases were skipped and their results read back from state). The rule:
        a green ✅ is claimed ONLY when the base env is present, at least one tool env
        was provisioned, none of the provisioned tools failed their mini-data
        (Tier-1) test, and no real-data agent run (Tier-2) that executed failed.
        Provisioning/testing **nothing** is ``incomplete`` — never ``ok`` — so the
        vacuous ``0/0`` a fresh clone hits can no longer read as "✅ finished
        successfully". A Tier-2 run that was skipped (no key, no dataset) is not a
        failure; one that ran and ended FAIL/ERROR/TIMEOUT is.
        """
        st = self.state
        base_present = st.phase_status("base_env") in (Status.DONE.value, Status.SKIPPED.value)
        provisioned = [k for k, r in st.tools.items() if r.get("provision_status") == Status.DONE.value]
        tier1_fail = [
            k for k in provisioned if st.tools[k].get("test_status") in (testing.FAIL, testing.ERROR, testing.TIMEOUT)
        ]
        tier1_pass = [k for k in provisioned if st.tools[k].get("test_status") == testing.PASS]
        # a --help import probe is a PASS, not a mini-data verification (hunt 2026-09-30, uL4-honesty-1):
        # the rollup names the servers a real mini-data case verified. An older state without the key
        # keeps the old count.
        rollup_t1 = (st.meta.get("test_rollup") or {}).get("tier1") or {}
        verified = rollup_t1.get("verified_on_mini_data")
        tier1_verified = tier1_pass if verified is None else [k for k in tier1_pass if k in set(verified)]
        # Selected tools whose build genuinely FAILED (distinct from healthy=DONE and
        # confirm-declined=SKIPPED). A run where the survivors pass Tier-1 but some selected
        # tools never built must NOT report a plain green "ok" and hide them.
        build_failed = [
            k for k in st.selected_servers if st.tools.get(k, {}).get("provision_status") == Status.FAILED.value
        ]
        t2counts = (((st.meta.get("test_rollup") or {}).get("tier2") or {}).get("counts")) or {}
        tier2_ran = sum(t2counts.values())
        # A Tier-2 FAIL is often the masked-env case -- a hollow answer hiding a broken tool env, which
        # _score_probe deliberately refuses to call a PASS -- and it was printed as a warning and then
        # crowned "✅ setup complete", exit 0 (hunt 2026-09-30, uL4-honesty-7).
        tier2_fail = sum(t2counts.get(s, 0) for s in (testing.FAIL, testing.ERROR, testing.TIMEOUT))

        if not base_present:
            level = "fail"
            headline = "the base environment was not established"
        elif not provisioned:
            level = "incomplete"
            headline = (
                "no analysis tools were selected — nothing was provisioned or validated"
                if not st.selected_servers
                else f"none of the {len(st.selected_servers)} selected tool(s) could be provisioned"
            )
        elif tier1_fail:
            level = "degraded"
            headline = f"{len(tier1_pass)}/{len(provisioned)} tool env(s) passed Tier-1; {len(tier1_fail)} failed"
        elif build_failed:
            # Survivors passed, but tools the user asked for never built — report it honestly
            # and point at the safe re-run, which rebuilds only the failures (see B2).
            level = "degraded"
            headline = (
                f"{len(st.selected_servers)} selected · {len(provisioned)} built · "
                f"{len(build_failed)} failed to build (re-run `sog-setup` to retry)"
            )
        elif st.meta.get("config_wiring_failed"):
            # Every selected tool built AND passed, but the post-build config wiring (setup config +
            # canonical rewrite) did not finish, so the agent would load a stale/blanket config. Report
            # it honestly (a "degraded" ⇒ exit 1, never a green ✅); finalize is left FAILED so the same
            # re-run re-attempts the wiring (Round-6 A).
            level = "degraded"
            headline = (
                f"{len(provisioned)} tool env(s) built & verified, but wiring the agent config "
                "didn't finish — re-run `sog-setup` to complete it"
            )
        elif tier2_fail:
            level = "degraded"
            # Counts what passed Tier-1, not what was built: with Tier-1 off (or a tool it never reached)
            # "N passed the mini-data test" claimed a test that did not run (hunt 2026-09-30,
            # uL4-honesty-7 repair).
            headline = (
                f"{len(tier1_pass)} tool env(s) passed Tier-1, but {tier2_fail} of {tier2_ran} "
                "real-data agent run(s) (Tier-2) failed"
            )
        else:
            level = "ok"
            probe_only = len(tier1_pass) - len(tier1_verified)
            headline = (
                f"{len(provisioned)} tool env(s) ready ({len(tier1_verified)} verified on mini data"
                + (f", {probe_only} by an import probe only" if probe_only else "")
                + ")"
            )

        return {
            "level": level,
            "headline": headline,
            "n_selected": len(st.selected_servers),
            "n_provisioned": len(provisioned),
            "tier1_pass": len(tier1_pass),
            "tier1_verified": len(tier1_verified),
            "tier1_fail": len(tier1_fail),
            "build_failed": len(build_failed),
            "config_wiring_failed": bool(st.meta.get("config_wiring_failed")),
            "tier2_ran": tier2_ran,
            "tier2_fail": tier2_fail,
        }

    def _finish(self) -> None:
        self.io.section("All done")
        base = self.state.basic_env_name or "(unset)"
        if self.opts.dry_run:
            self.io.ok(f"[dry-run] plan preview complete for base env {base} — nothing was changed")
            self.log.event("run_complete", run_id=self.state.run_id, dry_run=True)
            return

        v = self._final_verdict()
        loc = str(constants.artifact_dir())
        if v["level"] == "ok":
            self.io.ok(f"setup complete for base env {base} — {v['headline']}")
            if v["tier2_ran"] == 0:
                self.io.note(
                    "the real-data agent run (Tier-2) did not execute — add a working LLM key "
                    "(and a mini dataset) and re-run to validate an end-to-end analysis."
                )
        elif v["level"] == "degraded":
            self.io.warn(f"setup finished WITH PROBLEMS for base env {base} — {v['headline']}")
            self.io.note("inspect the failing tool env(s) above; re-run `sog-setup` to retry provisioning/repair.")
            if v.get("tier2_fail"):
                self.io.note("the Tier-2 lines above name each failed category and the tool it ran.")
        else:  # incomplete | fail — the vacuous "nothing validated" case must not read as success
            self.io.warn(f"setup did NOT validate anything for base env {base} — {v['headline']}")
            self.io.note("Re-run `sog-setup` and select at least one analysis category so tools get built and tested.")

        # A5 — self-healing roll-up: name the tool env(s) whose FIRST build failed but the
        # agent-monitored remediation recovered, so a green verdict that leaned on auto-repair
        # is transparent rather than silent.
        attempted = {k: r for k, r in self.state.tools.items() if r.get("repairs")}
        if attempted:
            recovered = [k for k, r in attempted.items() if any(rr.get("repaired") for rr in r["repairs"])]
            self.io.note(
                f"self-healing: recovered {len(recovered)}/{len(attempted)} tool env(s) that failed "
                f"their first build" + (f" ({', '.join(recovered)})" if recovered else "")
            )

        self.io.say(f"  results → {loc}")
        if v["n_provisioned"]:
            # Point at the end-to-end wiring self-check: it confirms the base env can import the agent
            # + MCP client stack AND that every enabled server's interpreter actually exists (catches a
            # tool env that was later removed / a stale config), which the per-env doctor doesn't.
            self.io.note("verify base env, MCP servers, and ST-Coscientist are wired together: `sog-setup conncheck`")
        self.io.note("guideline (install → test → demo → real-run, and how to read this): README.md")
        self.log.event("run_complete", run_id=self.state.run_id, dry_run=False, verdict=v)

    def _note_soft_failure(self, *, event: str, exc: BaseException, user_msg: str) -> None:
        """Record a full (secret-redacted) traceback for a *post-build* soft failure and show a
        terse, reassuring one-liner. Used by the config-wiring steps (setup-config write, finalize
        doctor + canonical rewrite): on a slow/loaded deploy box a live env probe can raise
        ``CondaError`` even with ``check=False`` (envtools ``_exec`` on timeout/OSError). Those steps
        run *after* every env is already built and tested, so a hiccup there must degrade to a
        warning — never unwind to the module-level ``CondaError`` catch and flip a green install to a
        scary exit 3. Mirrors :meth:`_phase_demo`'s soft-wrap; ``except Exception`` still lets
        ``KeyboardInterrupt``/``SystemExit`` (BaseException) through."""
        if self.log is not None:
            self.log.event(
                event,
                # redact BEFORE the length clip (R24-A3): SessionLog.event redacts these fields, but
                # a registered index-URL credential straddling the [:N] boundary is truncated FIRST, so
                # its surviving prefix no longer matches the full secret and slips past event()'s
                # redact into run-*.jsonl. redact-first masks it whole, then the clip is safe.
                error=redact(f"{type(exc).__name__}: {exc}")[:300],
                traceback=redact(traceback.format_exc())[:2000],
            )
        self.io.warn(user_msg)

    def _resolve_ticker(self, every: int = 15, label: str = "servers checked") -> Callable[[int, int, str], None]:
        """A progress callback for the install-aware MCP resolve sweep.

        That sweep health-probes every one of the ~88 servers with a real ``conda run`` import
        check. Measured on a box with a full set of tool envs: **12.5 minutes of silence** between
        "reusing your existing 'moscot' env for squidpy" and the closing "resolved N/M server(s)
        enabled" — with stdout block-buffered when redirected, a logfile tail stayed completely
        empty. Indistinguishable from a hang, right after the user was told their one selected
        tool needed no rebuild.

        Plain ``io.say`` lines (the same idiom ``provision`` uses per tool) rather than a spinner,
        so this reads the same on a TTY and in a redirected log. Ticks every ``every`` servers;
        the caller's "resolved N/M" line closes the phase, so the last tick is left to it.

        ``label`` distinguishes the run's TWO sweeps. The wizard deliberately resolves twice —
        once at the end of ``build`` (so the test phase has a config to run tools through) and
        again in ``finalize``, after ``doctor`` may have repaired an env, since only that second
        pass reflects post-repair truth. Measured on this box the first sweep ran 02:04:54 →
        02:24:22 (**19.5 minutes**), and the second began 30 s later. Without a label the counter
        silently restarts at 15/88 after having just reached 88/88 — which reads as an infinite
        loop, the exact anxiety this ticker exists to remove.
        """

        def tick(done: int, total: int, key: str) -> None:
            if done % every == 0 and done < total:
                self.io.say(f"    …{done}/{total} {label} (now: {key})")

        return tick

    def _phase_demo(self) -> bool:
        """The soft ``demo`` phase (Part D) — offer a demo on a passed tool + a real run on the
        freshly-installed env. Delegates to :func:`demo.run_demo_phase`; **always returns ``True``**
        (a declined/skipped demo never fails the run — ``demo`` is in ``state.SOFT_PHASES`` and does
        not gate ``_final_verdict``). Runs after ``_finish`` so it never precedes the honest verdict."""
        name = "demo"
        self.state.set_phase(name, Status.IN_PROGRESS)
        try:
            outcome = demo.run_demo_phase(
                state=self.state,
                io=self.io,
                conda=self.conda,
                specs=self.specs,
                cats=self.cats,
                log=self.log,
                dry_run=self.opts.dry_run,
                stated_goals=self.ctx.stated_goals or "",
                llm=self.choice,
            )
            self.state.set_phase(name, Status.SKIPPED if outcome.skipped else Status.DONE)
        except Exception as exc:  # a soft phase must never break a completed install
            if self.log is not None:
                # Record the full traceback (secret-redacted by SessionLog) so a soft-phase failure is
                # diagnosable from the JSONL alone — the terse user-facing warn stays a bare exception name.
                self.log.event(
                    "demo_phase_error",
                    # redact-before-clip: see the note in _note_soft_failure (R24-A3).
                    error=redact(f"{type(exc).__name__}: {exc}")[:300],
                    traceback=redact(traceback.format_exc())[:2000],
                )
            self.io.warn(f"demo step skipped (non-fatal): {type(exc).__name__}")
            self.state.set_phase(name, Status.SKIPPED)
        self._persist()
        return True

    def _open_log(self) -> None:
        """Open the per-run JSONL transcript and wire it into I/O + conda.

        Deferred until here because the log path embeds the (possibly just-loaded)
        run id, and secret masking must be live before any key is read.
        """
        # Extend this method's "masking is live before any secret is used" contract to the credentials
        # in any configured pip/conda/uv index URL, so a https://user:token@host source is redacted from
        # build stderr and the build-log FILE sinks (C1). Covers both the environment the build
        # subprocesses inherit and the .env file they may source; both calls are best-effort. Masking is
        # a pure in-memory op, so it runs even under --dry-run (nothing is written, but nothing leaks).
        register_index_url_secrets(env={**llm_setup.read_dotenv_values(), **os.environ})
        log_path = constants.logs_dir() / f"run-{self.state.run_id}.jsonl"
        if self.opts.dry_run:
            # Read-only plan preview: console echo only, no state dir, no JSONL transcript (N4).
            self.log = _DryRunLog(log_path)
        else:
            constants.ensure_state_dirs()
            self.log = SessionLog(log_path)
        self.io.log = self.log
        if self._conda is not None:
            self._conda.log = self.log
        self.log.event(
            "run_start",
            run_id=self.state.run_id,
            resume=self.state.has_progress(),
            dry_run=self.opts.dry_run,
            scripted=self.answers is not None,
        )

    # ------------------------------------------------------------------ #
    # State / resume
    # ------------------------------------------------------------------ #
    def _load_or_start_state(self) -> None:
        # A dry run is a read-only plan preview: start fresh, never touch (or read)
        # the real state file, so it can't make the next real run think there's a
        # run to resume. `_persist` is a no-op under --dry-run for the same reason.
        if self.opts.dry_run:
            self.state = SetupState()
            return
        prior = SetupState.load()
        if prior is None or not prior.has_progress():
            self.state = prior or SetupState()
            self._persist()
            return
        if self.opts.restart:
            arch = archive_state()
            self.io.note(f"--restart: archived prior state → {arch}")
            self.state = SetupState()
        elif self.opts.resume or self.answers is not None or prior.is_complete():
            # scripted runs resume silently; a finished run starts a fresh cursor
            if prior.is_complete():
                # Archive the completed run's ledger BEFORE the fresh cursor's _persist() (below)
                # overwrites setup_state.json — mirror the --restart/interactive-restart siblings.
                # The ledger is the only record of which <basic>_* envs this run built; without
                # archiving it here, a subsequent ledger-aware `reset` can't find them to clean. (S1)
                arch = archive_state()
                if arch is not None:
                    self.io.note(f"prior run complete: archived its ledger → {arch}")
            self.state = SetupState() if prior.is_complete() else prior
            # Wizard-F2: a scripted (--answers) run shares this resume-in-place path. If a reused workspace's
            # INCOMPLETE prior selected a DIFFERENT tool set than the current answers file, the resumed
            # selection wins and the file's is silently ignored — breaking the --answers reproducibility
            # guarantee. Don't auto-restart (an identical interrupted run must still resume its partial
            # build); warn loudly + point at --restart so the divergence is visible, never silent.
            if self.answers is not None and self.state is prior and self._scripted_selection_diverged(prior):
                self.io.warn(
                    "the --answers file's tool selection differs from the interrupted run being resumed; "
                    f"resuming the prior selection ({sorted(prior.selected_servers)}) and IGNORING the file's. "
                    "Pass --restart to discard the prior run and apply the answers file instead."
                )
        else:
            choice = self.io.select(
                "A previous setup run was found. Resume it, or restart from scratch?",
                [
                    ("resume", "Resume", "continue where it left off"),
                    ("restart", "Restart", "archive it and begin anew"),
                ],
                default="resume",
            )
            if choice == "restart":
                archive_state()
                self.state = SetupState()
            else:
                self.state = prior
        reset = self.state.reset_interrupted()
        if reset:
            self.io.note(f"re-running interrupted phase(s): {', '.join(reset)}")
        self._persist()

    def _scripted_selection_diverged(self, prior: SetupState) -> bool:
        """True if the current ``--answers`` file selects a DIFFERENT tool set than the incomplete prior
        state about to be resumed — the file's declared categories differ, or the tools it expands to
        differ from the prior run's. Returns ``False`` when the prior hasn't selected anything yet
        (``category_select`` will still run) so an identical interrupted run never trips it.

        The tools are compared EXPANDED, the way ``category_select`` would expand them: comparing the raw
        ``servers`` field missed a file narrowed to ``servers: [tangram]`` inside a prior 31-tool
        category (a subset), and a file widened back to the whole category (no ``servers``), so either
        resumed the other selection without a word (hunt 2026-09-30, u35-setup-ux-9)."""
        if not (prior.selected_categories or prior.selected_servers):
            return False
        ans = self.answers or {}
        new_cats = list(ans.get("categories") or [])
        new_servers = list(ans.get("servers") or [])
        if new_cats and set(new_cats) != set(prior.selected_categories):
            return True
        try:
            _ = self.cats  # populates ctx.categories
        except Exception as exc:
            # Said, not swallowed: the raw-field check below is the comparison this replaced, and it
            # misses a narrowed or widened selection (hunt 2026-09-30, u35-setup-ux-9 repair).
            self.io.warn(
                f"could not read the tool catalog ({type(exc).__name__}); comparing the --answers file's "
                "tool selection with the interrupted run by its raw categories/servers fields only"
            )
        if self.ctx.categories and (new_cats or new_servers):
            expanded = expand_category_servers(new_cats, new_servers or None, self.ctx.categories)
            return set(expanded) != set(prior.selected_servers)
        if new_servers and not set(new_servers).issubset(set(prior.selected_servers)):
            return True
        return False

    def _stale_completion(self, name: str) -> bool:
        """True if a phase marked done/skipped in the *loaded* state is not actually
        usable and must re-run despite the resume cursor.

        Two cases:

        * ``category_select`` that persisted **0 tools** — a run can never provision from
          an empty selection, so skipping it (and hydrating an empty ``ctx.selected``)
          would make the whole downstream pipeline a vacuous ``0/0``. Re-running it
          re-applies the fixed guard, which fails loudly instead.
        * ``provision``/``test``/``finalize`` when a prior run left a *selected* tool unbuilt
          (B2). The provision phase records ``DONE`` even when individual tool builds fail, so
          the resume cursor would otherwise skip straight past the failures to (and through)
          ``finalize``. Re-opening all three makes "2 tools failed — just run ``sog-setup``
          again" rebuild only the failures (``provision_one`` skips healthy envs), re-test the
          results, AND re-run ``finalize`` so the recovered tool is actually wired into the
          agent's canonical ``agent/MCP_server/mcp_config.yaml`` (``_apply_canonical`` is the sole
          writer, and it regenerates idempotently from live state). Without re-opening finalize,
          the retry rebuilds + passes the tool but the run reports success while the canonical
          still carries it ``enabled: false`` from the failed first run — the agent never sees
          it. The decision is snapshotted once in :meth:`run` so provision rebuilding a tool
          mid-loop can't flip ``test``/``finalize`` back to skipped.
        """
        if name == "category_select":
            return not self.state.selected_servers
        if name in ("provision", "test", "finalize"):
            return self._retry_unbuilt
        return False

    def _working_servers(self) -> list[str]:
        """The selected servers THIS invocation should provision/test: the full persisted selection,
        narrowed to ``--only`` when that flag is set.

        ``--only`` is a per-run *filter*, NOT a mutation of ``state.selected_servers`` — that stays the
        user's full intent, so (a) a fresh interactive run with ``--only`` never loses the other guide
        picks (a later plain re-run still builds them), and (b) the final config / ``doctor`` still cover
        every selected tool. Because the filter is recomputed here on EVERY invocation, ``--only`` also
        works on a *resume*, where ``category_select`` is skipped and the old in-phase narrowing never
        ran (so ``--only`` was silently ignored and provisioning rebuilt everything)."""
        servers = list(self.state.selected_servers)
        if not self.opts.only:
            return servers
        keep = set(self.opts.only)
        return [s for s in servers if s in keep]

    def _has_unbuilt_selected(self) -> bool:
        """True if any tool this run should build (``_working_servers`` — the selection, narrowed to
        ``--only``) never reached a terminal-good provision state — its build FAILED or no provision
        record exists. Healthy (``DONE``) and deliberately confirm-declined (``SKIPPED``) tools don't
        count, so this is ``False`` once every such tool is built or intentionally skipped (and on a
        fresh run, where nothing is selected yet). Honoring ``--only`` here means ``sog-setup --only X``
        does not trigger a retry just because some OTHER selected tool is still unbuilt."""
        good = (Status.DONE.value, Status.SKIPPED.value)
        for key in self._working_servers():
            if self.state.tools.get(key, {}).get("provision_status") not in good:
                return True
        return False

    def _rehydrate_after_skip(self, name: str) -> None:
        """When a Stage-B phase is skipped on resume, pull its result out of state
        into the live context so downstream phases have it."""
        if name == "category_select":
            self.ctx.selected = CategorySelection(
                categories=list(self.state.selected_categories),
                servers=list(self.state.selected_servers),
            )
        elif name == "base_env" and self.state.basic_env_name:
            self.ctx.base_env = BaseEnvDecision(
                mode=_coerce_base_mode(self.state.base_env_mode),
                basic_env_name=self.state.basic_env_name,
            )

    def _persist(self) -> None:
        if self.opts.dry_run:
            return  # a plan preview writes no resume-triggering state
        self.state.save()

    def _enter(self, name: str) -> None:
        self.state.set_phase(name, Status.IN_PROGRESS)
        self._persist()

    def _leave(self, name: str, status: Status) -> None:
        self.state.set_phase(name, status)
        self._persist()

    # ------------------------------------------------------------------ #
    # Signals — flush state on Ctrl-C so the run stays resumable
    # ------------------------------------------------------------------ #
    def _install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            # A signal handler MUST always reach its clean exit (a515#1): any failure while persisting
            # state or printing the notice — a broken stderr pipe (``BrokenPipeError``), a disk-full
            # during save, a lock momentarily held by the interrupted frame, or a reentrant second
            # Ctrl-C landing mid-handler — must NOT replace exit 130 with a scary secondary traceback.
            # So every side effect is best-effort (``except BaseException`` swallows even a nested
            # KeyboardInterrupt / SystemExit raised inside it) and ``raise SystemExit(130)`` is the
            # unconditional final statement, never buried in a ``finally`` a preceding line could preempt.
            try:
                # Route through _persist so a Ctrl-C during a --dry-run PREVIEW writes nothing —
                # `self.state.save()` here bypassed the dry-run guard and left a phantom
                # `.sog_setup/setup_state.json` (phases already advanced in memory) that a later
                # real run would resume/prompt on, breaking the dry-run non-interference invariant.
                self._persist()
            except BaseException:
                pass
            try:
                if self.log is not None:
                    self.log.event("interrupted", signal=signum)
            except BaseException:
                pass
            try:
                if self.opts.dry_run:
                    self.io.warn("\ninterrupted — dry-run, nothing was saved.")
                else:
                    self.io.warn("\ninterrupted — state saved; re-run `sog-setup` to resume.")
            except BaseException:
                pass
            raise SystemExit(130)

        self._saved_signals = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, _handler)
                if prev is not None:  # None: not installed from Python, so it cannot be put back
                    self._saved_signals.append((sig, prev))
            except (ValueError, OSError):
                pass  # not on the main thread (e.g. under a test harness) — skip

    def _restore_signal_handlers(self) -> None:
        for sig, prev in reversed(getattr(self, "_saved_signals", [])):
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError, TypeError):
                pass
        self._saved_signals = []

    # ------------------------------------------------------------------ #
    # Stage A
    # ------------------------------------------------------------------ #
    def _phase_preflight(self) -> bool:
        name = "preflight"
        self._enter(name)
        # A dry-run prunes nothing: the writability probe's default prune unlinked the oldest run logs
        # once the box held more than the cap, on a run whose contract is "make no changes" -- the F2
        # fix `reset --dry-run` already has (hunt 2026-09-30, u35-setup-ux-8).
        results = run_preflight(
            check_net=self.opts.check_net, prune=not self.opts.dry_run, create=not self.opts.dry_run
        )
        for r in results:
            (self.io.ok if r.level == "ok" else self.io.warn if r.level == "warn" else self.io.err)(
                f"{r.name}: {r.detail}"
            )
        self.ctx.preflight = summarize(results)
        self.ctx.current_env = os.environ.get("CONDA_DEFAULT_ENV")
        if has_hard_failure(results):
            self.io.err("preflight found a blocking problem — cannot continue.")
            self._leave(name, Status.FAILED)
            return False
        self._leave(name, Status.DONE)
        return True

    def _phase_onboarding(self) -> None:
        name = "onboarding"
        # If a prior run already connected the LLM and we're resuming, onboarding's
        # reuse path still runs (cheap) to rebuild the source — so we always execute.
        self._enter(name)
        llm_answers = (self.answers or {}).get("llm") if self.answers else None

        if self.opts.dry_run and self.answers is not None:
            # Scripted dry-run: synthesize the choice from the file — no prompts, no .env write.
            self.choice = _synthesize_choice(llm_answers or {})
            self.io.note(f"[dry-run] would configure LLM: {self.choice.source}/{self.choice.model}")
        else:
            # Steps BEFORE onboarding: the banner is printed as the LLM step starts, so "+ 1" showed
            # "[✓] 1. Connect your LLM" before any key was entered (hunt 2026-09-30, u35-setup-ux-16).
            roadmap_done = tuple(range(PHASES.index("onboarding")))
            # Interactive dry-run (no --answers) still collects/validates so the user sees what
            # would happen, but `persist=False` makes it write nothing — no .env/backup/vault (N4).
            self.choice = onboarding.run_onboarding(
                self.io,
                self.log,
                answers=llm_answers,
                knobs=(self.answers or {}).get("knobs"),
                roadmap_done=roadmap_done,
                persist=not self.opts.dry_run,
            )

        # Record NON-SECRET LLM facts in state (source/model/validated only).
        self.state.llm = {
            "source": self.choice.source,
            "model": self.choice.model,
            "validated": self.choice.validated,
        }
        self.source = self._build_source()
        self._leave(name, Status.DONE)

    def _build_source(self) -> DecisionSource:
        """Scripted (``--answers``) or LLM-guided, chosen by how the run was invoked."""
        if self.answers is not None:
            return ScriptedSource(self.answers)
        # Interactive: seed the stdlib chat client with the just-verified key.
        assert self.choice is not None
        client = ChatClient(
            source=self.choice.source,
            model=self.choice.model,
            field_values=self.choice.field_values,
        )
        return InteractiveGuide(self.io, chat_client=client, log=self.log)

    def _capture_stated_goals(self) -> None:
        """Grab the user's intent once, upstream of every Stage-B decision."""
        if self.answers is not None:
            self.ctx.stated_goals = str(self.answers.get("goals", "") or "")
            return
        if self.state.phase_status("category_select") in (Status.DONE.value, Status.SKIPPED.value):
            return  # resuming past selection — no need to ask again
        self.io.section("Now let's set up your tools")
        self.ctx.stated_goals = self.io.ask_text(
            "🧬 What are you trying to do? (e.g. 'Visium data, want cell-type deconvolution')",
            default="",
        )

    # ------------------------------------------------------------------ #
    # Stage B
    # ------------------------------------------------------------------ #
    def _phase_category_select(self) -> bool:
        name = "category_select"
        self._enter(name)
        _ = self.cats  # populate ctx.categories + ctx.tool_groups
        # A deterministic, goal-first starter set (never empty). The guide pre-selects it in
        # the picker whenever the LLM has no suggestion of its own (no key / failed reply), so
        # a non-expert always sees a sensible small set marked ready-to-accept. ScriptedSource
        # ignores ctx.extra, so headless/CI runs are unchanged.
        self.ctx.extra["recommended"] = recommended_servers(self._cats, self.ctx.stated_goals or "")
        sel = self.source.choose_categories(self.ctx)
        if not sel.servers:
            # 0 servers = nothing to provision. Refuse to complete this phase (which would
            # persist an empty selection and make every downstream phase a vacuous 0/0);
            # fail clearly so the user re-runs and picks categories that map to tools.
            picked = list(sel.categories)
            # Categories the user picked that genuinely back NO installable server — the skill-only
            # skills (omics / spatial_analysis / data_conversion). `installable` is the set of
            # categories that DO map to tools (empty only in a bare / unreadable-catalog env, where
            # we can't trust the classification and fall back to the generic message).
            skill_only = [c.name for c in (self._cats or []) if c.name in picked and not c.servers]
            installable = [c.name for c in (self._cats or []) if c.servers]
            if picked and installable and set(skill_only) == set(picked):
                # R3-F2: every category the user picked is a skill-only skill with no dedicated tool
                # env, so there is genuinely nothing to build — but they DID pick something. Don't
                # claim "no tools were selected"; name what they picked and steer them to installable
                # categories instead of a misleading dead-end.
                self.io.err(
                    f"the categor(ies) you picked ({', '.join(picked)}) are analysis skills that don't "
                    f"map to an installable tool, so there's nothing to build. Pick at least one "
                    f"category that installs tools (e.g. {', '.join(installable[:5])}), then re-run "
                    f"`sog-setup`."
                )
            else:
                self.io.err(
                    "no analysis tools were selected — pick at least one analysis category "
                    "that maps to tools, then re-run `sog-setup`."
                )
            self._leave(name, Status.FAILED)
            return False
        # NOTE: ``--only`` is applied later, as a per-run provisioning/testing FILTER (see
        # ``_working_servers`` / ``_phase_provision``), NOT here as a narrowing of the persisted
        # selection. Persisting the full guide/answers selection means a fresh ``--only`` run keeps every
        # pick for a later resume, and ``--only`` also takes effect on a resume (which skips this phase).
        self.ctx.selected = sel
        self.state.selected_categories = list(sel.categories)
        self.state.selected_servers = list(sel.servers)
        self.io.ok(f"selected {len(sel.servers)} tool(s) across {len(sel.categories)} categor(ies)")
        self._leave(name, Status.DONE)
        return True

    def _phase_base_env(self) -> bool:
        name = "base_env"
        self._enter(name)
        dec = self.source.choose_base_env(self.ctx)
        self.ctx.base_env = dec
        self.state.basic_env_name = dec.basic_env_name
        self.state.base_env_mode = dec.mode.value
        self._persist()
        if self.opts.dry_run:
            self.io.note(
                f"[dry-run] base env: {dec.mode.value} '{dec.basic_env_name}' (install_editable={dec.install_editable})"
            )
            self._leave(name, Status.DONE)
            return True
        try:
            res = base_env.establish(self.conda, dec, io=self.io, confirm=self._confirm(), log=self.log)
        except CondaError as exc:
            # N12: a base-env conda step can still raise CondaError on a TIMEOUT/OSError even with
            # check=False (envtools._exec translates those regardless). Degrade to a friendly
            # *phase-fail* — persist the full stderr for diagnosis, tell the user how to resume, mark
            # the phase FAILED — instead of unwinding to the module-level exit-3 catch and turning a
            # recoverable base-env hiccup into a raw traceback on the deploy box.
            if self.log is not None:
                self.log.event(
                    "base_env_conda_error",
                    # redact-before-clip: see the note in _note_soft_failure (R24-A3). The base-env
                    # solve echoes the (credentialed) index/channel URL in its stderr, so this site is
                    # the most likely of the three to actually carry a straddling secret.
                    error=redact(f"{type(exc).__name__}: {exc}")[:300],
                    stderr=redact(getattr(exc, "stderr", "") or "")[:4000],
                    traceback=redact(traceback.format_exc())[:2000],
                )
            self.io.err(f"  base env could not be built: {exc}")
            self.io.note("  the full error is in the run log; re-run `sog-setup` to resume from here.")
            self._leave(name, Status.FAILED)
            return False
        if not res.ok and not res.skipped:
            for m in res.messages:
                self.io.err(f"  {m}")
            self._leave(name, Status.FAILED)
            return False
        self._leave(name, Status.SKIPPED if res.skipped else Status.DONE)
        return True

    def _phase_provision(self) -> bool:
        name = "provision"
        self._enter(name)
        self._remove_orphan_tool_envs()  # clean any partial build a prior crash left
        basic = self.state.basic_env_name or "sog"
        if self.opts.only and self.state.selected_servers:
            working = self._working_servers()
            if not working:
                # `--only` matched none of the selection — fail fast (as the old in-phase check did), but
                # now from the provision phase so it also fires on a resume (where category_select is skipped).
                self.io.err(
                    f"--only={sorted(set(self.opts.only))} matched none of the "
                    f"{len(self.state.selected_servers)} selected tool(s) {sorted(self.state.selected_servers)}; "
                    "nothing to provision — check the server keys."
                )
                self._leave(name, Status.FAILED)
                return False
            skipped = [s for s in self.state.selected_servers if s not in set(working)]
            if skipped:
                self.io.note(
                    f"--only: provisioning {working} this run; skipping {skipped} "
                    "(still selected — a plain re-run builds them)"
                )
        candidates = self._candidate_tools(basic)
        self.ctx.extra["candidate_tools"] = candidates
        plan = self.source.plan_provision(self.ctx)

        self._collect_service_keys(plan.tools)

        if self.opts.dry_run:
            self.io.note("[dry-run] would provision, in order:")
            for tp in plan.tools:
                self.io.say(
                    f"    • {tp.target_env:28s} {tp.strategy.value}" + (f"  (~{tp.est_gb:.1f} GB)" if tp.est_gb else "")
                )
            self.io.note(f"[dry-run] would write install-aware MCP config → {constants.generated_mcp_config()}")
            self._leave(name, Status.DONE)
            return True

        def _mark_start(tp: ToolPlan) -> None:
            # Fingerprint the in-flight tool BEFORE its build starts, so a crash here
            # leaves an in_progress record that resume uses to force a clean rebuild.
            self.state.set_tool(tp.server_key, target_env=tp.target_env, provision_status=Status.IN_PROGRESS.value)
            self._persist()

        def _mark_done(r: provision.ToolResult) -> None:
            self.state.set_tool(
                r.server_key,
                target_env=r.target_env,
                provision_status=(
                    Status.DONE.value if r.ok else Status.SKIPPED.value if r.skipped else Status.FAILED.value
                ),
                strategy=r.strategy,
                fell_back=r.fell_back,
                self_review=r.self_review,
                repairs=r.repairs,
                messages=r.messages,
                needs_attention=r.needs_attention,
            )
            self._persist()

        # A4 — the agentic remediation planner runs ONLY when a validated LLM handle exists
        # (interactive + a working key; None for scripted/no-key) AND the opt-in
        # SOG_PROVISION_LLM_REMEDIATION is set (re-checked inside the planner). Otherwise the
        # self-heal stays fully deterministic (A2 loop + A3 off-index repair). It reuses the
        # wizard's own stdlib ChatClient — never pulls in the heavy agent stack.
        remediation = getattr(self.source, "chat", None)

        # Announce the reuse pre-scan (R1) so a "reusing your existing '<env>' env — no rebuild
        # needed" line reads as intended, not as a skipped tool: before creating anything, each
        # tool is checked against the envs already on this box and wired READ-ONLY to any that are
        # already healthy for it (most often the very source env a clone would duplicate), building
        # only what is genuinely missing. Suppressed under SOG_SETUP_NO_ENV_REUSE (reuse disabled).
        if mcp_resolver._reuse_source_envs_enabled():
            self.io.say(
                "Checking your existing conda envs first — I'll reuse any that are already healthy "
                "for a tool and only build what's missing."
            )

        # The build itself — per-tool state is persisted by the _mark_done callback, so the
        # return value is not needed; the wizard's config is derived from live env health below.
        provision.provision_all(
            self.conda,
            self.specs,
            plan,
            basic,
            confirm=self._confirm(),
            io=self.io,
            log=self.log,
            on_tool_start=_mark_start,
            on_tool_done=_mark_done,
            remediation=remediation,
        )

        # Resolve ALL servers from LIVE install state — not just this run's builds.
        # A build that reported ok but is actually unhealthy honestly lands enabled:false,
        # and servers satisfied by a shared/protected env (squidpy→moscot, svca→spatialomicsgym_e1)
        # light up without a static rule. This writes the wizard's own setup config only;
        # the agent's canonical config is rewritten later, in finalize (post-repair health).
        try:
            self.io.say("  checking which tool servers can actually run on this box…")
            cfg_path, entries = mcp_resolver.write_setup_config(
                self.conda, basic, self.specs, log=self.log, on_progress=self._resolve_ticker()
            )
            on = sum(1 for e in entries if e.enabled)
            self.io.ok(f"resolved {on}/{len(entries)} server(s) enabled → {cfg_path}")
            gaps = [e.server_key for e in entries if e.static_gap]
            if gaps:
                self.io.note(f"not config-repointable (no interpreter override): {', '.join(gaps)}")
        except Exception as exc:  # a config-wiring hiccup must never sink an otherwise-complete build
            self._note_soft_failure(
                event="setup_config_error",
                exc=exc,
                user_msg=(
                    f"couldn't write the tool config just now ({type(exc).__name__}) — your tool "
                    "envs are built; re-run `sog-setup` to finish wiring them."
                ),
            )
        self._leave(name, Status.DONE)
        return True

    def _remove_orphan_tool_envs(self) -> None:
        """Remove any ``<basic>_<server>`` env a prior *interrupted* provision left in a
        partial state, so this run rebuilds it cleanly instead of trusting a possibly
        half-built env (health probes can't see a tool with no import_check, so the env
        must be removed, not re-probed). A tool is fingerprinted ``in_progress`` the
        moment its build starts (``_phase_provision``'s ``on_tool_start``); on resume
        that record names the orphan. Honors the namespace guard and the dry-run gate —
        the concrete cleanup ``state.reset_interrupted`` documents but never performed."""
        if self.opts.dry_run:
            return
        basic = self.state.basic_env_name
        if not basic:
            return
        for target in self.state.orphan_tool_envs():
            try:
                constants.assert_deletable_env(basic, target)
            except PermissionError:
                continue  # never touch anything outside the '<basic>_*' namespace
            try:
                if self.conda.env_exists(target):
                    self.io.note(f"removing orphaned env '{target}' from an interrupted run")
                    self.conda.remove_env(target)
                    if self.log is not None:
                        self.log.event("orphan_env_removed", env=target)
            except (CondaError, OSError) as exc:
                # This pre-provision cleanup runs BEFORE a single tool is built and OUTSIDE the phase's
                # own error handling. A locked / half-written / slow-to-remove orphan (conda timeout or
                # a pkg-cache lock → CondaError even with check=False; an OSError under env_exists) must
                # NOT abort the entire wizard with exit 3 here. Note it and move to the next orphan — the
                # tool stays fingerprinted in_progress, so build() re-cleans and rebuilds it regardless,
                # and a genuinely-undeletable collision surfaces honestly at that build, not up front.
                self.io.note(f"could not remove orphaned env '{target}' ({type(exc).__name__}); continuing")
                if self.log is not None:
                    self.log.event("orphan_env_remove_failed", env=target, error=str(exc))

    def _candidate_tools(self, basic: str) -> list[ToolPlan]:
        """Turn the selected server keys (narrowed to ``--only`` for this run) into concrete per-tool
        provisioning plans. Reads ``_working_servers`` — NOT ``selected_servers`` directly — so ``--only``
        restricts what gets built on every invocation, resume included."""
        plans: list[ToolPlan] = []
        for key in self._working_servers():
            spec = self.specs.get(key)
            if spec is None:
                self.io.warn(f"no install spec for '{key}' — it will be skipped")
                continue
            plans.append(
                ToolPlan(
                    server_key=key,
                    target_env=spec.target_env(basic),
                    strategy=spec.build_strategy,
                    est_gb=spec.est_gb,
                    gpu=spec.gpu,
                    service_keys=list(spec.service_keys),
                    source_env=spec.source_env,
                    recipe=spec.recipe,
                )
            )
        return plans

    def _collect_service_keys(self, tools: list[ToolPlan]) -> None:
        """Ask (via the decision source) for any tool-gating credential a selected
        tool needs and isn't already set. Best-effort: a declined key never blocks
        provisioning — the env still builds; only that tool's external-API feature
        is unavailable until the key is added. Written through the safe ``.env`` merge."""
        # A credential counts as "already provided" only if it holds a *real* value. An empty
        # `UCD_TOKEN=` or a copied-in `<your-token>` placeholder (a `cp .env.example .env` leaves
        # exactly these) must still prompt — otherwise the env builds but that tool's external-API
        # feature silently fails at runtime (N6). os.environ wins over .env, so a real exported value
        # overrides a stub left in the file.
        env_vals = {**llm_setup.read_dotenv_values(), **os.environ}
        present = {
            k for k, v in env_vals.items() if isinstance(v, str) and not llm_setup.value_looks_like_placeholder(v)
        }
        needed: list[str] = []
        for tp in tools:
            for var in tp.service_keys:
                if var not in present and var not in needed:
                    needed.append(var)
        if not needed:
            return
        collected: dict[str, str] = {}
        skipped: list[str] = []
        for var in needed:
            val = None if self.opts.dry_run else self.source.collect_service_key(var, self.ctx)
            if val:
                collected[var] = val
            else:
                skipped.append(var)
        if collected and not self.opts.dry_run:
            llm_setup.write_dotenv(collected)
            self.io.ok(f"saved {len(collected)} service key(s) to .env (backed up first)")
        if skipped:
            self.io.note(f"no key provided for {', '.join(skipped)} — those tools build, but that feature stays off")
            self.state.meta["service_keys_skipped"] = sorted(
                set(self.state.meta.get("service_keys_skipped", []) + skipped)
            )

    def _phase_test(self) -> bool:
        name = "test"
        self._enter(name)
        tdec = self.source.choose_tests(self.ctx)
        basic = self.state.basic_env_name or "sog"
        if self.opts.dry_run:
            # The decision's categories as they are: an explicit [] is "no category tests", not the
            # build's categories (hunt 2026-09-30, u35-setup-ux-13).
            self.io.note(
                f"[dry-run] would test: tier1={tdec.run_tier1} tier2={tdec.run_tier2} "
                f"categories={tdec.categories_to_test}"
            )
            self._leave(name, Status.DONE)
            return True
        provisioned = [
            k for k, rec in self.state.tools.items() if rec.get("provision_status") == Status.DONE.value
        ] or self._working_servers()  # --only-aware fallback when nothing has a DONE record yet
        if self.opts.only:
            # --only restricts testing as well as provisioning: every tool a PRIOR run built was
            # re-tested (Tier-1 up to 900 s each) on a run told to touch one (hunt 2026-09-30,
            # u35-setup-ux-10).
            keep = set(self._working_servers())
            provisioned = [k for k in provisioned if k in keep]
        # Same stdlib chat handle the provision phase uses (None for scripted/no-key runs): with the
        # opt-in SOG_TEST_LLM_REMEDIATION gate it lets the test tiers' InstallerScientist loop consult
        # the fail-closed LLM planner. Deterministic self-heal needs no handle and stays default-on.
        remediation = getattr(self.source, "chat", None)
        report = testing.run_tests(
            self.conda,
            self.specs,
            provisioned,
            basic,
            tdec,
            self.cats,
            io=self.io,
            log=self.log,
            remediation=remediation,
        )
        for st in report.servers:
            self.state.set_tool(st.server_key, test_status=st.status, needs_attention=st.needs_attention)
        # Stash the rollup so the final verdict can tell a real-data (Tier-2) run that
        # actually executed from one that was skipped (no key / no dataset).
        self.state.meta["test_rollup"] = report.summary()
        self._persist()
        self._leave(name, Status.DONE)
        return True

    def _phase_finalize(self) -> bool:
        name = "finalize"
        self._enter(name)
        basic = self.state.basic_env_name or "sog"
        if not self.opts.dry_run:
            try:
                doctor.doctor(
                    basic,
                    servers=list(self.state.selected_servers),
                    specs=self.specs,
                    conda=self.conda,
                    io=self.io,
                    check_net=False,
                )
                # doctor may have repaired envs in THIS phase, so `enabled:` is only now most
                # accurate. Resolve ONCE here and refresh BOTH configs from that single post-repair
                # truth (a1c#1): the wizard's own setup config (first written pre-doctor in
                # _phase_build) AND the agent's canonical — otherwise mcp_config.setup.yaml (which
                # SOG_MCP_CONFIG advertises, and the env_overrides.env beside it) stays stale
                # relative to the canonical. The setup refresh runs even under --keep-agent-config
                # (it still carries the truth); only the canonical rewrite is gated inside it.
                # Second full sweep of the run (see ``_resolve_ticker``): say so up front, because
                # the per-server counter is about to start over from 1 right after the build phase
                # counted all the way to N.
                self.io.say("  re-checking every tool server now that repairs are done (last long step)…")
                resolved = mcp_resolver.resolve_full_config(
                    self.conda, basic, self.specs, on_progress=self._resolve_ticker(label="servers re-checked")
                )
                mcp_resolver.write_resolved_setup_config(*resolved, log=self.log)
                self._apply_canonical(basic, resolved=resolved)
                # Wiring completed — clear any stale degrade flag left by a PRIOR failed finalize
                # (this run is the retry that succeeded), so the verdict returns to a clean ✅.
                self.state.meta.pop("config_wiring_failed", None)
                # The USER-tool overlay (mcp_config_user.yaml) gets the same on-this-box rewrite
                # the canonical just got — but WARN-only: the user layer is optional, and a snag
                # here must never set ``config_wiring_failed`` (that flag owns the canonical's
                # FAILED/resume semantics). The helper swallows its own exceptions.
                self._rebase_user_layer(basic)
            except Exception as exc:  # finalize is post-build: never flip a green install to a scary exit 3
                # ...but don't silently report ✅ either. The agent config the tools rely on was NOT
                # (re)written, so the agent would load a stale/blanket config. Record a durable flag:
                # `_final_verdict` downgrades an otherwise-green run to "degraded" (honest exit 1, not the
                # scary exit 3), and `_leave` below marks finalize FAILED so a resume actually RE-ATTEMPTS
                # the wiring — a DONE phase is skipped on resume, which would make the "re-run to finish
                # wiring" advice hollow (Round-6 A). Cleanup-only snags (below) never set this flag.
                self.state.meta["config_wiring_failed"] = True
                self._note_soft_failure(
                    event="finalize_error",
                    exc=exc,
                    user_msg=(
                        f"finishing touches hit a snag ({type(exc).__name__}) — your tool envs are "
                        "built and tested; re-run `sog-setup` to finish wiring the agent config."
                    ),
                )
        else:
            verb = "keep" if self.opts.keep_agent_config else "rewrite (with a timestamped backup)"
            self.io.note(f"[dry-run] would {verb} the agent config {constants.original_mcp_config()}")
        # Optional post-build cleanup (delete the test artifacts). Like the doctor/canonical block
        # above, this runs AFTER every env is built + tested, so a hiccup here — a live-probe
        # CondaError inside the decision, an OSError deleting an artifact — must degrade to a warning,
        # never unwind to the module-level catch and flip a green install to exit 3. ``except
        # Exception`` still lets KeyboardInterrupt/SystemExit through to the wizard's SIGINT handling.
        try:
            cdec = self.source.choose_cleanup(self.ctx)
            if cdec.delete_artifacts and not self.opts.dry_run:
                if self._confirm()(
                    Proposal(
                        action="delete test artifacts",
                        detail=str(constants.artifact_dir()),
                        kind="delete_artifacts",
                    )
                ):
                    self._delete_artifacts()
        except Exception as exc:  # post-build cleanup is optional — a snag here never fails the install
            self._note_soft_failure(
                event="finalize_cleanup_error",
                exc=exc,
                user_msg=(
                    f"couldn't tidy up the test artifacts ({type(exc).__name__}) — harmless; your tool "
                    f"envs are built and tested. Remove them anytime with `sog-setup reset`."
                ),
            )
        # A config-wiring failure above set ``config_wiring_failed`` — mark finalize FAILED (not DONE)
        # so the resume cursor re-runs it (a DONE phase is skipped, stranding the wiring and making the
        # verdict's "re-run to complete it" hollow). A harmless cleanup-only snag leaves finalize DONE.
        self._leave(name, Status.FAILED if self.state.meta.get("config_wiring_failed") else Status.DONE)
        return True

    def _apply_canonical(self, basic: str, resolved: tuple | None = None) -> None:
        """Rewrite the agent's ``agent/MCP_server/mcp_config.yaml`` from live install state.

        This is the Q1="Both" write: after the wizard's own ``install/recipes/mcp_config.setup.yaml``,
        also regenerate the canonical config the agent actually loads, so a fresh clone's
        tools work without any manual path surgery. ``--keep-agent-config`` opts out (the
        setup config still carries the truth). The rewritten canonical holds absolute,
        machine-specific paths — it is per-box, not a commit artifact.

        ``resolved`` reuses an already-computed ``(config, entries)`` so ``_phase_finalize``
        can resolve ONCE post-doctor and drive both this canonical rewrite and the setup-config
        refresh from the same live truth (a1c#1); when ``None`` we resolve here as before."""
        if self.opts.keep_agent_config:
            mcp_resolver.apply_to_canonical({}, keep_agent_config=True, log=self.log)
            self.io.note("kept the agent's MCP_server/mcp_config.yaml (--keep-agent-config)")
            return
        cfg, entries = (
            resolved if resolved is not None else mcp_resolver.resolve_full_config(self.conda, basic, self.specs)
        )
        backup = mcp_resolver.apply_to_canonical(cfg, keep_agent_config=False, log=self.log)
        on = sum(1 for e in entries if e.enabled)
        canon = constants.original_mcp_config()
        self.io.ok(f"regenerated {canon} for THIS machine — {on}/{len(entries)} server(s) enabled")
        if backup is not None:
            self.io.note(f"previous config backed up → {backup}")
        self.io.warn(
            "this config holds absolute, machine-specific paths — do NOT commit it; "
            "re-run `sog-setup` per machine, or pass --keep-agent-config to leave it alone"
        )

    def _rebase_user_layer(self, basic: str) -> None:
        """Rebase ``agent/MCP_server/mcp_config_user.yaml`` onto this box — never fail finalize.

        Most installs have no user-tool overlay (``rebase_user_config`` returns ``None``);
        when one exists (tool-creation playbook, ``sog-setup unpack``), its wiring gets the
        same per-machine truth pass the canonical config just received. WARN-only by design:
        every failure degrades to a warning, and ``--keep-agent-config`` opts this rewrite
        out too — a hand-managed agent config includes its user overlay.
        """
        if self.opts.keep_agent_config:
            return
        try:
            summary = mcp_resolver.rebase_user_config(
                base_python=wiring.base_python_path(basic, conda=self.conda),
                conda=self.conda,
                log=self.log,
            )
        except Exception as exc:
            self.io.warn(f"couldn't rebase the user-tool config ({type(exc).__name__}: {exc}) — harmless")
            return
        if summary is None:
            return  # no user-tool overlay on this box — the common case
        if summary.get("error"):
            self.io.warn(f"user-tool config left untouched: {summary['error']}")
            return
        n = summary.get("servers", 0)
        if not summary.get("changed"):
            self.io.note(f"user-tool config already true on this machine ({n} server(s), unchanged)")
            return
        self.io.ok(f"rebased the user-tool config for THIS machine — {n} server(s) in mcp_config_user.yaml")
        disabled = summary.get("disabled") or []
        if disabled:
            self.io.warn(
                "user server(s) disabled — script missing on this machine: "
                + ", ".join(disabled)
                + " (restore the file under tools_user/ or re-run the tool creation playbook)"
            )

    def _delete_artifacts(self) -> None:
        """Remove ``test/installation/*`` only — guarded so it can never touch a conda
        env or the durable ``.sog_setup/`` state."""
        import shutil

        adir = constants.artifact_dir()
        removed = 0
        skipped = 0
        if adir.exists():
            for child in adir.iterdir():
                try:
                    constants.assert_deletable_artifact(child)  # rejects /opt/conda/**, .sog_setup/**
                except PermissionError:
                    # A stray protected entry under the artifact dir is skipped with a note,
                    # never aborting the whole cleanup (mirrors reset._artifact_children).
                    skipped += 1
                    self.io.note(f"skipped protected entry (not deleted): {child}")
                    continue
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except OSError as exc:
                    # Permission denied / Windows file-in-use / vanished mid-race: skip this one
                    # artifact, keep deleting the rest (mirrors reset._execute's per-child guard).
                    skipped += 1
                    self.io.note(f"could not delete artifact {child.name}: {exc} — skipping")
                    continue
                removed += 1
        msg = f"deleted {removed} test artifact(s) from {adir}"
        if skipped:
            msg += f" ({skipped} protected/undeletable, skipped)"
        self.io.ok(msg)
        self.log.event("artifacts_deleted", count=removed, dir=str(adir))


# --------------------------------------------------------------------------- #
# Small module-level helpers
# --------------------------------------------------------------------------- #
def _coerce_base_mode(value: str | None) -> BaseEnvMode:
    return BaseEnvMode.coerce(value or "new")


def _synthesize_choice(llm_block: dict) -> LLMChoice:
    """Build an (unvalidated) :class:`LLMChoice` from a scripted ``llm:`` block for
    dry runs — no network, no ``.env`` write."""
    from .credentials import DEFAULT_PROVIDER_KEY, default_provider, get_provider
    from .decisions import LLMChoice

    # The answers `llm:` block keys the provider as `provider` (menu id) — accept a
    # `source` (ALLOWED_SOURCES) alias too; get_provider() takes either.
    key = llm_block.get("provider") or llm_block.get("source") or DEFAULT_PROVIDER_KEY
    try:
        prov = get_provider(str(key))
    except KeyError:
        prov = default_provider()
    model = llm_block.get("model") or prov.default_model
    return LLMChoice(source=prov.source, model=model, field_values={}, validated=False)


def _verdict_exit_code(verdict: dict) -> int:
    """Map an honest :meth:`Wizard._final_verdict` result to a process exit code.

    ``ok`` — and the benign "user selected no analysis tools" case (nothing to build, so nothing to
    fail) — succeed with 0. Everything else exits 1: ``degraded`` (a provisioned tool failed its
    mini-data test, a selected tool never built, the agent config wiring did not finish, or a
    real-data agent run (Tier-2) that executed ended FAIL/ERROR/TIMEOUT), ``fail`` (no base env),
    and the non-benign ``incomplete`` where tools WERE selected but none could be provisioned. A
    Tier-2 run that was skipped (no key, no dataset) is not a failure. This is what lets CI tell
    a green install from a degraded one; the human-readable headline is already printed by
    :meth:`Wizard._finish`. Dry-run is handled by the caller (it changes nothing)."""
    level = verdict.get("level")
    if level == "ok":
        return 0
    if level == "incomplete" and not verdict.get("n_selected"):
        return 0
    return 1


def run(opts: WizardOptions, *, io: PromptIO | None = None) -> int:
    """Convenience entry point used by the CLI."""
    try:
        return Wizard(opts, io=io).run()
    except CondaError as exc:
        (io or PromptIO()).err(f"conda error: {exc}")
        return 3
