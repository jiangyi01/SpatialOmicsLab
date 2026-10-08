"""
``sog-setup reset`` — tear a run's *build products* down so setup can start clean.

Reset is the counterpart to the wizard: it removes what a run **built** while
preserving what a run **learned**. By default it removes the per-tool
``<basic>_<server>`` conda envs, clears the ``test/installation/`` artifacts, and
archives the state JSON — but it never references ``.sog_setup/logs/``, so the
run transcripts are preserved *by construction* (there is no code path here that
could delete a log).

There is **no new deletion authority** in this module — every removal reuses the
same guards the wizard already funnels through:

* tool envs   → :func:`constants.assert_deletable_env`      (only ``<basic>_*``; never base/protected)
* the base env → :func:`constants.assert_deletable_base_env` (only via ``--with-base``, separately confirmed)
* artifacts   → :func:`constants.assert_deletable_artifact` (only inside ``test/installation/``)

**Removes only what this run recorded building.** The namespace guard (``<basic>_*``)
is necessary but not sufficient: a user's unrelated env can *share* the base name's
prefix (e.g. base ``sog`` vs. a pre-existing ``sog_myproject``), and blind
prefix-matching would reap it. So reset reconciles the prefix-matched candidates
against the saved state's **build ledger** — every tool's ``target_env``, which the
wizard persists *before* each build starts, so the ledger provably covers even
crash-orphans. A ``<basic>_*`` env with no ledger entry is treated as *collateral*
and **kept** (reported, not removed); ``--prune-unrecorded`` is the explicit opt-in
that removes those look-alikes too.

Two layers of protection sit above even that opt-in. A ``<basic>_*`` env on
:data:`constants.PROTECTED_ENVS` (notably ``sog_reproduce`` under base ``sog``, a real
developer env that falls inside the namespace) is **never** removed — not by the default
sweep and not by ``--prune-unrecorded`` — but it is surfaced in the plan (``protected_envs``)
so the keep is visible rather than silent.

Flags:

======================  ====================================================================
(default)               remove *recorded* tool envs + artifacts, archive state, **keep** base env + logs
``--keep-envs``         keep **all** conda envs (tool + base); still clears artifacts + state
``--with-base``         **also** remove the base env (its own explicit confirmation)
``--prune-unrecorded``  also remove ``<basic>_*`` envs with no build record (look-alikes)
``--basic NAME``        operate on this base-env name instead of the one in saved state
``--dry-run``           print the plan and change nothing (via the conda dry-run gate)
``--yes``               skip the confirmation prompts (CI / the maintainer's test loop)
======================  ====================================================================

``--keep-envs`` and ``--with-base`` are mutually exclusive: one keeps every env,
the other removes even the base — asking for both is contradictory.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import constants
from .envtools import Conda, CondaError
from .prompts import PromptError, PromptIO
from .session_log import DryRunSessionLog, SessionLog
from .state import SetupState, archive_state, new_run_id

if TYPE_CHECKING:
    from pathlib import Path  # annotation-only (see ResetPlan.artifacts / _artifact_children)


# --------------------------------------------------------------------------- #
# Options + plan
# --------------------------------------------------------------------------- #
@dataclass
class ResetOptions:
    basic: str | None = None  # --basic NAME (default: read from saved state)
    keep_envs: bool = False  # --keep-envs : clear artifacts + state, keep every env
    with_base: bool = False  # --with-base : also remove the base env (separately confirmed)
    prune_unrecorded: bool = False  # --prune-unrecorded : also remove <basic>_* look-alikes with no build record
    dry_run: bool = False  # --dry-run   : print the plan, change nothing
    assume_yes: bool = False  # --yes       : skip confirmation prompts


@dataclass
class ResetPlan:
    """What a reset *would* remove — computed once, guarded, then executed."""

    basic_env: str
    tool_envs: list[str] = field(default_factory=list)
    base_env: str | None = None  # set only under --with-base and after the base-env guard
    artifacts: list[Path] = field(default_factory=list)
    # <basic>_* envs that matched the namespace but have NO build-ledger entry — kept as
    # likely collateral (an unrelated project's env sharing the prefix), reported not removed.
    unrecorded_envs: list[str] = field(default_factory=list)
    # <basic>_* envs on the PROTECTED_ENVS list (e.g. ``sog_reproduce`` under base ``sog``) —
    # never removed, NOT EVEN under ``--prune-unrecorded``. Surfaced (not silently dropped) so
    # the user can see the guard kept a real env that merely shares the prefix.
    protected_envs: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.tool_envs and self.base_env is None and not self.artifacts


# --------------------------------------------------------------------------- #
# Discovery + planning
# --------------------------------------------------------------------------- #
def _archive_order(path: Path) -> str:
    """Sort key for an archived state file: the ``new_run_id()`` stamp in its name (time-ordered)."""
    stem = path.name[len("setup_state.") : -len(".json")]
    return stem.removeprefix("unreadable-")


def _archived_states() -> list[dict]:
    """Every archived state document under ``archive_dir()``, newest first, as raw JSON.

    ``archive_state`` moves a run's ledger here on ``--restart``, when a completed run is re-launched
    (wizard S1), and on every reset -- including one that could not remove everything. Read raw and
    quietly: a file that does not parse, or a newer wizard's shape, contributes what it can and
    never a notice."""
    out: list[dict] = []
    try:
        files = sorted(constants.archive_dir().glob("setup_state.*.json"), key=_archive_order, reverse=True)
    except OSError:
        return out
    for p in files:
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # UnicodeDecodeError is a ValueError
            continue
        if isinstance(raw, dict):
            out.append(raw)
    return out


def _ledger_of(raw: dict, basic: str) -> set[str]:
    """The ``target_env`` names one raw state document recorded, when it belongs to ``basic``."""
    tools = raw.get("tools")
    if raw.get("basic_env_name") != basic or not isinstance(tools, dict):
        return set()
    return {t for rec in tools.values() if isinstance(rec, dict) and isinstance(t := rec.get("target_env"), str) and t}


def _live_basic() -> str | None:
    st = SetupState.load()
    return st.basic_env_name if st is not None and st.basic_env_name else None


def _newest_archived_basic() -> str | None:
    """The base env of the newest archived run that recorded one."""
    for raw in _archived_states():
        basic = raw.get("basic_env_name")
        if isinstance(basic, str) and basic.strip():
            return basic
    return None


def _resolve_basic_env(explicit: str | None) -> tuple[str | None, str]:
    """The base-env name to reset and where it came from: an explicit ``--basic`` wins
    (``"explicit"``), else saved state (``"live"``), else the newest archived ledger (``"archive"``);
    ``(None, "")`` when none names one.

    The archive fallback (hunt 2026-09-30, u35b-setup-state-1/3): a reset that could not remove an env
    still archives the state, and so does re-launching a completed run, so the live file is gone or
    names no base. Without it the "re-run ``sog-setup reset`` to retry" advice found nothing to reset.
    The one copy of this chain: ``run_reset`` used to repeat it inline to learn the origin
    (hunt 2026-09-30, rc-setup-state reset minor)."""
    if explicit:
        return explicit, "explicit"
    if live := _live_basic():
        return live, "live"
    if archived := _newest_archived_basic():
        return archived, "archive"
    return None, ""


def _discover_basic_env(explicit: str | None) -> str | None:
    """The base-env name to reset -- see :func:`_resolve_basic_env`."""
    return _resolve_basic_env(explicit)[0]


def _recorded_tool_envs(basic: str | None) -> set[str]:
    """The envs a saved run recorded creating: every tool's ``target_env``.

    The wizard writes ``target_env`` the instant a build starts (before ``conda
    create``), so this ledger is a superset of every env the run physically made —
    including crash-orphans. A state document is trusted only when it belongs to
    ``basic``: an explicit ``--basic`` that names a *different* run borrows nothing from
    it. Empty → every ``<basic>_*`` env is treated as unrecorded, so reset never assumes
    ownership of a look-alike env it has no record of building.

    The live state AND every archived one are read (hunt 2026-09-30, u35b-setup-state-1/3). The
    wizard archives a completed run's ledger before its next run starts "so a subsequent ledger-aware
    reset can find them" (S1), and reset archives the ledger even when a removal failed, but this
    read only the live file -- so a re-run reset kept every env it had built as "likely an unrelated
    env" and printed "reset complete.".
    """
    if not basic:
        return set()
    recorded: set[str] = set()
    st = SetupState.load()
    if st is not None and st.basic_env_name == basic:
        recorded |= {t for rec in st.tools.values() if (t := rec.get("target_env"))}
    for raw in _archived_states():
        recorded |= _ledger_of(raw, basic)
    return recorded


def _foreign_state_basic(basic: str) -> str | None:
    """The base env the live state file belongs to, when that is a different one than ``basic``.

    (hunt 2026-09-30, u35b-setup-state-2) ``reset --basic X`` used to archive whatever state was live,
    so an interrupted install of base Y lost its resume point and its ledger to a reset of X. A
    state with no base yet, or one that cannot be read, is not claimed by anyone and is archived."""
    try:
        raw = json.loads(constants.state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    other = raw.get("basic_env_name") if isinstance(raw, dict) else None
    return other if isinstance(other, str) and other and other != basic else None


def _artifact_children() -> list[Path]:
    """Guarded list of ``test/installation/*`` entries (skips anything the guard rejects)."""
    adir = constants.artifact_dir()
    out: list[Path] = []
    if adir.exists():
        for child in sorted(adir.iterdir()):
            try:
                constants.assert_deletable_artifact(child)
            except PermissionError:
                continue  # defense-in-depth: never delete outside the artifact dir
            out.append(child)
    return out


def build_plan(
    conda: Conda,
    basic_env: str,
    *,
    keep_envs: bool,
    with_base: bool,
    recorded: set[str] | frozenset[str] = frozenset(),
    prune_unrecorded: bool = False,
) -> ResetPlan:
    """Enumerate exactly what to remove, guarding every candidate up front.

    One ``env_list`` pass classifies conda envs. A ``<basic>_*`` tool env is collected
    for removal only when it is **recorded** in this run's build ledger (``recorded``);
    a namespace look-alike with no ledger entry is set aside in ``unrecorded_envs`` and
    kept (unless ``prune_unrecorded`` folds it back into the removal set). The base env
    is collected only under ``--with-base`` and only if it clears
    :func:`assert_deletable_base_env`. Artifacts are always collected (a reset always
    clears the per-run test dir). ``recorded`` defaults to empty so the *safe* outcome —
    keep anything we can't prove we built — is the default for any caller.
    """
    tool_envs: list[str] = []
    unrecorded: list[str] = []
    protected: list[str] = []
    base: str | None = None
    if not keep_envs:
        names = conda.env_list()
        for name in names:
            if name == basic_env:
                continue  # the base env is handled separately (with_base) below
            if not constants.is_managed_env(basic_env, name):
                continue  # not ours — never touch it
            try:
                constants.assert_deletable_env(basic_env, name)
            except PermissionError:
                # The base env and out-of-namespace names are already filtered above, so the
                # ONLY way the guard refuses a managed <basic>_* env here is that it is on the
                # PROTECTED_ENVS list. Surface it (kept, never removed — not even under
                # --prune-unrecorded) instead of dropping it silently.
                if name in constants.PROTECTED_ENVS:
                    protected.append(name)
                continue
            if name in recorded or prune_unrecorded:
                tool_envs.append(name)  # we built it (or --prune-unrecorded) → remove
            else:
                unrecorded.append(name)  # namespace match but no build record → keep as collateral
        if with_base and basic_env in names:
            try:
                constants.assert_deletable_base_env(basic_env, basic_env)
                base = basic_env
            except PermissionError:
                base = None  # protected base (e.g. spatialomicsgym_env) — never remove
    return ResetPlan(
        basic_env=basic_env,
        tool_envs=sorted(tool_envs),
        base_env=base,
        artifacts=_artifact_children(),
        unrecorded_envs=sorted(unrecorded),
        protected_envs=sorted(protected),
    )


# --------------------------------------------------------------------------- #
# Presentation + execution
# --------------------------------------------------------------------------- #
def _print_plan(io: PromptIO, plan: ResetPlan, *, keep_envs: bool, foreign_state: str | None = None) -> None:
    if keep_envs:
        io.say("  • conda envs: KEEPING all (--keep-envs)")
    elif plan.tool_envs or plan.base_env:
        for name in plan.tool_envs:
            io.say(f"  • remove tool env   {name}")
        if plan.base_env:
            io.say(f"  • remove BASE env   {plan.base_env}   ⚠️  (rebuilt from scratch next run)")
    elif plan.unrecorded_envs or plan.protected_envs:
        # Not "none in the namespace" directly above a list of <basic>_* envs being kept (hunt 2026-09-30,
        # u35b-setup-state-3): there are some, this setup just has no record of building them.
        io.say(f"  • conda envs: none recorded as built by this setup under '{plan.basic_env}_*'")
    else:
        io.say(f"  • conda envs: none in the '{plan.basic_env}_*' namespace")
    if plan.unrecorded_envs and not keep_envs:
        io.warn(
            f"  • KEEPING {len(plan.unrecorded_envs)} env(s) that match '{plan.basic_env}_*' "
            "but were not recorded as built by this setup:"
        )
        for name in plan.unrecorded_envs:
            io.say(f"        - {name}   (likely an unrelated env sharing the prefix)")
        io.say("      → re-run with --prune-unrecorded to remove these too")
    if plan.protected_envs and not keep_envs:
        io.note(
            f"  • KEEPING {len(plan.protected_envs)} protected env(s) that share the "
            f"'{plan.basic_env}_*' prefix — never removed, even with --prune-unrecorded:"
        )
        for name in plan.protected_envs:
            io.say(f"        - {name}   (protected)")
    io.say(f"  • delete {len(plan.artifacts)} test artifact(s) under {constants.artifact_dir()}")
    if foreign_state:
        io.say(
            f"  • KEEP the setup state JSON — it belongs to base env '{foreign_state}'"
            f" (reset that one with --basic {foreign_state})"
        )
    else:
        io.say("  • archive the setup state JSON   (logs under .sog_setup/logs/ are PRESERVED)")


def _res_reason(res) -> str:
    """A short one-line reason from a non-ok ``remove_env`` RunResult for the user + log."""
    # A whitespace-only stderr/stdout (conda has emitted a bare "\n" on some fast refusals) ``.strip()``s
    # to "" whose ``.splitlines()`` is ``[]`` — indexing ``[0]`` there raises IndexError, turning a
    # benign "couldn't remove" note into a crash. Compute the line list first and fall back to the
    # returncode line when it's empty; mirrors envtools' slice-safe first-line extraction.
    lines = (res.stderr or res.stdout or "").strip().splitlines()
    return (lines[0] if lines else f"conda exited {res.returncode}")[:200]


def _execute(conda: Conda, plan: ResetPlan, *, io: PromptIO, log: SessionLog, dry_run: bool) -> int:
    """Perform the plan. Conda removes ride the envtools dry-run gate; the
    non-conda steps (artifacts, state archive) guard ``dry_run`` explicitly.

    Returns the number of items that could NOT be removed (0 = fully clean). ``remove_env`` runs
    with ``check=False``, so a removal that FAILS (env busy/locked, permission denied, or the
    belt-and-braces PROTECTED refusal) returns a **non-ok RunResult without raising** — the old code
    discarded that result and printed "removed" + logged ``reset_env_removed`` unconditionally, a
    false-green that told the user an env was gone when it was not (F1). Now each removal is checked
    like the artifact loop already is, and the failure count flows up so the caller can report an
    honest verdict + a non-zero exit instead of a green "reset complete." Under ``--dry-run`` every
    ``remove_env`` returns an ok dry-run RunResult, so the count stays 0 and the preview stays green."""
    failed = 0
    for name in plan.tool_envs:
        constants.assert_deletable_env(
            plan.basic_env, name
        )  # re-assert at the very edge (namespace guard: MUST stay outside the try below)
        try:
            res = conda.remove_env(name)  # no-op under dry_run (gated inside envtools → ok dry-run RunResult)
        except CondaError as exc:
            # envtools._exec raises CondaError even at check=False when the removal hits a
            # TimeoutExpired/OSError — the "env busy/locked" teardown that hangs to the removal timeout,
            # the very case the docstring above cites. The docstring promises a failed removal is
            # counted and skipped, not raised; honor that here too so the remaining tool envs, the base
            # env, artifact cleanup, AND the state archive still run, instead of the CondaError unwinding
            # to main (exit 3) and leaving a half-done, un-archived reset. The env-removal message
            # carries no credential (mirrors _res_reason's already-unredacted stderr). (R27 F1)
            failed += 1
            io.warn(f"could not remove tool env {name}: {str(exc)[:200]} — skipping (re-run reset to retry)")
            log.event("reset_env_remove_failed", env=name, error=str(exc)[:200])
            continue
        if res.ok:
            io.ok(f"{'[dry-run] would remove' if dry_run else 'removed'} tool env {name}")
            log.event("reset_env_removed", env=name, dry_run=dry_run)
        else:
            failed += 1
            io.warn(f"could not remove tool env {name}: {_res_reason(res)} — skipping (re-run reset to retry)")
            log.event("reset_env_remove_failed", env=name, error=_res_reason(res))

    if plan.base_env:
        constants.assert_deletable_base_env(
            plan.basic_env, plan.base_env
        )  # re-assert (namespace guard: stays outside the try)
        try:
            res = conda.remove_env(plan.base_env)
            base_ok, base_reason = res.ok, _res_reason(res)
        except CondaError as exc:
            # same timeout/OSError → CondaError degradation as the tool-env loop above (R27 F1): count
            # it and fall through to the artifact + state-archive steps rather than aborting the reset.
            base_ok, base_reason = False, str(exc)[:200]
        if base_ok:
            io.ok(f"{'[dry-run] would remove' if dry_run else 'removed'} BASE env {plan.base_env}")
            log.event("reset_base_env_removed", env=plan.base_env, dry_run=dry_run)
        else:
            failed += 1
            io.warn(f"could not remove BASE env {plan.base_env}: {base_reason} — skipping (re-run reset to retry)")
            log.event("reset_base_env_remove_failed", env=plan.base_env, error=base_reason)

    # Artifacts — a plain rmtree, not a conda op, so dry-run must be guarded here.
    removed = 0
    artifact_failed = 0
    for child in plan.artifacts:
        constants.assert_deletable_artifact(child)  # re-assert: rejects /opt/conda/**, .sog_setup/**
        if dry_run:
            continue
        # Guard each delete against OSError (permission denied, Windows file-in-use, a broken
        # symlink, or a race where the artifact vanished between planning and now). Mirrors
        # ``_artifact_children``'s skip-on-guard-failure: one un-deletable artifact must not abort
        # the whole reset — the remaining artifacts AND the state-archive step below must still run,
        # instead of an OSError unwinding to ``main`` (which catches only CondaError/PromptError) as a
        # raw traceback + a half-done reset.
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as exc:
            artifact_failed += 1
            io.warn(f"could not delete artifact {child.name}: {exc} — skipping (re-run reset to retry)")
            log.event("reset_artifact_delete_failed", path=str(child), error=str(exc))
            continue
        removed += 1
    if dry_run:
        io.note(f"[dry-run] would delete {len(plan.artifacts)} test artifact(s)")
    else:
        io.ok(
            f"deleted {removed} test artifact(s)"
            + (f" ({artifact_failed} could not be removed)" if artifact_failed else "")
        )
        log.event("reset_artifacts_deleted", count=removed, failed=artifact_failed)
    failed += artifact_failed  # fold artifact-delete failures into the verdict alongside env-removal ones

    # State — archive (never unlink), so a reset is itself recoverable. Logs are
    # untouched by never referencing logs_dir() anywhere in this module. A state that belongs to a
    # different base env is not this reset's to archive (hunt 2026-09-30, u35b-setup-state-2).
    foreign = _foreign_state_basic(plan.basic_env)
    if foreign is not None:
        io.note(f"kept the setup state — it belongs to base env '{foreign}', not '{plan.basic_env}'")
        log.event("reset_state_kept", owner=foreign, dry_run=dry_run)
    elif dry_run:
        io.note("[dry-run] would archive the setup state (logs preserved)")
    else:
        arch = archive_state()
        if arch is not None:
            io.ok(f"archived setup state → {arch.name}  (logs preserved)")
            log.event("reset_state_archived", path=str(arch))
        else:
            io.note("no setup state file to archive")

    return failed


def run_reset(opts: ResetOptions, *, io: PromptIO | None = None, conda: Conda | None = None) -> int:
    """Discover → plan → confirm → execute. Returns a process exit code.

    ``io``/``conda`` are injectable test seams (mirroring the wizard); in normal use
    both are built here and the real conda rides the dry-run gate."""
    io = io or PromptIO()
    basic, origin = _resolve_basic_env(opts.basic)
    if not basic:
        io.warn("no saved setup state and no --basic NAME given — nothing to reset.")
        return 0
    if origin == "archive":
        io.note(f"no live setup state names a base env — using '{basic}' from the newest archived run ledger")

    # --dry-run is a read-only preview → ZERO side effects (mirrors the wizard's DryRunSessionLog
    # contract, R20): create no `.sog_setup` dir and write no reset-*.jsonl transcript — the help text
    # promises "change nothing". A real reset ensures the state dirs with prune=False and appends its
    # own reset-*.jsonl. Reset NEVER prunes the wizard's run logs — both the module contract (see the
    # module docstring: "no code path here that could delete a log") AND, crucially, because this runs
    # BEFORE the confirmation gate and the "nothing to remove" early-returns below: a prune=True here
    # size-caps (deletes oldest) run transcripts even on a reset the user ABORTS or one with nothing to
    # do — the exact opposite of "reset aborted — nothing changed". Log size-capping stays the wizard's
    # job (its own ensure_state_dirs default prunes at each run start).
    log: SessionLog
    if opts.dry_run:
        log = DryRunSessionLog(constants.logs_dir() / f"reset-{new_run_id()}.jsonl")
    else:
        constants.ensure_state_dirs(prune=False)
        log = SessionLog(constants.logs_dir() / f"reset-{new_run_id()}.jsonl")
    io.log = log
    if conda is None:
        conda = Conda(dry_run=opts.dry_run, log=log)
    else:
        conda.dry_run = opts.dry_run  # run_reset owns the dry-run contract, however conda arrived

    recorded = _recorded_tool_envs(basic)
    plan = build_plan(
        conda,
        basic,
        keep_envs=opts.keep_envs,
        with_base=opts.with_base,
        recorded=recorded,
        prune_unrecorded=opts.prune_unrecorded,
    )
    foreign_state = _foreign_state_basic(basic)
    state_exists = constants.state_file().exists() and foreign_state is None
    if plan.empty and not plan.unrecorded_envs and not state_exists:
        io.ok(f"'{basic}' is already clean — no envs, artifacts, or state to reset.")
        return 0

    io.section(f"Reset setup for base env '{basic}'")
    _print_plan(io, plan, keep_envs=opts.keep_envs, foreign_state=foreign_state)

    if plan.empty and not state_exists:
        # Only unrecorded look-alikes remain — nothing this run is authorized to remove.
        io.note("nothing to remove — the envs above were not built by this setup (use --prune-unrecorded to force).")
        return 0

    if not opts.dry_run and not opts.assume_yes:
        if not io.ask_yesno(f"Proceed — remove the above and archive state for '{basic}'?", default=False):
            io.note("reset aborted — nothing changed.")
            return 1
        # Base removal is a SEPARATE, explicit confirmation — never bundled into the
        # main gate (per the plan: base teardown is never automatic).
        if plan.base_env is not None and not io.ask_yesno(
            f"⚠️  ALSO remove the BASE env '{basic}'? It must be rebuilt from scratch next run.",
            default=False,
        ):
            io.note(f"keeping base env '{basic}' (removing tool envs + artifacts only).")
            plan.base_env = None

    n_failed = _execute(conda, plan, io=io, log=log, dry_run=opts.dry_run)
    log.event(
        "reset_complete",
        basic_env=basic,
        dry_run=opts.dry_run,
        tool_envs=plan.tool_envs,
        base_removed=plan.base_env is not None,
        failed=n_failed,
    )
    if opts.dry_run:
        io.ok("dry-run complete — nothing was changed.")
        return 0
    if n_failed:
        # Honest verdict: at least one env/artifact could not be removed. Never report a green
        # "reset complete." over a partial teardown — a non-zero exit lets a script/CI tell the two
        # apart, and the "re-run to retry" hint is real because the survivors are still on disk and
        # the archived ledger still names them (hunt 2026-09-30, u35b-setup-state-1). It names the
        # base env, because a plain re-run would follow whatever state is live by then.
        io.warn(
            f"reset finished, but {n_failed} item(s) could not be removed — "
            f"re-run `sog-setup reset --basic {basic}` to retry."
        )
        return 1
    io.ok("reset complete.")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
_EPILOG = (
    "examples:\n"
    "  sog-setup reset                    remove *recorded* tool envs + test artifacts, keep base + logs\n"
    "  sog-setup reset --keep-envs        clear only artifacts + state; keep every conda env\n"
    "  sog-setup reset --with-base        also remove the base env (asked again, on its own)\n"
    "  sog-setup reset --prune-unrecorded also remove '<basic>_*' envs with no build record\n"
    "  sog-setup reset --dry-run          preview the plan; change nothing\n"
)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="sog-setup reset",
        description=(
            "Remove a setup run's build products (per-tool conda envs + test artifacts) and "
            "archive its state so setup can start clean. Logs under .sog_setup/logs/ are always "
            "kept; the base env and every protected env are never removed unless you pass --with-base."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--basic", metavar="NAME", help="base-env name to reset (default: read from saved state)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--keep-envs", action="store_true", help="keep ALL conda envs; clear only artifacts + state")
    grp.add_argument("--with-base", action="store_true", help="ALSO remove the base env (asked to confirm separately)")
    ap.add_argument(
        "--prune-unrecorded",
        action="store_true",
        help="also remove '<basic>_*' envs with no build record (look-alikes are kept by default)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and change nothing")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompts (CI / unattended)")
    args = ap.parse_args(argv)

    opts = ResetOptions(
        basic=args.basic,
        keep_envs=args.keep_envs,
        with_base=args.with_base,
        prune_unrecorded=args.prune_unrecorded,
        dry_run=args.dry_run,
        assume_yes=args.yes,
    )
    try:
        return run_reset(opts)
    except CondaError as exc:
        PromptIO(stream=sys.stderr).err(f"conda error: {exc}")
        return 3
    except PromptError as exc:
        print(f"❌ needed input that wasn't available: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        # Defense in depth: the artifact loop already skips per-file OSErrors, but a filesystem
        # failure outside it (state-dir creation, the state archive) must still exit cleanly with a
        # friendly line, never a raw traceback.
        PromptIO(stream=sys.stderr).err(f"filesystem error during reset: {exc}")
        return 3


if __name__ == "__main__":
    # L2: route through the cli friendly wrapper so a direct ``python -m sog_install.reset``
    # also gets KeyboardInterrupt→130 and a generic-error→friendly-1, on top of ``reset.main``'s own
    # CondaError/PromptError/OSError handling — matching ``sog-setup reset`` exactly.
    from .cli import main as _cli_main

    raise SystemExit(_cli_main(["reset", *sys.argv[1:]]))
