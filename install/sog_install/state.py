"""
Durable setup state + atomic persistence + resume logic.

State is rewritten atomically after every transition to
``.sog_setup/setup_state.json`` (outside ``test/installation/`` so cleanup can
never destroy it). On the next launch the wizard reloads it, detects any phase
left ``in_progress`` by a crash, and computes a resume cursor.

**No secrets are ever stored here** — only the provider source, model name, and
a validated flag. Keys live solely in ``.env`` (and transiently in memory).

Stdlib only.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from . import constants
from .session_log import _sanitize, notice

SCHEMA_VERSION = 1

# Ordered phases (index == the number shown to the user).
PHASES: tuple[str, ...] = (
    "preflight",  # 0  Stage A
    "onboarding",  # 1  Stage A
    "category_select",  # 2  Stage B
    "base_env",  # 3  Stage B
    "provision",  # 4  Stage B
    "test",  # 5  Stage B
    "finalize",  # 6  Stage B
    "demo",  # 7  Stage B — SOFT (post-install demo + real-run offer; never gates completion)
)

# Soft phases are optional and never counted toward "did the install finish?": a run whose
# every REQUIRED phase is done/skipped is complete even if the soft phase never ran, and the
# resume cursor skips them. This keeps the soft ``demo`` phase from (a) making an older state
# file — written before ``demo`` existed, so ``demo`` defaults to ``pending`` — read as an
# unfinished run that must be resumed, and (b) failing/blocking a run that just declined the demo.
SOFT_PHASES: frozenset[str] = frozenset({"demo"})


class Status(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


# A phase whose work is behind us for cursor purposes.
_TERMINAL_FORWARD = {Status.DONE.value, Status.SKIPPED.value}


def new_run_id() -> str:
    """A unique, sortable run id: ``<timestamp>-<rand6>``."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically; verify parse-back before committing (clone of the
    proven ``tools_user/memory_manager._atomic_write_json``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            # Durability (#95c): force the bytes to disk BEFORE the atomic rename, so a crash/power-loss
            # right after os.replace can't leave a rename pointing at an unflushed (zero-length/torn) file
            # — the exact way a resume-state or key vault silently loses data on a deploy box. fsync can be
            # unsupported on exotic filesystems (tmpfs/some NFS), so it degrades to best-effort there.
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        with open(tmp_path, encoding="utf-8") as f:
            json.load(f)  # parse-back verification before replace
        os.replace(tmp, str(path))
    except BaseException:
        # BaseException (not just Exception) so a Ctrl-C landing mid-write still removes the temp file
        # (a515#2). The state/vault temps are harmless cruft, but this same atomic-write shape backs the
        # secret-bearing ``.env`` writer; leaking a ``.env.<rand>`` on interrupt is a secret-hygiene wart.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass
class SetupState:
    """The complete, resumable state of one setup run."""

    run_id: str = field(default_factory=new_run_id)
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # Phase → Status.value
    phases: dict[str, str] = field(default_factory=lambda: dict.fromkeys(PHASES, Status.PENDING.value))

    # Stage-B selections
    basic_env_name: str | None = None
    base_env_mode: str | None = None  # BaseEnvMode.value
    selected_categories: list[str] = field(default_factory=list)
    selected_servers: list[str] = field(default_factory=list)

    # LLM config — NO SECRETS (source/model/validated only)
    llm: dict = field(default_factory=dict)

    # Per-tool records: server_key → {provision_status, test_status, strategy,
    # target_env, self_review_rounds, error, timestamps, ...}
    tools: dict[str, dict] = field(default_factory=dict)

    # Run policies + free-form metadata
    policies: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # The file this instance was loaded from or last saved to — not state, never serialized. A first
    # save to any other path checks what it is about to overwrite (see :func:`_set_aside_unreadable`).
    _known_target: Path | None = field(default=None, init=False, repr=False, compare=False)

    # -- phase transitions ----------------------------------------------------
    def set_phase(self, name: str, status: Status | str) -> None:
        if name not in self.phases:
            raise KeyError(f"unknown phase {name!r}")
        self.phases[name] = Status(status).value  # accepts a Status or its str value
        self.touch()

    def phase_status(self, name: str) -> str:
        return self.phases[name]

    def touch(self) -> None:
        self.updated_at = datetime.now().isoformat(timespec="seconds")

    # -- per-tool records -----------------------------------------------------
    def tool(self, server_key: str) -> dict:
        return self.tools.setdefault(
            server_key, {"provision_status": Status.PENDING.value, "test_status": Status.PENDING.value}
        )

    def set_tool(self, server_key: str, **fields) -> dict:
        rec = self.tool(server_key)
        rec.update(fields)
        self.touch()
        return rec

    # -- resume logic ---------------------------------------------------------
    def has_progress(self) -> bool:
        """True if any phase moved off ``pending`` — i.e. a prior run exists."""
        return any(s != Status.PENDING.value for s in self.phases.values())

    def is_complete(self) -> bool:
        # Only REQUIRED phases decide completion — a declined/never-run soft ``demo`` phase must
        # not make a finished install read as incomplete (and so wrongly resumable next launch).
        if not all(v in _TERMINAL_FORWARD for k, v in self.phases.items() if k not in SOFT_PHASES):
            return False
        # …but a run whose provision phase reached DONE while a *selected* tool's build FAILED is
        # NOT complete: the next launch must KEEP this state so the B2 retry rebuilds only the
        # failures (wizard ``_has_unbuilt_selected``/``_stale_completion``). Without this guard,
        # ``is_complete() → True`` makes ``_load_or_start_state`` discard the state (a fresh
        # ``SetupState()``), the failed tool is forgotten, and the on-screen "re-run ``sog-setup``
        # to retry" becomes a lie. Mirror the verdict's ``build_failed`` predicate exactly
        # (``provision_status == FAILED`` over ``selected_servers``) — keying on FAILED (not
        # "non-terminal") keeps a genuinely-green run fresh-on-rerun and never loops on a
        # no-spec / no-record server.
        return not any(
            self.tools.get(k, {}).get("provision_status") == Status.FAILED.value for k in self.selected_servers
        )

    def interrupted_phases(self) -> list[str]:
        """Phases left ``in_progress`` — the fingerprint of a crash/kill."""
        return [p for p in PHASES if self.phases.get(p) == Status.IN_PROGRESS.value]

    def reset_interrupted(self) -> list[str]:
        """Reset any ``in_progress`` phase to ``pending`` so it re-runs cleanly.
        Returns the phases that were reset (their orphan envs must be removed
        with ``force=True`` before re-provisioning — see :meth:`orphan_tool_envs`)."""
        reset = self.interrupted_phases()
        for p in reset:
            self.phases[p] = Status.PENDING.value
        if reset:
            self.touch()
        return reset

    def orphan_tool_envs(self) -> list[str]:
        """Target envs a crash left mid-provision (``provision_status`` ``in_progress``).

        The wizard marks a tool ``in_progress`` the instant its build starts and flips
        it to a terminal status the instant it finishes; so any tool still
        ``in_progress`` in a *loaded* state is one whose build was interrupted and whose
        ``<basic>_<server>`` env may be a partial build. These must be removed (guarded)
        before re-provisioning — the concrete cleanup :meth:`reset_interrupted` promises.
        """
        return [
            rec["target_env"]
            for rec in self.tools.values()
            if rec.get("provision_status") == Status.IN_PROGRESS.value and rec.get("target_env")
        ]

    def resume_cursor(self) -> str | None:
        """First REQUIRED phase whose work is not behind us (pending/failed/in_progress).
        ``None`` when every required phase is done or skipped. Soft phases (``demo``) are never
        a resume target — they run unconditionally after the required pipeline finishes."""
        for p in PHASES:
            if p in SOFT_PHASES:
                continue
            if self.phases.get(p) not in _TERMINAL_FORWARD:
                return p
        return None

    # -- (de)serialization ----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phases": dict(self.phases),
            "basic_env_name": self.basic_env_name,
            "base_env_mode": self.base_env_mode,
            "selected_categories": list(self.selected_categories),
            "selected_servers": list(self.selected_servers),
            "llm": dict(self.llm),
            "tools": {k: dict(v) for k, v in self.tools.items()},
            "policies": dict(self.policies),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SetupState:
        st = cls(
            run_id=d.get("run_id", new_run_id()),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            created_at=d.get("created_at", datetime.now().isoformat(timespec="seconds")),
            updated_at=d.get("updated_at", datetime.now().isoformat(timespec="seconds")),
            basic_env_name=d.get("basic_env_name"),
            base_env_mode=d.get("base_env_mode"),
            selected_categories=list(d.get("selected_categories", [])),
            selected_servers=list(d.get("selected_servers", [])),
            llm=dict(d.get("llm", {})),
            tools={k: dict(v) for k, v in d.get("tools", {}).items()},
            policies=dict(d.get("policies", {})),
            meta=dict(d.get("meta", {})),
        )
        # Merge persisted phase statuses over the default (tolerant of new phases).
        phases = dict.fromkeys(PHASES, Status.PENDING.value)
        phases.update({k: v for k, v in d.get("phases", {}).items() if k in phases})
        st.phases = phases
        return st

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path | None = None) -> None:
        self.touch()
        target = path or constants.state_file()
        if self._known_target != target:
            _set_aside_unreadable(target)
            self._known_target = target
        # Defense-in-depth: the state file is the one persisted sink that does NOT flow through
        # SessionLog.event's redaction. _sanitize deep-redacts every string value (and masks any
        # secret-ish key) so a registered credential that reached a tool's messages/self_review/repairs
        # via a raw provider (before its per-site redact-before-clip) is masked in state.json too. This
        # MUST live here, not in _atomic_write_json — key_library reuses that shared writer for the
        # vault of REAL keys, which must round-trip verbatim.
        _atomic_write_json(target, _sanitize(self.to_dict()))

    @classmethod
    def load(cls, path: Path | None = None) -> SetupState | None:
        p = path or constants.state_file()
        if not p.exists():
            return None  # no prior run — normal, stay silent
        try:
            st = cls.from_dict(_read_state_dict(p))
        except _NewerSchemaError as exc:
            # Written by a newer wizard — its shape may differ; start fresh rather than
            # misread it. (Older/equal versions are merged tolerantly by from_dict.)
            notice(
                f"saved setup progress is from a newer version (v{exc.version} > v{SCHEMA_VERSION}); starting fresh."
            )
            return None
        except Exception as exc:
            # Corrupt / truncated / schema-skewed state must never brick a launch: degrade to a
            # fresh run (the phases just re-run; they're idempotent) with a plain-language notice.
            notice(f"could not read previous setup progress ({exc}); starting fresh.")
            return None
        st._known_target = p
        return st


class _NewerSchemaError(ValueError):
    def __init__(self, version: int) -> None:
        super().__init__(f"schema v{version} is newer than v{SCHEMA_VERSION}")
        self.version = version


def _read_state_dict(p: Path) -> dict:
    """The raw state document at ``p``, or an exception saying why this version cannot use it."""
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("state file is not a JSON object")
    ver = int(raw.get("schema_version", SCHEMA_VERSION))
    if ver > SCHEMA_VERSION:
        raise _NewerSchemaError(ver)
    return raw


def _set_aside_unreadable(target: Path) -> Path | None:
    """Move a state file this version cannot read out of the way before a save replaces it.

    (hunt 2026-09-30, u35b-setup-state-14) ``load`` reports a newer-schema or corrupt file as "no
    prior run", and the wizard's next ``_persist()`` wrote a fresh state over it -- destroying the
    newer wizard's resume point and the per-tool ``target_env`` ledger reset relies on, where every
    other fresh start (``--restart``, a completed run) archives first. Done here at the overwrite and
    not in ``load``, because doctor, conncheck and ``sog-setup chat`` also load the state and must not
    move a newer wizard's file just by reading it. The live file goes to ``archive_dir()``, where
    reset reads ledgers; any other path is set aside next to itself."""
    if not target.exists():
        return None
    try:
        SetupState.from_dict(_read_state_dict(target))
        return None  # readable — the caller is replacing it on purpose
    except Exception:
        pass
    if target == constants.state_file():
        constants.ensure_state_dirs(prune=False)  # prune=False for the reason archive_state gives
        dst = constants.archive_dir() / f"setup_state.unreadable-{new_run_id()}.json"
    else:
        dst = target.with_name(f"{target.stem}.unreadable-{new_run_id()}{target.suffix}")
    try:
        shutil.move(str(target), str(dst))
    except FileNotFoundError:
        return None  # someone else moved it first
    except OSError as exc:
        notice(f"could not set aside the unreadable setup progress {target.name} ({exc}); it will be replaced.")
        return None
    notice(f"kept the previous setup progress this version cannot read → {dst}")
    return dst


def archive_state(path: Path | None = None) -> Path | None:
    """Move the current state file into the archive dir (for ``--restart``).
    Returns the archive path, or ``None`` if there was nothing to archive."""
    src = path or constants.state_file()
    if not src.exists():
        return None
    # prune=False: this call exists ONLY to guarantee the archive dir is present before the move —
    # size-capping run logs is not archival's job. With the default prune=True, `sog-setup reset`
    # (whose _execute calls us) would delete the oldest run transcripts on every executed reset,
    # breaking reset's contract to "preserve what a run LEARNED" and its module-docstring promise
    # that "there is no code path here that could delete a log". The wizard's run-start
    # `ensure_state_dirs()` (wizard.py) remains the sole designated log size-cap point, so no
    # hygiene is lost — an over-grown log dir is still capped the next time the wizard runs. (R23-C-F1b)
    constants.ensure_state_dirs(prune=False)
    # #95a: a bare %Y%m%d-%H%M%S stamp collides at SECOND granularity — two archives in the same second
    # (a double `--restart`, or a scripted run) resolve to the same path and shutil.move OVERWRITES the
    # first, silently losing that state history. new_run_id() appends a random suffix to the same sortable
    # timestamp, so concurrent/rapid archives get distinct filenames while staying time-ordered.
    dst = constants.archive_dir() / f"setup_state.{new_run_id()}.json"
    try:
        shutil.move(str(src), str(dst))
    except FileNotFoundError:
        # TOCTOU (a515#4): the src vanished between the exists() check above and the move — a
        # concurrent ``--restart``/reset (or a shared NFS home) already archived it. Nothing left to
        # do, and no state history is lost (someone else moved it), so degrade to "nothing archived"
        # rather than let a bare FileNotFoundError traceback out of an otherwise-successful run.
        return None
    return dst
