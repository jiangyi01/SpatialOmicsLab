"""
Saved-key library — a hardened local vault of *validated* LLM keys.

Every key the wizard confirms actually works (a successful Tier-1 ``tier1_ping``) is remembered
here, so the next run can offer a pick-list — "reuse a saved LLM, or enter a new one" — at both
onboarding Step 1 and the run-phase key gate, instead of re-pasting a key each time. Each row is
labeled by its LLM (``Anthropic · claude-sonnet-4 · ****mnop``), with a short optional nickname
when a provider has more than one saved key.

Security posture — this is the FIRST secret-bearing file the wizard writes under ``.sog_setup/``
(the tree already holds secrets in ``.env`` backups), so the vault is deliberately hardened:

* it lives at ``.sog_setup/llm_keys.json`` — git-ignored (the repo-root ``.*`` rule *plus* a
  belt-and-braces ``.sog_setup/.gitignore`` this module writes, which also protects the existing
  secret-bearing ``backups/``),
* every save is ``chmod 0600`` (best-effort; POSIX only),
* every stored secret is :func:`~sog_install.session_log.register_secret`'d on load and
  save, so it is redacted from the session log and every console echo,
* it is shown only via :func:`~sog_install.session_log.mask_secret` (``****…last4``) —
  the raw value is never printed.

``.env`` stays the runtime source of truth; this vault only *feeds* a chosen key back into
``onboarding.configure_key`` → ``write_dotenv``. A read/write hiccup here must never block a key
write or a launch, so callers wrap :func:`remember` / :func:`save` in ``try/except``.

Stdlib only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from . import constants, credentials
from .session_log import _looks_secretish, mask_secret, notice, register_secret
from .state import _atomic_write_json, new_run_id

if TYPE_CHECKING:
    from pathlib import Path

    from .prompts import PromptIO

SCHEMA_VERSION = 1

# Menu sentinels — chosen so they can never collide with a real 8-hex entry id.
NEW_ID = "__new__"  # onboarding Step 1: "Enter a new key"
OTHER_ID = "__other__"  # run-phase gate: "Use a different key"

# Backward-compatible private aliases (call sites reference these).
_NEW_ID = NEW_ID
_OTHER_ID = OTHER_ID


def _now() -> str:
    """An ISO-second timestamp. Normal runtime — mirrors ``state.py`` / ``session_log.py``
    (the workflow-only ``datetime.now`` restriction does not apply to package code)."""
    return datetime.now().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class KeyEntry:
    """One remembered, validated LLM key.

    ``field_values`` holds ALL of the provider's fields — secrets included (e.g. an Azure key
    *and* its endpoint) — so selecting the entry fully reconstructs the config via
    ``assemble_owned_keys``. ``id`` is a deterministic fingerprint of the secret (see
    :func:`_fingerprint`), so re-saving the same key updates in place rather than duplicating."""

    id: str
    source: str
    model: str
    field_values: dict[str, str]
    validated: bool = True
    nickname: str = ""
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "model": self.model,
            "field_values": dict(self.field_values),
            "validated": bool(self.validated),
            "nickname": self.nickname,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KeyEntry:
        return cls(
            id=str(d.get("id", "")),
            source=str(d.get("source", "")),
            model=str(d.get("model", "")),
            field_values={str(k): str(v) for k, v in (d.get("field_values") or {}).items()},
            validated=bool(d.get("validated", True)),
            nickname=str(d.get("nickname", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            last_used_at=str(d.get("last_used_at", "")),
        )


@dataclass
class KeyLibrary:
    entries: list[KeyEntry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, d: dict) -> KeyLibrary:
        # `... or SCHEMA_VERSION` (not `.get(k, default)`): a present-but-null `schema_version:`
        # returns None (the default only applies when the key is ABSENT), and `int(None)` raises →
        # `load()`'s `except Exception` would then discard the WHOLE vault — every valid saved key —
        # over a null version field. Match the adjacent `entries or []` null-coalescing below. (R20)
        #
        # A present-but-NON-NUMERIC version — a JSON string (`"1.0"`, `"v1"`) or a container, from the
        # very same hand-edit / foreign-writer / partial-NFS-sync threat model R20 cites — makes
        # `int(...)` raise ValueError/TypeError, which `load()`'s `except Exception` again turns into a
        # whole-vault discard (the exact loss R20 set out to prevent; `or` only closed the null/falsy
        # case). Coerce TOTALLY: an unparseable version falls back to the current schema, never an
        # exception — mirroring the sibling `entries` clause tolerating a bad element (`if isinstance`). (R22)
        try:
            schema_version = int(d.get("schema_version") or SCHEMA_VERSION)
        except (ValueError, TypeError):
            schema_version = SCHEMA_VERSION
        # Fault-isolate EACH entry, don't merely skip non-dict elements: a dict entry whose inner
        # `field_values` is a truthy non-dict (a list/str/number from a hand-edit / foreign writer /
        # partial NFS sync) makes `KeyEntry.from_dict` raise AttributeError at `(... or {}).items()`,
        # which `load()`'s broad `except Exception` would turn into a WHOLE-vault discard — every other
        # validated key lost over one corrupt row. That is the exact loss the R20/R22 `schema_version`
        # hardening set out to prevent; extend it to entries so a single bad entry is dropped while the
        # good keys survive (true degradation, not empty-vault). (R27 F2)
        entries: list[KeyEntry] = []
        raw_entries = d.get("entries") or []
        if isinstance(raw_entries, list):  # a truthy non-list `entries: 5` would raise `for e in 5` → TypeError
            for e in raw_entries:
                if not isinstance(e, dict):
                    continue
                try:
                    entries.append(KeyEntry.from_dict(e))
                except Exception:
                    continue
        return cls(schema_version=schema_version, entries=entries)


# --------------------------------------------------------------------------- #
# Secret helpers (which values in an entry are credentials?)
# --------------------------------------------------------------------------- #
def _provider_for(source: str) -> credentials.ProviderSpec | None:
    try:
        return credentials.get_provider(source)
    except KeyError:
        return None


def _secret_values(source: str, field_values: dict[str, str]) -> list[str]:
    """The secret field values of an entry, in provider field order (used to register/mask and
    to fingerprint). Falls back to a field-*name* heuristic when the provider no longer resolves,
    so a stored key is still masked even after its provider is removed from the catalog."""
    prov = _provider_for(source)
    if prov is not None:
        return [field_values[f.env_var] for f in prov.all_fields() if f.secret and field_values.get(f.env_var)]
    return [v for k, v in field_values.items() if v and _looks_secretish(k)]


def _register_entry_secrets(entry: KeyEntry) -> None:
    for s in _secret_values(entry.source, entry.field_values):
        register_secret(s)


def _fingerprint(source: str, field_values: dict[str, str]) -> str:
    """A stable 8-hex id / dedupe key: ``sha256`` over the source + the entry's distinguishing material.

    Deterministic (no timestamp), so re-entering the same config upserts in place; a *different* key —
    or the same key under a *different provider config* — fingerprints differently and becomes a
    separate entry.

    The material is the sorted **secret** values PLUS the provider's declared **non-secret config**
    fields (``endpoint`` / ``base_url`` / ``region`` / ``api_version``). Two configs that share a
    secret but differ in one of those are genuinely different LLMs and must NOT collide — e.g. two
    OpenAI-compatible servers behind the *same* auth token but different ``base_url`` (SOG_CUSTOM), or
    the same AWS credentials used in two Bedrock ``region``s. Folding the declared non-secret fields in
    gives the keyed branch the SAME "distinguish by base_url/endpoint/region" safety the keyless branch
    below already has (F-sec-1); without it the second config silently overwrote the first on upsert.

    Only the provider's *declared* non-secret fields count, so a foreign/spurious ``field_values`` key
    can't fork identity, and a single-secret-field provider (Anthropic / OpenAI / Gemini / Groq — no
    declared non-secret field) keeps its old secret-only id: those entries never re-fingerprint on
    re-save (no migration). When the provider no longer resolves we cannot tell a config field from
    junk, so those (menu-skipped) entries keep the secret-only id too.

    For a **keyless** config (no secret value at all — e.g. a custom endpoint whose key field is left
    blank), there is no secret to key on: fall back to the full sorted ``field_values`` so two distinct
    keyless endpoints that differ only in a non-secret field don't both collapse to ``sha256(source)``.
    A truly fieldless keyless provider (e.g. Ollama) has nothing to distinguish, so those still dedupe.

    ``model`` is deliberately NOT folded in: upsert refreshes the model in place (one row per key,
    model mutable), and folding it would re-fingerprint *every* existing entry. Two Azure deployments on
    the *same* resource (same key+endpoint, differing only by deployment=model) therefore still share a
    row — disambiguate those with a nickname."""
    material = sorted(_secret_values(source, field_values))
    if not material:
        material = [f"{k}={v}" for k, v in sorted(field_values.items()) if v]
    else:
        # Keyed: also fold in the provider's DECLARED non-secret config fields, so a shared secret with
        # a differing endpoint/base_url/region is a distinct entry (not a silent overwrite). Restricted
        # to declared fields ⇒ no foreign key forks identity and single-secret-field providers keep
        # their old id (material unchanged). Unresolvable provider ⇒ leave the secret-only id as-is.
        prov = _provider_for(source)
        if prov is not None:
            nonsecret = {f.env_var for f in prov.all_fields() if not f.secret}
            material += [f"{k}={v}" for k, v in sorted(field_values.items()) if v and k in nonsecret]
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    for s in material:
        h.update(b"\x00")
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:8]


def _primary_secret(entry: KeyEntry) -> str:
    vals = _secret_values(entry.source, entry.field_values)
    return vals[0] if vals else ""


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
def label_for(entry: KeyEntry) -> str:
    """The pick-list label — *always* leads with the LLM (provider + model) so each row reads as
    "which LLM is this", then a masked ``****…last4`` and an optional nickname. Never the raw key."""
    prov = _provider_for(entry.source)
    prov_label = prov.label if prov is not None else (entry.source or "LLM")
    model = entry.model or (prov.default_model if prov is not None else "") or "?"
    base = f"{prov_label} · {model} · {mask_secret(_primary_secret(entry))}"
    return f"{base}  ({entry.nickname})" if entry.nickname else base


def _hint_for(entry: KeyEntry) -> str:
    parts: list[str] = []
    if entry.created_at:
        parts.append(f"saved {entry.created_at[:10]}")
    if entry.last_used_at and entry.last_used_at[:10] != (entry.created_at[:10] if entry.created_at else ""):
        parts.append(f"last used {entry.last_used_at[:10]}")
    if not entry.validated:
        parts.append("unvalidated")
    return ", ".join(parts)


def menu_options(lib: KeyLibrary) -> list[tuple[str, str, str]]:
    """``(id, label, hint)`` tuples for :meth:`PromptIO.select`, most-recently-used first.

    Entries whose provider no longer resolves are skipped — we cannot safely reconstruct or label
    them — so a stale catalog can never surface an unusable pick."""
    usable = [e for e in lib.entries if _provider_for(e.source) is not None]
    usable.sort(key=lambda e: (e.last_used_at or e.created_at or "", e.created_at or ""), reverse=True)
    return [(e.id, label_for(e), _hint_for(e)) for e in usable]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _ensure_gitignore() -> None:
    """Write a belt-and-braces ``.sog_setup/.gitignore`` (``*``) if absent, so the vault — and the
    existing secret-bearing ``.env`` backups — can never be committed even if the repo-root ``.*``
    rule is removed or someone ``git add -f``s the tree. Best-effort; never fatal."""
    try:
        gi = constants.state_dir() / ".gitignore"
        if not gi.exists():
            gi.write_text(
                "# machine-local setup state — never commit (secrets live here)\n*\n!.gitignore\n", encoding="utf-8"
            )
    except OSError:
        pass


def load(path: Path | None = None) -> KeyLibrary:
    """Read the vault (empty :class:`KeyLibrary` on missing/corrupt). Registers every stored secret
    for masking, so the moment the library is loaded its keys are redacted everywhere."""
    target = path or constants.key_library_file()
    if not target.exists():
        return KeyLibrary()
    try:
        with open(target, encoding="utf-8") as f:
            raw = json.load(f)
        # A complete-but-non-object vault (``[]`` / ``null`` / ``"x"`` from a hand-edit, a foreign
        # tool, or a partial NFS sync) would make ``from_dict`` call ``.get`` on a non-dict and raise
        # AttributeError — which the old narrow except let escape to the unwrapped onboarding call
        # site (``onboarding.py`` Step 1), aborting the wizard with a traceback. Guard the shape, and
        # broaden the catch to ``Exception`` so a malformed *inner* field (e.g. ``field_values`` as a
        # list) also degrades to an empty vault. The vault is a convenience cache (``.env`` is the
        # runtime source of truth), so it must never block a launch. Mirrors ``state.SetupState.load``.
        if not isinstance(raw, dict):
            raise ValueError("vault is not a JSON object")
        lib = KeyLibrary.from_dict(raw)
    except Exception:
        return KeyLibrary()
    for e in lib.entries:
        _register_entry_secrets(e)
    return lib


def _set_aside_unparseable(target: Path) -> None:
    """Move a vault that does not parse out of the way before a save replaces it.

    (hunt 2026-09-30, u35b-setup-state-13) :func:`load` reads such a vault as EMPTY -- the per-field
    guards above only help once the JSON parses -- so the next ``remember`` saved a vault holding just
    the new key, silently discarding every other one. Kept as ``llm_keys.json.corrupt-<id>`` (0600),
    and said so. If it cannot be moved, this raises: refusing the save is better than overwriting."""
    if not target.exists():
        return
    try:
        with open(target, encoding="utf-8") as f:
            if isinstance(json.load(f), dict):
                return  # parses — load() kept every entry it could
    except Exception:
        pass
    dst = target.with_name(f"{target.name}.corrupt-{new_run_id()}")
    os.replace(target, dst)
    with contextlib.suppress(OSError):
        os.chmod(dst, 0o600)
    notice(f"the saved-key vault could not be read; kept it as {dst.name} and started a new one")


def save(lib: KeyLibrary, path: Path | None = None) -> None:
    """Atomically persist the vault, then harden it: ``chmod 0600`` + a git-ignore guard. Registers
    every secret for masking. Raises only on a genuine write failure (callers swallow it)."""
    constants.ensure_state_dirs()
    _ensure_gitignore()
    target = path or constants.key_library_file()
    _set_aside_unparseable(target)
    _atomic_write_json(target, lib.to_dict())
    try:
        os.chmod(target, 0o600)
    except OSError:  # non-POSIX / unusual filesystem — git-ignore + masking still apply
        pass
    for e in lib.entries:
        _register_entry_secrets(e)


# --------------------------------------------------------------------------- #
# Cross-process lock (N16) — serialize a vault load→save read-modify-write
# --------------------------------------------------------------------------- #
_LOCK_TIMEOUT_SEC = 5.0  # wait at most this long for a concurrent writer before proceeding anyway
_LOCK_STALE_SEC = 30.0  # a lockfile older than this is a crashed run's orphan — steal it
_LOCK_POLL_SEC = 0.05


@contextlib.contextmanager
def _vault_lock(path: Path | None = None):
    """Best-effort cross-process lock around a vault load→save critical section.

    Two overlapping ``sog-setup`` runs (or a shared NFS home) otherwise lost-update the vault: both
    ``load`` the same file, both ``save``, and the second save wins the WHOLE file — silently dropping
    the first run's new entry (the atomic write prevents corruption, not the lost entry). This
    serializes the read-modify-write with an ``O_CREAT|O_EXCL`` lockfile that is portable (POSIX +
    Windows) and, crucially, **never fatal**: on contention it waits briefly, then proceeds anyway
    rather than block a key write; a stale lockfile from a crashed run is stolen. The vault is a
    convenience cache (``.env`` is the runtime source of truth), so availability beats strict mutual
    exclusion here — matching this module's other best-effort guards (``chmod``, git-ignore)."""
    with contextlib.suppress(OSError):
        constants.ensure_state_dirs()  # so the lockfile's parent dir exists
    target = path or constants.key_library_file()
    lock = target.parent / (target.name + ".lock")
    acquired = False
    deadline = time.monotonic() + _LOCK_TIMEOUT_SEC
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            # Held by someone else — steal a stale orphan, else wait a beat and eventually give up.
            try:
                if time.time() - os.stat(lock).st_mtime > _LOCK_STALE_SEC:
                    os.unlink(lock)
                    continue
            except OSError:
                pass  # the lock vanished between calls — loop and retry the create
            if time.monotonic() >= deadline:
                break  # proceed WITHOUT the lock rather than block a key write (best-effort)
            time.sleep(_LOCK_POLL_SEC)
        except OSError:
            break  # can't even attempt a lock on this filesystem — proceed best-effort
    try:
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                os.unlink(lock)


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #
def get(lib: KeyLibrary, entry_id: str) -> KeyEntry | None:
    return next((e for e in lib.entries if e.id == entry_id), None)


def upsert(
    lib: KeyLibrary,
    *,
    source: str,
    model: str | None,
    field_values: dict[str, str],
    validated: bool,
    now: str,
) -> tuple[KeyEntry, bool]:
    """Insert or update a key by fingerprint. Returns ``(entry, is_new)``. On an update we keep
    ``created_at`` and any ``nickname`` but refresh model/validated/timestamps."""
    fv = {str(k): str(v) for k, v in field_values.items() if v}
    fp = _fingerprint(source, fv)
    existing = get(lib, fp)
    if existing is not None:
        existing.model = model or existing.model
        existing.field_values = fv or existing.field_values
        existing.validated = bool(validated) or existing.validated
        existing.updated_at = now
        existing.last_used_at = now
        _register_entry_secrets(existing)
        return existing, False
    entry = KeyEntry(
        id=fp,
        source=source,
        model=model or "",
        field_values=fv,
        validated=bool(validated),
        created_at=now,
        updated_at=now,
        last_used_at=now,
    )
    lib.entries.append(entry)
    _register_entry_secrets(entry)
    return entry, True


def _ask_nickname(io: PromptIO) -> str:
    """Prompt for a short disambiguating nickname. Cosmetic and never fatal: robust in
    scripted/non-interactive mode (a missing canned answer or exhausted input yields '')."""
    try:
        return io.ask_text("  Short name for this key (to tell it apart)", default="").strip()
    except Exception:
        return ""


def remember(
    io: PromptIO,
    *,
    source: str,
    model: str | None,
    field_values: dict[str, str],
    validated: bool,
    now: str,
) -> KeyEntry:
    """Capture a validated key: load → upsert (dedupe by fingerprint) → optional nickname → save.

    The nickname is prompted **only** for a genuinely new key when the library already holds
    another key for the same provider (auto-label otherwise), so a single key per provider is
    never interrupted. Returns the stored entry.

    N16: the load→save read-modify-write runs under a best-effort cross-process lock so an
    overlapping run can't lost-update the new entry.

    #95b: the interactive nickname prompt is done BEFORE the lock is taken, never inside it. Holding
    the lock across a blocking ``input()`` is a latent correctness bug: a user pondering a name for
    longer than ``_LOCK_STALE_SEC`` (30s) lets a concurrent run judge this lock a crashed orphan and
    STEAL it — then this run's ``finally`` unlinks the concurrent run's fresh lock, breaking mutual
    exclusion for both. So we pre-read (no lock) only to decide whether a prompt is even warranted,
    prompt outside the lock, then hold the lock solely for the fast load→upsert→save."""
    # Pre-read (lock-free): a new key gets a disambiguating nickname ONLY when the vault already holds
    # another key for the same provider. This read is advisory — the authoritative state is re-derived
    # under the lock below — so it needn't be serialized.
    pre = load()
    fv = {str(k): str(v) for k, v in field_values.items() if v}
    prompt_nickname = _fingerprint(source, fv) not in {e.id for e in pre.entries} and any(
        e.source == source for e in pre.entries
    )
    nickname = _ask_nickname(io) if prompt_nickname else ""
    with _vault_lock():
        lib = load()
        entry, _is_new = upsert(
            lib, source=source, model=model, field_values=field_values, validated=validated, now=now
        )
        if nickname and not entry.nickname:  # apply the pre-prompted name (also covers a concurrent add)
            entry.nickname = nickname
        save(lib)
    return entry


def touch(entry_id: str, now: str, *, path: Path | None = None) -> None:
    """Bump ``last_used_at`` for a picked entry (load → set → save). Silent no-op if it's gone.

    N16: also under the vault lock — a concurrent ``remember`` adding a NEW entry must not be clobbered
    by this run's timestamp-only save (both do load→save; the later writer would otherwise win the
    whole file)."""
    with _vault_lock(path):
        lib = load(path)
        entry = get(lib, entry_id)
        if entry is None:
            return
        entry.last_used_at = now
        save(lib, path)


def remove(lib: KeyLibrary, entry_id: str) -> bool:
    """Drop an entry in memory (caller saves). Returns whether anything was removed. No UI wires
    this in v1 — it exists for tests and a future "forget this key" affordance."""
    before = len(lib.entries)
    lib.entries = [e for e in lib.entries if e.id != entry_id]
    return len(lib.entries) != before
