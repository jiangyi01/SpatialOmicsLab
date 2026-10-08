"""Memory manager for STCoscientist user-MCP-tool lifecycle.

Provides optional long-term (cross-tool) + short-term (per-tool)
memory of creation/deletion/modification attempts. Used by STCoscientist to
learn from prior attempts and avoid known pitfalls.

OFF BY DEFAULT. Enable via `default_config.memory_enabled = True`
before constructing STCoscientist (or env var `SOG_MEMORY_ENABLED=true`).

Bug mitigations from design plan (see debug_log/patterns/06_plan_memory_system.md):
- Atomic writes (L/F/S bug guards)
- Per-process threading.RLock + cross-process filelock (H23)
- Schema version + quarantine on corrupt (L1, L3, L20, H9)
- FIFO cap with keep-best pin (L8)
- UTC timestamps (H19)
- ensure_ascii=False for unicode (H22)
- IDN canonicalization of source_url (H21)
- SHA256 of URL as filename (F13)
- Prompt-injection sanitization (F6, L14)
- HMAC signing via per-install secret (H13, M22)
- No top-level module-load side effects
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    # No `Timeout as FileLockTimeout` here: nothing in this module catches it, deliberately --
    # see `_lock_for`. An unused import of it is how the same gap in `tools_user/self_review.py`
    # went unnoticed, so it stays out until there is a handler to justify it.
    from filelock import FileLock

    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False

SCHEMA_VERSION = "1.0"
LOGGER = logging.getLogger("spatialomicsgym.memory")

# --- Outcome enum (plan §2) ---
OUTCOME_FULL_PASS = "full_pass"
OUTCOME_VENDOR_FALLBACK = "vendor_fallback_pass"
OUTCOME_SOFT_DEGRADED = "soft_degraded"
OUTCOME_ROLLED_BACK = "rolled_back"
OUTCOME_HUNG = "hung"
OUTCOME_MOD_PASSED = "modification_passed"
OUTCOME_MOD_ROLLED_BACK = "modification_rolled_back"
VALID_OUTCOMES = {
    OUTCOME_FULL_PASS,
    OUTCOME_VENDOR_FALLBACK,
    OUTCOME_SOFT_DEGRADED,
    OUTCOME_ROLLED_BACK,
    OUTCOME_HUNG,
    OUTCOME_MOD_PASSED,
    OUTCOME_MOD_ROLLED_BACK,
}

# --- Caps from plan §2, §7 ---
DEFAULT_MAX_ATTEMPTS = 10
DEFAULT_STALENESS_DAYS = 30
DEFAULT_LOCK_TIMEOUT_SEC = 30
DEFAULT_BACKUP_RETENTION_DAYS = 7

# Where the store lives when nobody names a location. Anchored to this file rather than spelled
# relatively, because a relative path is resolved against whatever directory the process started
# in -- see _configured_root. This resolves to the same directory the old relative default did
# from the repo root, so an install with records already on disk keeps finding them.
DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent / ".memory"


def _utc_now() -> str:
    """UTC ISO 8601 with offset (H19, H20 fix)."""
    return datetime.now(timezone.utc).isoformat()


def _canonical_url(url: str) -> str:
    """Canonicalize source URL for consistent keying (F13, H21, L13).

    - Lowercase
    - Strip trailing /
    - Strip .git suffix
    - IDN-normalize host (Punycode)
    """
    if not url:
        return ""
    u = url.strip().lower()
    if u.endswith("/"):
        u = u[:-1]
    if u.endswith(".git"):
        u = u[:-4]
    try:
        parsed = urlparse(u)
        if parsed.hostname:
            try:
                idn_host = parsed.hostname.encode("idna").decode("ascii")
            except UnicodeError:
                idn_host = parsed.hostname
            u = parsed._replace(netloc=idn_host).geturl()
    except Exception:
        pass
    return u


def _key_for(url: str) -> str:
    """SHA256(canonical_url)[:16] — deterministic, cross-restart stable (F13)."""
    canonical = _canonical_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _sanitize_text(text: str, max_len: int = 2000) -> str:
    """Strip prompt-injection-y patterns + control chars + length cap (F6, L14)."""
    if not isinstance(text, str):
        text = str(text)
    # Remove ASCII control chars except \n, \t
    text = "".join(c for c in text if c >= " " or c in ("\n", "\t"))
    # Remove obvious meta-instruction patterns
    meta_patterns = [
        r"(?i)IGNORE\s+(PRIOR|PREVIOUS|ALL)\s+INSTRUCTIONS?",
        r"(?i)DISREGARD\s+(PRIOR|PREVIOUS|ALL)\s+(RULES?|INSTRUCTIONS?)",
        r"(?i)OVERRIDE\s+(SYSTEM|RULES?)",
        r"(?i)\bSYSTEM\s*:\s*YOU\b",
    ]
    for pat in meta_patterns:
        text = re.sub(pat, "[redacted]", text)
    # Escape angle brackets that could break <prior_attempt> tag
    text = text.replace("<prior_attempt>", "[prior_attempt]")
    text = text.replace("</prior_attempt>", "[/prior_attempt]")
    # Length cap
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically; verify parse-back before commit (H18, L2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        # H18 — parse-back verification before replace
        with open(tmp_path, encoding="utf-8") as f:
            _ = json.load(f)
        os.replace(tmp, str(path))
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _memory_disabled() -> bool:
    """Check memory_enabled via config or env var. Default False.

    Precedence: env var > default_config.memory_enabled > False (L13).

    The spellings mirror ``config._env_bool`` (strip, lower; true/1/yes/on against false/0/no/off),
    as ``self_review._enabled`` already does. A value this function does not recognise is not read
    as "off" -- it *falls through* to the config, so a narrower set here does not fail safe, it
    hands the decision back to a default that usually disagrees. ``SOG_MEMORY_ENABLED=on`` left
    memory off, and a ``.env`` line with a trailing space did the same to ``true``.
    """
    env_val = (os.environ.get("SOG_MEMORY_ENABLED") or os.environ.get("BIOMNI_MEMORY_ENABLED") or "").strip().lower()
    if env_val in ("true", "1", "yes", "on"):
        return False  # not disabled (i.e. enabled)
    if env_val in ("false", "0", "no", "off"):
        return True  # disabled
    # Fallback to config attribute if present (imported lazily — don't
    # require spatialomicsgym.config to be importable at memory-module load time)
    try:
        from spatialomicsgym.config import default_config

        return not bool(getattr(default_config, "memory_enabled", False))
    except Exception:
        return True  # default off


def _config_value(name: str):
    """One attribute off ``default_config``, or None when it cannot be read.

    ``getattr(obj, name, default)`` returns the default only for a *missing* attribute -- an
    exception raised inside ``__getattr__`` or a property propagates straight through it. That put
    a broken config object on the path that constructs ``MemoryConfig``, in a module whose whole
    contract is that a memory problem never halts a run. Unreadable reads as unset.
    """
    try:
        from spatialomicsgym.config import default_config

        return getattr(default_config, name, None)
    except Exception:
        return None


def _config_overrides() -> dict:
    """The ``MemoryConfig`` fields a user can set through ``default_config``.

    ``MemoryConfig(root=root)`` took every default, so ``memory_max_attempts_per_tool`` -- which is
    documented and has an env override (``SOG_MEMORY_MAX_ATTEMPTS``) -- had no effect on anything,
    and ``staleness_days`` was unreachable from any front door, which is why the freshness gate
    the playbook describes was never wired up.

    Only values the config actually carries are forwarded, and only if they are the right type, so
    a stub or partially-populated config object cannot break construction. A cap below 1 is
    ignored rather than honoured: ``_apply_cap`` would compute a negative keep_count and the pins
    would be the only survivors, which is not what a user setting "1" is asking for.
    """
    out: dict[str, int] = {}
    cap = _config_value("memory_max_attempts_per_tool")
    if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1:
        out["max_attempts_per_tool"] = cap
    staleness = _config_value("memory_staleness_days")
    if isinstance(staleness, int) and not isinstance(staleness, bool):
        # Non-positive is a legitimate setting: it turns the freshness gate off.
        out["staleness_days"] = staleness
    return out


def _configured_root() -> Path:
    """Where the store lives when the caller named no root and no environment variable did either.

    Was ``Path("tools_user/.memory")``. A relative path is resolved against the process CWD, so the
    store moved with the agent: a recipe recorded from the repo root was invisible to a run started
    from the user's data directory, which was then told the tool had never been attempted, and a
    stray ``tools_user/.memory`` tree was left wherever that run happened to start.

    ``default_config.memory_path`` is read here rather than by nobody. It is documented at
    ``config.py:136`` and is where ``SOG_MEMORY_PATH`` lands, but the manager only ever consulted
    the environment, so setting the attribute did nothing -- the dead-knob shape of #176.

    A path the user supplied is used exactly as given, relative or not: only our own default is
    anchored, because resolving somebody else's relative path for them is a different decision.
    """
    configured = _config_value("memory_path")
    if isinstance(configured, str) and configured.strip():
        return Path(configured)
    return DEFAULT_MEMORY_ROOT


@dataclass
class MemoryConfig:
    root: Path
    max_attempts_per_tool: int = DEFAULT_MAX_ATTEMPTS
    staleness_days: int = DEFAULT_STALENESS_DAYS
    lock_timeout_sec: int = DEFAULT_LOCK_TIMEOUT_SEC
    backup_retention_days: int = DEFAULT_BACKUP_RETENTION_DAYS
    enable_hmac: bool = True


class MemoryManager:
    """Per-process memory manager with atomic+locked writes.

    Lazily-instantiated. If memory is disabled, calling any mutating method
    raises RuntimeError — this is a guard against code paths that forgot to
    check the enable flag (S8).
    """

    _instance_lock = threading.RLock()
    _instance: MemoryManager | None = None

    def __init__(self, root: Path | None = None):
        env_root = os.environ.get("SOG_MEMORY_PATH") or os.environ.get("BIOMNI_MEMORY_PATH") or ""
        if root is None:
            root = Path(env_root) if env_root else _configured_root()
        else:
            root = Path(root)
        self.cfg = MemoryConfig(root=root, **_config_overrides())
        self.root = root
        self.short_term_dir = self.root / "short_term"
        self.archive_dir = self.root / "archive"
        self.quarantine_dir = self.root / ".quarantine"
        self.backups_dir = self.root / ".backups"
        self.long_term_path = self.root / "long_term.json"
        self.index_path = self.root / "index.json"
        self.secret_path = self.root / ".secret"
        self._rlock = threading.RLock()
        self._snapshot_cache: dict[str, dict] = {}  # H1 per-turn cache

    @classmethod
    def get(cls) -> MemoryManager:
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = MemoryManager()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Test helper — reset singleton."""
        with cls._instance_lock:
            cls._instance = None

    # --- Internal helpers ---

    def _ensure_layout(self) -> None:
        """Create dirs on first write. Skip if disabled (C4)."""
        if _memory_disabled():
            raise RuntimeError(
                "MemoryManager invoked while disabled — this is a bug. "
                "Check default_config.memory_enabled or SOG_MEMORY_ENABLED."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        self.short_term_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        # Create per-install secret for HMAC (H13, M22)
        if self.cfg.enable_hmac and not self.secret_path.exists():
            self.secret_path.write_text(secrets.token_hex(32))
            try:
                os.chmod(self.secret_path, 0o400)
            except OSError:
                pass

    def _load_secret(self) -> bytes:
        try:
            return self.secret_path.read_text().encode("utf-8")
        except Exception:
            return b""

    def _hmac_sign(self, data: dict) -> str:
        key = self._load_secret()
        if not key:
            return ""
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hmac.new(key, payload, hashlib.sha256).hexdigest()

    def _hmac_verify(self, data: dict, signature: str) -> bool:
        expected = self._hmac_sign(data)
        if not expected or not signature:
            return False
        return hmac.compare_digest(expected, signature)

    def _lock_for(self, path: Path, *, timeout: int | None = None):
        """Return a file+thread lock for ``path`` (a threading lock when filelock is absent).

        A lock that cannot be acquired within ``timeout`` raises ``filelock.Timeout`` out of the
        ``with`` statement into the caller. That is deliberate rather than an oversight: the writers
        that use it return ``None``, so swallowing the timeout here would be indistinguishable from
        a completed write, and ``seed_tool_creation_memory`` would report records it never stored.

        Tolerating a busy lock is therefore the caller's decision, and the callers make it:
        ``delete_short_term`` catches in its own body and returns False, and the agent-side write
        goes through ``stcoscientist._record_memory_attempt``, which keeps a memory failure from
        blocking the run.
        """
        timeout = timeout or self.cfg.lock_timeout_sec
        if HAS_FILELOCK:
            return FileLock(str(path) + ".lock", timeout=timeout)
        # Fallback: threading-only
        return self._rlock

    def _quarantine(self, path: Path, reason: str) -> None:
        """Move corrupt / schema-mismatched files to quarantine (L20, H9)."""
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            target = self.quarantine_dir / f"{ts}_{path.name}"
            os.replace(str(path), str(target))
            LOGGER.warning("[memory] quarantined %s -> %s: %s", path, target, reason)
        except Exception as e:
            LOGGER.warning("[memory] failed to quarantine %s: %s", path, e)

    def _migrate_if_needed(self, data: dict, path: Path) -> dict | None:
        """Schema migration hook (L1).

        Returns migrated data or None if the file should be quarantined.
        """
        v = data.get("schema_version")
        if v == SCHEMA_VERSION:
            return data
        if v is None:
            # Pre-versioned payload — upgrade in place
            data["schema_version"] = SCHEMA_VERSION
            return data
        # Unknown version → quarantine
        self._quarantine(path, f"unknown schema_version={v!r}")
        return None

    # --- Short-term ---

    def short_term_path(self, source_url: str) -> Path:
        return self.short_term_dir / f"{_key_for(source_url)}.json"

    def read_short_term(self, source_url: str, *, use_cache: bool = True) -> dict:
        """Read per-tool memory. Returns {} if disabled or absent."""
        if _memory_disabled():
            return {}
        key = _key_for(source_url)
        if use_cache and key in self._snapshot_cache:
            return self._snapshot_cache[key]
        path = self.short_term_path(source_url)
        if not path.exists():
            self._snapshot_cache[key] = {}
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self._quarantine(path, f"read error: {e}")
            self._snapshot_cache[key] = {}
            return {}
        # HMAC verify (H13)
        sig = data.pop("_hmac", None)
        if self.cfg.enable_hmac and sig is not None:
            if not self._hmac_verify(data, sig):
                self._quarantine(path, "hmac verification failed")
                self._snapshot_cache[key] = {}
                return {}
        migrated = self._migrate_if_needed(data, path)
        if migrated is None:
            self._snapshot_cache[key] = {}
            return {}
        self._snapshot_cache[key] = migrated
        return migrated

    def append_attempt(self, source_url: str, record: dict) -> None:
        """Append an attempt to short-term memory.

        record must include: outcome, started, finished, strategy (optional),
        api_shape (optional), gotchas (optional), error_classes_seen (optional).
        """
        if _memory_disabled():
            raise RuntimeError(
                "append_attempt called while memory is disabled. This is a bug "
                "— check default_config.memory_enabled before calling."
            )
        self._ensure_layout()
        outcome = record.get("outcome")
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"invalid outcome {outcome!r}; must be in {VALID_OUTCOMES}")
        canonical = _canonical_url(source_url)
        path = self.short_term_path(source_url)
        with self._lock_for(path):
            # Read existing (if any)
            existing = {}
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        existing = json.load(f) or {}
                        existing.pop("_hmac", None)
                except Exception:
                    self._quarantine(path, "read error during append")
                    existing = {}
            attempts = existing.get("attempts", [])
            # Sanitize record (F6, L14)
            sanitized = self._sanitize_record(record)
            sanitized["attempt_id"] = len(attempts) + 1
            sanitized.setdefault("started", _utc_now())
            sanitized.setdefault("finished", _utc_now())
            sanitized["last_verified"] = _utc_now()
            attempts.append(sanitized)
            # L8 — FIFO cap with keep-best pin
            attempts = self._apply_cap(attempts)
            # L5 — compute best_attempt_id
            best_id = self._compute_best_attempt_id(attempts)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "source_url": canonical,
                "tool_id_history": self._update_tool_id_history(
                    existing.get("tool_id_history", []), record.get("tool_id")
                ),
                "attempts": attempts,
                "best_attempt_id": best_id,
                "last_verified": _utc_now(),
                "unavailable_reason": self._derive_unavailable_reason(attempts),
            }
            if self.cfg.enable_hmac:
                payload["_hmac"] = self._hmac_sign({k: v for k, v in payload.items() if k != "_hmac"})
            _atomic_write_json(path, payload)
            # Invalidate cache
            self._snapshot_cache.pop(_key_for(source_url), None)

    def delete_short_term(self, source_url: str) -> bool:
        """Delete per-tool memory. Returns True if deleted, False if absent.
        Does NOT raise — memory failure never blocks lifecycle (F12).
        """
        if _memory_disabled():
            return False
        path = self.short_term_path(source_url)
        if not path.exists():
            return False
        try:
            with self._lock_for(path):
                path.unlink()
                self._snapshot_cache.pop(_key_for(source_url), None)
                return True
        except Exception as e:
            LOGGER.warning("[memory] failed to delete short_term for %s: %s", source_url, e)
            return False

    # --- Long-term ---

    def read_long_term(self) -> dict:
        """Read aggregate long-term memory. Returns {} if disabled."""
        if _memory_disabled():
            return {}
        if not self.long_term_path.exists():
            return {}
        try:
            with open(self.long_term_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self._quarantine(self.long_term_path, f"read error: {e}")
            return {}
        sig = data.pop("_hmac", None)
        if self.cfg.enable_hmac and sig is not None:
            if not self._hmac_verify(data, sig):
                self._quarantine(self.long_term_path, "hmac verification failed")
                return {}
        migrated = self._migrate_if_needed(data, self.long_term_path)
        return migrated or {}

    def update_long_term(
        self,
        *,
        outcome: str,
        language: str | None = None,
        install_path: str | None = None,
        time_sec: float | None = None,
        domain: str | None = None,
    ) -> None:
        """Increment aggregate counters. L19 — creation outcomes only."""
        if _memory_disabled():
            raise RuntimeError("update_long_term called while memory disabled.")
        self._ensure_layout()
        with self._lock_for(self.long_term_path):
            cur = self.read_long_term() or {
                "schema_version": SCHEMA_VERSION,
                "version": "1.0",
                "updated_at": _utc_now(),
                "aggregate_stats": {"total_creations": 0, "total_successes": 0, "by_outcome": {}},
                "by_language": {},
                "by_install_path": {},
                "by_domain": {},
                "known_hangs": [],
                "recurring_patterns": [],
                "common_pitfalls": [],
            }
            cur.setdefault("aggregate_stats", {}).setdefault("by_outcome", {})
            cur["aggregate_stats"]["total_creations"] = cur["aggregate_stats"].get("total_creations", 0) + 1
            if outcome in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK, OUTCOME_SOFT_DEGRADED):
                cur["aggregate_stats"]["total_successes"] = cur["aggregate_stats"].get("total_successes", 0) + 1
            cur["aggregate_stats"]["by_outcome"][outcome] = cur["aggregate_stats"]["by_outcome"].get(outcome, 0) + 1
            if language:
                bl = cur.setdefault("by_language", {}).setdefault(
                    language, {"attempts": 0, "success": 0, "total_time_sec": 0}
                )
                bl["attempts"] += 1
                if outcome in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK):
                    bl["success"] += 1
                if time_sec:
                    bl["total_time_sec"] = bl.get("total_time_sec", 0) + time_sec
                    bl["avg_time_sec"] = bl["total_time_sec"] // max(bl["attempts"], 1)
            if install_path:
                bi = cur.setdefault("by_install_path", {}).setdefault(install_path, {"attempts": 0, "success": 0})
                bi["attempts"] += 1
                if outcome in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK):
                    bi["success"] += 1
            if domain:
                bd = cur.setdefault("by_domain", {}).setdefault(domain, {"attempts": 0, "success": 0})
                bd["attempts"] += 1
                if outcome in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK):
                    bd["success"] += 1
            cur["updated_at"] = _utc_now()
            if self.cfg.enable_hmac:
                payload = {k: v for k, v in cur.items() if k != "_hmac"}
                cur["_hmac"] = self._hmac_sign(payload)
            _atomic_write_json(self.long_term_path, cur)

    def mark_known_hang(self, source_url: str, reason: str) -> None:
        """Record that a tool consistently hangs, so the next run is told before it retries.

        ``add_new_mcp_tool.md`` instructs the agent to call this after aborting a hung install.
        Read back by :meth:`known_hang_for`; until that existed the record was written and never
        looked at, and the next run was told the tool was a fresh creation.
        """
        if _memory_disabled():
            return
        self._ensure_layout()
        with self._lock_for(self.long_term_path):
            cur = self.read_long_term() or {"schema_version": SCHEMA_VERSION, "known_hangs": []}
            hangs = cur.setdefault("known_hangs", [])
            canonical = _canonical_url(source_url)
            existing = next((h for h in hangs if _canonical_url(h.get("source_url", "")) == canonical), None)
            if existing:
                existing["observed_times"] = existing.get("observed_times", 0) + 1
                existing["last_seen"] = _utc_now()
                existing["reason"] = _sanitize_text(reason, max_len=500)
            else:
                hangs.append(
                    {
                        "source_url": canonical,
                        "reason": _sanitize_text(reason, max_len=500),
                        "observed_times": 1,
                        "first_seen": _utc_now(),
                        "last_seen": _utc_now(),
                    }
                )
            cur["updated_at"] = _utc_now()
            if self.cfg.enable_hmac:
                payload = {k: v for k, v in cur.items() if k != "_hmac"}
                cur["_hmac"] = self._hmac_sign(payload)
            _atomic_write_json(self.long_term_path, cur)

    def known_hang_for(self, source_url: str) -> dict | None:
        """Return the recorded hang for ``source_url``, or None if there is none.

        The read half of :meth:`mark_known_hang`. It lives here, rather than at the call site,
        because the write stores the *canonical* URL: a caller comparing the URL it was handed
        against the stored one would miss every spelling difference the canonicaliser exists to
        absorb -- case, a trailing slash, a ``.git`` suffix.

        Read by ``stcoscientist._enrich_prompt_with_memory_hints`` on every run that names a GitHub
        source, so a tool the previous run had to kill is not announced as one never seen before.
        """
        if _memory_disabled():
            return None
        canonical = _canonical_url(source_url)
        if not canonical:
            return None
        for hang in self.read_long_term().get("known_hangs") or []:
            if _canonical_url(hang.get("source_url", "")) == canonical:
                return hang
        return None

    # --- Bulk / backup / helpers ---

    def delete_all_short_term(self, *, backup: bool = True) -> int:
        """Delete all short-term memory. Returns count deleted."""
        if _memory_disabled():
            return 0
        if not self.short_term_dir.exists():
            return 0
        if backup:
            self._backup_before_delete("short_term")
        count = 0
        for p in list(self.short_term_dir.glob("*.json")):
            try:
                p.unlink()
                count += 1
            except Exception as e:
                LOGGER.warning("[memory] delete %s failed: %s", p, e)
        self._snapshot_cache.clear()
        return count

    def delete_long_term(self, *, backup: bool = True) -> bool:
        if _memory_disabled():
            return False
        if not self.long_term_path.exists():
            return False
        if backup:
            self._backup_before_delete("long_term")
        try:
            self.long_term_path.unlink()
            return True
        except Exception as e:
            LOGGER.warning("[memory] delete long_term failed: %s", e)
            return False

    def _backup_before_delete(self, scope: str) -> Path:
        self._ensure_layout()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = self.backups_dir / f"{ts}_{scope}"
        backup.mkdir(parents=True, exist_ok=True)
        if scope in ("short_term", "all") and self.short_term_dir.exists():
            for p in self.short_term_dir.glob("*.json"):
                target = backup / p.name
                try:
                    with open(p, "rb") as src, open(target, "wb") as dst:
                        dst.write(src.read())
                except Exception:
                    pass
        if scope in ("long_term", "all") and self.long_term_path.exists():
            try:
                with open(self.long_term_path, "rb") as src, open(backup / self.long_term_path.name, "wb") as dst:
                    dst.write(src.read())
            except Exception:
                pass
        # Purge old backups
        self._purge_old_backups()
        return backup

    def _purge_old_backups(self) -> None:
        if not self.backups_dir.exists():
            return
        import time

        cutoff = time.time() - (self.cfg.backup_retention_days * 86400)
        for d in self.backups_dir.iterdir():
            try:
                if d.stat().st_mtime < cutoff:
                    for p in d.rglob("*"):
                        if p.is_file():
                            p.unlink()
                    d.rmdir()
            except Exception:
                pass

    # --- Prompt-injection helpers ---

    def best_attempt(self, source_url: str) -> dict | None:
        """The attempt ``best_attempt_id`` points at, or None when there is nothing to pick.

        One definition, because there were three. ``_compute_best_attempt_id`` RANKS rather than
        filters -- ``pool = successes or attempts`` -- so on a source where every attempt failed this
        returns a FAILURE. Every caller must therefore read ``outcome`` before describing what it is
        holding. Two of the three copies this replaces did not, which is how a rolled-back install
        came to be announced to the model as "Last winning strategy".

        Falls back to the newest attempt when the stored id matches none of them -- a hand-edited or
        partially-migrated record -- which is what the readers this replaces did.
        """
        record = self.read_short_term(source_url) or {}
        attempts = record.get("attempts") or []
        if not attempts:
            return None
        best_id = record.get("best_attempt_id")
        best = next((a for a in attempts if a.get("attempt_id") == best_id), attempts[-1])
        return best if isinstance(best, dict) else None

    def attempt_age_days(self, attempt: dict, record: dict | None = None) -> float | None:
        """Days since this attempt was last verified, or None when nothing dateable is on file.

        None is not zero. Records written before ``last_verified`` existed, and any timestamp that
        does not parse, leave the age genuinely unknown -- so callers must read None as "cannot
        say" and leave the memory alone. Discarding a user's recipe on a guess is worse than the
        stale hint this feeds.
        """
        for stamp in (attempt.get("last_verified"), attempt.get("finished"), (record or {}).get("last_verified")):
            if not isinstance(stamp, str) or not stamp:
                continue
            try:
                then = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - then).total_seconds() / 86400.0
        return None

    def staleness_note(self, attempt: dict, record: dict | None = None) -> str:
        """One sentence naming the age of a recipe past ``cfg.staleness_days``, else "".

        This is the condition ``know_how/memory_user_mcp_tools.md`` gates recipe reuse on. It was
        prose: ``staleness_days`` was the one MemoryConfig field with no consumer, so a 400-day-old
        install was handed over as "what WORKED LAST TIME" with its install path as the first rung
        of the ladder. A non-positive threshold turns the gate off.
        """
        limit = self.cfg.staleness_days
        if not isinstance(limit, int) or limit <= 0:
            return ""
        age = self.attempt_age_days(attempt, record)
        if age is None or age <= limit:
            return ""
        return (
            f"STALE: last verified {int(age)} days ago, past the {limit}-day threshold. Treat the "
            f"details below as history, not as a starting point -- pins and install paths go out of "
            f"date. Run the canonical ladder from the top."
        )

    def format_hint_for_prompt(self, source_url: str) -> str:
        """Format memory hints for safe injection at prompt SUFFIX (H5, L14).

        Returns an empty string in three cases, not two: memory is disabled, no attempts are
        recorded, and -- the one a caller cannot guess -- the best attempt's outcome is neither
        ``full_pass`` nor ``vendor_fallback_pass``. Only a success is injected (L14), so five of the
        seven recorded outcomes yield "" despite having attempts on file.

        Gate on this return value, not on ``read_short_term(url)["attempts"]``. The Phase-1 block in
        ``know_how/memory_user_mcp_tools.md`` gated on the attempt list and announced a hint it had
        not been given for every source whose every attempt failed.
        """
        if _memory_disabled():
            return ""
        mem = self.read_short_term(source_url)
        if not mem or not mem.get("attempts"):
            return ""
        best = self.best_attempt(source_url) or {}
        # Only include recent successful attempts as hint content
        if best.get("outcome") not in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK):
            return ""
        lines = [
            "<prior_attempt>",
            "PRIOR ATTEMPT HINT ONLY. You MUST still run Phase 1 discovery",
            "and canonical ladder from the start. The hint below tells you",
            "what WORKED LAST TIME but does not mean it will work this time",
            "or that you can skip phases.",
            f"  source_url: {_sanitize_text(best.get('source_url') or mem.get('source_url', ''))}",
            f"  last_outcome: {best.get('outcome')}",
            f"  last_verified: {best.get('last_verified')}",
        ]
        note = self.staleness_note(best, mem)
        if note:
            lines.append(f"  {note}")
        strat = best.get("strategy") or {}
        if strat:
            lines.append(f"  winning_install_path: {_sanitize_text(str(strat.get('install_path', '')))}")
            deps = strat.get("key_deps_pinned") or {}
            if deps:
                lines.append("  suggested_pins:")
                for k, v in list(deps.items())[:10]:
                    lines.append(f"    {_sanitize_text(k, max_len=60)}: {_sanitize_text(str(v), max_len=60)}")
        api = best.get("api_shape") or {}
        if api.get("primary_fn"):
            lines.append(f"  hint_primary_fn: {_sanitize_text(str(api['primary_fn']), max_len=80)}")
        gotchas = best.get("gotchas") or []
        for g in gotchas[:5]:
            lines.append(f"  gotcha: {_sanitize_text(g, max_len=200)}")
        lines.append("</prior_attempt>")
        return "\n".join(lines)

    # --- Internal helpers ---

    def _sanitize_record(self, record: dict) -> dict:
        """Scrub strings in a record before persisting."""
        clean: dict[str, Any] = {}
        for k, v in record.items():
            if isinstance(v, str):
                clean[k] = _sanitize_text(v)
            elif isinstance(v, list):
                clean[k] = [_sanitize_text(x) if isinstance(x, str) else x for x in v]
            elif isinstance(v, dict):
                clean[k] = self._sanitize_record(v)
            else:
                clean[k] = v
        return clean

    def _apply_cap(self, attempts: list[dict]) -> list[dict]:
        """FIFO cap with keep-best pin (L8)."""
        if len(attempts) <= self.cfg.max_attempts_per_tool:
            return attempts
        # Separate into successes and failures
        successes = [
            a
            for a in attempts
            if a.get("outcome") in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK, OUTCOME_SOFT_DEGRADED)
        ]
        failures = [a for a in attempts if a not in successes]
        pins = []
        if successes:
            pins.append(successes[-1])  # most recent success
        if failures:
            pins.append(failures[-1])  # most recent failure
        pin_ids = {id(p) for p in pins}
        rest = [a for a in attempts if id(a) not in pin_ids]
        keep_count = self.cfg.max_attempts_per_tool - len(pins)
        kept = rest[-keep_count:] if keep_count > 0 else []
        return kept + pins  # order by (old-FIFO) + (pins)

    def _compute_best_attempt_id(self, attempts: list[dict]) -> int | None:
        """L5 — tiebreaker: recent success > shortest time > lowest id."""
        if not attempts:
            return None
        successes = [
            a
            for a in attempts
            if a.get("outcome") in (OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK, OUTCOME_SOFT_DEGRADED)
        ]
        pool = successes or attempts
        pool_sorted = sorted(
            pool,
            key=lambda a: (
                -self._parse_ts(a.get("finished", "")),
                a.get("time_to_outcome_sec") or float("inf"),
                a.get("attempt_id") or 9999,
            ),
        )
        return pool_sorted[0].get("attempt_id")

    @staticmethod
    def _parse_ts(ts: str) -> float:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    def _update_tool_id_history(self, prior: list[str], tool_id: str | None) -> list[str]:
        if not tool_id:
            return prior
        if tool_id in prior:
            return prior
        return prior + [tool_id]

    def _derive_unavailable_reason(self, attempts: list[dict]) -> str | None:
        """If ALL attempts are rolled_back or hung, mark unavailable."""
        if not attempts:
            return None
        if all(a.get("outcome") in (OUTCOME_ROLLED_BACK, OUTCOME_HUNG) for a in attempts):
            return "all_attempts_failed"
        return None

    # --- Per-turn cache invalidation (H1) ---

    def invalidate_cache(self) -> None:
        self._snapshot_cache.clear()
