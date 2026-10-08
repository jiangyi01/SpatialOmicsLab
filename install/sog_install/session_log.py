"""
Append-only session log with hard secret masking.

Every run writes a human-readable JSONL transcript to
``.sog_setup/logs/run-<id>.jsonl`` so an interrupted run can be understood and
resumed. The log is also the *safety net* for secrets: any value registered via
:func:`register_secret` is redacted from every line and every console echo, so a
key can never leak into the transcript even if a caller passes it by mistake.

Stdlib only.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

# --------------------------------------------------------------------------- #
# Secret masking
# --------------------------------------------------------------------------- #
# Registered raw secret values, longest-first so overlapping secrets redact
# fully. Populated as the wizard reads keys; never itself logged or persisted.
_SECRETS: list[str] = []
# Guards _SECRETS so register_secret()'s append+sort can't race a concurrent redact() from the
# ThoughtBox animator daemon (which would otherwise iterate a transiently-unsorted / mid-mutation
# list — a redaction gap in the secret safety net).
#
# REENTRANT (N10): the SIGINT handler (wizard._on_sigint) prints its "interrupted…" line through
# redact() to keep secrets masked even on Ctrl-C. A Python signal handler runs on the MAIN thread,
# synchronously interrupting whatever it was doing — which may be inside register_secret()/redact()
# already holding this lock. A plain Lock would then self-deadlock (the same thread blocking on a
# lock it already owns), hanging the Ctrl-C instead of exiting 130. An RLock lets the same thread
# re-acquire, so the handler's redact() proceeds; it keeps the exact cross-thread guarantee above
# (a DIFFERENT thread still blocks until the holder releases).
_SECRETS_LOCK = threading.RLock()

# Field-name *words* whose value is always masked regardless of registration. Matched as
# whole tokens (see _looks_secretish), never substrings — so "pat" masks GITHUB_PAT but not
# the very common innocent field "path".
_SECRETISH_KEYS = frozenset(("key", "token", "secret", "password", "pat", "authorization"))

# Field-name separators: split "api_key"/"custom-base.url"/"GITHUB PAT" into words before matching.
_KEY_SEPARATORS = str.maketrans("_-./", "    ")

_MASK = "****"

# A value shorter than this cannot be a real credential (API keys, tokens, and endpoints are all
# far longer) and registering it would over-redact ordinary prose — a stray 1-char "k" would mask
# the letter from every word, a 3-char value would punch holes in unrelated log lines. write_dotenv
# registers *every* value it writes (model name, provider, region) so short non-secret bookkeeping
# strings reach here; the guard keeps the safety net from corrupting the transcript. Matches the
# >= 4 floor already used for URL-embedded passwords in _url_credentials.
_MIN_SECRET_LEN = 4


def register_secret(value: str | None) -> None:
    """Remember a raw secret so it is redacted everywhere it might appear.

    Values shorter than :data:`_MIN_SECRET_LEN` are ignored: they can't be real credentials and
    would over-redact ordinary text (see the constant's note). Display-side masking of short values
    is unaffected — :func:`mask_secret` still stars them when a caller explicitly renders one.
    """
    if not value or not isinstance(value, str) or len(value) < _MIN_SECRET_LEN:
        return
    with _SECRETS_LOCK:
        if value in _SECRETS:
            return
        _SECRETS.append(value)
        _SECRETS.sort(key=len, reverse=True)


# pip / conda / uv read these environment variables as their (optionally credentialed) package
# index sources; a ``https://user:token@host`` value echoed in build stderr would otherwise leak
# raw into the build-log FILE sinks (C1).
_INDEX_URL_ENV_VARS = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "CONDA_CHANNEL_ALIAS",
)


def _url_credentials(url: str) -> list[str]:
    """The credential substrings to register from one possibly-credentialed index URL.

    Returns the password (only when long enough to be a real token, so a 1–3 char value can't
    over-redact ordinary prose) and the full ``user:pass`` userinfo as it appears inline in the raw
    URL; empty when the URL carries no password. Never raises — a non-URL string yields nothing."""
    try:
        parts = urlsplit(url.strip())
        pwd = parts.password or ""
        user = parts.username or ""
    except (ValueError, AttributeError):
        return []
    if not pwd:
        return []
    out: list[str] = []
    if len(pwd) >= 4:
        out.append(pwd)
    if user:
        out.append(f"{user}:{pwd}")
    return out


def register_index_url_secrets(env: Mapping[str, str] | None = None, *, extra: Iterable[str] = ()) -> None:
    """Register the credentials embedded in any configured pip/conda/uv index URL as secrets (C1).

    So a ``https://user:token@host`` index URL is masked by :func:`redact` wherever it is echoed —
    crucially including the raw build-log FILE sinks (``provision._persist_build_log`` /
    ``envtools.Conda._tee_line``) that pip/conda stderr flows into. Reads the standard index-URL
    environment variables (``env`` defaults to :data:`os.environ`) plus any ``extra`` URL strings a
    caller loaded from ``.env``; each value may be a whitespace-separated list of URLs. Best-effort
    and never raises — a malformed value is simply skipped."""
    values = env if env is not None else os.environ
    raw: list[str] = []
    for name in _INDEX_URL_ENV_VARS:
        raw.extend((values.get(name) or "").split())
    for item in extra:
        raw.extend((item or "").split())
    for url in raw:
        for secret in _url_credentials(url):
            register_secret(secret)


def mask_secret(value: str | None, keep: int = 4) -> str:
    """Render a secret for display as ``****…last4`` (never the raw value).

    Short values (<= ``keep`` visible chars would reveal too much) are fully
    starred. ``None``/empty renders as ``(unset)``.
    """
    if value is None or value == "":
        return "(unset)"
    s = str(value)
    if len(s) <= keep + 2:
        return _MASK
    return f"{_MASK}…{s[-keep:]}"


def redact(text: str) -> str:
    """Replace every registered secret substring in ``text`` with the mask."""
    if not text:
        return text
    with _SECRETS_LOCK:
        # Snapshot under the lock so this iteration never runs over the list while
        # register_secret() is mid append+sort (the ThoughtBox animator daemon calls redact()
        # concurrently) — otherwise a secret could momentarily slip through unmasked.
        secrets = tuple(_SECRETS)
    if not secrets:
        return text
    out = text
    for secret in secrets:
        if secret and secret in out:
            out = out.replace(secret, f"{_MASK}…{secret[-4:]}" if len(secret) > 6 else _MASK)
    return out


def notice(message: str) -> None:
    """Print a friendly, secret-redacted one-line ``[sog-setup]`` notice to stderr.

    The graceful-degradation channel for the low-level loaders (:mod:`specs`, :mod:`state`,
    :mod:`categories`): surface a plain-language message instead of a raw traceback when a
    shipped file is missing or corrupt, without threading a :class:`SessionLog` instance down to
    module scope. Redacted like every other output.
    """
    print(f"[sog-setup] {redact(str(message))}", file=sys.stderr)


def _looks_secretish(key: str) -> bool:
    """True if a field *name* implies its value is a credential.

    Matches on whole separator-delimited tokens, NOT substrings: ``GITHUB_PAT`` /
    ``api_key`` / ``authorization`` mask, but an innocent ``path`` no longer trips the
    old ``"pat" in "path"`` substring bug that masked every logged file path.
    """
    tokens = key.lower().translate(_KEY_SEPARATORS).split()
    return any(tok in _SECRETISH_KEYS for tok in tokens)


def _sanitize(obj: Any) -> Any:
    """Deep-redact a JSON-able structure before it hits disk or the console."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for k, v in obj.items():
            if _looks_secretish(str(k)):
                # A secret-ish field name masks its WHOLE value, not just a scalar one (E-F1): a
                # list/dict under `token=`/`authorization=`/… must not fall through to `_sanitize`,
                # which only masks *registered* secrets and would round-trip an unregistered one to
                # disk raw. Serialize a container first so `mask_secret` (str-only) can redact it.
                clean[k] = mask_secret(v if isinstance(v, str) else json.dumps(v, default=str))
            else:
                clean[k] = _sanitize(v)
        return clean
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    # JSON-native scalars (int/float/bool — bool is an int subclass — and None) are secret-free, so
    # keep them as-is. ANYTHING ELSE (an Exception, a Path, a dataclass) would otherwise be returned
    # untouched and then str()-coerced by ``json.dumps(default=str)`` in :meth:`SessionLog.event`
    # AFTER this function returns — bypassing redaction and round-tripping a registered secret in its
    # ``repr`` to the transcript raw. That is exactly the "even if a caller passes it by mistake" leak
    # this module's docstring promises to prevent (a natural ``log.event("failed", error=exc)`` where
    # ``str(exc)`` embeds the key). Coerce+redact here so ``json.dumps`` only ever sees a clean,
    # already-redacted string; ``default=str`` degrades to a pure belt-and-braces backstop.
    if obj is None or isinstance(obj, (int, float)):
        return obj
    return redact(str(obj))


# --------------------------------------------------------------------------- #
# SessionLog
# --------------------------------------------------------------------------- #
class SessionLog:
    """Append-only JSONL transcript with optional console echo.

    Parameters
    ----------
    path:
        Destination JSONL file. Parent dirs are created on first write.
    echo:
        When True (default), :meth:`console` also prints to stdout.
    """

    def __init__(self, path: str | os.PathLike[str], echo: bool = True) -> None:
        self.path = Path(path)
        self.echo = echo
        self._fh = None
        self._degraded = False  # set once the log sink becomes unwritable (best-effort thereafter)

    # -- lifecycle ------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                # A disk-full / broken-pipe flush at teardown must not raise out of ``__exit__``
                # and flip an otherwise-green run — the record is already best-effort by then.
                pass
            finally:
                self._fh = None

    def __enter__(self) -> SessionLog:
        self._ensure_open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writing --------------------------------------------------------------
    def event(self, kind: str, **fields: Any) -> None:
        """Append one JSONL record ``{ts, kind, **fields}`` (secrets redacted).

        Best-effort: the run log is a side-channel, so a filesystem failure here — a full disk, a
        read-only/vanished log dir (``_ensure_open``'s mkdir+open), a broken pipe or a closed handle
        mid-write — must NEVER unwind past a phase handler and flip an otherwise-green install to a
        scary ``exit 1``. On the first such failure we mark the log degraded, emit ONE guarded stderr
        notice, and drop this and every later record silently. The record is built before the ``try``
        so a redaction/serialisation bug still surfaces loudly (it is a real defect, not an I/O hiccup)."""
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind}
        record.update(_sanitize(fields))
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        try:
            self._ensure_open()
            if self._fh is None:  # _ensure_open never returns with _fh unset, but never raise if it did
                return
            self._fh.write(line)
            self._fh.flush()
        except (OSError, ValueError) as exc:
            # OSError: disk full / read-only FS / broken pipe. ValueError: write to a closed handle
            # during an out-of-order teardown. Either way the sink is unusable — degrade, don't crash.
            self._note_degraded(exc)

    def _note_degraded(self, exc: BaseException) -> None:
        """Record that the log sink went unwritable; warn once, on stderr, itself guarded."""
        if self._degraded:
            return
        self._degraded = True
        try:
            print(f"[setup] run-log write failed ({exc}); continuing without a durable log.", file=sys.stderr)
        except Exception:
            pass

    def console(self, message: str, *, kind: str = "console", to_stderr: bool = False) -> None:
        """Log ``message`` and (if echo) print it — redacted both ways."""
        safe = redact(message)
        self.event(kind, message=safe)
        if self.echo:
            print(safe, file=sys.stderr if to_stderr else sys.stdout)

    def phase(self, name: str, status: str, **fields: Any) -> None:
        self.event("phase", phase=name, status=status, **fields)

    def warn(self, message: str) -> None:
        self.console(message, kind="warn", to_stderr=True)

    def error(self, message: str) -> None:
        self.console(message, kind="error", to_stderr=True)


class DryRunSessionLog(SessionLog):
    """A :class:`SessionLog` that echoes to the console but never opens a file or creates a dir.

    ``--dry-run`` (both the wizard's plan preview and ``sog-setup reset --dry-run``) is a read-only
    preview: the user still sees every ``console``/``warn``/``error`` line, but no ``*.jsonl``
    transcript is written and no ``.sog_setup`` directory is created. Overriding ``_ensure_open`` and
    ``event`` to no-ops keeps the whole logging surface intact — ``console`` still prints because it
    calls ``event`` (now inert) *and then* ``print`` — while making the sink write-free. Shared so
    both dry-run contracts ("change nothing") are honoured by ONE implementation (wizard N4 / reset R20).
    """

    def _ensure_open(self) -> None:  # never open a file / create .sog_setup
        return

    def event(self, kind: str, **fields: Any) -> None:  # drop the JSONL record
        return
