"""
LLM validation (Tier-1) and safe ``.env`` writing.

Two responsibilities, both stdlib-only so they run before any env exists:

* **Tier-1 validation** — a per-provider ``urllib`` REST ping that proves the
  key/endpoint work *now*, without importing langchain (which lives in the env
  being built). Tier-2 (a real ``get_llm().invoke``) runs later via
  :mod:`sog_install._llm_ping` once the base env exists.

* **Safe ``.env`` merge** — the only file shared with the agent. Every write is
  backed up first, is a parse-preserving merge (owned keys only, comments/order
  kept), de-duplicates repeated keys (the live ``.env`` duplicates
  ``OPENAI_API_KEY``), and migrates stale ``BIOMNI_*``/``CUSTOM_MODEL_*`` names
  the current code no longer reads. Secrets are never echoed or logged.
"""

from __future__ import annotations

import http.client
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from spatialomicsgym.provider_names import canonical_source

from . import constants
from .credentials import (
    STALE_BY_OLD,
    ProviderSpec,
    azure_deployment_from_endpoint,
    azure_resource_endpoint,
    get_provider,
)
from .session_log import _looks_secretish, notice, register_secret


# --------------------------------------------------------------------------- #
# Tier-1 REST ping
# --------------------------------------------------------------------------- #
@dataclass
class PingResult:
    ok: bool  # provider is usable as configured
    reachable: bool = False  # endpoint answered at all
    auth_ok: bool | None = None  # None = not determinable (e.g. presence-only)
    status: int | None = None  # HTTP status if any
    detail: str = ""  # human-readable outcome
    presence_only: bool = False  # True for Bedrock (no live call)


#: The only handlers a provider ping is built with. ``urlopen``'s default opener also carries
#: file:, ftp: and data: handlers, so a Custom base URL of ``file:///etc`` "validated" as a working
#: key and answered a file-existence oracle; and its redirect handler follows a 302 to ftp:. Without
#: those handlers a non-HTTP URL, first hop or redirect, is "unknown url type" (hunt 2026-09-30,
#: uL1-security-4). Loopback stays reachable on purpose: Custom and Ollama are local servers.
_PING_HANDLERS = (
    urllib.request.ProxyHandler,
    urllib.request.UnknownHandler,
    urllib.request.HTTPHandler,
    urllib.request.HTTPSHandler,
    urllib.request.HTTPDefaultErrorHandler,
    urllib.request.HTTPRedirectHandler,
    urllib.request.HTTPErrorProcessor,
)


def _http_get(url: str, headers: dict[str, str], timeout: int) -> tuple[int, bytes]:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {scheme or '(none)'!r} -- use an http:// or https:// endpoint")
    opener = urllib.request.OpenerDirector()
    for handler in _PING_HANDLERS:
        opener.add_handler(handler())
    req = urllib.request.Request(url, headers=headers, method="GET")
    with opener.open(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _field_with_a_control_char(provider: ProviderSpec, values: dict[str, str]) -> str | None:
    """The first of ``provider``'s fields whose value carries a line break or other control
    character, or ``None``.

    http.client refuses such a header value with ``ValueError("Invalid header value b'<the key>'")``
    -- the whole key, in the exception text. That text used to become the ping's detail, which
    onboarding prints and the run log keeps: a CI secret with a trailing newline was written out in
    full, and blamed on the network (hunt 2026-09-30, u35-setup-ux-1). Checked before any request is
    built, so the refusal names the field and never the value."""
    for f in provider.all_fields():
        val = values.get(f.env_var)
        if isinstance(val, str) and any(ord(c) < 32 or ord(c) == 127 for c in val):
            return f.env_var
    return None


def _scrub(text: str, headers: dict[str, str]) -> str:
    """``text`` with every header value we sent masked -- the belt behind the control-char check, for
    any other exception that quotes a header (hunt 2026-09-30, u35-setup-ux-1)."""
    for val in headers.values():
        # The value as sent, stripped, and without an auth scheme ("Bearer <key>" -> "<key>").
        for form in sorted({val, val.strip(), val.split(" ", 1)[-1].strip()}, key=len, reverse=True):
            if len(form) >= 4:
                text = text.replace(form, "****")
    return text


def tier1_ping(
    provider: ProviderSpec,
    values: dict[str, str],
    *,
    timeout: int = constants.LLM_PING_TIMEOUT_SEC,
) -> PingResult:
    """Validate a provider configuration over plain HTTP.

    ``values`` maps the provider's env vars to the values the user just entered.
    Distinguishes *unreachable* (network/DNS/refused) from *auth failure*
    (401/403) so onboarding can give an actionable message.
    """
    ping = provider.ping
    if not ping.supported:
        # Bedrock: presence-only. Usable iff region + some credential are present.
        region = values.get("AWS_REGION")
        has_cred = bool(
            values.get("AWS_BEARER_TOKEN_BEDROCK")
            or (values.get("AWS_ACCESS_KEY_ID") and values.get("AWS_SECRET_ACCESS_KEY"))
        )
        ok = bool(region and has_cred)
        return PingResult(
            ok=ok,
            reachable=False,
            auth_ok=None,
            presence_only=True,
            detail=(
                "credentials present (not live-validated — Bedrock uses SigV4)"
                if ok
                else "need AWS_REGION plus a bearer token or an access-key pair"
            ),
        )

    bad_field = _field_with_a_control_char(provider, values)
    if bad_field is not None:
        return PingResult(
            ok=False,
            detail=f"{bad_field} contains a line break or other control character -- re-enter it without one",
        )

    # Resolve the URL + auth headers for this provider.
    url = ping.url
    headers = {h[0]: h[1] for h in ping.extra_headers}
    try:
        if ping.auth == "x-api-key":  # Anthropic
            headers["x-api-key"] = values[provider.required[0].env_var]
        elif ping.auth == "bearer":  # OpenAI / Gemini / Groq / Custom
            key = (
                values.get("OPENAI_API_KEY")
                or values.get("GEMINI_API_KEY")
                or values.get("GROQ_API_KEY")
                or values.get("SOG_CUSTOM_API_KEY")
            )
            if "{base_url}" in url:  # Custom
                base = values["SOG_CUSTOM_BASE_URL"].rstrip("/")
                url = url.replace("{base_url}", base)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        elif ping.auth == "api-key-header":  # Azure
            # Through the shared helper, not a third hand-rolled strip: the two hand-rolled ones
            # destroyed the endpoint of any resource NAMED `openai...`.
            endpoint = azure_resource_endpoint(values.get("OPENAI_ENDPOINT") or "")
            version = values.get("OPENAI_API_VERSION") or "2024-12-01-preview"
            url = url.replace("{endpoint}", endpoint) + f"?api-version={version}"
            headers["api-key"] = values.get("OPENAI_API_KEY", "")
        # ping.auth == "none" (Ollama): no headers
    except KeyError as exc:
        return PingResult(ok=False, detail=f"missing required field {exc}")

    if not url:
        return PingResult(ok=False, detail="no validation endpoint configured")

    try:
        status, _ = _http_get(url, headers, timeout)
        return PingResult(ok=True, reachable=True, auth_ok=True, status=status, detail="key works")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return PingResult(
                ok=False,
                reachable=True,
                auth_ok=False,
                status=exc.code,
                detail="authentication rejected — check the key/endpoint",
            )
        if exc.code == 404 and provider.source in ("Custom", "AzureOpenAI"):
            # Custom: some OpenAI-compatible servers lack /models but are otherwise fine.
            # Azure: the data-plane /models route can be absent/blocked (an APIM gateway in
            # front, restricted networking, or a non-standard api-version), yet the resource
            # is fine — the real client uses the per-deployment chat/completions route, never
            # this one, so its deployment is validated on the first real call. A wrong host
            # (URLError) and a bad key (401/403) are still caught above, so a 404 here means
            # only that this probe route is unavailable — not a misconfiguration. Don't block.
            detail = (
                "reachable; couldn't confirm via the models route (your deployment is validated on the first real call)"
                if provider.source == "AzureOpenAI"
                else "reachable (no /models route; assuming OK)"
            )
            return PingResult(ok=True, reachable=True, auth_ok=None, status=404, detail=detail)
        return PingResult(ok=False, reachable=True, status=exc.code, detail=f"endpoint returned HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
        # A malformed endpoint (schemeless like ``res.openai.azure.com`` or a non-HTTP scheme →
        # ``ValueError`` from ``_http_get``; a bad port or embedded newline → ``http.client.InvalidURL``,
        # an HTTPException that is NOT a ValueError) must read as "unreachable, fix the endpoint" — not
        # abort the wizard with a raw traceback at exit 1. (SETUP-2c) The URL is NOT the only ValueError
        # source: http.client also raises one for a bad header value and quotes the value, i.e. the
        # key. The control-char check above stops that case; ``_scrub`` masks anything a header still
        # leaks into the text (hunt 2026-09-30, u35-setup-ux-1).
        reason = getattr(exc, "reason", exc)
        return PingResult(ok=False, reachable=False, detail=f"could not reach endpoint: {_scrub(str(reason), headers)}")


# --------------------------------------------------------------------------- #
# .env parsing / merge
# --------------------------------------------------------------------------- #
@dataclass
class _Line:
    kind: str  # blank | comment | kv | other
    raw: str
    key: str | None = None
    export: bool = False


@dataclass
class MergeReport:
    migrated: list[tuple[str, str]] = field(default_factory=list)  # (old, new)
    deduped: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    appended: list[str] = field(default_factory=list)
    dropped_stale: list[str] = field(default_factory=list)


def _parse_lines(text: str) -> list[_Line]:
    out: list[_Line] = []
    # Split on NEWLINE only -- NOT str.splitlines(), which ALSO breaks on \v \f \x1c \x1d \x1e \x85
    # U+2028 U+2029. Those are boundary chars `_format_kv` never escapes (it neutralises only \n/\r), so
    # a value carrying one would be TRUNCATED on read and its tail injected as a spurious `KEY=VALUE`
    # (the SETUP-1c / a93#1 corruption class, for the residual boundary chars a93#1 didn't cover -- e.g.
    # a copy-pasted key with a NEL \x85). `\n` + a stripped trailing `\r` reproduces splitlines() for the
    # only line endings the writer (or any normal editor) emits -- `\n`, `\r\n` -- while leaving an
    # exotic char intact INSIDE the value. Dropping one trailing empty keeps splitlines() parity
    # (a trailing newline does not create an empty final line).
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for raw in lines:
        if raw.endswith("\r"):
            raw = raw[:-1]
        s = raw.strip()
        if not s:
            out.append(_Line("blank", raw))
        elif s.startswith("#"):
            out.append(_Line("comment", raw))
        elif "=" in s:
            lhs = s.split("=", 1)[0].strip()
            export = lhs.startswith("export ")
            key = lhs[len("export ") :].strip() if export else lhs
            if key.replace("_", "").isalnum():
                out.append(_Line("kv", raw, key=key, export=export))
            else:
                out.append(_Line("other", raw))
        else:
            out.append(_Line("other", raw))
    return out


def _format_kv(key: str, value: str, export: bool = False) -> str:
    # Quote if the value has whitespace or shell-special chars. Two styles, because the agent + the
    # Tier-2 ping read `.env` through **python-dotenv**, not a shell:
    #
    #   * `$`/backtick in a DOUBLE-quoted value is wrong for that reader (SETUP-1c). python-dotenv
    #     INTERPOLATES `$VAR` inside double quotes, and its escape table omits `\$`/`` \` ``, so an
    #     escaped `\$` reaches the agent with a STRAY BACKSLASH — a real custom/vLLM key like
    #     `p@ss$word` (a `$` is common in generated strong passwords) silently corrupts, yet Tier-1
    #     stays green because it validates the in-memory value, not the on-disk one. A SINGLE-quoted
    #     value is a shell/dotenv literal — no interpolation, no escape decoding — so emit those
    #     verbatim whenever the value carries a `$`/backtick but no single-quote or newline (which
    #     single-quoting can't represent). :func:`_decode_dotenv_value` reads a single-quoted value
    #     back verbatim, keeping the setup-side reader in agreement.
    #   * Otherwise DOUBLE-quote. Escape backslash FIRST, then `"`, then control chars — the important
    #     one (a93#1) being that an embedded newline must be `\n`-escaped or it would be written as a
    #     SECOND physical line, truncating the value AND injecting whatever followed as a spurious
    #     `KEY=VALUE`. We deliberately DO NOT escape `$`/backtick here (MY-2): python-dotenv can't
    #     decode `\$` (SETUP-1c), and the ONLY way a `$`/backtick value reaches this double-quote
    #     branch is if it ALSO carries a `'` or newline (else the single-quote branch above claimed it),
    #     so a *raw* `$` is what python-dotenv reads back correctly — an escaped `\$` would leave the
    #     same stray backslash SETUP-1c set out to kill. `_dq_unescape` still DECODES `\$`/`` \` `` on
    #     read so a `.env` written by the pre-MY-2 code keeps round-tripping. (Residual, irreducible:
    #     a value that is BOTH `$<an-existing-env-var>` AND single-quote-bearing can't be written
    #     python-dotenv-literally at all — double quotes interpolate, single quotes can't hold the `'`.)
    if value and ("$" in value or "`" in value) and not any(c in value for c in "'\n\r"):
        value = "'" + value + "'"
    elif value == "" or any(c in value for c in " \t\"'#$\\`\n\r"):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        value = '"' + escaped + '"'
    return f"{'export ' if export else ''}{key}={value}"


# Reverse table for :func:`_dq_unescape` — a superset of :func:`_format_kv`'s escape chain that also
# read-decodes escapes python-dotenv honors but _format_kv never WRITES, so a hand-edited value decodes
# the SAME way the agent's python-dotenv reader sees it (round-trip fidelity — see _decode_dotenv_value).
# A backslash before one of these keys decodes to the mapped char: ``\n``/``\r``/``\t`` to a real
# newline/CR/TAB, the rest to the literal char. ``$``/`` ` `` are legacy read-compat (SETUP-1c); ``\t`` is
# python-dotenv-compat (_format_kv emits a RAW tab, never ``\t``, so this never breaks a written value's
# round-trip — a ``\\t`` it wrote still decodes ``\\``→``\`` then a literal ``t``).
_DQ_UNESCAPE = {'"': '"', "\\": "\\", "$": "$", "`": "`", "n": "\n", "r": "\r", "t": "\t"}


def _dq_unescape(s: str) -> str:
    """Reverse :func:`_format_kv`'s double-quote escaping: ``\\\\``→``\\``, ``\\"``→``"``, ``\\$``→``$``,
    `` \\` ``→`` ` ``, ``\\n``→newline, ``\\r``→CR.

    Single left-to-right pass so an escaped backslash immediately before a quote (``\\\\"``) is not
    mis-parsed as an escaped quote. Only applied to values we wrote as double-quoted; single-quoted
    values keep shell literal semantics (never unescaped)."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == "\\" and i + 1 < n and s[i + 1] in _DQ_UNESCAPE:
            out.append(_DQ_UNESCAPE[s[i + 1]])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


# A ``.env`` value that OPENS with a quote may be FOLLOWED by a whitespace-preceded inline ``# comment``
# that python-dotenv strips (``KEY="v" # note`` → ``v``). Match the quoted body (``\"`` does not close a
# double-quoted value), then require the remainder to be only whitespace + an optional comment.
_DQ_VALUE_RE = re.compile(r'"((?:\\.|[^"\\])*)"')  # double-quoted body, honoring \" escapes
_SQ_VALUE_RE = re.compile(r"'([^']*)'")  # single-quoted body (shell-literal — no escapes)
_AFTER_CLOSE_QUOTE_RE = re.compile(r"\s*(#.*)?\Z", re.S)  # ws + optional inline comment after close-quote


def _decode_dotenv_value(raw_val: str) -> str:
    """Decode a raw ``.env`` RHS (everything after the first ``=``) to its logical value: strip a
    *matched* surrounding quote pair and, for a double-quoted value, reverse :func:`_format_kv`'s
    escaping (``\\\\``/``\\"``/``\\$``); a single-quoted value keeps shell-literal semantics.

    The single decode used by BOTH :func:`read_dotenv_values` and :func:`merge_dotenv`'s stale-name
    migration, so a read and a rename decode a value identically. Before this was shared, migration
    did ``.strip('"')`` with no unescape (F-sec-2 / N14 gap): a double-quoted stale value containing
    ``$``/``\\``/``"`` was re-escaped by ``_format_kv`` on rename → double-escaped → corrupt on the
    next read; and single-quoted / unmatched-quote values were mangled by the blunt ``strip('"')``."""
    raw_val = raw_val.strip()
    # A value that OPENS with a quote: take the properly-closed quoted body and DISCARD a trailing
    # whitespace-preceded ``# comment`` (python-dotenv does the same). This MUST run before the old
    # ``raw_val[0] == raw_val[-1]`` matched-pair test, which failed whenever a ``# comment`` followed the
    # closing quote — the last char is then comment text, not a quote, so the value kept its surrounding
    # quotes: ``KEY="sk-ant-key" # note`` decoded to ``"sk-ant-key"`` (quoted). That quoted form is the
    # exact N8/N14 double hazard the unquoted branch below already guards — a redaction MISS (the quoted
    # string is registered but the agent's python-dotenv reads the BARE key, so ``redact()`` can't match)
    # AND a reuse-corruption 401 (``_format_kv`` re-quotes the ``"``-bearing value → ``"\"sk-ant-key\""``,
    # which python-dotenv then reads literally). Matching python-dotenv here closes both.
    if raw_val[:1] in ('"', "'"):
        m = (_DQ_VALUE_RE if raw_val[0] == '"' else _SQ_VALUE_RE).match(raw_val)
        if m and _AFTER_CLOSE_QUOTE_RE.fullmatch(raw_val, m.end()):
            inner = m.group(1)
            return _dq_unescape(inner) if raw_val[0] == '"' else inner
        # An unmatched / mid-value quote (``"a"b"``, ``no_close"x``) is malformed — fall through to the
        # unquoted branch (python-dotenv drops such a line entirely; we keep the raw text, which is a
        # redaction superset and never a real key, so it can't leak or 401).
    # UNQUOTED value: python-dotenv strips a *whitespace-preceded* inline `# comment` before yielding
    # the value (parser.py ``parse_unquoted_value``: ``re.sub(r"\s+#.*$", "", value).rstrip()``). We
    # MUST decode identically. If we don't: a value the agent reads via python-dotenv as `sk-ant-key`
    # is read here as `sk-ant-key # note`; the reuse/persist path then hands that to ``_format_kv``,
    # which quotes it (space + ``#`` trip the quote predicate at :236) and rewrites the `.env` line as
    # ``KEY="sk-ant-key # note"`` — and python-dotenv reads a *quoted* value LITERALLY (comment kept),
    # so a previously-working credential silently becomes `sk-ant-key # note` → 401 on the next call
    # (and ``read_dotenv_values`` would register the comment-suffixed string for redaction, letting the
    # real key slip past ``redact()``). A ``#`` with no preceding whitespace (``p@ss#word``) and any
    # quoted ``#`` are preserved, exactly as python-dotenv does.
    return re.sub(r"\s+#.*$", "", raw_val).rstrip()


def merge_dotenv(text: str, updates: dict[str, str], *, migrate: bool = True) -> tuple[str, MergeReport]:
    """Parse-preserving merge.

    Order of operations: migrate stale names → de-dup repeats (last wins) →
    update-or-append owned keys. Any key not owned (not in ``updates``, not
    stale) is left byte-for-byte untouched.
    """
    report = MergeReport()
    lines = _parse_lines(text)

    # --- 1. migrate stale names (rename the line, keep the value) -----------
    present = {ln.key for ln in lines if ln.kind == "kv"}
    if migrate:
        for ln in lines:
            if ln.kind != "kv" or ln.key not in STALE_BY_OLD:
                continue
            new = STALE_BY_OLD[ln.key].new
            old = ln.key
            if new is None:
                continue
            value = _decode_dotenv_value(ln.raw.split("=", 1)[1])
            if new in updates or new in present:
                # Modern name is being set (or already exists) — drop the stale line.
                ln.kind, ln.raw, ln.key = "drop", "", None
                report.dropped_stale.append(old)
            else:
                ln.key, ln.export = new, ln.export
                # decode → re-encode so a double-quoted / escaped stale value round-trips intact.
                ln.raw = _format_kv(new, value, ln.export)
                present.add(new)
                report.migrated.append((old, new))

    # --- 2. de-dup repeated keys (keep the LAST occurrence's line) ----------
    last_index: dict[str, int] = {}
    for i, ln in enumerate(lines):
        if ln.kind == "kv":
            last_index[ln.key] = i
    for i, ln in enumerate(lines):
        if ln.kind == "kv" and last_index.get(ln.key) != i:
            ln.kind, ln.raw = "drop", ""
            if ln.key not in report.deduped:
                report.deduped.append(ln.key)

    # --- 3. update-or-append owned keys -------------------------------------
    surviving = {ln.key: ln for ln in lines if ln.kind == "kv"}
    for key, value in updates.items():
        if key in surviving:
            ln = surviving[key]
            new_raw = _format_kv(key, value, ln.export)
            if new_raw != ln.raw:
                ln.raw = new_raw
                report.updated.append(key)
        else:
            lines.append(_Line("kv", _format_kv(key, value), key=key))
            report.appended.append(key)

    body = "\n".join(ln.raw for ln in lines if ln.kind != "drop")
    if body and not body.endswith("\n"):
        body += "\n"
    return body, report


# --------------------------------------------------------------------------- #
# Safe .env write
# --------------------------------------------------------------------------- #
def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            # Durability (a1c#3, mirrors state._atomic_write_json's #95c): force the bytes to disk BEFORE
            # the atomic rename, so a crash/power-loss right after os.replace can't leave the rename
            # pointing at an unflushed (zero-length/torn) file. This writer backs ``.env`` — the user's
            # KEYS — and the setup MCP config, so a torn write here is worse than a torn state file.
            # fsync is unsupported on some filesystems (tmpfs/NFS), so it degrades to best-effort.
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, str(path))
    except BaseException:
        # BaseException (a515#2): a Ctrl-C mid-write must still remove the temp — critically here it is a
        # 0600 ``.env.<rand>`` holding the freshly-merged secrets, which must not be left in the repo root.
        Path(tmp).unlink(missing_ok=True)
        raise


def backup_dotenv(path: Path | None = None) -> Path | None:
    """Copy the current ``.env`` to ``.sog_setup/backups/.env.<ts>``. No-op if absent."""
    src = path or constants.dotenv_path()
    if not src.exists():
        return None
    constants.ensure_state_dirs()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backups = constants.backups_dir()
    # Claim a unique backup name ATOMICALLY. The old check-then-act (a `while dst.exists()` loop, then a
    # separate copy) had a TOCTOU: two backups resolving in the same wall-clock second — overlapping runs /
    # a shared NFS home, or several write_dotenv sites firing close together in one run (key write,
    # MCP-config pointer, knobs, the demo key sync) — could both resolve `.env.<ts>.1`, and the later copy
    # would silently OVERWRITE the earlier secret-bearing snapshot. O_CREAT|O_EXCL makes the name-claim
    # win-or-retry (a loser bumps the suffix), exactly like mcp_resolver._unique_backup_path and
    # state.archive_state guard their siblings. Creating the fd 0600 up front also subsumes the old
    # post-copy chmod (N11): the backup is never even briefly world-readable, on any umask.
    n = 0
    while True:
        dst = backups / (f".env.{ts}" if n == 0 else f".env.{ts}.{n}")
        try:
            fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            n += 1
            if n > 100000:  # pathological collision storm — never spin forever
                raise
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(src.read_bytes())
    except OSError:
        # Best-effort snapshot: on a write failure don't leave a 0-byte claim masquerading as a backup.
        try:
            os.unlink(dst)
        except OSError:
            pass
        raise
    return dst


def _read_text_lenient(target: Path) -> str:
    """Read ``target`` as text, tolerating a non-UTF-8 ``.env``.

    ``.env`` is the one file users routinely hand-edit (``cp .env.example .env``), so a byte pasted
    from a browser / Word / PDF on a Windows box — a cp1252 smart-quote / en-dash / NBSP in a comment
    or value — makes ``read_text(encoding="utf-8")`` raise ``UnicodeDecodeError`` (a ``ValueError``
    subclass, NOT an ``OSError``). Unguarded, that escaped BOTH dotenv readers and aborted the *very
    first* step of onboarding — the reuse-detector reads ``.env`` before any prompt — with a raw
    traceback, the exact outcome this module's other guards (see the ``.env``-parse comments) prevent.

    Fall back to latin-1, which maps every one of the 256 byte values to a code point and so NEVER
    raises: all ASCII keys/values/paths (every real API key is ASCII) round-trip byte-for-byte, and
    only a non-ASCII free-text comment/value is cosmetically remapped when the file is later merged +
    rewritten as UTF-8. That is strictly better than the alternatives — a hard crash, or treating the
    file as ``""`` and silently dropping every other key the user has (the reason we do NOT just
    swallow-to-empty). ``write_dotenv`` backs the original up first, so even the remap is recoverable.
    """
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        notice(f"{target.name} is not valid UTF-8 — reading it leniently (latin-1); non-ASCII bytes may be remapped")
        return target.read_text(encoding="latin-1")


def write_dotenv(
    updates: dict[str, str],
    *,
    path: Path | None = None,
    backup: bool = True,
    secret: bool = True,
) -> tuple[MergeReport, Path | None]:
    """Back up, merge, and atomically write ``updates`` into ``.env``.

    Returns the merge report and the backup path (``None`` if there was no
    existing file). By default every **secret-named** written value (one whose key
    passes ``_looks_secretish`` — every API key in the credentials catalog does) is
    registered so it is masked in logs; non-secret bookkeeping written alongside
    (``SOG_SOURCE``/``SOG_LLM``/knob values) is deliberately NOT registered, so common
    words like the model name aren't masked everywhere in the transcript. Pass
    ``secret=False`` to skip registration entirely for a purely non-sensitive write
    (e.g. the ``SOG_MCP_CONFIG`` path pointer) so it is not redacted from the log.
    """
    target = path or constants.dotenv_path()
    if secret:
        # Register only the SECRET-NAMED values for masking (a93#4), not every value written. ``updates``
        # carries non-secret bookkeeping too (``SOG_SOURCE``, ``SOG_LLM``, knob values like ``false`` /
        # ``./data`` / ``gpt-4o``); registering those as secrets made ``redact()`` mask common words like
        # "false" and the provider/model name everywhere in the transcript. Name-gating via
        # ``_looks_secretish`` is safe: every secret env_var in the credentials catalog is secretish
        # (locked by a test), so no real key is under-registered. We deliberately do NOT also
        # placeholder-gate here — matching the read side (``read_dotenv_values``, C-LOW): for a
        # secret-named value the safe bias is to over-mask a stub, never risk under-masking a real key.
        for k, v in updates.items():
            if _looks_secretish(k):
                register_secret(v)
    existing = _read_text_lenient(target) if target.exists() else ""
    backup_path = backup_dotenv(target) if backup else None
    merged, report = merge_dotenv(existing, updates)
    _atomic_write_text(target, merged)
    return report, backup_path


# --------------------------------------------------------------------------- #
# Assembling the owned-key set + detecting an existing config
# --------------------------------------------------------------------------- #
def assemble_owned_keys(
    provider: ProviderSpec,
    field_values: dict[str, str],
    model: str | None = None,
    knobs: dict[str, str] | None = None,
) -> dict[str, str]:
    """The exact set of env vars the wizard OWNS for a provider choice: the
    fields the user supplied + ``LLM_SOURCE``/``SOG_SOURCE`` + ``SOG_LLM`` +
    any chosen knobs. Empty values are dropped."""
    owned: dict[str, str] = {}
    for f in provider.all_fields():
        val = field_values.get(f.env_var)
        if val:
            # A field value can arrive as a YAML scalar from an --answers file: the documented Azure
            # toggle `llm.fields: {OPENAI_USE_RESPONSES_API: true}` yields a bool, a numeric
            # OPENAI_API_VERSION yields an int. answers._validate guards that `llm.fields` IS a dict but
            # never its value types, so an unquoted scalar reaches here. Coerce to str — mirroring the
            # knob belt below (C-F1) — so it never hits _format_kv's `"$" in value` as a non-str (a raw
            # TypeError -> generic exit 1 on the flagship --answers CI path).
            owned[f.env_var] = val if isinstance(val, str) else str(val)
    owned["LLM_SOURCE"] = provider.source
    owned["SOG_SOURCE"] = provider.source
    # Model resolution. Azure is special: its "model" is the *deployment* name baked into the
    # request URL, so `default_model` (gpt-5.5) is a placeholder, NOT a working fallback —
    # writing it points the agent at a deployment that may not exist (an inference-time 404
    # right after a *successful* key validation). When no model was chosen and none is pinned
    # via SOG_LLM, recover the deployment from the endpoint the user actually configured
    # before falling back to the (last-resort) provider default.
    azure_deployment = (
        azure_deployment_from_endpoint(field_values.get("OPENAI_ENDPOINT", "")) if provider.key == "azure" else None
    )
    chosen_model = model or field_values.get("SOG_LLM") or azure_deployment or provider.default_model
    if chosen_model:
        owned["SOG_LLM"] = chosen_model
    for k, v in (knobs or {}).items():
        if v not in (None, ""):
            # A knob value can arrive as a YAML scalar (int ``30`` / bool ``true``) from an --answers
            # file. Coerce to ``str`` so it lands as a clean ``.env`` RHS and never reaches
            # ``_format_kv``'s ``"$" in value`` test as a non-str (a raw ``TypeError`` → generic exit 1,
            # C-F1). Container values are rejected upstream (``answers._validate``); this coercion is the
            # belt for any other caller.
            owned[k] = v if isinstance(v, str) else str(v)
    return owned


def azure_deployment_is_placeholder(provider: ProviderSpec, field_values: dict[str, str], model: str | None) -> bool:
    """True when persisting this Azure config would fall back to the bare placeholder
    deployment (``provider.default_model``, e.g. ``gpt-5.5``) because no *real* deployment
    was supplied — none embedded in the endpoint URL, none pinned via ``SOG_LLM``, and none
    chosen by the user.

    This is the one Azure config that passes Tier-1 key validation (the ``/models`` ping
    only proves the resource + key) yet 404s with ``DeploymentNotFound`` on the agent's first
    real call. Callers (onboarding ``_persist``) warn on it so the failure surfaces at setup
    time, not as an opaque inference-time error. Always ``False`` for non-Azure providers."""
    if provider.key != "azure":
        return False
    embedded = azure_deployment_from_endpoint(field_values.get("OPENAI_ENDPOINT", ""))
    return not (model or field_values.get("SOG_LLM") or embedded)


@dataclass
class ExistingLLM:
    source: str | None
    provider_key: str | None
    model: str | None
    looks_complete: bool
    detail: str
    # True when ``model`` is the provider's default because ``.env`` names none. For Azure that
    # default is a placeholder deployment, and the reuse path must not hand it to ``_persist`` as if
    # the user had configured it -- that silenced the DeploymentNotFound warning (hunt 2026-09-30,
    # u35-setup-ux-14).
    model_is_default: bool = False


def read_dotenv_values(path: Path | None = None) -> dict[str, str]:
    """Read current ``.env`` into ``{KEY: value}`` (last wins). Values are held only in memory; a
    secret-named, non-placeholder value is registered for masking as it is read.

    That registration matters (N8): a real key the Tier-2 gate or the reuse detector *reads* here —
    but which ``sog-setup`` never *writes* itself — would otherwise never enter the redaction
    registry (``write_dotenv`` is the only other registrant), so it could leak verbatim into a log.
    Name-gated via :func:`session_log._looks_secretish`. Unlike the reuse/``present`` decision it is
    NOT placeholder-gated: a case-folded substring placeholder scan false-positives on a real key whose
    body happens to contain ``xxx``/``here``/``your`` (~0.12% of ``sk-ant-`` keys), which would leave a
    real credential OUT of the redaction net. Mirroring ``write_dotenv``'s bias — over-mask a stub, never
    risk under-masking a real key — registration drops the gate (``register_secret`` already ignores
    sub-``_MIN_SECRET_LEN`` values, so a short stub can't over-redact ordinary log text)."""
    target = path or constants.dotenv_path()
    if not target.exists():
        return {}
    values: dict[str, str] = {}
    for ln in _parse_lines(_read_text_lenient(target)):
        if ln.kind == "kv":
            # matched-quote strip + double-quote unescape (N14), shared with merge_dotenv migration.
            raw_val = _decode_dotenv_value(ln.raw.split("=", 1)[1])
            values[ln.key] = raw_val
            # NOT placeholder-gated (see docstring): a real key whose body contains a placeholder
            # substring must still register, mirroring write_dotenv's over-mask-a-stub bias. register_secret
            # ignores sub-_MIN_SECRET_LEN values, so a short stub can't over-redact ordinary text.
            if raw_val and _looks_secretish(ln.key):
                register_secret(raw_val)
    return values


# --------------------------------------------------------------------------- #
# Placeholder detection — is a .env value a real credential or a template stub?
# --------------------------------------------------------------------------- #
# Substrings that betray a not-yet-filled-in value. The shipped ``.env.example`` uses
# ``sk-ant-...`` / ``sk-...`` / ``gsk_...`` / ``<your-deployment-name>`` etc., and the README tells
# users to ``cp .env.example .env`` — so a placeholder-holding ``.env`` is a normal starting state.
# Matched case-insensitively anywhere in the trimmed value. Kept deliberately NAME-AGNOSTIC (unlike
# the stricter Tier-2 key-gate, which keys off provider prefixes) so a real key for any provider —
# including an Azure hex key with no ``sk-`` prefix — is never mistaken for a stub.
_PLACEHOLDER_SENTINELS = (
    "...",
    "xxx",
    "your",
    "placeholder",
    "changeme",
    "example",
    "dummy",
    "replace",
    "redacted",
    "here",
    "<",
    ">",
)


#: Sentinels no real credential or endpoint contains anywhere: matched as substrings.
_PLACEHOLDER_MARKS = ("...", "<", ">")


def value_looks_like_placeholder(value: str) -> bool:
    """True when ``value`` is empty or an obvious template stub (e.g. ``sk-ant-...``,
    ``<your-key>``) rather than a real credential. Used to avoid *keeping* a ``.env`` that was only
    ``cp``-d from ``.env.example`` — see :func:`detect_existing_llm`'s onboarding consumers.

    The word sentinels (``your``, ``here``, ``example``, ``xxx``...) are matched as WHOLE words --
    the runs of letters and digits between separators -- not as substrings. As substrings they
    fired inside the random body of real keys ("...Xxx...", "...hEre...") -- about 0.3% of keys --
    and the CLI refused a working key with "run sog-setup", which cannot help (hunt 2026-09-30,
    u17-cli-report-9). Template stubs are words: ``your-key-here``, ``sk-ant-xxxxxxxx``,
    ``https://example.openai.azure.com``.
    """
    v = (value or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    if not v:
        return True
    low = v.lower()
    if any(mark in low for mark in _PLACEHOLDER_MARKS):
        return True
    words = [w for w in re.split(r"[^a-z0-9]+", low) if w]
    return any(w in _PLACEHOLDER_SENTINELS or (len(w) >= 3 and set(w) == {"x"}) for w in words)


def detect_existing_llm(path: Path | None = None) -> ExistingLLM | None:
    """Inspect ``.env`` for an already-working LLM config so onboarding can
    offer to keep it by default (never silently repoint the agent)."""
    values = read_dotenv_values(path)
    if not values:
        return None
    source = values.get("SOG_SOURCE") or values.get("LLM_SOURCE")
    model = values.get("SOG_LLM") or values.get("SOG_LLM_MODEL")
    provider = None
    if source:
        # Through the agent's own spelling rule: it reads ``SOG_SOURCE=azureopenai`` as AzureOpenAI,
        # but ``get_provider`` is exact-case, so the key-presence fallback below took the Azure key
        # for an OpenAI one -- reuse pinged api.openai.com with it and rewrote .env to
        # SOG_SOURCE=OpenAI (hunt 2026-09-30, u35-setup-ux-2).
        try:
            provider = get_provider(canonical_source(source) or source)
        except KeyError:
            provider = None
    # Fall back to inferring provider from which key is present.
    if provider is None:
        for pkey, var in (
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("gemini", "GEMINI_API_KEY"),
            ("groq", "GROQ_API_KEY"),
        ):
            if values.get(var):
                provider = get_provider(pkey)
                break
        # OPENAI_API_KEY is Azure's key variable too; an Azure resource endpoint beside it says which.
        if provider is not None and provider.key == "openai" and _is_azure_host(values.get("OPENAI_ENDPOINT", "")):
            provider = get_provider("azure")
    if provider is None and not source:
        return None

    complete = True
    detail_parts = []
    if provider is not None:
        for f in provider.required:
            if not values.get(f.env_var):
                complete = False
                detail_parts.append(f"missing {f.env_var}")
    # For Azure, an absent SOG_LLM must NOT read as the gpt-4o default — that would make the
    # reuse prompt offer to keep "model gpt-4o" (default-yes) and re-persist the wrong
    # deployment. Recover the real deployment from the endpoint the user configured; only a
    # bare host (nothing to parse) falls through to the provider default.
    if not model and provider is not None and provider.key == "azure":
        model = azure_deployment_from_endpoint(values.get("OPENAI_ENDPOINT", ""))
    return ExistingLLM(
        source=source,
        provider_key=provider.key if provider else None,
        model=model or (provider.default_model if provider else None),
        looks_complete=complete and bool(source or provider),
        detail="; ".join(detail_parts) or "looks complete",
        model_is_default=not model and provider is not None,
    )


def _is_azure_host(endpoint: str) -> bool:
    """True when ``endpoint`` names an Azure OpenAI / AI Services resource -- the agent's own rule
    (``llm._is_azure_endpoint``, restated because llm imports langchain).

    A schemeless ``res.openai.azure.com`` was read as OpenAI, and any ``*.azure.com`` or
    ``*.azure-api.net`` host was read as Azure, unlike the agent (hunt 2026-09-30, u35-setup-ux-2 repair).
    """
    raw = (endpoint or "").strip()
    try:
        host = (urllib.parse.urlsplit(raw if "//" in raw else f"//{raw}").hostname or "").lower()
    except ValueError:
        return False
    return host.endswith((".openai.azure.com", ".cognitiveservices.azure.com", ".services.ai.azure.com"))
