"""One definition of what a credential looks like, for every surface that prints text.

Four copies of this list had drifted apart before it was centralised here. ``chat_cli`` carried a
comment claiming it mirrored ``sog_portal.server`` while missing that module's JWT shape -- on the front
door whose ``--json`` mode writes to stdout, a CI-log and machine-capture surface, and whose error
redactor exists to mask provider 401 bodies, which is exactly where a bearer token appears. The
dataset builder under ``huggingface_data/`` had a fifth list again, missing four shapes, for a corpus
built to be published.

This module is a leaf on purpose: it imports only :mod:`re` at module scope, so the CLI (which loads
the package ``__init__`` already and must stay well under its measured 11 ms startup) pays nothing,
and ``report`` can use it without the import cycle that made it copy the pattern in the first place.
"""

from __future__ import annotations

import re
from typing import Any

#: Token shapes masked on every surface. The union of what the four previous copies knew.
#:
#: ``sk-ant-`` precedes the general ``sk-`` rule so the longer prefix is the one reported when these
#: are read by a human; either would mask the whole token. The trailing 32+ hex run is the loosest
#: rule and stays last: it also matches a checksum or a git SHA quoted in an answer. That
#: over-masking is deliberate and predates this module -- the surfaces it guards are fed by stdout of
#: code the agent wrote and ran, and by MCP worker stderr, and leaving those unmasked is worse than
#: redacting the occasional SHA.
#:
#: The prefix rules share a leading ``\b`` because without one they matched *inside* ordinary words:
#: ``task-specific`` is ``ta`` plus something shaped exactly like an OpenAI key, and came out as
#: ``ta[redacted]``. That destroyed the agent's own prompt vocabulary on the CLI, the web UI and the
#: report, and put 1,329 spurious markers into a rebuild of the published trajectory corpus. A real
#: credential is always preceded by ``=``, ``:``, a quote, whitespace or a line start, every one of
#: which is a word boundary, so nothing that used to be masked stops being masked; a key glued
#: straight onto the tail of a word with no separator is the one deliberate exception. The PEM rule
#: is excluded because it opens with a dash, where ``\b`` would demand a preceding word character.
#:
#: The hex rule already had boundaries on both sides, and needed a different exclusion: it has no
#: prefix to anchor on, and every decimal digit is a hex digit, so a long run of *digits* satisfied
#: it. That is not a hypothetical either -- all 13 redactions in the whole published trajectory
#: corpus are 39-digit MERFISH cell identifiers in the middle of a dataframe the agent printed, and
#: the rule caught no real credential there at all. ``(?![0-9]+\b)`` drops the all-decimal case only:
#: inside a run of digits there is no word boundary, so ``[0-9]+\b`` can match nothing shorter than
#: the whole run, and a run carrying any of ``a``-``f`` -- which every real digest, session token and
#: git SHA does -- fails the lookahead and is still masked.
#:
#: The bearer rule is the one shape with no prefix of its own: an opaque OAuth2 token is identified
#: only by the scheme word in front of it. It came from a sixth copy of this list, in the agent's
#: provider-error note, which is the surface where a 401 body most often lands. Three details, each
#: measured rather than guessed:
#:
#: * A **lookbehind**, so only the token is replaced and ``Authorization: Bearer [redacted]`` still
#:   says which scheme failed. Matching the word too would rewrite five strings that already mask
#:   correctly today; as written, all 127 strings the redaction tests guard are byte-identical.
#: * A lookahead requiring a **digit or a capital**. Without it the rule eats the English that
#:   follows the word -- "Bearer authentication-scheme", "Bearer authorization_header_missing" -- and
#:   that is the R81/W defect, which the sixth copy still had at ``{8,}``. Real tokens are base64,
#:   hex or UUID and always carry one; lowercase prose does not.
#: * ``{16,}``, which clears "authentication" (14) and "credentials" (11) on length alone.
#:
#: Its one accepted narrowing is a single space: ``(?<=...)`` must be fixed width, and every HTTP
#: serialiser writes exactly one. A Title-Cased hyphenated phrase of 16+ characters straight after
#: the word is the one false positive left, and no measured line has that shape.
#:
#: ``uc_`` is the one shape here that came from a leak rather than from an earlier copy of this list.
#: A live UCDeconvolve key sat in a served MCP parameter default and was recognised by none of the
#: rules above, so every surface in this module printed it verbatim. Its body is alphanumeric --
#: measured on the real key: 18 lowercase, 20 uppercase, 10 digits, no separators -- and the class is
#: kept that way rather than widened to ``[A-Za-z0-9_\-]``, because ``uc_`` followed by word
#: characters is also the shape of a snake_case identifier, and these surfaces print the agent's own
#: code and its stdout. Stopping at the first ``_`` means no identifier can match. Same reasoning as
#: the ``gh[pousr]_`` and ``gsk_`` rules, which are alphanumeric for the same reason. Measured blast
#: radius at ``{20,}``: zero hits across all 2,011 tracked files and the 17.8 MB published corpus.
SECRET_RE = re.compile(
    r"\b(?:"
    r"sk-ant-[A-Za-z0-9_\-]{16,}"
    r"|sk-[A-Za-z0-9_\-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"  # GitHub PAT / OAuth / user / server / refresh
    r"|gsk_[A-Za-z0-9]{20,}"  # Groq
    r"|AIza[A-Za-z0-9_\-]{20,}"  # Google
    r"|AKIA[A-Z0-9]{12,}"  # AWS access key id
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"  # Slack
    r"|uc_[A-Za-z0-9]{20,}"  # UCDeconvolve
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"  # JWT
    r")"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|(?<=(?i:bearer) )(?=[A-Za-z0-9._\-]*[0-9A-Z])[A-Za-z0-9._\-]{16,}"  # opaque bearer token
    r"|\b(?![0-9]+\b)[A-Fa-f0-9]{32,}\b"
)

#: What a masked token is replaced with, everywhere.
PLACEHOLDER = "[redacted]"

#: Field names that hold a credential, as opposed to naming a slot in the data.
#:
#: Deliberately much narrower than "contains the word key", and the reason is measured. Across both
#: shipped MCP configs, every parameter whose name carries ``key`` names an ``obs``/``obsm``/``uns``
#: slot -- ``spatial_key``, ``annotation_key``, ``cluster_key``, ``layer_key``, ``key_added`` -- and
#: each has a short, legitimate default the model needs. A rule keying on the substring would blank
#: dozens of those to catch one token. So a bare ``key`` matches nothing here; only a *qualified* one
#: does (``api_key``, ``access_key``, ``private_key``, ``client_secret``), alongside the words that
#: can mean nothing else (``token``, ``secret``, ``password``, ``credential``).
CREDENTIAL_NAME_RE = re.compile(
    r"(?:^|_)(?:tokens?|secrets?|passwo?r?ds?|credentials?)(?:_|$)"
    r"|(?:^|_)(?:api|access|private|client|auth|secret)_keys?(?:_|$)"
    r"|(?:^|_)api_?keys?(?:_|$)",
    re.I,
)

#: Short query-parameter names a password travels under in a typed or scripted URL. Kept out of
#: :data:`CREDENTIAL_NAME_RE`, which also judges MCP parameter names, where ``pw`` could name anything;
#: in a request's query string, and only there (:func:`is_credential_query_name`), they mean one thing.
#:
#: ``current`` joined on 2026-10-02. It is the name ``/api/auth/password``'s own body gives the current
#: password, and no route in the portal reads a ``current`` from a query, so in this portal's URLs it
#: means nothing else. It stays out of :data:`CREDENTIAL_NAME_RE` for the same reason as ``pw``.
_PASSWORD_PARAM_RE = re.compile(r"^(?:pass|pwd|pw|passphrase|current)$", re.I)

#: A lowercase letter or digit followed by a capital: where ``newPassword`` reads as ``new_Password``.
_CAMEL_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

#: How many times :func:`redact_query` opens a value looking for a query inside it. One is the gate's
#: ``next=/app?password=...``; two is the same URL encoded twice. A real request stops at one.
_QUERY_NESTING = 3

#: Below this, a value under a credential name is a placeholder, not a secret.
#:
#: Real keys start around 20 characters; the shortest thing this needs to catch is 51. The floor is
#: set at 12 so that ``changeme``, ``your-key``, ``unset`` and ``none`` keep their defaults -- a
#: placeholder is not a leak, and blanking one would tell the model a knob has no default when it
#: does.
_CREDENTIAL_MIN_LEN = 12


def looks_like_credential(name: str, value: Any) -> bool:
    """True when ``value`` is a credential -- by its own shape, or by what it is called.

    Two independent signals, either sufficient:

    * :data:`SECRET_RE` recognises the value. This is the one that works whatever the field is
      called, and is what catches a key pasted into a field named something innocuous.
    * :data:`CREDENTIAL_NAME_RE` matches the field name *and* the value is a long unbroken string.
      Needed because a vendor can mint any shape it likes: the token that prompted this predicate is
      ``uc_`` followed by 48 characters, and matched none of the shapes we knew. The name is the only
      evidence available for a shape nobody has seen.

    Non-strings are never credentials, and neither is a value containing whitespace -- prose and
    connection strings turn up under credential-ish names, and no real token has a space in it.

    This answers "should this be published?", not "is this definitely a secret". A false positive
    costs a default; a false negative mails a key to a third party.
    """
    if not isinstance(value, str):
        return False
    v = value.strip()
    if not v:
        return False
    if SECRET_RE.search(v):
        return True
    if len(v) < _CREDENTIAL_MIN_LEN or any(c.isspace() for c in v):
        return False
    return bool(CREDENTIAL_NAME_RE.search(str(name)))


def redact(text: Any) -> str:
    """Mask every credential in ``text``. Never raises.

    Two passes, in this order:

    1. The setup redaction **registry** -- ``session_log.redact`` knows every key this process wrote
       or read through ``write_dotenv``/``read_dotenv_values``, so a real credential is masked
       whatever its shape, including one no pattern would recognise.
    2. :data:`SECRET_RE`, which catches key-shaped tokens that were never registered.

    Registry first because it is exact; the shape match is the backstop. The import is deferred so
    this module stays a leaf and importing it costs nothing at startup.

    Callers that also truncate must redact **before** clipping, never after: clipping first can sever
    a token so that neither pass recognises the surviving prefix, and the front of a real credential
    reaches the output.
    """
    s = "" if text is None else str(text)
    try:
        from sog_install import session_log

        s = session_log.redact(s)
    except Exception:
        pass  # registry unavailable -> the shape match below still covers the common cases
    return SECRET_RE.sub(PLACEHOLDER, s)


def is_credential_query_name(name: Any) -> bool:
    """Whether a query parameter called ``name`` carries a credential. ``name`` as it appears in the query
    string, percent-encoded or not. Never raises; anything that is not text, or is empty, is not a name.

    The one rule for the request's own URL: :func:`redact_query` masks the values it picks out of an
    access-log line, and the login gate drops the parameters it picks out of the ``next`` it echoes
    (``sog_portal.api.middleware.session``). :data:`CREDENTIAL_NAME_RE` with ``-`` read as ``_``, plus the
    short names in :data:`_PASSWORD_PARAM_RE`, and since 2026-10-02 two more readings of the same names:

    * **camel case** -- ``newPassword``, ``currentPassword``, ``apiKey`` -- read hump by hump as
      ``new_Password``, so the snake-case word boundaries the rule already has apply. The frontend
      spells its fields this way.
    * **brackets** -- ``password[]``, ``user[password]`` -- each part checked on its own.

    Each part is read **both** as written and hump by hump, and either reading is enough. The hump
    reading alone splits a capital inside the credential word itself -- ``passWord`` as ``pass_Word``,
    ``PassPhrase`` as ``Pass_Phrase`` -- and lost names the rule had always masked (repair review,
    2026-10-02); read as written as well, it can only add names, never remove one.

    A name whose parts only contain credential letters (``passport``, ``tokenizer``, ``currentTab``)
    is still no credential.
    """
    if not isinstance(name, str) or not name:
        return False
    try:
        from urllib.parse import unquote_plus

        raw = unquote_plus(name).strip()
        for part in re.split(r"[\[\]]", raw):
            if not part:
                continue
            for word in (part, _CAMEL_HUMP.sub("_", part)):
                word = word.replace("-", "_")
                if CREDENTIAL_NAME_RE.search(word) or _PASSWORD_PARAM_RE.match(word):
                    return True
    except Exception:
        return True  # fails closed, like redact_query: a name that cannot be read is masked, or dropped
    return False


def redact_query(target: Any) -> str:
    """``target`` -- a request path and its query string, as an HTTP access log prints it -- with the
    value of every query parameter named like a credential replaced by :data:`PLACEHOLDER`. Never raises.

    For the access log, where :func:`redact` cannot help: it recognises a key by its SHAPE, and a
    password has none, so ``GET /login?username=a&password=hunter2`` passed through it unchanged into
    the portal's log (hunt 2026-10-01, leaks-access-log-records-query-strings). Here the parameter's
    NAME is the evidence: :func:`is_credential_query_name`, the rule the login gate also drops by.

    * A credential's value runs to the next ``&``. The query is read both ways a server reads it:
      Python's and Starlette's parsers split on ``&`` only, so ``password=Pa;ss`` is the password
      ``Pa;ss``, and older frameworks also start a new name after each ``;`` (since 2026-10-02:
      ``?a=1;password=x`` reached the log in clear). Under either reading, from a credential's ``=`` to
      the next ``&`` is masked: splitting on ``;`` as well cut a password at its first ``;`` and wrote the
      rest in clear (repair review 2, 2026-10-02).
    * A query inside a value is opened too: the login gate used to send ``/app?password=x`` on as
      ``/login?next=%2Fapp%3Fpassword%3Dx`` (it drops the parameter now, 2026-10-02), and a hand-typed
      ``/login?next=...`` still carries one into the sign-in page's ``/api/auth/login?next=...``. The value is
      opened whole, to the next ``&``, so a ``;`` typed raw inside it neither cuts a password short nor
      hides a query that only shows across it.
    * Everything else is kept byte for byte, so a line with no credential in it reads exactly as before.
    * Fails closed: a query that cannot be taken apart is masked whole, and the path is kept.
    """
    s = "" if target is None else str(target)
    if "?" not in s:
        return s
    path, _, query = s.partition("?")
    try:
        return f"{path}?{_redact_pairs(query, 0)}"
    except Exception:
        return f"{path}?{PLACEHOLDER}"


def _redact_pairs(query: str, depth: int) -> str:
    # One '&'-separated part at a time, each put back exactly as it came unless something in it is masked.
    return "&".join(_redact_part(part, depth) for part in query.split("&"))


def _redact_part(part: str, depth: int) -> str:
    """One ``&``-separated part of a query, masked from a credential's ``=`` to its end.

    A name starts where either reading starts one: at the part's start, running to its first ``=`` (the
    server's reading, whose value -- ``;`` and all -- runs to the next ``&``), and after each ``;`` (the
    older frameworks' reading). The first of them, in order, that is a credential's name masks the rest of
    the part, since under the server's reading all of it is one value and a password may hold a ``;``.

    What comes before that name is the first name's value, and it is opened as one value: a query inside
    it (the gate's ``next``) is read to the next ``&`` too, so ``next=/app?password=Pa;ss`` is masked
    whole. One opening per part, so the work grows with the line's length, never with its square.
    """
    first = part.find("=")
    if first < 0:
        return part  # no value anywhere in it
    names = [(0, first)]  # (where a name starts, where its '=' is), in order
    at = 0
    for piece in part.split(";"):
        eq = piece.find("=")
        if at and eq >= 0:
            names.append((at, at + eq))
        at += len(piece) + 1
    cut = next((n for n in names if n[1] + 1 < len(part) and is_credential_query_name(part[n[0] : n[1]])), None)
    if cut is not None:
        part = f"{part[: cut[1] + 1]}{PLACEHOLDER}"
    # The server's reading: the first name's value runs to the end of the part, a kept ``;name=`` included,
    # so it is opened whole. Opening only what came before the ``;`` printed the tail of a nested password
    # that held a ``;`` (``next=/app?password=Pa;ss%26newPassword=Qw``; review of 2026-10-02).
    if cut != (0, first) and first + 1 < len(part):
        part = f"{part[: first + 1]}{_redact_value(part[first + 1 :], depth)}"
    return part


def _redact_value(value: str, depth: int) -> str:
    """``value`` with any query inside it redacted, re-encoded as it came; ``value`` itself when there
    was nothing to mask, so an ordinary parameter is never re-spelled.

    A value with a ``?`` is read as a URL -- a path, ``?``, its query -- and, when its path holds an ``=``,
    as a query as well: read only as a URL, ``password=x?y=1`` kept ``password=x`` as a path nobody looked
    at; read only as a query, a path with three ``=`` before its ``?`` (``/results/run=3/x=1/y=2?password=``)
    spent every nesting level before the ``?`` was reached (review of 2026-10-02).
    """
    if not value or depth >= _QUERY_NESTING:
        return value
    from urllib.parse import quote, unquote_plus

    inner = unquote_plus(value)
    head, mark, rest = inner.partition("?")
    if mark:
        masked = f"{head}?{_redact_pairs(rest, depth + 1)}"
        if "=" in head:
            masked = _redact_pairs(masked, depth + 1)
    elif "=" in inner:
        masked = _redact_pairs(inner, depth + 1)
    elif inner != value:
        masked = _redact_value(inner, depth + 1)  # encoded more than once
    else:
        return value
    if masked == inner:
        return value
    return quote(masked, safe="/") if inner != value else masked


def _filter_stdin_to_stdout() -> int:
    """Redact standard input line by line onto standard output. ``python -m spatialomicsgym.redaction``.

    For a process whose stdout is redirected to a file that outlives it. ``restart_portal.sh``
    sends the portal's stdout -- which carries the full ``pretty_print`` ReAct transcript, prompts
    and answers included -- straight into ``.portal.log`` at the default umask, while
    :mod:`sog_portal.services.history` redacts the same text before it reaches ``portal.db`` (``0600``). One of
    those two is wrong about the same bytes.

    **Line buffered, and never fatal.** A log is read while it is being written, so a filter that
    buffers by block makes ``tail -f`` useless and turns a start-up failure into a silent hang. A
    line this module cannot process is passed through **unchanged** rather than dropped: losing the
    line that says why the portal would not start is a worse outcome than an unredacted one, and
    :func:`redact` is already defensive enough that reaching the fallback means something rarer
    than a credential.

    Redaction is not a substitute for the mode bit. Both are applied: this masks what it
    recognises, and ``0600`` covers what it does not.
    """
    import sys

    for line in sys.stdin:
        try:
            sys.stdout.write(redact(line))
        except Exception:
            sys.stdout.write(line)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess by the launcher
    import sys as _sys

    _sys.exit(_filter_stdin_to_stdout())
