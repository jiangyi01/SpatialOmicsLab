"""Find a UCDeconvolve token without being asked for one, and remember it once it works.

WHY THIS EXISTS. ``ucdeconvolve`` keeps its token in ``settings.token``, which is an in-memory
attribute on a module-level object: nothing on disk, gone when the process exits. Every UCD run is
a fresh worker process, so every run arrived with no token and the wrapper had to be handed one
again -- ``payload.token`` or ``UCD_TOKEN`` in the environment, and a hard failure otherwise. That
is the whole reason a token had to be supplied per call.

WHAT "AUTOMATIC" CAN AND CANNOT MEAN HERE, stated up front because the limit is in the service and
not in this file. Acquiring a *brand new* token is ``register()`` then ``activate(code)``, and the
activation code is delivered **by email**. No amount of code on this box can read that mailbox, so
first acquisition needs one human paste, once. Everything on either side of that paste is
automatic: after the first success the token is validated, cached at ``0600``, and every later run
finds it with no argument, no environment variable and no prompt.

THE HAZARD THIS FILE IS SHAPED AROUND. ``ucdeconvolve.api.register`` defaults to ``dynamic=True``,
and in that mode it calls ``input()`` and ``getpass()`` -- five times, plus confirmation loops. A
worker has no terminal. Depending on how stdin is wired that either raises ``EOFError`` from
somewhere deep in a third-party call stack, or **blocks until the agent's 600-second step budget
expires**, which reads to the user as a hung analysis rather than a missing credential. So
:func:`provision` never lets ``dynamic`` default: it passes every field and forces it off, and
:func:`_assert_no_prompting` fails loudly if a future version of the package reintroduces a prompt
on that path. A credential helper that can hang the turn is worse than one that refuses.

RESOLUTION ORDER, most explicit first, so a caller can always override what is cached:

1. the token passed to the tool call -- an operator naming one means it
2. ``UCD_TOKEN`` in the environment -- what the MCP config and CI already set
3. ``UCD_TOKEN`` in the project ``.env`` -- where ``sog-setup`` writes service keys
4. the local cache written by :func:`remember` -- the rung that makes this automatic

Rungs 2 and 3 are read through the same machinery the rest of the platform uses
(``sog_install.llm_setup.read_dotenv_values``), so a value with an inline comment or quoting is decoded
the one way this repo decodes it, rather than a second time slightly differently here.

NO SECRET IS EVER LOGGED OR RETURNED IN AN ERROR. Every diagnostic prints the *rung* a token came
from and a fingerprint, never the value. ``redaction.py`` already carries the ``uc_`` shape because
a UCDeconvolve token leaked into a served config once; this module does not add a second way to
print one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

#: The one environment variable this token has ever been called. ``setup/credentials.py`` declares
#: it as a ``ServiceKey``, which is what puts it in the ``.env`` ``sog-setup`` writes.
ENV_VAR = "UCD_TOKEN"

#: Cache filename inside the platform state directory. Not in the repo, not in the run directory,
#: and not beside the data: a credential should not travel with an exported dataset.
CACHE_NAME = "ucd_token.json"


class TokenError(RuntimeError):
    """No usable token, with a sentence saying what to do about it."""


@dataclass(frozen=True)
class Resolved:
    """A token and where it came from. ``token`` is secret; ``source`` and ``fingerprint`` are not."""

    token: str
    source: str

    @property
    def fingerprint(self) -> str:
        """A stable, non-reversing 8-char tag, so two runs can be compared without printing a secret."""
        return hashlib.sha256(self.token.encode("utf-8")).hexdigest()[:8]

    def __repr__(self) -> str:  # pragma: no cover - defensive; keeps a secret out of a traceback
        return f"Resolved(source={self.source!r}, fingerprint={self.fingerprint!r})"


# --------------------------------------------------------------------------- #
# where a token can be found
# --------------------------------------------------------------------------- #
def _from_env() -> str:
    return (os.environ.get(ENV_VAR) or "").strip()


def _from_dotenv() -> str:
    """The project ``.env``, decoded the way the rest of the platform decodes it.

    Imported lazily and defensively: this module runs inside the per-tool worker environment, which
    has ``ucdeconvolve`` and its dependencies but is not guaranteed to have the agent package on
    ``sys.path``. A missing platform is a skipped rung, not a crash -- the environment variable and
    the cache still work.
    """
    try:
        from sog_install.llm_setup import read_dotenv_values
    except Exception:
        return ""
    try:
        return (read_dotenv_values().get(ENV_VAR) or "").strip()
    except Exception:
        return ""


def cache_path() -> Path:
    """Where :func:`remember` writes. Falls back to the home directory if the platform is absent."""
    try:
        from sog_install.constants import state_dir

        return Path(state_dir()) / CACHE_NAME
    except Exception:
        return Path.home() / ".spatialomicsgym" / CACHE_NAME


def _from_cache() -> str:
    p = cache_path()
    try:
        if not p.is_file():
            return ""
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        # A corrupt cache is a skipped rung, never an exception: the operator can still pass a
        # token, and failing the run over an unreadable convenience file would be absurd.
        return ""
    return (data.get("token") or "").strip() if isinstance(data, dict) else ""


def remember(token: str) -> Path | None:
    """Persist a token that has been shown to work. Best effort; returns the path or ``None``.

    Written ``0600`` and via a ``.partial`` + ``os.replace`` so a reader never sees a half-file --
    the same idiom the rest of the platform uses for state. Failure to write is not failure to run:
    the token in hand is still good for this process, so a read-only state directory costs the
    caching, not the analysis.
    """
    token = (token or "").strip()
    if not token:
        return None
    p = cache_path()
    tmp = None
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates the file 0600 under a name nobody can predict. The old fixed
        # ``ucd_token.json.partial`` was created 0644 (umask 022) and chmodded only after the token was
        # in it, so another account polling that name in a traversable state dir could read it in
        # between (hunt 2026-09-30, u29a-mcp-transport-18).
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".partial")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"token": token}, indent=2) + "\n")
        os.replace(tmp, p)
        return p
    except Exception:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return None


def resolve(explicit: str | None = None) -> Resolved | None:
    """Walk the ladder and return the first token found, or ``None``. Never raises, never prompts."""
    for source, value in (
        ("argument", (explicit or "").strip()),
        (f"${ENV_VAR}", _from_env()),
        (".env", _from_dotenv()),
        ("cache", _from_cache()),
    ):
        if value:
            return Resolved(token=value, source=source)
    return None


# --------------------------------------------------------------------------- #
# proving a token before a long run leans on it
# --------------------------------------------------------------------------- #
def validate(token: str, *, authenticate: Callable[[str], Any] | None = None) -> bool:
    """Is this token accepted by the service? Uses the package's own validation endpoint.

    ``ucdeconvolve.api.authenticate`` posts to ``api/validate_token`` and, on a non-OK response,
    **logs an error and returns ``None`` anyway** -- it does not raise and it does not return a
    boolean, so calling it proves nothing by itself. What it does do is set ``settings.token`` only
    when the service said OK. So the honest test is: clear the setting, call it, and read back
    whether the package accepted it. That is what this does.
    """
    token = (token or "").strip()
    if not token:
        return False
    try:
        import ucdeconvolve as ucd
        from ucdeconvolve._settings import settings
    except Exception as exc:  # pragma: no cover - worker env always has it; be explicit if not
        raise TokenError(f"ucdeconvolve is not importable in this environment: {exc}") from exc

    fn = authenticate or ucd.api.authenticate
    before = getattr(settings, "token", None)
    try:
        settings.token = None
        fn(token)
        return bool(getattr(settings, "token", None))
    except Exception:
        # A network failure is not an invalid token, and must not be reported as one -- an operator
        # told their key is bad will replace a key that was fine.
        settings.token = before
        raise
    finally:
        if not getattr(settings, "token", None):
            settings.token = before


def resolve_or_raise(explicit: str | None = None, *, verify: bool = False) -> Resolved:
    """:func:`resolve`, but with a sentence a user can act on when there is nothing to find."""
    found = resolve(explicit)
    if found is None:
        raise TokenError(
            "No UCDeconvolve token. Any one of these fixes it, cheapest first: pass `token` to the "
            f"tool call; export {ENV_VAR}; put {ENV_VAR} in the project .env (`sog-setup` writes it "
            "there); or run `python -m tools.ucd_token provision` once to register and cache one. "
            "Register at https://ucdeconvolve.org -- activation arrives by email, so first-time "
            "setup needs one paste and nothing after that."
        )
    if verify and not validate(found.token):
        raise TokenError(
            f"The UCDeconvolve token from {found.source} (fingerprint {found.fingerprint}) was "
            "refused by the service. If it was rotated, update that source -- a cached copy is at "
            f"{cache_path()} and can be deleted safely."
        )
    return found


# --------------------------------------------------------------------------- #
# acquiring one, for the case where there is nothing to resolve
# --------------------------------------------------------------------------- #
def _assert_no_prompting(fn: Callable[..., Any], name: str) -> None:
    """Refuse to call into a code path that can block a worker on stdin.

    ``register``'s ``dynamic=True`` default drives five ``input()``/``getpass()`` calls. We always
    pass ``dynamic=False``, but that is our discipline and not the package's promise -- so this
    checks the signature still has the parameter we are relying on. If a future version renames or
    drops it, this raises a sentence naming the problem instead of the worker hanging until the
    step budget expires and the user being told the analysis timed out.
    """
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
        return
    # A ``**kwargs``-only signature counts as "does not declare it", deliberately. If the package
    # dropped ``dynamic`` but kept ``**kwargs``, our ``dynamic=False`` would be accepted and
    # silently ignored -- and that is precisely the case that ends in a prompt and a hung worker.
    # A signature that dropped it with no ``**kwargs`` would at least raise TypeError on its own.
    if "dynamic" not in params:
        raise TokenError(
            f"ucdeconvolve.api.{name}() no longer takes `dynamic`, so this helper cannot guarantee "
            "it will not prompt for input. Refusing to call it from a worker, which has no "
            "terminal and would block until the step budget expires. Acquire the token "
            "interactively instead and set it with `python -m tools.ucd_token save` (it asks for the token)."
        )


def provision(
    *,
    username: str,
    password: str,
    firstname: str,
    lastname: str,
    email: str,
    institution: str,
) -> dict[str, Any]:
    """Register a new account, non-interactively. Returns the service's response.

    This is step one of two. The service emails an activation code; feed it to :func:`activate`,
    which is what actually yields a token. Every field is required here precisely so ``dynamic``
    can be forced off -- the package skips its prompts only when all six are supplied.
    """
    missing = [
        n
        for n, v in (
            ("username", username),
            ("password", password),
            ("firstname", firstname),
            ("lastname", lastname),
            ("email", email),
            ("institution", institution),
        )
        if not str(v or "").strip()
    ]
    if missing:
        raise TokenError(
            "Registration needs every field, because supplying all of them is what stops the "
            f"library prompting on stdin. Missing: {', '.join(missing)}."
        )
    import ucdeconvolve as ucd

    _assert_no_prompting(ucd.api.register, "register")
    return ucd.api.register(
        username=username,
        password=password,
        firstname=firstname,
        lastname=lastname,
        email=email,
        institution=institution,
        dynamic=False,
    )


def activate(code: str, *, remember_it: bool = True) -> Resolved:
    """Exchange an emailed activation code for a token, and cache it.

    ``ucdeconvolve.api.activate`` returns ``None`` and leaves the token on ``settings.token``, so
    the value is read back from there rather than from a return value.
    """
    code = (code or "").strip()
    if not code:
        raise TokenError("An activation code is required. It arrives by email after `provision`.")
    import ucdeconvolve as ucd
    from ucdeconvolve._settings import settings

    # ``activate`` has no ``dynamic`` switch -- it prompts via ``getpass`` only when ``code`` is
    # falsy. The empty-code check above is therefore the whole guard: with a real code in hand this
    # path cannot reach stdin.
    ucd.api.activate(code)
    token = (getattr(settings, "token", None) or "").strip()
    if not token:
        raise TokenError(
            "Activation did not yield a token. The usual cause is a code that was already used or "
            "has expired; request a new one by registering again."
        )
    if remember_it:
        remember(token)
    return Resolved(token=token, source="activation")


# --------------------------------------------------------------------------- #
# a tiny CLI, so first-time setup is one command and not a Python session
# --------------------------------------------------------------------------- #
def _read_secret(prompt: str) -> str:
    """A secret from the terminal without echo, or the first line of a piped stdin."""
    import getpass
    import sys

    if sys.stdin is not None and sys.stdin.isatty():
        return getpass.getpass(prompt).strip()
    return (sys.stdin.readline() if sys.stdin is not None else "").strip()


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m tools.ucd_token", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    # A secret given as an argument is on the command line, which every local account can read in
    # ps or /proc/<pid>/cmdline, and it lands in the shell's history. Both still work, but leaving
    # them out (or passing "-") reads the token from $UCD_TOKEN or a no-echo prompt / stdin, and the
    # password from the prompt (hunt 2026-09-30, u29a-mcp-transport-5, the same class on this CLI).
    sub.add_parser("show", help="say which rung a token would come from (never prints the token)")
    p_save = sub.add_parser("save", help="validate a token you already have, then cache it")
    p_save.add_argument(
        "token", nargs="?", default="-", help=f"omit (or '-') to read ${ENV_VAR}, else a prompt or stdin"
    )
    p_prov = sub.add_parser("provision", help="register a new account (activation code arrives by email)")
    for f in ("username", "password", "firstname", "lastname", "email", "institution"):
        if f == "password":
            p_prov.add_argument("--password", default="-", help="omit (or '-') to be prompted, or pipe it on stdin")
        else:
            p_prov.add_argument(f"--{f}", required=True)
    p_act = sub.add_parser("activate", help="exchange the emailed code for a token and cache it")
    p_act.add_argument("code")

    args = ap.parse_args(argv)
    if args.cmd == "save" and args.token == "-":
        args.token = _from_env() or _read_secret("UCD token: ")
    if args.cmd == "provision" and args.password == "-":
        args.password = _read_secret("UCD account password: ")

    if args.cmd == "show":
        found = resolve()
        if found is None:
            print(f"no token found; looked at: argument, ${ENV_VAR}, .env, {cache_path()}")
            return 1
        print(f"token from {found.source} (fingerprint {found.fingerprint})")
        return 0

    if args.cmd == "save":
        if not validate(args.token):
            print("the service refused that token; nothing cached")
            return 1
        where = remember(args.token)
        print(f"cached at {where}" if where else "valid, but the cache could not be written")
        return 0

    if args.cmd == "provision":
        provision(
            username=args.username,
            password=args.password,
            firstname=args.firstname,
            lastname=args.lastname,
            email=args.email,
            institution=args.institution,
        )
        print("registered; check the inbox for an activation code, then:")
        print("  python -m tools.ucd_token activate <code>")
        return 0

    if args.cmd == "activate":
        got = activate(args.code)
        print(f"activated and cached (fingerprint {got.fingerprint})")
        return 0

    return 2  # pragma: no cover - argparse rejects anything else first


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(_main(sys.argv[1:]))
