"""One outbound HTTP path, with timeouts, bounded retry and logging.

Every tool vendored from ToolUniverse calls out through here. Nothing else in the tree does yet --
this module is additive, and the sixteen pre-existing ``requests`` call sites are deliberately left
alone (see ``DECISIONS.md`` D-004: Exit Gate 1 requires proving no existing backend logic changed,
so retrofitting them is out of scope for the task that introduces this file).

What it is for, in order of how much it matters:

**A host allowlist that the caller must name.** ``request_json`` will not fetch a host its caller did
not declare. The vendored catalog is a set of thin API clients whose parameters come from the model,
and several of them take something that looks like an identifier and interpolate it into a path; an
allowlist means a model that invents ``../`` or a full ``http://`` URL reaches a refusal instead of a
request. The allowlist is a module constant in each tool module, never an argument the model can set.

**A timeout on every request.** ``requests`` has no default timeout: a call with none can hang until
the socket dies, which inside the ReAct loop is a turn that never returns an observation.

**Retry only where retrying is meaningful.** Connection failures, read timeouts, 429 and 5xx are
retried with exponential backoff. A 4xx is not -- a malformed query retried three times is three
identical refusals and three times the latency. ``tenacity`` was already a pinned dependency with no
call sites; this is its first.

**Failures that read as failures.** ``HttpError`` carries the status and a truncated body, and its
``str`` begins with ``Error:`` so that ``agent/execution.py:_EXEC_ERROR_RE`` recognises a failed tool
call as a failure rather than as prose (that regex deliberately does not match the bare word
"error"; see its comment).
"""

from __future__ import annotations

import http.cookiejar
import ipaddress
import json
import logging
import socket
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

logger = logging.getLogger(__name__)

#: Seconds before a request is abandoned. Public biomedical APIs are mostly sub-second; the
#: outliers are full-text fetches and ontology-wide searches, which is what the ceiling is for.
DEFAULT_TIMEOUT = 30.0

#: Attempts, not retries: 3 means one try and two more.
DEFAULT_ATTEMPTS = 3

#: Bytes of a failed response kept in the error message. Enough to carry an API's own explanation
#: ("unknown dataset", "invalid accession") without pasting a 2 MB error page into the transcript.
_ERROR_BODY_CHARS = 500

#: Status codes worth trying again. 429 is rate limiting, 5xx is the far end being unwell; both
#: usually resolve on their own. Everything else is an answer, even when it is "no".
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

_USER_AGENT = "SpatialOmicsGym/1.0 (biomedical research agent; +https://github.com/jiangyi01/SpatialOmicsLab)"

#: Fallbacks ONLY. The live values come from `policy/egress.yaml`'s `defaults:` block -- see
#: :func:`_limits`. These are what applies when the policy cannot be read at all, which must not be
#: a reason for an outbound call to be unbounded.
_FALLBACK_MAX_REDIRECTS = 4
_FALLBACK_MAX_BODY_BYTES = 8 * 1024 * 1024
_FALLBACK_DEADLINE_GRACE = 60.0


def _limits() -> tuple[int, int, float, bool]:
    """``(max_redirects, max_response_bytes, deadline_grace_seconds, refuse_private_ranges)``.

    Read from the central policy, not hard-coded here. ``policy/egress.yaml`` has declared all four
    since Part 1 -- ``max_redirects: 4``, ``max_response_bytes: 8388608``,
    ``deadline_grace_seconds: 60``, ``refuse_private_ranges: true`` -- and **nothing read them**,
    so the block an operator edits and the numbers that actually ran were two different things. The
    first version of this module's redirect work made that worse by inventing a third set (5 hops,
    64 MB, a multiplier instead of a grace). One file decides.

    Never raises. This module is a leaf that tool code imports at call time, and a policy that
    cannot be parsed must not turn every outbound request into an exception -- it falls back to the
    constants above, which are the policy's own current values, so the failure mode is "the limits
    are the shipped ones" and never "there are no limits".
    """
    try:
        from spatialomicsgym.policy import egress

        declared = egress.policy().defaults or {}
    except Exception:
        declared = {}

    def _int(name: str, fallback: int) -> int:
        try:
            value = int(declared[name])
        except (KeyError, TypeError, ValueError):
            return fallback
        return value if value > 0 else fallback

    return (
        _int("max_redirects", _FALLBACK_MAX_REDIRECTS),
        _int("max_response_bytes", _FALLBACK_MAX_BODY_BYTES),
        float(_int("deadline_grace_seconds", int(_FALLBACK_DEADLINE_GRACE))),
        bool(declared.get("refuse_private_ranges", True)),
    )


_session: requests.Session | None = None


class HttpError(RuntimeError):
    """An outbound call that did not produce a usable response.

    The message is prefixed ``Error:`` on purpose -- that is the literal the ReAct loop's failure
    regex looks for at the start of a line, so a tool that lets this propagate is reported to the
    agent as a failed action rather than as an observation it should reason about.
    """

    def __init__(self, message: str, *, url: str, status: int | None = None, body: str = "") -> None:
        self.url = url
        self.status = status
        self.body = body
        #: The message without the ``Error:`` prefix, for a caller that wants to put it inside its
        #: own ``{"status": "error"}`` payload without saying "Error: Error: ...".
        self.detail = message
        super().__init__(f"Error: {message}")


class _RetryableStatus(Exception):
    """Internal: a 429/5xx, raised so tenacity can see it and retry."""

    def __init__(self, response: requests.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


#: Two threads' first calls could each build a session, one overwriting the other mid-request.
_SESSION_LOCK = threading.Lock()


def get_session() -> requests.Session:
    """The process-wide session, created on first use.

    One session means one connection pool. The vendored tools fan out over about two dozen hosts and
    a multi-step agent turn will hit the same one repeatedly, so keeping the TLS handshake is worth
    more than the isolation a per-call session would buy.
    """
    global _session
    if _session is None:
        with _SESSION_LOCK:
            if _session is None:
                made = requests.Session()
                made.headers.update({"User-Agent": _USER_AGENT})
                # No cookie jar. The one session is shared by every account's concurrent turns, and
                # a cookie one upstream set during alice's call was sent on bob's next call to it
                # (SECURITY_FINDINGS MED-17; hunt 2026-09-30, u02-server-b-14). Nothing here needs
                # a server to remember it.
                made.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
                _session = made
    return _session


def _refuse_private(host: str, url: str) -> None:
    """Refuse a host that resolves to an address inside this network (R5.3).

    An allowlist of NAMES is not an allowlist of DESTINATIONS. A name in it can resolve to
    ``127.0.0.1``, to ``169.254.169.254`` -- the cloud metadata endpoint, which is the classic
    way an outbound fetch becomes a credential read -- or to anything else on the private side of
    the boundary this module exists to sit on. A literal IP in the URL does the same with no DNS
    at all, which is why the literal is checked first and the lookup only happens when it is not
    one.

    **What this does not close, stated rather than implied:** the address checked here is not the
    address the socket connects to. ``requests`` resolves again when it connects, so a name whose
    record changes between the two -- DNS rebinding -- still reaches the second answer. Closing
    that means connecting to the checked IP with the hostname carried in SNI and the ``Host``
    header, which is a transport-level change this function cannot make on its own. It is the
    residual the plan's Part 2 proxy and seatbelt are for, and ``sog-web boundary check`` reports
    the egress gap rather than claiming isolation.

    A name that does not resolve is left alone: the connection is about to fail anyway, and
    refusing it here would turn a transient DNS outage into a security-shaped error message.
    """
    if not _limits()[3]:
        return  # the policy turned the check off; `sog-web boundary check` reports that it is off
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    candidates = [address] if address is not None else []
    if address is None:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return  # cannot resolve; let the connection fail on its own terms
        for info in infos:
            try:
                candidates.append(ipaddress.ip_address(info[4][0]))
            except ValueError:
                continue
    from spatialomicsgym.policy.egress import is_internal

    for candidate in candidates:
        if is_internal(candidate):
            raise HttpError(
                f"refusing a request to {host!r}: it resolves to {candidate}, which is inside this "
                "network. An allowlist of names is not an allowlist of destinations.",
                url=url,
            )


def _check_host(url: str, allowed_hosts: tuple[str, ...]) -> str:
    """Refuse anything the calling module did not declare, before a socket is opened.

    Returns the host so the caller can log it. Raises ``HttpError`` rather than returning a flag:
    a blocked host is a programming or injection fault, not a queryable result.

    Called on EVERY redirect hop, not only on the URL the caller wrote -- see ``_MAX_REDIRECTS``.
    """
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise HttpError(f"refusing a non-HTTPS request to {parts.scheme or '(no scheme)'}://", url=url)
    host = parts.hostname or ""
    if host not in allowed_hosts:
        raise HttpError(
            f"refusing a request to {host!r}, which this tool module does not declare "
            f"(allowed: {', '.join(allowed_hosts)})",
            url=url,
        )
    _refuse_private(host, url)
    # And the platform's egress policy, which egress.yaml and SECURITY.md say this client enforces:
    # it read only the module's own constant, so taking a host out of egress.yaml stopped nothing
    # (u16-llm-config-5). A policy that cannot be read refuses: this is the allow decision, and the
    # policy module fails closed.
    try:
        from spatialomicsgym.policy import egress

        permitted, why = egress.allows(host, scope="tool")
    except Exception as exc:
        permitted, why = False, f"the egress policy could not be read ({type(exc).__name__}: {exc})"
    if not permitted:
        raise HttpError(why or f"refusing a request to {host!r}: the egress policy does not allow it", url=url)
    return host


def _shut(response: requests.Response) -> None:
    """Shut down the socket under ``response`` (never raises): a recv blocked on it returns."""
    sock = getattr(getattr(getattr(response, "raw", None), "connection", None), "sock", None)
    try:
        if sock is not None:
            sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        response.close()
    except Exception:
        pass


def _read_bounded(response: requests.Response, url: str, deadline: float, max_bytes: int | None = None) -> None:
    """Read a streamed response into ``response._content``, refusing one that will not end (R5.4).

    ``stream=True`` is what makes the redirect loop above able to close a hop without downloading
    it; the cost is that the body is not read until something asks, and `.text` would then read it
    unbounded. This reads it here, with a ceiling and the same wall clock the hops use, so
    everything after this point can treat the response as an ordinary one.

    The refusal happens at the first chunk that crosses the ceiling, so a 10 GB body costs 64 MB
    and not 10 GB.
    """
    ceiling = _limits()[1] if max_bytes is None else max_bytes
    total = 0
    chunks: list[bytes] = []
    # The clock is also enforced from OUTSIDE the read. Checked only between chunks, it never fired
    # on a peer trickling one byte per ``timeout``: each recv resets the socket timeout, and one
    # 64 KiB chunk read could then run for 65536 x timeout (hunt 2026-09-30, u16-llm-config-4). At
    # the deadline the socket is shut down, which ends a blocked recv.
    watchdog = threading.Timer(max(0.0, deadline - time.monotonic()), _shut, args=(response,))
    watchdog.daemon = True
    watchdog.start()
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > ceiling:
                response.close()
                raise HttpError(
                    f"{url} returned more than {ceiling // (1024 * 1024)} MB; refusing to read further",
                    url=url,
                    status=response.status_code,
                )
            if time.monotonic() > deadline:
                response.close()
                raise HttpError(f"{url} was still sending after the deadline", url=url, status=response.status_code)
            chunks.append(chunk)
    except (requests.RequestException, OSError, ValueError, AttributeError) as exc:
        if time.monotonic() >= deadline:
            raise HttpError(
                f"{url} was still sending after the deadline", url=url, status=response.status_code
            ) from exc
        raise
    finally:
        watchdog.cancel()
        response.close()
    # `_content` is what `.text`, `.json()` and `.content` all read. Setting it (and clearing the
    # consumed flag) turns this streamed response into one indistinguishable from a buffered one,
    # so nothing downstream -- including the error paths that read `response.text` -- has to know.
    response._content = b"".join(chunks)
    response._content_consumed = True


def request_json(
    url: str,
    *,
    allowed_hosts: tuple[str, ...],
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    form_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Any:
    """Fetch ``url`` and return its decoded JSON.

    ``allowed_hosts`` is required and is a module constant at every call site -- see the module
    docstring for why it is not optional.

    Raises ``HttpError`` for a refused host, a transport failure that outlived the retries, a status
    outside 2xx, or a body that does not parse as JSON.
    """
    text = request_text(
        url,
        allowed_hosts=allowed_hosts,
        method=method,
        params=params,
        json_body=json_body,
        form_data=form_data,
        headers={"Accept": "application/json", **(headers or {})},
        timeout=timeout,
        attempts=attempts,
    )
    try:
        return json.loads(text)
    except ValueError as exc:
        raise HttpError(
            f"{url} returned a body that is not JSON ({exc})",
            url=url,
            body=text[:_ERROR_BODY_CHARS],
        ) from exc


def request_text(
    url: str,
    *,
    allowed_hosts: tuple[str, ...],
    method: str = "GET",
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    form_data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    attempts: int = DEFAULT_ATTEMPTS,
) -> str:
    """Fetch ``url`` and return its body as text.

    The non-JSON half of the same path: a few upstream endpoints answer in XML or TSV, and they get
    the same allowlist, timeout, retry and logging as everything else rather than a private
    ``requests.get`` beside it.
    """
    host = _check_host(url, allowed_hosts)
    session = get_session()
    sent = {k: v for k, v in (params or {}).items() if v is not None}
    # One wall clock for the whole call, retries included -- what SECURITY.md and egress.yaml say it
    # is. It was set inside each attempt, so three attempts that each stalled just short of it ran
    # about three times the documented bound plus the backoff (hunt 2026-09-30, u16-llm-config-extra-22).
    call_budget = timeout + _limits()[2]
    call_deadline = time.monotonic() + call_budget

    def _attempt() -> requests.Response:
        # One hop at a time, with `_check_host` on every one.
        #
        # `requests` follows up to 30 redirects on its own and checks none of them, so this module
        # verified the URL the CALLER wrote and then let an allowed host send it anywhere -- the
        # whole point of the allowlist, undone by one `Location` header. That is R5.3 and
        # `SECURITY_FINDINGS` HIGH-5. `allow_redirects=False` takes the following back from
        # `requests` so each destination can be checked before a socket opens for it.
        max_hops, max_bytes, _grace, _ = _limits()
        # The policy's model, not a multiplier: a wall clock BEYOND the caller's own timeout. It
        # bounds what `timeout` cannot -- a server sending one byte at a time, or a chain where
        # every hop is individually fast.
        budget, deadline = call_budget, call_deadline
        current = url
        verb = method.upper()
        body_json, body_form, query = json_body, form_data, sent or None
        for hop in range(max_hops + 1):
            if time.monotonic() > deadline:
                raise HttpError(f"{url} did not finish within {budget:.0f}s across {hop} hop(s)", url=url)
            if hop:
                _check_host(current, allowed_hosts)
            response = session.request(
                verb,
                current,
                params=query,
                json=body_json,
                data=body_form,
                headers=headers,
                # Never past the call's own deadline: a retry gets what is left of it, not a fresh
                # `timeout` of its own.
                timeout=max(0.1, min(timeout, deadline - time.monotonic())),
                allow_redirects=False,
                stream=True,
            )
            if not (response.is_redirect or response.is_permanent_redirect):
                # A 3xx that `requests` does not call a redirect is one with no `Location` header
                # at all -- a broken server. It must not fall through: `Response.ok` is
                # `status_code < 400`, so the not-ok check at the end of this function treats 302
                # as success and the redirect PAGE's body is returned as the answer. Pre-dates the
                # hop loop; found by a test written for the empty-Location case next to it.
                if 300 <= response.status_code < 400:
                    _read_bounded(response, url, deadline, max_bytes)
                    raise HttpError(
                        f"{current} answered {response.status_code} with no Location to follow",
                        url=url,
                        status=response.status_code,
                        body=response.text[:_ERROR_BODY_CHARS],
                    )
                _read_bounded(response, url, deadline, max_bytes)
                if response.status_code in _RETRY_STATUSES:
                    raise _RetryableStatus(response)
                return response
            location = response.headers.get("Location") or ""
            response.close()
            if not location.strip():
                raise HttpError(f"{current} answered {response.status_code} with no Location", url=url)
            current = urljoin(current, location)
            # What `requests` itself does across a redirect, kept so behaviour does not change for
            # the chains that already worked: a 303, and a 301/302 on a non-GET, become a GET with
            # no body. The query is already in the resolved location.
            if response.status_code == 303 or (response.status_code in (301, 302) and verb not in ("GET", "HEAD")):
                verb, body_json, body_form = "GET", None, None
            query = None
        raise HttpError(f"{url} redirected more than {max_hops} times", url=url)

    retrying = Retrying(
        stop=stop_after_attempt(max(1, attempts)) | stop_after_delay(call_budget),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type((_RetryableStatus, requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )

    try:
        response = retrying(_attempt)
    except _RetryableStatus as exc:  # every attempt came back 429/5xx
        response = exc.response
        logger.warning("%s %s gave up after %d attempts: HTTP %s", method, host, attempts, response.status_code)
        raise HttpError(
            f"{url} returned HTTP {response.status_code} on every one of {attempts} attempts",
            url=url,
            status=response.status_code,
            body=response.text[:_ERROR_BODY_CHARS],
        ) from exc
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("%s %s failed after %d attempts: %s", method, host, attempts, exc)
        raise HttpError(f"could not reach {url} after {attempts} attempts ({exc})", url=url) from exc
    except RetryError as exc:  # defensive: reraise=True should mean this never arrives
        logger.warning("%s %s exhausted its retries: %s", method, host, exc)
        raise HttpError(f"could not reach {url} after {attempts} attempts", url=url) from exc
    except requests.RequestException as exc:
        logger.warning("%s %s failed: %s", method, host, exc)
        raise HttpError(f"request to {url} failed ({exc})", url=url) from exc

    if not response.ok:
        logger.info("%s %s returned HTTP %s", method, host, response.status_code)
        raise HttpError(
            f"{url} returned HTTP {response.status_code}",
            url=url,
            status=response.status_code,
            body=response.text[:_ERROR_BODY_CHARS],
        )
    return response.text
