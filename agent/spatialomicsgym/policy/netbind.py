"""Is this bind exposed? One answer for every front door that opens a port.

``sog-web`` worked this out first and wrote down why it matters: our chat front doors have no
login of their own and the agent executes model-written code, so a non-loopback bind hands this
machine to anyone who can reach the port. The legacy Gradio demo (removed 2026-09-30) shipped the
opposite default for the same agent, which is what put the check here instead of in one server
module.

Two subtleties are easy to copy wrong, and both fail *open* when they are:

* an EMPTY host is all-interfaces. ``sock.bind(("", port))`` and ``uvicorn.run(host="")`` listen
  on every interface exactly like ``0.0.0.0``, so ``sog-web --host "$UNSET_VAR"`` -- a launch
  script's unset variable collapsing to ``""`` -- must warn rather than pass for loopback;
* the whole ``127.0.0.0/8`` block is loopback, not just ``127.0.0.1``.

stdlib-only and import-cheap on purpose: every caller is a front door deciding what to print
before it starts listening.
"""

from __future__ import annotations

# Deliberately omits "" — see the module docstring. The names, not the block: `127.` is matched
# separately by prefix so 127.0.0.5 is loopback too.
LOOPBACK_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def is_exposed_bind(host: str | None) -> bool:
    """True when binding ``host`` puts the port on the network rather than on this machine only.

    Fails safe: an unset/blank host is all-interfaces, and anything unrecognised is treated as
    exposed, so a new spelling produces a warning rather than silence.
    """
    normalized = (host or "").strip().lower()
    if not normalized:
        return True
    return not (normalized in LOOPBACK_HOSTNAMES or normalized.startswith("127."))
