"""The one place that answers "may this platform reach that?".

``egress.yaml`` beside this file is the policy; this module loads it once and answers questions
about it. Every boundary that can enforce something reads the same answers, so the proxy, the
broker, ``http_client`` and the recipe lens cannot disagree about what is allowed.

**Why YAML and not Python.** A policy file is data a root process reads. Making it code would mean
the thing that constrains the agent is itself executable by the process it constrains, which is the
one property a policy file must not have. Everything operator-facing in this repo is already YAML
and ``yaml.safe_load`` cannot execute.

**"Read-only to every agent", precisely.** Three different mechanisms, and only the first is a
permission bit:

* the worker runs as ``sog-agent`` and the file is ``root:root`` 0644 in a ``root:root`` tree, so
  it genuinely cannot write it -- and ``boundary.check()`` audits the mode the same way it audits
  ``users.json``;
* the broker's path rule is a positive allowlist of ``<tool_id>_*`` names under ``tools_user/``, so
  no brokered write can name this file at all;
* the **portal** runs as root, so no permission bit stops it. What stands in for one is that no
  writer exists -- pinned by a lens that AST-walks the package for a write whose target mentions
  this file -- and that :func:`fingerprint` is recorded at every boot, so a change is visible in
  the audit log even when it was legitimate.

That third bullet is a weaker guarantee than the first two and is written down rather than papered
over.

**Fail closed, and loudly.** A missing or unparseable policy raises :class:`EgressPolicyError`
rather than returning an empty allowlist. Empty would be fail-closed for egress and would also turn
eleven working tools into permanent refusals that the model reports as "the tool is broken", which
sends an operator chasing the wrong thing. The portal's start-up and ``sog-web boundary check`` both
call :func:`policy` early so the failure is named at boot.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

POLICY_NAME = "egress.yaml"
OVERLAY_NAME = "egress_overlay.yaml"

#: Where an operator may put additions without editing an installed wheel. Read only when it is
#: owned by root and not group- or world-writable -- the ssh discipline, for the same reason.
OVERLAY_ENV = "SOG_EGRESS_POLICY"

#: Ranges a resolved address may never be in. ``is_private`` covers RFC1918 and unique-local;
#: the rest are the ones it does not. 100.64/10 is CGNAT, which Python does not class as private,
#: and 169.254.169.254 -- the cloud metadata endpoint this control mainly exists for -- is caught
#: by ``is_link_local``.
_CGNAT = ipaddress.ip_network("100.64.0.0/10")


class EgressPolicyError(RuntimeError):
    """The policy could not be loaded, or a caller asked about something it does not describe."""


@dataclass(frozen=True)
class Policy:
    hosts: dict[str, dict[str, Any]]
    index_hosts: frozenset[str]
    git_hosts: frozenset[str]
    conda_channels: frozenset[str]
    defaults: dict[str, Any]
    source: str
    digest: str
    repo_paths: dict[str, frozenset[str]] = field(default_factory=dict)

    def scope_hosts(self, scope: str) -> tuple[str, ...]:
        return tuple(sorted(h for h, meta in self.hosts.items() if scope in (meta.get("scope") or ())))


_LOCK = threading.Lock()
_POLICY: Policy | None = None


def policy_path() -> Path:
    return Path(__file__).resolve().parent / POLICY_NAME


def _overlay_path() -> Path | None:
    raw = (os.environ.get(OVERLAY_ENV) or "").strip()
    if not raw:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except (OSError, ValueError):
        return None


def _read_overlay(path: Path) -> dict[str, Any]:
    """An operator overlay, or a refusal. Never a silent skip.

    Mode is checked before content: a policy anyone can edit is not a policy, and the failure mode
    of reading one is exactly the failure this whole module exists to prevent.
    """
    import yaml

    try:
        stat = path.stat()
    except OSError as exc:
        raise EgressPolicyError(f"{path} is named by {OVERLAY_ENV} and cannot be read") from exc
    if stat.st_uid != 0:
        raise EgressPolicyError(f"{path} is not owned by root; refusing to read an egress overlay")
    if stat.st_mode & 0o022:
        raise EgressPolicyError(f"{path} is group- or world-writable; refusing to read it")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        # yaml's parse errors are not ValueErrors: a broken overlay escaped as a raw ParserError past
        # the broker's EgressPolicyError handler, unrefused and unaudited (u16-llm-config-7).
        raise EgressPolicyError(f"{path} could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise EgressPolicyError(f"{path} is not a mapping; refusing to read it as an egress overlay")
    return data


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Union the two. Additive only: an overlay may widen the policy, never narrow it silently."""
    out = dict(base)
    for key in ("package_indexes", "conda_channels"):
        out[key] = list(base.get(key) or []) + list(overlay.get(key) or [])
    out["hosts"] = _merge_hosts(list(base.get("hosts") or []), list(overlay.get("hosts") or []))
    if isinstance(overlay.get("defaults"), dict):
        out["defaults"] = {**(base.get("defaults") or {}), **overlay["defaults"]}
    return out


def _merge_hosts(base: list[Any], overlay: list[Any]) -> list[Any]:
    """Host entries of both, one per host: scopes and repo paths unioned, the base's never dropped.

    Concatenating them let the overlay's later entry for a host REPLACE the base's in the loader, so
    adding one repository on github.com dropped the 25 shipped ones and github's tool and worker
    scopes with them -- a narrowing the docstring above says an overlay cannot do (u16-llm-config-6).
    An overlay entry with no ``paths`` for a host the base restricts by path keeps the base's paths:
    widening to every repository on a host is not something an overlay does by omission.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in [*base, *overlay]:
        if not isinstance(entry, dict) or not entry.get("host"):
            continue
        host = str(entry["host"]).strip().lower()
        if host not in merged:
            merged[host] = dict(entry)
            order.append(host)
            continue
        have = merged[host]
        scopes = list(have.get("scope") or ())
        scopes += [s for s in entry.get("scope") or () if s not in scopes]
        have["scope"] = scopes
        if have.get("paths") and entry.get("paths"):
            seen = {str(p.get("path") or "") for p in have["paths"] if isinstance(p, dict)}
            have["paths"] = list(have["paths"]) + [
                p for p in entry["paths"] if isinstance(p, dict) and str(p.get("path") or "") not in seen
            ]
    return [merged[h] for h in order]


def policy() -> Policy:
    """The loaded policy. Built once; a file that changes under a running process is a state channel."""
    global _POLICY
    with _LOCK:
        if _POLICY is not None:
            return _POLICY
        import yaml

        path = policy_path()
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise EgressPolicyError(f"the egress policy is missing at {path}") from exc
        try:
            data = yaml.safe_load(raw_bytes.decode("utf-8")) or {}
        except (ValueError, yaml.YAMLError) as exc:
            raise EgressPolicyError(f"the egress policy at {path} could not be parsed: {exc}") from exc
        if not isinstance(data, dict):
            raise EgressPolicyError(f"the egress policy at {path} is not a mapping")

        digest_src = bytearray(raw_bytes)
        source = str(path)
        overlay_path = _overlay_path()
        if overlay_path is not None and overlay_path.is_file():
            data = _merge(data, _read_overlay(overlay_path))
            digest_src += overlay_path.read_bytes()
            source = f"{path} + {overlay_path}"

        hosts: dict[str, dict[str, Any]] = {}
        repo_paths: dict[str, frozenset[str]] = {}
        for entry in data.get("hosts") or []:
            if not isinstance(entry, dict) or not entry.get("host"):
                continue
            host = str(entry["host"]).strip().lower()
            hosts[host] = {"scope": tuple(entry.get("scope") or ()), "reason": str(entry.get("reason") or "")}
            paths = entry.get("paths")
            if paths:
                repo_paths[host] = frozenset(
                    str(p.get("path") or "").rstrip("/").lower() for p in paths if isinstance(p, dict)
                )

        indexes = frozenset(
            str(e.get("host")).strip().lower() for e in (data.get("package_indexes") or []) if e.get("host")
        )
        channels = frozenset(str(e.get("name")).strip() for e in (data.get("conda_channels") or []) if e.get("name"))
        if not hosts or not indexes or not channels:
            raise EgressPolicyError(f"the egress policy at {source} is empty or malformed")

        _POLICY = Policy(
            hosts=hosts,
            index_hosts=indexes,
            git_hosts=frozenset(repo_paths),
            conda_channels=channels,
            defaults=dict(data.get("defaults") or {}),
            source=source,
            digest=hashlib.sha256(bytes(digest_src)).hexdigest(),
            repo_paths=repo_paths,
        )
        return _POLICY


def reset_for_tests() -> None:
    """Drop the cached snapshot. Tests only; nothing in the product calls it."""
    global _POLICY
    with _LOCK:
        _POLICY = None


def fingerprint() -> str:
    return policy().digest


# --------------------------------------------------------------------------- #
# the questions the boundaries ask
# --------------------------------------------------------------------------- #
#: Which module owns which hosts. The one mapping that is not in the YAML, because it is a fact
#: about this tree's imports rather than about the policy: the YAML records WHO reaches a host in
#: prose, and this records it in a form ``hosts_for`` can answer with.
_MODULE_HOSTS: dict[str, tuple[str, ...]] = {
    "spatialomicsgym.tool.gene_identifiers": (
        "mygene.info",
        "myvariant.info",
        "mychem.info",
        "api.togoid.dbcls.jp",
        "webservice.bridgedb.org",
        "rest.genenames.org",
    ),
    "spatialomicsgym.tool.ontology": ("www.ebi.ac.uk", "purl.obolibrary.org"),
    "spatialomicsgym.tool.model_organisms": ("www.alliancegenome.org",),
    "spatialomicsgym.tool.expression_atlases": ("www.proteinatlas.org", "gtexportal.org", "www.ebi.ac.uk"),
    "spatialomicsgym.tool.scholarly_literature": (
        "api.crossref.org",
        "api.openalex.org",
        "openalex.org",
        "doi.org",
        "dx.doi.org",
        "eutils.ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
        "www.ncbi.nlm.nih.gov",
        "jats.nlm.nih.gov",
    ),
    "spatialomicsgym.tool.spatial_atlases": (
        "api.brain-map.org",
        "entity.api.hubmapconsortium.org",
        "ontology.api.hubmapconsortium.org",
        "search.api.hubmapconsortium.org",
    ),
}


def hosts_for(module: str) -> tuple[str, ...]:
    """The hosts one module may reach. Raises for a module the policy does not describe.

    Raising rather than returning ``()`` is deliberate: an empty tuple would refuse every call in a
    module that used to work, and the model would report the tool as broken rather than the policy
    as missing an entry.
    """
    known = policy().hosts
    try:
        declared = _MODULE_HOSTS[module]
    except KeyError:
        raise EgressPolicyError(
            f"{module} asked for its egress allowlist and the policy does not describe it. "
            f"Add it to _MODULE_HOSTS and to {POLICY_NAME}."
        ) from None
    missing = [h for h in declared if h not in known]
    if missing:
        raise EgressPolicyError(f"{module} declares hosts absent from {POLICY_NAME}: {missing}")
    return tuple(declared)


def allows(host: Any, *, scope: str, path: Any = None) -> tuple[bool, str]:
    """Is ``host`` reachable in ``scope``? ``(ok, why not)``."""
    name = str(host or "").strip().lower()
    if not name:
        return False, "no host was named"
    meta = policy().hosts.get(name)
    if meta is None:
        return False, (
            f"refusing a request to {name!r} -- SECURITY_RULES R5.3. It is not one of the "
            f"{len(policy().hosts)} destinations this platform's egress policy allows. The policy is "
            f"read-only to every agent and every tool; only an operator can change it, on the shell."
        )
    if scope not in (meta.get("scope") or ()):
        return False, (
            f"{name!r} is allowed for {', '.join(meta.get('scope') or ('nothing',))} but not for "
            f"{scope} -- SECURITY_RULES R5.3."
        )
    if path is not None and name in policy().repo_paths:
        wanted = "/" + str(path).strip("/").lower()
        if wanted not in policy().repo_paths[name]:
            return False, (
                f"{name}{wanted} is not one of the {len(policy().repo_paths[name])} repositories this "
                f"policy allows on {name}. Being on an allowed host is not enough -- that is the whole "
                f"point of the rule. Adding one is an operator action on the shell."
            )
    return True, ""


def _split_repo(url: str) -> tuple[str, str]:
    text = str(url or "").strip()
    if text.startswith("git+"):
        text = text[4:]
    parts = urlsplit(text)
    path = parts.path
    for cut in ("#", "@"):
        if cut in path:
            path = path.split(cut, 1)[0]
    if path.endswith(".git"):
        path = path[:-4]
    return (parts.hostname or "").lower(), path.rstrip("/")


def repo_allowed(url: Any) -> tuple[bool, str]:
    """Path-granular. ``github.com`` being allowed does not allow every repository on it."""
    host, path = _split_repo(str(url or ""))
    if not host:
        return False, f"{str(url)[:80]!r} names no host"
    return allows(host, scope="setup", path=path)


def index_allowed(url: Any) -> tuple[bool, str]:
    host = (urlsplit(str(url or "").strip()).hostname or "").lower()
    if not host:
        return False, f"{str(url)[:80]!r} names no host"
    if host in policy().index_hosts:
        return True, ""
    return False, (
        f"{host!r} is not one of this platform's approved package indexes "
        f"({', '.join(sorted(policy().index_hosts))}). Adding one is an operator action on the shell."
    )


def channel_allowed(name: Any) -> tuple[bool, str]:
    """Judged on the leading segment, so ``conda-forge/label/broken`` is judged on ``conda-forge``."""
    head = str(name or "").strip().split("/", 1)[0]
    if head and head in policy().conda_channels:
        return True, ""
    return False, (
        f"{head!r} is not one of this platform's approved conda channels "
        f"({', '.join(sorted(policy().conda_channels))})."
    )


def redirect_allowed(from_host: Any, to_host: Any, allowed: Any) -> tuple[bool, str]:
    """R5.2: the allowlist must survive a redirect. Every hop is re-checked against the same list."""
    target = str(to_host or "").strip().lower()
    if not target:
        return False, f"{from_host} redirected to a location with no host -- SECURITY_RULES R5.2."
    if target in {str(h).lower() for h in (allowed or ())}:
        return True, ""
    return False, (
        f"{from_host} redirected to {target!r}, which this tool does not declare -- SECURITY_RULES "
        f"R5.2: every hop is re-checked against the same allowlist, and a redirect cannot leave it."
    )


def refuse_address(host: Any, port: int = 443) -> str:
    """Resolve ``host`` and return ONE address safe to connect to. Raises on a refused range.

    Resolve once, check the answers, and hand back the address the caller must connect to -- so
    there is no second resolution between the check and the connect for an attacker's DNS to win.
    """
    name = str(host or "").strip()
    try:
        answers = socket.getaddrinfo(name, int(port), type=socket.SOCK_STREAM)
    except OSError as exc:
        raise EgressPolicyError(f"{name!r} could not be resolved: {exc}") from exc
    for family, _type, _proto, _canon, sockaddr in answers:
        try:
            address = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        for candidate in _unwrapped(address):
            if is_internal(candidate):
                raise EgressPolicyError(
                    f"refusing to connect to {name!r}: it resolves to {candidate}, which is a private, "
                    f"loopback, link-local or carrier-NAT address -- SECURITY_RULES R5.3. This is the "
                    f"control that stops a public name being pointed at this machine or at a cloud "
                    f"metadata endpoint."
                )
        del family
        return sockaddr[0]
    raise EgressPolicyError(f"{name!r} resolved to no usable address")


def is_internal(address: Any) -> bool:
    """Whether one address is inside this network: the one predicate both outbound checks use.

    ``utils.http_client`` had its own copy without the carrier-NAT row, so a declared host resolving
    to 100.100.100.200 (a cloud metadata address) or a tailnet address was fetched there and refused
    here (u16-llm-config-17). The IPv6 unwrapping is applied too, so the two cannot drift again.
    """
    return any(
        c.is_private
        or c.is_loopback
        or c.is_link_local
        or c.is_multicast
        or c.is_reserved
        or c.is_unspecified
        or (c.version == 4 and c in _CGNAT)
        for c in _unwrapped(address)
    )


def _unwrapped(address: Any) -> list[Any]:
    """An IPv6 address and whatever IPv4 address it embeds.

    ``::ffff:169.254.169.254`` is not link-local to ``IPv6Address.is_link_local``, so checking only
    the outer form lets the metadata endpoint straight through.
    """
    out = [address]
    for attr in ("ipv4_mapped", "sixtofour"):
        inner = getattr(address, attr, None)
        if inner is not None:
            out.append(inner)
    teredo = getattr(address, "teredo", None)
    if teredo:
        out.extend(teredo)
    return out


def safe_url(url: Any) -> str:
    """A URL safe to put in a log line or an error message.

    Userinfo is dropped entirely and every query VALUE is replaced while the keys are kept -- the
    keys are what make a message diagnostic ("you did not send an api_key"), the values are what
    leak. Several allowlisted APIs take their key in the query string, so this is not hypothetical.
    """
    try:
        parts = urlsplit(str(url or ""))
    except ValueError:
        return "(unparseable url)"
    netloc = parts.hostname or ""
    if ":" in netloc:
        netloc = f"[{netloc}]"  # an IPv6 literal keeps its brackets, or the port reads as part of it
    try:
        port = parts.port
    except ValueError:
        # A malformed port raised here, inside the refusal path, so the refusal and its audit record
        # never happened (u16-llm-config-8). It is left out rather than repeated.
        port = None
    if port:
        netloc = f"{netloc}:{port}"
    query = "&".join(f"{pair.split('=', 1)[0]}=..." if "=" in pair else pair for pair in parts.query.split("&") if pair)
    cleaned = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
    try:
        from spatialomicsgym import paths as _paths
        from spatialomicsgym import redaction as _redaction

        # Redact BEFORE scrubbing and before any clipping: severing a token first leaves a prefix
        # neither pass recognises. `redaction.redact`'s own docstring states the ordering rule.
        return _paths.scrub(_redaction.redact(cleaned))
    except Exception:
        return cleaned
