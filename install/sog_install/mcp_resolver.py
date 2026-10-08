"""
Install-aware MCP config resolution — the "dynamic management system".

The canonical ``agent/MCP_server/mcp_config.yaml`` ships 88 servers frozen to the
build host's paths (``/opt/conda/envs/<src>``, ``/workspace/…/tools/…``) and a
blanket ``enabled: true``. On any *other* machine every path is wrong and the
``enabled`` flag lies, so each tool hard-fails at call time. The wizard already
knows the live truth — it just built the envs — but historically never wrote it
back.

This module closes that loop. For **every** server it derives, from live state:

* ``enabled`` — ``True`` iff an env that satisfies the tool is actually present
  *and* passes the same import/``library()`` health probe :mod:`provision` uses.
  Not "we tried to build it" — "it works right now on this box".
* ``command`` — ``[<base-env python>, <this clone>/agent/tools/<server_file>]``.
* ``env`` — the interpreter override (``{PREFIX}_PYTHON`` / ``SEURAT_RSCRIPT``)
  repointed at the **live prefix** of whichever env actually satisfies the tool,
  and the worker override rebased onto this clone's ``agent/tools/``. A server with no
  ``env:`` block in the canonical still gets one synthesized (its worker reads
  the override regardless), so currently-dormant overrides activate.

Two writes, per the "Both" decision:

1. :func:`write_setup_config` — the wizard's OWN full config at
   ``install/recipes/mcp_config.setup.yaml`` (all 88, install-aware), plus the flat
   ``install/recipes/env_overrides.env``, and records ``SOG_MCP_CONFIG`` in ``.env``. Never
   touches the canonical.
2. :func:`apply_to_canonical` — rewrites ``agent/MCP_server/mcp_config.yaml`` from that
   same resolution, after a timestamped git-ignored backup. Opt out with
   ``keep_agent_config=True``.

Re-running after building more envs flips newly-healthy servers on and corrects
their prefixes — the config is *dynamic to setting up*.

"Whichever env actually exists is the truth": a tool captured from a shared or
protected source env (``squidpy``→``moscot``, ``svca``→``spatialomicsgym_e1``) resolves
against an ordered candidate list and takes the first present+healthy env, so a
shared build satisfies its partners without a static special-case.

Reuses the live-truth primitives rather than reinventing them:
:func:`provision._healthy`, ``Conda.env_exists/env_prefix/env_list``,
:func:`wiring.base_python_path`/``rebase_tools_path``/``load_original_servers``/
``env_overrides_text``, and the :mod:`constants` path + namespace helpers.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from . import constants, llm_setup, provision, wiring
from .decisions import BuildStrategy

if TYPE_CHECKING:
    from collections.abc import Callable

    from .envtools import Conda
    from .session_log import SessionLog
    from .specs import ToolSpec


# --------------------------------------------------------------------------- #
# Result of resolving one server
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedEntry:
    """The live-resolved state of one MCP server.

    ``meta`` is the exact dict written into the config (original order + keys
    preserved, ``command``/``env``/``enabled`` overwritten). The scalar fields
    mirror it for logging / ``doctor`` and to source the wizard's "N/88 on"
    count without re-parsing the dict.
    """

    server_key: str
    enabled: bool
    chosen_env: str | None
    command: list[str]
    env: dict[str, str]
    reason: str
    meta: dict = field(default_factory=dict)
    static_gap: bool = False


# --------------------------------------------------------------------------- #
# Live environment geometry
# --------------------------------------------------------------------------- #
def envs_root(conda: Conda) -> str:
    """The live directory that holds named conda envs on *this* machine.

    Never hardcode ``/opt/conda/envs`` — a user's conda may live anywhere. Derive
    it from the parent of the first non-``base`` env prefix (they all share one
    ``envs/`` dir); fall back to ``<base prefix>/envs``; last resort the
    conventional path.
    """
    # Any conda probe failure — env_list OR a per-env / base env_prefix — must not crash config
    # resolution. A narrow guard around env_list ALONE is not enough: on a slow/loaded box the
    # env_prefix calls below hit the same `conda env list --json` subprocess and raise the same
    # CondaError (envtools `_exec` translates TimeoutExpired/OSError even with check=False). Wrap the
    # whole geometry derivation and fall through to the interpreter-derived last resort. (N1/R20)
    try:
        names = [n for n in conda.env_list() if n != "base"]
        for name in names:
            prefix = conda.env_prefix(name)
            if prefix:
                parent = os.path.dirname(prefix.rstrip("/"))
                if parent:
                    return parent
        base_prefix = conda.env_prefix("base")
        if base_prefix:
            return str(Path(base_prefix.rstrip("/")) / "envs")
    except Exception:  # a conda probe failure must not crash config resolution
        pass
    # Last resort: derive from the running interpreter's prefix (CONDA_PREFIX/sys.prefix) rather than
    # hardcoding the Linux ``/opt/conda/envs`` — correct on a ``~/miniconda3`` / micromamba mac/Win box too.
    return constants.conda_envs_root()


def _reuse_source_envs_enabled() -> bool:
    """Whether a pre-existing *raw source* env (the clone/export source the user already has,
    e.g. ``card_env`` / ``tangram-env`` / ``SpatialDE``) may satisfy a tool WITHOUT a rebuild —
    the "reuse the env you already have instead of cloning a fresh copy" behavior. On by default.

    Set ``SOG_SETUP_NO_ENV_REUSE=1`` to restore strict managed-namespace wiring: only the managed
    ``<basic>_<server>`` env and KNOWN shared/protected sources are ever wired, and every plain
    clone tool builds its own dedicated ``<basic>_<server>`` env even when an importable source env
    is already present. Read at call time so a test / a locked-down redeploy can toggle it per-run.
    """
    return os.environ.get("SOG_SETUP_NO_ENV_REUSE", "").strip().lower() not in {"1", "true", "yes", "y", "on"}


def candidate_target_envs(spec: ToolSpec, basic_env: str) -> list[str]:
    """Ordered, deduped envs that could satisfy ``spec`` on this box.

    The managed ``<basic>_<server>`` env is always tried first (the env this run
    builds). A **shared** source (e.g. ``squidpy``'s ``moscot``) adds the managed
    ``<basic>_<source>`` then the raw ``<source>``; a **protected** source (e.g.
    ``svca``'s ``spatialomicsgym_e1``) adds the raw ``<source>``. With env reuse on
    (the default), a **plain** clone source (``card_env`` / ``tangram-env`` / …) is
    also appended as a read-only last fallback, so a box that already has the source
    env installed reuses it instead of cloning a byte-identical copy. This is how
    "target = the env that actually exists" *emerges from live probing* instead of a
    static rule — :func:`resolve_interpreter` picks the first that exists + is healthy.
    """
    out: list[str] = [constants.tool_env_name(basic_env, spec.server_key)]
    src = spec.source_env
    if src:
        if spec.shared_env:
            _append_unique(out, constants.tool_env_name(basic_env, src))
        # The raw source env is a READ-ONLY reuse fallback. Shared/protected sources are always
        # wired (managed-namespace discipline already trusts them). A plain clone source is added
        # only when env reuse is enabled (the default) — the wizard only ever creates/repairs/
        # deletes <basic>_* envs (assert_deletable_env), so pointing a worker's interpreter at a
        # foreign source env never risks mutating it. The opt-out drops it, restoring strict wiring.
        if spec.shared_env or src in constants.PROTECTED_ENVS or _reuse_source_envs_enabled():
            _append_unique(out, src)
    # In-place adoption of a pre-rename box: if a candidate is a branded env that
    # was historically provisioned under a legacy name, probe the legacy name LAST
    # so an upgrade box reuses its existing env with no rebuild. No-op on a fresh
    # deployment (legacy env absent) and once migration completes.
    for cand in list(out):
        legacy = constants.LEGACY_ENV_ALIASES.get(cand)
        if legacy:
            _append_unique(out, legacy)
    return out


def _append_unique(seq: list[str], value: str) -> None:
    if value not in seq:
        seq.append(value)


def resolve_interpreter(conda: Conda, spec: ToolSpec, basic_env: str) -> tuple[str | None, bool, str]:
    """First candidate env that exists *and* passes the health probe.

    Returns ``(chosen_env, healthy, reason)``. Only existing envs are probed
    (absent ones short-circuit on the cached env map — no subprocess), so the
    cost is bounded to the handful of envs a run actually built.
    """
    tried_unhealthy: list[str] = []
    for cand in candidate_target_envs(spec, basic_env):
        try:
            if not conda.env_exists(cand):
                continue
            if provision._healthy(conda, spec, cand):
                return cand, True, f"healthy:{cand}"
            tried_unhealthy.append(cand)
        except Exception:
            # A slow/loaded deploy box can make the health probe (or even the env-list) time out
            # and raise CondaError mid-resolution. That must never abort the whole config resolve —
            # a single unprobeable candidate is simply not-confirmed-healthy (mirrors envs_root's
            # `except Exception` guard). Keep trying the remaining candidates.
            tried_unhealthy.append(cand)
            continue
    if tried_unhealthy:
        return None, False, "unhealthy:" + ",".join(tried_unhealthy)
    return None, False, "absent"


# --------------------------------------------------------------------------- #
# One server entry
# --------------------------------------------------------------------------- #
def _interpreter_overrides_for_live_env(spec: ToolSpec, prefix: str) -> dict[str, str]:
    """Repoint every ``override_vars`` interpreter at ``prefix``'s ``bin/``.

    The captured value carries the authoritative ``bin/python`` vs ``bin/Rscript``
    tail (that's *why* it was captured, not derived); preserve it. A value with no
    ``/bin/`` segment can't be safely relocated — keep it verbatim.
    """
    out: dict[str, str] = {}
    for var, path in spec.override_vars.items():
        tail = str(path).rsplit("/bin/", 1)
        if len(tail) == 2:
            # Re-point at ``prefix`` via the per-OS interpreter layout (a1c#2): on Windows the exe is
            # ``<env>\python.exe`` / ``<env>\Scripts\Rscript.exe``, not the POSIX ``<env>/bin/<exe>``.
            out[var] = constants.interp_path(prefix, tail[1])
        else:
            out[var] = str(path)
    return out


def _rebase_source_env_extras(env: dict[str, str], spec: ToolSpec, prefix: str) -> None:
    """Follow *non-interpreter* source-env-rooted vars onto the live ``prefix`` (mutates ``env``).

    ``_interpreter_overrides_for_live_env`` rebases the ``override_vars`` interpreters, but a
    ``conda_clone`` spec can carry *other* source-env-rooted vars in the captured canonical ``env``
    that no ``/bin/`` rebase touches — GraphST ships ``GRAPHST_R_HOME = …/envs/GraphST/lib/R``. Left
    frozen, that points at the build-host **source** env (never present on a fresh clone), and the
    GraphST worker uses ``GRAPHST_R_HOME`` **verbatim** (no ``isdir`` check), which *suppresses* its
    own correct ``sys.prefix/lib/R`` auto-detect and hard-fails ``mclust`` on an ``enabled`` server.

    Rebase only a var whose value points into **this spec's own** ``source_env`` (the env it was
    cloned FROM) and that isn't already an ``override_vars`` interpreter — so a var legitimately
    pointing at a *shared/foreign* env (squidpy→moscot, svca→spatialomicsgym_e1) is left untouched. The tail
    under ``/envs/<source_env>/`` is preserved and re-rooted at ``prefix`` — exactly the relocation
    :func:`_interpreter_overrides_for_live_env` performs for ``/bin/`` tails, generalized to any tail
    (for GraphST: ``lib/R`` → ``<prefix>/lib/R``, which its cloned env ships via ``r-base``).
    """
    source = (spec.source_env or "").strip()
    if not source:
        return
    marker = f"/envs/{source}/"
    for var, val in list(env.items()):
        if var in spec.override_vars:
            continue  # interpreters already relocated by _interpreter_overrides_for_live_env
        idx = str(val).find(marker)
        if idx != -1:
            tail = str(val)[idx + len(marker) :]
            if tail:
                env[var] = str(Path(prefix) / tail)


def resolve_server_entry(
    conda: Conda,
    spec: ToolSpec,
    original_meta: dict,
    *,
    basic_env: str,
    base_python: str,
    repo_root: Path,
    root_envs: str,
) -> ResolvedEntry:
    """Resolve one specced server into an install-aware config entry."""
    meta = dict(original_meta or {})

    # command: [base python, rebased server script] — from the original if present,
    # else synthesized from the spec's authoritative server_file basename.
    # A hand-authored scalar `command: "python foo.py"` must NOT be char-split by list(str); guard to
    # a list, else fall through to the spec-synthesis path below (correct, authoritative). (C)
    raw_cmd = meta.get("command")
    cmd = list(raw_cmd) if isinstance(raw_cmd, list) else []
    server_script = ""
    if len(cmd) >= 2:
        server_script = wiring.rebase_tools_path(cmd[1], repo_root)
    elif spec.server_file:
        server_script = str(constants.tools_dir(repo_root) / spec.server_file)
    command = [base_python, server_script] if server_script else [base_python]

    # Guard a hand-authored non-mapping `env:` (a YAML list/scalar) exactly as the sibling `command`
    # guard above: `dict()` on a truthy non-mapping raises ValueError, and resolve_full_config's loop
    # has no per-server try/except, so one bad entry would abort the resolve of ALL servers on every
    # finalize (the nested twin of the R24-D1 top-level `load_original_servers` guard). A malformed
    # original env is dropped to {}; the authoritative spec-derived overrides are added below anyway.
    raw_env = meta.get("env")
    env = dict(raw_env) if isinstance(raw_env, dict) else {}

    # A build-NONE server that carries an interpreter override (spatial_library) has NO dedicated
    # env — its worker runs *inside* the base env under base python. It is therefore healthy exactly
    # when the base env exists, and its override must point at the base env, never a phantom
    # `<basic>_<server>` that is never built. This mirrors doctor._env_report's NONE-awareness: its
    # `import_check` is the worker's own name (not a real pip module), so we trust the base env's
    # presence rather than probe it. Without this branch the generic candidate probe below finds only
    # the never-built `<basic>_<server>`, returns `absent`, and deterministically flips the committed
    # `enabled: true` to `false` with a bogus interpreter path on every run — silently dropping
    # `search_spatial_datasets`. (An override-LESS NONE server has no interpreter to wire, so it falls
    # through to the static-gap path below, unchanged.)
    if spec.build_strategy is BuildStrategy.NONE and spec.override_vars:
        try:
            base_ok = conda.env_exists(basic_env)
            base_prefix = conda.env_prefix(basic_env) if base_ok else ""
        except Exception:
            # A slow/loaded box can make even env_exists / env_prefix time out and raise CondaError.
            # Mirror resolve_interpreter's guard (D-F4): a probe hiccup here must never abort the whole
            # canonical rewrite. Treat as base-absent → keep the override pointed at base python by
            # convention (the else-branch below), so the path is correct the moment the env appears.
            base_ok, base_prefix = False, ""
        if base_prefix:
            # rebase each override interpreter onto the live base prefix (preserves the
            # authoritative bin/python vs bin/Rscript tail, exactly like the healthy branch below)
            env.update(_interpreter_overrides_for_live_env(spec, base_prefix))
        else:
            # base env absent (or its prefix momentarily unavailable): keep the override pointed at
            # base python by convention so the path is already correct the moment the env is present
            for var in spec.override_vars:
                env[var] = base_python
        _rebase_worker(env, spec, repo_root)
        meta["command"] = command
        if env:
            meta["env"] = env
        meta["enabled"] = base_ok
        return ResolvedEntry(
            server_key=spec.server_key,
            enabled=base_ok,
            chosen_env=basic_env if base_ok else None,
            command=command,
            env=env,
            reason="base-env:present" if base_ok else "base-env:absent",
            meta=meta,
        )

    # A spec with no interpreter override can't be health-wired (e.g. build NONE).
    if not spec.override_vars:
        _rebase_worker(env, spec, repo_root)
        meta["command"] = command
        if env:
            meta["env"] = env
        meta["enabled"] = False
        return ResolvedEntry(
            server_key=spec.server_key,
            enabled=False,
            chosen_env=None,
            command=command,
            env=env,
            reason="no-override-vars",
            meta=meta,
            static_gap=True,
        )

    chosen, healthy, reason = resolve_interpreter(conda, spec, basic_env)

    # `command[0]` is ALWAYS the base-env python (see `command` synthesis above), so NO server can launch
    # when the base env is absent — even one whose tool/source env probes healthy. The dangerous case is a
    # shared/protected source env that ships or pre-exists (svca→spatialomicsgym_e1, squidpy→moscot): it probes
    # healthy regardless of the base env, so `enabled` derived from tool-env health ALONE would advertise
    # `enabled:true` for a server whose own interpreter path points at a missing base python. Reachable on
    # a resume run whose base env was since deleted (the base_env phase is skipped without an existence
    # re-check) or any out-of-happy-path caller (doctor) on such a box. Gate on base-env presence too,
    # exactly as the NONE+override branch already does (`enabled = base_ok`). env_exists reads the cached
    # env map (cheap — resolve_interpreter just warmed it); guard it like that branch so a slow-box
    # CondaError degrades to base-absent rather than aborting the whole canonical rewrite.
    try:
        base_ok = conda.env_exists(basic_env)
    except Exception:
        base_ok = False
    enabled = bool(healthy) and base_ok

    if healthy and chosen:
        try:
            prefix = conda.env_prefix(chosen)
        except Exception:
            # Parity with the base_ok guard above (~line 362): env_prefix reads the SAME warm env map
            # resolve_interpreter just populated, so on today's code path it cannot raise — but the
            # author already deemed a hypothetical slow-box CondaError on that identical cached read
            # worth degrading rather than aborting the whole canonical rewrite. A None here routes into
            # the exact "map lost it — best-effort convention" fallback the else-branch already provides,
            # so a raise should too, not unwind past every remaining tool's wiring.
            prefix = None
        if prefix:
            env.update(_interpreter_overrides_for_live_env(spec, prefix))
            _rebase_source_env_extras(env, spec, prefix)  # e.g. GRAPHST_R_HOME → <prefix>/lib/R
        else:  # existed a moment ago but the map lost it — best-effort convention
            env.update(spec.rewired_override_vars(chosen, envs_root=root_envs))
    else:
        # Not built (or unhealthy): point at the env a re-run WILL build, so the
        # path is already correct the moment it exists. enabled stays False.
        best_guess = constants.tool_env_name(basic_env, spec.server_key)
        env.update(spec.rewired_override_vars(best_guess, envs_root=root_envs))

    _rebase_worker(env, spec, repo_root)

    meta["command"] = command
    meta["env"] = env
    meta["enabled"] = enabled
    return ResolvedEntry(
        server_key=spec.server_key,
        enabled=enabled,
        chosen_env=chosen,
        command=command,
        env=env,
        # Annotate ONLY the deciding case (tool env healthy but base absent) so every existing
        # reason string — base present, or tool env absent — is unchanged.
        reason=f"{reason};base-env:absent" if (healthy and not base_ok) else reason,
        meta=meta,
    )


def _rebase_worker(env: dict[str, str], spec: ToolSpec, repo_root: Path) -> None:
    """Repoint the worker override at this clone's ``agent/tools/`` (mutates ``env``)."""
    if spec.worker_var and spec.worker_file:
        env[spec.worker_var] = str(constants.tools_dir(repo_root) / spec.worker_file)
    elif spec.worker_var and env.get(spec.worker_var):
        env[spec.worker_var] = wiring.rebase_tools_path(env[spec.worker_var], repo_root)


def _rebase_only_entry(
    server_key: str,
    original_meta: dict,
    *,
    repo_root: Path,
    base_python: str,
) -> ResolvedEntry:
    """A canonical server with no matching spec: rebase paths, don't health-probe.

    Defensive — on the shipped repo the canonical set is exactly the 88 specs, so
    this never fires — but a drifted clone shouldn't crash resolution. We can't
    verify health without a spec, so the server is forced ``enabled: false`` — the
    fresh-clone canonical is blanket-true, and advertising an unverifiable server
    would let the agent call a portal whose env/worker may not exist.
    """
    meta = dict(original_meta or {})
    # Guard against a scalar `command` (list(str) would char-split it); a non-list is left untouched —
    # this server is forced enabled:false below, so an un-rebased malformed command is never run. (C)
    raw_cmd = meta.get("command")
    cmd = list(raw_cmd) if isinstance(raw_cmd, list) else []
    if len(cmd) >= 2:
        meta["command"] = [base_python, wiring.rebase_tools_path(cmd[1], repo_root)]
    elif cmd:
        meta["command"] = [base_python]
    # Same nested non-mapping `env:` guard as resolve_server_entry (a truthy non-mapping → ValueError
    # in dict(), which would abort the whole resolve loop). This server is forced enabled:false below.
    raw_env = meta.get("env")
    env = dict(raw_env) if isinstance(raw_env, dict) else {}
    for key, val in list(env.items()):
        if isinstance(val, str) and ("/tools/" in val or val.startswith("tools/")):
            env[key] = wiring.rebase_tools_path(val, repo_root)
    if env:
        meta["env"] = env
    # Can't verify health without a spec ⇒ don't advertise it (N7). Force disabled regardless of
    # the declared value (fresh-clone canonical ships blanket-true), so a spec dropped by the
    # tolerant H2 loader can never surface as an unverified enabled:true server.
    enabled = False
    meta["enabled"] = enabled
    return ResolvedEntry(
        server_key=server_key,
        enabled=enabled,
        chosen_env=None,
        command=meta.get("command", [base_python]),
        env=env,
        reason="no-spec:rebased-only",
        meta=meta,
        static_gap=True,
    )


# --------------------------------------------------------------------------- #
# Whole config
# --------------------------------------------------------------------------- #
def resolve_full_config(
    conda: Conda,
    basic_env: str,
    specs: dict[str, ToolSpec],
    *,
    repo_root: Path | None = None,
    original_servers: dict | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict, list[ResolvedEntry]]:
    """Resolve **every** server (the union of the canonical set and the specs).

    Returns ``({"mcp_servers": {...}}, entries)``. Iterating the union keeps the
    output drift-robust: a specced server missing from the canonical is still
    emitted (synthesized command), and a canonical server with no spec is
    rebased-only.

    ``on_progress(done, total, server_key)`` fires after each server resolves. Every server is
    health-probed with a real ``conda run`` (3–40 s each), so on a box with many envs this sweep
    runs for tens of minutes; without a tick the wizard looks hung. Purely advisory — a callback
    that raises is swallowed, because this runs *after* every env is built and a display hiccup
    must not discard that work.
    """
    repo_root = repo_root or constants.repo_root()
    original = original_servers if original_servers is not None else wiring.load_original_servers()
    base_python = wiring.base_python_path(basic_env, conda=conda)
    root_envs = envs_root(conda)

    servers: dict[str, dict] = {}
    entries: list[ResolvedEntry] = []
    keys = sorted(set(original) | set(specs))
    for key in keys:
        spec = specs.get(key)
        orig_meta = original.get(key, {})
        if spec is None:
            entry = _rebase_only_entry(key, orig_meta, repo_root=repo_root, base_python=base_python)
        else:
            entry = resolve_server_entry(
                conda,
                spec,
                orig_meta,
                basic_env=basic_env,
                base_python=base_python,
                repo_root=repo_root,
                root_envs=root_envs,
            )
        servers[key] = entry.meta
        entries.append(entry)
        if on_progress is not None:
            try:
                on_progress(len(entries), len(keys), key)
            except Exception:  # progress is advisory; never lose a completed sweep to a display hiccup
                pass
    return {"mcp_servers": servers}, entries


# --------------------------------------------------------------------------- #
# Write 1 — the wizard's own setup config (never the canonical)
# --------------------------------------------------------------------------- #
def write_setup_config(
    conda: Conda,
    basic_env: str,
    specs: dict[str, ToolSpec],
    *,
    repo_root: Path | None = None,
    log: SessionLog | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, list[ResolvedEntry]]:
    """Resolve all servers and write the FULL install-aware setup config.

    Writes ``install/recipes/mcp_config.setup.yaml`` + ``install/recipes/env_overrides.env`` and
    records ``SOG_MCP_CONFIG`` (unredacted — it is a path, not a secret) in
    ``.env``. The agent's canonical config is untouched here.

    ``on_progress`` is forwarded to :func:`resolve_full_config` — see there for why the sweep
    needs to be able to report where it is.
    """
    repo_root = repo_root or constants.repo_root()
    config, entries = resolve_full_config(conda, basic_env, specs, repo_root=repo_root, on_progress=on_progress)
    return write_resolved_setup_config(config, entries, log=log)


def write_resolved_setup_config(
    config: dict,
    entries: list[ResolvedEntry],
    *,
    log: SessionLog | None = None,
) -> tuple[Path, list[ResolvedEntry]]:
    """Write the setup config + overrides from an ALREADY-resolved ``(config, entries)``.

    Split out of :func:`write_setup_config` (a1c#1) so a caller that has just resolved can
    refresh the setup config from the SAME resolution it also drives :func:`apply_to_canonical`
    with — instead of re-resolving. ``wizard._phase_finalize`` needs this: ``doctor`` runs its
    repairs there, so a fresh resolve is only then most accurate; resolving ONCE and feeding both
    writers keeps ``install/recipes/mcp_config.setup.yaml`` from going stale relative to the canonical the
    agent actually loads.
    """
    # Atomic writes (tmp + os.replace): a kill / ENOSPC mid-write must never leave a
    # half-written setup config or overrides file that a later run/agent would misread.
    cfg_path = constants.generated_mcp_config()
    llm_setup._atomic_write_text(cfg_path, yaml.safe_dump(config, sort_keys=False))

    ov_path = constants.generated_env_overrides()
    llm_setup._atomic_write_text(ov_path, wiring.env_overrides_text(config))

    _record_config_pointer(cfg_path, log=log)

    if log is not None:
        enabled = [e.server_key for e in entries if e.enabled]
        log.event(
            "setup_config_written",
            config=str(cfg_path),
            enabled_count=len(enabled),
            total=len(entries),
        )
    return cfg_path, entries


def _record_config_pointer(cfg_path: Path, *, log: SessionLog | None = None) -> None:
    """Record ``SOG_MCP_CONFIG=<cfg_path>`` in ``.env`` (idempotent, non-secret).

    Skips the backup+rewrite when the pointer is already recorded identically, so
    an idempotent re-run doesn't spew ``.env`` backups.
    """
    desired = str(cfg_path)
    dotenv = constants.dotenv_path()
    if dotenv.exists():
        try:
            # Compare the DECODED value. write_dotenv writes through _format_kv, which double-quotes a
            # value containing a space/#/$/quote -- so a raw line-split comparison against the unquoted
            # `desired` never matches for a repo path like ".../My Projects/...", and every finalize
            # would rewrite .env AND drop a fresh secret-bearing backup (never pruned) -> unbounded
            # accumulation. read_dotenv_values decodes the quoting and reads leniently (non-UTF-8 safe),
            # so the idempotency guard actually holds.
            if llm_setup.read_dotenv_values().get("SOG_MCP_CONFIG") == desired:
                return  # already recorded — nothing to do
        except Exception:
            # Any read failure must not abort the idempotency peek -> fall through to write_dotenv,
            # which reads leniently and merges, so the pointer is still recorded and other keys kept.
            pass
    llm_setup.write_dotenv({"SOG_MCP_CONFIG": desired}, backup=True, secret=False)
    if log is not None:
        log.event("mcp_config_pointer_recorded", path=desired)


# --------------------------------------------------------------------------- #
# Write 2 — rewrite the agent's canonical config from the live resolution
# --------------------------------------------------------------------------- #
def apply_to_canonical(
    config: dict,
    *,
    repo_root: Path | None = None,
    keep_agent_config: bool = False,
    log: SessionLog | None = None,
) -> Path | None:
    """Rewrite ``agent/MCP_server/mcp_config.yaml`` from ``config`` after a backup.

    Returns the backup path (``None`` if ``keep_agent_config`` or there was no
    prior file). The backup lands in the git-ignored ``.sog_setup/backups/`` with
    a timestamped, collision-proof name, so an existing hand-edited config is
    always recoverable and repeated applies never clobber each other.

    The rewritten canonical is a **per-machine** artifact (absolute, live-resolved
    paths) — regenerated per box, not meant to be committed. Callers surface that
    to the user; ``keep_agent_config=True`` opts out entirely.
    """
    root = Path(repo_root) if repo_root else constants.repo_root()
    canon = constants.agent_path(constants.ORIGINAL_MCP_CONFIG_REL, root=root)

    if keep_agent_config:
        if log is not None:
            log.event("canonical_kept", reason="keep_agent_config", path=str(canon))
        return None

    constants.ensure_state_dirs()
    backup: Path | None = None
    out = dict(config)  # shallow copy — we may layer preserved top-level keys beneath it
    if canon.exists():
        backup = _atomic_backup(canon, constants.backups_dir(), "mcp_config.yaml")
        # Preserve any top-level key the on-disk canonical carries BEYOND `mcp_servers` — a
        # hand-added global `settings:`/`version:` block, say. `resolve_full_config` emits ONLY
        # `{"mcp_servers": …}`, so a blind rewrite would silently drop those sibling keys. Seed the
        # output from the prior doc and let `config` overlay it: the freshly resolved `mcp_servers`
        # wins (and keeps its leading position), everything else survives. Best-effort — a
        # malformed/unreadable/non-mapping canonical simply isn't merged, identical to the old
        # behaviour on the shipped 1-key repo config (D-D).
        try:
            prior = yaml.safe_load(canon.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError, UnicodeDecodeError):
            prior = None
        if isinstance(prior, dict) and set(prior) - set(out):
            merged = dict(prior)
            merged.update(out)
            out = merged

    # Atomic (tmp + os.replace): the agent reads this canonical to launch every MCP server, so a
    # kill / ENOSPC mid-rewrite must never truncate it to an unparseable half-file. The pre-write
    # backup above lets the user recover a hand-edited config; atomicity protects the live swap.
    llm_setup._atomic_write_text(canon, yaml.safe_dump(out, sort_keys=False))

    if log is not None:
        log.event("canonical_applied", path=str(canon), backup=str(backup) if backup else None)
    return backup


# --------------------------------------------------------------------------- #
# Write 3 — rebase the USER-tool overlay (mcp_config_user.yaml) onto this box
# --------------------------------------------------------------------------- #
# The marker :func:`rebase_user_config` writes when it disables a server for a missing script -- and
# the only ``disabled_reason`` it will clear by re-enabling the server.
_SCRIPT_MISSING_REASON = "server script not found on this machine"


def user_config_path(repo_root: Path | None = None) -> Path:
    """The user-tool config the agent merges on this install -- the file to report on and rebase.

    ``SOG_MCP_USER_CONFIG`` first, exactly as the agent reads it (``mcp_user_config.
    resolve_user_config_path``) and the tool-creation writers write it. ``conncheck`` and
    :func:`rebase_user_config` read only the repo-anchored copy, so with the override set they
    reported on, and rebased, a file the agent never merged (hunt 2026-09-30, u37-setup-checks-9).

    An explicitly named root -- ``repo_root`` here, or the ``SOG_SETUP_REPO_ROOT`` seam -- asks about
    THAT root's copy, the same carve-out the agent's resolver makes for an injected root. The agent's
    CWD rung is deliberately not followed (see ``conncheck._load_user_servers``)."""
    from spatialomicsgym.mcp_user_config import user_config_override

    if repo_root is None and not os.environ.get("SOG_SETUP_REPO_ROOT"):
        override = user_config_override()
        if override is not None:
            return override
    root = Path(repo_root) if repo_root else constants.repo_root()
    return constants.agent_path(constants.USER_MCP_CONFIG_REL, root=root)


def rebase_user_config(
    *,
    base_python: str,
    repo_root: Path | None = None,
    conda: Conda | None = None,
    log: SessionLog | None = None,
) -> dict | None:
    """Rewrite ``agent/MCP_server/mcp_config_user.yaml`` (or the ``SOG_MCP_USER_CONFIG`` override the agent
    reads instead -- :func:`user_config_path`) so its wiring is true on THIS machine.

    The canonical config gets exactly this treatment on every finalize
    (``resolve_full_config`` → ``apply_to_canonical``); the user-tool overlay got none, so
    after a clone or a ``sog-setup unpack`` its entries kept machine-A absolute paths and a
    bare ``python`` interpreter — launchable only through ``base_mcp``'s launch-time
    self-heal, and lied about by every on-disk reader (conncheck, resync, a human).

    Per server block (declared under ``mcp_servers:`` or recovered from a stray top-level
    block, which this rewrite folds into its proper place):

    * ``command[0]`` → ``base_python`` (the thin FastMCP portal runs in the base env,
      exactly like every canonical server).
    * ``command[1]`` → ``<repo>/agent/tools_user/<basename>`` whenever that local copy exists —
      the root's copy is the authority, because on a same-box transplant a recorded path
      into ANOTHER checkout still exists and would silently launch that other tree's
      file. A path with no local copy is kept as-is when it exists (a deliberately
      out-of-tree script); missing everywhere → ``enabled: false`` plus a
      ``disabled_reason`` the merger ignores and a human/conncheck can read.
    * ``env`` pins ``<KEY>_PYTHON`` when ``<envs root>/<server_key>/bin/python`` exists
      and ``<KEY>_WORKER`` when a local worker file does (KEY = the server key upper-cased,
      the ``get_worker_paths("USER_<ID>", ...)`` convention every playbook server uses).
      A recorded pin whose path is missing here is dropped rather than kept as a lie —
      ``base_mcp``'s dispatch-time fallback chain then owns the rescue.

    Anchored via ``constants.repo_root()`` (NEVER a ``parents[N]`` walk — on a pip-only
    install the instance root is not this file's grandparent). Missing overlay → ``None``
    (a fresh clone has none; not an error). Unreadable/malformed → ``{"error": ...}`` with
    the file left untouched. Otherwise: timestamped backup (``_atomic_backup``) + atomic
    rewrite, skipped entirely when the rewrite would be byte-identical, and a summary dict
    ``{path, backup, changed, servers, recovered, pinned, disabled}``.
    """
    root = Path(repo_root) if repo_root else constants.repo_root()
    path = user_config_path(repo_root)
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        if log is not None:
            log.event("user_config_rebase_skipped", path=str(path), error=str(exc)[:200])
        return {"path": str(path), "error": f"unreadable user config left untouched: {str(exc)[:200]}"}
    if not isinstance(doc, dict):
        return {"path": str(path), "error": "user config is not a mapping; left untouched"}

    declared = doc.get("mcp_servers")
    if not isinstance(declared, dict):
        declared = {}
    try:
        recovered = wiring.merger_module()._recover_top_level_servers(doc, declared)
    except Exception:
        recovered = {}  # best-effort: an unloadable merger must not block the rebase

    try:
        root_envs = envs_root(conda) if conda is not None else constants.conda_envs_root()
    except Exception:
        root_envs = constants.conda_envs_root()

    out_servers: dict[str, dict] = {}
    pinned: dict[str, list[str]] = {}
    disabled: list[str] = []
    for key, meta in {**declared, **recovered}.items():
        if not isinstance(meta, dict):
            out_servers[key] = meta  # not a server block; carry through untouched
            continue
        entry = dict(meta)

        # -- command: [base python, local server script] --------------------------------- #
        raw_cmd = entry.get("command")
        cmd = list(raw_cmd) if isinstance(raw_cmd, list) else []
        script = str(cmd[1]) if len(cmd) >= 2 else ""
        script_path = Path(script) if script else None
        if script_path is not None and not script_path.is_absolute():
            script_path = constants.agent_root(root) / script_path  # relative to the agent part (tools_user/…)
        if script_path is not None:
            local = constants.tools_user_dir(root) / script_path.name
            if script_path != local and local.exists():
                # The root's own copy is the authority: on a same-box transplant a path
                # into ANOTHER checkout still exists, and keeping it would silently launch
                # that other tree's file. A script with no local copy is handled below.
                script_path = local
            elif not script_path.exists():
                script_path = local  # predictable home whether or not the copy exists (checked below)
        if script_path is not None:
            entry["command"] = [base_python, str(script_path), *[str(a) for a in cmd[2:]]]
            if script_path.exists():
                # Re-enable what THIS rewrite disabled, now that the script is back. Popping only the
                # reason left the ``enabled: false`` it had written, so following the reason's own advice
                # ("restore the file") left the server off for good with no reason recorded -- and the
                # merger passes a disabled server over in silence (hunt 2026-09-30, u37-setup-checks-8).
                # A user's own ``enabled: false`` carries no such marker and is left alone.
                if str(entry.get("disabled_reason") or "").startswith(_SCRIPT_MISSING_REASON):
                    entry["enabled"] = True
                entry.pop("disabled_reason", None)
            else:
                entry["enabled"] = False
                entry["disabled_reason"] = (
                    f"{_SCRIPT_MISSING_REASON} (looked for tools_user/{script_path.name}); "
                    "restore the file or re-run the tool creation playbook"
                )
                disabled.append(key)
        elif cmd:
            entry["command"] = [base_python]

        # -- env pins: <KEY>_PYTHON / <KEY>_WORKER, only where the target exists --------- #
        raw_env = entry.get("env")
        env = dict(raw_env) if isinstance(raw_env, dict) else {}
        prefix = re.sub(r"[^A-Za-z0-9]", "_", str(key)).upper()
        py_var, worker_var = f"{prefix}_PYTHON", f"{prefix}_WORKER"
        env_python = Path(constants.interp_path(f"{root_envs}/{key}", "python"))
        if env_python.exists():
            if env.get(py_var) != str(env_python):
                pinned.setdefault(key, []).append(py_var)
            env[py_var] = str(env_python)
        elif py_var in env and not Path(str(env[py_var])).exists():
            env.pop(py_var)  # a dead pin outranks the defaults; base_mcp's fallback chain owns it now
        worker = _local_user_worker(root, key, recorded=env.get(worker_var))
        if worker is not None:
            if env.get(worker_var) != str(worker):
                pinned.setdefault(key, []).append(worker_var)
            env[worker_var] = str(worker)
        elif worker_var in env and not Path(str(env[worker_var])).exists():
            env.pop(worker_var)
        if env:
            entry["env"] = env
        else:
            entry.pop("env", None)
        out_servers[key] = entry

    # Fold recovered blocks into their proper place; preserve every other top-level key.
    out = {k: v for k, v in doc.items() if k != "mcp_servers" and k not in recovered}
    out = {"mcp_servers": out_servers, **out}
    new_text = yaml.safe_dump(out, sort_keys=False)
    if new_text == text:
        if log is not None:
            log.event("user_config_rebase_unchanged", path=str(path))
        return {
            "path": str(path),
            "backup": None,
            "changed": False,
            "servers": len(out_servers),
            "recovered": sorted(recovered),
            "pinned": pinned,
            "disabled": disabled,
        }
    constants.ensure_state_dirs()
    backup = _atomic_backup(path, constants.backups_dir(), "mcp_config_user.yaml")
    llm_setup._atomic_write_text(path, new_text)
    if log is not None:
        log.event(
            "user_config_rebased",
            path=str(path),
            backup=str(backup),
            servers=len(out_servers),
            recovered=sorted(recovered),
            disabled=disabled,
        )
    return {
        "path": str(path),
        "backup": str(backup),
        "changed": True,
        "servers": len(out_servers),
        "recovered": sorted(recovered),
        "pinned": pinned,
        "disabled": disabled,
    }


def _local_user_worker(root: Path, server_key: str, *, recorded: object = None) -> Path | None:
    """This checkout's worker file for a user server, or ``None`` when it has none.

    Tried in order: the recorded ``<KEY>_WORKER`` pin's basename under ``agent/tools_user/``
    (authoritative when present — the name the server was actually created with), then the
    ``<id>_worker.py`` / ``<id>_worker.R`` convention for a ``user_<id>`` key. Only paths
    that EXIST are returned — a pin must never be written pointing at nothing.
    """
    candidates: list[str] = []
    if isinstance(recorded, str) and recorded.strip():
        candidates.append(Path(recorded).name)
    tool_id = server_key[len("user_") :] if server_key.startswith("user_") else server_key
    candidates += [f"{tool_id}_worker.py", f"{tool_id}_worker.R"]
    for name in candidates:
        cand = constants.tools_user_dir(root) / name
        if cand.exists():
            return cand
    return None


def _atomic_backup(src: Path, directory: Path, base: str) -> Path:
    """Back up ``src`` to ``<directory>/<base>.<ts>.bak``, claiming the name ATOMICALLY.

    The old two-step (pick a name with a ``while dst.exists()`` check, then a separate ``shutil.copy2``)
    had a TOCTOU: two finalizes resolving in the same wall-clock second -- overlapping runs / a shared
    NFS home -- both saw the ``.bak`` name free, and the later copy silently OVERWROTE the earlier
    pre-rewrite backup of a hand-edited canonical config. ``O_CREAT|O_EXCL`` makes the name-claim
    win-or-retry (a loser bumps the suffix), mirroring ``llm_setup.backup_dotenv`` /
    ``state.archive_state``; the 0600 fd also keeps the snapshot from being briefly world-readable.
    """
    import os

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    n = 0
    while True:
        dst = directory / (f"{base}.{ts}.bak" if n == 0 else f"{base}.{ts}.{n}.bak")
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
        try:  # don't leave a 0-byte claim masquerading as a backup on a write failure
            os.unlink(dst)
        except OSError:
            pass
        raise
    return dst
