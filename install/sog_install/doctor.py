"""
Doctor — a self-contained, read-only health report for a setup run.

``sog-setup doctor`` (or :func:`doctor`) answers "is my install actually healthy?"
without touching anything:

1. re-runs the Stage-A **preflight** (conda / disk / gpu / network / mini-data / writable),
2. inspects every ``<basic>_*`` conda env this run owns — does it exist, and does its
   import/library probe pass (the *same* read-only signal provisioning uses),
3. folds in the latest Tier-1/Tier-2 verdicts written under ``test/installation/``,
4. prints a per-tool table + a one-line roster and returns the report as a dict.

Scope note: this deliberately does **not** call
``tools_user.knowledge_manager.health_check_tools`` — that reports only on
*user-created* tools (via ``install_log.json``) and imports the heavy agent stack,
whereas the wizard needs a stdlib view of the whole provisioned ``<basic>_*`` fleet.
Nothing here writes to a conda env or the agent's config.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING

from . import constants, preflight
from .decisions import BuildStrategy
from .envtools import Conda, CondaError
from .prompts import PromptIO
from .provision import _healthy, effective_tool_env
from .specs import load_all_specs
from .state import SetupState

if TYPE_CHECKING:
    from .specs import ToolSpec

_OK, _WARN, _FAIL = "ok", "warn", "fail"


# --------------------------------------------------------------------------- #
# Read the last test verdicts (written by testing.py)
# --------------------------------------------------------------------------- #
def _load_test_verdicts() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(tier1_by_server, tier2_by_category)`` from the artifact dir.

    Prefers ``summary.json``; falls back to per-server ``<server>.json``. Missing
    files just mean "not tested yet" — never an error.
    """
    tier1: dict[str, str] = {}
    tier2: dict[str, str] = {}
    adir = constants.artifact_dir()
    summ = adir / "summary.json"
    if summ.exists():
        try:
            doc = json.loads(summ.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            # A non-UTF-8 file raises UnicodeDecodeError (a ValueError, NOT an OSError/JSONDecodeError);
            # without it a torn/foreign summary.json would escape this reader whose contract is
            # "missing/unreadable just means not-tested-yet, never an error". Fall through to per-server.
            doc = None
        # Shape-guard every level: a valid-but-skewed summary (`{"tier1": null}` or a top-level list
        # from a hand-edit/partial sync) must not `.get`-crash with AttributeError past the guard above.
        # A readable dict is authoritative (return what we can extract); anything else falls through.
        if isinstance(doc, dict):
            t1 = doc.get("tier1")
            if isinstance(t1, dict) and isinstance(t1.get("servers"), dict):
                tier1 = dict(t1["servers"])
            t2 = doc.get("tier2")
            if isinstance(t2, dict) and isinstance(t2.get("categories"), dict):
                tier2 = dict(t2["categories"])
            return tier1, tier2
    if adir.exists():
        for p in sorted(adir.glob("*.json")):
            if p.name == "summary.json":
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
            # `d.get(...)` below is OUTSIDE the try: a stray top-level-list/scalar `*.json` (valid JSON,
            # wrong shape) would AttributeError here. Guard the shape rather than widen the try.
            if isinstance(d, dict) and d.get("server"):
                tier1[d["server"]] = d.get("status", "?")
    return tier1, tier2


# --------------------------------------------------------------------------- #
# Per-env health probe (read-only)
# --------------------------------------------------------------------------- #
def _env_report(conda: Conda, basic_env: str, spec: ToolSpec | None, env_name: str) -> dict:
    # NONE-strategy servers (e.g. spatial_library, eval) run *inside* the base env — they never get a
    # dedicated `<basic>_<server>` env, so its absence is the correct, healthy state, not a broken
    # install. Provisioning already counts them ready (``ToolStatus.NONE`` → ok) and Tier-1 SKIPs them;
    # the doctor must agree, or a NONE-only install renders as `[✗] absent` / "0/ N healthy" and
    # `sog-setup doctor` exits 1 on a perfectly good machine. Its ``import_check`` is not a real pip
    # module (the worker uses base-env deps), so we trust the base env's presence rather than probe it.
    if spec is not None and spec.build_strategy is BuildStrategy.NONE:
        base_ok = conda.env_exists(basic_env)
        return {
            "env": basic_env,
            "exists": base_ok,
            "healthy": base_ok,
            "detail": "runs in base env (no dedicated env)" if base_ok else "base env absent",
        }
    # Env-reuse (R1): when a tool's managed <basic>_<server> env was never built because an existing
    # healthy env already satisfies it, the config is wired read-only to that reused env. Probe the env
    # the config actually points at — the managed env when built, else the reused env — mirroring
    # mcp_resolver.resolve_interpreter. Without this, every reused tool renders as [✗] absent and
    # `sog-setup doctor` exits 1 on a perfectly healthy box (the caller passes the managed name).
    if spec is not None:
        env_name = effective_tool_env(conda, spec, basic_env)
    exists = conda.env_exists(env_name)
    if not exists:
        return {"env": env_name, "exists": False, "healthy": False, "detail": "absent"}
    if spec is None or not spec.import_check:
        return {"env": env_name, "exists": True, "healthy": True, "detail": "present (no probe)"}
    try:
        ok = _healthy(conda, spec, env_name)
    except Exception:
        # A slow/loaded box can make the import probe raise CondaError (envtools maps a conda-run
        # TimeoutExpired/OSError → CondaError even at check=False). Report THIS env as FAILED rather than
        # let one probe abort the whole per-tool loop — which would leave `sog-setup doctor` with no health
        # table at all (cli.main maps CondaError → exit 3, discarding every other env's health) and skip the
        # wizard's post-repair config refresh, leaving a stale canonical mcp_config. Mirrors
        # provision.py:148-155's per-candidate `_healthy` guard ("that must never abort provisioning").
        ok = False
    return {
        "env": env_name,
        "exists": True,
        "healthy": ok,
        "detail": f"import {spec.import_check}: {'ok' if ok else 'FAILED'}",
    }


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def doctor(
    basic_env: str,
    *,
    servers: list[str] | None = None,
    specs: dict[str, ToolSpec] | None = None,
    conda: Conda | None = None,
    io: PromptIO | None = None,
    check_net: bool = True,
) -> dict:
    """Build (and print) the health report for ``basic_env``. Returns the report dict."""
    io = io or PromptIO()
    conda = conda or Conda()
    specs = specs if specs is not None else load_all_specs()

    io.banner("SPATIALOMICSLAB — DOCTOR", f"health report for base env '{basic_env}'")

    # 1) preflight
    io.section("Machine preflight")
    # L1: doctor is a READ-ONLY health report — prune=False so the writability probe never unlinks
    # the user's accumulated run logs just because they inspected a run.
    pf = preflight.run_preflight(check_net=check_net, prune=False)
    for r in pf:
        (io.ok if r.level == _OK else io.warn if r.level == _WARN else io.err)(f"{r.name}: {r.detail}")
    pf_summary = preflight.summarize(pf)

    # 2) which envs to inspect: the ones this run owns
    if servers is None:
        # discover from live envs matching the namespace. Exclude PROTECTED_ENVS exactly as ``reset``
        # does: under base ``sog`` the unrelated real ``sog_reproduce`` env matches ``sog_*`` and would
        # otherwise be reported as a healthy tool env "reproduce", inflating the roster (F-F2).
        try:
            all_envs = conda.env_list()
        except CondaError as exc:
            # First conda enumeration on a cold Conda instance can raise (TimeoutExpired/OSError) even at
            # check=False. A read-only health report must still render its table — degrade to "none
            # discovered" rather than unwinding to cli.main's CondaError→exit 3, which would ALSO
            # misdiagnose a slow/loaded conda as "not on PATH". Mirrors the per-env _healthy guard above.
            io.warn(f"could not enumerate conda envs ({str(exc)[:120]}) — reporting base env only")
            all_envs = []
        servers = sorted(
            e[len(basic_env) + 1 :]
            for e in all_envs
            if e.startswith(f"{basic_env}_") and e not in constants.PROTECTED_ENVS
        )
    try:
        base_exists = conda.env_exists(basic_env)
    except CondaError as exc:
        io.warn(f"could not query base env '{basic_env}' ({str(exc)[:120]}) — treating as not confirmed")
        base_exists = False

    # 3) per-tool table (health probe + last test verdict)
    tier1, tier2 = _load_test_verdicts()
    io.section(f"Base env '{basic_env}': {'present' if base_exists else 'ABSENT'}")
    io.section("Tool environments")
    rows = []
    healthy_n = 0
    for key in servers:
        env_name = constants.tool_env_name(basic_env, key)
        try:
            rep = _env_report(conda, basic_env, specs.get(key), env_name)
        except CondaError as exc:
            # _env_report calls conda.env_exists (line 100/114). On a cold/slow box the very first conda
            # call raises (envtools maps a conda-run TimeoutExpired/OSError → CondaError even at check=False),
            # AND a failed cold call leaves _env_map's cache empty, so a later call re-attempts it. If the
            # guarded env_list/env_exists above already degraded (cache still empty), this per-env probe is
            # the next cold attempt and would raise UNGUARDED — aborting the whole loop to cli.main's
            # CondaError→exit 3, discarding every other env's health. Degrade THIS row to unknown instead,
            # mirroring the _healthy per-probe guard (line 119-128).
            io.warn(f"could not probe env '{env_name}' ({str(exc)[:120]}) — reporting as unknown")
            rep = {"env": env_name, "exists": False, "healthy": False, "detail": "probe error (conda unavailable)"}
        rep["server"] = key
        rep["tier1"] = tier1.get(key, "-")
        if rep["healthy"]:
            healthy_n += 1
        rows.append(rep)
        mark = constants.CHECK_DONE if rep["healthy"] else constants.CHECK_FAIL
        io.say(f"  {mark} {key:24s} {rep['detail']:28s} test:{rep['tier1']}")
    if not rows:
        io.note("no tool envs found in this namespace yet")

    report = {
        "basic_env": basic_env,
        "base_env_present": base_exists,
        "preflight": pf_summary,
        "tools": rows,
        "tier2": tier2,
        "summary": {"tools": len(rows), "healthy": healthy_n, "unhealthy": len(rows) - healthy_n},
    }

    io.section("Summary")
    io.say(f"  {healthy_n}/{len(rows)} tool env(s) healthy; base env {'present' if base_exists else 'ABSENT'}")
    if tier2:
        passed = sum(1 for v in tier2.values() if v == "PASS")
        io.say(f"  Tier-2: {passed}/{len(tier2)} category pipeline(s) passed")
    # ``summarize`` is keyed by check *name* ({name: {level, detail, hint}}), not by level — so
    # count the hard failures off the raw results instead of the never-present ``["fail"]`` key
    # (the old ``pf_summary.get("fail")`` silently never fired, hiding a broken preflight).
    n_fail = sum(1 for r in pf if r.level == _FAIL)
    # Surface the preflight hard-failure count at the report top level (beside ``base_env_present``) so
    # the CLI exit code can reflect it — a broken machine (failed net/disk/conda preflight) must NOT
    # report exit 0 just because no tool env happens to be marked unhealthy. Kept OUT of ``summary`` so
    # that dict stays the {tools, healthy, unhealthy} triple callers assert on.
    report["preflight_failures"] = n_fail
    if n_fail:
        io.warn(f"preflight has {n_fail} hard failure(s) — see above")
    return report


# --------------------------------------------------------------------------- #
# CLI: sog-setup doctor
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sog-setup doctor", description="Read-only health report for a setup run.")
    ap.add_argument("--base", help="basic env name (defaults to the last run's, from state)")
    ap.add_argument("--no-net", action="store_true", help="skip the network preflight probe")
    ap.add_argument("--json", action="store_true", help="print the report as JSON (in addition to the table)")
    args = ap.parse_args(argv)

    basic_env = args.base
    servers: list[str] | None = None
    if not basic_env:
        st = SetupState.load()
        if st and st.basic_env_name:
            basic_env = st.basic_env_name
            servers = st.selected_servers or None
    if not basic_env:
        print("no base env given and no saved state found — pass --base <name>")
        return 2

    report = doctor(basic_env, servers=servers, check_net=not args.no_net)
    if args.json:
        print(json.dumps(report, indent=2))
    # Exit 0 only when the box is actually healthy: no unhealthy tool env, the base env is present,
    # and the machine preflight has no hard failures. The old check looked at ``unhealthy`` alone, so
    # a box with the base env ABSENT (or a failing net/disk/conda preflight) but zero built tool envs
    # reported exit 0 — a false green that hid a broken deploy. A caller/CI reads this exit code.
    healthy = (
        report["summary"]["unhealthy"] == 0
        and report.get("base_env_present", False)
        and report.get("preflight_failures", 0) == 0
    )
    return 0 if healthy else 1


if __name__ == "__main__":
    # L2: route the direct ``python -m sog_install.doctor`` entry through the same friendly
    # cli wrapper as ``sog-setup doctor`` (KeyboardInterrupt→130, CondaError→3, any other error→a
    # one-line message + 1). ``doctor.main`` wraps nothing, so without this a Ctrl-C — or simply a box
    # with no conda on PATH, which the wizard actively steers people here to diagnose — would dump a
    # raw traceback. ``cli`` imports doctor lazily, so this back-import does not cycle.
    import sys

    from .cli import main as _cli_main

    raise SystemExit(_cli_main(["doctor", *sys.argv[1:]]))
