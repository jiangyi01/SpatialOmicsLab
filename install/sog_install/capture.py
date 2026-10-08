"""
Capture — freeze the live per-tool envs into committed, rebuildable specs.

Run **once**, on a host that already has the worker envs (this repo's dev box):

    sog-setup capture            # all servers, with hybrid env exports
    sog-setup capture --no-export --only cell2location,tangram

For each MCP **server** it emits ``install/recipes/tool_specs/<server>.yaml`` (a
:class:`~sog_install.specs.ToolSpec`) so a fresh clone can rebuild the
env without the original machine. The load-bearing facts it records that
``registry.build_registry`` does *not*:

* the exact worker **override var-names** — ``get_worker_paths`` always reads
  ``{PREFIX}_PYTHON`` / ``{PREFIX}_WORKER`` (even for R tools, where the
  ``_PYTHON`` var holds an *Rscript* path), while a few servers (seurat) read a
  literal ``SEURAT_RSCRIPT`` / ``SEURAT_WORKER``. Not derivable by rule → parsed
  from the server source with :mod:`ast` (a read-only, non-invasive scan; we
  never edit ``registry.py`` or the server files).
* the worker **basename**, so wiring can repoint it to the clone (cell2location
  hardcodes an absolute worker default that is wrong on a fresh checkout).

Build strategy per server: ``eval`` → ``none``; a shipped ``agent/tools_user/*_env.yaml``
→ ``env_yaml``; otherwise ``conda_clone`` of the present source env (fast +
faithful on the same host) with a pinned ``conda env export`` stashed under
``env/<server>.env.yaml`` as the ``conda_export`` fallback recipe.

Stdlib + pyyaml only.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from . import constants, credentials, session_log
from .decisions import BuildStrategy
from .envtools import Conda
from .specs import FunctionSpec, ToolSpec, load_spec, write_spec

if TYPE_CHECKING:
    from .session_log import SessionLog

# Var-name suffixes we care about in the AST scan.
_INTERP_SUFFIXES = ("_PYTHON", "_RSCRIPT")
_WORKER_SUFFIX = "_WORKER"


@dataclass
class WorkerVars:
    """What the AST scan pulls out of one ``tools/<x>_mcp_server.py``."""

    interp_var: str = ""
    interp_default: str = ""
    worker_var: str = ""
    worker_default: str = ""


# --------------------------------------------------------------------------- #
# AST scan (read-only; never mutates the server files or registry.py)
# --------------------------------------------------------------------------- #
def _const_str(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def scan_worker_vars(server_file: Path) -> WorkerVars:
    """Parse a server module for its interpreter/worker override var-names."""
    wv = WorkerVars()
    try:
        tree = ast.parse(server_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return wv
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        fn = call.func
        # get_worker_paths(PREFIX, default_interp, default_worker)
        if isinstance(fn, ast.Name) and fn.id == "get_worker_paths" and len(call.args) >= 3:
            prefix = _const_str(call.args[0])
            if prefix:
                wv.interp_var = f"{prefix}_PYTHON"  # the impl always uses _PYTHON
                wv.interp_default = _const_str(call.args[1]) or ""
                wv.worker_var = f"{prefix}_WORKER"
                wv.worker_default = _const_str(call.args[2]) or ""
        # os.environ.get("LITERAL_KEY", default)
        elif isinstance(fn, ast.Attribute) and fn.attr == "get" and call.args:
            key = _const_str(call.args[0])
            if not key:
                continue
            default = _const_str(call.args[1]) if len(call.args) > 1 else ""
            if key.endswith(_INTERP_SUFFIXES) and not wv.interp_var:
                wv.interp_var, wv.interp_default = key, default or ""
            elif key.endswith(_WORKER_SUFFIX) and not wv.worker_var:
                wv.worker_var, wv.worker_default = key, default or ""
    return wv


# --------------------------------------------------------------------------- #
# Function → category (skill) index
# --------------------------------------------------------------------------- #
def _function_category_index() -> dict[str, str]:
    """Map each ``mcp_function`` to its owning skill name (best-effort)."""
    try:
        from skills.registry import SkillRegistry
    except Exception:
        return {}
    reg = SkillRegistry.create_default()
    out: dict[str, str] = {}
    for name in reg.list_skills():
        for v in reg.get_skill(name).get_mcp_tool_mapping().values():
            fn = v.get("mcp_function")
            if fn and fn not in out:
                out[fn] = name
    return out


# --------------------------------------------------------------------------- #
# Recipe / strategy resolution
# --------------------------------------------------------------------------- #
def _tools_user_recipe(server_key: str) -> str | None:
    """Return a repo-relative path to a shipped ``agent/tools_user/<key>_env.yaml``, if any."""
    p = constants.tools_user_dir() / f"{server_key}_env.yaml"
    return f"{constants.AGENT_DIRNAME}/tools_user/{server_key}_env.yaml" if p.exists() else None


def _strip_export(text: str) -> str:
    """Drop ``name:``/``prefix:`` lines from a ``conda env export`` dump."""
    keep = [ln for ln in text.splitlines() if not ln.startswith(("name:", "prefix:"))]
    return "\n".join(keep).rstrip() + "\n"


# ``conda env export`` copies channel/pip URLs verbatim from ``.condarc`` and the env, so a raw dump
# can carry credentials into the committed ``tool_specs/env/<key>.env.yaml`` recipe:
#  * Anaconda.org token channels embed the token in the path — ``.../t/<TOKEN>/<channel>``;
#  * a pip ``--extra-index-url`` / VCS pin can embed URL userinfo — ``scheme://<user[:pass]>@host``;
#  * a private index (artifactory/cloudsmith/devpi) can put a token in the QUERY STRING — ``?token=<T>``.
# ``capture_all`` registers the env's configured index-URL credentials so ``redact()`` masks any it
# recognizes, but a ``.condarc``-only token never reaches that env scan — so mask all three URL shapes
# structurally too. (A token in a bare PATH SEGMENT — ``…/<TOKEN>/org/…`` — can't be masked without
# over-redacting legitimate channel paths, so that shape stays the maintainer's diff-review responsibility.)
_CONDA_TOKEN_RE = re.compile(r"(/t/)[^/\s]+(/)")  # anaconda.org token channel: .../t/<token>/<channel>
_URL_USERINFO_RE = re.compile(r"(://)[^/\s@]+(@)")  # URL userinfo: scheme://<user[:pass]>@host
# credential query params (private artifactory/cloudsmith/devpi indices); value runs to the next &, ws, or quote.
# Deliberately high-precision names only — `api-version`/`label`/`channel` and a bare `key=` are NOT masked.
_URL_QUERY_TOKEN_RE = re.compile(
    r"([?&](?:token|api[-_]?key|access[-_]?token|sig|signature|password|secret)=)[^&\s\"']+",
    re.IGNORECASE,
)


def _scrub_export_secrets(text: str) -> str:
    """Mask any credential a ``conda env export`` may have copied verbatim, before the dump is written
    to the committed spec tree: registered secrets (via :func:`session_log.redact`), Anaconda.org
    channel tokens (``/t/<token>/``), URL-embedded ``user:pass@`` credentials, and credential
    query-string params (``?token=``/``?api_key=``…). ASCII round-trips unchanged; a clean public recipe
    (conda-forge, download.pytorch.org, data.pyg.org) is untouched."""
    text = session_log.redact(text)
    text = _CONDA_TOKEN_RE.sub(r"\1****\2", text)
    text = _URL_USERINFO_RE.sub(r"\1****\2", text)
    text = _URL_QUERY_TOKEN_RE.sub(r"\1****", text)
    return text


# A `conda env export` is a *lossy* rendering of an env: it cannot express a pip index directive
# (`--extra-index-url`, `--find-links`) and it renders a `git+<url>@<sha>` install as a bare
# distribution name. Both are load-bearing in the curated recipes -- without the index directive the
# `+cpu`/`+cu118` wheels do not resolve at all, and a bare name is uninstallable where no PyPI
# project exists and *wrong* where an unrelated one does (PyPI `stride` is not STRIDE). An export
# also stamps the exporting host's exact conda build strings onto every dependency.
#
# So capture writes to the same paths a maintainer curates by hand, and the two must be told apart:
# whatever capture generated carries this marker and is freely refreshed on the next run; a recipe
# without it was written by a human and is left exactly as found.
GENERATED_MARKER = "GENERATED by `sog-setup capture`"
_GENERATED_HEADER = (
    f"# {GENERATED_MARKER} -- a pinned `conda env export` of the source env. Safe to regenerate.\n"
    "# A recipe WITHOUT this header is hand-curated (pip --extra-index-url / --find-links directives,\n"
    "# git+<url>@<sha> pins) -- things an export cannot round-trip -- and capture will not overwrite it.\n"
    "# To re-export a curated recipe deliberately, delete it first.\n"
)


def _is_generated_recipe(path: Path) -> bool:
    """Whether ``path`` is a recipe capture itself wrote. Unreadable → treated as curated, because
    the safe answer to "is this a human's file?" is yes."""
    try:
        with path.open(encoding="utf-8", errors="replace") as fh:
            head = fh.read(len(_GENERATED_HEADER) + 256)
    except OSError:
        return False
    return GENERATED_MARKER in head


def export_env_recipe(conda: Conda, env_name: str, server_key: str, *, log: SessionLog | None = None) -> str | None:
    """Write a pinned ``conda env export`` (name/prefix stripped, credentials scrubbed) as the export
    fallback. The recipe lands in the *committed* spec tree, so any channel/index token a ``conda env
    export`` copied from ``.condarc``/the env is masked first (see :func:`_scrub_export_secrets`).

    A curated recipe already at that path is preserved and returned unchanged -- see
    :data:`GENERATED_MARKER`. The export is not even run in that case: it is a conda subprocess with
    a five-minute timeout whose result would only be discarded.
    """
    d = constants.spec_dir() / "env"
    out = d / f"{server_key}.env.yaml"
    rel = f"{constants.SPEC_DIRNAME}/env/{server_key}.env.yaml"
    if out.exists() and not _is_generated_recipe(out):
        if log is not None:
            log.event("capture_recipe_preserved", server=server_key, recipe=rel)
        return rel
    dump = conda.export_full(env_name)
    if not dump.strip():
        return None
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(_GENERATED_HEADER + _scrub_export_secrets(_strip_export(dump)), encoding="utf-8")
    return rel


def _committed_spec(server_key: str) -> ToolSpec | None:
    """The spec already committed for this server, or ``None`` (absent, unreadable or invalid).

    Several of its fields are reviewed decisions a re-capture has no evidence to overturn: ``gpu`` and
    ``est_gb`` (capture cannot measure either), a ``build_strategy: none`` set by hand for a server
    that runs in the agent env, and the recipe pointer when this run produced no export. An invalid
    spec yields ``None`` for the reason :func:`_reviewed_import_check` gives."""
    p = constants.spec_dir() / f"{server_key}.yaml"
    if not p.exists():
        return None
    try:
        return load_spec(p)
    except Exception:
        return None


def _kept_recipe(server_key: str, committed: ToolSpec | None) -> str | None:
    """The recipe a capture that exported nothing should keep: the committed pointer, else an
    ``env/<server>.env.yaml`` already on disk, else ``None``."""
    if committed is not None and committed.recipe:
        return committed.recipe
    if (constants.spec_dir() / "env" / f"{server_key}.env.yaml").exists():
        return f"{constants.SPEC_DIRNAME}/env/{server_key}.env.yaml"
    return None


def _reviewed_import_check(server_key: str) -> str:
    """The ``import_check`` already committed for this server, if any.

    ``import_check`` is the module the installer imports to decide an env is healthy, and which
    module that should be is a *reviewed* decision rather than a derivable fact: a worker-bound tool
    with no installable package probes the ``__future__`` sentinel on purpose, and a tool that does
    ship a package probes that package -- never a generic dependency such as ``h5py`` or ``anndata``,
    which is present whether or not the tool itself installed. The registry's ``worker_package``
    guess carries none of that history, so it is used only where nothing has been reviewed yet.

    An unreadable or invalid spec yields ``""`` and falls back to the guess: capture is how a broken
    spec tree gets rebuilt, so it must not be the thing that a broken spec tree stops.
    """
    p = constants.spec_dir() / f"{server_key}.yaml"
    if not p.exists():
        return ""
    try:
        return load_spec(p).import_check or ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Build one spec from a registry record
# --------------------------------------------------------------------------- #
def build_spec(
    rec,
    *,
    conda: Conda,
    fn_category: dict[str, str],
    env_counts: dict[str, int],
    do_export: bool,
    log: SessionLog | None = None,
    warnings: list[str] | None = None,
) -> ToolSpec:
    """One spec from one registry record. ``warnings``, when given, receives a line for anything this
    run was asked to do and could not (today: an export that produced nothing)."""
    server_key = rec.server_key
    worker_kind = "rscript" if str(rec.language).lower() in ("r", "rscript") else "python"

    wv = scan_worker_vars(rec.server_file)
    # Prefer the interpreter path the registry already resolved (rec.interpreter);
    # fall back to the AST default.
    interp_path = rec.interpreter or wv.interp_default
    override_vars = {wv.interp_var: interp_path} if wv.interp_var and interp_path else {}

    # strategy + recipe. What the committed spec already says wins wherever this run has no new
    # evidence (hunt 2026-09-30, u35b-setup-state-8/9): `--no-export`, or a source env this box does not
    # have, used to write `recipe: null` over the committed pointer -- and provision adds the export
    # fallback only `if spec.recipe`, so a fresh clone could no longer build that tool -- and a reviewed
    # `build_strategy: none` (an agent-core server) flipped to a clone of the protected agent env.
    committed = _committed_spec(server_key)
    source_env = rec.conda_env or None
    if server_key == "eval":
        strategy, recipe = BuildStrategy.NONE, None
    elif committed is not None and committed.build_strategy is BuildStrategy.NONE:
        strategy, recipe, source_env = BuildStrategy.NONE, None, committed.source_env
    elif (r := _tools_user_recipe(server_key)) is not None:
        strategy, recipe = BuildStrategy.ENV_YAML, r
    elif source_env and conda.env_exists(source_env):
        strategy = BuildStrategy.CONDA_CLONE
        if do_export:
            recipe = export_env_recipe(conda, source_env, server_key, log=log)
            if recipe is None:
                # A `conda env export` that exits non-zero answers "" -> None, and that `recipe: null`
                # then overwrote the committed pointer and exited 0: the u35b-setup-state-8 defect by a
                # third road (hunt 2026-09-30, rc-setup-state capture minor). Keep it, and say so.
                recipe = _kept_recipe(server_key, committed)
                kept = f"kept {recipe}" if recipe else "no recipe recorded"
                if warnings is not None:
                    warnings.append(f"{server_key}: `conda env export -n {source_env}` produced nothing; {kept}")
                if log is not None:
                    log.event("capture_export_failed", server=server_key, source_env=source_env, kept_recipe=recipe)
        else:
            recipe = _kept_recipe(server_key, committed)
    elif committed is not None and committed.recipe:
        strategy, recipe = committed.build_strategy, committed.recipe  # the source env is not on this box
    else:
        strategy, recipe = BuildStrategy.CONDA_EXPORT, _kept_recipe(server_key, None)  # best we can record

    # service keys the tool needs (scan server + worker source)
    text = _read_text(rec.server_file) + _read_text(rec.worker_file)
    svc_keys = credentials.service_keys_in(text)

    functions = [
        FunctionSpec(name=fn, category=fn_category.get(fn, "")) for fn in (rec.func_names or [rec.spatialomicsgym_name])
    ]

    spec = ToolSpec(
        server_key=server_key,
        worker_kind=worker_kind,
        env_prefix=wv.interp_var.rsplit("_", 1)[0] if wv.interp_var else "",
        override_vars=override_vars,
        worker_var=wv.worker_var,
        worker_file=Path(rec.worker_file).name if rec.worker_file else "",
        server_file=Path(rec.server_file).name if rec.server_file else "",
        source_env=source_env,
        # Reviewed answer first; the registry's derived guess only for a server nothing has
        # committed yet (see :func:`_reviewed_import_check`).
        import_check=_reviewed_import_check(server_key) or (getattr(rec, "worker_package", "") or ""),
        build_strategy=strategy,
        recipe=recipe,
        # Reviewed by hand; capture cannot measure either (hunt 2026-09-30, u35b-setup-state-9). A
        # hard-coded False turned cast's GPU-shaped failure on a CPU box from SKIP into a FAIL.
        gpu=committed.gpu if committed is not None else False,
        shared_env=env_counts.get(source_env or "", 0) > 1,
        est_gb=committed.est_gb if committed is not None else None,
        service_keys=[k.env_var for k in svc_keys],
        functions=functions,
    )
    if log is not None:
        log.event(
            "capture_spec",
            server=server_key,
            strategy=strategy.value,
            interp_var=wv.interp_var,
            source_env=source_env,
            shared=spec.shared_env,
        )
    return spec


def _read_text(p) -> str:
    try:
        return Path(p).read_text(encoding="utf-8") if p else ""
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Top-level capture
# --------------------------------------------------------------------------- #
def write_index(idx_path: Path, entries: dict[str, dict], *, merge: bool) -> dict[str, dict]:
    """Write ``index.yaml``; with ``merge`` the entries are laid OVER the index already there.

    ``--only`` captures a subset of servers, and the index used to be written from that subset
    alone -- so ``sog-setup capture --only pathway_enrichment`` replaced an 88-entry index with a
    one-entry one (2026-09-20, caught by a lens that had asserted only its own key). A partial
    capture updates the entries it captured and keeps the rest; a full capture is authoritative
    and drops entries whose server is gone. Returns what was written.
    """
    index: dict[str, dict] = {}
    if merge and idx_path.is_file():
        try:
            existing = yaml.safe_load(idx_path.read_text(encoding="utf-8")) or {}
            if isinstance(existing, dict) and isinstance(existing.get("servers"), dict):
                index.update(existing["servers"])
        except Exception:
            index = {}
    index.update(entries)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(yaml.safe_dump({"servers": index}, sort_keys=True), encoding="utf-8")
    return index


def capture_all(
    *,
    only: list[str] | None = None,
    do_export: bool = True,
    conda: Conda | None = None,
    log: SessionLog | None = None,
) -> dict:
    """Capture every server (or ``only`` a subset) → ``install/recipes/tool_specs/*.yaml``.

    Returns a summary dict. Read-only w.r.t. envs and the repo's runtime code —
    it only *writes* new files under ``install/recipes/tool_specs/``. The server registry comes
    from the development tree's ``test/smoke/registry.py`` (loaded by path); without ``test/``
    capture stops with that sentence rather than an import error.
    """
    try:
        build_registry = constants.load_test_module("smoke.registry").build_registry  # heavy-ish; lazily
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"sog-setup capture reads the server registry from test/smoke: {exc}") from exc

    conda = conda or Conda()
    # Make redact() live in a standalone `sog-setup capture` process — nothing else registers secrets
    # here (the wizard's startup registration doesn't run), so register the credentials embedded in any
    # configured pip/conda/uv index-URL env var. Then _scrub_export_secrets can mask them via redact()
    # even when they aren't URL-userinfo / anaconda-token shaped. Best-effort; never raises.
    session_log.register_index_url_secrets()
    fn_category = _function_category_index()
    # build_registry yields one record per config tool-entry, so a multi-function
    # server appears several times with an identical (complete) ``func_names``.
    # Dedup to one record per server_key (first wins) — nothing is lost.
    seen_keys: set[str] = set()
    recs = [r for r in build_registry() if not (r.server_key in seen_keys or seen_keys.add(r.server_key))]
    # The registry now unions in mcp_config_user.yaml (origin == "user"), but the spec dir is
    # the 88-builtin freeze: user tools transplant via their own layer (tools_user/ yamls +
    # `sog-setup pack`), and writing user_* specs here would commit machine-local state.
    # Filtered BEFORE ``only`` so even an explicit --only user_x stays a skip, not a spec.
    user_skipped = sorted(r.server_key for r in recs if getattr(r, "origin", "builtin") == "user")
    recs = [r for r in recs if getattr(r, "origin", "builtin") != "user"]

    # Counted over every builtin server, BEFORE `only`: `--only squidpy` used to see moscot used once
    # and write `shared_env: false` over the committed `true` (hunt 2026-09-30, u35b-setup-state-10).
    env_counts: dict[str, int] = {}
    for r in recs:
        if r.conda_env:
            env_counts[r.conda_env] = env_counts.get(r.conda_env, 0) + 1

    if only:
        wanted = set(only)
        recs = [r for r in recs if r.server_key in wanted]

    written: list[str] = []
    warnings: list[str] = []
    index: dict[str, dict] = {}
    for rec in recs:
        try:
            spec = build_spec(
                rec,
                conda=conda,
                fn_category=fn_category,
                env_counts=env_counts,
                do_export=do_export,
                log=log,
                warnings=warnings,
            )
        except Exception as exc:
            # build_spec calls conda.env_exists / conda.export_full, both of which reach _exec(check=False)
            # and so raise CondaError on a TimeoutExpired/OSError (a slow/loaded box, or a large env whose
            # `conda env export` exceeds CONDA_EXPORT_TIMEOUT_SEC). One env's failure must NOT abort the whole
            # capture — that would leave the specs already written on disk while index.yaml (written only
            # after this loop) is never refreshed → a partial, internally-inconsistent committed spec tree.
            # Record it (redacted; exit-nonzero via `warnings`) and skip to the next server.
            warnings.append(f"{rec.server_key}: {session_log.redact(str(exc))}")
            continue
        try:
            spec.validate()
        except Exception as exc:
            warnings.append(f"{rec.server_key}: {exc}")
        # write_spec serializes only ToolSpec's fields, so rewriting an unchanged spec drops its
        # reviewers' YAML comments and notes (cast's recipe_notes) for nothing -- leave it as found.
        committed = _committed_spec(spec.server_key)
        if committed is None or committed.to_dict() != spec.to_dict():
            write_spec(spec)
        written.append(spec.server_key)
        index[spec.server_key] = {
            "strategy": spec.build_strategy.value,
            "source_env": spec.source_env,
            "worker_kind": spec.worker_kind,
            "shared_env": spec.shared_env,
            "has_override": bool(spec.override_vars),
            "service_keys": spec.service_keys,
        }

    idx_path = constants.spec_dir() / "index.yaml"
    write_index(idx_path, index, merge=bool(only))

    summary = {
        "captured": len(written),
        "servers": written,
        "warnings": warnings,
        # Informational, never in ``warnings``: a skipped user server is the designed outcome,
        # not a capture failure, and must not flip the exit code.
        "user_skipped": user_skipped,
        "missing_override": [k for k, v in index.items() if not v["has_override"]],
        "index": str(idx_path),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sog-setup capture", description="Freeze live tool envs into specs.")
    ap.add_argument("--only", help="comma-separated server keys to capture")
    ap.add_argument("--no-export", action="store_true", help="skip the pinned env export (clone-only specs)")
    args = ap.parse_args(argv)

    only = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None
    summary = capture_all(only=only, do_export=not args.no_export)
    print(f"captured {summary['captured']} spec(s) -> {constants.spec_dir()}")
    if summary.get("user_skipped"):
        skipped = summary["user_skipped"]
        print(f"  -- skipped {len(skipped)} user server(s) (owned by the tools_user layer): {', '.join(skipped)}")
    if summary["missing_override"]:
        print(f"  no interpreter override for: {', '.join(summary['missing_override'])}")
    if summary["warnings"]:
        print(f"  {len(summary['warnings'])} warning(s):")
        for w in summary["warnings"][:20]:
            print(f"    - {w}")
    # ``warnings`` holds ``spec.validate()`` failures — an invalid spec that a fresh clone cannot
    # rebuild — a server whose capture raised, and an export that was asked for and produced nothing
    # (hunt 2026-09-30, rc-setup-state). Exit non-zero so a CI/scripted ``sog-setup capture`` catches
    # it, instead of returning a false green after writing broken specs. (``missing_override`` is informational —
    # NONE-strategy servers legitimately have none — so it does NOT affect the exit code.)
    return 1 if summary["warnings"] else 0


if __name__ == "__main__":
    # L2: same friendly cli wrapper as ``sog-setup capture`` (KeyboardInterrupt→130, CondaError→3,
    # any other error→a one-line message + 1). ``capture.main`` wraps nothing.
    import sys

    from .cli import main as _cli_main

    sys.exit(_cli_main(["capture", *sys.argv[1:]]))
