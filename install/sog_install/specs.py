"""
The per-server install-spec schema — the bridge from ``capture`` to ``provision``.

A fresh clone has no per-tool conda envs and no reproducible recipe for them
(only the 20 ``tools_user/*_env.yaml`` ship full recipes; the rest are assumed
pre-existing). ``capture`` (run once on a host that HAS the envs) freezes each
server into an ``install/recipes/tool_specs/<server>.yaml`` conforming to :class:`ToolSpec`;
``provision`` reads those specs back to rebuild.

One spec == one MCP **server** (one interpreter / PREFIX), which may expose
several ``@mcp.tool`` functions that share the env. The crucial captured bits are
``env_prefix`` + ``override_vars`` — the exact ``{PREFIX}_PYTHON`` / ``_RSCRIPT``
env-var name(s) and path(s) the worker dispatch reads — because they are NOT
derivable by rule (``card``'s ``CARD_PYTHON`` holds an *Rscript* path;
``seurat`` uses ``SEURAT_RSCRIPT``).

Stdlib + pyyaml only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import constants
from .decisions import BuildStrategy
from .session_log import notice

WORKER_KINDS = ("python", "rscript")


class SpecError(ValueError):
    """A tool spec is missing required fields or has an invalid value."""


@dataclass
class FunctionSpec:
    """One ``@mcp.tool`` function within a server."""

    name: str
    category: str = ""
    mini_test: dict = field(default_factory=dict)  # mirrors a test/smoke TESTS tuple

    def to_dict(self) -> dict:
        d = {"name": self.name}
        if self.category:
            d["category"] = self.category
        if self.mini_test:
            d["mini_test"] = self.mini_test
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FunctionSpec:
        if not d.get("name"):
            raise SpecError("function entry needs a `name`")
        return cls(name=d["name"], category=d.get("category", ""), mini_test=d.get("mini_test") or {})


@dataclass
class ToolSpec:
    """The reproducible recipe + wiring facts for one MCP server."""

    server_key: str
    worker_kind: str = "python"
    env_prefix: str = ""  # e.g. TANGRAM, CARD, SEURAT
    override_vars: dict[str, str] = field(default_factory=dict)  # {TANGRAM_PYTHON: /opt/.../bin/python}
    # The worker script is repo-resident (not env-resident); wiring repoints it to
    # the live clone, so we record its env-var NAME + basename separately from the
    # interpreter override. (cell2location hardcodes an absolute worker default that
    # is wrong on a fresh clone → the override is mandatory, not cosmetic.)
    worker_var: str = ""  # e.g. CELL2LOCATION_WORKER, SEURAT_WORKER
    worker_file: str = ""  # basename, e.g. cell2location_worker.py
    server_file: str = ""  # basename, e.g. cell2location_mcp_server.py
    source_env: str | None = None  # the env name captured from (clone/export source)
    import_check: str = ""  # a package name to import as a health probe (e.g. cell2location)
    build_strategy: BuildStrategy = BuildStrategy.CONDA_EXPORT
    recipe: str | None = None  # path to an env yaml (tools_user/* or captured env/*)
    gpu: bool = False
    shared_env: bool = False  # env shared by multiple servers (moscot/novosparc/spatialomicsgym_e1)
    est_gb: float | None = None
    service_keys: list[str] = field(default_factory=list)  # env vars a selected function needs
    functions: list[FunctionSpec] = field(default_factory=list)

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "server_key": self.server_key,
            "worker_kind": self.worker_kind,
            "env_prefix": self.env_prefix,
            "override_vars": dict(self.override_vars),
            "worker_var": self.worker_var,
            "worker_file": self.worker_file,
            "server_file": self.server_file,
            "source_env": self.source_env,
            "import_check": self.import_check,
            "build_strategy": self.build_strategy.value,
            "recipe": self.recipe,
            "gpu": self.gpu,
            "shared_env": self.shared_env,
            "est_gb": self.est_gb,
            "service_keys": list(self.service_keys),
            "functions": [f.to_dict() for f in self.functions],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolSpec:
        if not d.get("server_key"):
            raise SpecError("spec is missing `server_key`")
        try:
            strategy = BuildStrategy.coerce(d.get("build_strategy", "conda_export"))
        except ValueError as exc:
            raise SpecError(f"{d.get('server_key')}: {exc}") from exc
        kind = d.get("worker_kind", "python")
        if kind not in WORKER_KINDS:
            raise SpecError(f"{d['server_key']}: worker_kind must be one of {WORKER_KINDS}, got {kind!r}")
        # `service_keys` may be authored as a scalar (`service_keys: UCD_TOKEN`, one key) or a list. A
        # bare `list(...)` char-splits the scalar into ['U','C','D',...]; normalize str|list exactly as
        # the registry does for filename_pattern/prediction_key (a durable str|list convention).
        sk = d.get("service_keys") or []
        service_keys = [sk] if isinstance(sk, str) else list(sk)
        # Coerce the scalar path/env fields to str for the same reason as `server_key` below: a
        # hand-authored non-string (a YAML list `source_env:\n  - env`, or a numeric `worker_file: 2`)
        # otherwise flows uncoerced into the loop-less `resolve_full_config` — `src in PROTECTED_ENVS`
        # (unhashable list -> TypeError), `(source_env or "").strip()` and `constants.tools_dir(repo_root) /
        # worker_file` (list/int -> TypeError) — which has no per-server guard, so ONE bad spec aborts
        # the resolve of EVERY server. str() degrades a bad value to a non-matching name (safe
        # absent/enabled:false); an absent source_env must stay None (candidate_target_envs gates on
        # `if src:`), so don't stringify None into the literal "None".
        src_env = d.get("source_env")
        spec = cls(
            # Coerce to str (the declared `server_key: str` type): a YAML-authored numeric key
            # (`server_key: 2` → int) otherwise flows through load_all_specs into the resolver's
            # `sorted(set(original) | set(specs))`, where a mixed str/int set raises TypeError and
            # aborts the whole resolve. Mirrors the service_keys str-normalization just above; the 88
            # shipped specs are already strings, so this is a no-op on the healthy repo.
            server_key=str(d["server_key"]),
            worker_kind=kind,
            env_prefix=d.get("env_prefix", ""),
            override_vars=dict(d.get("override_vars") or {}),
            worker_var=d.get("worker_var", ""),
            worker_file=str(d.get("worker_file") or ""),
            server_file=str(d.get("server_file") or ""),
            source_env=None if src_env is None else str(src_env),
            import_check=d.get("import_check", ""),
            build_strategy=strategy,
            recipe=d.get("recipe"),
            gpu=bool(d.get("gpu", False)),
            shared_env=bool(d.get("shared_env", False)),
            est_gb=d.get("est_gb"),
            service_keys=service_keys,
            functions=[FunctionSpec.from_dict(f) for f in (d.get("functions") or [])],
        )
        spec.validate()
        return spec

    # -- validation -----------------------------------------------------------
    def validate(self) -> None:
        if self.build_strategy is BuildStrategy.NONE:
            return  # e.g. the `eval` server — nothing to build or wire
        if not self.override_vars:
            raise SpecError(
                f"{self.server_key}: override_vars is empty — the worker interpreter "
                "cannot be wired (need e.g. {PREFIX}_PYTHON -> /opt/conda/envs/<env>/bin/python)"
            )
        # every override path should look like an interpreter under an env prefix
        for var, path in self.override_vars.items():
            if not var.isupper():
                raise SpecError(f"{self.server_key}: override var {var!r} should be an UPPER_CASE env var name")
            if not str(path).strip():
                raise SpecError(f"{self.server_key}: override var {var} has an empty path")

    # -- convenience ----------------------------------------------------------
    def target_env(self, basic_env_name: str) -> str:
        """The env this spec builds into for a given base name: ``<basic>_<server>``."""
        return constants.tool_env_name(basic_env_name, self.server_key)

    def categories(self) -> list[str]:
        seen: list[str] = []
        for f in self.functions:
            if f.category and f.category not in seen:
                seen.append(f.category)
        return seen

    def rewired_override_vars(self, target_env: str, envs_root: str | None = None) -> dict[str, str]:
        """Return ``override_vars`` repointed from ``source_env`` to ``target_env``.

        The captured paths point at the *source* env; when we rebuild into
        ``<basic>_<server>`` the interpreter lives under the new prefix. Rewrite
        the env-dir segment while preserving the ``bin/python`` vs ``bin/Rscript``
        tail (that distinction is why we captured, not derived, these).

        N13: ``envs_root`` defaults to the LIVE conda envs dir on this machine
        (``constants.conda_envs_root()``) — never a hardcoded ``/opt/conda/envs``, which
        is wrong on a micromamba / ``~/miniconda3`` / relocated-Miniforge deploy box.
        """
        if envs_root is None:
            envs_root = constants.conda_envs_root()
        envs_root = envs_root.rstrip("/")
        out: dict[str, str] = {}
        for var, path in self.override_vars.items():
            p = str(path)
            # …/envs/<source>/bin/xxx  ->  …/envs/<target>/<per-OS interpreter> (a1c#2): on Windows the
            # rebased interpreter is <env>\python.exe / <env>\Scripts\Rscript.exe, not …/bin/<exe>.
            tail = p.rsplit("/bin/", 1)
            if len(tail) == 2:
                out[var] = constants.interp_path(f"{envs_root}/{target_env}", tail[1])
            else:
                out[var] = p
        return out


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_spec(path: str | Path) -> ToolSpec:
    p = Path(path)
    if not p.exists():
        raise SpecError(f"spec file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        # load_spec is documented to raise SpecError; catching only YAMLError let an OSError
        # (a race where the file becomes unreadable after the exists() check) or a
        # UnicodeDecodeError (a non-UTF-8 spec — a ValueError, NOT an OSError) escape as a
        # non-SpecError. load_all_specs's broad `except Exception` currently masks it, but the
        # contract leak is real; wrap it here so the friendly SpecError is what propagates.
        raise SpecError(f"could not read/parse {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"{p}: spec must be a mapping")
    return ToolSpec.from_dict(data)


def load_all_specs(specs_dir: str | Path | None = None) -> dict[str, ToolSpec]:
    """Load every ``<server>.yaml`` under the specs dir into ``{server_key: ToolSpec}``."""
    d = Path(specs_dir) if specs_dir else constants.spec_dir()
    out: dict[str, ToolSpec] = {}
    if not d.exists():
        return out
    for p in sorted(d.glob("*.yaml")):
        if p.name in ("index.yaml",):
            continue
        try:
            spec = load_spec(p)
        except Exception as exc:  # one bad file must not abort loading every other tool
            # A malformed/hand-corrupted spec on a deployed box is skipped with a friendly
            # notice, not a launch-blocking traceback — the rest of the catalog still loads.
            notice(f"skipping unreadable tool spec {p.name}: {exc}")
            continue
        out[spec.server_key] = spec
    return out


def write_spec(spec: ToolSpec, specs_dir: str | Path | None = None) -> Path:
    d = Path(specs_dir) if specs_dir else constants.spec_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{spec.server_key}.yaml"
    p.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    return p
