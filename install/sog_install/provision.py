"""
Provision — build one ``<basic>_<server>`` conda env per selected tool.

Every env this module creates, repairs, or removes is in the ``<basic>_*``
namespace and passes :func:`constants.assert_deletable_env` before any removal,
so a provisioning run can never touch a pre-existing env you already use.

Per server the flow is: **classify** health (read-only) → **propose** → (gated
``confirm``) → **build** by strategy, with a **clone fallback** and an optional
**self-review** auto-repair on build failure.

Classification:

* ``CREATE``   — target env absent (the fresh-clone common case; no probe needed).
* ``SKIP``     — target env present *and* its import/health probe passes.
* ``RECREATE`` — target env present but unhealthy → remove (guarded) + rebuild.
* ``NONE``     — nothing to build (e.g. the ``eval`` server).

Build strategies (``specs.BuildStrategy``): ``env_yaml`` / ``conda_export``
materialize the recipe (name/prefix stripped, ``-n <target>``); ``conda_clone``
clones the present source env (fast + faithful on the same host); ``pip`` makes a
minimal env and pip-installs the package; ``github_creation`` defers to the
tool-creation flow (a hook — not attempted inline).

Stdlib + pyyaml only.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import yaml

from . import constants, progress
from .decisions import BuildStrategy, OnToolFail, Proposal, ProvisionDecision
from .envtools import Conda, CondaError, RunResult
from .session_log import redact

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .decisions import ToolPlan
    from .prompts import PromptIO
    from .session_log import SessionLog
    from .specs import ToolSpec

_TRUTHY = {"1", "true", "yes", "y", "on"}  # opt-in flag values (mirrors remediation_planner / llm_chat)


class ToolStatus(StrEnum):
    CREATE = "create"  # first-create (env absent)
    SKIP = "skip"  # healthy, nothing to do
    RECREATE = "recreate"  # present but unhealthy → rebuild
    REUSE = "reuse"  # managed env absent, but an existing healthy env (usually the clone source) satisfies it → wire read-only, no build
    NONE = "none"  # nothing to build (eval)


@dataclass
class ToolResult:
    server_key: str
    target_env: str
    status: str = ""  # ToolStatus that was acted on
    strategy: str = ""  # BuildStrategy actually used
    ok: bool = False
    skipped: bool = False  # confirm declined
    built: bool = False
    fell_back: bool = False  # spec strategy failed → clone fallback used
    self_review: dict | None = None
    repairs: list[dict] = field(default_factory=list)  # envdoctor/LLM remediation attempts (A2/A4)
    messages: list[str] = field(default_factory=list)
    # A specific, actionable Lane-3 message when a build failure is real but NOT an env problem
    # (external token / special input / data shape / …). Persisted so the Part-D demo summary can
    # point the user at exactly what to fix. Empty when the build worked or was cleanly repaired.
    needs_attention: str = ""
    # The pre-existing env a REUSE tool was wired to (read-only), set when the managed env was
    # absent but an existing healthy env — usually the clone source the run would otherwise
    # duplicate — already satisfied the tool. Empty for every build/skip path.
    reused_env: str = ""


# --------------------------------------------------------------------------- #
# Health classification (read-only)
# --------------------------------------------------------------------------- #
def _health_run(conda: Conda, spec: ToolSpec, target_env: str) -> RunResult:
    """Run the cheap import/library probe inside ``target_env`` and return its full ``RunResult``,
    so callers that need the *failure text* (the remediation loop's re-observation, which surfaces a
    cascading missing dependency) get the probe stderr, not just a pass/fail bool."""
    pkg = spec.import_check
    if not pkg:
        # No probe declared → we cannot verify health, so treat env existence as healthy (a1c#4). We
        # deliberately do NOT fall back to ``spec.server_key`` as an import name (unlike the doctor's
        # remediation probes): server_key is frequently NOT the importable package name (e.g.
        # ``scanpy_spatial`` / ``spatialde``), so probing it would FALSE-NEGATIVE and trigger needless
        # rebuilds. Every shipped buildable spec declares an import_check (locked by a test), so this
        # branch is defensive only — a future spec that omits it trips that test rather than silently
        # shipping an unverified "healthy".
        return RunResult(returncode=0)
    if spec.worker_kind == "rscript":
        return conda.run(
            target_env, ["Rscript", "-e", f"library({pkg})"], timeout=constants.CONDA_RUN_TIMEOUT_SEC, check=False
        )
    return conda.run(
        target_env, ["python", "-c", f"import {pkg}"], timeout=constants.CONDA_RUN_TIMEOUT_SEC, check=False
    )


def _healthy(conda: Conda, spec: ToolSpec, target_env: str) -> bool:
    """Cheap import/library probe inside the target env."""
    return _health_run(conda, spec, target_env).ok


def classify(conda: Conda, spec: ToolSpec, target_env: str) -> ToolStatus:
    if spec.build_strategy is BuildStrategy.NONE:
        return ToolStatus.NONE
    if not conda.env_exists(target_env):
        return ToolStatus.CREATE
    return ToolStatus.SKIP if _healthy(conda, spec, target_env) else ToolStatus.RECREATE


def _reusable_existing_env(conda: Conda, spec: ToolSpec, basic_env: str) -> str | None:
    """An already-present env (other than the managed ``<basic>_<server>``) that is *already
    healthy* for ``spec`` — the "reuse the env you already have" scan. Returns the first such env
    name, or ``None`` when the tool genuinely has to be built.

    Read-only and cheap: it probes only envs that exist (absent candidates short-circuit on the
    cached env map, no subprocess) and never creates or mutates anything. The candidate list — the
    managed name excluded — comes from :func:`mcp_resolver.candidate_target_envs`, so the reuse scan
    and the live interpreter resolver agree on exactly which envs may satisfy a tool. The dominant
    hit is the tool's own *source* env (``card_env`` / ``tangram-env`` / ``SpatialDE`` / …) that a
    plain ``conda_clone`` build would otherwise duplicate: with the source already installed and
    importable, cloning a byte-identical copy is pure waste. Honors ``SOG_SETUP_NO_ENV_REUSE`` via
    ``candidate_target_envs`` (the flag drops the raw-source candidate, so the scan finds nothing
    to reuse and the tool builds as before)."""
    from . import mcp_resolver  # lazy: mcp_resolver imports provision (._healthy) — avoid an import cycle

    managed = spec.target_env(basic_env)
    for cand in mcp_resolver.candidate_target_envs(spec, basic_env):
        if cand == managed:
            continue  # the managed env is what CREATE would build — never a "reuse" of an existing one
        try:
            if conda.env_exists(cand) and _healthy(conda, spec, cand):
                return cand
        except Exception:
            # A slow/loaded box can make a health probe raise CondaError. That must never abort
            # provisioning — treat this candidate as unusable and keep scanning (mirrors the
            # resolver's per-candidate guard). Worst case we build the managed env, as before.
            continue
    return None


def effective_tool_env(conda: Conda, spec: ToolSpec, basic_env: str) -> str:
    """The env a tool's config is *actually* wired to: the managed ``<basic>_<server>`` env when it
    was built, otherwise a healthy already-present env that env-reuse (R1) wired read-only in its
    place, else the managed name (so a genuinely-missing tool still reports as its own absent env).

    This mirrors :func:`mcp_resolver.resolve_interpreter`'s candidate walk (managed-first, reuse
    fallback) at the env-*name* level, so post-provision consumers — ``doctor``'s health report and
    Tier-1's worker run — probe the SAME env the emitted config points at. Without it a perfectly
    healthy reuse install reads as ``[✗] absent`` / ``sog-setup doctor`` exit 1, and Tier-1 SKIPs the
    reused tool with a misleading "interpreter missing". Probes only envs that already exist.

    It walks the resolver's own candidates with the resolver's own rule — the first env that exists
    AND is healthy (hunt 2026-09-30, u36-setup-install-9). It used to return the managed env on mere
    existence, so a half-built ``<basic>_<server>`` beside a healthy source env had doctor and Tier-1
    test (and RECREATE) the broken managed env while the config ran the healthy one. When no
    candidate is healthy the managed name is returned, as before, so the broken env is the one
    reported and repaired."""
    from . import mcp_resolver  # lazy: mcp_resolver imports provision (._healthy) — avoid an import cycle

    managed = spec.target_env(basic_env)
    chosen, healthy, _reason = mcp_resolver.resolve_interpreter(conda, spec, basic_env)
    return chosen if (healthy and chosen) else managed


# --------------------------------------------------------------------------- #
# Proposals
# --------------------------------------------------------------------------- #
def propose(spec: ToolSpec, target_env: str, status: ToolStatus, *, conda: Conda | None = None) -> Proposal:
    verb = {ToolStatus.CREATE: "create env", ToolStatus.RECREATE: "rebuild env"}[status]
    src = f" (clone of {spec.source_env})" if spec.build_strategy is BuildStrategy.CONDA_CLONE else ""
    detail = f"{target_env} via {spec.build_strategy.value}{src}"
    # Say which build will actually run (hunt 2026-09-30, u36-setup-install-18): a clone needs its source
    # env HERE, and on a fresh machine it is absent, so build() falls through to solving and downloading
    # the full recipe — the user was asked to consent to "clone of X", a local copy that never happens.
    if conda is not None and spec.build_strategy is BuildStrategy.CONDA_CLONE:
        try:
            source_here = bool(spec.source_env) and conda.env_exists(spec.source_env)
        except Exception:
            source_here = True  # unprobeable → keep the plain description rather than guess
        if not source_here:
            if spec.recipe:
                detail = (
                    f"{target_env} via {BuildStrategy.CONDA_EXPORT.value} from {spec.recipe} — the clone source "
                    f"{spec.source_env or '(none)'} is not on this machine, so this solves and downloads the "
                    "full recipe (network, can take a long time)"
                )
            else:
                detail = (
                    f"{target_env} via {spec.build_strategy.value} — but the clone source "
                    f"{spec.source_env or '(none)'} is not on this machine and there is no recipe to build from"
                )
    return Proposal(
        action=verb,
        detail=detail,
        est_gb=spec.est_gb,
        reversible=True,  # a <basic>_* env can be removed cleanly
        kind="create_env",
    )


# --------------------------------------------------------------------------- #
# Recipe materialization
# --------------------------------------------------------------------------- #
def materialize_recipe(
    recipe_rel: str,
    target_env: str,
    *,
    strip_builds: bool = False,
    no_gpu: bool | None = None,
    foreign_platform: bool | None = None,
) -> Path:
    """Load a recipe yaml, strip ``name:``/``prefix:``, write a temp for ``-n <target>``.

    ``strip_builds`` drops exact conda build strings — used by the loosen-on-failure retry so a
    recipe whose pinned build rotted out of the channel can still solve. ``no_gpu`` (auto-detected
    from the host when ``None``) relaxes CUDA-only pip pins so a recipe captured on a GPU-Linux box
    still installs on a CPU/mac target. ``foreign_platform`` (auto-detected as *non-Linux* when
    ``None``) strips the linux-64-only GNU toolchain / runtime / system / X11 conda pins that would
    otherwise hard-fail an osx/win solve. The default call (build strings kept, GPU- and OS-gate auto)
    keeps the strict recipe verbatim on a GPU Linux host and only relaxes provably-unusable pins on a
    CPU or non-Linux target.
    """
    src = constants.resolve_recipe_path(recipe_rel)  # maps a recorded pre-re-layout 'setup/...' string
    try:
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError) as exc:
        # A missing/malformed recipe must fail *this* tool, not crash the whole run.
        # CondaError is caught by build()'s fallback loop and honors on_tool_fail. A non-UTF-8
        # recipe raises UnicodeDecodeError (a ValueError, NOT an OSError) — without it here the
        # raw ValueError escapes the CondaError-only fallback loop and sinks the entire run.
        raise CondaError(f"recipe {recipe_rel!r} unreadable: {exc}") from exc
    if not isinstance(data, dict):
        raise CondaError(f"recipe {recipe_rel!r} is not a mapping (got {type(data).__name__})")
    if no_gpu is None:
        no_gpu = not _host_has_gpu()
    if foreign_platform is None:
        foreign_platform = not _host_is_linux()
    data = portable_recipe_data(data, strip_builds=strip_builds, no_gpu=no_gpu, foreign_platform=foreign_platform)
    data.pop("name", None)
    data.pop("prefix", None)
    tmp_dir = constants.state_dir() / "tmp"
    out = tmp_dir / f"{target_env}.recipe.yaml"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except OSError as exc:
        # Mirror the read-side wrap above: a materialize *write* failure (ENOSPC mid-build, a
        # read-only/over-quota state_dir) must fail THIS tool via CondaError so build()'s fallback
        # loop owns it (loosen-retry / next strategy / remediation) — not escape as a raw OSError
        # that slips past build()'s `except CondaError` and only the run-level catch-all stops. (R20)
        raise CondaError(f"recipe {recipe_rel!r} could not be materialized: {exc}") from exc
    return out


def read_recipe_pip_and_python(recipe_rel: str) -> tuple[list[str], str]:
    """Return ``(pip_requirements, python_version)`` from a recipe env yaml.

    The pip requirements are the raw strings under ``dependencies: - pip: [...]`` (each a
    ``pkg==ver`` or an embedded pip flag line); the python version is parsed from the conda
    ``python=3.9.23=...`` pin (major.minor only). Used by the off-index-wheel repair
    (:mod:`envdoctor`) to reproduce a PyTorch-Geometric-style pip stack that ``conda env
    create`` cannot rebuild verbatim. Returns ``([], "")`` for a recipe with no pip section.
    Never raises — a missing/unreadable recipe yields empties so the caller degrades cleanly.
    """
    src = constants.resolve_recipe_path(recipe_rel)  # maps a recorded pre-re-layout 'setup/...' string
    try:
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, UnicodeDecodeError):
        # "Never raises" (see docstring): a non-UTF-8 recipe raises UnicodeDecodeError (a
        # ValueError, NOT an OSError), which would otherwise escape and crash the off-index
        # repair that relies on this degrading cleanly to empties.
        return [], ""
    if not isinstance(data, dict):
        return [], ""
    pip_reqs: list[str] = []
    python_ver = ""
    for dep in data.get("dependencies", []) or []:
        if isinstance(dep, dict) and "pip" in dep:
            # `pip:` is a LIST of req strings; a scalar `- pip: torch==2.1.0` (malformed/hand-edited
            # recipe, valid YAML) must NOT be char-split into ['t','o','r','c','h',...] by list(str).
            # Guard the same way the sibling portable_recipe_data does (isinstance(pip_list, list)). (R20)
            raw = dep.get("pip")
            pip_reqs = [str(x) for x in raw] if isinstance(raw, list) else ([str(raw)] if raw else [])
        elif isinstance(dep, str):
            # conda pin like "python=3.9.23=hc30ae73_0_cpython" → keep major.minor
            token = dep.split("=", 1)
            if token[0].strip() == "python" and len(token) == 2:
                ver = token[1].lstrip("=").split("=", 1)[0]
                parts = ver.split(".")
                python_ver = ".".join(parts[:2]) if len(parts) >= 2 else ver
    return pip_reqs, python_ver


# --------------------------------------------------------------------------- #
# Recipe portability (H1) — build-string stripping + GPU-gate
# --------------------------------------------------------------------------- #
# A captured recipe pins every conda package to an exact build string
# (``bzip2=1.0.8=hda65f42_8``). conda-forge garbage-collects old builds even on
# linux-64, and no build string is valid on osx-64/win-64 — so a verbatim recipe
# stops solving over time and never solves cross-platform. Dropping the third
# ``=<build>`` field back to ``name=version`` lets conda re-solve a
# platform-appropriate build while preserving the intended version.

# A conda *solve* failure that dropping build strings can fix: the exact pinned
# build/version is not in the channels for this platform. Deliberately does NOT
# match the pip "No matching distribution" / "Could not find a version" surface —
# those are off-index-wheel problems in the pip section that build-string
# stripping cannot fix (they route through envdoctor's OFF_INDEX_WHEEL repair).
_SOLVE_FAILURE_RE = re.compile(
    r"PackagesNotFoundError|ResolvePackageNotFound|nothing provides|"
    r"Encountered problems while solving|Could not solve for environment|"
    r"The following packages are not available from current channels|"
    r"packages are missing from the (?:target )?channels",
    re.I,
)

# nvidia CUDA runtime wheels torch drags in on a CUDA-Linux capture. They ship
# only as linux CUDA wheels, so an explicit pin hard-fails on osx/win and is dead
# weight on CPU. They are transitive (torch re-pulls the right ones per platform),
# so dropping the pin on a no-GPU host is safe and restores portability. Two
# packagings coexist in the wild and BOTH must be caught: the CUDA-12 ``-cuNN``
# *suffix* form (``nvidia-cublas-cu12==12.8.4.1``) and the CUDA-13 repackaging that
# *dropped* the suffix (``nvidia-cublas==13.1.0.3``, ``nvidia-cuda-cupti==13.0.85``) —
# so we match on the CUDA-component *stem* (``cu*`` / ``nccl`` / ``nvtx`` / ``nvjitlink`` /
# ``nvshmem``), which anchors both. ``nvidia-ml-py`` (pynvml) is deliberately EXCLUDED:
# it is pure-python, installs cross-platform, and is a runtime GPU-query binding — not a
# torch-dragged CUDA binary wheel — so a no-GPU host keeps it rather than ImportError-ing.
_CUDA_PIP_RE = re.compile(r"^\s*nvidia-(?:cu[a-z0-9]+|nccl|nvtx|nvjitlink|nvshmem)\b", re.I)
# A PEP 440 local-version CUDA pin → strip the local version so the rewritten CPU index /
# find-links below supplies the CPU wheel: ``torch==2.1.0+cu118`` → ``torch==2.1.0``. The
# ``[\w.]*`` before ``cu\d+`` also catches PyG's *compound* local version —
# ``torch-scatter==2.1.2+pt21cu118`` (torch-2.1 + CUDA-11.8) → ``torch-scatter==2.1.2`` —
# which a bare ``\+cu`` misses because the ``+`` is followed by ``pt21`` first. An already-CPU
# ``+cpu`` / ``+pt22cpu`` carries no ``cuNN`` and is left untouched.
_CUDA_LOCAL_VER_RE = re.compile(r"^(\s*[A-Za-z0-9._-]+==[0-9][\w.]*)\+[\w.]*cu\d+\b", re.I)
# A CUDA wheel *index* in a raw pip flag line — PyTorch's ``.../whl/cu118`` and PyG's
# ``.../torch-2.2.0+cu118.html``. On a no-GPU host these resolve to CUDA-only wheels that won't
# install; rewriting the ``cuNN`` token to ``cpu`` points pip at the CPU channel the same servers
# publish. Package pins are handled above; these two catch the flag *lines* (``--extra-index-url`` /
# ``--find-links``) that carry no ``==`` for _CUDA_LOCAL_VER_RE to bite.
_CUDA_INDEX_RE = re.compile(r"(whl/)cu\d+\b", re.I)
_CUDA_FINDLINKS_RE = re.compile(r"\+cu\d+(\.html)", re.I)
# Conda-*side* CUDA-only runtime libs a GPU-box capture bakes into the recipe (today only graphst:
# ``cuda-version``/``cuda-nvrtc``/``cudnn``/``libcublas``/``libcudnn``/``libcudnn-dev``). When the tool's
# torch/jax is a pip CPU wheel these go unused on CPU, and none ships an osx/win build — so a stray pin
# breaks a Mac deploy solve outright and needlessly pulls GBs of CUDA runtime onto a CPU box. They are
# leaf runtime libs (nothing else in a recipe conda-depends on them), so dropping them on a no-GPU host
# is solve-safe. This is an *explicit allowlist* of CUDA package stems — deliberately NOT ``libcu.*``,
# which would also eat ``libcurl``/``libcups``; the trailing ``(?:$|[=<> ])`` anchors each stem to a full
# name so ``libcurand`` never matches ``libcurl``. The pip-side ``nvidia-*-cuNN`` pins are _CUDA_PIP_RE.
_CUDA_CONDA_RE = re.compile(
    r"^(?:cuda(?:-[a-z0-9._-]+)?|cudatoolkit|cudnn|libcudnn(?:-[a-z0-9._-]+)?"
    r"|libcublas(?:lt)?(?:-[a-z0-9._-]+)?|libcufft|libcurand|libcusolver|libcusparse(?:lt)?"
    r"|libcufile|libnvjitlink|libnvfatbin|libnvjpeg|libnvrtc|nccl|libnccl(?:-[a-z0-9._-]+)?"
    r"|pytorch-cuda)(?:$|[=<> ])",
    re.I,
)
# Conda packages that only exist for linux-64 on the channels these recipes use, so a Linux
# capture's verbatim pin (``libgomp=15.2.0``, ``ld_impl_linux-64=2.45.1``, ``xorg-libx11=…``)
# hard-fails the *solve* on osx-arm64/osx-64/win. Every stem here is one of:
#   * the GNU cross-compiler toolchain (``*_linux-64`` + the bare ``gcc``/``gxx``/``gfortran``
#     activation metapackages) — build-time only; macOS builds against clang (Xcode CLT) and
#     conda re-adds an osx compiler transitively if a package genuinely build-requires one;
#   * the GNU C/C++/Fortran/OpenMP *runtimes* (``libgcc*``/``libstdcxx*``/``libgomp``/
#     ``libgfortran*``/``_openmp_mutex``/``_libgcc_mutex``) — macOS uses ``libcxx``/``llvm-openmp``
#     and its own ``libgfortran5``, all pulled in transitively by numpy/scipy/… on the target;
#   * Linux-only *system* libraries (``libuuid``/``libnsl``/``libxcrypt``/``keyutils``/``libdrm``/
#     ``libpciaccess``/``libudev1``/``libsystemd0``/``libcap``) — macOS provides the equivalents in
#     libSystem, so nothing on the target depends on the conda build;
#   * the Linux X11 / GL display stack (``xorg-lib*``/``libglvnd``) — macOS plots through
#     quartz/cocoa/agg backends and never loads these.
# Dropping the *explicit* pin (not the capability) lets conda re-solve a platform-appropriate build,
# exactly like the CUDA-conda strip above. Anchored to full names via ``(?:$|[=<> ])`` so ``gcc`` never
# eats ``gccxml``, ``attr`` is untouched (kept out to protect the cross-platform ``attrs``), ``libgcc``
# never eats an unrelated ``libgcc``-prefixed osx pkg, and ``libcap`` never bites ``libcaca``. This only
# fires on a non-Linux host (:func:`_host_is_linux`); a Linux build is left byte-for-byte unchanged.
_LINUX_ONLY_CONDA_RE = re.compile(
    r"^(?:"
    r"[a-z0-9_.-]*_linux-64"  # arch-tagged GNU cross toolchain (ld_impl/gcc_impl/sysroot/…_linux-64)
    r"|g(?:cc|xx|fortran)"  # bare conda-forge compiler activation metapackages
    r"|libgcc(?:-ng)?|libstdcxx(?:-ng)?|libgomp"  # GNU C/C++/OpenMP runtimes
    r"|libgfortran(?:-ng|[45])?"  # GNU Fortran runtime
    r"|_openmp_mutex|_libgcc_mutex"  # GNU/OpenMP mutex shims
    r"|libuuid|libnsl|libxcrypt|keyutils"  # Linux system libs (macOS: libSystem)
    r"|libpciaccess|libdrm|libudev1|libsystemd0|libcap"
    r"|xorg-lib[a-z0-9]+|libglvnd"  # Linux X11 / GL display stack
    r")(?:$|[=<> ])",
    re.I,
)


def _strip_build_string(dep: str) -> str:
    """``pkg=ver=build`` → ``pkg=ver``; leave ``pkg=ver`` / ``pkg`` / pip ``pkg==ver`` unchanged."""
    if "==" in dep:  # a pip-style pin that slipped into the conda list — never touch
        return dep
    parts = dep.split("=")
    # conda match-spec grammar is name[=version[=build]]; three non-empty leading
    # fields means a build string is present → keep name=version, drop the build.
    if len(parts) >= 3 and parts[0].strip() and parts[1].strip():
        return f"{parts[0]}={parts[1]}"
    return dep


def _degpu_pip_req(req: str) -> str | None:
    """A pip requirement/flag rewritten for a no-GPU host, or ``None`` to drop it entirely."""
    if _CUDA_PIP_RE.match(req):
        return None  # drop explicit nvidia-*-cuNN pins (torch re-adds the right ones)
    if req.lstrip().startswith("-"):
        # a raw pip flag line (``--extra-index-url``/``--find-links``): repoint a CUDA wheel
        # index at the CPU channel the same host publishes, rather than dropping the line and
        # losing the (still-needed) CPU wheel source.
        rewritten = _CUDA_INDEX_RE.sub(r"\1cpu", req)
        return _CUDA_FINDLINKS_RE.sub(r"+cpu\1", rewritten)
    m = _CUDA_LOCAL_VER_RE.match(req)
    if m:
        return m.group(1).strip()  # torch==2.1.0+cu118 → torch==2.1.0
    return req


def portable_recipe_data(data: dict, *, strip_builds: bool, no_gpu: bool, foreign_platform: bool = False) -> dict:
    """A copy of a recipe mapping made portable.

    ``strip_builds`` drops exact conda build strings; ``no_gpu`` drops/relaxes CUDA-only pins on both
    sides of the recipe — pip (``nvidia-*-cuNN``, ``torch==X+cuNN``, CUDA wheel indexes) *and* conda
    (``cuda-version``/``libcublas``/``cudnn`` runtime libs, via :data:`_CUDA_CONDA_RE`). ``foreign_platform``
    (a non-Linux host — macOS/Windows) additionally drops the Linux-only GNU toolchain / runtime / system /
    X11 conda pins a linux-64 capture bakes in (via :data:`_LINUX_ONLY_CONDA_RE`), which would otherwise
    hard-fail the osx/win *solve*; conda re-adds the platform-appropriate equivalents transitively.
    Pure and defensive — an unexpected shape is passed through untouched so a malformed recipe still
    reaches conda (which reports the real error) rather than crashing here.
    """
    if not isinstance(data, dict):
        return data
    deps = data.get("dependencies")
    if not isinstance(deps, list):
        return data
    new_deps: list = []
    for dep in deps:
        if isinstance(dep, str):
            if no_gpu and _CUDA_CONDA_RE.match(dep):
                continue  # drop a conda-side CUDA-only runtime lib on a no-GPU host (Mac/CPU portability)
            if foreign_platform and _LINUX_ONLY_CONDA_RE.match(dep):
                continue  # drop a linux-64-only conda pin on a non-Linux host (macOS/Windows portability)
            new_deps.append(_strip_build_string(dep) if strip_builds else dep)
        elif isinstance(dep, dict) and "pip" in dep:
            pip_list = dep.get("pip")
            if no_gpu and isinstance(pip_list, list):
                kept = [r for r in (_degpu_pip_req(str(x)) for x in pip_list) if r is not None]
                new_deps.append({**dep, "pip": kept})
            else:
                new_deps.append(dep)
        else:
            new_deps.append(dep)
    return {**data, "dependencies": new_deps}


def _host_has_gpu() -> bool:
    """Cheap NVIDIA probe — the same ``nvidia-smi`` check preflight/testing use."""
    return shutil.which("nvidia-smi") is not None


def _host_is_linux() -> bool:
    """True on a Linux host. Drives the recipe cross-platform strip: a non-Linux host (macOS,
    Windows) is 'foreign', so the linux-64-only conda pins are dropped from the materialized recipe."""
    return sys.platform.startswith("linux")


# --------------------------------------------------------------------------- #
# Build dispatch
# --------------------------------------------------------------------------- #
def _build_by_strategy(
    conda: Conda, spec: ToolSpec, target_env: str, strategy: BuildStrategy, *, loosen: bool = False
) -> RunResult:
    # Live build-log streaming (opt-in, off by default): tee this build's output to a tail-able
    # <target>.build.log — the SAME path _persist_build_log writes post-hoc, so a failed build's
    # authoritative joined stderr is appended there afterward (C7: the prior live lines are kept,
    # not clobbered — the newest block is always at the tail). Passed ONLY when streaming is on,
    # so every non-streaming caller (and every FakeConda in the suite) sees the exact prior call
    # signature and never receives an unexpected kwarg.
    stream_log = (
        constants.logs_dir() / f"{target_env}.build.log" if getattr(conda, "stream_build_logs", False) else None
    )
    kw = {"stream_log_path": stream_log} if stream_log is not None else {}
    if strategy in (BuildStrategy.ENV_YAML, BuildStrategy.CONDA_EXPORT):
        if not spec.recipe:
            raise CondaError(f"{spec.server_key}: {strategy.value} has no recipe")
        # loosen=True (the H1b retry) drops exact build strings so a recipe whose pinned build
        # rotted out of the channel — or was captured on another platform — can re-solve. The
        # strict first attempt keeps the two-positional-arg call so any monkeypatch still matches.
        recipe = (
            materialize_recipe(spec.recipe, target_env, strip_builds=True)
            if loosen
            else materialize_recipe(spec.recipe, target_env)
        )
        # check=False: a recipe rebuild that dies on off-index pins must return its FULL stderr
        # (not a truncated CondaError) so build() surfaces it verbatim and the A2/A3 classifier
        # can see the "No matching distribution found for …+cpu/+ptNN" signature. Without this the
        # only text reaching classify_failure was "`conda env create …` exited 1" — unclassifiable,
        # so the off-index remediation never fired.
        return conda.create_from_yaml(str(recipe), name=target_env, check=False, **kw)
    if strategy is BuildStrategy.CONDA_CLONE:
        if not spec.source_env or not conda.env_exists(spec.source_env):
            raise CondaError(f"{spec.server_key}: clone source {spec.source_env!r} not present")
        return conda.clone(spec.source_env, target_env, **kw)
    if strategy is BuildStrategy.PIP:
        conda.create_named(target_env, python="3.11")
        pkg = spec.import_check or spec.server_key
        # check=False mirrors the recipe branch above: a pip build that dies on an off-index wheel
        # (`No matching distribution found for …+cpu/+ptNN`) must return its FULL stderr so build()
        # surfaces the signature the A2/A3 off-index classifier keys on. With the default check=True,
        # `_exec` raises a CondaError carrying only the last 8 lines — the signature is truncated away
        # and the off-index remediation never fires for a PIP-strategy tool.
        return conda.pip_install(target_env, [pkg], check=False, **kw)
    if strategy is BuildStrategy.GITHUB_CREATION:
        raise CondaError(f"{spec.server_key}: github_creation is not attempted inline (needs the tool-creation flow)")
    raise CondaError(f"{spec.server_key}: unknown strategy {strategy}")


def _fallback_chain(spec: ToolSpec) -> list[BuildStrategy]:
    """The ordered build candidates: the captured strategy first, then whatever
    else the spec makes possible.

    The two fallbacks cover the two ways a build can be reproduced without the
    original machine: a **materialized recipe** (``conda_export``) needs no source
    env — the fresh-*machine* case — while a **clone** is fastest when the source
    env is present, as it is for a fresh *clone* on this same host. Both are added
    only when the spec actually supports them, and ``_build_by_strategy`` guards
    each again at call time.
    """
    chain = [spec.build_strategy]
    recipe_strats = (BuildStrategy.ENV_YAML, BuildStrategy.CONDA_EXPORT)
    if spec.recipe and not any(s in chain for s in recipe_strats):
        chain.append(BuildStrategy.CONDA_EXPORT)
    if spec.source_env and BuildStrategy.CONDA_CLONE not in chain:
        chain.append(BuildStrategy.CONDA_CLONE)
    return chain


def build(conda: Conda, spec: ToolSpec, target_env: str, basic_env: str = "") -> tuple[RunResult, BuildStrategy, bool]:
    """Build ``target_env`` by walking the fallback chain; first success wins.

    Returns ``(result, strategy_used, fell_back)`` where ``fell_back`` is True iff
    the strategy that succeeded was not the captured primary.

    ``basic_env`` (this run's base env) enables the inter-attempt cleanup: a strategy
    that fails partway can leave a PARTIAL ``target_env`` prefix on disk (an interrupted
    clone/solve, or a ``PIP`` create that succeeded before its pip step failed). The next
    create/clone would then collide with that half-written prefix ("prefix already exists",
    or a clone into a dirty dir) and a genuinely-recoverable fallback fails spuriously. So
    before each *retry* attempt we guard-remove the managed target, mirroring
    :func:`provision_one`'s RECREATE pre-clean. Omitted (empty) ⇒ no cleanup, exactly the
    prior behavior — so a bare ``build(conda, spec, target)`` call is unchanged.
    """
    chain = _fallback_chain(spec)
    primary = chain[0]
    errors: list[str] = []
    loosen_tried = False

    def _clear_partial_target() -> None:
        """Guard-remove a partial ``target_env`` left by a prior failed attempt so the next
        create/clone starts clean. No-op unless we own the namespace and the env is present; a
        non-managed/protected name is left untouched (the collision then surfaces honestly rather
        than risking a wrong delete)."""
        if not (basic_env and conda.env_exists(target_env)):
            return
        try:
            constants.assert_deletable_env(basic_env, target_env)  # raises unless <basic>_* & unprotected
        except PermissionError:
            return
        # Best-effort: a locked/slow partial env makes remove_env raise CondaError (envtools _exec maps
        # TimeoutExpired/OSError to CondaError even at check=False) or OSError. This pre-clean runs
        # OUTSIDE build()'s per-strategy try (it is called at i>0, before it), so an escape would turn a
        # recoverable fallback-cleanup hiccup into a whole-tool crash (provision_all's per-tool guard),
        # skipping the remaining strategies' self-heal + full-log. Swallow it and let the next
        # create/clone collide honestly (mirrors wizard._remove_orphan_tool_envs; build() exposes no
        # io/log to note through, so this is a silent best-effort).
        with contextlib.suppress(CondaError, OSError):
            conda.remove_env(target_env)

    # With live streaming on, start this build's tail-able log fresh so it shows only THIS build's
    # attempts (not a previous rebuild's). No-op when streaming is off — the file is written post-hoc
    # by _persist_build_log exactly as before.
    if getattr(conda, "stream_build_logs", False):
        with contextlib.suppress(OSError):
            logs = constants.logs_dir()
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"{target_env}.build.log").write_text("", encoding="utf-8")
    for i, strat in enumerate(chain):
        # Falling to a later strategy: clear any partial env the prior strategy left so this
        # create/clone doesn't collide with a half-written prefix (the first attempt is pre-cleaned
        # by provision_one's own RECREATE path / starts from no env, so only i>0 needs it).
        if i > 0:
            _clear_partial_target()
        try:
            res = _build_by_strategy(conda, spec, target_env, strat)
        except CondaError as exc:
            # A raising strategy (a clone with an absent source, or any check=True create) still
            # carries its captured stderr tail on the exception — join it so the multi-strategy
            # error text keeps every classifiable signature, not just the bare "exited N" message.
            detail = getattr(exc, "stderr", "") or ""
            errors.append(f"[{strat.value}] {exc}\n{detail}".rstrip())
            continue
        if res.ok or res.dry_run:
            return res, strat, i > 0
        errors.append(f"[{strat.value}] {res.stderr}".rstrip())
        # H1b loosen-on-failure: a recipe solve that died on an exact build/version pin can often
        # be rebuilt by dropping the build strings. Retry once, in place, before falling through to
        # the next strategy — the strict recipe stays the primary attempt (no regression when it
        # already solves). Gated to conda *solve* failures so a pip off-index error (which stripping
        # cannot fix) is left for the envdoctor OFF_INDEX_WHEEL repair instead.
        if (
            not loosen_tried
            and strat in (BuildStrategy.ENV_YAML, BuildStrategy.CONDA_EXPORT)
            and spec.recipe
            and _SOLVE_FAILURE_RE.search(res.stderr or "")
        ):
            loosen_tried = True
            # The strict recipe attempt may have left a partial prefix; clear it so the loosened
            # re-solve (same env name) creates cleanly instead of hitting "prefix already exists".
            _clear_partial_target()
            try:
                relaxed = _build_by_strategy(conda, spec, target_env, strat, loosen=True)
            except CondaError as exc:
                errors.append(f"[{strat.value}+loosened] {exc}".rstrip())
            else:
                if relaxed.ok or relaxed.dry_run:
                    return relaxed, strat, True  # a loosened recipe is a fallback → fell_back=True
                errors.append(f"[{strat.value}+loosened] {relaxed.stderr}".rstrip())
    # A failed build leaves no prefix (hunt 2026-09-30, u36-setup-install-3). conda does not roll back
    # an `env create` whose pip layer died, so the last strategy can leave a prefix with a working
    # python and no tool packages. classify() then read it as SKIP "already healthy" on the next run
    # whenever the health probe cannot see the missing layer (13 tools probe `__future__`), and the
    # self-heal loop's health re-observation came back clean and reported "recovered". Same guard as
    # the inter-attempt cleanup: only a managed, unprotected <basic>_* target is ever removed.
    _clear_partial_target()
    # Every attempted strategy's full error, strategy-labeled — so a build-failure log shows
    # the WHOLE chain (not just the last strategy) and the remediation classifier (A2) sees
    # every signature. The final RunResult carries the joined text as its stderr.
    return RunResult(returncode=1, stderr="\n\n".join(errors)), primary, False


# --------------------------------------------------------------------------- #
# Optional self-review auto-repair (best-effort; never hard-fails provisioning)
# --------------------------------------------------------------------------- #
def _maybe_self_review(spec: ToolSpec, target_env: str, stderr: str, log: SessionLog | None) -> dict | None:
    # The documented SOG_* boolean set -- ``config._env_bool``: strip, lower, true/1/yes/on. Kept
    # inline (this module's import head is stdlib-only) but deliberately NOT this module's
    # ``_TRUTHY``, which also carries ``y``: ``self_review_loop`` re-checks its own ``_enabled()``
    # on entry, so a gate WIDER than that check imports, builds a context, calls the loop, and
    # records ``attempted`` for a repair that returned instantly. Narrower is how this line read
    # before -- ("1","true","yes"), unstripped -- so ``SOG_SELF_REVIEW_ENABLED=on``, which the web
    # settings panel writes through verbatim and then reports back as enabled, did nothing here.
    if os.environ.get("SOG_SELF_REVIEW_ENABLED", "").strip().lower() not in ("true", "1", "yes", "on"):
        return None
    try:
        from tools_user.self_review import self_review_loop  # type: ignore

        from .envdoctor import build_remediation_context
    except Exception as exc:
        if log is not None:
            log.event("self_review_unavailable", server=spec.server_key, error=str(exc))
        return None
    try:
        # Fully-populated RemediationContext (7 fields) + positional error_msg — the
        # previously-inert call passed only env_name/error_text and always TypeError'd.
        ctx = build_remediation_context(spec, target_env)
        success, final_error, history = self_review_loop(ctx, stderr, stderr=stderr)
        return {
            "attempted": True,
            "success": bool(success),
            # redact-before-clip (Class-4): final_error is the LLM/conda remediation transcript tail —
            # it can echo a registered index-URL token. This dict is persisted to state.json via the
            # tool's self_review, so a bare [:200] would split a straddling secret into a head fragment
            # redact()/_sanitize can no longer match. Redact the FULL text, then clip.
            "final_error": redact(str(final_error))[:200],
            "rounds": len(history),
        }
    except Exception as exc:
        return {"attempted": True, "error": redact(str(exc))[:200]}


# --------------------------------------------------------------------------- #
# Agent-monitored self-healing provisioning (A1 capture + A2 loop + A4 planner)
# --------------------------------------------------------------------------- #
def _ev(log: SessionLog | None, kind: str, **fields) -> None:
    if log is not None:
        log.event(kind, **fields)


# C7 — on a re-failure the prior build log is KEPT: the new authoritative stderr is appended under
# this delimiter instead of clobbering the earlier attempt (a resume / second failure used to lose
# it). ASCII-only, so it never trips a downstream code-block parser.
_BUILD_LOG_DELIM = "=" * 24 + " next build attempt " + "=" * 24
_BUILD_LOG_CLIP_MARK = "...[older build-log attempts clipped to the tail]...\n"


def _persist_build_log(basic_env: str, server_key: str, text: str) -> Path | None:
    """Write the build stderr to ``<state>/logs/<basic>_<server>.build.log`` and return the path
    (``None`` if it could not be written). Never raises — a logging failure must not sink provisioning.

    This is where to see what actually went wrong: the summary keeps only a
    300-char snippet, but the whole multi-strategy error lands here. Two round-2 (Part C) guarantees:

    * **C1 (redact).** The text is masked with :func:`session_log.redact` before it touches disk. This
      file is fed straight into ``log_monitor.read_tail`` → the planner LLM prompt, so a credentialed
      index URL or any ``register_secret``'d token echoed in pip/conda stderr must never land here raw.
    * **C7 (keep prior evidence + cap).** The FIRST failure for a target writes the (redacted) stderr
      verbatim — the common single-failure case is byte-identical to before. A LATER failure **appends**
      under a delimiter instead of clobbering the earlier attempt, and the whole file is clipped to its
      TAIL under :data:`constants.BUILD_LOG_CLIP_BYTES` so repeated failures can't grow it without bound
      (the newest attempt is always at the tail, exactly what ``log_monitor.read_tail`` surfaces).
    """
    try:
        logs = constants.logs_dir()
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / f"{basic_env}_{server_key}.build.log"
        body = redact(text or "")
        prior = ""
        with contextlib.suppress(OSError):
            prior = path.read_text(encoding="utf-8", errors="replace")
        if prior:
            combined = prior.rstrip("\n") + "\n" + _BUILD_LOG_DELIM + "\n" + body
        else:
            combined = body
        cap = constants.BUILD_LOG_CLIP_BYTES
        if len(combined) > cap:
            combined = _BUILD_LOG_CLIP_MARK + combined[-cap:]
        path.write_text(combined, encoding="utf-8")
        return path
    except OSError:
        return None


def _attempt_provision_repairs(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    result: ToolResult,
    stderr: str,
    *,
    io: PromptIO,
    log: SessionLog | None,
    remediation=None,
) -> bool:
    """Detect an ENVIRONMENT problem behind a build FAILURE and self-heal it, bounded.

    The provision-phase twin of :func:`testing._attempt_env_repairs`: classify the (joined,
    multi-strategy) build ``stderr``; a non-env failure classifies as ``None`` → no
    deterministic repair (the honest FAIL is kept). A recognized env issue — an
    ``OFF_INDEX_WHEEL`` PyTorch/PyG/git stack, a missing module, a broken env — is repaired
    with the guarded :mod:`envdoctor` primitives, then health is re-confirmed by
    re-classification: ``result.ok/built`` flip to ``True`` **only** on a genuine ``SKIP``
    (healthy) re-pass. Bounded by ``MAX_ENV_REPAIRS`` with a no-progress guard on the
    ``(kind, package)`` signature.

    When the deterministic classifier can't name the failure **and** an LLM ``remediation``
    planner is wired (A4, interactive + opt-in only), that planner is consulted as a last
    resort. Env-only; never touches agent/tool source. A no-op when ``SOG_PROVISION_AUTOREPAIR``
    is falsy. Runs *outside* the per-tool checklist frame (a RECREATE opens its own box).
    Returns ``True`` iff the tool now builds and imports healthily.
    """
    # The documented SOG_* falsy set -- ``config._env_bool``: strip, lower, false/0/no/off. Kept
    # inline (this module's import head is stdlib-only) and deliberately NOT this module's
    # ``_TRUTHY``, for the reason recorded at ``_maybe_self_review``. ``off`` and the strip were
    # both missing, so the documented way to switch this default-ON loop off did not switch it off.
    if os.environ.get("SOG_PROVISION_AUTOREPAIR", "1").strip().lower() in ("false", "0", "no", "off"):
        return False
    from .installer_scientist import InstallerScientist

    _ev(log, "provision_remediate_start", server=spec.server_key, target=target)

    # observe(): the FIRST turn seeds with the captured multi-strategy build stderr; each later turn
    # re-derives the failure text from a health probe (NOT a fresh build() — repair_env already did any
    # needed recreate; rebuilding would wipe an in-place pip_install and error on an existing env). The
    # probe stderr is what surfaces a *cascading* missing dependency (fix six → probe reveals tqdm).
    seed = {"first": True}

    def observe() -> str:
        if seed["first"]:
            seed["first"] = False
            return stderr
        res = _health_run(conda, spec, target)
        if res.ok:
            return ""
        # build() removes the prefix of a build that failed, so an absent target means nothing has
        # built it since (a transient-retry turn changes nothing; a repair whose own rebuild failed).
        # Say so in the shape classify_failure reads as ENV_ABSENT, so the next turn rebuilds through
        # the budgeted recreate lane — the probe's own error for a missing env depends on the conda
        # frontend and need not classify at all (hunt 2026-09-30, u36-setup-install-3).
        try:
            present = conda.env_exists(target)
        except Exception:
            present = True  # unprobeable → report the probe's own text
        if not present:
            return f"Could not find conda environment: {target} (the last build of it failed and left no env)"
        return (res.stderr or res.stdout or f"import {spec.import_check or '?'} failed").strip()

    def recheck() -> bool:
        return classify(conda, spec, target) is ToolStatus.SKIP  # authoritative health re-check

    # Lane 2 opt-in (A4): the planner runs ONLY with a validated chat handle AND
    # SOG_PROVISION_LLM_REMEDIATION set; otherwise the self-heal stays fully deterministic.
    gate = os.environ.get("SOG_PROVISION_LLM_REMEDIATION", "").strip().lower() in _TRUTHY
    chat = remediation if (gate and remediation is not None) else None

    agent = InstallerScientist(conda, spec, basic_env, io=io, chat=chat, llm_enabled=bool(chat), log=log)
    try:
        # Live "thinking box": the self-heal loop's per-turn reasoning/action + the build-log line it is
        # watching redraw in place. Inert (zero bytes, no thread) on a non-TTY / muted stream, so the
        # deterministic-only path stays byte-for-byte what it was — see progress.think().
        with progress.think(getattr(conda, "progress", None), f"Self-heal · {target}", fallback_stream=io.out) as box:
            outcome = agent.remediate(observe=observe, recheck=recheck, box=box)
    except Exception as exc:  # the agent must never sink the provision phase
        io.err(f"{target}: provision remediation crashed: {exc}")
        _ev(log, "provision_remediated", server=spec.server_key, target=target, repaired=False, error=str(exc)[:200])
        return False

    result.repairs.extend(outcome.repairs)
    if outcome.repaired:
        result.ok = True
        result.built = True
        result.strategy = f"{result.strategy or spec.build_strategy.value}+remediation:{outcome.last_action}"
        _ev(
            log,
            "provision_remediated",
            server=spec.server_key,
            target=target,
            repaired=True,
            action=outcome.last_action,
            strategy=result.strategy,
        )
        return True
    # A real-but-not-env problem (Lane 3): tell the user exactly what to do, then keep the honest FAIL.
    if outcome.needs_attention:
        io.warn(f"{target}: {outcome.needs_attention}")
        result.needs_attention = outcome.needs_attention
        result.messages.append(f"needs attention: {outcome.needs_attention}"[:300])
    _ev(log, "provision_remediated", server=spec.server_key, target=target, repaired=False, reason=outcome.reason)
    return False


def _run_llm_remediation(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    result: ToolResult,
    stderr: str,
    remediation,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> bool:
    """A4 last resort: ask the wizard's own LLM for an env-only, allowlist-validated fix plan
    and execute it via the guarded primitives, then re-confirm health. Lazily imports the
    planner so the ``sog_install`` core stays stdlib+pyyaml; any failure degrades to
    an honest ``False`` (the FAIL stands). ``remediation`` is the live ``ChatClient`` handle
    (``self.source.chat``) threaded down from the wizard; gating already happened at the call
    site (handle present + ``SOG_PROVISION_LLM_REMEDIATION`` opt-in).

    Retained as a direct single-shot entry point (and a stable monkeypatch seam). The default
    provision path now routes the whole three-lane self-heal — deterministic repair, this LLM
    hand-off, honest surfacing — through :class:`installer_scientist.InstallerScientist`, which
    reaches the *multi-turn* planner (:func:`remediation_planner.plan_env_fix_iter`) itself."""
    try:
        from .remediation_planner import run_llm_remediation
    except Exception as exc:  # planner import must never crash provisioning
        # redact-before-clip (Class-4): redact FULL text before clipping (matches this file's siblings).
        _ev(log, "provision_llm_unavailable", server=spec.server_key, error=redact(str(exc))[:200])
        return False
    try:
        return bool(run_llm_remediation(conda, spec, basic_env, target, result, stderr, remediation, io=io, log=log))
    except Exception as exc:
        io.warn(f"{target}: LLM remediation errored ({exc}) — keeping FAIL")
        _ev(
            log,
            "provision_remediated",
            server=spec.server_key,
            target=target,
            repaired=False,
            action="llm",
            # redact-before-clip (Class-4): redact FULL text before clipping.
            error=redact(str(exc))[:200],
        )
        return False


# --------------------------------------------------------------------------- #
# Live checklist-box steps (Fix 3)
# --------------------------------------------------------------------------- #
def _box_title(spec: ToolSpec, index: int | None, total: int | None) -> str:
    """``server · strategy · i/total`` — the header of this tool's checklist box."""
    parts = [spec.server_key, spec.build_strategy.value]
    if index and total:
        parts.append(f"{index}/{total}")
    return " · ".join(parts)


def _strategy_steps(spec: ToolSpec, target: str, status: ToolStatus) -> list[str]:
    """The ordered step labels this build WILL emit — each string is byte-identical to the
    ``label=`` a :class:`~sog_install.envtools.Conda` mutating call passes, so the
    box's :meth:`~sog_install.progress.StepBox.task` matches and advances it.

    Covers the primary strategy only; if :func:`build` falls back to another strategy at
    runtime its (different) label is appended live as a new box step.
    """
    steps: list[str] = []
    if status is ToolStatus.RECREATE:
        steps.append(f"removing env {target}")
    strat = spec.build_strategy
    if strat is BuildStrategy.PIP:
        steps.append(f"creating env {target}")
        steps.append(f"pip installing into {target}")
    elif strat is BuildStrategy.CONDA_CLONE:
        steps.append(f"cloning {spec.source_env} → {target}")
    elif strat in (BuildStrategy.ENV_YAML, BuildStrategy.CONDA_EXPORT):
        steps.append(f"building env {target}")
    return steps


# --------------------------------------------------------------------------- #
# Provision one / all
# --------------------------------------------------------------------------- #
def provision_one(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    *,
    confirm: Callable[[Proposal], bool],
    io: PromptIO,
    log: SessionLog | None = None,
    index: int | None = None,
    total: int | None = None,
    remediation=None,
    _allow_repairs: bool = True,
) -> ToolResult:
    # ``_allow_repairs`` breaks a re-entrancy cycle (a94#1). A build failure here can invoke
    # ``_attempt_provision_repairs`` → InstallerScientist → envdoctor → ``_repair_via_recreate``,
    # which calls back into ``provision_one`` to rebuild the env. If that rebuild also fails it would
    # spawn *another* remediation, recursing until RecursionError and blowing past MAX_ENV_RECREATES
    # (budgets are per-InstallerScientist and a fresh one is built at each level, so they never bind).
    # The recreate path therefore calls us with ``_allow_repairs=False``: a rebuild becomes a plain
    # build whose failure is observed by the requesting InstallerScientist (via recheck), not a new
    # nested remediation.
    target = spec.target_env(basic_env)
    result = ToolResult(server_key=spec.server_key, target_env=target)

    try:
        status = classify(conda, spec, target)
    except Exception as exc:
        # A slow/loaded box (or a stalled NFS mount) can make the cheap import/exists probe raise
        # CondaError mid-run — most often on a *resume*, where the env was already built last run and
        # `_healthy`'s import probe merely times out under load (a genuinely broken package returns
        # False, not raises — `_health_run` uses check=False). That transient infra error must NEVER
        # abort the whole run: unguarded it unwinds to `provision_all`'s catch → ToolResult(ok=False) →
        # and under `on_tool_fail=abort` it breaks the ENTIRE loop, exactly the invariant
        # `_reusable_existing_env` guards ("a health probe raising CondaError must never abort
        # provisioning"). Defer just this one tool — mark it skipped (never a false-healthy claim,
        # never a destructive rebuild on an unknown env state) and let the finalize resolver, which
        # independently re-probes every env, make the real enable/disable wiring call. A tool whose env
        # is in fact healthy still gets wired at config time; re-running provisions it once the box frees.
        result.skipped = True
        result.messages.append(f"probe error, deferred: {exc}"[:200])
        io.note(
            f"couldn't probe {spec.server_key} right now ({type(exc).__name__}) — "
            "skipping it this run; re-run to pick it up"
        )
        _ev(log, "provision_probe_error", server=spec.server_key, target=target, error=str(exc)[:200])
        return result
    result.status = status.value

    if status is ToolStatus.NONE:
        result.ok = True
        return result
    if status is ToolStatus.SKIP:
        result.ok = True
        io.ok(f"{target} already healthy — skipping")
        return result

    # Env reuse (R1): the managed <basic>_<server> env is absent (CREATE), but the user may
    # already have a healthy env that satisfies this tool — most often the very *source* env this
    # run would otherwise clone (card_env, tangram-env, SpatialDE, …). Wire the worker to it
    # READ-ONLY and skip the (re)build entirely: "don't clone what you already have." The scan is
    # health-gated (the env must actually import the tool's package), so a merely same-named env
    # that lacks the package is never adopted. result.target_env stays the MANAGED name, so every
    # downstream namespace guard (assert_deletable_env, orphan cleanup) still refuses to touch the
    # reused env, and the live mcp_resolver independently re-derives the same interpreter when it
    # writes the config. Opt out with SOG_SETUP_NO_ENV_REUSE=1 to restore strict building.
    if status is ToolStatus.CREATE:
        reuse_env = _reusable_existing_env(conda, spec, basic_env)
        if reuse_env:
            result.status = ToolStatus.REUSE.value
            result.reused_env = reuse_env
            result.ok = True
            io.ok(f"reusing your existing '{reuse_env}' env for {spec.server_key} — no rebuild needed")
            _ev(log, "provision_reuse", server=spec.server_key, target=target, reused_env=reuse_env)
            return result

    if not confirm(propose(spec, target, status, conda=conda)):
        result.skipped = True
        result.messages.append("user declined")
        io.note(f"skipped {spec.server_key}")
        return result

    # A live checklist box (Fix 3) frames the remove+build on a real TTY; its steps mirror
    # the labels the conda calls below emit, so each call advances the matching step. On a
    # muted / scripted / non-TTY run ``box_open`` is False and we keep the plain io.say lines
    # (byte-for-byte unchanged).
    box = getattr(conda, "progress", None)
    box_open = box is not None and hasattr(box, "build") and getattr(box, "enabled", False)
    frame = (
        box.build(_box_title(spec, index, total), _strategy_steps(spec, target, status))
        if box_open
        else contextlib.nullcontext()
    )

    with frame:
        # RECREATE: remove the unhealthy managed env first (guarded).
        if status is ToolStatus.RECREATE:
            constants.assert_deletable_env(basic_env, target)  # raises unless <basic>_* and unprotected
            if not box_open:
                io.say(f"  removing unhealthy {target} before rebuild…")
            # Best-effort: if the unhealthy env can't be removed (locked/slow → CondaError even at
            # check=False, or OSError), don't let it escape provision_one into provision_all's per-tool
            # crash guard — that skips build()'s multi-strategy self-heal + full-log. Note it and fall
            # through to build(), which surfaces any real collision honestly (like
            # wizard._remove_orphan_tool_envs). The protection guard above stays hard — a foreign/
            # protected env must still raise, never be swallowed here.
            try:
                conda.remove_env(target)
            except (CondaError, OSError) as exc:
                io.note(f"could not remove {target} before rebuild ({type(exc).__name__}); continuing")
                _ev(log, "recreate_remove_failed", server=spec.server_key, target=target, error=str(exc))

        if not box_open:
            io.say(f"  building {target} via {spec.build_strategy.value}…")
        res, strategy_used, fell_back = build(conda, spec, target, basic_env)
    result.strategy = strategy_used.value
    result.fell_back = fell_back

    # A build that exits 0 is not yet a working env (hunt 2026-09-30, u38b-specs-a-14): a recipe can
    # solve and install cleanly and still lack the tool's own package (an R library installed out of
    # band in the source env and absent from the recipe). This used to print "<target> ready", mark
    # the tool DONE and skip every repair, until the resolver quietly wired it enabled:false. Run the
    # same import/library probe classify() uses; a failing probe is a build failure carrying the probe's
    # own stderr, so the self-heal loop and the honest FAILED state apply.
    probe_failed = False
    if res.ok and not res.dry_run:
        try:
            health = _health_run(conda, spec, target)
        except Exception as exc:
            health = None
            io.note(
                f"built {target}, but its health probe could not run ({type(exc).__name__}); "
                "the config step re-probes it before wiring"
            )
        if health is not None and not health.ok:
            probe_failed = True
            is_r = getattr(spec, "worker_kind", "") == "rscript"
            what = f"library({spec.import_check})" if is_r else f"import {spec.import_check}"
            detail = (health.stderr or health.stdout or "").strip() or f"{what} failed"
            res = RunResult(
                returncode=1,
                stderr=f"[{strategy_used.value}] {target} was built, but its health probe `{what}` failed:\n{detail}",
            )

    if res.ok or res.dry_run:
        result.built = True
        result.ok = True
        if fell_back:
            # Truthful (A1): name the strategy that ACTUALLY succeeded. For a clone-primary tool
            # the working fallback is the recipe rebuild (conda_export), not a clone — the old
            # hardcoded "used clone of X" read as if a clone had failed and confused the user.
            io.note(f"(primary '{spec.build_strategy.value}' unavailable here — built via '{result.strategy}')")
        io.ok(f"{target} ready")
    else:
        # A1 — persist the WHOLE multi-strategy error where the user can read it; keep only a
        # 300-char snippet in the summary. Point them at the full log explicitly.
        # redact-before-clip (Class-4): res.stderr is raw multi-strategy conda/pip output that can
        # carry a registered index-URL token; this snippet is persisted to state.json via the tool's
        # messages (through wizard.set_tool), so redact the FULL text FIRST — a bare [:300] would split
        # a straddling secret into a head fragment redact()/_sanitize can no longer match. `or ""` also
        # closes a latent None[:300]. Mirrors the sibling transcript-row clip just below.
        result.messages.append(redact(res.stderr or "")[:300])
        log_path = _persist_build_log(basic_env, spec.server_key, res.stderr)
        _ev(
            log,
            "provision_build_failed",
            server=spec.server_key,
            target=target,
            strategy=result.strategy,
            # C7 — clip the transcript row to the last N KB (the FULL, redacted text lives in the
            # <target>.build.log the log_file points at); redact-before-clip so a secret straddling
            # the tail boundary is masked, not split into a leaking fragment.
            stderr=redact(res.stderr or "")[-constants.BUILD_LOG_CLIP_BYTES :],
            log_file=str(log_path) if log_path else "",
        )
        io.err(f"{target} was built but fails its health probe" if probe_failed else f"failed to build {target}")
        if log_path is not None:
            io.note(f"full log: {log_path}")
        # A2/A4 — agent-monitored self-heal (env-only, guarded). Flips ok only on a confirmed
        # rebuild+import; falls back to the legacy self-review hook only if nothing recovered it.
        # ``_allow_repairs`` is False when we were called *by* a recreate-repair (a94#1): a nested
        # rebuild must not spawn a fresh remediation, or the two loops recurse without bound.
        if _allow_repairs and _attempt_provision_repairs(
            conda, spec, basic_env, target, result, res.stderr, io=io, log=log, remediation=remediation
        ):
            io.ok(f"{target} recovered after remediation")
        else:
            result.self_review = _maybe_self_review(spec, target, res.stderr, log)
            if result.self_review and result.self_review.get("attempted"):
                # re-classify after a repair attempt (guard the probe: a transient CondaError here
                # must not mask the already-recorded build failure or skip the log bookkeeping below —
                # treat an unconfirmable re-probe as "not repaired", which is the honest default).
                try:
                    repaired = classify(conda, spec, target) is ToolStatus.SKIP
                except Exception:
                    repaired = False
                if repaired:
                    result.ok = True
                    result.built = True
                    io.ok(f"{target} repaired via self-review")

    if log is not None:
        log.event(
            "provision_tool",
            server=spec.server_key,
            target=target,
            status=result.status,
            strategy=result.strategy,
            ok=result.ok,
            fell_back=result.fell_back,
            skipped=result.skipped,
            repairs=len(result.repairs),
        )
    return result


def provision_all(
    conda: Conda,
    specs: dict[str, ToolSpec],
    plan: ProvisionDecision,
    basic_env: str,
    *,
    confirm: Callable[[Proposal], bool],
    io: PromptIO,
    log: SessionLog | None = None,
    on_tool_start: Callable[[ToolPlan], None] | None = None,
    on_tool_done: Callable[[ToolResult], None] | None = None,
    remediation=None,
) -> list[ToolResult]:
    """Provision every tool in ``plan.tools`` (order preserved). Honors ``on_tool_fail``.

    ``on_tool_start``/``on_tool_done`` (optional) fire immediately before and after each
    tool so the caller can persist *live* per-tool progress. That live marking is what
    makes a crash recoverable: the in-flight tool is left ``in_progress`` (its env a
    possible partial build), and resume force-removes that orphan before rebuilding
    (see :meth:`state.SetupState.orphan_tool_envs`)."""
    results: list[ToolResult] = []
    total = len(plan.tools)
    for index, tp in enumerate(plan.tools, start=1):
        spec = specs.get(tp.server_key)
        if spec is None:
            io.warn(f"no spec for {tp.server_key} — skipping")
            r = ToolResult(server_key=tp.server_key, target_env=tp.target_env, messages=["no spec"])
            results.append(r)
            if on_tool_done is not None:
                on_tool_done(r)
            continue
        if on_tool_start is not None:
            on_tool_start(tp)
        try:
            r = provision_one(
                conda,
                spec,
                basic_env,
                confirm=confirm,
                io=io,
                log=log,
                index=index,
                total=total,
                remediation=remediation,
            )
        except Exception as exc:  # one tool's crash must never sink the rest of the run
            io.err(f"{tp.server_key} crashed during provisioning: {exc}")
            # redact-before-clip (Class-4): this is a GENERIC except, so `exc` is not guaranteed to be a
            # CondaError (whose str is message-only) — an arbitrary crash could embed a registered secret.
            # Clipping first would leave a straddling secret's head unmatchable by log.event's / the result
            # sink's redact. Redact the FULL text, then clip (mirrors this module's :651/:655).
            r = ToolResult(server_key=tp.server_key, target_env=tp.target_env, messages=[redact(f"crash: {exc}")[:300]])
            if log is not None:
                log.event(
                    "provision_tool_crash", server=tp.server_key, target=tp.target_env, error=redact(str(exc))[:300]
                )
        results.append(r)
        if on_tool_done is not None:
            on_tool_done(r)
        if not r.ok and not r.skipped and plan.on_tool_fail is OnToolFail.ABORT:
            io.err(f"aborting: {tp.server_key} failed and on_tool_fail=abort")
            break
    ok = sum(1 for r in results if r.ok)
    io.say(f"  provisioned {ok}/{len(results)} tool env(s)")
    return results
