"""
Preflight machine checks (Stage A, phase 0).

Read-only probes of the host: Python version, a conda/mamba manager, free disk,
GPU presence, outbound network, the local mini datasets, and a writable state
dir. Returns a list of :class:`CheckResult`; the driver hard-stops on any
``fail`` and asks to confirm-continue on ``warn``.

Stdlib only, no side effects beyond creating (and removing) a probe file to test
writability.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import constants

_OK, _WARN, _FAIL = "ok", "warn", "fail"


@dataclass
class CheckResult:
    name: str
    level: str  # ok | warn | fail
    detail: str
    hint: str = ""

    @property
    def is_fail(self) -> bool:
        return self.level == _FAIL

    @property
    def is_warn(self) -> bool:
        return self.level == _WARN


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def check_python() -> CheckResult:
    v = sys.version_info
    if (v.major, v.minor) >= (3, 11):
        return CheckResult("python", _OK, f"Python {v.major}.{v.minor}.{v.micro}")
    return CheckResult(
        "python",
        _FAIL,
        f"Python {v.major}.{v.minor} is too old",
        hint="SpatialOmicsLab requires Python >= 3.11.",
    )


def _candidate_managers() -> list[str]:
    """Ordered, de-duplicated conda/mamba/micromamba executables to probe.

    ``PATH`` first (a ``which`` hit is already resolved + executable, so it is trusted as-is),
    then the ``$CONDA_EXE`` / ``$MAMBA_EXE`` pointers conda exports, then the usual install roots
    — so a Miniconda/Miniforge that is installed but not ``conda activate``-d is still found
    instead of tripping a hard preflight fail. Constructed (non-PATH) paths are verified to be
    real executables before use.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _add(path: str | None, *, verify: bool) -> None:
        if not path:
            return
        if verify:
            path = os.path.realpath(path)
            if not (os.path.isfile(path) and os.access(path, os.X_OK)):
                return
        if path not in seen:
            seen.add(path)
            out.append(path)

    for exe in ("conda", "mamba", "micromamba"):
        _add(shutil.which(exe), verify=False)
    for var in ("CONDA_EXE", "MAMBA_EXE"):
        _add(os.environ.get(var), verify=True)
    roots = [
        os.environ.get("CONDA_ROOT"),
        os.environ.get("MAMBA_ROOT_PREFIX"),
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniforge3"),
        os.path.expanduser("~/mambaforge"),
        os.path.expanduser("~/micromamba"),
        "/opt/conda",
    ]
    _win = os.name == "nt"
    for root in roots:
        if not root:
            continue
        for exe in ("conda", "mamba", "micromamba"):
            if _win:
                # Windows conda has no per-root ``bin/``: console launchers live under ``Scripts\*.exe``
                # and ``condabin\*.bat``; a standalone micromamba often ships as ``<root>\micromamba.exe``
                # (or under ``Library\bin``). A POSIX-only ``bin/<exe>`` probe finds none of these.
                _add(os.path.join(root, "Scripts", exe + ".exe"), verify=True)
                _add(os.path.join(root, "condabin", exe + ".bat"), verify=True)
                _add(os.path.join(root, "Library", "bin", exe + ".exe"), verify=True)
                _add(os.path.join(root, exe + ".exe"), verify=True)
            else:
                _add(os.path.join(root, "bin", exe), verify=True)
                _add(os.path.join(root, "condabin", exe), verify=True)
    return out


def find_conda() -> tuple[str | None, str]:
    """Return ``(manager_exe, version_str)`` — prefers conda, then mamba/micromamba.

    Searches beyond ``PATH`` (see :func:`_candidate_managers`) so a conda that is installed but
    not activated no longer trips a hard preflight fail. Returns the first candidate that exists,
    even if the ``--version`` probe itself fails (it is still usable by absolute path)."""
    for exe in _candidate_managers():
        name = Path(exe).name
        try:
            out = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                errors="replace",  # a non-UTF-8 byte (localized/LANG=C output) must not raise and abort preflight
                timeout=constants.CONDA_RUN_TIMEOUT_SEC,
            )
            return exe, out.stdout.strip() or f"{name} (version unknown)"
        except (subprocess.SubprocessError, OSError):
            return exe, f"{name} (version probe failed)"
    return None, ""


def check_conda() -> CheckResult:
    exe, version = find_conda()
    if exe:
        return CheckResult("conda", _OK, version)
    return CheckResult(
        "conda",
        _FAIL,
        "no conda/mamba/micromamba on PATH",
        hint="Install Miniconda/Miniforge — the wizard creates one env per tool.",
    )


def _envs_mount() -> str:
    """The mount where conda creates the per-tool envs, so the disk gate measures THIS filesystem
    (tool envs land there, not beside the repo — often a different mount).

    Derived from the running interpreter's prefix via :func:`constants.conda_envs_root`, which is
    portable across ``/opt/conda``, ``~/miniconda3``, and standalone/symlinked ``micromamba`` (the
    old ``<root>/bin/<exe>`` assumption measured the wrong filesystem for those). Falls back to the
    base prefix, then the discovered manager's root, then the repo root."""
    envs_root = constants.conda_envs_root()
    for candidate in (envs_root, os.path.dirname(envs_root.rstrip("/"))):
        if candidate and os.path.isdir(candidate):
            return candidate
    exe, _ = find_conda()
    if exe:
        root = Path(exe).resolve().parent.parent  # <root>/bin/<exe> -> <root>
        for candidate in (root / "envs", root):
            if candidate.exists():
                return str(candidate)
    return str(constants.repo_root())


def check_disk() -> CheckResult:
    try:
        free_gb = shutil.disk_usage(_envs_mount()).free / (1024**3)
    except OSError as exc:
        return CheckResult("disk", _WARN, f"could not read free space: {exc}")
    if free_gb < constants.MIN_DISK_GB_BASE:
        return CheckResult(
            "disk",
            _FAIL,
            f"only {free_gb:.0f} GB free",
            hint=f"Need at least {constants.MIN_DISK_GB_BASE} GB for a base env; tool envs add several GB each.",
        )
    if free_gb < constants.WARN_DISK_GB:
        return CheckResult(
            "disk",
            _WARN,
            f"{free_gb:.0f} GB free (tight)",
            hint=f"Heavy tool envs may not fit; {constants.WARN_DISK_GB}+ GB recommended.",
        )
    return CheckResult("disk", _OK, f"{free_gb:.0f} GB free")


def check_gpu() -> CheckResult:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return CheckResult(
            "gpu",
            _WARN,
            "no NVIDIA GPU detected",
            hint="GPU tools fall back to CPU where possible (some may be skipped).",
        )
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            errors="replace",  # a non-UTF-8 byte in nvidia-smi output must not raise and abort preflight
            timeout=constants.CONDA_RUN_TIMEOUT_SEC,
        )
        # Trust stdout only on a clean exit. A present-but-broken driver (the classic
        # "Failed to initialize NVML: Driver/library version mismatch") leaves nvidia-smi on PATH but
        # makes this query exit non-zero with the error on STDERR and an EMPTY stdout — so the old
        # ``... or ["GPU present"]`` fallback reported the GPU as _OK and suppressed the guide's
        # CPU-fallback note (guide.py:310, gated on level == "warn"). Fold a failed/empty query into the
        # same _WARN the OSError arm already emits, so the user is honestly told the GPU is unusable.
        if out.returncode != 0 or not out.stdout.strip():
            return CheckResult("gpu", _WARN, "nvidia-smi present but query failed")
        first = out.stdout.strip().splitlines()[0]
        return CheckResult("gpu", _OK, first)
    except (subprocess.SubprocessError, OSError):
        return CheckResult("gpu", _WARN, "nvidia-smi present but query failed")


def check_network(url: str = "https://pypi.org", timeout: int | None = None) -> CheckResult:
    timeout = timeout or constants.NETWORK_TIMEOUT_SEC
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return CheckResult("network", _OK, f"reachable ({url})")
    except Exception as exc:
        return CheckResult(
            "network",
            _WARN,
            f"no outbound network ({exc})",
            hint="Provisioning from remote recipes needs network; local recipes still work.",
        )


def check_mini_data() -> CheckResult:
    d = constants.mini_data_dir()
    spatial = d / constants.MINI_SPATIAL
    sc_ref = d / constants.MINI_SC_REF
    missing = [p.name for p in (spatial, sc_ref) if not p.exists()]
    if not missing:
        return CheckResult("mini_data", _OK, f"mini datasets present in {d}")
    # The mini_*.h5ad are gitignored even in the development tree — but the small creation_demo
    # dataset sits beside them. If it's present, the real-data probe still has a genuine
    # dataset, so this is not a problem worth a scary warning.
    if constants.fallback_spatial_dataset().is_file():
        return CheckResult(
            "mini_data",
            _OK,
            f"using the demo dataset ({constants.FALLBACK_SPATIAL_REL}) for the real-data probe",
            hint=f"the full mini datasets ({', '.join(missing)}) aren't in this checkout; "
            "the small demo dataset will be staged instead.",
        )
    why = constants.mini_data_note()
    return CheckResult(
        "mini_data",
        _WARN,
        f"missing {', '.join(missing)}" + (f" ({why})" if why else ""),
        hint="Tier-1 worker tests need these; they ship in the development tree's test/test_data/.",
    )


def check_writable(*, prune: bool = True, create: bool = True) -> CheckResult:
    if not create and not constants.state_dir().exists():
        # A dry run creates nothing: judge the nearest existing ancestor instead of making the tree
        # (hunt 2026-09-30, u35-setup-ux-8).
        anc = constants.state_dir()
        while not anc.exists() and anc != anc.parent:
            anc = anc.parent
        ok = os.access(anc, os.W_OK)
        return CheckResult("writable", _OK if ok else _FAIL, f"state dir would be created under {anc}")
    try:
        # L1: ``prune=False`` still creates the state-dir tree and probes writability, but does NOT
        # unlink accumulated run logs. A read-only caller (``sog-setup doctor``) passes it so merely
        # *inspecting* a run never silently deletes the user's oldest logs; the wizard keeps the
        # default (it is about to write logs anyway, so per-run size-capping is correct there).
        constants.ensure_state_dirs(prune=prune)
        probe = constants.state_dir() / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult("writable", _OK, f"state dir writable ({constants.state_dir()})")
    except OSError as exc:
        return CheckResult(
            "writable",
            _FAIL,
            f"cannot write state dir: {exc}",
            hint="The wizard needs to persist resume state under .sog_setup/.",
        )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_preflight(*, check_net: bool = True, prune: bool = True, create: bool = True) -> list[CheckResult]:
    # ``prune`` forwards to ``check_writable``: a read-only run (doctor) passes ``prune=False`` so the
    # writability probe never deletes accumulated logs. The wizard keeps the default. ``create=False``
    # (a dry run) judges writability without creating the state dir.
    checks = [
        check_python(),
        check_conda(),
        check_disk(),
        check_gpu(),
        check_mini_data(),
        check_writable(prune=prune, create=create),
    ]
    if check_net:
        checks.insert(4, check_network())
    return checks


def has_hard_failure(results: list[CheckResult]) -> bool:
    return any(r.is_fail for r in results)


def summarize(results: list[CheckResult]) -> dict:
    return {r.name: {"level": r.level, "detail": r.detail, "hint": r.hint} for r in results}
