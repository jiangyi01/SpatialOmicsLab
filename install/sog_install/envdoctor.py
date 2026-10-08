"""
envdoctor — turn a tool-test failure into a *classified environment problem* and
(optionally) repair it, without ever touching agent/tool/package source.

Two cleanly separated responsibilities:

1. :func:`classify_failure` — **PURE**. Parse failure text — a Tier-1 worker
   ``stderr``/detail line, OR a Tier-2 in-log ``"Error: …"`` observation that the
   agent's REPL swallowed (``run_python_repl`` returns the exception as a *string*
   instead of raising) — for the signatures of a broken conda env. Returns a typed
   :class:`EnvIssue` or ``None``. **Anything that is not an env problem
   (LLM/auth/rate-limit/task/timeout) returns ``None`` and is never repaired.**

2. :func:`repair_env` — dispatch an :class:`EnvIssue` to the EXISTING
   namespace-guarded primitives (``Conda.pip_install`` into the ``<basic>_<server>``
   env, ``provision.provision_one`` RECREATE). Env-only by construction; every
   mutation passes ``assert_deletable_env`` (via ``provision_one``) or targets a
   ``<basic>_<server>`` env by name (``pip_install``). Never raises on a repair
   failure — it returns ``repaired=False`` so the caller keeps the FAIL verdict.
   A dry-run :class:`~.envtools.Conda` short-circuits every repair (no mutation,
   ``repaired=False``), so a ``--dry-run`` wizard never flips a verdict.

The classifier is deliberately *conservative*: it matches only unambiguous env
signatures and falls through to ``None`` otherwise. A false negative merely leaves a
failure reported (the status quo); a false positive would trigger a mutation on a
non-env problem, which is the outcome we must never risk.

Stdlib only.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING

from . import constants
from .session_log import redact  # C6: RepairResult.detail is persisted (report doc + state), redact its stderr

if TYPE_CHECKING:
    from pathlib import Path

    from .envtools import Conda
    from .prompts import PromptIO
    from .session_log import SessionLog
    from .specs import ToolSpec


class EnvIssueKind(StrEnum):
    MISSING_PY_MODULE = "missing_py_module"  # a Python import is absent → pip install (then RECREATE)
    MISSING_PIP_DIST = (
        "missing_pip_dist"  # a tool named its own pip target ("install X via pip install Y") → pip install Y
    )
    MISSING_R_PACKAGE = "missing_r_package"  # an R library is absent → RECREATE from recipe
    ENV_BROKEN = "env_broken"  # env present but import machinery / ABI is broken → RECREATE
    ENV_ABSENT = "env_absent"  # the env / its interpreter is gone → (re)build
    WIRING_MISSING = "wiring_missing"  # the generated MCP config is absent → re-wire
    OFF_INDEX_WHEEL = "off_index_wheel"  # a build needs a custom-index wheel / git dist → recreate w/ index+git
    BUILD_TOOLCHAIN = "build_toolchain"  # a pip build failed for lack of a C/C++ compiler → add toolchain + finish pip


# --------------------------------------------------------------------------- #
# Trusted-source allowlists — shared by the deterministic off-index repair (A3)
# and the gated LLM remediation planner (A4). Any install URL a repair uses (an
# ``--index-url`` / ``--extra-index-url`` / ``--find-links`` host, or a ``git+``
# origin) MUST be on one of these lists; nothing else is ever fetched.
# --------------------------------------------------------------------------- #
TRUSTED_INDEX_HOSTS: frozenset[str] = frozenset(
    {"pypi.org", "files.pythonhosted.org", "download.pytorch.org", "data.pyg.org"}
)
TRUSTED_GIT_HOSTS: frozenset[str] = frozenset({"github.com"})
# Conda channels a deterministic repair may add (a missing system ``.so`` → ``conda_install``
# from conda-forge, a compiler toolchain, …). Any ``-c <channel>`` a repair passes MUST be on
# this list; an off-list channel aborts the repair before it runs — the same trust discipline
# as :data:`TRUSTED_INDEX_HOSTS`. Reused to tighten the gated planner's channel validation (C5).
TRUSTED_CONDA_CHANNELS: frozenset[str] = frozenset({"conda-forge", "bioconda", "pytorch", "nvidia"})

# Non-PyPI pip packages whose git origin is known and verified. A recipe pins only the
# name/version (e.g. ``stagate-pyg==1.0.0``); the source repo is not derivable from that,
# so this tiny curated map supplies it. (``QIFEIDKN/STAGATE_pyG`` — the PyTorch-Geometric
# variant declaring ``packages=['STAGATE_pyG']`` — verified against the source env's egg
# metadata; the plain ``QIFEIDKN/STAGATE`` repo is the TensorFlow build.) An unknown
# non-PyPI package is left to the LLM planner, never guessed.
KNOWN_GIT_SOURCES: dict[str, str] = {
    # At the sha stagate.env.yaml pins, so a repair rebuilds what the recipe does (hunt 2026-09-30,
    # u36-setup-install-8).
    "stagate-pyg": "git+https://github.com/QIFEIDKN/STAGATE_pyG.git@ae1158ca8cf1eb6bb8ee198298552d44c9ac21db",
    "stagate_pyg": "git+https://github.com/QIFEIDKN/STAGATE_pyG.git@ae1158ca8cf1eb6bb8ee198298552d44c9ac21db",
}

# A pip pin carrying a PEP 440 *local version* (``torch==2.1.0+cpu``,
# ``torch-scatter==2.1.2+pt21cpu``) — such wheels live only on a custom index
# (download.pytorch.org / data.pyg.org), never on PyPI, so a plain rebuild fails.
_LOCAL_VERSION_PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==([0-9][\w.]*)\+([a-z0-9]+)$")
# The requirement token pip names when it cannot resolve a distribution.
_NO_DIST_RE = re.compile(
    r"(?:No matching distribution found for|Could not find a version that satisfies the requirement)\s+(\S+)",
    re.I,
)
_TORCH_DISTS: frozenset[str] = frozenset({"torch", "torchvision", "torchaudio"})


@dataclass(frozen=True)
class EnvIssue:
    """A classified environment fault. ``package`` is the module/library name when
    known (top-level, pip-installable); ``detail`` is the matched evidence line."""

    kind: EnvIssueKind
    package: str = ""
    detail: str = ""


# --------------------------------------------------------------------------- #
# Signature patterns (ordered by precedence in classify_failure)
# --------------------------------------------------------------------------- #
# The whole env / its interpreter is gone — most severe, checked first.
_ENV_ABSENT_RES = (
    re.compile(r"(/[\w./+-]+/bin/(?:python[\d.]*|Rscript))\s*:\s*(?:No such file|not found)", re.I),
    re.compile(r"EnvironmentLocationNotFound", re.I),
    re.compile(r"Could not find conda environment", re.I),
    # smoke run_one SKIP detail ``env <e> missing`` (fed as ``SKIP | env <e> missing``). Anchored to
    # start-of-line OR just after a ``|`` status separator (B-F2) so it matches the harness signal but
    # NOT a mid-sentence ``… env CUDA_HOME missing …`` in arbitrary tool/agent prose (which, matched
    # whole-text and checked FIRST, would have flipped a passing run to a false ENV_ABSENT → RECREATE).
    re.compile(r"(?:^|\|)\s*env\s+[\w.+-]+\s+missing\b", re.I | re.M),
    re.compile(r"\binterpreter missing\b", re.I),  # Tier-1 help-probe detail
)

# R: a library()/requireNamespace() call found no package, across R's several phrasings.
_R_NO_PACKAGE_RE = re.compile(r"there is no package called ['\"]([\w.]+)['\"]", re.I)
_R_LIBRARY_RE = re.compile(r"Error in library\(([\w.]+)\)")
_R_NAMESPACE_LOAD_RE = re.compile(r"package or namespace load failed for ['\"]([\w.]+)['\"]", re.I)
_R_NOT_AVAILABLE_RE = re.compile(r"package ['\"]([\w.]+)['\"] is not available", re.I)
# Anchored to R's own error surface (``Error in loadNamespace("x") :`` / ``Error: requireNamespace('x')
# returned FALSE``), never the bare call (hunt 2026-09-30, u36-setup-install-7): agent code routinely
# carries ``if (!requireNamespace("SPOTlight", quietly = TRUE))``, so a fully successful R run read as
# MISSING_R_PACKAGE, was scored FAIL and had its env destroyed and rebuilt.
_R_LOADNAMESPACE_RE = re.compile(
    r"\bError(?:\s+in|\s*:)\s*(?:loadNamespace|requireNamespace)\(\s*['\"]([\w.]+)['\"]", re.I
)
_R_RES = (_R_NO_PACKAGE_RE, _R_LIBRARY_RE, _R_NAMESPACE_LOAD_RE, _R_NOT_AVAILABLE_RE, _R_LOADNAMESPACE_RE)

# Python: a top-level import is missing. A *genuine* surface is the exception line itself — a
# traceback tail (``ModuleNotFoundError: …``), the Tier-2 REPL's swallowed ``Error: …`` string,
# or a line that STARTS with the phrase — never a mid-line mention inside a warning or the
# agent's narration. ``_find_py_module`` enforces that anchoring per line and drops benign
# optional-dependency lines, so a *passing* run that logs ``optional backend unavailable: No
# module named 'louvain'; falling back to leiden`` is never misread as a fault.
_PY_MODULE_RE = re.compile(
    r"(?:(?:ModuleNotFoundError|ImportError|Error)\s*:\s*|^)\s*No module named\s*['\"]?([A-Za-z_][\w.]*)"
)
_PY_MODULE_BENIGN_RE = re.compile(r"optional|fall(?:ing|s)?\s*back|retr(?:y|ied|ies)|succe(?:ss|eded)", re.I)

# Signatures that ALSO legitimately appear on a *benign* line and so must be scanned line-by-line and
# skipped on a benign marker (mirroring ``_find_py_module``) — a soft event on a PASSING run must not
# trigger a needless RECREATE (nuke + rebuild) of a perfectly healthy env:
#   • ``cannot import name X from Y``  → optional-import fallback  (``_CANNOT_IMPORT_NAME_RE``, skip on
#     ``optional / falling back / retry / success`` via ``_PY_MODULE_BENIGN_RE``);
#   • the three numpy C-API-skew phrases below (``_ENV_BROKEN_WARN_RES``) → numpy emits them as a
#     ``RuntimeWarning`` and keeps running, so skip on a ``…warning…`` line.
# The remaining ``_ENV_BROKEN_RES`` signatures only ever surface on a genuinely fatal line and stay a
# hard whole-text match.
_CANNOT_IMPORT_NAME_RE = re.compile(r"cannot import name ['\"][\w.]+['\"] from ['\"]([\w.]+)['\"]")

# Env present but its import machinery / native ABI is broken → a clean rebuild fixes it. HARD
# signatures only: matched whole-text (below).
_ENV_BROKEN_RES = (
    # (``undefined symbol`` is matched line by line below — benign warnings print it too.)
    # NB (C8): `GLIBCXX_… not found` / `version `GLIBC_…` (a too-old system libstdc++/glibc) were moved
    # OUT of ENV_BROKEN to a targeted `diagnose` system-lib REPAIR lane (_GLIBCXX_RE / _GLIBC_VER_RE
    # below): a blind RECREATE just rebuilds against the SAME too-old host runtime, whereas
    # conda-installing libstdcxx-ng/libgcc-ng ships a newer libstdc++.so.6 that satisfies the symbol.
    # `undefined symbol` (a genuine cross-package ABI break, above) stays here → RECREATE.
    re.compile(r"numpy\.core\.multiarray failed to import", re.I),  # numpy 1↔2 native ABI break (fatal)
    re.compile(r"No module named ['\"]?numpy[._]+core", re.I),  # numpy 2 moved numpy.core→numpy._core (fatal)
    re.compile(r"compiled using NumPy 1\.x cannot be run in NumPy 2", re.I),  # fatal numpy-2 load
    re.compile(r"DLL load failed", re.I),
)

# numpy C-API-skew phrases that numpy prints as a *RuntimeWarning* (execution continues) on the mixed
# conda+pip envs this project builds (B-F1). Scanned line-by-line and skipped on a ``…warning…`` line:
# a genuinely fatal break surfaces on a SEPARATE hard line above (``multiarray failed to import`` /
# ``numpy._core`` / ``1.x cannot be run``), so no fatal detection is lost — but a healthy tool that
# merely logged ``RuntimeWarning: numpy.dtype size changed …`` is no longer nuked + rebuilt (and
# scored FAILED) for nothing.
_ENV_BROKEN_WARN_RES = (
    re.compile(r"numpy\.dtype size changed", re.I),  # classic ABI-skew RuntimeWarning across a partial upgrade
    re.compile(r"compiled against API version", re.I),  # "module compiled against API version … but this numpy …"
    re.compile(r"_ARRAY_API not found", re.I),  # numpy 2 ABI break — but usually printed on a RuntimeWarning line
)
_NUMPY_ABI_WARNING_RE = re.compile(r"warning", re.I)  # RuntimeWarning / UserWarning / warnings.warn line marker

# ``undefined symbol`` is a genuine ABI break on an ``ImportError: …: undefined symbol: …`` line, but it is
# also printed by warnings that let execution continue (hunt 2026-09-30, u36-setup-install-7): torchvision's
# ``UserWarning: Failed to load image Python extension: …image.so: undefined symbol: …`` and PyG's
# ``UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. …``. As a whole-text
# hard match it turned a run that failed for a data reason (its stderr tail carrying the warning) into
# ENV_BROKEN, a forced FAIL and a RECREATE that rebuilt the same pinned wheels — and the same warning. So it
# is matched line by line and a line carrying a warning LABEL is skipped. The label, not the bare word: a
# mangled torch symbol (``_ZN3c107Warning4warnE…``) contains "Warning" and must still count.
_UNDEFINED_SYMBOL_RE = re.compile(r"undefined symbol", re.I)
_WARNING_LABEL_RE = re.compile(r"\b\w*Warning:|\bwarnings\.warn\(", re.I)

# The wizard's own generated MCP config is absent (Tier-2 SKIP reason).
_WIRING_RES = (
    re.compile(r"no generated MCP config", re.I),
    re.compile(r"wiring did not run", re.I),
)

# A runtime "please install X (via `pip install Y`)" hint — the tool is importable-but-absent and
# *names its own pip target*. scanpy's ``highly_variable_genes(flavor="seurat_v3")`` emits the
# canonical case: ``Please install skmisc package via `pip install --user scikit-misc```. Distinct
# from a bare ``No module named`` (handled by the anchored py-module rule): here the tool prints the
# exact remedy, so we install *that* distribution verbatim (dropping ``--user`` so it lands in the
# managed env, not ``~/.local``). Tightly anchored to the "install <mod> … via pip install <dist>"
# idiom so an incidental "you could pip install X" in prose never triggers a mutation.
_PIP_HINT_RE = re.compile(
    r"(?:please\s+)?install\s+(?P<mod>[\w.\-]+)\s+(?:package\s+)?via\s+[`'\"]*"
    r"pip\s+install\s+(?P<flags>(?:--?[A-Za-z][\w-]*\s+)*)(?P<dist>[A-Za-z0-9][\w.\-]+)",
    re.I,
)

# A pip build that failed for lack of a C/C++ compiler / Python dev headers. Build-time and
# unambiguous; the deterministic fix is to add the conda toolchain and finish the pip layer (never
# a source-code change). Checked before the broad py-module rule so a compile failure whose
# downstream symptom is a missing extension module is not misread as an ordinary missing import.
_TOOLCHAIN_RES = (
    re.compile(r"fatal error:\s*Python\.h", re.I),
    re.compile(r"\bPython\.h:\s*No such file", re.I),
    # allow a Debian/Ubuntu arch-vendor prefix on the compiler name (x86_64-linux-gnu-gcc, …);
    # order the alternation longest-first so g++/clang++ win over cc/clang.
    re.compile(r"error:\s*command\s+'?(?:[\w./+-]+-)?(?:g\+\+|clang\+\+|gcc|clang|cc)'?\s+failed", re.I),
    re.compile(r"unable to execute '?(?:[\w./+-]+-)?(?:g\+\+|clang\+\+|gcc|clang|cc)'?:", re.I),
    re.compile(r"\b(?:[\w./+-]+-)?(?:g\+\+|gcc|cc):\s*(?:command\s+not\s+found|not\s+found)", re.I),
)


def _top_level(module: str) -> str:
    """``numba.core`` → ``numba`` — pip installs distributions, not submodules."""
    return module.split(".")[0].strip()


# Import name → PyPI distribution, for the handful where they differ. Used ONLY to choose the
# pip *target* (in ``_repair_py_module``); the in-env re-probe always imports the original
# module name, so ``issue.package`` stays the import name everywhere else.
_IMPORT_TO_DIST = {
    "cv2": "opencv-python",
    "ot": "POT",  # POT imports as ``ot``; unmapped, a repair pip-installed the bare name (u37-setup-checks-2 repair)
    "paste": "paste-bio",  # PyPI ``paste`` is the unrelated WSGI toolkit (hunt 2026-09-30, u38c-specs-b-11)
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "bs4": "beautifulsoup4",
    "yaml": "PyYAML",
    "OpenGL": "PyOpenGL",
    "Bio": "biopython",
    "igraph": "python-igraph",
    "cairo": "pycairo",
    # scanpy's ``highly_variable_genes(flavor="seurat_v3")`` imports ``skmisc`` but its dist is
    # ``scikit-misc`` — so the plain ``No module named 'skmisc'`` form resolves to the right pip
    # target too (the runtime ``Please install skmisc via pip install scikit-misc`` hint is caught
    # by :data:`MISSING_PIP_DIST`, which carries the tool-named dist verbatim).
    "skmisc": "scikit-misc",
    # Two tools' own packages whose PyPI dist differs from the import (hunt 2026-09-30, u38d-specs-c-11):
    # the tangram recipe pins ``tangram-sc`` and the stacker recipe ``antspyx``. Unmapped, a missing-module
    # repair pip-installed the bare import name — a different project, or none at all.
    "tangram": "tangram-sc",
    "ants": "antspyx",
}


def _dist_for(import_name: str) -> str:
    """Map a top-level import name to its PyPI distribution when the two differ
    (``cv2`` → ``opencv-python``); otherwise return it unchanged."""
    return _IMPORT_TO_DIST.get(import_name, import_name)


def _norm_dist(name: str) -> str:
    """PEP 503 name normalization: ``Tangram_SC`` and ``tangram-sc`` are the same distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _recipe_pin_for(spec: ToolSpec, dist: str) -> str:
    """The recipe's own ``dist==version`` pin for ``dist``, or ``""``.

    A missing-module repair installs the version the tool was captured with, not whatever PyPI
    serves today (hunt 2026-09-30, u38d-specs-c-11). Only a plain PyPI pin qualifies: a ``+local``
    version (``torch==2.1.0+cpu``) needs a custom index this one-package install does not pass, so
    it is left to the off-index / RECREATE paths. Never raises."""
    from . import provision  # lazy: envdoctor↔provision decoupled at import

    try:
        pip_reqs, _py = provision.read_recipe_pip_and_python(getattr(spec, "recipe", "") or "")
    except Exception:
        return ""
    want = _norm_dist(dist)
    for raw in pip_reqs:
        req = str(raw).strip()
        if _norm_dist(_req_name(req)) == want and "==" in req and "+" not in req and "://" not in req:
            return req
    return ""


# Missing system shared object → the conda-forge package that provides it. Mirrors
# :data:`_IMPORT_TO_DIST` for the OS layer: a *mapped* ``.so`` is a Lane-1 deterministic repair
# (``conda_install`` from conda-forge into the tool env); an *unmapped* ``.so`` is surfaced
# honestly (``SYSTEM_LIB_NEEDS_ROOT`` — likely needs a distro package the wizard can't install).
# Keyed on the bare soname without its version suffix so ``libGL.so.1`` and ``libGL.so`` both hit.
_SO_TO_CONDA_PKG = {
    "libGL.so": "libgl",  # OpenGL runtime (opencv/vtk/pyvista) — conda-forge ``libgl``/mesa
    "libEGL.so": "libegl",
    "libgthread-2.0.so": "glib",
    "libglib-2.0.so": "glib",
    "libgomp.so": "libgomp",  # OpenMP runtime (scikit-learn / xgboost native)
    "libSM.so": "libsm",
    "libICE.so": "libice",
    "libXrender.so": "libxrender",
    "libXext.so": "libxext",
    "libX11.so": "xorg-libx11",
}


def _soname_root(soname: str) -> str:
    """``libGL.so.1`` / ``libGL.so.1.7.0`` → ``libGL.so`` — the version-stripped key for
    :data:`_SO_TO_CONDA_PKG` (a repair installs the package, never a specific ABI version)."""
    m = re.match(r"(.+?\.so)(?:\.[\d.]+)?$", soname.strip())
    return m.group(1) if m else soname.strip()


def _conda_pkg_for_so(soname: str) -> str:
    """conda-forge package that provides ``soname``, or ``""`` if unmapped (→ Lane 3)."""
    return _SO_TO_CONDA_PKG.get(_soname_root(soname), "")


def _find_py_module(text: str) -> str | None:
    """Return the missing top-level module from a *genuine* import-failure line, or ``None``.

    A line qualifies only when ``No module named 'X'`` is a real exception surface — it
    *starts* the line or is directly preceded by a ``ModuleNotFoundError:`` / ``ImportError:`` /
    ``Error:`` label (the Tier-2 REPL's swallowed form) — and carries no benign marker
    (``optional`` / ``falling back`` / ``retried`` / ``succeeded``). A mid-line mention inside a
    warning or the agent's narration therefore never matches, so a *passing* run's log can't be
    misread as a missing module.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _PY_MODULE_BENIGN_RE.search(line):
            continue
        m = _PY_MODULE_RE.search(line)
        if m:
            return _top_level(m.group(1))
    return None


def _find_pip_hint(text: str) -> tuple[str, str] | None:
    """``(import_name, dist)`` from a runtime "install X via ``pip install Y``" hint, or ``None``.

    Returns both the module the tool tried to import (to re-probe after the install) and the exact
    distribution it named (to install) — e.g. ``("skmisc", "scikit-misc")``. Pip flags (``--user``)
    are dropped so the repair targets the managed env. Skipped if the hint sits on a benign line
    (``optional`` / ``falling back``), so a soft "you could install X for extra features" note on a
    passing run is never read as a fault."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _PY_MODULE_BENIGN_RE.search(line):
            continue
        m = _PIP_HINT_RE.search(line)
        if m:
            dist = m.group("dist").strip().strip("`'\"")
            mod = _top_level(m.group("mod").strip().strip("`'\""))
            if dist:
                return (mod, dist)
    return None


def _req_name(req: str) -> str:
    """The distribution name from a pip requirement pin (``torch==2.1.0+cpu`` → ``torch``).

    Empty for a flag line (starts with ``-``) or a URL/VCS requirement (``git+https://…``,
    ``https://…/x.whl``) — those carry no plain name to key a lookup on."""
    r = req.strip()
    if not r or r.startswith("-") or "://" in r:
        return ""
    return re.split(r"[<>=!~ ;\[]", r, maxsplit=1)[0].strip()


def _find_off_index_dist(text: str) -> str | None:
    """Return the requirement token of a build failure that needs a *custom index* or a
    *git source*, or ``None`` when the failure is not that signature.

    Two build-time surfaces qualify — both a plain ``conda env create`` recipe rebuild
    emits when its pip section pins wheels PyPI does not carry:

    * a PEP 440 *local-version* pin (``torch==2.1.0+cpu``, ``torch-scatter==2.1.2+pt21cpu``)
      — those wheels live only on ``download.pytorch.org`` / ``data.pyg.org``; or
    * a **known non-PyPI git package** (``stagate-pyg==1.0.0``) whose source repo is in
      :data:`KNOWN_GIT_SOURCES`.

    Only these two are claimed. An ordinary ``No matching distribution`` for a typo'd or
    yanked PyPI package returns ``None`` (not our case) and is never auto-"fixed"."""
    for m in _NO_DIST_RE.finditer(text):
        token = m.group(1).strip().strip("'\"()")
        if _LOCAL_VERSION_PIN_RE.match(token):
            return token
        if _req_name(token).lower() in KNOWN_GIT_SOURCES:
            return token
    return None


def classify_failure(text: str, *, spec: ToolSpec | None = None) -> EnvIssue | None:
    """Classify ``text`` as an environment problem, or return ``None``.

    ``spec`` is accepted for context (its ``worker_kind`` biases R vs. Python and its
    ``import_check`` is the tool's own top-level package) but classification never
    *requires* it — the raw error text carries enough signal on its own.
    """
    if not text:
        return None
    is_r = bool(spec and getattr(spec, "worker_kind", "") == "rscript")

    # 1) Whole env / interpreter gone.
    for rx in _ENV_ABSENT_RES:
        m = rx.search(text)
        if m:
            return EnvIssue(EnvIssueKind.ENV_ABSENT, detail=m.group(0)[:200])

    # 1b) A recipe rebuild that needs an off-PyPI wheel (a ``+local`` pin) or a known non-PyPI
    #     git package — ``conda env create`` forwards the pip pins verbatim and dies "No matching
    #     distribution found" until the right custom index / git source is supplied. Checked early:
    #     the signature is build-time-specific and disjoint from the runtime import rules below.
    off = _find_off_index_dist(text)
    if off:
        return EnvIssue(EnvIssueKind.OFF_INDEX_WHEEL, package=off, detail=f"cannot resolve {off} from PyPI")

    # 2) R package missing (check before the Python module rule; the signatures are
    #    disjoint, but an R worker's traceback should never be read as a py import).
    for rx in _R_RES:
        m = rx.search(text)
        if m:
            return EnvIssue(EnvIssueKind.MISSING_R_PACKAGE, package=m.group(1), detail=m.group(0)[:200])

    # 3) Env present but its import machinery / native ABI is broken → a clean rebuild fixes it.
    #    Checked BEFORE the broad "No module named" rule so a numpy 1↔2 ABI skew — which surfaces
    #    as BOTH "numpy.core.multiarray failed to import" AND "No module named 'numpy._core'" —
    #    routes to RECREATE, not a futile `pip install numpy` that leaves the ABI mismatched.
    for rx in _ENV_BROKEN_RES:
        m = rx.search(text)
        if m:
            pkg = m.group(1) if rx.groups else ""
            return EnvIssue(EnvIssueKind.ENV_BROKEN, package=_top_level(pkg) if pkg else "", detail=m.group(0)[:200])

    # 3-sym) ``undefined symbol`` on a line that is not a warning (see _UNDEFINED_SYMBOL_RE).
    for raw in text.splitlines():
        line = raw.strip()
        if line and not _WARNING_LABEL_RE.search(line):
            m = _UNDEFINED_SYMBOL_RE.search(line)
            if m:
                return EnvIssue(EnvIssueKind.ENV_BROKEN, package="", detail=m.group(0)[:200])

    # 3-warn) The numpy C-API-skew signatures that ALSO print as a benign RuntimeWarning (B-F1): scan
    #    line-by-line and skip a ``…warning…`` line (mirrors 3a). A PASSING run that merely logged
    #    ``RuntimeWarning: numpy.dtype size changed …`` must not be nuked + rebuilt; a genuinely fatal
    #    break is already caught whole-text by the HARD signatures above.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _NUMPY_ABI_WARNING_RE.search(line):
            continue
        for rx in _ENV_BROKEN_WARN_RES:
            m = rx.search(line)
            if m:
                return EnvIssue(EnvIssueKind.ENV_BROKEN, package="", detail=m.group(0)[:200])

    # 3a) ``cannot import name X from Y`` is an ENV_BROKEN signature too, but — unlike the hard ABI
    #     signatures above — it ALSO appears inside benign optional-import fallbacks, so it is scanned
    #     line-by-line here and skipped on a benign marker (mirrors ``_find_py_module``). A real
    #     ``ImportError: cannot import name …`` on its own line still routes to ENV_BROKEN; an
    #     ``optional: cannot import name … ; falling back`` on a passing run no longer forces a RECREATE.
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _PY_MODULE_BENIGN_RE.search(line):
            continue
        m = _CANNOT_IMPORT_NAME_RE.search(line)
        if m:
            return EnvIssue(EnvIssueKind.ENV_BROKEN, package=_top_level(m.group(1)), detail=m.group(0)[:200])

    # 3b) A pip build failed for lack of a C/C++ compiler → add the conda toolchain and finish the
    #     pip layer (python workers only; an R build failure is left to recreate/planner, not this
    #     pip-oriented repair). Before the broad module rule: a compile failure whose symptom is a
    #     missing extension module must route to the toolchain fix, not a futile `pip install`.
    if not is_r:
        for rx in _TOOLCHAIN_RES:
            m = rx.search(text)
            if m:
                return EnvIssue(EnvIssueKind.BUILD_TOOLCHAIN, detail=m.group(0)[:200])

    # 3c) The tool named its OWN pip target ("install X via `pip install Y`") — install that exact
    #     distribution. ``package`` is the dist to install, ``detail`` the import name to re-probe
    #     (the two differ for scikit-misc → skmisc). Python workers only.
    if not is_r:
        hint = _find_pip_hint(text)
        if hint:
            mod, dist = hint
            return EnvIssue(EnvIssueKind.MISSING_PIP_DIST, package=dist, detail=mod)

    # 4) Python module missing — but only from a genuine, anchored exception line (not a benign
    #    optional-dep log). For an R worker a stray "No module named" is a broken env instead.
    py_pkg = _find_py_module(text)
    if py_pkg:
        if is_r:
            return EnvIssue(EnvIssueKind.ENV_BROKEN, detail=f"No module named '{py_pkg}'")
        return EnvIssue(EnvIssueKind.MISSING_PY_MODULE, package=py_pkg, detail=f"No module named '{py_pkg}'")

    # 5) Generated wiring absent.
    for rx in _WIRING_RES:
        m = rx.search(text)
        if m:
            return EnvIssue(EnvIssueKind.WIRING_MISSING, detail=m.group(0)[:200])

    # Not an environment problem (LLM / auth / rate-limit / task / timeout / unknown).
    return None


# --------------------------------------------------------------------------- #
# Phase 2 — repair a classified EnvIssue with the existing guarded primitives
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RepairResult:
    """Outcome of one :func:`repair_env` dispatch.

    ``repaired`` is the *only* signal the caller may use to flip a FAIL→(retry):
    it is True only when a real mutation happened AND a cheap in-env probe confirmed
    it. ``dry_run`` marks a simulated repair (no mutation, never ``repaired``).
    """

    kind: str  # the EnvIssueKind acted on
    action: str  # "pip_install" | "recreate" | "wire" | "none"
    attempted: bool = False
    repaired: bool = False
    dry_run: bool = False
    package: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "action": self.action,
            "attempted": self.attempted,
            "repaired": self.repaired,
            "dry_run": self.dry_run,
            "package": self.package,
            "detail": self.detail,
        }


def _log(log: SessionLog | None, kind: str, **fields) -> None:
    if log is not None:
        log.event(kind, **fields)


def _probe_import(conda: Conda, spec: ToolSpec, target_env: str, module: str) -> bool:
    """Cheap in-env confirmation that ``module`` now imports. Read-only (``conda.run``
    is never dry-run-gated), so it reflects the env's *real* state after a repair."""
    argv = ["python", "-c", f"import {module}"]
    if getattr(spec, "worker_kind", "") == "rscript":
        argv = ["Rscript", "-e", f"library({module})"]
    res = conda.run(target_env, argv, timeout=constants.CONDA_RUN_TIMEOUT_SEC, check=False)
    return bool(res.ok)


def repair_env(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None = None,
) -> RepairResult:
    """Repair a classified :class:`EnvIssue` in the ``<basic>_<server>`` env.

    Dispatch:

    * ``MISSING_PY_MODULE`` → ``conda.pip_install`` the package, re-probe the import;
      on failure escalate to a guarded RECREATE from the spec recipe.
    * ``MISSING_R_PACKAGE`` → for an R worker, ``conda.r_install`` that one library first; else (or if
      that did not fix it) guarded RECREATE, then re-probe the *specific* R library (a transitive dep
      that ``provision_one``'s ``import_check`` confirm would miss).
    * ``ENV_BROKEN`` / ``ENV_ABSENT`` → guarded RECREATE/build via ``provision.provision_one``
      (which re-classifies + confirms health itself).
    * ``WIRING_MISSING`` → reported only (regenerating the whole config needs the full
      provisioned-server set, which Tier-1 does not hold; Tier-2 owns re-wiring).

    Env-only and namespace-guarded by construction; never raises. A dry-run ``conda``
    short-circuits with ``repaired=False`` so no verdict is ever flipped on a --dry-run.
    """
    kind = issue.kind
    target = spec.target_env(basic_env)

    # A dry-run wizard must never mutate an env or flip a verdict — report intent only.
    if getattr(conda, "dry_run", False):
        _log(log, "env_repair", server=spec.server_key, env=target, issue_kind=str(kind), dry_run=True)
        return RepairResult(
            kind=str(kind),
            action="(dry-run)",
            attempted=True,
            repaired=False,
            dry_run=True,
            package=issue.package,
            detail=f"dry-run: would repair {kind}",
        )

    # Dispatch is wrapped so the module's "never raises" contract holds even if a guarded
    # primitive raises (an in-env probe hitting OSError/timeout, a pip erroring under check=True
    # somewhere): a crashing repair degrades to "not repaired", it never takes down the phase.
    try:
        if kind is EnvIssueKind.MISSING_PY_MODULE:
            return _repair_py_module(conda, spec, basic_env, target, issue, io=io, log=log)
        if kind is EnvIssueKind.MISSING_PIP_DIST:
            return _repair_pip_dist(conda, spec, basic_env, target, issue, io=io, log=log)
        if kind is EnvIssueKind.BUILD_TOOLCHAIN:
            return _repair_toolchain(conda, spec, basic_env, target, issue, io=io, log=log)
        if kind is EnvIssueKind.MISSING_R_PACKAGE:
            return _repair_r_package(conda, spec, basic_env, issue, io=io, log=log)
        if kind in (EnvIssueKind.ENV_BROKEN, EnvIssueKind.ENV_ABSENT):
            return _repair_via_recreate(conda, spec, basic_env, issue, io=io, log=log)
        if kind is EnvIssueKind.OFF_INDEX_WHEEL:
            return _repair_off_index_wheel(conda, spec, basic_env, target, issue, io=io, log=log)
        if kind is EnvIssueKind.WIRING_MISSING:
            return RepairResult(
                kind=str(kind),
                action="wire",
                attempted=False,
                repaired=False,
                detail="wiring is regenerated by the Tier-2 wiring step, not per-server repair",
            )
        return RepairResult(kind=str(kind), action="none", attempted=False)
    except Exception as exc:  # the module's contract: repair_env NEVER raises
        _log(log, "env_repair", server=spec.server_key, env=target, action="error", error=str(exc)[:200])
        return RepairResult(
            kind=str(kind),
            action="error",
            attempted=True,
            repaired=False,
            package=issue.package,
            detail=f"repair crashed: {exc}"[:200],
        )


def _repair_py_module(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """Fast path for a missing Python import: pip install the *distribution* into the tool
    env, confirm the import, and escalate to a full guarded RECREATE if pip did not resolve
    it. Never raises — a pip that itself errors (bad dist name, network) escalates too."""
    # Defense-in-depth (mirrors _repair_toolchain / _repair_off_index_wheel / _repair_system_lib):
    # the pip_install below MUTATES `target`, so refuse a non-`<basic>_*` / PROTECTED env before
    # touching it — never mutate an env we don't own. (target is structurally in-namespace today, so
    # this only fires on a pathological basic/server-key collision; the guard makes it can't-happen.)
    try:
        constants.assert_deletable_env(basic_env, target)
    except PermissionError as exc:
        return RepairResult(
            kind=str(issue.kind), action="refused", attempted=False, repaired=False, detail=str(exc)[:200]
        )
    import_name = issue.package or spec.import_check or spec.server_key
    dist = _dist_for(import_name)  # bug C: cv2 → opencv-python, etc.; identity when they match
    dist = _recipe_pin_for(spec, dist) or dist  # the recipe's own pin when it lists this dist
    shown = dist if dist == import_name else f"{dist} (for import '{import_name}')"
    io.say(f"  envdoctor: {target} is missing '{import_name}' — trying pip install {shown}…")
    try:
        # bug 5 — check=False: a genuine pip failure must RETURN ok=False so the RECREATE
        # escalation below is reachable, not raise CondaError (which made it dead code).
        res = conda.pip_install(target, [dist], check=False)
        pip_ran_clean = bool(res.ok)
    except Exception as exc:  # OSError / timeout / a CondaError from some other layer
        io.warn(f"{target}: pip install '{dist}' errored ({exc}) — recreating env from recipe…")
        pip_ran_clean = False

    if pip_ran_clean and _probe_import(conda, spec, target, import_name):
        io.ok(f"{target}: installed '{dist}' (envdoctor)")
        _log(
            log,
            "env_repair",
            server=spec.server_key,
            env=target,
            action="pip_install",
            package=import_name,
            repaired=True,
        )
        return RepairResult(
            kind=str(issue.kind),
            action="pip_install",
            attempted=True,
            repaired=True,
            package=import_name,
            detail=f"pip install '{dist}' + import confirmed",
        )

    # pip could not resolve it (wrong distribution name, or a deeper break) → RECREATE.
    if pip_ran_clean:  # only note "did not resolve" when pip itself exited 0
        io.warn(f"{target}: pip install '{dist}' did not resolve '{import_name}' — recreating env from recipe…")
    rec = _repair_via_recreate(conda, spec, basic_env, issue, io=io, log=log)

    # RECREATE honesty (bug 2): provision_one keys env health off spec.import_check. A missing
    # *transitive* dep (import_name != import_check) is NOT covered by that confirmation, so
    # re-probe the specific import before trusting the repair.
    repaired = rec.repaired
    detail = f"pip did not fix → {rec.detail}"
    if repaired and import_name and import_name != (spec.import_check or ""):
        try:
            import_ok = _probe_import(conda, spec, target, import_name)
        except Exception:
            # _repair_via_recreate already did a destructive remove_env + rebuild. A terminal probe that
            # itself raises (conda.run maps a loaded-box TimeoutExpired/OSError → CondaError even at
            # check=False) must NOT unwind to repair_env's catch-all, which relabels this to action="error"
            # — a spelling installer_scientist._action_recreates does NOT count, so the finished recreate
            # would escape the MAX_ENV_RECREATES tally and remediation could nuke+rebuild again. Read an
            # unverifiable import as FAILED, KEEPING the honest pip_install→recreate action (still counted).
            # (R33 — mirrors R28's _repair_toolchain / _repair_off_index_wheel terminal-probe guards.)
            import_ok = False
        if not import_ok:
            repaired = False
            detail = f"RECREATE rebuilt {target} but '{import_name}' still does not import"

    return RepairResult(
        kind=str(issue.kind),
        action=f"pip_install→{rec.action}",
        attempted=True,
        repaired=repaired,
        dry_run=rec.dry_run,
        package=import_name,
        detail=detail,
    )


def _repair_pip_dist(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """Install the exact distribution a tool named for itself ("install X via ``pip install Y``").

    ``issue.package`` is the dist to install (``scikit-misc``); ``issue.detail`` is the import to
    re-probe (``skmisc``) — the two differ here, which is why the tool's own naming beats a guess.
    ONE guarded pip install, then confirm the import. No RECREATE escalation: the recipe does not
    carry this dist, so a rebuild would not add it; an unresolved install surfaces honestly."""
    # Defense-in-depth: the pip_install below MUTATES `target` — refuse a non-`<basic>_*` / PROTECTED
    # env before touching it (mirrors the guarded destructive repairs).
    try:
        constants.assert_deletable_env(basic_env, target)
    except PermissionError as exc:
        return RepairResult(
            kind=str(issue.kind), action="refused", attempted=False, repaired=False, detail=str(exc)[:200]
        )
    dist = issue.package or _dist_for(issue.detail or spec.import_check or spec.server_key)
    import_name = issue.detail or spec.import_check or spec.server_key
    io.say(f"  envdoctor: {target} needs '{dist}' (named by the tool) — pip installing…")
    try:
        res = conda.pip_install(target, [dist], check=False)  # tool-provided flags (--user) dropped
        pip_ok = bool(res.ok)
    except Exception as exc:  # OSError / timeout / a CondaError from a lower layer
        io.warn(f"{target}: pip install '{dist}' errored ({exc})")
        pip_ok = False
    repaired = bool(pip_ok and import_name and _probe_import(conda, spec, target, import_name))
    if repaired:
        io.ok(f"{target}: installed '{dist}' (envdoctor)")
    _log(log, "env_repair", server=spec.server_key, env=target, action="pip_dist", package=dist, repaired=repaired)
    return RepairResult(
        kind=str(issue.kind),
        action="pip_dist",
        attempted=True,
        repaired=repaired,
        package=dist,
        detail=f"pip install '{dist}' + import '{import_name}' {'confirmed' if repaired else 'still fails'}",
    )


def _repair_toolchain(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """A pip build failed for lack of a compiler. Rebuild the env from its recipe (so every conda +
    pip dep is honored), add the conda-forge C/C++ toolchain on top, re-run the recipe's pip stack
    with a compiler now present, and confirm the import. Namespace-guarded; never raises out.

    Deliberately uses ``create_from_yaml`` (not a bare ``create_named``) so the recipe's *conda*
    layer is not dropped — the failure is only the pip build, and the fix must not throw away the
    tool's real dependencies. Every channel it adds is on :data:`TRUSTED_CONDA_CHANNELS`."""
    from . import provision  # lazy: envdoctor↔provision decoupled at import

    recipe = getattr(spec, "recipe", "") or ""
    if not recipe:
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=False,
            repaired=False,
            detail="no recipe to rebuild for a toolchain repair",
        )
    channels = ["conda-forge"]
    for chan in channels:  # defensive: the literal is on the list; enforce the discipline anyway
        if chan not in TRUSTED_CONDA_CHANNELS:
            return RepairResult(
                kind=str(issue.kind),
                action="toolchain",
                attempted=False,
                repaired=False,
                detail=f"refusing untrusted conda channel '{chan}'",
            )
    constants.assert_deletable_env(basic_env, target)
    # Materialize the recipe BEFORE anything is removed (hunt 2026-09-30, u36-setup-install-5). The
    # spec carries a repo-relative path, and handing it to `conda env create` straight resolved it
    # against whatever directory sog-setup was started from: outside the repo root (the pip-wheel /
    # instance-root deploy) conda answered EnvironmentFileNotFound AFTER the env had been removed.
    # materialize_recipe resolves through constants.resolve_recipe_path and applies the same GPU / foreign-
    # platform portability the main build does, so the rebuild is the recipe build() would run.
    try:
        staged = provision.materialize_recipe(recipe, target)
    except Exception as exc:
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=False,
            repaired=False,
            detail=redact(f"recipe could not be materialized, env left as it was: {exc}")[:200],
        )
    pip_reqs, _py = provision.read_recipe_pip_and_python(str(staged))
    io.say(f"  envdoctor: {target} build needs a C/C++ toolchain — rebuilding with compilers…")

    if conda.env_exists(target):
        conda.remove_env(target)
    # Recreate the FULL conda layer from the recipe (check=False: the pip layer may fail again for
    # the same compiler reason; we finish it manually below once the toolchain is in place).
    try:
        conda.create_from_yaml(str(staged), name=target, check=False)
    except Exception as exc:  # a lower layer may still raise despite check=False
        _log(log, "env_repair", server=spec.server_key, env=target, action="toolchain", error=str(exc)[:200])
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=True,
            repaired=False,
            detail=f"recipe rebuild failed: {exc}"[:200],
        )
    if not conda.env_exists(target, refresh=True):
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=True,
            repaired=False,
            detail="recipe rebuild did not create the env",
        )
    try:
        ires = conda.conda_install(target, ["c-compiler", "cxx-compiler"], channels=channels, check=False)
    except Exception as exc:
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=True,
            repaired=False,
            detail=f"toolchain install failed: {exc}"[:200],
        )
    if not (ires.ok or ires.dry_run):
        return RepairResult(
            kind=str(issue.kind),
            action="toolchain",
            attempted=True,
            repaired=False,
            detail=f"toolchain install rc={ires.returncode}",
        )
    pip_ok = True
    if pip_reqs:
        try:
            pres = conda.pip_install(
                target, _pip_argv(pip_reqs), timeout=constants.PIP_INSTALL_TIMEOUT_SEC, check=False
            )
        except Exception as exc:
            return RepairResult(
                kind=str(issue.kind),
                action="toolchain",
                attempted=True,
                repaired=False,
                detail=f"pip layer failed after toolchain add: {exc}"[:200],
            )
        # The pip result decides too (hunt 2026-09-30, u36-setup-install-6): it was discarded, so for a
        # tool whose health probe cannot see its pip layer (13 probe `__future__`) a pip stack that never
        # installed was reported "finished pip + import confirmed".
        pip_ok = bool(pres.ok or pres.dry_run)
    probe = spec.import_check or spec.server_key
    try:
        import_ok = _probe_import(conda, spec, target, probe) if probe else True
    except Exception:
        # The destructive remove_env + recipe rebuild already completed above. A terminal probe that
        # itself raises (conda.run maps a TimeoutExpired/OSError → CondaError even at check=False on a
        # loaded box) must NOT let repair_env's catch-all relabel this completed recreate as
        # action="error" — that spelling slips past installer_scientist._action_recreates (which counts
        # only "recreate"/"toolchain"/"off_index"), so the finished rebuild would escape the
        # MAX_ENV_RECREATES tally and remediation could nuke+rebuild again next turn. Read an
        # unverifiable import as FAILED, preserving the honest "toolchain" action + its recreate count.
        import_ok = False
    repaired = bool(pip_ok and import_ok)
    _log(log, "env_repair", server=spec.server_key, env=target, action="toolchain", repaired=repaired)
    if repaired:
        detail = "added c/cxx compiler + finished pip + import confirmed"
    elif not pip_ok:
        detail = "toolchain added but the recipe's pip layer still fails to install"
    else:
        detail = f"toolchain added but import '{probe}' still fails"
    return RepairResult(
        kind=str(issue.kind),
        action="toolchain",
        attempted=True,
        repaired=repaired,
        detail=detail,
    )


def _pip_argv(pip_reqs: list[str]) -> list[str]:
    """A recipe's ``pip:`` items as ``pip install`` argv (hunt 2026-09-30, u36-setup-install-6).

    conda writes those items into a requirements file, where ``--extra-index-url https://…`` is one
    valid line; passed to ``pip install`` as ONE argv element pip answers ``no such option:
    --extra-index-url https://…`` (rc 2) and installs nothing. A flag line is split into its words;
    a requirement stays one element (``pkg; python_version < "3.9"`` must not be split)."""
    argv: list[str] = []
    for raw in pip_reqs:
        item = str(raw).strip()
        if not item:
            continue
        if item.startswith("-"):
            try:
                argv.extend(shlex.split(item))
            except ValueError:
                argv.extend(item.split())
        else:
            argv.append(item)
    return argv


def _repair_via_recreate(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """Rebuild the tool env from its recipe via ``provision.provision_one``. That call
    owns the namespace guard (``assert_deletable_env`` before any removal) and confirms
    health by re-classification, so a returned ``ok`` means the env imports cleanly."""
    from . import provision  # lazy: envdoctor↔provision stay decoupled at import time

    target = spec.target_env(basic_env)
    io.say(f"  envdoctor: recreating {target} from recipe ({issue.kind})…")

    def _confirm(proposal) -> bool:
        # Human-in-the-loop interactively; auto-yes on the scripted/non-interactive path.
        return io.ask_yesno(f"Repair {target}: {proposal.action} — {proposal.detail}?", default=True)

    try:
        # ``_allow_repairs=False`` (a94#1): this rebuild is itself the repair, so a failure here must
        # NOT re-enter the remediation loop (that path is what called us). It becomes a plain build;
        # its failure surfaces via ``result.ok`` below and the requesting InstallerScientist stops.
        result = provision.provision_one(conda, spec, basic_env, confirm=_confirm, io=io, log=log, _allow_repairs=False)
    except Exception as exc:  # a repair must never crash the test phase
        _log(log, "env_repair", server=spec.server_key, env=target, action="recreate", error=str(exc)[:200])
        return RepairResult(
            kind=str(issue.kind),
            action="recreate",
            attempted=True,
            repaired=False,
            package=issue.package,
            detail=f"recreate raised: {exc}"[:200],
        )
    repaired = bool(result.ok and not result.skipped)
    _log(log, "env_repair", server=spec.server_key, env=target, action="recreate", repaired=repaired)
    return RepairResult(
        kind=str(issue.kind),
        action="recreate",
        attempted=True,
        repaired=repaired,
        package=issue.package,
        detail="provision_one rebuilt + confirmed" if repaired else "provision_one could not rebuild",
    )


def _repair_r_package(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """RECREATE for a missing R library, then re-probe the *specific* package.

    ``_repair_via_recreate`` trusts ``provision_one``, which keys env health off
    ``spec.import_check`` — the tool's own top-level library. A missing *transitive* R dep
    (``issue.package`` != ``import_check`` — e.g. a ``SpatialExperiment`` that a newer
    Bioconductor pin dropped) is NOT covered by that confirmation, so a rebuild that imports
    the tool cleanly but still lacks the named library would be reported ``repaired=True`` —
    flipping the tool FAIL→retry straight into an identical ``no package called '<pkg>'``
    re-failure. Mirror ``_repair_py_module``'s RECREATE-honesty re-probe (bug 2) for R:
    recreate, then confirm the classified package actually loads before trusting the repair.

    The targeted install comes first (hunt 2026-09-30, u36-setup-install-14). A package the recipe
    lacks — a transitive Bioconductor dependency, or a library installed out of band in the source
    env — is not added by rebuilding from that same recipe, so the RECREATE alone spent the
    destructive budget on an env that still failed. ``Conda.r_install`` (BiocManager, which also
    serves CRAN) installs just that package into the managed env; only if the package and the tool
    still do not load does the RECREATE run.
    """
    target = spec.target_env(basic_env)
    pkg = issue.package
    if pkg and getattr(spec, "worker_kind", "") == "rscript":
        try:
            constants.assert_deletable_env(basic_env, target)  # only ever mutate a managed <basic>_* env
            if conda.env_exists(target):
                io.say(f"  envdoctor: {target} is missing R library '{pkg}' — installing it (BiocManager)…")
                ires = conda.r_install(target, [pkg], check=False)
                tool = spec.import_check or ""
                if (ires.ok or ires.dry_run) and _probe_import(conda, spec, target, pkg):
                    if not tool or tool == pkg or _probe_import(conda, spec, target, tool):
                        io.ok(f"{target}: installed R library '{pkg}' (envdoctor)")
                        _log(
                            log,
                            "env_repair",
                            server=spec.server_key,
                            env=target,
                            action="r_install",
                            package=pkg,
                            repaired=True,
                        )
                        return RepairResult(
                            kind=str(issue.kind),
                            action="r_install",
                            attempted=True,
                            repaired=True,
                            package=pkg,
                            detail=f"BiocManager installed '{pkg}' + library() confirmed",
                        )
                io.warn(f"{target}: installing R library '{pkg}' did not fix it — recreating env from recipe…")
        except PermissionError as exc:
            return RepairResult(
                kind=str(issue.kind), action="refused", attempted=False, repaired=False, detail=str(exc)[:200]
            )
        except Exception as exc:  # an unsafe name (CondaError), a timeout, a probe crash → the RECREATE
            io.warn(f"{target}: R install of '{pkg}' errored ({type(exc).__name__}) — recreating env from recipe…")
    rec = _repair_via_recreate(conda, spec, basic_env, issue, io=io, log=log)
    pkg_missing = False
    if rec.repaired and pkg and pkg != (spec.import_check or ""):
        try:
            pkg_missing = not _probe_import(conda, spec, target, pkg)
        except Exception:
            # Same as _repair_py_module: the destructive recreate already ran, so a raising terminal probe
            # must read as "unverifiable → FAILED" while KEEPING the "recreate" action counted — never let
            # repair_env relabel it "error" and slip the finished recreate past MAX_ENV_RECREATES.
            # (R33 — mirrors R28's terminal-probe guards.)
            pkg_missing = True
    if pkg_missing:
        _log(log, "env_repair", server=spec.server_key, env=target, action="recreate", repaired=False, package=pkg)
        return replace(
            rec,
            repaired=False,
            detail=f"RECREATE rebuilt {target} but R library '{pkg}' still does not load",
        )
    return rec


# --------------------------------------------------------------------------- #
# OFF_INDEX_WHEEL repair — rebuild a recipe whose pip stack needs a custom index
# (PyTorch / PyTorch-Geometric wheels) or a non-PyPI git package. Deterministic;
# every fetched host is checked against the trusted allowlists above.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _OffIndexPlan:
    """A rebuildable ``pip install`` derived from a recipe's verbatim pins.

    ``pip_args`` is the full argv (trusted index flags first, then the pinned packages with
    any known git package swapped for its ``git+https`` URL). ``index_urls`` / ``git_urls``
    are every source it will fetch from — surfaced for the host-allowlist check and the audit
    log; ``torch`` is the ``<ver>+<compute>`` label (for the log/summary), empty if git-only.
    """

    pip_args: list[str]
    index_urls: list[str]
    git_urls: list[str]
    torch: str


def _host_of(url: str) -> str:
    """The network host of an index / find-links / ``git+`` URL (``git+https://github.com/…``
    → ``github.com``); empty for anything that is not an ``…://host/…`` URL."""
    m = re.match(r"\s*(?:git\+)?[a-z][a-z0-9+.-]*://([^/]+)", url, re.I)
    return m.group(1).lower() if m else ""


def _derive_off_index_plan(pip_reqs: list[str]) -> _OffIndexPlan | None:
    """Turn a recipe's verbatim pip pins into a rebuildable install, or ``None`` if none can
    be safely derived (no off-index/git pin to fix, an un-keyable PyG stack, or an off-allowlist
    host).

    The torch ``+local`` pin supplies both custom indexes: the compute variant (``cpu`` /
    ``cuNNN``) keys ``download.pytorch.org/whl/<compute>`` (torch itself) and, together with the
    torch version, ``data.pyg.org/whl/torch-<ver>+<compute>.html`` (the PyTorch-Geometric family
    wheels). Known non-PyPI packages are swapped for their verified ``git+https`` origin. Every
    other pin is forwarded verbatim (they resolve from PyPI as usual). Recipe-embedded pip flag
    lines are dropped — this repair supplies only its own trusted index flags, never an unvetted
    ``--index-url`` from a recipe.
    """
    torch_ver = compute = ""
    has_local = needs_pyg = False
    git_urls: list[str] = []
    reqs_out: list[str] = []
    for raw in pip_reqs:
        req = raw.strip()
        if not req or req.startswith("-"):
            continue  # drop flag lines; we supply our own trusted index flags
        if _req_name(req).lower() in KNOWN_GIT_SOURCES:
            git = KNOWN_GIT_SOURCES[_req_name(req).lower()]
            git_urls.append(git)
            reqs_out.append(git)
            continue
        m = _LOCAL_VERSION_PIN_RE.match(req)
        if m:
            has_local = True
            if m.group(1).lower() == "torch":
                torch_ver, compute = m.group(2), m.group(3)
            else:
                needs_pyg = True  # a torch-geometric-family +local wheel
        reqs_out.append(req)

    if not has_local and not git_urls:
        return None  # nothing off-index and no git package → not our signature

    index_flags: list[str] = []
    index_urls: list[str] = []
    if torch_ver and compute:
        torch_url = f"https://download.pytorch.org/whl/{compute}"
        pyg_url = f"https://data.pyg.org/whl/torch-{torch_ver}+{compute}.html"
        index_flags += ["--extra-index-url", torch_url, "--find-links", pyg_url]
        index_urls += [torch_url, pyg_url]
    elif needs_pyg:
        # PyG ``+local`` wheels but no torch pin to key the find-links URL — can't build it safely.
        return None

    for url in index_urls:
        if _host_of(url) not in TRUSTED_INDEX_HOSTS:
            return None
    for git in git_urls:
        if _host_of(git) not in TRUSTED_GIT_HOSTS:
            return None

    return _OffIndexPlan(
        pip_args=[*index_flags, *reqs_out],
        index_urls=index_urls,
        git_urls=git_urls,
        torch=f"{torch_ver}+{compute}" if torch_ver else "",
    )


def _repair_off_index_wheel(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    issue: EnvIssue,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """Rebuild ``<basic>_<server>`` for a recipe whose pip stack needs a custom index / git
    source. Read the recipe's own pins + python; recreate a minimal env at that python; then
    ONE guarded ``pip_install`` of the full stack with the derived ``--extra-index-url`` (torch)
    + ``--find-links`` (PyG family) + ``git+https`` swap. Every index/git host was checked
    against the trusted allowlists in :func:`_derive_off_index_plan`; an off-list host aborts
    the repair (``repaired=False``) before any fetch. Namespace-guarded, never raises out."""
    from . import provision  # lazy: envdoctor↔provision stay decoupled at import time

    pip_reqs, py_ver = provision.read_recipe_pip_and_python(getattr(spec, "recipe", "") or "")
    if not pip_reqs:
        _log(log, "env_repair", server=spec.server_key, env=target, action="off_index", repaired=False)
        return RepairResult(
            kind=str(issue.kind),
            action="off_index",
            attempted=False,
            repaired=False,
            package=issue.package,
            detail="recipe has no pip section to rebuild",
        )
    plan = _derive_off_index_plan(pip_reqs)
    if plan is None:
        return RepairResult(
            kind=str(issue.kind),
            action="off_index",
            attempted=False,
            repaired=False,
            package=issue.package,
            detail="could not derive a trusted-index install from the recipe pins",
        )

    # Namespace guard — only ever recreate a <basic>_* env, never a protected/foreign one.
    constants.assert_deletable_env(basic_env, target)
    sources = ", ".join(plan.index_urls + plan.git_urls) or "recipe pins"
    io.say(f"  envdoctor: rebuilding {target} from trusted sources ({sources})…")

    if conda.env_exists(target):
        conda.remove_env(target)
    try:
        cres = conda.create_named(target, python=py_ver or "3.11")
    except Exception as exc:  # create_named raises CondaError on a non-zero exit
        _log(log, "env_repair", server=spec.server_key, env=target, action="off_index", error=str(exc)[:200])
        return RepairResult(
            kind=str(issue.kind),
            action="off_index",
            attempted=True,
            repaired=False,
            package=issue.package,
            detail=redact(f"env create failed: {exc}")[:200],  # redact-before-clip: detail is persisted (C6)
        )
    if not (cres.ok or cres.dry_run):
        return RepairResult(
            kind=str(issue.kind),
            action="off_index",
            attempted=True,
            repaired=False,
            package=issue.package,
            detail=f"env create failed: {redact(cres.stderr or '')[:160]}",  # redact raw stderr before clip (C6)
        )

    try:
        pres = conda.pip_install(target, plan.pip_args, timeout=constants.PIP_INSTALL_TIMEOUT_SEC, check=False)
    except Exception as exc:  # OSError / timeout / a CondaError from a lower layer
        _log(log, "env_repair", server=spec.server_key, env=target, action="off_index", error=str(exc)[:200])
        return RepairResult(
            kind=str(issue.kind),
            action="off_index",
            attempted=True,
            repaired=False,
            package=issue.package,
            detail=f"pip install crashed: {exc}"[:200],
        )

    probe = spec.import_check or spec.server_key
    try:
        import_ok = _probe_import(conda, spec, target, probe) if probe else bool(pres.ok)
    except Exception:
        # Same as the toolchain rebuild: the destructive recreate already happened, so a raising terminal
        # probe must not relabel it action="error" and slip the finished recreate past the
        # MAX_ENV_RECREATES tally (installer_scientist._action_recreates). Unverifiable import → FAILED.
        import_ok = False
    repaired = bool(pres.ok and import_ok)
    _log(
        log,
        "env_repair",
        server=spec.server_key,
        env=target,
        action="off_index",
        repaired=repaired,
        torch=plan.torch,
        indexes=plan.index_urls,
        git=plan.git_urls,
    )
    detail = (
        f"rebuilt via custom index{'+git' if plan.git_urls else ''} ({plan.torch or 'git pins'}) + import confirmed"
        if repaired
        else f"pip rc={pres.returncode}; import '{probe}' {'ok' if import_ok else 'failed'}"
    )
    return RepairResult(
        kind=str(issue.kind),
        action="off_index",
        attempted=True,
        repaired=repaired,
        package=issue.package,
        detail=detail,
    )


# --------------------------------------------------------------------------- #
# diagnose — the ADVISORY layer the installer-scientist consults AFTER
# classify_failure returns None. It NEVER changes classify_failure's verdict (so
# every negative pin — a bare ``.so``, a CUDA runtime error, a plain-PyPI miss —
# stays None there), and it splits the "not a plain repairable env problem" space:
#
#   • Lane 1 REPAIR  — a *mapped* missing system ``.so`` → ``conda_install``. The one
#     deterministic repair kept out of classify_failure to honor its bare-``.so``
#     negative pin; dispatched via :func:`repair_diagnosis`.
#   • Lane 3 SURFACE — a specific, actionable message, NO mutation, STOP the loop:
#     missing external token, a special input the mini data lacks, an input-shape
#     mismatch, a full disk, no DNS, an unmapped system lib, an unusable GPU.
#   • Lane 2 HANDOFF — an ambiguous *env* failure to route to the gated multi-turn
#     LLM planner (a dependency-resolution / solver / python-pin conflict); never a
#     deterministic guess. Key-free it surfaces its message; with a key it's planned.
#   • Lane 4 RETRY   — a transient network blip → retry the SAME build unchanged.
#
# Every Lane-3 message tells the user exactly what to do; nothing here mutates an env
# except the Lane-1 system-lib repair, which is namespace- and channel-guarded.
# --------------------------------------------------------------------------- #
class DiagLane(IntEnum):
    REPAIR = 1  # deterministic env repair (system lib)
    HANDOFF = 2  # ambiguous env failure → gated LLM planner
    SURFACE = 3  # honest, actionable, no-mutation surface — stops the loop
    RETRY = 4  # transient — retry the same build


class DiagKind(StrEnum):
    # Lane 1 — deterministic repair (kept out of classify_failure for the .so negative pin)
    MISSING_SYSTEM_LIB = "missing_system_lib"
    # Lane 3 — surface honestly, no mutation, stop the loop
    EXTERNAL_TOKEN_REQUIRED = "external_token_required"
    SPECIAL_INPUT_REQUIRED = "special_input_required"
    DATA_SHAPE = "data_shape"
    DISK_FULL = "disk_full"
    NETWORK_UNREACHABLE = "network_unreachable"
    SYSTEM_LIB_NEEDS_ROOT = "system_lib_needs_root"
    GPU_UNAVAILABLE = "gpu_unavailable"
    BENIGN_FALLBACK = "benign_fallback"
    # Lane 2 — ambiguous env failure → gated LLM planner
    PIP_RESOLUTION_CONFLICT = "pip_resolution_conflict"
    PIP_DIST_NOT_FOUND = "pip_dist_not_found"
    CONDA_UNSAT_VERSION = "conda_unsat_version"
    CONDA_UNSAT_CHANNEL = "conda_unsat_channel"
    PYTHON_VERSION = "python_version"
    # Lane 4 — transient
    TRANSIENT_NETWORK = "transient_network"


@dataclass(frozen=True)
class Diagnosis:
    """An advisory read of a failure that :func:`classify_failure` did not claim.

    ``message`` is user-facing and actionable — printed verbatim on a SURFACE outcome.
    ``package`` carries the conda package (system-lib), the env var (token), or the
    conflicting dependency (handoff) when known. ``lane`` decides what the agent does.
    """

    kind: DiagKind
    lane: DiagLane
    message: str
    package: str = ""
    detail: str = ""

    @property
    def terminal(self) -> bool:
        """A SURFACE diagnosis stops the loop — no env action will change the outcome."""
        return self.lane == DiagLane.SURFACE


# -- Lane 3 / Lane 4 signatures (all disjoint from classify_failure's repairable set) -- #
_DISK_FULL_RE = re.compile(r"No space left on device|\[Errno 28\]|Disk quota exceeded", re.I)
_DNS_DOWN_RE = re.compile(
    r"Temporary failure in name resolution|Could not resolve host|Name or service not known|"
    r"getaddrinfo failed|Network is unreachable",
    re.I,
)
_MISSING_SO_RE = re.compile(r"\b(lib[\w.+-]+\.so(?:\.[\d.]+)?)\s*:\s*cannot open shared object file", re.I)
# A too-old system C++/C runtime: the .so EXISTS but lacks the SYMBOL VERSION a compiled extension
# needs (`version `GLIBCXX_3.4.30' not found`, `version `GLIBC_2.34' not found`). Distinct from a
# *missing* .so (above): the conda-forge fix is to pull a NEWER libstdcxx-ng / libgcc-ng INTO the env
# (they ship a newer libstdc++.so.6 satisfying the symbol), not a blind RECREATE against the same host
# runtime. GLIBCXX is checked before GLIBC so a libstdc++ fault maps to both packages, not just libgcc.
_GLIBCXX_RE = re.compile(r"GLIBCXX_[\d.]+['\"]?\s+not found", re.I)
_GLIBC_VER_RE = re.compile(r"version\s+`?GLIBC_[\d]", re.I)
_GPU_RES = (
    re.compile(r"CUDA error: no kernel image is available", re.I),
    re.compile(r"CUDA driver version is insufficient", re.I),
    re.compile(r"no CUDA-capable device is detected", re.I),
    re.compile(r"Torch not compiled with CUDA enabled", re.I),
)
# External credential: a marker that a token/key is *required*, plus an UPPER_SNAKE env-var name.
_TOKEN_MARKER_RE = re.compile(
    r"missing token|token[^\n]{0,20}required|provide[^\n]{0,20}token|api[\s_]?key|"
    r"not authenticated|authenticate\(|register at",
    re.I,
)
_TOKEN_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")  # case-sensitive: UCD_TOKEN, OPENAI_API_KEY
# A special input the generic mini data does not carry (Seurat RDS, spaceranger dir, DAPI/segmentation).
_SPECIAL_RDS_RE = re.compile(r"requires?\s+Seurat\s+RDS|Convert\s+h5ad\s+to\s+RDS|readRDS", re.I)
_SPECIAL_SPACERANGER_RE = re.compile(
    r"\bspaceranger\b|requires?\s+outs_dir|Missing required Visium files|tissue_positions", re.I
)
_SPECIAL_IMAGING_RE = re.compile(r"\bDAPI\b|fp_dapi|imaging[- ]segmentation|nucle(?:us|i)\s*segmentation", re.I)
# Input-shape mismatch: obsm['spatial'] absent, a required param unset, a graph too small for the tool.
_DATA_SHAPE_RES = (
    re.compile(r"missing\s+obsm\[['\"]spatial['\"]\]", re.I),
    re.compile(r"['\"]spatial['\"]\s+not found in\s+adata\.obsm", re.I),
    re.compile(r"KeyError:\s*['\"](?:spatial|X_spatial|spatial_connectivities)['\"]", re.I),
    re.compile(r"n_clusters must be provided", re.I),
    re.compile(r"too few (?:spots|cells|nodes|neighbors)", re.I),
    re.compile(r"No common genes found", re.I),
)
# Lane 2 — ambiguous env failures that route to the gated planner (never a deterministic guess).
_PIP_CONFLICT_RE = re.compile(r"ResolutionImpossible|conflicting dependencies|cannot install.+because", re.I)
# A plain-PyPI miss (classify_failure already claimed the +local/git OFF_INDEX form, so anything reaching
# diagnose is an ordinary "renamed / yanked / private / typo'd" distribution — a planner decision, not a guess).
_PIP_NOT_FOUND_RE = re.compile(
    r"No matching distribution found for|Could not find a version that satisfies the requirement", re.I
)
_CONDA_UNSAT_VERSION_RE = re.compile(r"UnsatisfiableError", re.I)
_CONDA_UNSAT_CHANNEL_RE = re.compile(r"PackagesNotFoundError", re.I)
_PYTHON_VERSION_RE = re.compile(r"Requires-Python|requires a different [Pp]ython", re.I)
# Lane 4 — transient network (distinct from Lane-3 DNS-down: here the name resolves, the transfer blips).
_TRANSIENT_NET_RE = re.compile(
    r"Connection reset by peer|Read timed out|ReadTimeoutError|Connection aborted|IncompleteRead|"
    r"ProtocolError|Connection broken|\b50[23]\s+(?:Bad Gateway|Service|Gateway)",
    re.I,
)
# "Is this text ONLY a benign optional-dep fallback?" — a benign marker with no hard-error token.
_HARD_ERROR_RE = re.compile(r"Traceback|Error:|Exception|Errno|\brc=\s*[1-9]|exit(?:ed)?\s+[1-9]|\bfatal\b", re.I)


def _find_required_token(text: str) -> str | None:
    """The UPPER_SNAKE credential name a failure says is required (``UCD_TOKEN``), or ``None``.
    Gated on a 'token/key required' marker so an unrelated UPPER_SNAKE constant never trips it."""
    if not _TOKEN_MARKER_RE.search(text):
        return None
    names = _TOKEN_NAME_RE.findall(text)
    for n in names:  # prefer a credential-shaped name
        if n.endswith(("TOKEN", "KEY", "SECRET", "APIKEY")):
            return n
    return names[0] if names else "the required credential"


def diagnose(text: str, *, spec: ToolSpec | None = None) -> Diagnosis | None:
    """Advisory read of a failure ``classify_failure`` did not claim. Returns a :class:`Diagnosis`
    (repair / surface / handoff / retry) or ``None`` when even this layer cannot place it (a truly
    unknown failure the agent may hand to the planner). Pure; never mutates.

    Ordering: unambiguous infrastructure (disk → DNS → missing ``.so`` → GPU) first, then external
    requirements (token → special input → data shape), then ambiguous env handoffs, then transient
    retry, then a benign-only fallback. First match wins."""
    if not text:
        return None

    # --- Lane 3: unambiguous infrastructure ------------------------------------------------ #
    if m := _DISK_FULL_RE.search(text):
        return Diagnosis(
            DiagKind.DISK_FULL,
            DiagLane.SURFACE,
            "No space left on device — free disk and re-run; the env build cannot proceed here.",
            detail=m.group(0),
        )
    if m := _DNS_DOWN_RE.search(text):
        return Diagnosis(
            DiagKind.NETWORK_UNREACHABLE,
            DiagLane.SURFACE,
            f"No network/DNS ({m.group(0)}) — check connectivity/proxy and re-run; not an env problem to repair.",
            detail=m.group(0),
        )
    if m := _MISSING_SO_RE.search(text):
        soname = m.group(1)
        pkg = _conda_pkg_for_so(soname)
        if pkg:  # Lane 1 — a mapped .so is a deterministic conda-forge install
            return Diagnosis(
                DiagKind.MISSING_SYSTEM_LIB,
                DiagLane.REPAIR,
                f"Missing system library '{soname}' → conda-install '{pkg}' from conda-forge.",
                package=pkg,
                detail=soname,
            )
        return Diagnosis(  # Lane 3 — unmapped: likely needs a distro package the wizard cannot install
            DiagKind.SYSTEM_LIB_NEEDS_ROOT,
            DiagLane.SURFACE,
            f"Missing system library '{soname}' with no conda package mapping — it likely needs a distro "
            f"package (e.g. `apt-get install` the providing lib), which the wizard cannot install; ask an "
            f"admin, then re-run.",
            detail=soname,
        )
    # A too-old system libstdc++/glibc symbol version (C8): pull a NEWER libstdcxx-ng/libgcc-ng into
    # the env rather than blindly RECREATE against the same host runtime. Moved here from
    # classify_failure's ENV_BROKEN set; `undefined symbol` (a real ABI break) still routes to RECREATE.
    if _GLIBCXX_RE.search(text):
        return Diagnosis(
            DiagKind.MISSING_SYSTEM_LIB,
            DiagLane.REPAIR,
            "A compiled extension needs a newer libstdc++ (a GLIBCXX_ symbol version is not found) → "
            "conda-install 'libstdcxx-ng libgcc-ng' from conda-forge to ship a newer runtime.",
            package="libstdcxx-ng libgcc-ng",
            detail="GLIBCXX (libstdc++ too old)",
        )
    if _GLIBC_VER_RE.search(text):
        return Diagnosis(
            DiagKind.MISSING_SYSTEM_LIB,
            DiagLane.REPAIR,
            "A compiled extension needs a newer glibc runtime symbol (a GLIBC_ version is not found) → "
            "conda-install 'libgcc-ng' from conda-forge to ship a newer runtime.",
            package="libgcc-ng",
            detail="GLIBC (runtime too old)",
        )
    for rx in _GPU_RES:
        if m := rx.search(text):
            return Diagnosis(
                DiagKind.GPU_UNAVAILABLE,
                DiagLane.SURFACE,
                f"GPU/CUDA unavailable or incompatible ({m.group(0)}) — this tool needs a working CUDA GPU; "
                f"it cannot run on this CPU-only box.",
                detail=m.group(0),
            )

    # --- Lane 3: external requirements the mini data / box does not satisfy ----------------- #
    if tok := _find_required_token(text):
        return Diagnosis(
            DiagKind.EXTERNAL_TOKEN_REQUIRED,
            DiagLane.SURFACE,
            f"This tool needs the '{tok}' credential — set it (e.g. `export {tok}=…`) and re-run; "
            f"it is not an environment build problem.",
            package=tok,
        )
    if _SPECIAL_RDS_RE.search(text):
        need = "a Seurat .rds object (with the embedded H&E image)"
        return Diagnosis(DiagKind.SPECIAL_INPUT_REQUIRED, DiagLane.SURFACE, _special_msg(need), detail="seurat_rds")
    if _SPECIAL_SPACERANGER_RE.search(text):
        need = "a Visium spaceranger 'outs/' directory (tissue image + tissue_positions)"
        return Diagnosis(DiagKind.SPECIAL_INPUT_REQUIRED, DiagLane.SURFACE, _special_msg(need), detail="spaceranger")
    if _SPECIAL_IMAGING_RE.search(text):
        need = "an imaging-segmentation input directory (DAPI image + transcript table)"
        return Diagnosis(DiagKind.SPECIAL_INPUT_REQUIRED, DiagLane.SURFACE, _special_msg(need), detail="imaging")
    for rx in _DATA_SHAPE_RES:
        if m := rx.search(text):
            return Diagnosis(
                DiagKind.DATA_SHAPE,
                DiagLane.SURFACE,
                f"Input-shape mismatch ({m.group(0)}) — the tool's expected data/params are not met by this "
                f"input; this is a data/parameter issue, not an environment problem.",
                detail=m.group(0),
            )

    # --- Lane 2: ambiguous env failures → the gated multi-turn planner ---------------------- #
    if m := _PIP_CONFLICT_RE.search(text):
        return Diagnosis(
            DiagKind.PIP_RESOLUTION_CONFLICT,
            DiagLane.HANDOFF,
            "pip cannot resolve a compatible set of versions — this needs assisted remediation "
            "(enable the LLM planner) or a recipe pin adjustment.",
            detail=m.group(0)[:160],
        )
    if m := _PIP_NOT_FOUND_RE.search(text):
        return Diagnosis(
            DiagKind.PIP_DIST_NOT_FOUND,
            DiagLane.HANDOFF,
            "pip cannot find a matching distribution — the package may be renamed, yanked, private, or "
            "mistyped; enable assisted remediation or correct the requirement.",
            detail=m.group(0)[:160],
        )
    if m := _CONDA_UNSAT_VERSION_RE.search(text):
        return Diagnosis(
            DiagKind.CONDA_UNSAT_VERSION,
            DiagLane.HANDOFF,
            "the conda solver cannot satisfy the pinned versions — enable assisted remediation or relax a pin.",
            detail=m.group(0)[:160],
        )
    if m := _CONDA_UNSAT_CHANNEL_RE.search(text):
        return Diagnosis(
            DiagKind.CONDA_UNSAT_CHANNEL,
            DiagLane.HANDOFF,
            "a package is not in the configured conda channels — enable assisted remediation to add a "
            "trusted channel, or install it manually.",
            detail=m.group(0)[:160],
        )
    if m := _PYTHON_VERSION_RE.search(text):
        return Diagnosis(
            DiagKind.PYTHON_VERSION,
            DiagLane.HANDOFF,
            "a dependency needs a different Python than the recipe pins — enable assisted remediation to "
            "rebuild at a compatible Python.",
            detail=m.group(0)[:160],
        )

    # --- Lane 4: transient network → retry the same build ----------------------------------- #
    if m := _TRANSIENT_NET_RE.search(text):
        return Diagnosis(
            DiagKind.TRANSIENT_NETWORK,
            DiagLane.RETRY,
            f"transient network error ({m.group(0)}) — retrying the same build.",
            detail=m.group(0),
        )

    # --- benign-only: an optional-dependency fallback with no hard error → nothing to fix --- #
    if _PY_MODULE_BENIGN_RE.search(text) and not _HARD_ERROR_RE.search(text):
        return Diagnosis(
            DiagKind.BENIGN_FALLBACK,
            DiagLane.SURFACE,
            "only an optional-dependency fallback was found (not a hard failure) — nothing to repair.",
        )
    return None


def _special_msg(need: str) -> str:
    return (
        f"This tool needs {need}, which the generic mini demo data does not provide — supply that input "
        f"and re-run; it is not an environment build problem."
    )


def repair_diagnosis(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    diag: Diagnosis,
    *,
    io: PromptIO,
    log: SessionLog | None = None,
) -> RepairResult:
    """Dispatch a Lane-1 (REPAIR) :class:`Diagnosis` to a guarded repair. Today only
    ``MISSING_SYSTEM_LIB`` is repairable; any other (Lane 2/3/4) is advisory and returns
    ``action='none'`` with the diagnosis message. Mirrors :func:`repair_env` for the diagnose
    layer: dry-run short-circuits, never raises, ``repaired`` is the only flip signal."""
    if getattr(conda, "dry_run", False):
        return RepairResult(
            kind=str(diag.kind),
            action="(dry-run)",
            attempted=True,
            repaired=False,
            dry_run=True,
            package=diag.package,
            detail=f"dry-run: would repair {diag.kind}",
        )
    try:
        if diag.kind is DiagKind.MISSING_SYSTEM_LIB:
            return _repair_system_lib(conda, spec, basic_env, spec.target_env(basic_env), diag, io=io, log=log)
        return RepairResult(kind=str(diag.kind), action="none", attempted=False, repaired=False, detail=diag.message)
    except Exception as exc:  # same contract as repair_env: never raises
        _log(log, "env_repair", server=spec.server_key, action="error", error=str(exc)[:200])
        return RepairResult(
            kind=str(diag.kind), action="error", attempted=True, repaired=False, detail=f"repair crashed: {exc}"[:200]
        )


def _repair_system_lib(
    conda: Conda,
    spec: ToolSpec,
    basic_env: str,
    target: str,
    diag: Diagnosis,
    *,
    io: PromptIO,
    log: SessionLog | None,
) -> RepairResult:
    """``conda_install`` the conda-forge package that provides a missing system ``.so`` into the
    tool env, then re-probe the tool's import. Namespace-guarded (``assert_deletable_env`` proves
    ``target`` is a mutable ``<basic>_*`` env); the channel is on :data:`TRUSTED_CONDA_CHANNELS`."""
    pkg = diag.package
    if not pkg:
        return RepairResult(
            kind=str(diag.kind),
            action="system_lib",
            attempted=False,
            repaired=False,
            detail="no conda package known for the missing .so",
        )
    if "conda-forge" not in TRUSTED_CONDA_CHANNELS:  # defensive; the literal is on the list
        return RepairResult(
            kind=str(diag.kind),
            action="system_lib",
            attempted=False,
            repaired=False,
            detail="conda-forge not on the trusted channel list",
        )
    constants.assert_deletable_env(basic_env, target)  # only ever mutate a managed <basic>_* env
    io.say(f"  envdoctor: {target} is missing '{diag.detail}' — conda installing '{pkg}' (conda-forge)…")
    try:
        # ``pkg`` may name MORE THAN ONE conda package (C8: a GLIBCXX fault installs both
        # ``libstdcxx-ng`` and ``libgcc-ng``); split on whitespace so each becomes its own spec.
        # ``.split()`` on a single-token package (every _conda_pkg_for_so mapping) is a no-op list.
        res = conda.conda_install(target, pkg.split(), channels=["conda-forge"], check=False)
        installed = bool(res.ok or res.dry_run)
    except Exception as exc:  # OSError / timeout / a CondaError from a lower layer
        io.warn(f"{target}: conda install '{pkg}' errored ({exc})")
        installed = False
    probe = spec.import_check or spec.server_key
    import_ok = _probe_import(conda, spec, target, probe) if probe else installed
    repaired = bool(installed and import_ok)
    if repaired:
        io.ok(f"{target}: installed system lib '{pkg}' (envdoctor)")
    _log(log, "env_repair", server=spec.server_key, env=target, action="system_lib", package=pkg, repaired=repaired)
    return RepairResult(
        kind=str(diag.kind),
        action="system_lib",
        attempted=True,
        repaired=repaired,
        package=pkg,
        detail=f"conda install '{pkg}' + import '{probe}' {'ok' if repaired else 'still fails'}",
    )


# --------------------------------------------------------------------------- #
# RemediationContext builder for the OPTIONAL, gated LLM self-review deep tier
# --------------------------------------------------------------------------- #
def _worker_dir(worker_file: str) -> Path:
    """Which agent dir holds this worker basename — ``agent/tools/`` or ``agent/tools_user/``."""
    for d in (constants.tools_dir(), constants.tools_user_dir()):
        if worker_file and (d / worker_file).exists():
            return d
    return constants.tools_dir()


def build_remediation_context(spec: ToolSpec, env_name: str):
    """Construct a fully-populated ``RemediationContext`` (all 7 required fields) for the
    optional, ``SOG_SELF_REVIEW_ENABLED``-gated LLM self-review path.

    The agent-side dataclass is imported lazily so a bare launcher env can still import
    ``envdoctor``. ``source_url`` is empty — setup-side repair never git-clones a tool;
    it only fixes the environment. ``knowledge_dir`` is the wizard's own git-ignored
    durable state, never an agent/tool/package path.

    Refuses outright (``PermissionError``) for an empty ``env_name`` or a
    :data:`constants.PROTECTED_ENVS` env (SETUP-2). The agent-side ``self_review_loop`` MUTATES this
    env via ``conda install -n <env>``, so this builder — the single choke point every self-review
    caller funnels through on its way into the forbidden ``tools_user`` module — is where we guarantee
    the shared ``base`` / ``spatialomicsgym_e1`` / ``spatialomicsgym_env`` / ``sog_reproduce`` envs can never be
    touched, mirroring the deletion-side guard (``Conda.remove_env`` / ``assert_deletable_env``). Each
    caller already wraps this in a try/except that records the refusal and skips the review untouched.
    """
    if not env_name or not env_name.strip():
        raise PermissionError("refusing to self-review an empty env name")
    if env_name in constants.PROTECTED_ENVS:
        raise PermissionError(f"refusing to self-review protected env {env_name!r} (PROTECTED_ENVS)")
    from tools_user.self_review import RemediationContext  # lazy: heavy, agent-side

    worker_file = getattr(spec, "worker_file", "") or ""
    server_file = getattr(spec, "server_file", "") or ""
    wdir = _worker_dir(worker_file)
    # Portals live beside their workers in tools/ (or tools_user/) — MCP_server/ holds only the two
    # config yamls (hunt 2026-09-30, u36-setup-install-13). Pointing server_path there made every
    # SCHEMA_VIOLATION remediation raise FileNotFoundError on its first read, for every tool.
    sdir = _worker_dir(server_file)
    knowledge = constants.state_dir() / "self_review"
    try:
        knowledge.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # a read-only / full state dir must not sink the (opt-in) self-review context build
    return RemediationContext(
        tool_id=spec.server_key,
        env_name=env_name,
        source_url="",
        worker_path=(wdir / worker_file) if worker_file else wdir,
        server_path=(sdir / server_file) if server_file else sdir,
        language="r" if getattr(spec, "worker_kind", "") == "rscript" else "python",
        knowledge_dir=knowledge,
    )
