"""Read the persisted build log and surface its diagnosis-dense lines.

This is the "let the ReAct strengthen the log monitor" half of the self-heal upgrade: instead of
reasoning only over the caller's short ``observe()`` string, the loop reads the full-fidelity
``<target>.build.log`` that :func:`provision._persist_build_log` writes on every failed build, and
pulls out the lines an env classifier keys on.

Design guarantees (why the self-heal loop can trust this even when a monitor read fails):

* **Stdlib only** (``pathlib`` / ``re`` / ``dataclasses`` + the sibling ``constants`` module) so it
  imports cleanly inside :mod:`installer_scientist` without breaking the setup import-boundary test.
* **Every public entry point is ``_safe``-wrapped** — a missing file, a permission error, a decode
  problem, anything — degrades to the plain observe text (``read_tail`` → ``""``, ``salient_lines``
  → ``[]``, ``gather`` → evidence built from the observe string alone). The loop is never perturbed.
* **Read-only.** Nothing here mutates an env or decides which repair runs; it only *enriches the
  evidence*. :func:`gather` returns both a bounded ``salient`` tail (for display + the planner block)
  and a non-lossy ``classify_text`` (what the classifier should actually run on), so surfacing the
  build log can only *add* signal, never drop a line the classifier would have matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import constants
from .session_log import redact

# --- cross-turn signature ---------------------------------------------------------------------
# Kept byte-for-byte in sync with ``installer_scientist._signature`` so the no-progress guard can
# compare a signature computed here against the loop's own notion of "same error again". (We cannot
# import it: installer_scientist imports *this* module, and the reverse edge would be a cycle.)
_SIG_NUM_RE = re.compile(r"0x[0-9a-fA-F]+|\d+")
_SIG_WS_RE = re.compile(r"\s+")


def _signature(text: str) -> str:
    """Volatile-stripped fingerprint: lowercase, drop hex/ints, collapse whitespace, clip to 400."""
    t = _SIG_NUM_RE.sub("", (text or "").lower())
    return _SIG_WS_RE.sub(" ", t).strip()[:400]


# --- salient-line vocabulary ------------------------------------------------------------------
# A broad, line-level "this line carries a diagnosis" set. Intentionally a SUPERSET of
# ``envdoctor.classify_failure``'s vocabulary: extra matches are harmless (the classifier re-applies
# its own anchoring + benign filters downstream), but a *dropped* line would change what the
# classifier sees, so we err toward matching. Order is display-only and does not affect correctness.
_SALIENT_RES = (
    # python import failures
    re.compile(r"\bNo module named\b", re.I),
    re.compile(r"\bModuleNotFoundError\b", re.I),
    re.compile(r"\bImportError\b", re.I),
    re.compile(r"cannot import name ", re.I),
    # dependency resolution / off-index wheels
    re.compile(r"No matching distribution found for", re.I),
    re.compile(r"Could not find a version", re.I),
    re.compile(r"ResolutionImpossible|conflicting dependencies", re.I),
    # conda resolver
    re.compile(r"CondaError|ResolvePackageNotFound|PackagesNotFoundError", re.I),
    # environment absent / wrong interpreter
    re.compile(r"EnvironmentLocationNotFound|Could not find conda environment", re.I),
    re.compile(r"bin/(?:python|Rscript)[\d.]*\s*:\s*(?:No such file|not found)", re.I),
    re.compile(r"\binterpreter missing\b|\benv\s+[\w.+-]+\s+missing\b", re.I),
    # native / ABI breakage
    re.compile(r"undefined symbol", re.I),
    re.compile(r"GLIBCXX_|version `?GLIBC", re.I),
    re.compile(r"numpy\.dtype size changed|multiarray failed to import|_ARRAY_API not found", re.I),
    re.compile(r"compiled against API version|cannot be run in NumPy 2", re.I),
    re.compile(r"DLL load failed", re.I),
    re.compile(r"cannot open shared object file", re.I),
    # C toolchain
    re.compile(r"fatal error:", re.I),  # covers Python.h + any generic missing-header
    re.compile(r"\bPython\.h\b", re.I),
    re.compile(r"error:\s*command\s+.*(?:gcc|g\+\+|clang|cc)'?\s+failed", re.I),
    re.compile(r"(?:gcc|g\+\+|clang|cc):\s*(?:command\s+)?not found", re.I),
    # R package load failures
    re.compile(r"there is no package called", re.I),
    re.compile(r"Error in library\(", re.I),
    re.compile(r"package or namespace load failed", re.I),
    re.compile(r"package ['\"][\w.]+['\"] is not available", re.I),
    re.compile(r"(?:loadNamespace|requireNamespace)\(", re.I),
    # wiring
    re.compile(r"no generated MCP config|wiring did not run", re.I),
    # infra: disk / network / GPU
    re.compile(r"No space left on device|\[Errno 28\]|Disk quota exceeded", re.I),
    re.compile(r"Temporary failure in name resolution|Could not resolve host|Network is unreachable", re.I),
    re.compile(r"CUDA error|CUDA driver version|no CUDA-capable device|Torch not compiled with CUDA", re.I),
    # pip's own guidance lines
    re.compile(r"\bpip install\b", re.I),
    # pip build-backend (PEP 517) failures — the modern "wheel build blew up" markers (C7)
    re.compile(r"metadata-generation-failed|subprocess-exited-with-error", re.I),
    re.compile(r"Failed building wheel for|Failed to build \S", re.I),
    # libmamba — conda's default solver since 23.10; its unsat/HTTP/SSL errors read differently
    # from the classic ``CondaError``/``ResolvePackageNotFound`` above (C7)
    re.compile(r"nothing provides|LibMambaUnsatisfiableError|Could not solve for environment specs", re.I),
    re.compile(r"CondaHTTPError|CondaSSLError|CondaVerificationError", re.I),
    # transient network timeouts — the retry lane's fingerprints (C7)
    re.compile(r"Read timed out|ReadTimeoutError|Max retries exceeded|Connection reset|Connection aborted", re.I),
    # download integrity (C7)
    re.compile(r"Hash mismatch|ChecksumMismatchError|SHA256 mismatch", re.I),
    # filesystem permissions (C7)
    re.compile(r"Permission denied|\[Errno 13\]|Access is denied", re.I),
    # generic last-resort markers (broad on purpose — keeps the superset honest)
    re.compile(r"^Traceback \(most recent call last\)", re.I),
    re.compile(r"^\s*ERROR:", re.I),
    re.compile(r"Segmentation fault|\bKilled\b|Out of memory|\bOOM\b", re.I),
)

# R logs get a couple of extra broad markers that would be too noisy on a Python build log.
_SALIENT_R_EXTRA = (
    re.compile(r"^Error in ", re.I),
    re.compile(r"Execution halted", re.I),
)

# Cap the joined salient tail so it never bloats the thinking box or the planner prompt.
_MAX_SALIENT_CHARS = 4000


@dataclass(frozen=True)
class LogEvidence:
    """What the log monitor recovered for one self-heal turn.

    ``salient``       — joined diagnostic tail, bounded; feeds the thinking box + the planner's
                        "SALIENT LOG LINES" block + the no-progress signature.
    ``classify_text`` — the text the loop should run ``classify_failure`` on. Never lossy: equals the
                        observe string when it already carries a diagnosis, else the observe string
                        merged with the persisted build-log tail. (We classify on this, not on the
                        possibly-truncated ``salient`` tail, so a high-precedence line near the top of
                        a long log can't be dropped and silently change the verdict.)
    ``full_tail``     — the raw build-log tail actually read (``""`` when none was needed/found).
    ``source``        — ``"observe"`` or ``"observe+build.log"``, for the audit trail.
    """

    signature: str = ""
    salient: str = ""
    classify_text: str = ""
    full_tail: str = ""
    source: str = "observe"


def _safe(fn, default):
    """Run ``fn`` and return ``default`` on ANY exception — the monitor must never break the loop."""
    try:
        return fn()
    except Exception:
        return default


def build_log_path(target: str) -> Path:
    """Where :func:`provision._persist_build_log` writes ``<basic>_<server>``'s joined build stderr."""
    return constants.logs_dir() / f"{target}.build.log"


def _read_tail_impl(path: Path | str, *, max_chars: int) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        # For a pathologically large log, seek near the end rather than reading it whole. ``*4``
        # over-reads to cover multibyte chars; the partial first line is then dropped.
        if size > max_chars * 4:
            fh.seek(max(0, size - max_chars * 4))
            fh.readline()
        data = fh.read()
    # The operative error sits at the tail of a multi-strategy build log — keep the end.
    return data[-max_chars:] if len(data) > max_chars else data


def read_tail(path: Path | str, *, max_chars: int = 6000) -> str:
    """``_safe`` read of the last ``max_chars`` of a build log (missing/oversize handled; never raises)."""
    return _safe(lambda: _read_tail_impl(path, max_chars=max_chars), "")


def _salient_lines_impl(text: str, *, is_r: bool, limit: int) -> list[str]:
    pats = _SALIENT_RES + (_SALIENT_R_EXTRA if is_r else ())
    out: list[str] = []
    prev: str | None = None
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if any(p.search(raw) for p in pats):
            if stripped != prev:  # collapse consecutive duplicates (retries repeat the same line)
                out.append(stripped)
                prev = stripped
    # Keep the most recent matches — the latest build strategy's failure is what matters for display.
    return out[-limit:] if len(out) > limit else out


def salient_lines(text: str, *, is_r: bool = False, limit: int = 40) -> list[str]:
    """Pull the diagnosis-dense lines from ``text`` (``_safe`` → ``[]`` on any failure)."""
    return _safe(lambda: _salient_lines_impl(text, is_r=is_r, limit=limit), [])


def _gather_impl(
    observe_text: str, *, target: str, spec, use_build_log: bool, max_chars: int, log_path: Path | str | None = None
) -> LogEvidence:
    observe_text = observe_text or ""
    is_r = bool(spec is not None and getattr(spec, "worker_kind", "") == "rscript")
    own = _salient_lines_impl(observe_text, is_r=is_r, limit=40)
    if own or not use_build_log:
        # Either the current text already carries a diagnosis (authoritative), or build-log
        # augmentation is disabled for this observation (a post-mutation re-observe, where the
        # persisted build log is potentially STALE — the authoritative ``observe()`` wins). Either
        # way, do not pull in the build log.
        classify_text = observe_text
        salient = own
        full_tail = ""
        source = "observe"
    else:
        # Low-signal SEED observation (e.g. a short summary while the full multi-strategy stderr sits
        # in the build log): recover the ORIGINAL rich failure so the loop has something to reason on.
        # ``log_path`` lets a caller point at a tier-specific sink (``<target>.tier1.log`` /
        # ``<target>.tier2.log``) instead of the provision-owned ``<target>.build.log`` — so a test-phase
        # self-heal reads THIS run's real error, never a stale provision-era log (C6).
        src = Path(log_path) if log_path is not None else build_log_path(target)
        full_tail = _read_tail_impl(src, max_chars=max_chars)
        classify_text = (observe_text + "\n" + full_tail).strip() if full_tail else observe_text
        salient = _salient_lines_impl(classify_text, is_r=is_r, limit=40)
        # Name the ACTUAL sink in the audit label (a tier log when overridden, else the provision build
        # log) while keeping the default path's label byte-for-byte "observe+build.log".
        source = ("observe+" + (src.name if log_path is not None else "build.log")) if full_tail else "observe"
    # redact BEFORE the tail-clip (R24-B/Finding-1). `salient` is built from RAW build stderr and this
    # is the single choke point every consumer reads (installer audit JSONL, the third-party
    # remediation-planner prompt, the live thinking box) — all of which do redact-before-their-own-clip
    # on `ev.salient`. But redact() matches a registered credential by EXACT substring: a secret
    # straddling this [-4000:] cut loses its head here, and the surviving tail (now at salient_text[0])
    # is unmatchable downstream, leaking a credential fragment to a durable file and an external LLM.
    # Redacting the full join first masks the whole secret, so the clip can't split one. session_log is
    # stdlib-only, so this keeps log_monitor's import boundary.
    salient_text = redact("\n".join(salient))[-_MAX_SALIENT_CHARS:]
    return LogEvidence(
        # Signature is ALWAYS taken from the raw observe text, never the (possibly build-log-enriched)
        # classify_text — so the loop's cross-turn no-progress comparison stays apples-to-apples with
        # its pre-monitor behavior (a constant build-log tail must never masquerade as "no progress").
        signature=_signature(observe_text),
        salient=salient_text,
        classify_text=classify_text or observe_text,
        full_tail=full_tail,
        source=source,
    )


def gather(
    observe_text: str,
    *,
    target: str,
    spec=None,
    use_build_log: bool = True,
    max_chars: int = 6000,
    log_path: Path | str | None = None,
) -> LogEvidence:
    """Merge the caller's ``observe_text`` with the persisted build-log tail into :class:`LogEvidence`.

    When ``observe_text`` already carries a diagnosis it is used as-is (fidelity identical to today).
    Only when it is short/low-signal *and* ``use_build_log`` is set do we read the persisted log to
    recover the original error — the caller passes ``use_build_log=True`` only for the SEED observation
    (before any repair has run, when the persisted log still reflects reality) and ``False`` for
    post-mutation re-observes (where the authoritative ``observe()`` is trusted and the build log may be
    stale). ``log_path`` overrides which file is read: by default it is ``<target>.build.log`` (the
    provision-owned sink), but a test-phase caller passes its own tier sink (``<target>.tier1.log`` /
    ``<target>.tier2.log``) so the loop reads THIS run's real error instead of a stale provision log
    (C6). Fully ``_safe``: on any failure it falls back to evidence built from the observe string alone,
    so the self-heal loop behaves exactly as it did before the monitor existed.
    """
    return _safe(
        lambda: _gather_impl(
            observe_text,
            target=target,
            spec=spec,
            use_build_log=use_build_log,
            max_chars=max_chars,
            log_path=log_path,
        ),
        LogEvidence(
            signature=_signature(observe_text or ""),
            salient="",
            classify_text=observe_text or "",
            full_tail="",
            source="observe",
        ),
    )
