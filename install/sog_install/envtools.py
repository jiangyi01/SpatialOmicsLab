"""
Thin stdlib wrapper around the conda/mamba/micromamba CLI.

Every conda interaction the wizard needs — list, exists, create, clone, remove,
``conda run``, pip-install, and the export pair used by ``capture`` — funnels
through one :class:`Conda` object so timeouts, logging, and the **dry-run gate**
live in a single place.

The dry-run gate is the mechanism behind the plan's "zero side effects" promise:
read-only calls (``env_list``/``env_exists``/``export_*``/``run``) always
execute, while every **mutating** call (``create_*``/``clone``/``remove``/
``pip_install``) is short-circuited when ``dry_run`` is set — it logs the exact
command it *would* run and returns a synthetic success.

This module never deletes anything on its own; ``remove_env`` executes whatever
name it's handed. The *namespace guard* that keeps deletions inside ``<basic>_*``
lives in :mod:`~sog_install.constants` and is applied by the callers
(``provision``/``wizard``) before they reach here.

Stdlib only.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import constants
from .preflight import find_conda
from .session_log import redact

if TYPE_CHECKING:
    from .progress import InstallProgress
    from .session_log import SessionLog

# A syntactically valid R package name: a letter, then letters/digits/dot/underscore. Bioc/CRAN
# names (SpotSweeper, SingleCellExperiment, S4Vectors) all satisfy it. Used to prove a name carries
# no quote/semicolon/space before it is interpolated into an ``Rscript -e`` string.
_R_PKG_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")


# --------------------------------------------------------------------------- #
# Portable process-group kill (N3). A build timeout must reap the WHOLE tree: mamba/conda spawns a
# detached solver + pip/gcc grandchildren that hold the pkgs/env lock, so a lone ``proc.kill()``
# (direct child only) orphans them → the next loosen-retry / RECREATE collides with the stale lock
# and hangs or half-writes the env. POSIX puts the child in its own session (``start_new_session``)
# so a timeout ``killpg``s the group; Windows lacks setsid/killpg/SIGKILL → degrade to a single-child
# terminate. Mirrors ``testing.py``'s proven helpers (kept local so this low-level module stays
# import-cycle-free — ``testing`` imports ``envtools``, not the reverse).
# --------------------------------------------------------------------------- #
_POSIX = os.name == "posix"
_SIGKILL = getattr(signal, "SIGKILL", getattr(signal, "SIGTERM", 15))


def no_capture_output_args(exe: str | None) -> list[str]:
    """The ``--no-capture-output`` flag for ``<mgr> run`` — but ONLY for conda/mamba.

    conda and mamba(1.x) require the flag for real-time streaming (the self-heal thinking box's log
    monitor reads the child's output live). micromamba's ``run`` streams by default and — being
    CLI11-based — does NOT define ``--no-capture-output``; it errors on the unknown option. Passing
    the flag there makes ``Conda.run`` (the single read-only in-env probe primitive) exit non-zero for
    EVERY caller, so on a micromamba-only host every health probe misreads as "unhealthy", the base-env
    scan reports a freshly-built env broken, and env-reuse never confirms. Omitting it for micromamba is
    correct (output still streams). Gated on the manager basename so conda/mamba are unchanged. (mamba
    >= 2.0 shares micromamba's ``run`` core and may eventually need the same treatment.)

    The manager is matched on the extension-stripped, lower-cased basename: on Windows the manager
    resolves to ``micromamba.exe`` (``preflight.find_conda`` builds exactly that, and ``shutil.which``
    returns the ``.exe`` path), so a bare ``== "micromamba"`` on the raw basename missed it and wrongly
    injected ``--no-capture-output`` — breaking EVERY ``conda run`` probe on a Windows+micromamba host,
    the exact failure this guard exists to prevent."""
    stem = os.path.splitext(os.path.basename(exe or ""))[0].lower()
    return [] if stem == "micromamba" else ["--no-capture-output"]


def _session_kwargs() -> dict:
    """``start_new_session=True`` on POSIX (own process group for group-kill); ``{}`` on Windows,
    where the kwarg has no effect and ``setsid`` does not exist."""
    return {"start_new_session": True} if _POSIX else {}


def _hard_kill(proc: subprocess.Popen) -> None:
    """Kill the whole process tree on POSIX (``killpg``), else just the child. Never raises —
    ``os.killpg``/``os.getpgid``/``SIGKILL`` are absent on Windows, so we guard on them."""
    if _POSIX and hasattr(os, "killpg") and hasattr(os, "getpgid"):
        try:
            os.killpg(os.getpgid(proc.pid), _SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass  # fall through to a direct child kill
    with contextlib.suppress(Exception):
        proc.kill()


def _run_captured(cmd: list[str], *, timeout: int) -> RunResult:
    """A ``subprocess.run``-equivalent capture whose child runs in its OWN session, so a timeout can
    group-kill the entire solver/pip tree instead of orphaning it (see the N3 note above; this path
    runs real env builds whenever live streaming is disabled — the common CI/non-TTY case).

    Raises exactly what ``subprocess.run`` would — :class:`subprocess.TimeoutExpired` on timeout,
    :class:`OSError` on a launch failure — so :meth:`Conda._exec`'s handlers translate both into the
    identical :class:`CondaError` the captured path always produced."""
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",  # conda/pip emitting non-UTF-8 must not raise UnicodeDecodeError
        **_session_kwargs(),
    ) as proc:
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _hard_kill(proc)  # reap the whole group, not just the direct child
            with contextlib.suppress(Exception):
                out, err = proc.communicate(timeout=10)  # drain pipes so the re-raise carries output
            raise  # same TimeoutExpired → _exec → CondaError (parity with the old run() path)
        except BaseException:
            # A Ctrl-C (KeyboardInterrupt) or the wizard SIGINT handler's SystemExit(130) unwinding
            # through here must NOT be left to Popen.__exit__: for any non-KeyboardInterrupt exception
            # __exit__ runs a BLOCKING wait() (bpo-25942), and the child — in its OWN session, so it
            # never received the terminal's SIGINT — would hang the wizard forever. Group-kill first so
            # __exit__'s wait() returns at once, best-effort drain, then re-raise the interrupt unchanged.
            _hard_kill(proc)
            with contextlib.suppress(Exception):
                proc.communicate(timeout=10)
            raise
        return RunResult(proc.returncode, out or "", err or "")


class CondaError(RuntimeError):
    """A conda CLI call failed (non-zero exit) or the manager is missing."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass
class RunResult:
    """Outcome of a conda CLI call (or a dry-run stand-in)."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass
class Conda:
    """Bound to one manager exe for the whole run."""

    exe: str | None = None
    version: str = ""
    dry_run: bool = False
    log: SessionLog | None = None
    progress: InstallProgress | None = None
    # Opt-in (default off): tee the three long build calls (create_from_yaml / pip_install / clone)
    # to a live, `tail -f`-able per-tool log via a Popen reader instead of capturing silently. The
    # returned RunResult is byte-identical either way. The wizard sets this from SOG_STREAM_BUILD_LOGS;
    # every existing caller and test double leaves it False and keeps the plain subprocess.run path.
    stream_build_logs: bool = False
    _env_cache: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.exe is None:
            self.exe, self.version = find_conda()
        if not self.exe:
            raise CondaError("no conda/mamba/micromamba found on PATH")

    # -- low-level exec -------------------------------------------------------
    def _exec(
        self,
        args: list[str],
        *,
        mutating: bool,
        timeout: int,
        check: bool = True,
        label: str | None = None,
        stream: bool = False,
        stream_log_path: str | os.PathLike[str] | None = None,
    ) -> RunResult:
        cmd = [self.exe, *args]
        if mutating and self.dry_run:
            self._log("conda_dryrun", cmd=cmd)
            return RunResult(returncode=0, dry_run=True)
        self._log("conda_exec", cmd=cmd, mutating=mutating)
        # A long, output-captured mutating call (env build / pip install) otherwise looks hung.
        # Show a friendly spinner for its duration — a no-op on non-TTY / scripted / dry runs.
        spinner = (
            self.progress.task(label)
            if (label and mutating and self.progress is not None)
            else contextlib.nullcontext()
        )
        # The streaming path (opt-in) and the captured path build the *same* RunResult and raise the
        # *same* exceptions (TimeoutExpired → CondaError below, OSError → CondaError below), so the
        # dry-run gate, spinner, timeout/OSError translation, and the check→raise policy are shared
        # and live here exactly once. Streaming only changes *how* the bytes are read, never the
        # returned value.
        try:
            with spinner:
                if stream:
                    result = self._run_streaming(cmd, timeout=timeout, stream_log_path=stream_log_path)
                else:
                    # Own-session capture so a build timeout group-kills the whole solver/pip tree (N3),
                    # not just the direct child — same exceptions the old subprocess.run raised.
                    result = _run_captured(cmd, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise CondaError(f"timed out after {timeout}s: {' '.join(args[:3])}…", stderr=str(exc)) from exc
        except OSError as exc:
            raise CondaError(f"could not run {self.exe}: {exc}") from exc
        if check and not result.ok:
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-8:]
            raise CondaError(
                f"`{self.exe} {' '.join(args[:4])}…` exited {result.returncode}",
                returncode=result.returncode,
                stderr="\n".join(tail),
            )
        return result

    # -- live-streaming exec (opt-in) -----------------------------------------
    def _run_streaming(
        self,
        cmd: list[str],
        *,
        timeout: int,
        stream_log_path: str | os.PathLike[str] | None,
    ) -> RunResult:
        """Run ``cmd`` via :class:`subprocess.Popen`, teeing every line **live** to the per-tool
        ``.build.log`` (a ``tail -f``-able, full-fidelity sink) and a bounded number of ``build_line``
        events to the :class:`SessionLog` JSONL, then return the **byte-identical** :class:`RunResult`
        the captured path would have produced — the same split ``stdout``/``stderr`` strings, the same
        ``returncode``, the same universal-newline handling (``text=True``).

        Two daemon reader threads (one per pipe) keep the ``stdout``/``stderr`` split intact — a single
        merged reader would collapse ``stderr`` into ``stdout`` and break that contract — and drain the
        tail after the child exits so no line is lost. Timeout parity: a run exceeding ``timeout`` kills
        the child and re-raises :class:`subprocess.TimeoutExpired`, the exact exception ``_exec`` already
        translates into the same ``CondaError`` as the captured path. Every I/O sink is best-effort:
        a failed log open/write degrades to no tee, never to a failed build.
        """
        logf = self._open_stream_log(stream_log_path)
        out_buf: list[str] = []
        err_buf: list[str] = []
        counter = [0]  # SessionLog build_line events emitted so far (bounded by STREAM_LOG_MAX_LINES)
        lock = threading.Lock()  # serializes the two readers' writes to logf / self.log / counter

        def _reader(pipe, name: str, buf: list[str]) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    buf.append(line)
                    self._tee_line(name, line, logf, counter, lock)
            except (OSError, ValueError):  # pipe torn down under us (kill) — stop quietly
                pass
            finally:
                with contextlib.suppress(Exception):
                    pipe.close()

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                bufsize=1,  # line-buffered: readline() returns whole lines as they arrive
                **_session_kwargs(),  # own session → a timeout can group-kill the whole build tree (N3)
            )
        except OSError:
            if logf is not None:
                with contextlib.suppress(Exception):
                    logf.close()
            raise  # → _exec's `except OSError` → CondaError (same as the captured path)
        self._log("build_stream_open", cmd=cmd, log_file=str(stream_log_path) if stream_log_path else "")
        t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout", out_buf), daemon=True)
        t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr", err_buf), daemon=True)
        t_out.start()
        t_err.start()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _hard_kill(proc)  # group-kill the whole build tree, not just the direct child (N3)
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)
            self._close_stream_log(logf, counter, timed_out=True, returncode=None)
            # Re-raise the SAME timeout, now carrying whatever the readers captured, so _exec's
            # handler produces the identical CondaError the captured path would.
            raise subprocess.TimeoutExpired(cmd, timeout, output="".join(out_buf), stderr="".join(err_buf)) from exc
        except BaseException:
            # Ctrl-C / the wizard SIGINT handler's SystemExit(130) unwinding through here: this is a bare
            # Popen (no `with`), so without an explicit reap the child — in its OWN session, so it never
            # saw the terminal's SIGINT — orphans along with the mamba/pip/gcc tree it spawned, holding
            # the pkgs/env lock and colliding with the next retry/RECREATE. Group-kill, drain, close the
            # log, then re-raise the interrupt unchanged (the wizard still exits 130).
            _hard_kill(proc)
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
            t_out.join(timeout=2.0)
            t_err.join(timeout=2.0)
            self._close_stream_log(logf, counter, timed_out=False, returncode=None)
            raise
        # Child exited → both pipes hit EOF → readers drain the final buffered lines and stop.
        # Join (bounded) so every line is captured before we assemble the identical RunResult.
        t_out.join(timeout=5.0)
        t_err.join(timeout=5.0)
        self._close_stream_log(logf, counter, timed_out=False, returncode=proc.returncode)
        return RunResult(proc.returncode, "".join(out_buf), "".join(err_buf))

    def _open_stream_log(self, stream_log_path: str | os.PathLike[str] | None):
        """Open the per-tool build log for live appending, or ``None`` (no path / unwritable)."""
        if not stream_log_path:
            return None
        try:
            p = Path(stream_log_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            return open(p, "a", encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _tee_line(self, name: str, line: str, logf, counter: list[int], lock: threading.Lock) -> None:
        """Append one line to the live build log and (bounded) the SessionLog. Never raises."""
        with lock:
            if logf is not None:
                try:
                    # C1 — redact before the RAW file sink. The JSONL twin below is redacted by
                    # session_log._sanitize, but this tail-able <target>.build.log bypassed it, so a
                    # credentialed index URL / registered token echoed in a build line landed on disk
                    # raw and then flowed into log_monitor.read_tail → the planner prompt.
                    logf.write(redact(line))
                    logf.flush()
                except OSError:
                    pass
            if self.log is not None and counter[0] < constants.STREAM_LOG_MAX_LINES:
                counter[0] += 1
                # Redact BEFORE the 500-char clip. session_log._sanitize redacts this JSONL twin too,
                # but a registered token straddling char 500 (e.g. pip's `Looking in indexes:` line with a
                # credentialed --extra-index-url) would have its span cut first, so redact() could no longer
                # match the whole secret and its surviving prefix hit the durable run-*.jsonl raw. Redacting
                # the full line first (mirroring the RAW sink at logf.write(redact(line)) above) masks the
                # token whole, then the clip is safe. C1/R18-F1 redact-before-egress class.
                self.log.event("build_line", stream=name, line=redact(line.rstrip("\n"))[:500])
                if counter[0] == constants.STREAM_LOG_MAX_LINES:
                    self.log.event("build_line_truncated", after=constants.STREAM_LOG_MAX_LINES)

    def _close_stream_log(self, logf, counter: list[int], *, timed_out: bool, returncode: int | None) -> None:
        self._log("build_stream_close", lines=counter[0], timed_out=timed_out, returncode=returncode)
        if logf is not None:
            with contextlib.suppress(Exception):
                logf.close()

    def _log(self, kind: str, **fields) -> None:
        if self.log is not None:
            self.log.event(kind, **fields)

    # -- read-only inspection (always executes) -------------------------------
    def _env_map(self, *, refresh: bool = False) -> dict[str, str]:
        """Return ``{env_name: prefix_path}`` from ``conda env list --json``."""
        if self._env_cache is not None and not refresh:
            return self._env_cache
        res = self._exec(
            ["env", "list", "--json"], mutating=False, timeout=constants.CONDA_ENVLIST_TIMEOUT_SEC, check=False
        )
        mapping: dict[str, str] = {}
        if res.ok:
            try:
                # `... or []` (not `.get("envs", [])`): the default applies only when the key is ABSENT,
                # so `{"envs": null}` would return None and the `for pfx in prefixes` loop below would
                # raise an uncaught TypeError (not in the except tuple), crashing every _env_map caller.
                # Matches the repo-wide `... or []` coalescing idiom (specs.py `d.get("functions") or []`).
                prefixes = json.loads(res.stdout).get("envs") or []
            except (json.JSONDecodeError, AttributeError):
                prefixes = []
            # Resolve the base prefixes ONCE — not per env. (Doing it in the loop
            # re-spawned `find_conda` for every one of ~120 envs.)
            base_prefixes = _conda_base_prefixes(self.exe)
            for pfx in prefixes:
                name = os.path.basename(pfx.rstrip("/")) or pfx
                # `base` prefix has no trailing env dir; label it explicitly.
                if pfx.rstrip("/") in base_prefixes:
                    name = "base"
                mapping.setdefault(name, pfx)
        self._env_cache = mapping
        return mapping

    def env_list(self, *, refresh: bool = False) -> list[str]:
        return sorted(self._env_map(refresh=refresh))

    def env_exists(self, name: str, *, refresh: bool = False) -> bool:
        return name in self._env_map(refresh=refresh)

    def env_prefix(self, name: str, *, refresh: bool = False) -> str | None:
        return self._env_map(refresh=refresh).get(name)

    def _invalidate(self) -> None:
        self._env_cache = None

    # -- mutating operations (gated by dry_run) -------------------------------
    # Every mutating call drops the env cache in a ``finally`` (hunt 2026-09-30, u36-setup-install-4): a
    # ``check=True`` failure or a timeout raises out of ``_exec`` AFTER conda may have written a partial
    # prefix, and an invalidate placed after ``_exec`` never ran — so ``env_exists`` kept answering from
    # the pre-call list, ``build()``'s partial-prefix cleanup skipped the leftover, and the next
    # strategy died on "prefix already exists".
    def create_named(self, name: str, *, python: str = "3.11", extra: list[str] | None = None) -> RunResult:
        args = ["create", "-y", "-n", name, f"python={python}", *(extra or [])]
        try:
            return self._exec(
                args, mutating=True, timeout=constants.CONDA_CREATE_TIMEOUT_SEC, label=f"creating env {name}"
            )
        finally:
            self._invalidate()

    def create_from_yaml(
        self,
        yaml_path: str,
        *,
        name: str | None = None,
        check: bool = True,
        stream_log_path: str | os.PathLike[str] | None = None,
    ) -> RunResult:
        """``conda env create -f <yaml> [-n <name>]``.

        ``check`` defaults ``True`` (a non-zero exit raises ``CondaError``). The base-env caller
        (:func:`base_env.create_base`, N12) and a recipe rebuild whose pip section pins off-index
        wheels both pass ``check=False`` so the FULL pip stderr (``No matching distribution found
        for …+cpu``) comes back as ``ok=False`` instead of being truncated into an 8-line
        ``CondaError`` — a friendly phase-fail, and the signature the provision-phase remediation
        classifier needs to recognize an off-index stack. ``stream_log_path`` (opt-in) tees this build live to that log when
        ``self.stream_build_logs`` is set; the returned RunResult is identical either way."""
        args = ["env", "create", "-f", str(yaml_path)]
        if name:
            args += ["-n", name]
        label = f"building env {name}" if name else "building env from spec"
        try:
            return self._exec(
                args,
                mutating=True,
                timeout=constants.CONDA_CREATE_TIMEOUT_SEC,
                check=check,
                label=label,
                stream=self.stream_build_logs,
                stream_log_path=stream_log_path,
            )
        finally:
            self._invalidate()

    def clone(self, source: str, target: str, *, stream_log_path: str | os.PathLike[str] | None = None) -> RunResult:
        args = ["create", "-y", "-n", target, "--clone", source]
        try:
            return self._exec(
                args,
                mutating=True,
                timeout=constants.CONDA_CLONE_TIMEOUT_SEC,
                label=f"cloning {source} → {target}",
                stream=self.stream_build_logs,
                stream_log_path=stream_log_path,
            )
        finally:
            self._invalidate()

    def remove_env(self, name: str) -> RunResult:
        """Execute ``conda env remove -n <name>``. Caller MUST have namespace-guarded ``name``.

        C8 belt-and-braces: this primitive is the last gate before an irreversible delete, so it
        **hard-refuses** a :data:`constants.PROTECTED_ENVS` name outright (``base`` / ``spatialomicsgym_e1`` /
        ``spatialomicsgym_env`` / ``sog_reproduce``) — no subprocess is launched — even though every
        current caller already namespace-guards via ``assert_deletable_env``. One forgetful future
        caller can no longer nuke a protected env; the refusal returns a non-ok ``RunResult`` (which
        callers already tolerate from a failed removal), and never raises.
        """
        if name in constants.PROTECTED_ENVS:
            self._log("remove_env_refused", name=name, reason="protected")
            return RunResult(returncode=1, stderr=f"refusing to remove protected env {name!r} (PROTECTED_ENVS)")
        args = ["env", "remove", "-y", "-n", name]
        try:
            return self._exec(
                args,
                mutating=True,
                timeout=constants.CONDA_REMOVE_TIMEOUT_SEC,
                check=False,
                label=f"removing env {name}",
            )
        finally:
            self._invalidate()

    def pip_install(
        self,
        name: str,
        pip_args: list[str],
        *,
        timeout: int | None = None,
        check: bool = True,
        stream_log_path: str | os.PathLike[str] | None = None,
    ) -> RunResult:
        """``conda run -n <name> pip install <pip_args…>``.

        ``check`` defaults to ``True`` (a non-zero exit raises ``CondaError``, unchanged for
        existing callers). The self-heal repair path passes ``check=False`` so a genuine pip
        failure returns ``ok=False`` and can escalate to RECREATE instead of raising.
        ``stream_log_path`` (opt-in) tees this install live when ``self.stream_build_logs`` is set."""
        args = ["run", "-n", name, "python", "-m", "pip", "install", *pip_args]
        return self._exec(
            args,
            mutating=True,
            timeout=timeout or constants.PIP_INSTALL_TIMEOUT_SEC,
            check=check,
            label=f"pip installing into {name}",
            stream=self.stream_build_logs,
            stream_log_path=stream_log_path,
        )

    def conda_install(
        self, name: str, packages: list[str], *, channels: list[str] | None = None, check: bool = True
    ) -> RunResult:
        """``conda install -y -n <name> [-c <chan>…] <packages…>`` — install conda-layer
        packages into an EXISTING managed env.

        Mutating (dry-run-gated like every other build call), so a ``--dry-run`` wizard only
        logs the command. Caller MUST have namespace-checked ``name`` to ``<basic>_*``. Used by
        the gated LLM remediation planner for the rare system-library fault a pip install can't
        fix; ``check`` defaults ``True`` but the planner passes ``check=False`` so a failed install
        returns ``ok=False`` and the authoritative health re-check (not an exception) decides."""
        chans: list[str] = []
        for c in channels or []:
            chans += ["-c", c]
        args = ["install", "-y", "-n", name, *chans, *packages]
        try:
            return self._exec(
                args,
                mutating=True,
                timeout=constants.CONDA_CREATE_TIMEOUT_SEC,
                check=check,
                label=f"conda installing into {name}",
            )
        finally:
            self._invalidate()

    def r_install(
        self,
        name: str,
        packages: list[str],
        *,
        bioc: bool = True,
        timeout: int | None = None,
        check: bool = False,
    ) -> RunResult:
        """``conda run -n <name> Rscript -e '<install expr>'`` — install R/Bioconductor packages into
        an EXISTING managed env.

        This is the **guarded** R-install primitive: it is ``mutating=True`` (so the dry-run gate
        short-circuits it, exactly like ``pip_install``/``conda_install``) — an R install must NOT slip
        past the gate through the read-only :meth:`run`. Caller MUST have namespace-checked ``name`` to
        ``<basic>_*`` (envdoctor's ``MISSING_R_PACKAGE`` repair does, via ``assert_deletable_env``).

        ``bioc=True`` uses ``BiocManager::install`` (bootstrapping BiocManager from CRAN if absent) —
        the right path for Bioconductor packages like ``SpotSweeper``; ``bioc=False`` uses
        ``install.packages`` from CRAN. ``check`` defaults ``False`` so a failed install returns
        ``ok=False`` and the authoritative ``library()`` re-probe (not an exception) decides.

        Every package name is validated against a strict R-identifier regex BEFORE it is interpolated
        into the ``Rscript -e`` string, so a name can never inject R/shell syntax."""
        for p in packages:
            if not _R_PKG_RE.match(p):
                raise CondaError(f"unsafe R package name {p!r}")
        quoted = ", ".join(f'"{p}"' for p in packages)
        if bioc:
            r_expr = (
                'if (!requireNamespace("BiocManager", quietly=TRUE)) '
                'install.packages("BiocManager", repos="https://cloud.r-project.org"); '
                f"BiocManager::install(c({quoted}), update=FALSE, ask=FALSE)"
            )
        else:
            r_expr = f'install.packages(c({quoted}), repos="https://cloud.r-project.org")'
        args = ["run", "-n", name, "Rscript", "-e", r_expr]
        return self._exec(
            args,
            mutating=True,
            timeout=timeout or constants.CONDA_CREATE_TIMEOUT_SEC,
            check=check,
            label=f"R-installing into {name}",
        )

    # -- generic in-env execution (read-only by contract) ---------------------
    def run(
        self,
        name: str,
        argv: list[str],
        *,
        timeout: int | None = None,
        check: bool = False,
    ) -> RunResult:
        """``conda run -n <name> <argv…>`` — used for probes and worker tests."""
        args = ["run", *no_capture_output_args(self.exe), "-n", name, *argv]
        return self._exec(args, mutating=False, timeout=timeout or constants.CONDA_RUN_TIMEOUT_SEC, check=check)

    # -- capture helpers (read-only) ------------------------------------------
    def export_history(self, name: str) -> str:
        """Portable conda layer via ``conda env export --from-history``."""
        res = self._exec(
            ["env", "export", "-n", name, "--from-history"],
            mutating=False,
            timeout=constants.CONDA_EXPORT_TIMEOUT_SEC,
            check=False,
        )
        return res.stdout if res.ok else ""

    def export_full(self, name: str) -> str:
        res = self._exec(
            ["env", "export", "-n", name], mutating=False, timeout=constants.CONDA_EXPORT_TIMEOUT_SEC, check=False
        )
        return res.stdout if res.ok else ""

    def pip_freeze(self, name: str) -> str:
        res = self.run(name, ["python", "-m", "pip", "freeze"], timeout=constants.CONDA_EXPORT_TIMEOUT_SEC, check=False)
        return res.stdout if res.ok else ""


def _conda_base_prefixes(exe: str | None = None) -> set[str]:
    """Best-effort set of prefixes that represent the conda ``base`` env.

    ``exe`` is the already-resolved manager path (``Conda.exe``); pass it to avoid
    re-probing. Only when it is ``None`` (a standalone caller) do we fall back to
    :func:`find_conda`, so the hot ``_env_map`` path never spawns a subprocess.
    """
    out = set()
    # Not ``CONDA_PREFIX_1`` (hunt 2026-09-30, u36-setup-install-16): conda's activator stores whatever
    # env was active at shell level 1 there, which is base only when base was auto-activated. With
    # ``auto_activate_base`` off, ``conda activate sog`` then ``conda activate other`` leaves the ``sog``
    # prefix in it, so it was relabelled "base", dropped by ``_env_map``'s setdefault, and ``env_exists``
    # answered False for an env that exists.
    for var in ("CONDA_ROOT", "MAMBA_ROOT_PREFIX"):
        v = os.environ.get(var)
        if v:
            out.add(v.rstrip("/"))
    if exe is None:
        exe, _ = find_conda()
    # Derive from the manager exe path: /opt/conda/bin/conda -> /opt/conda
    if exe and os.path.basename(os.path.dirname(exe)) == "bin":
        out.add(os.path.dirname(os.path.dirname(exe)).rstrip("/"))
    # Also derive the base from the running interpreter's prefix — handles a standalone/symlinked
    # manager whose exe is not under <root>/bin (where the exe-path rule above misses it).
    base = os.path.dirname(constants.conda_envs_root().rstrip("/"))
    if base:
        out.add(base.rstrip("/"))
    return out
