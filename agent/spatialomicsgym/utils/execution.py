import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from spatialomicsgym.utils.pandas_display import WIDTH as _READER_WIDTH

_UTF8_R_LOCALE = "C.UTF-8"


def _is_utf8_locale(value: str) -> bool:
    """Does this locale name request a UTF-8 charmap? ``C.UTF-8`` and ``en_US.utf8`` both do."""
    return "utf-8" in value.lower().replace("utf8", "utf-8")


def _r_child_env() -> dict[str, str]:
    """The caller's environment, with a UTF-8 ctype guaranteed for an R child.

    ``LC_ALL=C`` is the default in minimal containers, cron, systemd units and most Slurm jobs, and
    it reaches the child verbatim because subprocess inherits ``os.environ``. Python is immune --
    PEP 540 turns on UTF-8 mode under any non-UTF-8 locale -- but R has no equivalent: under a C
    ctype R marks strings ``Encoding() == "unknown"`` and ``print()`` renders every non-ASCII byte
    as an octal escape. ``run_r_code`` hands the child's stdout straight back to the ReAct loop, so
    ``print(head(markers))`` returns ``M\\303\\274ller glia`` where the annotation says ``Müller
    glia``. ``cat()`` passes the bytes through unharmed, so a single block can print one gene both
    ways. Nothing errors; the agent simply reasons about a name that exists in no nomenclature.

    A deployment that already speaks UTF-8 is left exactly as the operator set it -- only a
    non-UTF-8 (or absent) locale is overridden. The override has to land on ``LC_ALL`` because
    POSIX precedence puts it above both ``LC_CTYPE`` and ``LANG``; measured, ``LC_ALL=C`` plus
    ``LC_CTYPE=C.UTF-8`` still corrupts.

    ``SOG_WORKER_LOCALE`` names the replacement for hosts that lack ``C.UTF-8`` (macOS, older
    glibc) -- the same knob the MCP worker door uses, because it is the same question.

    This mirrors ``tools/base_mcp.py::_worker_env``, which covers the other door (a ``*_worker.R``
    launched by its MCP portal). The two cannot share an import: ``base_mcp`` is loaded inside each
    per-tool conda env, none of which has ``spatialomicsgym`` installed, so it has to stay
    dependency-free. ``test/test_the_agents_own_r_block_survives_a_c_locale.py`` drives both over
    one table so they cannot drift apart.
    """
    env = dict(os.environ)
    for var in ("LC_ALL", "LC_CTYPE", "LANG"):
        value = env.get(var, "")
        if not value:
            continue  # POSIX: an empty value defers to the next category
        if _is_utf8_locale(value):
            return env
        break  # the winning category is not UTF-8, and nothing below it can rescue the child
    env["LC_ALL"] = env.get("SOG_WORKER_LOCALE", "").strip() or _UTF8_R_LOCALE
    return env


#: Seconds a timed-out cell gets between SIGTERM and SIGKILL -- and, after the SIGKILL, the most
#: that is spent waiting for it to be gone.
KILL_GRACE_SECONDS = 2.0

#: Every process a shell/R cell starts carries ``SOG_CELL=<id>[:<id>...]`` -- the ids of the cells it
#: belongs to, innermost last -- so a timeout can find all of them, including one that has left the
#: tree: a double fork, a ``setsid``, a ``nohup ... &`` whose parent has already exited.
CELL_ENV_VAR = "SOG_CELL"

_HAVE_PROC = os.path.isdir("/proc/self")

# The cells a shell/R runner is waiting on, by the thread that waits. ``run_with_timeout`` abandons
# that thread at the budget, and on Ctrl-C; it ends these first, so the observation that says the
# step was killed is true by the time it is returned.
_cells_by_thread: dict[int, list["_Cell"]] = {}
_cells_lock = threading.Lock()


class _Cell:
    """One shell/R child: its ``Popen``, the id its processes carry, and whether it is over.

    ``over`` is set once the cell has finished inside its budget, or has been ended. From then on
    ending it is a no-op -- whatever a finished cell left running it left on purpose, and an entry
    a thread failed to unregister must not reach anything later.
    """

    __slots__ = ("proc", "cell_id", "over")

    def __init__(self, proc: subprocess.Popen, cell_id: str):
        self.proc, self.cell_id, self.over = proc, cell_id, False


def _carries(pid: int, cell_id: bytes) -> bool:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            environ = f.read()
    except OSError:  # gone, another user's, or not dumpable: found by descent or not at all
        return False
    prefix = CELL_ENV_VAR.encode() + b"="
    return any(
        entry.startswith(prefix) and cell_id in entry[len(prefix) :].split(b":") for entry in environ.split(b"\0")
    )


def _members(cell: _Cell) -> set[int]:
    """The live processes of ``cell``: those carrying its id, its leader until reaped, everything below.

    One pass over ``/proc``, so a parent link is read before a signal can break it. A leader that has
    been reaped is left out: its pid may already be someone else's. Never this process.
    """
    cell_id = cell.cell_id.encode()
    leader = cell.proc.pid if cell.proc.returncode is None else None
    children: dict[int, list[int]] = {}
    roots = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        state, ppid = stat[stat.rindex(b")") + 2 :].split()[:2]
        if state in (b"Z", b"X"):  # exited; only waiting to be collected
            continue
        children.setdefault(int(ppid), []).append(pid)
        if pid == leader or _carries(pid, cell_id):
            roots.append(pid)
    found: set[int] = set()
    while roots:
        pid = roots.pop()
        if pid not in found:
            found.add(pid)
            roots.extend(children.get(pid, ()))
    found.discard(os.getpid())
    return found


def _signal(pids, sig) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _gone_within(cell: _Cell, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not _members(cell):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _end_cell(cell: _Cell, grace: float = KILL_GRACE_SECONDS) -> None:
    """SIGTERM every process of ``cell``, SIGKILL whatever is left after ``grace``, wait for it to go.

    Returns once none is left, or ``grace`` after the SIGKILL (a process in uninterruptible IO can
    outlast even that). Safe from any thread, and more than once. Without ``/proc`` to find the
    cell's processes in, it kills the leader alone -- what ``subprocess.run`` did.
    """
    if cell.over:
        return
    if not _HAVE_PROC:
        cell.proc.kill()
        cell.over = True
        return
    _signal(_members(cell), signal.SIGTERM)
    if not _gone_within(cell, grace):
        # Freeze before the kill, so nothing can fork, or be reparented out of sight, in between.
        frozen: set[int] = set()
        for _ in range(20):
            fresh = _members(cell) - frozen
            if not fresh:
                break
            _signal(fresh, signal.SIGSTOP)
            frozen |= fresh
        _signal(frozen, signal.SIGKILL)
        _gone_within(cell, grace)
    cell.over = True


def _cells_waited_on_by(thread_id: int | None) -> list[_Cell]:
    with _cells_lock:
        return list(_cells_by_thread.get(thread_id, ()))


def _run_cell(args, *, timeout: float | None, env: dict[str, str], **popen_kwargs):
    """``subprocess.run(args, capture_output=True, timeout=timeout, env=env, **popen_kwargs)``, except
    that a child which has to be stopped is stopped with everything it started.

    ``subprocess.run`` kills only the process it launched -- for ``run_bash_script``, the shell in
    front of the script -- so a pipeline, a backgrounded job or a ``python - <<PY`` went on running
    after the observation told the model the step was killed, and could still write its outputs,
    ``eval_answer.json`` included. E-03 (SPATIAL07 r1) ran ``find / ... | head -50`` into its
    1800 s budget, and that kill reached the shell alone. Here a timeout (or ``run_with_timeout``'s
    SystemExit, or a Ctrl-C in this thread) ends every process of the cell before the exception
    goes on.

    The child stays in the CALLER's process group, as it always was. Whatever ends the agent as a
    group -- a trial supervisor's ``killpg``, coreutils ``timeout``, a terminal hangup, the REPL
    client's SIGKILL of its worker -- has to reach a cell that is still running, and a group or
    session of the cell's own is exactly what those kills cannot reach. So the cell's processes
    are found by the id they carry (``CELL_ENV_VAR``) and by descent, not by a group.
    """
    cell_id = os.urandom(8).hex()
    env = dict(env)
    env[CELL_ENV_VAR] = f"{env[CELL_ENV_VAR]}:{cell_id}" if env.get(CELL_ENV_VAR) else cell_id
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, **popen_kwargs) as proc:
        cell = _Cell(proc, cell_id)
        me = threading.get_ident()
        with _cells_lock:
            _cells_by_thread.setdefault(me, []).append(cell)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            cell.over = True
        except BaseException:
            _end_cell(cell)
            raise
        finally:
            with _cells_lock:
                mine = _cells_by_thread.get(me, [])
                if cell in mine:
                    mine.remove(cell)
                if not mine:
                    _cells_by_thread.pop(me, None)
        retcode = proc.poll()
    return subprocess.CompletedProcess(proc.args, retcode, stdout, stderr)


# Add these new functions for running R code and CLI commands
def _rscript() -> str | None:
    """The Rscript a ``#!R`` cell runs with: ``SOG_RSCRIPT`` when set and executable, else PATH's."""
    import shutil

    chosen = os.environ.get("SOG_RSCRIPT", "").strip()
    if chosen and os.path.isfile(chosen) and os.access(chosen, os.X_OK):
        return chosen
    return shutil.which("Rscript")


def run_r_code(code: str, timeout: float | None = None) -> str:
    """Run R code using subprocess.

    Args:
        code: R code to run
        timeout: Optional wall-clock limit (seconds) for the Rscript subprocess. When set, an
            overrunning process is killed -- with every process it started -- rather than
            orphaned; ``None`` (default) keeps the previous unbounded behavior. See ``_run_cell``.

    Returns:
        Output of the R code

    """
    temp_file = None
    rscript = _rscript()
    if rscript is None:
        # Said as what it is. The raw `[Errno 2] No such file or directory: 'Rscript'` read as a
        # broken cell, so the model retried the same R step until the repeated-error guard ended
        # the turn -- on boxes whose agent env simply has no R (u12-react-17). The per-tool R
        # methods run in their own envs through the tool functions, which still work.
        return (
            "Error running R code: R is not installed for the agent itself on this machine (no "
            "Rscript on PATH, and SOG_RSCRIPT is not set), so #!R cells cannot run here. R-based "
            "methods still run through their tool functions in a python cell; do this step in python "
            "or through a tool instead of retrying it in R."
        )
    try:
        # Create a temporary file to store the R code
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run the R code using Rscript. env= carries a UTF-8 ctype so R's print() does not hand the
        # agent octal escapes where the data has gene and cell-type names -- see _r_child_env.
        result = _run_cell(
            [rscript, temp_file],
            text=True,
            errors="replace",
            timeout=timeout,
            env=_r_child_env(),
        )

        # Return the output
        if result.returncode != 0:
            # Keep any stdout emitted before the failure — an R script that cat()s progress or
            # partial results then stop()s must still hand that context to the agent (the Python-REPL
            # executor does the same). stderr alone loses *where* the run got to.
            out = result.stdout or ""
            tail = f"\n[stdout before error]\n{out}" if out.strip() else ""
            return f"Error running R code:\n{result.stderr}{tail}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running R code: {str(e)}"
    finally:
        # Remove the temp file even when Rscript is absent (FileNotFoundError) or times out.
        if temp_file is not None and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass


#: Holds the ``sitecustomize`` that gives a Python started from a bash cell the backed-indexing shim.
CHILD_SITE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "child_site")


def run_bash_script(script: str, timeout: float | None = None) -> str:
    """Run a Bash script using subprocess.

    Args:
        script: Bash script to run
        timeout: Optional wall-clock limit (seconds) for the script subprocess. When set, an
            overrunning script is killed -- pipeline, background jobs and all -- rather than
            orphaned; ``None`` (default) keeps the previous unbounded behavior. See ``_run_cell``.

    Returns:
        Output of the Bash script

    Example:
        This is how to use the function

        .. code-block:: python

            # Example of a complex Bash script
            script = '''
            #!/bin/bash

            # Define variables
            DATA_DIR="/path/to/data"
            OUTPUT_FILE="results.txt"

            # Create output directory if it doesn't exist
            mkdir -p $(dirname $OUTPUT_FILE)

            # Loop through files
            for file in $DATA_DIR/*.txt; do
                echo "Processing $file..."
                # Count lines in each file
                line_count=$(wc -l < $file)
                echo "$file: $line_count lines" >> $OUTPUT_FILE
            done

            echo "Processing complete. Results saved to $OUTPUT_FILE"
            '''
            result = run_bash_script(script)
            print(result)

    """
    temp_file = None
    try:
        # Trim any leading/trailing whitespace
        script = script.strip()

        # If the script is empty, return an error
        if not script:
            return "Error: Empty script"

        # Create a temporary file to store the Bash script
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            # The script's own shebang, if it has one, must stay the FIRST line. ``set -e`` used to be
            # written above it, so the kernel saw no shebang and /bin/sh (dash) ran a #!/bin/bash or
            # #!/usr/bin/env zsh cell -- bash-only syntax then failed (hunt 2026-09-30, u12-react-2).
            if script.startswith("#!"):
                shebang, _, body = script.partition("\n")
            else:
                shebang, body = "#!/bin/bash", script
            f.write(shebang + "\n")
            # Exit on the first error -- for a shell only; a #!/usr/bin/env python cell is not one.
            is_shell = re.search(r"\b(ba|z|k|da)?sh\b", shebang) is not None
            if is_shell and "set -e" not in script:
                f.write("set -e\n")
            f.write(body)
            temp_file = f.name

        # Make the script executable
        os.chmod(temp_file, 0o755)

        # Get current environment variables and working directory
        env = os.environ.copy()
        cwd = os.getcwd()
        # FM-03 (L-1b): a Python the script starts gets the backed-indexing shim the REPL has.
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (CHILD_SITE_DIR, *env.get("PYTHONPATH", "").split(os.pathsep)) if p
        )
        # N5: the script's stdout is a pipe the model reads, so a program sizing its output to "the
        # terminal" -- pandas under ``python -I``, which skips the hook above -- gets the display
        # width the REPL uses, not shutil's 80-column fallback or the width of whatever terminal
        # launched the agent. Set, not defaulted; a script that exports its own still wins.
        env["COLUMNS"] = str(_READER_WIDTH)

        # Run the Bash script with the current environment and working directory
        result = _run_cell(
            [temp_file],
            shell=True,
            text=True,
            errors="replace",  # non-UTF-8 tool output must not crash decode even on a successful (rc=0) run
            env=env,
            cwd=cwd,
            timeout=timeout,
        )

        # Return the output
        if result.returncode != 0:
            traceback.print_stack()
            print(result)
            # Keep stdout emitted before the non-zero exit (echoed progress / partial counts) so the
            # agent's observation shows *where* the script failed, not just stderr.
            out = result.stdout or ""
            tail = f"\n[stdout before error]\n{out}" if out.strip() else ""
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}{tail}"
        else:
            return result.stdout
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"
    finally:
        # Remove the temp file even when the interpreter is missing or the script times out.
        if temp_file is not None and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass


# Keep the run_cli_command for backward compatibility
def run_cli_command(command: str) -> str:
    """Run a CLI command using subprocess.

    Args:
        command: CLI command to run

    Returns:
        Output of the CLI command

    """
    try:
        # Trim any leading/trailing whitespace
        command = command.strip()

        # If the command is empty, return an error
        if not command:
            return "Error: Empty command"

        # Split the command into a list of arguments, handling quoted arguments correctly
        import shlex

        args = shlex.split(command)

        # Run the command
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", check=False)

        # Return the output
        if result.returncode != 0:
            return f"Error running command '{command}':\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running command '{command}': {str(e)}"


def timeout_message(timeout, *, still_running: bool = False) -> str:
    """The observation a cell gets when it ran past its budget. One text, two callers.

    The old tail here read "try with simpler inputs" -- and the step that actually hits this
    budget is a full-dataset tool call, where "simpler inputs" is an invitation to subsample
    the data. That is the one remedy this platform forbids. Name the real ones instead, in
    the order the ReAct loop can act on them. The "ERROR: Code execution timed out" prefix is
    load-bearing: the failed-execution detector in agent/execution.py keys on it.

    Factored out of :func:`run_with_timeout` so the REPL worker client (``tool/repl_client.py``),
    which kills its worker on the same budget, says the same words -- plus one sentence of its
    own about the restart, because there the variables really are gone. The in-process REPL adds
    the opposite sentence, :data:`TIMEOUT_KEPT_NOTE`, because there nothing was restarted. Both
    reach it through :func:`timeout_observation`, which puts what the cell had printed in front of it.
    """
    # `still_running`: the in-process runner can only ASK a thread to stop, and the request is not
    # delivered while it sits in a C call or waits on an MCP tool. Telling the model the step was
    # "killed" and to re-run it then started a second full-dataset fit racing the first in one
    # namespace and one output folder (u12-react-20).
    first = (
        "The step could NOT be stopped: it is still running in the background, and any files it is "
        "writing are incomplete until it finishes. Do not re-run it now -- a second copy would race "
        "the first in the same session and output folder; check what it wrote in a later cell instead. "
        if still_running
        else "Any files the killed step was mid-writing are incomplete - re-run that step rather "
        "than reusing its outputs. "
    )
    return (
        f"ERROR: Code execution timed out after {timeout} seconds. "
        + first
        + "Do NOT subsample or shrink the data to fit the budget. "
        "Run one long tool call per execute block so it gets the whole budget, and if the "
        "full-dataset step legitimately needs longer, say so in your answer: the budget is "
        "raised via STCoscientist(timeout_seconds=...), the --timeout CLI flag, or the "
        "SOG_TIMEOUT_SECONDS environment variable."
    )


#: Said after :func:`timeout_message` when a python cell times out in THIS process (process isolation
#: off -- the scored path): the execute node passes it as ``run_with_timeout(..., after=...)``. The
#: counterpart of ``repl_client.TIMEOUT_RESTART_NOTE``, and its opposite. Nothing is restarted here:
#: the cell's thread is abandoned and sent a ``SystemExit``, and the namespace it was writing into is
#: the one the next cell runs in, so every name bound before the deadline is still bound.
#:
#: The message alone read as the opposite. E-06, curio_ovary_batch_driven_clustering r1-r3: each cell
#: printed that its Harmony embedding was done and then ran out of 1,800 s inside Leiden. Told only
#: "re-run that step rather than reusing its outputs" -- a sentence about FILES -- r2 listed as unknown
#: "whether the in-memory state from the previous Python execution persisted", and all three
#: recomputed the embedding the session still held (r1 and r3 from the raw file). The file warning
#: stands. What this adds is the rest of what is true, including what the unfinished step leaves
#: unsafe, and that a call it was blocked in is not interrupted: ``PyThreadState_SetAsyncExc`` lands
#: only when the thread next runs Python code, and the only processes ended at the deadline are those
#: ``run_bash_script`` / ``run_r_code`` started on the cell's thread (see :func:`_run_cell`).
TIMEOUT_KEPT_NOTE = (
    " The Python session was NOT restarted: every variable bound before the deadline -- by earlier "
    "cells, and by this cell up to where it stopped -- is still defined, and an object whose steps "
    "printed their completion still carries their results, so those steps need no re-run (print a "
    "shape or a key to check). Only what the unfinished step touched is unsafe: a name it was "
    "assigning keeps its previous value, if it had one; an object it was changing may be "
    "half-updated; files it was mid-writing are still incomplete; and a call it was blocked in "
    "(compiled code, a subprocess) may run on in the background until that call returns."
)


# --------------------------------------------------------------------------------------------- #
# What a cell had printed when its budget ran out.
#
# The cell's stdout is captured in a buffer local to ``PythonREPL.run``, so the thread that gives
# up on it at the deadline has no way to read it -- and it used to return the timeout message
# alone. Measured on E-03 (curio_ovary_batch_driven_clustering r2): the cell had printed "Harmony
# done" and was inside Leiden when 1,800 s ran out; the model was told only that the step timed
# out, treated the finished embedding as lost, and answered wrong. The exception and SystemExit
# paths already kept what the cell printed; this is the timeout path keeping it too.
#
# A slot per running function, on the thread that runs it: ``run_with_timeout`` opens one in the
# thread it starts and the REPL worker opens one around each cell (``repl_host.Host.op_exec``);
# ``PythonREPL`` publishes a reader into it and nothing else does. A function that publishes
# nothing leaves the slot empty, and its timeout observation is exactly what it always was.
#
# The observation is built to pass ``clip_observation`` whole: clipped, it would be headed "The
# code already ran to completion; this is only a display limit", which for a cell that never
# finished is false. The reader is asked for what fits (:func:`partial_room`), and
# :func:`timeout_observation` holds the whole to the limit whatever the reader hands back.
# --------------------------------------------------------------------------------------------- #
_running = threading.local()

#: What ``clip_observation`` (``agent/execution.py``) lets through unclipped.
OBSERVATION_CHARS = 10000


@contextlib.contextmanager
def partial_output_slot(slot: dict | None = None):
    """Open ``slot`` (a fresh dict by default) for the code about to run on THIS thread."""
    slot = {} if slot is None else slot
    previous = getattr(_running, "slot", None)
    _running.slot = slot
    try:
        yield slot
    finally:
        _running.slot = previous


def publish_partial_output(reader):
    """Say how to read what the code running here has printed so far: ``reader(room)`` returns it
    in at most ``room`` characters, or unbounded for ``None``. Returns the reader it replaced
    (``None`` if none), so a nested REPL call can put the outer one back. A no-op outside a slot."""
    slot = getattr(_running, "slot", None)
    if slot is None:
        return None
    previous = slot.get("reader")
    slot["reader"] = reader
    return previous


def read_partial_output(slot, room: int | None = None) -> str:
    """What the published reader returns now for ``room``, or ``""``. Never raises: it runs on the
    timeout branch, and nothing there may cost the model the timeout message itself."""
    reader = (slot or {}).get("reader")
    if reader is None:
        return ""
    try:
        return str(reader(room) or "")
    except Exception:
        return ""


def abandoned_by_its_caller() -> bool:
    """True on a thread ``run_with_timeout`` has given up on.

    Its caller has moved on and has been handed ``sys.stdout`` back, so this thread no longer owns
    it: code that saved a stream on the way in (``mcp_integration._put_back_stdio``) must not put
    it back on the way out, or the stream it restores is a buffer nobody reads -- or the capture
    of the step that is running NOW.
    """
    slot = getattr(_running, "slot", None)
    return bool(slot and slot.get("abandoned"))


def _still_running_line(timeout, reported_every: float | None) -> str:
    when = "The output above is what it printed before the deadline"
    if reported_every is not None:
        when += (
            f", as of the worker's last report (sent every {reported_every:g} s while the output changes and "
            f"the cell lets the reporter run -- a long call into compiled code can hold it off -- so the end "
            f"of it may be missing)"
        )
    return (
        f"[timeout] The cell was still running when its {timeout}-second budget ran out. "
        f"{when}; the step it was in at the deadline did not finish, and whatever came after its last "
        f"printed line may not have run."
    )


def partial_room(timeout, *, reported_every: float | None = None, after: str = "") -> int:
    """How much of what a cell printed :func:`timeout_observation` can carry, in characters: what
    :data:`OBSERVATION_CHARS` leaves once the ``[timeout]`` line, the message and ``after`` have
    theirs. The number a reader is asked to fit."""
    # The longer of the two messages: whether the step could be stopped is known only after the read.
    message = max(len(timeout_message(timeout)), len(timeout_message(timeout, still_running=True)))
    framing = len(_still_running_line(timeout, reported_every)) + message + len(after)
    return OBSERVATION_CHARS - framing - 2  # the newlines either side of the [timeout] line


#: Said in place of the start of a partial that came back longer than its room.
PARTIAL_CUT_NOTE = (
    "[... the start of what this cell printed is not shown: it was longer than an observation can carry ...]\n"
)


def timeout_observation(
    partial: str, timeout, *, reported_every: float | None = None, after: str = "", still_running: bool = False
) -> str:
    """The observation for a cell that ran past its budget: what it printed, then the message.

    A cell that printed nothing gets :func:`timeout_message` alone (then ``after``), byte for byte
    as before. One that printed something gets that first, then one line saying the cell was still
    running, then the message unchanged -- so the prefix the failed-execution detector keys on is
    still there, and still the last thing said before ``after``.

    ``after`` is the caller's own closing sentence (the worker's restart note), and it counts
    against the limit. ``reported_every`` is for the worker, whose client holds only the last
    report the worker sent before it was killed: the line then says how stale that can be rather
    than claiming the deadline.

    Never longer than :data:`OBSERVATION_CHARS`. ``PythonREPL`` fits what it hands over to
    :func:`partial_room` itself, keeping the notices and warnings and giving up tail lines; a
    partial that comes back longer anyway (another reader, or a report a cell forged) keeps its end
    -- where the cell had got to -- behind :data:`PARTIAL_CUT_NOTE`.
    """
    message = timeout_message(timeout, still_running=still_running) + after
    partial = (partial or "").rstrip("\n")
    if not partial.strip():
        return message
    room = partial_room(timeout, reported_every=reported_every, after=after)
    if len(partial) > room:
        keep = max(0, room - len(PARTIAL_CUT_NOTE))
        partial = PARTIAL_CUT_NOTE + partial[len(partial) - keep :]
    return f"{partial}\n{_still_running_line(timeout, reported_every)}\n{message}"


def _interrupt_thread(thread, exc_type: type[BaseException]) -> None:
    """Raise ``exc_type`` inside ``thread``. Best effort: not delivered while it is inside a C call."""
    import ctypes

    try:
        # By ident, not ``is_alive()``: a ^C that interrupts ``join`` makes CPython release the
        # thread's state lock and mark it stopped while it is still running (the bpo-45274 path in
        # ``_wait_for_tstate_lock``), so ``is_alive()`` answers False for a live cell. An ident whose
        # thread really has ended is refused by the interpreter (``res == 0``) and costs nothing.
        thread_id = thread.ident
        if thread_id:
            # This is a bit dangerous and not 100% reliable
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(exc_type))
            if res > 1:
                # Oops, we raised too many exceptions
                ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
    except Exception as e:
        print(f"Error trying to terminate thread: {e}")


def run_with_timeout(func, args=None, kwargs=None, timeout=600, *, after: str = ""):
    """Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.

    ``after`` is said after the timeout message and only there (:func:`timeout_observation`), inside
    the same :data:`OBSERVATION_CHARS`: the caller's sentence about what the timeout left behind.
    The execute node passes :data:`TIMEOUT_KEPT_NOTE` for a python cell; nothing else here knows
    whether ``func`` keeps a namespace, so the default says nothing.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    import ctypes
    import queue

    result_queue = queue.Queue()
    # Read on the timeout branch, and only there: see ``partial_output_slot``.
    slot: dict = {}
    # The stream this caller was printing to. The cell swaps in its own capture, and at the
    # deadline the abandoned thread still holds it -- so this function's own "TIMEOUT:" line, and
    # everything the caller printed after it (the probe's echo of the observation), went into a
    # buffer nobody would ever read. A later cell then saved THAT buffer as its "old" stdout and
    # restored it on exit, so the loss lasted for the rest of the run.
    entry_stdout = sys.stdout

    def thread_func(func, args, kwargs, result_queue):
        """Function to run in a separate thread."""
        try:
            with partial_output_slot(slot):
                result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except BaseException as e:
            # The TYPE as well as the message. `str(IndexError())` is the empty string, and several
            # common exceptions carry no text at all, so this used to hand the model the observation
            # "Error in execution: " with nothing after it -- an error it cannot diagnose and cannot
            # fix. The worker path already spells it `f"{type(exc).__name__}: {exc}"`; this is the
            # in-process path saying the same thing.
            result_queue.put(("error", f"{type(e).__name__}: {e}" if str(e) else type(e).__name__))

    # Start a separate thread
    thread = threading.Thread(target=thread_func, args=(func, args, kwargs, result_queue))
    thread.daemon = True  # Set as daemon so it will be killed when main thread exits
    thread.start()

    # Wait for the specified timeout. A Ctrl-C lands HERE, not in the thread running the cell. The
    # terminal's SIGINT reaches the cell's processes as well, but a command the script started with a
    # bare `&` inherits SIGINT ignored, so end the cell before the interrupt goes on -- and send the same
    # interrupt into the cell's Python: the CLI printed "turn cancelled" while a python cell carried on
    # writing, and the next question could start a second cell racing the first in one namespace (hunt
    # 2026-09-30, u17-cli-report-7).
    try:
        thread.join(timeout)
    except BaseException as exc:
        for cell in _cells_waited_on_by(thread.ident):
            _end_cell(cell)
        if isinstance(exc, KeyboardInterrupt):
            _interrupt_thread(thread, KeyboardInterrupt)
            # Not ``thread.join``: after the interrupted join the Thread object already reads as
            # stopped. The cell's own report on the queue is what says it has let go.
            try:
                result_queue.get(timeout=2.0)
            except queue.Empty:
                pass
        raise

    # Check if the thread is still running after timeout. Boundary race: a cell that finished right at
    # the deadline may still read is_alive()==True while its result is already queued — only declare a
    # timeout when NO result is available, else fall through and return the just-completed result.
    if thread.is_alive() and result_queue.empty():
        # Hand stdout back BEFORE printing. ``PythonREPL``'s restore is guarded
        # (``if sys.stdout is mystdout``), so the orphan leaves this alone when it unwinds; and the
        # ``abandoned`` flag tells the other code that saved a stream (the MCP wrapper) not to put
        # it back.
        slot["abandoned"] = True
        if sys.stdout is not entry_stdout:
            sys.stdout = entry_stdout
        # Read before the SystemExit below can start the cell unwinding.
        partial = read_partial_output(slot, partial_room(timeout, after=after))
        print(f"TIMEOUT: Code execution timed out after {timeout} seconds")
        cells = _cells_waited_on_by(thread.ident)  # read before the SystemExit lets the thread unregister them

        # Unfortunately, there's no clean way to force terminate a thread in Python
        # The recommended approach is to use daemon threads and let them be killed when main thread exits
        # Here, we'll try to raise an exception in the thread to make it stop
        try:
            # Get thread ID and try to terminate it
            thread_id = thread.ident
            if thread_id:
                # This is a bit dangerous and not 100% reliable
                # It attempts to raise a SystemExit exception in the thread
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                if res > 1:
                    # Oops, we raised too many exceptions
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        except Exception as e:
            print(f"Error trying to terminate thread: {e}")

        # A #!BASH / #!R cell -- the pipeline, not only the shell in front of it -- is gone before the
        # message below says the step was killed. The SystemExit comes first, while the thread is
        # known to be alive: ended first, the runner would see its child die and return like any
        # failed script (the bash one printing a stack), and the SystemExit would go to a thread
        # that has finished -- or to a new thread that got its ident.
        for cell in cells:
            _end_cell(cell)
        # Did it take? A thread inside a native call or an MCP tool's select does not see the
        # exception until it returns to Python, so say which of the two happened (u12-react-20).
        thread.join(2.0)
        still_running = thread.is_alive()
        if still_running:
            # The cell is an orphan now: disown its stdout capture as well, so that when it ends it does
            # not hand a finished turn's stream back as the process's stdout (2026-10-01 gate; rp-u23).
            try:
                from spatialomicsgym.tool import support_tools

                support_tools.abandon_capture_of(thread.ident)
            except Exception:
                pass
        return timeout_observation(partial, timeout, after=after, still_running=still_running)

    # Get the result from the queue if available
    try:
        status, result = result_queue.get(block=True, timeout=5)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"
