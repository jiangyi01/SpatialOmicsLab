"""The manifest: the only thing L2 (review) and L3 (report/webui) are allowed to read.

``docs/design/post_analysis_contract.md`` is the specification; this module is its single
implementation. Two properties matter more than anything else here:

* **every key is always present** -- consumers must never need ``.get()`` with a default;
* **every path is relative to the results dir** -- the manifest has to survive the directory being
  moved, tarred, or served over HTTP from a different root. An absolute path in it is a bug, so
  :meth:`Manifest.add_figure` and :meth:`Manifest.add_table` refuse one.

The write is atomic (tmp + ``os.replace``), so a consumer never reads a half-written manifest. It
happens at every exit from a run *and* once up front: the engine stamps a ``failed`` placeholder the
moment the results directory exists, so a run killed mid-figure leaves a directory that says it was
interrupted rather than one that says nothing at all. Read ``status`` -- finding this file no longer
means the run finished, only that post-analysis owns the directory.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1

#: The directory name L1 writes into when the caller names no ``results_dir``. It lives here, in the
#: stdlib-only schema module, because L3 needs it too: the portal lists runs by directory name, and
#: "the default one is called this" is the difference between a card that identifies a run and 92
#: cards all reading ``post_analysis``. :mod:`spatialomicsgym.postanalysis.engine` re-exports it.
DEFAULT_RESULTS_DIRNAME = "post_analysis"

#: Exactly the seven strings the contract defines, and the contract is the only source for them.
#: The registry is *not*: ``tool_output_registry`` keeps its own ``task_type`` vocabulary, and it
#: emits values outside this tuple (``resolution``, ``visualization`` and ``functional_enrichment``
#: on 2026-09-30, when it held 88 profiles over eight task types; the counts this sentence used to
#: quote were an earlier registry's and had gone stale -- hunt 2026-09-30, u18-postanalysis-5). A
#: registry ``task_type`` is therefore one *input* to detection that may legally come back outside
#: this vocabulary; :func:`engine.run_post_analysis` warns and degrades when it does, and the design
#: contract requires exactly that -- "``resolution`` is a valid registry task type with no
#: post-analysis handler. Treat it, and any other unlisted string, as ``status: "partial"`` with one
#: warning -- not ``"failed"``."
TASK_TYPES: tuple[str, ...] = (
    "spatial_clustering",
    "svg_detection",
    "deconvolution",
    "cell_communication",
    "alignment",
    "imputation",
    "trajectory",
)


def _no_handler_prefix(task_type: str) -> str:
    return f"Task type {task_type!r} has no post-analysis handler"


def no_handler_warning(task_type: str, why: str) -> str:
    """L1's word for "the registry named a task type this contract has no handler for".

    Lives here, beside :data:`TASK_TYPES`, because two layers need the same sentence and they need
    it for opposite reasons: :func:`engine.run_post_analysis` writes it, and
    :func:`review._manifest_checks` reads it to tell a manifest L1 degraded on purpose from one that
    is malformed. A second spelling in the reader would drift the first time either is re-worded,
    and the drift is silent -- the reader simply stops recognising the degrade and goes back to
    calling it a contract violation.
    """
    return (
        f"{_no_handler_prefix(task_type)} ({why}); the output was left unanalysed. The "
        f"{len(TASK_TYPES)} supported task types are: " + ", ".join(TASK_TYPES) + "."
    )


def declined_for_no_handler(payload: dict[str, Any], task_type: str) -> bool:
    """Did L1 record ``task_type`` because it had no handler for it, rather than by accident?

    Anchored to the start of the sentence and to this task type: a warning about ``resolution`` does
    not excuse a manifest that says ``super_resolution``, and a loose paraphrase does not excuse
    anything. Everything after the prefix is the ``why``, which varies per run.
    """
    prefix = _no_handler_prefix(task_type)
    warnings = payload.get("warnings")
    if not isinstance(warnings, (list, tuple)):  # an agent-writable manifest; review never raises on it
        return False
    return any(isinstance(w, str) and w.startswith(prefix) for w in warnings)


#: The vocabulary L3 renders against. A new kind is a contract change, not a local decision.
FIGURE_KINDS: frozenset[str] = frozenset(
    {"spatial_map", "bar", "heatmap", "histogram", "boxplot", "grid", "scatter", "line", "inventory"}
)

STATUS_OK = "ok"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

#: Findings that are post-analysis counting its own scan, not something the tool found.
#: :func:`engine._write_scan` appends these before any task runner has appended anything, so one of
#: them is finding #1 on every run that gets that far -- and every surface that shows "the first few
#: findings" spends a slot on it. Two surfaces do: :func:`next_step._named_findings`, where an SVG
#: run's prompt named the genes *tested* and never the genes found significant, and
#: :func:`review._verdict`, where all 49 recorded ``ok`` runs cite a file count as a reason the
#: result can be trusted -- on the ncem run it is half of the reviewer's whole case.
#:
#: It lives here, in the stdlib-only schema module, for the same reason
#: :data:`DEFAULT_RESULTS_DIRNAME` does: both consumers can reach it and neither can reach the
#: other. ``next_step`` imports ``review``, so ``review`` cannot import ``next_step``, and one list
#: is the point -- this named only ``n_output_files`` for a while, and on the five recorded runs
#: with staged input the freed slot went straight to ``n_staged_inputs``, a count of the files the
#: tool was *handed*. Anything ``_write_scan`` appends belongs here, and
#: ``test_the_prompt_does_not_spend_a_finding_slot_on_the_tools_input`` runs it and fails if it
#: grows a key this set has not been told about.
BOOKKEEPING_FINDINGS: frozenset[str] = frozenset({"n_output_files", "n_staged_inputs"})


@dataclass
class Manifest:
    """The in-memory manifest. Mutated by the task runners, serialised once at the end."""

    tool_name: str | None = None
    task_type: str | None = None
    source_outputs: list[str] = field(default_factory=list)
    status: str = STATUS_OK
    figures: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # ---- mutation -------------------------------------------------------------------

    def warn(self, message: str) -> None:
        """Record a warning once. Repeats are noise for a human and duplicates for L2."""
        text = str(message).strip()
        if text and text not in self.warnings:
            self.warnings.append(text)

    def degrade(self) -> None:
        """ok -> partial. Never upgrades, and never overwrites a hard failure."""
        if self.status == STATUS_OK:
            self.status = STATUS_PARTIAL

    def fail(self, reason: str) -> None:
        self.warn(reason)
        self.status = STATUS_FAILED

    def add_figure(self, path: str, title: str, caption: str, kind: str) -> None:
        self._check_relative(path, "figures/")
        if kind not in FIGURE_KINDS:
            raise ValueError(f"unknown figure kind {kind!r}; contract allows {sorted(FIGURE_KINDS)}")
        _replace_or_append(self.figures, {"path": path, "title": title, "caption": caption, "kind": kind})

    def add_table(self, path: str, title: str, rows: int) -> None:
        self._check_relative(path, "tables/")
        _replace_or_append(self.tables, {"path": path, "title": title, "rows": int(rows)})

    def add_finding(self, key: str, value: Any, label: str) -> None:
        """Record a scalar result. Later writes of the same key replace the earlier one."""
        entry = {"key": key, "value": _jsonable(value), "label": label}
        for i, existing in enumerate(self.findings):
            if existing["key"] == key:
                self.findings[i] = entry
                return
        self.findings.append(entry)

    def has_finding(self, key: str) -> bool:
        return any(f["key"] == key for f in self.findings)

    # ---- serialisation --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "tool_name": self.tool_name,
            "task_type": self.task_type,
            "source_outputs": list(self.source_outputs),
            "status": self.status,
            "figures": list(self.figures),
            "tables": list(self.tables),
            "findings": list(self.findings),
            "warnings": list(self.warnings),
            # L2 owns both of these. L1 writes the empty shape so a consumer can read
            # manifest["review"] without a KeyError before L2 has ever run.
            "review": None,
            "next_steps": [],
        }

    @staticmethod
    def _check_relative(path: str, prefix: str) -> None:
        check_artifact_path(path, prefix)


def check_artifact_path(path: str, prefix: str) -> None:
    """Raise ``ValueError`` unless ``path`` is a ``prefix``-rooted path that stays in the results dir.

    Public because the check has to happen *before* the bytes are written, not when the entry is
    registered afterwards: ``plots.save_figure`` and ``DataFrame.to_csv`` both create the file at
    whatever the name points to, so a check that only ran in ``add_figure``/``add_table`` rejected
    the manifest entry for a file that already existed outside the tree.

    ``..`` is the case that got through. ``tables/../../../x.csv`` is not absolute and does start
    with ``tables/``, and those were the only two tests -- but a prefix says how a path *begins*,
    not where it ends up. L3 refuses ``..`` when reading (``report/manifest.py::safe_subpath``), so
    the result was a file written outside the results dir *and* a table silently absent from the
    report, with the contract's "one run writes exactly one directory" broken in both directions.
    """
    if os.path.isabs(path) or path.startswith(("../", "./")):
        raise ValueError(f"manifest paths are relative to the results dir, got {path!r}")
    if not path.startswith(prefix):
        raise ValueError(f"expected a {prefix}* path, got {path!r}")
    if ".." in PurePosixPath(path).parts:
        raise ValueError(f"manifest paths must stay inside the results dir, got {path!r}")


def check_artifact_filename(filename: str) -> None:
    """Raise ``ValueError`` unless ``filename`` will land *inside* the directory it is joined to.

    Checking the joined path is not enough, because joining is what destroys the evidence:
    ``Path("/r") / "tables" / "/etc/x.csv"`` is ``/etc/x.csv`` -- an absolute right-hand operand
    discards everything to its left -- while the joined *string* ``"tables//etc/x.csv"`` is not
    absolute, does start with ``tables/`` and contains no ``..``. The composed form therefore passes
    every test that :func:`check_artifact_path` can make, with the write already redirected. The
    name has to be judged before it is used.

    A subdirectory (``sub/x.csv``) stays allowed. No runner writes one today, but nothing about it
    escapes, and forbidding it would be a new restriction rather than a fix.
    """
    text = str(filename)
    if not text.strip():
        raise ValueError("artifact filenames must not be empty")
    if os.path.isabs(text) or ".." in PurePosixPath(text).parts:
        raise ValueError(f"artifact filenames must stay inside the results dir, got {filename!r}")


#: Distinguishes concurrent writers inside one process. ``next`` on an ``itertools.count`` is a
#: single bytecode under the GIL, so two threads cannot draw the same value.
_staging_seq = itertools.count()


def staging_name(filename: str, tag: str = "") -> str:
    """A staging filename for ``filename`` that no other writer will pick.

    Shared by the three writers that publish a post-analysis artefact by ``os.replace``: this
    module's :func:`write_manifest`, L2's ``review.write_review``, and L3's
    ``report.render.write_report``.

    All three used to name the staging file after the destination alone -- ``manifest.json.tmp``,
    ``manifest.json.l2tmp``, ``.report.html.tmp`` -- which every writer of that destination in that
    directory therefore computed identically. It makes the atomicity hold against a reader and not
    against a second writer: A's write truncates the file B is part-way through, A's ``os.replace``
    renames that inode into place, and B's still-open descriptor writes its remaining bytes into the
    published file. B then fails replacing a path that no longer exists, and its cleanup unlinks
    whatever has since taken the name.

    PID plus a process-local counter separates writers across processes and threads alike. A stale
    file from a killed process is simply truncated when that PID next comes round, which is the right
    outcome: it was an orphan.

    Not :func:`tempfile.mkstemp`, which is what the four atomic writers under ``sog_install/`` and
    ``sog_portal/`` use: it creates the file 0600 and ``os.replace`` carries the mode onto the
    destination. For ``.env`` that is the point. For ``manifest.json`` and ``report.html`` it would
    quietly make a results directory unreadable to anyone the user shares it with, so the file is
    created by the ordinary write and the umask still decides.
    """
    marker = f"{tag}." if tag else ""
    # The random part is a security boundary, not decoration. In the portal these writers run as root
    # over directories the sandboxed agent worker can write, and the PID and counter alone were
    # predictable: a worker that planted symlinks at the next names had root write the manifest or
    # page through them -- over admins.json, for one (hunt 2026-09-30, u11-stcoscientist-1). The
    # name is now unguessable and :func:`open_staging` refuses to create it through a link.
    return f".{filename}.{marker}{os.getpid()}.{next(_staging_seq)}.{secrets.token_hex(8)}.tmp"


def open_staging(tmp: Path | str, mode: str = "w", encoding: str | None = "utf-8", *, dir_fd: int | None = None):
    """Create the staging file ``tmp`` for writing, never through a symlink and never over an existing path.

    ``O_CREAT | O_EXCL`` fails if anything -- a planted symlink included -- already has the name, and
    ``O_NOFOLLOW`` makes the same refusal explicit. Mode 0666 so the umask still decides, as
    :func:`staging_name` explains. The caller publishes with ``os.replace``, which replaces a link at
    the destination rather than writing through it. ``dir_fd`` makes ``tmp`` relative to a directory
    the caller already holds open, so the staging file lands in that directory and not in whatever
    has since taken its path (hunt 2026-09-30, rp-u18).
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(tmp, flags, 0o666, dir_fd=dir_fd)
    if "b" in mode:
        return os.fdopen(fd, mode)
    return os.fdopen(fd, mode, encoding=encoding)


def discard_staging(tmp: Path | str, *, dir_fd: int | None = None) -> None:
    """Remove a staging file whose write did not reach ``os.replace``. Never raises.

    The counterpart to :func:`staging_name`, and shared by the same three writers. Unique names are
    what makes it necessary: the shared name they used to compute was truncated by whoever wrote next,
    so a killed write cost one stale file in total, whereas a per-writer name is never handed out
    again and every interrupted run would leave its own behind in the user's results directory.

    Callers reach it from ``except BaseException`` -- not ``except Exception`` -- because a Ctrl-C or
    a ``SystemExit`` landing between the open and the replace is precisely the case that strands one,
    and neither derives from ``Exception``. That is the pattern ``sog_install/state.py`` already documents.

    Swallowing ``OSError`` here is deliberate: this runs while another exception is propagating, and
    the write's own failure is the one worth reporting. Anything that stops the unlink -- the file
    already gone, a read-only directory -- also stops us from doing better than leaving it.
    """
    try:
        # ``os.unlink`` rather than ``Path.unlink``: the name is relative to ``dir_fd`` when one is
        # given, the descriptor :func:`open_staging` created it under.
        os.unlink(tmp, dir_fd=dir_fd)
    except OSError:
        pass


def write_manifest(results_dir: Path, manifest: Manifest) -> Path:
    """Serialise atomically: full content into a staging file, then one ``os.replace``.

    A half-written ``manifest.json`` is worse than a missing one -- L3 would render a truncated
    report and L2 would review a file that does not describe the run. The staging name is unique per
    writer; :func:`staging_name` explains why the destination's own name was not enough.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "manifest.json"
    tmp = results_dir / staging_name("manifest.json")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=False, default=str)
    try:
        with open_staging(tmp) as fh:
            fh.write(payload)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        # ``BaseException`` so a Ctrl-C landing mid-write still removes the staging file, matching
        # ``sog_install.state._atomic_write_json``. It matters more now the name is unique: the old shared
        # one was truncated by whoever wrote next, while an orphan under a per-writer name is never
        # reclaimed, so every interrupted run would leave one behind in the user's results directory.
        discard_staging(tmp)
        raise
    return target


#: The counterpart to :func:`write_manifest` -- how a manifest we wrote is recognised on the way
#: back in. It is defined one level up, in :mod:`spatialomicsgym.generated_report`, and only
#: re-exported here.
#:
#: Two layers ask the question. :func:`sources.collect_files` skips directories holding one, so a
#: second post-analysis run does not read the first run's findings table back as a tool result; the
#: scoring layer must skip them for the same reason, since the engine writes ``post_analysis/``
#: *inside* the tool's output directory, which is the directory that layer is later pointed at.
#: :class:`tests.test_postanalysis_engine.TestEvalNeutrality` fences the two packages apart in both
#: directions, so a rule they share can live in neither -- and duplicating it is exactly the drift
#: it exists to prevent.
from spatialomicsgym.generated_report import (
    declared_artifact_paths,  # noqa: F401  re-exported
    is_generated_report,
    is_generated_report_payload,
    split_report_files,  # noqa: F401  re-exported
)

#: The name this package asks the question under, and what :mod:`sources` aliases in turn.
is_our_manifest = is_generated_report

#: The same question for a payload already read. L2 holds one -- ``read_manifest`` parsed it to
#: decide the file was a JSON object at all -- so asking by path would read it twice.
is_our_manifest_payload = is_generated_report_payload


@contextmanager
def step(manifest: Manifest, label: str):
    """Run one analysis or figure. A failure degrades the run; it never ends it.

    Contract non-negotiable #4. Every task runner wraps each of its steps in this, so a tool output
    that breaks the co-localization heatmap still produces the composition bar chart, the tables and
    a manifest that says what went wrong.
    """
    try:
        yield
    except Exception as exc:
        manifest.warn(f"{label} failed: {type(exc).__name__}: {exc}")
        manifest.degrade()


def _replace_or_append(entries: list[dict[str, Any]], entry: dict[str, Any]) -> None:
    """One artifact path is one manifest entry -- the rule :meth:`Manifest.add_finding` applies to keys.

    Writing the same filename twice leaves *one* file on disk, so a second entry described a file
    that no longer had the content it claimed: the report listed the same table twice, once with a
    row count belonging to the version that had been overwritten. The position is kept rather than
    moved to the end, so re-writing a table does not reorder the report around it.
    """
    for i, existing in enumerate(entries):
        if existing["path"] == entry["path"]:
            entries[i] = entry
            return
    entries.append(entry)


def _jsonable(value: Any) -> Any:
    """numpy scalars and pandas types are not JSON-serialisable; findings must be.

    ``nan`` and the infinities are the case that looks handled and is not: they *are* Python floats,
    so they used to pass straight through, and ``json.dumps`` writes them as the bare tokens ``NaN``
    and ``Infinity``. JSON has no such literals (RFC 8259). Python's ``json.loads`` accepts them, so
    nothing on the Python side of L3 ever complained -- but ``manifest.json`` is served over HTTP,
    and ``JSON.parse`` rejects the *whole document*, so one unlucky finding takes the entire report
    with it. ``null`` is the honest encoding: it is what a NaN mean already meant -- "this could not
    be computed" -- and every consumer already handles a null finding.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    for attr in ("item", "tolist"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                break
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
