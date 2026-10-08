"""Where this installation writes analysis output, and where a reader should look for it.

A single run currently writes to more than one root, and that is not an accident waiting to be
tidied away -- it is structural:

============================  ==================================================================
``<agent.path>/outputs``      code the model writes itself. Bound into the system prompt by
                              ``agent/prompt_builder.py`` as ``{output_path}``.
``$SOG_WORK_DIR`` -> a
writable ``/workspace/work``
-> ``./work``                 every MCP portal whose caller omitted ``output_dir``. Resolved by
                              ``tools/base_mcp.py::default_output_dir``.
============================  ==================================================================

The portals run as subprocesses in their own conda envs, where this package is not installed, so
``tools/base_mcp.py`` cannot import from here. And ``tools/`` has no ``__init__.py`` and reaches a
wheel only as a read-only data copy (``spatialomicsgym/_platform/tools/``, executed by path, never
importable), so this package cannot import from *there* either -- ``from tools.base_mcp import
default_output_dir`` succeeds only when the repo root happens to be on ``sys.path``, and on a wheel
install it raises ``ModuleNotFoundError``. Where that import was wrapped in a bare ``except``, the
failure was silent and the caller simply stopped searching the root the portals actually write to.

So the write-side resolver is **mirrored** here rather than imported, the same choice
``benchmarking/workflow_gates.py`` already documents. ``tools/base_mcp.py`` stays the source of
truth for writing; :func:`tool_output_root` restates it for readers, and
``test/test_output_roots_agree.py`` fails if the two ever disagree.

This module also owns the *reading* half of the same question -- **how a path is named to somebody
who is not on this machine**. :func:`shown_path` and :func:`scrub` turn
``/home/alice/proj/spatialomicsgym_data/outputs/run_7`` into ``outputs/run_7``, because a browser
page, a shareable report and an exported transcript all travel off the box that wrote them and the
sender's home directory is nothing any of them are for. It lives here and not in ``redaction`` for a
reason worth writing down: ``redaction`` answers "what does a credential look like", is pinned
byte-identical on 127 strings, and is consumed by the HuggingFace corpus builder and ``chat_cli`` --
and the CLI **must** keep printing real paths, because it runs on the user's own machine and the
path is the answer. A path is not a credential. It is a name that is only meaningful next to a root,
and the roots are already here.

Everything here is stdlib-only and filesystem-read-only at import: ``sog_portal.api.routers.results`` imports it into
the web app, which must stay cheap in the 1.6 GB agent env, and nothing may create a directory as a
side effect of resolution.
"""

from __future__ import annotations

import os
import re
import threading

__all__ = [
    "WORK_DIR_ENV",
    "RESULTS_ROOTS_ENV_VAR",
    "AGENT_OUTPUT_DIRNAME",
    "tool_output_root",
    "work_output_roots",
    "agent_output_root",
    "results_search_roots",
    "ABS_PATH_RE",
    "OUTSIDE_WORKSPACE",
    "workspace_anchors",
    "set_anchors",
    "clear_anchors",
    "shown_path",
    "scrub",
]

#: Deployment override for the portal scratch root. Read by ``tools/base_mcp.py`` too.
WORK_DIR_ENV = "SOG_WORK_DIR"

#: Deployment override for where *finished runs* are looked for, ``os.pathsep``-separated. Distinct
#: from :data:`WORK_DIR_ENV`, which says where new output is written: an operator browsing an
#: archive is not redirecting the next tool into it. Defined here so the web portal, the CLI and
#: ``python -m spatialomicsgym.report`` all spell it the same way.
RESULTS_ROOTS_ENV_VAR = "SOG_RESULTS_ROOTS"

#: Subdirectory of ``agent.path`` that the system prompt names as the place to write results.
AGENT_OUTPUT_DIRNAME = "outputs"

_DEPLOYMENT_WORK_ROOT = "/workspace/work"
_ADVERTISED_WORK_ROOT = "./work"


#: ``SOG_WORK_DIR`` as it stood before the first of the portal turns now bound (``sog_portal.binding``), and how
#: many turns hold it. Empty when no turn is. Counted, because turns overlap: two accounts chatting at once
#: are two bound turns in one process, and the first one out used to clear the value while the second still
#: ran, so ``deployment_work_root`` answered that turn's chat folder (hunt 2026-10-07). See
#: :func:`deployment_work_root`.
_OUTER_WORK_DIR: list[str | None] = []
_OUTER_HOLDS = [0]
_OUTER_LOCK = threading.Lock()


def hold_outer_work_dir(value: str | None) -> None:
    """Record ``SOG_WORK_DIR``'s pre-turn value. Called by each binding about to change it, once.

    Only the first concurrent holder's value is kept: a later turn's "value before me" is an earlier
    turn's chat folder, not the deployment's.
    """
    with _OUTER_LOCK:
        if _OUTER_HOLDS[0] <= 0 or not _OUTER_WORK_DIR:
            _OUTER_WORK_DIR[:] = [value]
            _OUTER_HOLDS[0] = 0
        _OUTER_HOLDS[0] += 1


def release_outer_work_dir() -> None:
    """One holder is done; the value is forgotten when the last one is. Extra calls are no-ops."""
    with _OUTER_LOCK:
        _OUTER_HOLDS[0] = max(0, _OUTER_HOLDS[0] - 1)
        if _OUTER_HOLDS[0] == 0:
            _OUTER_WORK_DIR.clear()


def deployment_work_root() -> str:
    """:func:`tool_output_root` as it reads outside any turn.

    A portal turn sets ``SOG_WORK_DIR`` -- process-wide -- to its own chat folder for as long as it
    runs, and a results listing served meanwhile computed its second root from that. Every run
    under the deployment's work root (``/workspace/work``, where most tool output and every legacy
    run lives) vanished from Results for the length of the turn, and a saved link to one of them
    404'd (hunt 2026-09-30, u03-results-5). Readers ask this instead.
    """
    with _OUTER_LOCK:
        held = list(_OUTER_WORK_DIR)
    if not held:
        return tool_output_root()
    override = (held[0] or "").strip()
    root = override or (
        _DEPLOYMENT_WORK_ROOT
        if os.path.isdir(_DEPLOYMENT_WORK_ROOT) and os.access(_DEPLOYMENT_WORK_ROOT, os.W_OK)
        else _ADVERTISED_WORK_ROOT
    )
    return root


def tool_output_root(subdir: str = "") -> str:
    """The root an MCP portal writes to when its caller supplies no ``output_dir``.

    Mirrors ``tools/base_mcp.py::default_output_dir`` branch for branch -- ``$SOG_WORK_DIR``, then
    ``/workspace/work`` when it exists *and* is writable, then the relative ``./work`` that a fresh
    clone resolves against its own cwd. Read the module docstring before changing an branch here;
    the two copies are pinned equal by a test precisely because they cannot import each other.

    Never creates the directory. Resolution runs at import time on the writing side (the value ends
    up in portal signatures as a default argument), and an import must not touch the filesystem.
    """
    override = os.environ.get(WORK_DIR_ENV, "").strip()
    root = override or (
        _DEPLOYMENT_WORK_ROOT
        if os.path.isdir(_DEPLOYMENT_WORK_ROOT) and os.access(_DEPLOYMENT_WORK_ROOT, os.W_OK)
        else _ADVERTISED_WORK_ROOT
    )
    return os.path.join(root, subdir) if subdir else root


def work_output_roots() -> list[str]:
    """*Every* branch :func:`tool_output_root` could have taken, in priority order, de-duplicated.

    For a reader hunting output that a portal may have written under a different ``SOG_WORK_DIR``,
    by an earlier run, or by a portal whose own default differed -- ``benchmarking``'s misplaced-
    output recovery is the case that needs it. The writer must pick exactly one root; the reader
    checking all of them costs one failed ``is_dir()`` per miss.

    Unconditional on purpose: ``/workspace/work`` is listed even when it is not currently writable,
    because output written while it *was* is still there. That is why this is a separate function
    rather than something :func:`tool_output_root` could return -- the writer must never pick a
    root it cannot write to.

    Kept here so the ``SOG_WORK_DIR`` / ``/workspace/work`` / ``./work`` branch set exists once in
    this package. It previously existed a second time in ``benchmarking/workflow_gates.py``, where
    a fourth branch added to the writer would have had to be remembered twice.
    """
    roots: list[str] = []
    override = os.environ.get(WORK_DIR_ENV, "").strip()
    if override:
        roots.append(override)
    roots.extend([_DEPLOYMENT_WORK_ROOT, _ADVERTISED_WORK_ROOT])
    seen: set[str] = set()
    unique: list[str] = []
    for root in roots:
        try:
            key = os.path.abspath(root)
        except (OSError, ValueError):
            key = root
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def agent_output_root(agent_path) -> str | None:
    """``<agent.path>/outputs`` -- where the system prompt tells generated code to write.

    ``None`` when the agent has no path, so a caller can splice the result into a list without
    first checking. Note ``agent.path`` is already ``<--path>/spatialomicsgym_data``; the constructor
    appends that, so this returns ``<--path>/spatialomicsgym_data/outputs``.
    """
    if not agent_path:
        return None
    text = str(agent_path).strip()
    if not text:
        return None
    return os.path.join(text, AGENT_OUTPUT_DIRNAME)


def results_search_roots(agent_path=None, extra=None, *, work_root: str | None = None) -> list[str]:
    """Every root a finished run could be under, most-specific first, de-duplicated.

    A *union*, not a re-run of the write-side resolver, because the reader and the writer want
    different things: the writer picks exactly one root, while the reader is hunting for output that
    may have been written under a different ``SOG_WORK_DIR``, by an earlier run, or by a portal
    whose own default differs. Searching all of them costs one failed ``is_dir()`` per miss.

    ``agent_path`` is the agent's data root (``agent.path``), not the ``--path`` the user typed.

    ``SOG_RESULTS_ROOTS`` (or an ``extra`` the caller passes instead) **replaces** both defaults
    rather than preceding them: someone who set it is describing a deployment the defaults got
    wrong, and leaving the machine's work directory and the agent's own results listed beside the
    curated tree is the opposite of narrowing the scope.

    Read here and not only by the caller. The web portal's ``results_api.results_roots`` had its own
    copy of this rule and ``paths`` had none, so on any box that set the variable the portal browsed
    the operator's tree while ``chat_cli`` and ``python -m spatialomicsgym.report`` -- the two
    callers of this function -- browsed the defaults, and the three front doors disagreed about
    which runs exist. That is exactly the divergence a shared resolver exists to prevent.

    ``work_root`` replaces the live :func:`tool_output_root` in the defaults; see the comment below.

    De-duplication is by absolute path, but the returned strings keep their original spelling: a
    relative ``./work`` must stay relative so a caller that resolves it against a different cwd
    still gets that cwd's tree, which is exactly what the portals do.
    """
    # ``not extra`` and not ``extra is None``: an empty list is a caller who named nothing, exactly
    # like passing nothing, and the two branches below already agree about that -- the loop yields
    # no candidates either way. Only this line disagreed, which made ``extra=cfg.get("roots", [])``
    # skip the operator's ``SOG_RESULTS_ROOTS`` and browse the deployment defaults instead: a silent
    # wrong answer about which runs exist, not an error anyone would see.
    if not extra:
        extra = [chunk for chunk in (os.environ.get(RESULTS_ROOTS_ENV_VAR) or "").split(os.pathsep) if chunk.strip()]
    candidates: list[str] = []
    for chunk in extra or ():
        if isinstance(chunk, str) and chunk.strip():
            candidates.append(chunk.strip())
    if not candidates:
        agent_root = agent_output_root(agent_path)
        if agent_root:
            candidates.append(agent_root)
        # ``work_root`` is for a reader that must not see a running turn's own folder here -- the
        # portal passes :func:`deployment_work_root`. Everyone else gets the live value, which is
        # what an in-turn reader (the review of the run that just wrote there) needs.
        candidates.append(work_root if work_root is not None else tool_output_root())

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        try:
            key = os.path.abspath(candidate)
        except (OSError, ValueError):
            # An unresolvable spelling (a NUL byte, a path longer than the platform allows) cannot
            # be compared for identity, so it cannot be shown to be a duplicate. Keep it: a root
            # that never matches anything is a wasted stat, while a dropped root loses results.
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


# --------------------------------------------------------------------------------------------- #
# Naming a path to a reader who is not on this machine.
# --------------------------------------------------------------------------------------------- #

#: What an absolute path looks like in prose. Deliberately *not* "anything path-shaped": the
#: lookbehind refuses a match that continues a word, a relative path or a URL authority, so
#: ``and/or``, ``w/``, ``https://host/x`` and ``re.compile("/tmp/[a-z]+")`` are left alone. Trailing
#: sentence punctuation is stripped by :func:`scrub` rather than excluded here, because a directory
#: may legitimately end in a dot.
ABS_PATH_RE = re.compile(r"(?<![\w:/])/(?:[A-Za-z0-9._+@%~-]+)(?:/[A-Za-z0-9._+@%~-]+)*/?")

#: How an absolute path that matches no anchor is named. Never the empty string, and never
#: ``[redacted]``: suppressing the path must not suppress the *fact* that something was read from
#: somewhere else, which is the rule ``report.render`` wrote down first and the one case where a
#: silent drop would change what the page asserts.
OUTSIDE_WORKSPACE = "outside the workspace"

_ANCHOR_LOCK = threading.RLock()
_ANCHORS: dict[str, str] = {}

#: Anchors resolved at call time rather than registered, because the value can change under a
#: running server: ``SOG_WORK_DIR`` is an environment variable an operator edits, and a cached copy
#: would keep renaming tool output against a root nothing writes to any more.
# deployment_work_root, not tool_output_root: a portal turn rebinds SOG_WORK_DIR process-wide, and the
# live anchor moved with it for every concurrent request (hunt 2026-09-30, uL2-concurrency-10).
_LIVE_ANCHORS = (("tool-output", deployment_work_root),)


def set_anchors(anchors: dict[str, object] | None = None, **kwargs: object) -> None:
    """Register ``label -> root`` pairs. Replaces the whole registry; ``None``/blank roots are dropped.

    Called once at web-app start-up and again whenever the configured data path changes, which is
    why it replaces rather than merges: a root that moved must stop being an anchor, or a stale
    label keeps claiming paths that are no longer under it.

    Roots are stored absolute (``~`` expanded) because that is what a match is tested against, but
    an unresolvable spelling is kept verbatim rather than dropped -- an anchor that never matches
    costs one failed comparison, while a dropped anchor leaks the path it existed to rename.
    """
    merged: dict[str, object] = dict(anchors or {})
    merged.update(kwargs)
    resolved: dict[str, str] = {}
    for label, root in merged.items():
        text = str(root or "").strip()
        if not text or not str(label).strip():
            continue
        try:
            text = os.path.abspath(os.path.expanduser(text))
        except (OSError, ValueError):
            pass
        resolved[str(label).strip()] = text.rstrip("/") or "/"
    with _ANCHOR_LOCK:
        _ANCHORS.clear()
        _ANCHORS.update(resolved)


def clear_anchors() -> None:
    """Forget every registered anchor. For tests, and for a server tearing an app down."""
    with _ANCHOR_LOCK:
        _ANCHORS.clear()


def workspace_anchors() -> list[tuple[str, str]]:
    """Every ``(label, root)`` a path may be named against, **longest root first**.

    Longest first is the whole of the matching rule: ``<path>/spatialomicsgym_data/outputs`` sits
    inside ``<path>``, and a shortest-first scan would name every result ``workspace/...`` and never
    ``outputs/...``. Sorting here rather than at each call site is what keeps the three choke points
    from disagreeing about which label wins.

    The live anchors are appended after the registered ones and then sorted with them, so an
    operator who registered a more specific root still beats ``$SOG_WORK_DIR``.
    """
    with _ANCHOR_LOCK:
        pairs = list(_ANCHORS.items())
    seen = {root for _, root in pairs}
    for label, resolve in _LIVE_ANCHORS:
        try:
            root = os.path.abspath(str(resolve() or "")).rstrip("/")
        except (OSError, ValueError, TypeError):
            continue
        if root and root != "/" and root not in seen:
            seen.add(root)
            pairs.append((label, root))
    pairs.sort(key=lambda item: (-len(item[1]), item[0]))
    return pairs


def _under_for_display(path: str, root: str) -> str | None:
    """``path`` relative to ``root``, or ``None`` when it is not inside it. **Display only.**

    String containment is not the test: ``/data/run`` is not inside ``/data/run_2`` even though one
    spells the other. Compared component-wise via ``relpath`` + a ``..`` check, the same shape
    ``report.render`` used, so a sibling directory whose name merely starts with the root's cannot
    borrow its label.

    **Not a containment check, and the name now says so.** It works on ``abspath`` and never
    ``resolve()``, so a symlink inside ``root`` is reported as inside it -- which is correct for
    this module's job (labelling a path the way the user wrote it) and wrong for deciding whether
    a write may happen. Its two callers, :func:`shown_path` and :func:`scrub`, only ever print.

    A caller that needs the security answer wants one of the two predicates that already resolve
    before they contain: :func:`spatialomicsgym.report.manifest.safe_subpath` for a path under a
    known root, or :func:`sog_portal.services.datastore.resolve_derived` for one under the
    allowed derived roots. This function is deliberately NOT a third of them.
    """
    if not root:
        return None
    try:
        rel = os.path.relpath(path, root)
    except (OSError, ValueError):
        return None
    if rel == os.curdir:
        return ""
    if rel.startswith(os.pardir + os.sep) or rel == os.pardir:
        return None
    return rel.replace(os.sep, "/")


def shown_path(
    value: object,
    *,
    base: object = None,
    anchors: list[tuple[str, str]] | None = None,
    outside: str = OUTSIDE_WORKSPACE,
) -> str:
    """An absolute path renamed against the roots this installation knows about.

    Four branches, tried in order, the first two being ``report.render._shown_path``'s original
    behaviour and the third the generalisation this module exists for:

    1. not absolute (or empty) -- returned untouched. A relative path is already a name.
    2. under ``base`` -- returned relative to it, or with one ``../`` when it is a sibling. ``base``
       is the caller's local frame of reference (a report's own results directory); it wins over
       the anchors because "the directory this page is about" is a better name than "somewhere
       under outputs/".
    3. under a registered anchor -- ``label/rel``, longest root first (see :func:`workspace_anchors`).
    4. anything else -- ``basename (outside)``. **Never the empty string**: a source read from
       another directory is still named, and still marked as coming from elsewhere.

    Idempotent on its own output, because every branch returns something relative and branch 1
    returns a relative input unchanged -- so a value that passes through two choke points (a run
    label scrubbed on write and again on read) is not renamed twice.
    """
    text = str(value or "").strip()
    if not text or not os.path.isabs(text):
        return text
    try:
        target = os.path.abspath(text)
    except (OSError, ValueError):
        return os.path.basename(text.rstrip("/")) or text
    base_text = str(base or "").strip()
    if base_text:
        try:
            base_abs = os.path.abspath(base_text)
        except (OSError, ValueError):
            base_abs = ""
        if base_abs:
            rel = _under_for_display(target, base_abs)
            if rel is not None:
                return rel or os.path.basename(base_abs)
            rel = _under_for_display(target, os.path.dirname(base_abs))
            if rel is not None and rel:
                return "../" + rel
    for label, root in workspace_anchors() if anchors is None else anchors:
        rel = _under_for_display(target, root)
        if rel is not None:
            return f"{label}/{rel}" if rel else label
    return f"{os.path.basename(text.rstrip('/')) or text} ({outside})"


#: Characters a sentence may put immediately after a path that are not part of it. Stripped from a
#: match before it is resolved and put back afterwards, so ``... in /a/b/outputs.`` keeps its stop.
_TRAILING_PUNCTUATION = ".,;:!?)]}>\"'"


def scrub(text: object) -> str:
    """Rewrite the absolute paths in prose that resolve under a known anchor. Leave the rest alone.

    Deliberately conservative, and the conservatism is the design: "rewrite anything path-shaped"
    mangles ``and/or``, ``w/``, URLs and regex literals, while "rename the directories that are
    ours" is exactly what was asked for. A path under no anchor is **not** touched -- it is somebody
    else's name for somebody else's file, and a portal that rewrote it would be corrupting data it
    does not own rather than protecting anything.

    The output is always an *anchored relative* form, never a placeholder, because this runs over
    the agent's reasoning trace: a command the model printed as
    ``--output-dir /srv/x/spatialomicsgym_data/outputs/run_7`` becomes ``--output-dir
    outputs/run_7``, which still says which run. Replacing it with ``[redacted]`` would leave the
    reader a command they cannot act on and cannot even ask about.

    Best-effort on prose and complete on fields, and the difference is worth stating rather than
    hiding: a path split across two lines by a wrapped log, or one the agent assembled with
    ``os.path.join`` at runtime, is not a match here and will not be renamed. The structured fields
    the pages actually render go through :func:`shown_path` directly, where there is no guessing.
    """
    s = str(text if text is not None else "")
    if "/" not in s:
        return s
    anchors = workspace_anchors()
    if not anchors:
        return s

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in _TRAILING_PUNCTUATION:
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        if not raw or raw == "/":
            return match.group(0)
        for _label, root in anchors:
            if _under_for_display(os.path.abspath(raw), root) is not None:
                return shown_path(raw, anchors=anchors) + trailing
        return match.group(0)

    return ABS_PATH_RE.sub(_replace, s)
