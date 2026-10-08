"""
``spatialomicsgym.report`` -- the L3 presentation layer of the post-analysis subsystem.

It **consumes** ``manifest.json`` (``docs/design/post_analysis_contract.md``) and turns it into
something a human reads. It never re-reads raw tool outputs to decide what exists, and it never
imports analysis or plotting code -- so this package imports in the 1.6 GB minimal agent env with
nothing but the standard library.

Two entry points:

* :func:`write_report` -- render ``<results_dir>/report.html``, fully self-contained (base64
  figures, inline CSS, no network, no JavaScript). This is the file a user emails.
* :func:`render_report` / :func:`render_report_body` -- the same renderer with the figures pointed
  at URLs instead, which is how the web portal shows a run without copying megabytes into the page.

:mod:`spatialomicsgym.report.discover` finds runs on disk for the portal;
:mod:`spatialomicsgym.report.manifest` reads and contains them (``safe_subpath`` is the single
path-containment primitive both surfaces use).
"""

from __future__ import annotations

from .manifest import (
    FIGURE_KINDS,
    MANIFEST_NAME,
    SCHEMA_VERSION,
    STATUSES,
    TASK_TYPES,
    VERDICTS,
    ManifestError,
    is_run_dir,
    load,
    normalize,
    safe_subpath,
)
from .render import (
    REPORT_CSS,
    analysed_at,
    badge_class,
    esc_url,
    redact_text,
    render_report,
    render_report_body,
    write_report,
)

__all__ = [
    "FIGURE_KINDS",
    "MANIFEST_NAME",
    "REPORT_CSS",
    "SCHEMA_VERSION",
    "STATUSES",
    "TASK_TYPES",
    "VERDICTS",
    "ManifestError",
    "analysed_at",
    "badge_class",
    "esc_url",
    "is_run_dir",
    "load",
    "normalize",
    "redact_text",
    "render_report",
    "render_report_body",
    "safe_subpath",
    "write_report",
]
