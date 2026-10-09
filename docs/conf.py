"""Sphinx configuration for the SpatialOmicsLab documentation.

Built by Read the Docs from ``.readthedocs.yaml`` at the repository root, and locally with
``make -C docs html`` (see ``docs/README.md``). The build needs only ``docs/requirements.txt``: the API
reference is produced by ``sphinx-autoapi`` from the source files, so the package and its scientific
dependencies are never imported, and the tool catalogue is generated from the tool specs and the MCP
config by the local extension in ``docs/_ext/``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR / "_ext"))

import sog_tool_catalogue  # docs/_ext, on sys.path just above

if TYPE_CHECKING:
    from sphinx.application import Sphinx


def _read_version() -> str:
    """``spatialomicsgym.version.__version__`` without importing the package (which would load its ``.env``)."""
    text = (REPO_ROOT / "agent" / "spatialomicsgym" / "version.py").read_text(encoding="utf-8")
    match = re.search(r"""__version__\s*=\s*["']([^"']+)["']""", text)
    return match.group(1) if match else "0.0.0"


# -- Project information ------------------------------------------------------------------------------------------
project = "SpatialOmicsLab"
author = "The SpatialOmicsLab team"
copyright = f"{datetime.now(tz=UTC).year}, the SpatialOmicsLab authors. AGPL-3.0-only"
release = _read_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",  # Google-style ``Args:`` sections, which the package uses
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",  # AutoAPI feeds it the source, so nothing is imported
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx_design",
    "sog_tool_catalogue",  # docs/_ext: writes docs/tools/*.md from the specs and the MCP config
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md", "**/.ipynb_checkpoints"]
nitpicky = False
# The package's ``Args:`` blocks name types loosely (``str``, ``AnnData``, ``dict[str, Any]``); unresolved
# cross-references are rendered as plain text rather than reported. ``ref.python`` is the "more than one target
# found" notice for a bare :mod:`tables`-style reference in a docstring (two modules share the short name); the
# first match is linked, which is what the docstring meant.
suppress_warnings = ["autoapi.python_import_resolution", "myst.xref_missing", "ref.python"]

# -- MyST (Markdown) ----------------------------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_linkify_fuzzy_links = False
# ``{{ n_servers }}``, ``{{ n_mcp_functions }}``, ``{{ n_in_process }}`` and ``{{ n_tools_total }}`` are counted from
# the tool specs and the MCP config, so the numbers in the prose cannot drift from the catalogue.
myst_substitutions = {
    "repo": "https://github.com/jiangyi01/SpatialOmicsLab",
    "version": release,
    **sog_tool_catalogue.substitutions(),
}

# -- Napoleon ------------------------------------------------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# ``Attributes:`` sections become ``:ivar:`` fields. As ``.. attribute::`` directives they would duplicate the
# attributes AutoAPI already documents from a dataclass's annotations ("duplicate object description").
napoleon_use_ivar = True
# Type specs are left as written: a ``Returns:`` block such as ``dict: {"status": ...}`` is prose here, not a
# type expression, and tokenising it only yields "invalid value set" warnings.
napoleon_preprocess_types = False

# -- AutoAPI (API reference, parsed statically) -------------------------------------------------------------------
autoapi_type = "python"
autoapi_dirs = [
    str(REPO_ROOT / "agent" / "spatialomicsgym"),  # the agent package
    str(REPO_ROOT / "install" / "sog_install"),  # the ``sog-setup`` installer
    str(REPO_ROOT / "agent" / "skills"),  # the skill catalogue used for tool routing
]
autoapi_root = "api"
autoapi_add_toctree_entry = True
autoapi_keep_files = False
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]
# Not documented: Biomni-era code with no production caller, vendored helper scripts, conda recipes, the
# know-how corpus and the shims copied into child interpreters. Patterns are matched against full paths.
autoapi_ignore = [
    "*/__pycache__/*",
    "*/legacy/*",
    "*/spatialomicsgym_env/*",
    "*/know_how/*",
    "*/tool/omics_scripts/*",
    "*/tool/protocols/*",
    "*/tool/schema_db/*",
    "*/tool/tool_description/*",
    "*/utils/child_site/*",
    "*/spatial3d/_calibrate.py",
]

# -- Intersphinx ---------------------------------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
}
intersphinx_timeout = 30

# -- Copy button ---------------------------------------------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_exclude = ".linenos, .gp, .go"

# -- HTML output ---------------------------------------------------------------------------------------------------
html_theme = "furo"
html_title = f"SpatialOmicsLab {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False
html_theme_options = {
    "source_repository": "https://github.com/jiangyi01/SpatialOmicsLab/",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view"],
    "light_css_variables": {
        "color-brand-primary": "#1f5f8b",
        "color-brand-content": "#1f5f8b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb8e6",
        "color-brand-content": "#7fb8e6",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/jiangyi01/SpatialOmicsLab",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17'
                ".55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-"
                ".52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-"
                ".89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 "
                "2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 "
                "0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 "
                '8.013 0 0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# -- Other builders ------------------------------------------------------------------------------------------------
latex_documents = [(root_doc, "spatialomicslab.tex", "SpatialOmicsLab Documentation", author, "manual")]


# -- Hooks ---------------------------------------------------------------------------------------------------------
def _hide_view_button_on_generated_pages(
    app: Sphinx, pagename: str, templatename: str, context: dict[str, Any], doctree: Any
) -> None:
    """Drop Furo's "view this page" button on generated pages.

    The button links to ``docs/<page>.md`` on GitHub; the tool catalogue and the API reference are generated at
    build time and git-ignored, so on those pages the link would be a 404.
    """
    if pagename.startswith((f"{autoapi_root}/", "tools/")):
        context.pop("page_source_suffix", None)


def _warning_location(record: logging.LogRecord) -> str:
    """Where a Sphinx warning points: a ``path:line`` string, a ``(docname, line)`` pair, or a docutils node."""
    location = getattr(record, "location", None)
    if location is None:
        return record.getMessage()
    if isinstance(location, str):
        return location
    if isinstance(location, tuple):
        return str(location[0])
    # A docutils node: walk up to the nearest ancestor that knows its source file (what Sphinx itself does).
    node = location
    while node is not None:
        source = getattr(node, "source", None)
        if source:
            return os.path.abspath(str(source))
        node = getattr(node, "parent", None)
    return ""


class _MuteGeneratedApiWarnings(logging.Filter):
    """Drop warnings whose source is the generated API reference (``docs/api/``).

    AutoAPI renders every docstring of the package as reStructuredText. Docstrings written as plain prose (an
    indented example, a bare ``----`` rule, an unmatched backtick, a module constant assigned twice) produce
    docutils and domain messages that say nothing about this documentation and would otherwise fail a strict
    build. Hand-written pages are not affected: a broken link or directive there still fails ``make strict``.
    Set ``SOG_DOCS_SHOW_API_WARNINGS=1`` to see the muted messages.
    """

    _PREFIXES = (f"{DOCS_DIR / autoapi_root}{os.sep}", f"{autoapi_root}/")

    def filter(self, record: logging.LogRecord) -> bool:
        return not _warning_location(record).startswith(self._PREFIXES)


def _mute_warnings_from_generated_api() -> None:
    """Install the filter ahead of Sphinx's own, so a muted warning is neither printed nor counted."""
    from sphinx.util.logging import WarningStreamHandler

    for handler in logging.getLogger("sphinx").handlers:
        if isinstance(handler, WarningStreamHandler):
            handler.filters.insert(0, _MuteGeneratedApiWarnings())


_FIRST_API_OBJECT_FOR_NAME: dict[str, int] = {}


def _skip_duplicate_api_members(
    app: Sphinx, what: str, name: str, obj: Any, skip: bool, options: Any
) -> bool | None:
    """Document a name once.

    A module constant assigned on two paths (``ACCESS_TOKEN = ...`` at the top, then again under a condition) is
    two objects to AutoAPI and would be rendered twice, which Sphinx reports as a duplicate object description.
    AutoAPI asks about the same object more than once (for the summary, then for the rendering), so the decision
    is keyed on the object, not the name: the first object seen under a name is kept, every other one is skipped.
    """
    if skip:
        return True
    first = _FIRST_API_OBJECT_FOR_NAME.setdefault(name, id(obj))
    return True if first != id(obj) else None


def setup(app: Sphinx) -> None:
    app.connect("html-page-context", _hide_view_button_on_generated_pages)
    app.connect("autoapi-skip-member", _skip_duplicate_api_members)
    if not os.environ.get("SOG_DOCS_SHOW_API_WARNINGS"):
        _mute_warnings_from_generated_api()
