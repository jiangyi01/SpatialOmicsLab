# docs/: the SpatialOmicsLab documentation

**What this is.** The source of the Read the Docs site: Markdown (MyST) pages, a Sphinx configuration, and a small
extension that generates the tool catalogue. Read the Docs builds it from `.readthedocs.yaml` at the repository root.

**How to build it locally.**

```bash
pip install -r docs/requirements.txt
make -C docs html           # -> docs/_build/html/index.html
make -C docs strict         # same, but any warning fails the build
make -C docs linkcheck      # check external links
```

The package itself does not need to be installed: the API reference is parsed from the source files by
`sphinx-autoapi`, and the tool catalogue is read from `install/recipes/tool_specs/`, `agent/MCP_server/mcp_config.yaml`
and `agent/spatialomicsgym/tool/tool_description/`.

**What is generated, and what is written by hand.**

| Path | Kind | Source |
|---|---|---|
| `tools/*.md` | generated at build time (git-ignored) | `docs/_ext/sog_tool_catalogue.py` over the specs, the MCP config and the in-process tool descriptions |
| `api/` | generated at build time (git-ignored) | `sphinx-autoapi` over `agent/spatialomicsgym`, `install/sog_install` and `agent/skills` |
| everything else | written by hand | this directory |

To regenerate the tool pages without a Sphinx build: `python docs/_ext/sog_tool_catalogue.py`.

**Warnings.** Hand-written pages are held to zero warnings (`make -C docs strict`). Warnings that originate in the
generated API reference (a docstring written as prose rather than reStructuredText, a module constant assigned
twice) are muted by a filter in `conf.py`, because they describe the package's docstrings, not this documentation;
run with `SOG_DOCS_SHOW_API_WARNINGS=1 make -C docs html` to see them, and fix them in the docstrings if you want
the API pages to render better.

**Where things are.**

| Path | Contents |
|---|---|
| `conf.py` | Sphinx configuration (theme, MyST, AutoAPI, intersphinx) |
| `requirements.txt` | the build's Python dependencies; nothing here is a runtime dependency of the package |
| `index.md` | the landing page and the top-level table of contents |
| `getting-started/` | installation, first run, verifying the install |
| `usage/` | the terminal chat, the Python API, what an analysis writes to disk |
| `configuration.md` | the `.env` reference |
| `sog-setup.md` | the installer: wizard, subcommands, scripted runs, moving an install |
| `tools/` | the tool catalogue (generated) |
| `architecture/` | repository layout, the agent package, how MCP tools run |
| `tutorials/` | worked examples |
| `_ext/` | the catalogue generator |
| `_static/` | CSS |

**Conventions.** Pages are MyST Markdown. Cross-reference other pages with relative links (`[text](../configuration.md)`)
and Python objects with roles (`` {py:class}`spatialomicsgym.agent.STCoscientist` ``). Keep the prose in sync with the
root `README.md`: the README is the short version, these pages are the long one.
