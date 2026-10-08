"""
SpatialOmicsLab setup wizard (``sog-setup``).

A single, resumable provisioning flow that takes a fresh clone to a working
agent: a deterministic onboarding front-door (LLM key + API config) that hands
off to an LLM-guided walkthrough for tool-category selection, per-tool conda
env provisioning, non-destructive MCP wiring, and end-to-end verification.

Design invariants:
  * **Stdlib + PyYAML only.** The package must import and run *before* any
    conda env exists, so it never depends on the agent runtime (langchain,
    torch, the bio stack). Heavy modules are imported lazily inside the
    Stage-B engines, after the base env is built.
  * **Non-interference.** The wizard never mutates the user's running project:
    it writes a *separate* generated MCP config, only ever touches conda envs
    in the ``<basic>_*`` namespace, and merges ``.env`` in place (backed up
    first, owned keys only). See :mod:`sog_install.constants`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"

# Editable installs (``pip install -e .``) expose only the distribution's declared
# packages, so ``tools_user`` — which the Stage-B engines import lazily for the
# self-review — is unreachable when ``sog-setup`` runs as a console script. Append the
# agent part (``agent/``) here, at the one import chokepoint both entry points
# (``python -m sog_install`` and the console script) pass through, so no engine silently
# degrades. Append-only ⇒ no shadowing. The ``test/`` smoke harness is loaded by path,
# on demand (``constants.load_test_module``), and says so plainly when ``test/`` is absent.
from . import constants as _constants

_constants.ensure_repo_importable()
