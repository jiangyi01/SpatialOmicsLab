"""Post-analysis subsystem.

Layers are built independently against `docs/design/post_analysis_contract.md`:

  L1 engine        -- analysis + plotting + manifest writing (`engine.py` and its helpers)
  L2 review        -- `review.py` (results self-check) and `next_step.py` (what to do next)
  L3 presentation  -- HTML report / webui, outside this package

Usage::

    from spatialomicsgym.postanalysis import run_post_analysis

    results_dir = run_post_analysis("/path/to/tool/output", tool_name="run_spagcn")

**Nothing is imported eagerly here.** Contract non-negotiable 3 requires
``import spatialomicsgym.postanalysis`` to succeed in the 1.6 GB agent env, and importing
:mod:`~spatialomicsgym.postanalysis.engine` pulls in the plotting and dataframe stack. The single
public name is therefore resolved through the PEP 562 module ``__getattr__`` below: the submodule is
imported on first *attribute access*, not on import of this package. Deleting ``__getattr__`` in
favour of a plain ``from .engine import run_post_analysis`` would make ``import
spatialomicsgym.postanalysis`` cost matplotlib again -- silently, since no unit test on a full dev
box would notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["run_post_analysis"]

if TYPE_CHECKING:  # pragma: no cover - type checkers only; never executed at runtime
    from .engine import run_post_analysis


def __getattr__(name: str):
    if name == "run_post_analysis":
        from .engine import run_post_analysis

        return run_post_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
