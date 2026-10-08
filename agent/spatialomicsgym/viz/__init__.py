"""Analysis-aware visualization for spatial transcriptomics and single-cell data.

A figure here is not a picture the caller asked for; it is a picture the data can support. Every
plot declares what it needs, the dataset is profiled once and checked against that, and a request
that cannot be answered comes back naming what is missing, which call would produce it, and what
can be drawn instead. A figure that is drawn carries a record of how -- which matrix slot, which
transform, which colour limits, what was sampled -- and that record is what makes it both
reproducible and revisable without redoing the analysis behind it.

The modules, in the order a request moves through them:

``profile``       one backed read: what the dataset holds, and what could not be established
``capabilities``  the catalogue: what can be drawn, what each plot needs, what is unsupported here
``layers``        the only place values are read, and the only place a matrix slot is chosen
``palette``       colour, and the disclosures that stop it lying
``render``        drawing primitives, each taking an axes so panels compose onto one canvas
``spec``          the figure's own record, its identity, and the caption composed from it
``manifest_io``   the declaration without which a figure never reaches the chat
``pipelines``     one function per tool call, tying the six together
``sample``        the explorer's display sample: which points a view draws, reproducibly, and the caption that says so

**Importing this package does not import matplotlib.** The capability registry is read by a
portal that has to stay importable in an environment with no plotting stack, so every heavy
import is inside a function body. The accessor below keeps that true while still letting callers
write ``from spatialomicsgym.viz import pipelines``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["capabilities", "layers", "manifest_io", "palette", "pipelines", "profile", "render", "sample", "spec"]


def __getattr__(name: str) -> Any:
    """Import a submodule on first use, so the package itself stays cheap."""
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
