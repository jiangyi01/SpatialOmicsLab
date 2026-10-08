"""Cell-cell communication results, read for the explorer's interactive panels (Program 10).

A communication tool's output is told by its files' names and their header or key shape (``detect``), never by what
the analysis found and never by a record's ``produced_by``. Each backend has an adapter that fills one shared set of
facets (``facets.FACETS``) from what that tool really wrote, and says in ``warnings`` what it cannot give. COMMOT's
direction field and its group-by-group sums are derived here from the spot-by-spot matrices the h5ad already holds
(``direction``, ``groups``): the COMMOT worker never writes them, and no transport problem is solved again.

The modules, in the order a request moves through them:

``detect``    which backend and which role a file name is, and which siblings belong with it (stdlib only)
``facets``    the facet ids, every bound, and ``fit`` -- an answer trimmed to the reader's line (stdlib only)
``errors``    ``CCCRefusal``, the one exception a read answers with
``h5``        obsp, uns and obs reads over :class:`spatialomicsgym.viz.h5lite.H5AD`
``tables``    CSV through a descriptor, in blocks, under the table caps
``direction`` per-spot signal direction and the binned arrow field (numpy)
``groups``    group-by-group sums, means and permutation z/p (numpy)
``adapters``  one per backend
``child``     ``serve(spec, rows)``: the reader child's two operations

**Importing this package imports nothing.** ``detect`` and ``facets`` are imported by the portal process, which never
loads numpy, pandas or h5py (``test/test_webui_stays_out_of_the_heavy_stack.py``); the rest is imported only inside
the explorer's reader child.
"""

from __future__ import annotations

from typing import Any

__all__ = ["adapters", "child", "detect", "direction", "errors", "facets", "groups", "h5", "tables"]


def __getattr__(name: str) -> Any:
    """Import a submodule on first use, so the package itself stays cheap."""
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
