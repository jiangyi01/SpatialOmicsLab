"""Task runners: one module per contract task type, reached only through :func:`runner_for`.

The mapping is the engine's whole dispatch table. Imports are deferred so that pulling in the
package -- or running a deconvolution -- never costs the import of the clustering runner's scanpy.

Three of the seven task types share :mod:`.shallow`. That is a statement about depth, not about
coverage: a shallow run still scans the output, still draws at least one figure, and still says in
its warnings that the task type has no deep support yet. (It said four, from before alignment got
its own deep runner -- hunt 2026-09-30, u19-pa-tasks-research-6.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from spatialomicsgym.postanalysis.context import AnalysisContext

#: Contract task type -> ``module:function``. The three shallow ones share a single runner.
_RUNNERS: dict[str, tuple[str, str]] = {
    "spatial_clustering": (".clustering", "run"),
    "deconvolution": (".deconvolution", "run"),
    "svg_detection": (".svg", "run"),
    "cell_communication": (".shallow", "run"),
    "alignment": (".alignment", "run"),
    "imputation": (".shallow", "run"),
    "trajectory": (".shallow", "run"),
}


def depth_for(task_type: str) -> str | None:
    """``"deep"``, ``"shallow"``, or ``None`` when nothing runs for ``task_type``.

    Three of the seven task types route to the shallow runner, which reads a result and reports on
    it rather than re-deriving it. That is a real difference to someone deciding whether a
    follow-up will answer their question, and it was visible only by reading the dispatch table.
    Exposed as the consequence rather than as the table: callers get the depth, not the module
    name, so the split can be rearranged without breaking them.
    """
    entry = _RUNNERS.get(task_type)
    if entry is None:
        return None
    return "shallow" if entry[0] == ".shallow" else "deep"


def runner_for(task_type: str) -> Callable[[AnalysisContext], None] | None:
    """The runner for ``task_type``, imported on demand, or ``None`` if there is none."""
    entry = _RUNNERS.get(task_type)
    if entry is None:
        return None
    import importlib

    module = importlib.import_module(entry[0], __name__)
    return getattr(module, entry[1])
