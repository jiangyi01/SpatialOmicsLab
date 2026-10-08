"""The tool package. One name is re-exported here, and lazily.

``from spatialomicsgym.tool import get_tool_decorated_functions`` has worked since the rename, and
it resolved through ``spatialomicsgym.utils`` -- whose ``tool_conversion`` imports pandas and
langchain_core.tools at load. That made importing ANY ``spatialomicsgym.tool.*`` module -- the
tool-description dicts ``read_module2api`` walks to count what ships -- pay for the heavy stack,
on the portal's settings routes (found red in the 2026-09-20 gate). The name is still here; it is
resolved the first time it is asked for, and nothing is imported until then.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_tool_decorated_functions"]


def __getattr__(name: str) -> Any:
    if name == "get_tool_decorated_functions":
        from spatialomicsgym.utils.tool_conversion import get_tool_decorated_functions

        globals()[name] = get_tool_decorated_functions
        return get_tool_decorated_functions
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
