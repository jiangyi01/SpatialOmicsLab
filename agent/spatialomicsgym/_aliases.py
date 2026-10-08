"""Old import names that keep working after a move: one table, one meta-path finder.

Two moves use it:

* the re-layout: the installer moved to ``sog_install`` (``install/``), so ``spatialomicsgym.setup`` maps there;
* the agent/ re-organisation: 21 modules and packages of this package moved into role folders
  (``providers/``, ``turn/``, ``contracts/``, ``software_catalog/``, ``policy/``, ``legacy/``), so
  ``spatialomicsgym.llm`` maps to ``spatialomicsgym.providers.llm`` and so on (table below).

The finder maps the old name and every dotted name under it to the SAME module object as the new name, so a
monkeypatch through either name reaches the code, and no module is ever loaded twice. Nothing is imported until
someone asks for an old name, so importing ``spatialomicsgym`` stays light.

Stdlib only. The ``setup`` entry are removed in a later release; the agent/ re-organisation entries are
removed only by a deliberate rewrite of their importers (Phase C of that re-organisation), never silently.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys

#: old prefix -> new prefix. Every key starts with ``spatialomicsgym.`` (the fast path in ``target_of`` relies on it).
ALIASES = {
    "spatialomicsgym.setup": "sog_install",
    # agent/ re-organisation: same module object under both names.
    "spatialomicsgym.llm": "spatialomicsgym.providers.llm",
    "spatialomicsgym.provider_names": "spatialomicsgym.providers.provider_names",
    "spatialomicsgym.provider_backoff": "spatialomicsgym.providers.provider_backoff",
    "spatialomicsgym.responses_stream": "spatialomicsgym.providers.responses_stream",
    "spatialomicsgym.action": "spatialomicsgym.turn.action",
    "spatialomicsgym.answer": "spatialomicsgym.turn.answer",
    "spatialomicsgym.stream_events": "spatialomicsgym.contracts.stream_events",
    "spatialomicsgym.task_types": "spatialomicsgym.contracts.task_types",
    "spatialomicsgym.generated_report": "spatialomicsgym.contracts.generated_report",
    "spatialomicsgym.env_desc": "spatialomicsgym.software_catalog.env_desc",
    "spatialomicsgym.env_desc_cm": "spatialomicsgym.software_catalog.env_desc_cm",
    "spatialomicsgym.netbind": "spatialomicsgym.policy.netbind",
    "spatialomicsgym.generate_function": "spatialomicsgym.legacy.generate_function",
    "spatialomicsgym.extract_biorxiv_tasks": "spatialomicsgym.legacy.extract_biorxiv_tasks",
    "spatialomicsgym.process_all_subjects": "spatialomicsgym.legacy.process_all_subjects",
    "spatialomicsgym.agent.env_collection": "spatialomicsgym.legacy.env_collection",
    "spatialomicsgym.agent.function_generator": "spatialomicsgym.legacy.function_generator",
    "spatialomicsgym.agent.qa_llm": "spatialomicsgym.legacy.qa_llm",
    "spatialomicsgym.eval": "spatialomicsgym.legacy.eval",
    "spatialomicsgym.task": "spatialomicsgym.legacy.task",
    "spatialomicsgym.tool.example_mcp_tools": "spatialomicsgym.legacy.example_mcp_tools",
}


def target_of(fullname: str) -> str | None:
    """The new dotted name for an old one, or ``None`` when ``fullname`` is not an aliased name.

    A key matches the name itself and every dotted name under it, never a name that merely shares its spelling
    (``spatialomicsgym.task`` does not match ``spatialomicsgym.task_types``). When several keys match, the longest
    wins, so a more specific entry can never be shadowed by a shorter one, whatever the table's order.
    """
    # Fast path: this finder sits first on ``sys.meta_path`` and is asked about every import in the process.
    if not fullname.startswith("spatialomicsgym."):
        return None
    best = None
    for old in ALIASES:
        if (fullname == old or fullname.startswith(old + ".")) and (best is None or len(old) > len(best)):
            best = old
    if best is None:
        return None
    return ALIASES[best] + fullname[len(best) :]


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str, target_spec) -> None:
        self._target = target
        self._target_spec = target_spec
        self._real_spec = None

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        self._real_spec = getattr(module, "__spec__", None)
        return module

    def exec_module(self, module) -> None:
        # The target module is already executed; re-running it would load a second copy. Module creation
        # stamped the alias spec onto it -- put the real one back so __spec__.name/.parent stay the target's.
        if self._real_spec is not None:
            module.__spec__ = self._real_spec

    # ``python -m <old name>`` (runpy) asks the loader for code: hand over the target's.
    def get_code(self, fullname):
        return self._target_spec.loader.get_code(self._target)

    def get_source(self, fullname):
        get_source = getattr(self._target_spec.loader, "get_source", None)
        return get_source(self._target) if get_source else None

    def is_package(self, fullname) -> bool:
        return self._target_spec.submodule_search_locations is not None

    def get_filename(self, fullname):
        return self._target_spec.origin


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        new = target_of(fullname)
        if new is None:
            return None
        # Locate the target without executing it (runpy looks up ``<pkg>.__main__`` this way).
        target_spec = importlib.util.find_spec(new)
        if target_spec is None:
            return None
        spec = importlib.machinery.ModuleSpec(
            fullname,
            _AliasLoader(new, target_spec),
            origin=target_spec.origin,
            is_package=target_spec.submodule_search_locations is not None,
        )
        spec.has_location = target_spec.has_location
        return spec


def install() -> None:
    """Put the finder first on ``sys.meta_path`` (idempotent)."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
