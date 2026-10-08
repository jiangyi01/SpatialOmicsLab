"""Reading one assistant turn: the action to run (``action``) versus the final answer (``answer``).

``spatialomicsgym.action`` and ``spatialomicsgym.answer`` resolve to the same module objects here through
``spatialomicsgym._aliases``.

Docstring only, on purpose: ``import spatialomicsgym`` loads ``turn.answer``, and that import must stay stdlib-only
(the bare-``python3`` log redactor that ``restart_portal.sh`` starts runs it).
"""
