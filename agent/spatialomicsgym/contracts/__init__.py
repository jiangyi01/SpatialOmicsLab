"""Vocabularies that several layers share and none of them owns.

``stream_events`` is the portal's SSE frame-event vocabulary, ``task_types`` the analysis task names, and
``generated_report`` the rule that recognises a report this system wrote about a tool's output. The old dotted names
(``spatialomicsgym.stream_events``, ...) resolve to the same module objects here through ``spatialomicsgym._aliases``.

Docstring only, and it must stay that way: ``task_types`` imports ``spatialomicsgym.postanalysis``, so any import here
would pull the post-analysis package into the benchmarking package's import of ``generated_report``.
"""
