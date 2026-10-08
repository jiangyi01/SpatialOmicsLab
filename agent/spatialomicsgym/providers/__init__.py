"""The LLM vendor layer: the model factory, provider-name rules, retry backoff and the Responses-API stream reader.

``llm`` builds the chat model, ``provider_names`` holds the stdlib-only provider rules, ``provider_backoff`` decides
retries, and ``responses_stream`` reads a streamed Responses-API reply. The folder is named ``providers`` rather than
``llm`` so that ``spatialomicsgym.llm`` can stay a module name: every old dotted name (``spatialomicsgym.llm``,
``spatialomicsgym.provider_names``, ...) resolves to the same module object here through ``spatialomicsgym._aliases``.

Docstring only, on purpose: the installer's stdlib-only probe imports ``provider_names``, so this file imports nothing.
"""
