"""Code from the Biomni era with no production caller: importable, kept out of the reader's way, not extended.

``generate_function``, ``extract_biorxiv_tasks`` and ``process_all_subjects`` came from the top of the package;
``env_collection``, ``function_generator`` and ``qa_llm`` from ``agent/``; ``eval``, ``task`` and
``example_mcp_tools`` are moved whole. Every old dotted name (``spatialomicsgym.generate_function``,
``spatialomicsgym.agent.qa_llm``, ``spatialomicsgym.task``, ...) resolves to the same module object here through
``spatialomicsgym._aliases``. New benchmark work belongs under ``agent/benchmarks/``.

Docstring only.
"""
