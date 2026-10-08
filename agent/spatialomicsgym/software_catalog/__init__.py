"""The data-lake and software lists the system prompt shows: ``env_desc`` (academic) and ``env_desc_cm`` (commercial).

``spatialomicsgym.env_desc`` and ``spatialomicsgym.env_desc_cm`` resolve to the same module objects here through
``spatialomicsgym._aliases``. The identity matters: code and tests compare and patch these module-level dicts, so
both names must reach one object.

Docstring only.
"""
