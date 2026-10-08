"""``python -m sog_install`` → the setup CLI.

Kept trivial so the module entry point and the ``sog-setup`` console script share
exactly one code path (:func:`sog_install.cli.main`).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
