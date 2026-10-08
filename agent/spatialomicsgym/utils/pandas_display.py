"""Pandas display options for output a model reads rather than a terminal (N5).

WHAT BREAKS WITHOUT THIS. A cell's stdout is captured -- a ``StringIO`` in the REPL, a pipe in a
bash cell -- and pandas cannot tell. Outside IPython it assumes a terminal: ``display.max_columns``
defaults to 0, meaning "fit the screen", it asks ``shutil.get_terminal_size()`` how wide the screen
is, gets the 80-column fallback, and drops the middle columns of any wider frame to fit, leaving one
``...`` column where they were. The only trace is the ``[13 rows x 9 columns]`` line under the
table. Measured in the E-03 trials: an 8 x 8 crosstab reached the model as 4 columns and DR01 r2
ticked off the check it was printed for with half its diagonal unseen; a 9-column marker table with a
30-character index arrived as its first and last columns (qc_01 r1); NORM01 r1 lost a check the same
way.

THE FIX turns "fit the screen" off. With ``max_columns`` non-zero pandas no longer drops a column to
fit: it wraps a frame wider than ``display.width`` into a further block, and every column is printed.
Row truncation is left alone -- pandas marks it with a ``...`` row, and 60 rows is its own bound.

Loaded two ways: imported by ``support_tools`` for the REPL (in-process and the worker), and by file
path from ``child_site/sitecustomize.py`` in a Python a bash cell starts, which may be another conda
env's interpreter -- so this file imports nothing from ``spatialomicsgym`` and never imports pandas at
module import. Written for the oldest Python and pandas on the box (3.7, pandas 1.1).
"""

from __future__ import annotations

#: Characters per line before a printed frame wraps into another block of columns. Wrapping loses
#: nothing and adds only the index repeated per block, so this decides layout, not content or cost.
#: 250 keeps a typical 8-15 column table in one block: 15 columns of ~12-character names and values
#: beside a 30-character cell-type index is ~220.
WIDTH = 250

#: Columns shown before pandas elides the middle ones -- and says so, with a ``...`` column and the
#: ``[rows x columns]`` line. Twice the widest typical table, so an ``obs`` frame of 20-odd fields
#: prints whole, while a frame built from a gene matrix prints 30 columns rather than 30,000. More
#: would not show more: 60 rows (pandas' row cap) of 30 columns is ~20K characters, already twice
#: what ``clip_observation`` lets through.
MAX_COLUMNS = 30

#: Characters of one cell before pandas ends it with ``...``. Pandas' 50 cuts a typical absolute path
#: or a short comma-joined gene list mid-name; 100 holds those, and one text blob still cannot pad
#: every row of its column out to a page.
MAX_COLWIDTH = 100

OPTIONS = {"display.width": WIDTH, "display.max_columns": MAX_COLUMNS, "display.max_colwidth": MAX_COLWIDTH}

#: Set on the pandas module once the options are applied: they are applied once per process.
_APPLIED_ATTR = "_spatialomicsgym_readable_display"


def apply_readable_display(pd=None) -> bool:
    """Give this process's pandas :data:`OPTIONS` -- once, and only where nobody chose otherwise.

    ``pd`` is the pandas module when the caller already holds it (the child's import hook); otherwise
    pandas is imported here, and a process without pandas is left alone. Returns True when pandas is
    present. Never raises.

    Once, because the REPL calls this before every cell: a cell that sets its own display options,
    or resets one to pandas' default, keeps them for the rest of the session. Only where nobody chose
    otherwise, because the in-process REPL shares pandas with whoever hosts the agent: an option a
    notebook or an operator already moved off pandas' default is left as they set it.
    """
    if pd is None:
        try:
            import pandas as pd
        except Exception:
            return False
    if getattr(pd, _APPLIED_ATTR, False):
        return True
    try:
        from pandas._config.config import get_default_val
    except Exception:  # a pandas that moved it: no way to tell a choice from a default, so set all
        get_default_val = None
    for key, value in OPTIONS.items():
        try:
            if get_default_val is None or pd.get_option(key) == get_default_val(key):
                pd.set_option(key, value)
        except Exception:
            pass
    try:
        setattr(pd, _APPLIED_ATTR, True)
    except Exception:
        pass
    return True
