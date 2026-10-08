"""Startup hook for a Python process a model's bash cell launches (FM-03, L-1b; N5).

The backed-indexing shim (``spatialomicsgym/utils/anndata_compat.py``) is applied inside the agent's
REPL, but a model also runs Python from a ``#!BASH`` cell -- ``python - <<'PY' ... PY`` -- and that is
a fresh interpreter the shim never reached: 4 of the 21 E-01 trials with the error, and 12 of 146
in the archive, raised it only there. ``run_bash_script`` puts this directory first on the child's
``PYTHONPATH``, and Python imports the first ``sitecustomize`` it finds at startup.

It costs nothing to a script that never touches anndata: nothing is imported until
``anndata._core.sparse_dataset`` is, and then the shim is loaded from its file -- not as
``spatialomicsgym``, because the child may be another conda env's interpreter -- and applied only
where that env's scipy lacks the method; where its anndata already copes, anndata's own code keeps
answering (see the shim's WHEN IT APPLIES). Any other ``sitecustomize`` further along the path still
runs. Written for every Python the box has (3.7 in the oldest worker env), and it never raises.

The same watch gives ``pandas`` the REPL's display options the moment it has been imported
(``spatialomicsgym/utils/pandas_display.py``, loaded from its file the same way), so a frame the child
prints wraps into blocks instead of losing its middle columns to an 80-column screen nobody has. A
script that never imports pandas never loads it. The wrapped loader answers everything else about the
module as the real one does -- ``pkgutil.get_data("pandas", ...)`` asks it for ``get_data``.
"""

import importlib.abc
import importlib.util
import os
import sys

_TARGET = "anndata._core.sparse_dataset"
_PANDAS = "pandas"
_HERE = os.path.dirname(os.path.abspath(__file__))
_COMPAT = os.path.join(os.path.dirname(_HERE), "anndata_compat.py")
_DISPLAY = os.path.join(os.path.dirname(_HERE), "pandas_display.py")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_shim(module):
    try:
        _load("_spatialomicsgym_anndata_compat", _COMPAT).ensure_backed_sparse_indexing()
    except Exception:
        pass


def _apply_display(module):
    try:
        _load("_spatialomicsgym_pandas_display", _DISPLAY).apply_readable_display(module)
    except Exception:
        pass


class _ThenShim(importlib.abc.Loader):
    def __init__(self, inner, then):
        self._inner = inner
        self._then = then

    def __getattr__(self, name):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._then(module)


class _WatchImport(importlib.abc.MetaPathFinder):
    """Watches for the import of ``name`` and runs ``then`` on the module once it has executed.

    It stays on ``sys.meta_path`` until the module has executed, not until it is first looked up: a
    lookup -- ``importlib.util.find_spec``, ``pkgutil.find_loader``, a library checking for an optional
    dependency -- reaches this finder as the import does and executes nothing, and a watch that left
    at the first one was gone before the import that followed it.
    """

    def __init__(self, name, then):
        self.name = name
        self.then = then
        self.looking = False

    def find_spec(self, name, path, target=None):
        if name != self.name or self.looking:
            return None
        self.looking = True  # the lookup below must not find us; a finder runs under the import lock
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            return None
        finally:
            self.looking = False
        if spec is not None and spec.loader is not None:
            spec.loader = _ThenShim(spec.loader, self._executed)
        return spec

    def _executed(self, module):
        try:
            sys.meta_path.remove(self)  # once is enough
        except ValueError:
            pass
        self.then(module)


def _run_the_next_sitecustomize():
    for entry in sys.path:
        try:
            if not entry or os.path.abspath(entry) == _HERE:
                continue
            candidate = os.path.join(entry, "sitecustomize.py")
            if os.path.isfile(candidate):
                spec = importlib.util.spec_from_file_location("_spatialomicsgym_next_sitecustomize", candidate)
                spec.loader.exec_module(importlib.util.module_from_spec(spec))
                return
        except Exception:
            return


try:
    for _name, _then in ((_TARGET, _apply_shim), (_PANDAS, _apply_display)):
        if not any(isinstance(f, _WatchImport) and f.name == _name for f in sys.meta_path):
            sys.meta_path.insert(0, _WatchImport(_name, _then))
except Exception:
    pass
_run_the_next_sitecustomize()
