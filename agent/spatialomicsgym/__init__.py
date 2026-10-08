import os as _os

# Stdlib-only (``re``), so importing the package stays instant. `clean_answer` is here because the
# documented pattern -- `log, answer = agent.go(task)` -- hands back the RAW final message, which in
# every recorded live session carried `<solution>` tags, a routing `Classification:` line and the
# Deliberative-Thinking-Protocol scaffold. `go()` cannot clean its own return (benchmarks and
# huggingface_data/build_dataset.py parse `<solution>` out of it), so the cleaner is exposed here
# instead: `from spatialomicsgym import clean_answer`. `answer` is bound as well, so
# `spatialomicsgym.answer` stays an attribute of the package now that the module lives in turn/.
from .turn import answer  # noqa: F401
from .turn.answer import clean_answer
from .version import __version__


def mirror_legacy_env() -> None:
    """Mirror legacy BIOMNI_* variables onto their SOG_* names (the new name wins).

    For existing .env files after the biomni->spatialomicsgym rename. Run here for the shell's
    variables, and again by every door that loads a .env: this package runs before any of them, so
    a BIOMNI_* key living only in a .env was never mirrored, which is the case the mirror is for
    (u16-llm-config-22).
    """
    for key in list(_os.environ):
        if key.startswith("BIOMNI_"):
            _os.environ.setdefault("SOG_" + key[len("BIOMNI_") :], _os.environ[key])


mirror_legacy_env()

# Old import names: ``spatialomicsgym.webui`` -> ``sog_portal``, ``spatialomicsgym.llm`` ->
# ``spatialomicsgym.providers.llm``, ... (lazy; the full table is in _aliases).
from . import _aliases as _aliases_mod

_aliases_mod.install()

__all__ = ["__version__", "clean_answer", "mirror_legacy_env"]
