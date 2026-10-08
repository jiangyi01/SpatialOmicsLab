"""One adapter per communication backend, keyed by ``detect.BACKENDS``'s names."""

from __future__ import annotations

from . import commot, deeplinc, mistyr, ncem, neighborseq, spacet, spaotsc, squidpy

ADAPTERS = {
    module.ADAPTER.backend: module.ADAPTER
    for module in (commot, squidpy, neighborseq, deeplinc, spacet, mistyr, ncem, spaotsc)
}

__all__ = ["ADAPTERS"]
