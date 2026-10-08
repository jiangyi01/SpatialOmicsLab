"""Serial sections into one 3D object: diagnose, align, validate, then analyse.

Ten alignment portals already ship here. What did not exist was everything around them -- a way to
ask whether a stack needs aligning at all, an agreed place to put the answer, and a way to show
that the alignment helped. This package is that layer, and it is deliberately the only part of the
3D path that knows where a coordinate lives.

The modules, in the order a study moves through them:

``contract``    the keys, the provenance block, and the rule that ``obsm['spatial']`` survives
``profile``     per-slice coordinate metadata: units, ranges, pitch, coverage, slice order
``geometry``    adjacent-pair geometry -- centroid, principal axis, scale, outline overlap
``biology``     adjacent-pair expression agreement, against a null that makes it readable
``batch``       per-slice expression differences, computed without touching a coordinate
``thresholds``  every number the classifier compares against, with where it was measured
``classify``    A (already aligned) / B (rigid) / C (non-rigid) / unknown, with its evidence
``adapters``    where each of the shipped aligners actually left its answer
``validate``    the same metrics again, before and after, and whether any pair got worse
``pipelines``   one function per portal call

Two separations are structural rather than conventional, and a test asserts each. ``geometry``
imports nothing from ``biology``, and ``batch`` imports nothing from ``geometry``: coordinate
misalignment and expression batch effects are different findings with different remedies, and a
module that could reach both would eventually report one as the other. And ``validate`` calls the
same functions ``classify`` does -- a second implementation of "the same metric, after" is how a
before/after comparison quietly stops being one.

**Importing this package does not import anndata, scipy or matplotlib.** It is read by two portals
that must stay importable where the analysis stack is absent, so every heavy import sits inside a
function body.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "adapters",
    "batch",
    "biology",
    "classify",
    "contract",
    "geometry",
    "pipelines",
    "profile",
    # 'report' was listed here and in the docstring, but no such module exists, so a star import
    # raised ModuleNotFoundError (hunt 2026-09-30, u21-3d-16).
    "thresholds",
    "validate",
]


def __getattr__(name: str) -> Any:
    """Import a submodule on first use, so the package itself stays cheap."""
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
