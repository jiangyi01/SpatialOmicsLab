"""Backed sparse indexing for anndata 0.12.x under scipy >= 1.17 (FM-03).

WHAT BREAKS WITHOUT THIS. ``anndata._core.sparse_dataset.validate_indices`` calls
``mtx._validate_indices(indices)`` as a **method** of the backed CSR/CSC matrix. SciPy 1.17 turned
that method into a module-level function, ``scipy.sparse._index._validate_indices(key, shape,
format)``, and anndata pins ``scipy>=1.12`` with no upper bound -- the agent-core env holds exactly
anndata 0.12.6 + scipy 1.17.1. Two failures follow, and the second is the one that costs answers:

1. Any index into a backed matrix raises ``AttributeError: 'backed_csr_matrix' object has no
   attribute '_validate_indices'`` -- even ``adata.X[:5]``, the opening sniff of most trials.
2. ``AnnData.to_memory()`` collects ``X`` with ``getattr(self, "X", None)``, which **swallows** that
   AttributeError, so ``adata[cells, genes].to_memory()`` returns the right shape with ``X is None``.
   ``pd.DataFrame(None, columns=genes)`` then has zero rows, every marker's detection is ``NaN``,
   and the trial reports that a cell population is absent. No step of that involves the model
   reasoning badly. Measured: the error appears in 146 of 456 archived SpatialBench trajectories;
   ``BASELINE.md`` independently counts the silent form in 98 of 360 scored trials.

THE FIX re-expresses the one call over scipy's module-level function and returns its first element,
the validated index tuple -- which is what anndata took from the old method (``res[0] if
SCIPY_1_15``). Verified against the oracle this box happens to have: anndata 0.12.6 with scipy
1.16.3 (the last scipy with the method) in the ``COMMOT`` env, over every index shape in
``test/test_a_backed_matrix_can_be_read_under_scipy_1_17.py``.

WHEN IT APPLIES. A scipy that restores the method -- the backed matrix classes have it again -- is
left untouched: nothing is replaced. Otherwise anndata's ``validate_indices`` is WRAPPED, not
replaced: the wrapper calls anndata's own function first, and uses scipy's module-level function only
when that call raises the AttributeError naming ``_validate_indices``. An anndata that already
handles scipy 1.17 therefore answers every read with its own code, and the shim can only turn that
one AttributeError into a result.

The wrap is needed because whether anndata still makes the method call cannot be read off the
classes: under scipy 1.17 they lack the method in every anndata version, fixed or not. The first
version of this gate asked only that, and so REPLACED working functions. Measured 2026-09-25: five
envs on this box (anndata 0.12.10, 0.12.14 and 0.12.18, all with scipy 1.17.1) carry anndata's own
fix -- ``if hasattr(mtx, "_validate_indices") ... elif scipy >= 1.17`` -- and the old gate
overrode it in all five, which a bash cell reaches through ``sitecustomize`` whenever it starts one
of their interpreters. The bodies matched that day; a later anndata that does more would have been
silently overruled.
"""

from __future__ import annotations

from typing import Any

_PATCHED_ATTR = "_spatialomicsgym_scipy117_compat"


def _needs_patch(sd: Any) -> bool:
    classes = [getattr(sd, name, None) for name in ("backed_csr_matrix", "backed_csc_matrix")]
    classes = [c for c in classes if c is not None]
    return bool(classes) and not all(hasattr(c, "_validate_indices") for c in classes)


def ensure_backed_sparse_indexing() -> bool:
    """Make backed sparse indexing work in this process if (and only if) it is broken.

    Returns True when backed indexing is usable afterwards -- because it already was, or because the
    shim is now in place -- and False when anndata/scipy are absent or have moved in a way this does
    not recognise. Idempotent and cheap after the first call. Never raises.
    """
    try:
        import anndata._core.sparse_dataset as sd
    except Exception:
        return False
    current = getattr(sd, "validate_indices", None)
    if current is None:
        return False
    if getattr(current, _PATCHED_ATTR, False) or not _needs_patch(sd):
        return True
    try:
        from scipy.sparse._index import _validate_indices as _module_level
    except Exception:
        return False

    def validate_indices(mtx: Any, indices: Any) -> Any:
        # anndata's own answer whenever it has one. Only the missing-method error is ours to repair;
        # any other AttributeError is anndata's and goes to the caller unchanged.
        try:
            return current(mtx, indices)
        except AttributeError as err:
            if "_validate_indices" not in str(err):
                raise
        return _module_level(indices, mtx.shape, mtx.format)[0]

    validate_indices.__doc__ = current.__doc__
    validate_indices.__wrapped_original__ = current  # so a test can prove the REPL re-applies it
    setattr(validate_indices, _PATCHED_ATTR, True)
    sd.validate_indices = validate_indices
    return True
