"""HVG pre-filter for SVG-detection inputs.

Centralizes the "top ~1500 Seurat HVGs before any SVG tool" step.
Used by:
  - benchmarks/workflows/benchmark_runner.py (agent / multi-LLM pipeline)
  - hands_by_myself/runners/_common.py (manual baseline)

Both call `prepare_svg_input_hvg(input_h5ad)` at dataset-load time so every SVG
worker / MCP server receives the SAME HVG-filtered AnnData. This replaces:
  - the SVG_HVG_PREFILTER paragraph that used to be appended to the STCoscientist prompt
  - per-tool internal HVG selection (which was inconsistent across tools)
  - post-hoc HVG re-filter (still useful for legacy results, not for new runs)

Cache: written next to input as `<stem>_hvg<N>.h5ad`. Treated as fresh if its
mtime ≥ original h5ad's mtime AND its `uns['_hvg_n']` matches `n_hvg`.

Skip: if `n_vars <= n_hvg` (e.g. MERFISH 648-gene targeted panel) the original
path is returned unchanged — no point selecting "top 1500" out of 648.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

_CACHE_VERSION = 2


def prepare_svg_input_hvg(input_path: str | Path, n_hvg: int = 1500) -> Path:
    """Return path to a HVG-filtered copy of the input spatial h5ad.

    Args:
        input_path: original spatial_transcriptomics.h5ad
        n_hvg: number of top Seurat HVGs to keep (default 1500)

    Returns:
        Path to HVG-filtered h5ad. Same as `input_path` if `n_vars <= n_hvg`.
    """
    import anndata as ad
    import scanpy as sc

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"SVG input h5ad not found: {input_path}")

    cache_path = input_path.with_name(f"{input_path.stem}_hvg{n_hvg}.h5ad")

    if cache_path.exists() and cache_path.stat().st_mtime >= input_path.stat().st_mtime:
        try:
            import h5py

            with h5py.File(cache_path, "r") as f:
                uns = f.get("uns", {})
                hvg_n = uns["_hvg_n"][()] if "_hvg_n" in uns else None
                ver = uns["_hvg_cache_version"][()] if "_hvg_cache_version" in uns else None
            if hvg_n is not None and int(hvg_n) == n_hvg and int(ver) == _CACHE_VERSION:
                log.info("SVG HVG cache hit: %s", cache_path.name)
                return cache_path
            log.info("SVG HVG cache stale or unstamped; regenerating")
        except Exception as e:
            log.warning("SVG HVG cache probe failed (%s); regenerating", e)

    log.info("Computing SVG HVG cache: %s (n_hvg=%d) ...", input_path.name, n_hvg)
    a_raw = ad.read_h5ad(input_path)

    if a_raw.n_vars <= n_hvg:
        log.info(
            "Skipping HVG filter: dataset has %d genes ≤ n_hvg=%d (likely targeted panel); "
            "returning original input unchanged",
            a_raw.n_vars,
            n_hvg,
        )
        return input_path

    a_norm = a_raw.copy()
    sc.pp.normalize_total(a_norm, target_sum=1e4)
    sc.pp.log1p(a_norm)
    sc.pp.highly_variable_genes(a_norm, n_top_genes=n_hvg, flavor="seurat")
    hvg_mask = a_norm.var["highly_variable"].values

    a = a_raw[:, hvg_mask].copy()
    a.var["highly_variable"] = True
    a.uns["_hvg_n"] = n_hvg
    a.uns["_hvg_flavor"] = "seurat"
    a.uns["_hvg_source_h5ad"] = str(input_path)
    a.uns["_hvg_cache_version"] = _CACHE_VERSION

    # Written beside the dataset, where a concurrent benchmark arm may be probing or reading it: never
    # at the final name. A probe that met a half-written cache "failed", regenerated and wrote the
    # same path under the first writer -- an HDF5 lock error out of a call no try guards, or a
    # truncated input handed to a worker (hunt 2026-09-30, u31-benchmarking-16). A per-process
    # partial name plus os.replace means a reader sees no file or a whole one; two arms racing both
    # write whole files and the last replace wins, which is the same file.
    partial = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.partial.h5ad")
    try:
        a.write_h5ad(partial)
        os.replace(partial, cache_path)
    finally:
        if partial.exists():
            partial.unlink()
    log.info("SVG HVG cache written: %s (n_vars=%d, raw counts preserved)", cache_path.name, a.n_vars)
    return cache_path
