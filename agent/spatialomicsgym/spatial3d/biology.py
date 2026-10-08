"""Do adjacent sections agree about where the genes are, and how would we know?

Geometry can be satisfied by two sections that overlap perfectly and share no biology -- a
rectangle on a rectangle. This module asks the other question: at matched locations, do the same
genes have the same values? It is what separates "these outlines coincide" from "this is the same
tissue, continued".

Four decisions carry it.

**A few genes, all cells -- never a cell subset.** The Zhuang-ABCA-1 matrix is chunked at
(16281, 5): five genes across sixteen thousand cells per chunk. Reading one gene for every one of
4.2 million cells costs ``n / 16281`` chunk reads and is cheap; reading every gene for a subset of
cells touches nearly every chunk and is pathological. The whole module is built around the first
access pattern, which is also why there is no downsampling anywhere in it.

**"Robust" is a filter, not an adjective.** A gene qualifies only if it is detected in at least
:data:`MIN_DETECTION_RATE` of cells in *every* section, and if its per-section mean is stable.
A gene dominated by a section-level shift is precisely the gene that makes misaligned sections
look correlated and aligned ones look uncorrelated -- it carries the batch, not the anatomy.
Genes rejected on the second test are not discarded; they are handed to :mod:`batch`, and both
lists are reported.

**Normalisation happens within a section, before anything is compared across sections.** A
difference in sequencing depth between two sections would otherwise enter the correlation and be
read as a difference in tissue. That single choice is what keeps this module's answer independent
of :mod:`batch`'s.

**The number reported is a gap against a null, not a correlation.** A Spearman rho of 0.3 between
adjacent sections is uninterpretable on its own, because some of it is merely that two sections of
one organ have similar composition wherever you lay them. So the same computation is run again
with one section rotated ninety degrees about its centroid -- a wrong but entirely plausible
registration -- and the statistic is the difference. The null is also a second measurement through
the identical pipeline, so a metric that has silently become a constant reports a gap of exactly
zero instead of a healthy-looking correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: A gene must be detected in at least this fraction of cells in *every* section to be eligible.
#: Five per cent: low enough to keep a regional marker that is absent from most of the tissue,
#: high enough to drop a gene whose correlation would be computed over a handful of cells.
MIN_DETECTION_RATE = 0.05

#: A gene whose per-section mean varies by more than this (as a coefficient of variation across
#: sections, after within-section normalisation) is treated as carrying a batch effect and is
#: routed to the batch report instead of the consistency metric.
MAX_SECTION_CV = 0.35

#: Neighbours used to smooth a gene within its own section. Fifteen is a local average over
#: roughly two rings of a hexagonal lattice -- enough to suppress per-cell dropout, small enough
#: that a domain boundary is not averaged away.
SMOOTH_K = 15

#: A nearest-neighbour match further than this many median spacings apart is not a match. Without
#: the rejection a cell is paired with something across the tissue and the correlation becomes
#: noise with a respectable-looking value.
MATCH_RADIUS_PITCHES = 3.0

#: Below this matched fraction the pair has no shared region to compare and returns a refusal
#: rather than a number computed over a sliver.
MIN_MATCHED_FRACTION = 0.20

#: How many robust genes the consistency metric uses. The request asks for five to ten.
N_GENES = 8

#: Cap on the gene scan when no highly-variable flag is stored. A 1,122-gene panel is fully
#: scannable; a 30,000-gene transcriptome is not, and a capped scan that says so is honest.
MAX_GENE_SCAN = 2000


@dataclass
class GeneSelection:
    """Which genes were chosen, which were rejected, and why."""

    chosen: list[str] = field(default_factory=list)
    rejected_batchy: list[str] = field(default_factory=list)
    rejected_sparse: list[str] = field(default_factory=list)
    rejected_nonfinite: list[str] = field(default_factory=list)
    scanned: int = 0
    scan_capped: bool = False
    source: str = ""


@dataclass
class PairBiology:
    """Expression agreement between one adjacent pair, against its null."""

    a: str
    b: str
    bio_consistency: float = float("nan")
    bio_null: float = float("nan")
    bio_gap: float = float("nan")
    matched_fraction: float = 0.0
    n_genes: int = 0
    refusal: str = ""
    per_gene: dict[str, float] = field(default_factory=dict)


class AnnDataGenes:
    """Gene-wise access to an in-memory or backed AnnData."""

    def __init__(self, adata: Any):
        self._a = adata
        self.genes = [str(g) for g in adata.var_names]
        self.source = "anndata"

    def column(self, gene: str) -> Any:
        import numpy as np

        j = self.genes.index(gene)
        col = self._a.X[:, j]
        col = col.toarray() if hasattr(col, "toarray") else np.asarray(col)
        return np.asarray(col, dtype=float).ravel()


class H5adGenes:
    """Gene-wise access straight through h5py, for a matrix too large to hold.

    Only the dense case is handled here, because that is the shape the large atlases arrive in
    (Zhuang-ABCA-1 is a dense float32 of 4,167,870 x 1,122). A sparse file raises rather than
    silently falling back to loading everything.
    """

    def __init__(self, path: str):
        import h5py

        self._path = path
        self._f = h5py.File(path, "r")
        x = self._f["X"]
        if not hasattr(x, "shape") or len(getattr(x, "shape", ())) != 2:
            raise ValueError(f"{path}: X is not a dense 2-D dataset; gene-wise streaming needs one")
        self._x = x
        var = self._f["var"]
        idx = var.attrs.get("_index", "_index")
        raw = var[idx][:]
        self.genes = [g.decode() if isinstance(g, bytes) else str(g) for g in raw]
        self.source = f"h5py:{path}"

    def column(self, gene: str) -> Any:
        import numpy as np

        return np.asarray(self._x[:, self.genes.index(gene)], dtype=float).ravel()

    def close(self) -> None:
        self._f.close()


def _cpm_log1p_within(values: Any, sections: Any, totals: Any) -> Any:
    """log1p of counts-per-million, with the library size taken within each section.

    This is where the depth difference between sections is removed, and it is removed before any
    cross-section comparison happens rather than after.
    """
    import numpy as np

    out = np.zeros(len(values), dtype=float)
    for s in np.unique(sections):
        m = sections == s
        tot = totals[m]
        scale = np.median(tot[tot > 0]) if (tot > 0).any() else 1.0
        out[m] = np.log1p(values[m] / np.maximum(tot, 1.0) * max(scale, 1.0))
    return out


def select_genes(
    source: Any,
    sections: Any,
    totals: Any,
    *,
    n_genes: int = N_GENES,
    candidates: list[str] | None = None,
) -> GeneSelection:
    """Pick robust, spatially usable genes, and say what was rejected and why."""
    import numpy as np

    sel = GeneSelection(source=getattr(source, "source", ""))
    totals = np.asarray(totals, dtype=float)
    names = list(candidates or source.genes)
    sel.scan_capped = len(names) > MAX_GENE_SCAN
    if sel.scan_capped:
        names = names[:MAX_GENE_SCAN]
    sel.scanned = len(names)

    uniq = list(np.unique(sections))
    scored: list[tuple[float, str]] = []
    for g in names:
        try:
            raw = source.column(g)
        except Exception:
            continue
        # Non-finite BEFORE anything else, and in a bucket of its own.
        #
        # Measured on the Moffitt hypothalamus atlas: exactly one of its 156 gene columns is NaN
        # for every cell. That single column makes each cell's library size NaN, which makes every
        # per-section mean NaN, which makes every coefficient of variation infinite -- so all 155
        # readable genes were rejected, and the report said "155 genes carry a section-level
        # shift". One unreadable column silently disabled the whole expression metric and the
        # failure was presented as a finding about the biology.
        if not np.isfinite(raw).all():
            sel.rejected_nonfinite.append(g)
            continue
        # Detected everywhere? A gene missing from one section cannot describe its continuity.
        if any((raw[sections == s] > 0).mean() < MIN_DETECTION_RATE for s in uniq):
            sel.rejected_sparse.append(g)
            continue
        v = _cpm_log1p_within(raw, sections, totals)
        means = np.array([v[sections == s].mean() for s in uniq])
        mu = float(means.mean())
        cv = float(means.std() / mu) if mu > 0 else float("inf")
        if cv > MAX_SECTION_CV:
            sel.rejected_batchy.append(g)
            continue
        # Rank by within-section variance: a gene that is flat inside a section carries no
        # spatial pattern for an alignment to preserve, however abundant it is.
        within = float(np.mean([v[sections == s].var() for s in uniq]))
        scored.append((within, g))

    scored.sort(reverse=True)
    sel.chosen = [g for _, g in scored[:n_genes]]
    return sel


def smooth_within_section(values: Any, xy: Any, k: int = SMOOTH_K) -> Any:
    """Mean of a gene over each cell's k in-plane neighbours, within one section."""
    import numpy as np
    from scipy.spatial import cKDTree

    a = np.asarray(xy, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(a) <= 1:
        return v
    kk = min(k, len(a) - 1)
    _, nb = cKDTree(a).query(a, k=kk + 1)
    return v[nb].mean(axis=1)


def _matched(a_xy: Any, b_xy: Any) -> tuple[Any, Any, float]:
    import numpy as np
    from scipy.spatial import cKDTree

    from . import geometry as geom

    pitch = max(geom.median_pitch(a_xy), geom.median_pitch(b_xy), 1e-12)
    d, j = cKDTree(a_xy).query(b_xy)
    ok = d <= MATCH_RADIUS_PITCHES * pitch
    return np.flatnonzero(ok), j[ok], float(ok.mean())


def _rotate90(xy: Any) -> Any:
    """Ninety degrees about the trimmed centroid: a wrong but plausible registration."""
    import numpy as np

    from . import geometry as geom

    c = geom.trimmed_centroid(xy)
    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    return (np.asarray(xy, float) - c) @ R.T + c


def pair_biology(
    name_a: str,
    xy_a: Any,
    smoothed_a: dict[str, Any],
    name_b: str,
    xy_b: Any,
    smoothed_b: dict[str, Any],
) -> PairBiology:
    """Spearman agreement at nearest-neighbour-matched locations, minus its rotated null."""
    import numpy as np
    from scipy.stats import spearmanr

    out = PairBiology(a=name_a, b=name_b, n_genes=len(smoothed_a))
    a = np.asarray(xy_a, float)[:, :2]
    b = np.asarray(xy_b, float)[:, :2]
    if not smoothed_a or len(a) < 10 or len(b) < 10:
        out.refusal = "too few cells or no eligible gene to compare"
        return out

    def rho_for(b_frame: Any) -> float:
        bi, aj, frac = _matched(a, b_frame)
        if frac < MIN_MATCHED_FRACTION or len(bi) < 10:
            return float("nan")
        vals = []
        for g in smoothed_a:
            r = spearmanr(smoothed_b[g][bi], smoothed_a[g][aj]).statistic
            if np.isfinite(r):
                vals.append(float(r))
                if b_frame is b:
                    out.per_gene[g] = float(r)
        return float(np.median(vals)) if vals else float("nan")

    _, _, frac = _matched(a, b)
    out.matched_fraction = frac
    if frac < MIN_MATCHED_FRACTION:
        out.refusal = (
            f"only {frac:.1%} of section {name_b} found a neighbour in {name_a} within "
            f"{MATCH_RADIUS_PITCHES:g} spacings, so there is no shared region to compare"
        )
        return out

    out.bio_consistency = rho_for(b)
    out.bio_null = rho_for(_rotate90(b))
    if np.isfinite(out.bio_consistency) and np.isfinite(out.bio_null):
        out.bio_gap = out.bio_consistency - out.bio_null
    return out
