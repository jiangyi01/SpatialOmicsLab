"""Adjacent-section geometry: how far apart, how turned, how stretched, how much overlap.

Coordinates only. This module never opens an expression matrix and imports nothing from
``biology``, because "the sections are in different places" and "the sections were sequenced
differently" are different findings with different remedies, and a module that could reach both
would eventually report one as the other.

Every length is reported twice: in the coordinates' own units, and as a fraction of the pooled
robust bounding-box diagonal. The fraction is the one thresholds are written against, because the
unit verdict is frequently ``unknown`` and a threshold in micrometres cannot be compared against a
number that might be pixels.

Four of the decisions here are the module rather than details of it.

**The principal-axis angle is folded into [0, 90] degrees, and that folding *is* the 180-degree
disambiguation.** An eigenvector's sign is arbitrary, so theta and 180 - theta are the same
physical statement; ``arccos(|v_i . v_j|)`` says so. The *signed* rotation is not recoverable from
PCA at all and is estimated separately, from correspondences, and reported as an estimate with a
residual. A recorded run in this repository reported rotations of 136.9, 176.1 and -158.0 degrees
between coronal sections fifty micrometres apart, every one of them with a negative improvement
score. Those were eigenvector sign flips presented as anatomy.

**A near-isotropic slice has no principal axis and the angle is refused.** Below an eigenvalue
ratio of :data:`ISOTROPY_FLOOR` the leading eigenvector is noise, and an angle computed from it
varies wildly between two slices that are the same shape. This is the single largest way the
metric lies, so it returns ``None`` and a reason.

**Overlap is binned occupancy, not an alpha shape.** An alpha shape or concave hull needs
``alphashape`` or full ``shapely``, neither of which is guaranteed in the agent environment, and
both are super-linear. ``numpy.histogram2d`` is O(n), which is what makes 4.2 million cells free,
and its one parameter -- the bin size -- is disclosed in the output rather than hidden. Its
stability is checked rather than assumed.

**Non-rigidity is measured as disagreement between local block shifts, not as a residual.** Two
residual-based discriminators were implemented and measured on the serial-section fixtures before
this one, and both were removed for not varying with the thing they named. The numbers are in
:func:`local_shift_dispersion`. The short version is that the distance between two serial sections
has a floor -- they are different cells -- and that floor is larger than the deformation.

**Scale comes from the convex hull, not the bounding box.** A rotated tissue has a larger bounding
box and the same hull, so a bbox ratio confounds rotation with scale. The aspect ratio is reported
separately, and it is what separates an anisotropic stretch (class C) from a uniform rescale
(class B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Below this ratio of the two coordinate eigenvalues the leading eigenvector is not identified.
#: 1.3 is a shape whose long axis is only 14% longer than its short one once the square root is
#: taken -- for a tissue section that is round enough that "which way is it pointing" has no
#: answer, and any angle computed from it is noise dressed as a measurement.
ISOTROPY_FLOOR = 1.3

#: The robust bounding box. Percentiles rather than min/max: stage coordinates routinely carry a
#: handful of cells far outside the tissue, and one of them moves a min by more than the tissue.
BBOX_LO_PCT = 1.0
BBOX_HI_PCT = 99.0

#: Trim for the centroid, per axis, for the same reason.
CENTROID_TRIM = 0.10

#: Occupancy bin edge, as a multiple of the sparser section's median nearest-neighbour spacing.
#: Two, so a bin holds a point and its neighbours rather than isolating every point in its own
#: cell -- at one, a jittered resample of the same tissue overlaps itself only by chance.
BIN_PITCH_MULTIPLE = 2.0

#: The occupancy grid never exceeds this on a side, so a large tissue with a fine pitch cannot
#: allocate an enormous array.
MAX_GRID = 512

#: If IoU at min_count 1 and 2 differ by more than this, the estimate depends on the bin size and
#: the classifier is told to treat it as weak evidence rather than a measurement.
BIN_SENSITIVITY_TOLERANCE = 0.05

#: A correspondence further than this many median-NN-spacings apart is not a correspondence.
MATCH_RADIUS_PITCHES = 3.0

#: The local-shift grid. Four on a side: coarse enough that a block holds enough cells for its
#: centroid to mean something, fine enough that a smooth deformation makes neighbouring blocks
#: disagree. See :func:`local_shift_dispersion`.
LOCAL_BLOCKS = 4
LOCAL_MIN_PER_BLOCK = 15
LOCAL_MIN_BLOCKS = 4

#: Iterations of the alternating correspondence/fit loop. Small: this is an estimate reported with
#: its residual, not a registration, and a long loop would invite reading it as one.
FIT_ITERATIONS = 8

#: A similarity fit must beat translation-only by this factor on the residual before its rotation
#: and scale are reported as anything but ``translation_only``. Without it every pair acquires a
#: spurious rotation, which is exactly the recorded failure above.
SIMILARITY_MARGIN = 0.95


@dataclass
class PairGeometry:
    """Every geometric statement about one adjacent pair, with its refusals."""

    a: str
    b: str
    n_a: int = 0
    n_b: int = 0
    scale_length: float = 0.0
    centroid_offset: float = 0.0
    centroid_offset_frac: float = 0.0
    axis_angle_deg: float | None = None
    axis_refusal: str = ""
    eigenvalue_ratio_a: float = 0.0
    eigenvalue_ratio_b: float = 0.0
    hull_area_ratio_log2: float = 0.0
    aspect_change_log2: float = 0.0
    iou: float = 0.0
    iou_recentred: float = 0.0
    containment: float = 0.0
    bin_edge: float = 0.0
    bin_sensitive: bool = False
    resid_rigid: float = float("nan")
    local_shift_dispersion: float = float("nan")
    matched_fraction: float = 0.0
    fit_mode: str = "not_fitted"
    fit_rotation_deg: float = 0.0
    fit_tx: float = 0.0
    fit_ty: float = 0.0
    fit_scale: float = 1.0
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row = dict(self.__dict__)
        row["notes"] = "; ".join(self.notes)
        return row


def median_pitch(xy: Any) -> float:
    """Median nearest-neighbour distance: the natural length scale of a point set."""
    import numpy as np
    from scipy.spatial import cKDTree

    a = np.asarray(xy, dtype=float)
    if len(a) < 2:
        return 0.0
    d, _ = cKDTree(a).query(a, k=2)
    return float(np.median(d[:, 1]))


def robust_bbox(xy: Any) -> tuple[Any, Any]:
    import numpy as np

    a = np.asarray(xy, dtype=float)
    return np.percentile(a, BBOX_LO_PCT, axis=0), np.percentile(a, BBOX_HI_PCT, axis=0)


def bbox_diagonal(xy: Any) -> float:
    import numpy as np

    lo, hi = robust_bbox(xy)
    return float(np.hypot(*(hi - lo)))


def trimmed_centroid(xy: Any, trim: float = CENTROID_TRIM) -> Any:
    """Per-axis trimmed mean. Follows the tissue rather than the farthest stray cell."""
    import numpy as np

    a = np.asarray(xy, dtype=float)
    out = np.empty(a.shape[1])
    for i in range(a.shape[1]):
        col = np.sort(a[:, i])
        k = int(len(col) * trim)
        out[i] = col[k : len(col) - k].mean() if len(col) - 2 * k > 0 else col.mean()
    return out


def principal_axis(xy: Any) -> tuple[Any, float]:
    """The leading eigenvector of the coordinate covariance, and the eigenvalue ratio.

    The ratio is returned so the caller can decide the axis is not identified; this function does
    not make that decision, because the threshold belongs with the thresholds.
    """
    import numpy as np

    a = np.asarray(xy, dtype=float)
    c = a - a.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    order = np.argsort(w)[::-1]
    w, v = w[order], v[:, order]
    ratio = float(w[0] / w[1]) if w[1] > 0 else float("inf")
    return v[:, 0], ratio


def axis_angle_deg(va: Any, vb: Any) -> float:
    """The angle between two *axes*, in [0, 90].

    The absolute value of the dot product is the disambiguation, not a convenience: an eigenvector
    and its negation describe the same axis, so an angle of 170 degrees and one of 10 degrees are
    the same physical statement and both must come back as 10.
    """
    import numpy as np

    c = float(abs(np.dot(np.asarray(va, float), np.asarray(vb, float))))
    return float(np.degrees(np.arccos(min(1.0, max(0.0, c)))))


def hull_area(xy: Any) -> float:
    """Convex-hull area, or the robust bbox area when a hull cannot be built."""
    import numpy as np

    a = np.unique(np.asarray(xy, dtype=float), axis=0)
    if len(a) >= 3:
        try:
            from scipy.spatial import ConvexHull

            return float(ConvexHull(a).volume)  # 'volume' is area in 2D
        except Exception:
            pass
    lo, hi = robust_bbox(a)
    return float(max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0))


def _occupancy(a: Any, b: Any, bin_edge: float, min_count: int):
    import numpy as np

    lo = np.minimum(a.min(0), b.min(0)) - bin_edge
    hi = np.maximum(a.max(0), b.max(0)) + bin_edge
    nb = [max(2, min(MAX_GRID, int(np.ceil((hi[i] - lo[i]) / bin_edge)))) for i in (0, 1)]
    edges = [np.linspace(lo[i], hi[i], nb[i] + 1) for i in (0, 1)]
    A = np.histogram2d(a[:, 0], a[:, 1], bins=edges)[0] >= min_count
    B = np.histogram2d(b[:, 0], b[:, 1], bins=edges)[0] >= min_count
    inter = int((A & B).sum())
    union = int((A | B).sum())
    smaller = int(min(A.sum(), B.sum()))
    return (inter / union if union else 0.0), (inter / smaller if smaller else 0.0)


def occupancy_overlap(a: Any, b: Any) -> dict[str, Any]:
    """IoU, containment, recentred IoU, and whether the bin size is carrying the answer.

    ``containment`` is what tells partial overlap apart from displacement: a section that is half
    missing has a low IoU and a high containment, while a section that has merely slid has both
    low. ``iou_recentred`` is the cleanest single discriminator between a translation and a real
    difference in outline, and it costs one extra histogram.
    """
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bin_edge = BIN_PITCH_MULTIPLE * max(median_pitch(a), median_pitch(b), 1e-12)

    iou1, contain1 = _occupancy(a, b, bin_edge, 1)
    iou2, _ = _occupancy(a, b, bin_edge, 2)
    b_rec = b - trimmed_centroid(b) + trimmed_centroid(a)
    iou_rec, _ = _occupancy(a, b_rec, bin_edge, 1)

    return {
        "iou": float(iou1),
        "containment": float(contain1),
        "iou_recentred": float(iou_rec),
        "bin_edge": float(bin_edge),
        "bin_sensitive": bool(abs(iou1 - iou2) > BIN_SENSITIVITY_TOLERANCE),
    }


def binned_centroids(xy: Any, bin_edge: float) -> Any:
    """Occupied-bin centroids: the outline, with per-cell sampling noise averaged out.

    Fitting on raw points measures the wrong thing. Two serial sections are different cells, so
    even a perfect registration leaves a median nearest-neighbour distance of roughly half the
    spot pitch, and on a sparse section that floor is larger than any deformation. Averaging
    within a bin removes it while keeping the shape.
    """
    import numpy as np

    a = np.asarray(xy, dtype=float)
    if len(a) == 0 or bin_edge <= 0:
        return a
    key = np.floor(a / bin_edge).astype(np.int64)
    order = np.lexsort((key[:, 1], key[:, 0]))
    a, key = a[order], key[order]
    new = np.ones(len(a), dtype=bool)
    new[1:] = (key[1:] != key[:-1]).any(1)
    idx = np.flatnonzero(new)
    sums = np.add.reduceat(a, idx, axis=0)
    counts = np.diff(np.append(idx, len(a))).reshape(-1, 1)
    return sums / counts


def _umeyama(P: Any, Q: Any, with_scale: bool = True):
    """Least-squares similarity taking P onto Q. Returns (R, scale, t)."""
    import numpy as np

    pm, qm = P.mean(0), Q.mean(0)
    Pc, Qc = P - pm, Q - qm
    H = Pc.T @ Qc / len(P)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    var = (Pc**2).sum() / len(P)
    s = float((S * np.array([1.0, d])).sum() / var) if (with_scale and var > 0) else 1.0
    t = qm - s * (R @ pm)
    return R, s, t


def fit_similarity(a: Any, b: Any) -> dict[str, Any]:
    """Estimate the transform taking section ``b`` onto section ``a``, and say how well it did.

    There are no correspondences between two serial sections -- they are different cells -- so
    they are built by alternating nearest-neighbour matching with a closed-form fit, from a
    translation-only start. Matches beyond :data:`MATCH_RADIUS_PITCHES` spacings are dropped; an
    unrejected match pairs a cell with something on the other side of the tissue.

    The result is an *estimate with a residual*, never a fact. Its rotation and scale are reported
    only when the similarity fit beats translation-only by :data:`SIMILARITY_MARGIN`; otherwise the
    mode is ``translation_only`` and the angle is zero. That guard is the one the recorded Moffitt
    run lacked, and it is why nine of its eleven pairs got worse while it reported success.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out: dict[str, Any] = {
        "mode": "not_fitted",
        "rotation_deg": 0.0,
        "tx": 0.0,
        "ty": 0.0,
        "scale": 1.0,
        "resid": float("nan"),
        "matched_fraction": 0.0,
    }
    if len(a) < 3 or len(b) < 3:
        out["note"] = "fewer than three points in one section"
        return out

    pitch = max(median_pitch(a), median_pitch(b), 1e-12)
    # Fit on the outline, not on the cells -- see binned_centroids for why the raw-point residual
    # is a floor rather than a measurement.
    bin_edge = BIN_PITCH_MULTIPLE * pitch
    a_f, b_f = binned_centroids(a, bin_edge), binned_centroids(b, bin_edge)
    if len(a_f) < 3 or len(b_f) < 3:
        a_f, b_f = a, b
    radius = MATCH_RADIUS_PITCHES * pitch
    tree = cKDTree(a_f)

    def residual(moved):
        d, _ = tree.query(moved)
        ok = d <= radius
        return (float(np.median(d[ok])) if ok.any() else float("inf")), float(ok.mean())

    # Translation-only baseline.
    shift = trimmed_centroid(a_f) - trimmed_centroid(b_f)
    trans_resid, trans_frac = residual(b_f + shift)

    # Alternating correspondence and closed-form similarity, from that baseline.
    R_tot, s_tot, t_tot = np.eye(2), 1.0, shift.copy()
    moved = b_f + shift
    best = (trans_resid, np.eye(2), 1.0, shift.copy(), trans_frac)
    for _ in range(FIT_ITERATIONS):
        d, j = tree.query(moved)
        ok = d <= radius
        if ok.sum() < 3:
            break
        R, s, t = _umeyama(moved[ok], a_f[j[ok]])
        moved = (s * (moved @ R.T)) + t
        R_tot = R @ R_tot
        s_tot = s * s_tot
        t_tot = s * (R @ t_tot) + t
        r, frac = residual(moved)
        if r < best[0]:
            best = (r, R_tot.copy(), s_tot, t_tot.copy(), frac)

    sim_resid, R, s, t, frac = best
    if not np.isfinite(sim_resid) or sim_resid > SIMILARITY_MARGIN * trans_resid:
        out.update(
            mode="translation_only",
            rotation_deg=0.0,
            tx=float(shift[0]),
            ty=float(shift[1]),
            scale=1.0,
            resid=float(trans_resid),
            matched_fraction=float(trans_frac),
            note=(
                "a similarity fit did not beat translation alone by the required margin, "
                "so no rotation or scale is claimed"
            ),
        )
        return out

    out.update(
        mode="similarity",
        rotation_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
        tx=float(t[0]),
        ty=float(t[1]),
        scale=float(s),
        resid=float(sim_resid),
        matched_fraction=float(frac),
    )
    return out


def local_shift_dispersion(a: Any, b: Any, nblocks: int = LOCAL_BLOCKS) -> float:
    """How much a single global similarity fails to explain: the class B / class C discriminator.

    After the best similarity has been fitted and removed, the tissue is cut into a fixed grid and
    each block is asked what translation it *still* wants, as the difference of the two trimmed
    block centroids. A rigid pair has already been explained, so every block wants roughly the
    same nothing. A deformed pair has not, and the blocks disagree -- smoothly, because a
    deformation is smooth. The statistic is the spatial standard deviation of those per-block
    shifts, as a fraction of the tissue diagonal.

    **Two more obvious statistics were implemented, measured on the fixtures, and removed.**
    Neither discriminated, and a metric that does not vary with the thing it names is worse than
    no metric because it will be read as evidence:

    * the ratio of a non-rigid residual to a rigid one -- 0.99 / 0.98 / 0.95 for the aligned,
      rigid and warped stacks. The residual magnitude sits on a floor set by the two sections
      being different cells, which no alignment can lower.
    * Moran's I of the post-fit residual vectors -- -0.013 / 0.016 / 0.014. Nearest-neighbour
      correspondence destroys the field it is trying to measure: every residual points at the
      closest available point, so it is short and locally random by construction.

    The block centroid avoids both, because the block is fixed in space rather than chosen per
    point. Measured on the same three fixtures: **0.0084 / 0.0305 / 0.0658** (the medians in
    ``docs/design/spatial_3d_thresholds.csv``).
    """
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fit = fit_similarity(a, b)
    if fit["mode"] == "not_fitted":
        return float("nan")

    r = np.deg2rad(fit["rotation_deg"])
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    moved = fit["scale"] * (b @ R.T) + np.array([fit["tx"], fit["ty"]])

    L = 0.5 * (bbox_diagonal(a) + bbox_diagonal(moved))
    if not L:
        return float("nan")
    lo = np.minimum(a.min(0), moved.min(0))
    hi = np.maximum(a.max(0), moved.max(0))
    edges = [np.linspace(lo[i], hi[i], nblocks + 1) for i in (0, 1)]

    shifts = []
    for i in range(nblocks):
        for j in range(nblocks):
            ma = (
                (a[:, 0] >= edges[0][i])
                & (a[:, 0] < edges[0][i + 1])
                & (a[:, 1] >= edges[1][j])
                & (a[:, 1] < edges[1][j + 1])
            )
            mb = (
                (moved[:, 0] >= edges[0][i])
                & (moved[:, 0] < edges[0][i + 1])
                & (moved[:, 1] >= edges[1][j])
                & (moved[:, 1] < edges[1][j + 1])
            )
            if ma.sum() >= LOCAL_MIN_PER_BLOCK and mb.sum() >= LOCAL_MIN_PER_BLOCK:
                shifts.append(trimmed_centroid(a[ma]) - trimmed_centroid(moved[mb]))
    if len(shifts) < LOCAL_MIN_BLOCKS:
        return float("nan")
    return float(np.linalg.norm(np.asarray(shifts).std(0)) / L)


def pair_geometry(name_a: str, xy_a: Any, name_b: str, xy_b: Any) -> PairGeometry:
    """Every geometric statement about one adjacent pair."""
    import numpy as np

    a = np.asarray(xy_a, dtype=float)[:, :2]
    b = np.asarray(xy_b, dtype=float)[:, :2]
    # A row with no finite coordinate is not a position. SPIRAL's spatial_aligned and ST-GEARS's
    # spatial_elas_reuse hold NaN rows by design, and the KD-tree below raised 'data must be
    # finite' on the first one -- while the profile told the user those cells were excluded. They
    # are now, here, and counted (hunt 2026-09-30, u21-3d-4).
    dropped = []
    for name, arr in ((name_a, a), (name_b, b)):
        n_bad = int((~np.isfinite(arr).all(axis=1)).sum()) if len(arr) else 0
        if n_bad:
            dropped.append(f"{n_bad} of {len(arr)} cells of {name} have a non-finite coordinate and were left out")
    if dropped:
        a = a[np.isfinite(a).all(axis=1)]
        b = b[np.isfinite(b).all(axis=1)]
    g = PairGeometry(a=name_a, b=name_b, n_a=len(a), n_b=len(b))
    g.notes.extend(dropped)
    if len(a) < 3 or len(b) < 3:
        g.notes.append("a section with fewer than three points supports no geometry")
        return g

    L = 0.5 * (bbox_diagonal(a) + bbox_diagonal(b))
    g.scale_length = L

    off = float(np.linalg.norm(trimmed_centroid(b) - trimmed_centroid(a)))
    g.centroid_offset = off
    g.centroid_offset_frac = off / L if L else 0.0

    va, ra = principal_axis(a)
    vb, rb = principal_axis(b)
    g.eigenvalue_ratio_a, g.eigenvalue_ratio_b = ra, rb
    if ra < ISOTROPY_FLOOR or rb < ISOTROPY_FLOOR:
        g.axis_refusal = (
            f"the principal axis is not identified: eigenvalue ratios {ra:.2f} and {rb:.2f} are "
            f"below {ISOTROPY_FLOOR}, so the leading eigenvector is noise and an angle from it "
            f"would be a measurement of nothing"
        )
    else:
        g.axis_angle_deg = axis_angle_deg(va, vb)

    ha, hb = hull_area(a), hull_area(b)
    g.hull_area_ratio_log2 = float(abs(np.log2(hb / ha))) if ha > 0 and hb > 0 else 0.0
    (la, ha_), (lb, hb_) = robust_bbox(a), robust_bbox(b)
    asp_a = (ha_[0] - la[0]) / max(ha_[1] - la[1], 1e-12)
    asp_b = (hb_[0] - lb[0]) / max(hb_[1] - lb[1], 1e-12)
    g.aspect_change_log2 = float(abs(np.log2(asp_b / asp_a))) if asp_a > 0 and asp_b > 0 else 0.0

    ov = occupancy_overlap(a, b)
    g.iou, g.containment = ov["iou"], ov["containment"]
    g.iou_recentred, g.bin_edge, g.bin_sensitive = ov["iou_recentred"], ov["bin_edge"], ov["bin_sensitive"]
    if g.bin_sensitive:
        g.notes.append("the overlap estimate moved with the bin size and is weak evidence")

    fit = fit_similarity(a, b)
    g.fit_mode = fit["mode"]
    g.fit_rotation_deg, g.fit_tx, g.fit_ty, g.fit_scale = (fit["rotation_deg"], fit["tx"], fit["ty"], fit["scale"])
    g.matched_fraction = fit["matched_fraction"]
    g.resid_rigid = fit["resid"] / L if (L and np.isfinite(fit["resid"])) else float("nan")
    if fit.get("note"):
        g.notes.append(fit["note"])

    g.local_shift_dispersion = local_shift_dispersion(a, b)
    if np.isnan(g.local_shift_dispersion):
        g.notes.append(
            "too few populated blocks to say whether one global transform explains this pair, "
            "so rigid and non-rigid cannot be told apart here"
        )
    return g
