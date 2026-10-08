"""Measure the classifier's thresholds instead of inventing them. Not imported by anything.

Run this to regenerate ``docs/design/spatial_3d_thresholds.csv``, which is the file every entry in
:mod:`~spatialomicsgym.spatial3d.thresholds` cites. A threshold that cannot name a row in that
table fails a test, and that is the whole mechanism: it is the reason no number in this package is
somebody's guess.

Two sources, and the difference between them is itself a finding.

**Zhuang-ABCA-1 is a real, known class-A stack** -- 147 MERFISH coronal sections of one mouse
brain, placed in a shared frame by the Allen Institute before this repository ever saw them. Its
146 adjacent pairs are what "already aligned" actually looks like, and they look nothing like a
synthetic stack. Adjacent sections 100 micrometres apart have genuinely *different outlines*,
because the brain changes shape along the anterior-posterior axis: the measured median IoU is
0.37 and the maximum over all pairs is 0.465. A threshold of 0.70 -- the number that looks right
when you picture two slices of the same tissue -- classifies the entire atlas as misaligned.

**The four sections at exactly z = 0 are excluded**, and that exclusion is not a convenience. The
provider clamps negative anterior estimates to zero, so sections .001 to .004 occupy one plane;
"adjacent" is undefined among them and any ordering is invented. Their pairs measure a centroid
offset of 0.20 and a local shift of 0.08 -- i.e. they look badly misaligned -- purely because
they were sorted. Calibrating on them would move every class-A threshold far enough to admit real
misalignment.

**The synthetic fixtures supply the class-B and class-C side**, because no misaligned real stack
in this repository has a known transform. They are deliberately not used for the class-A numbers:
a synthetic stack is more self-similar than real tissue, and thresholds set from it would be far
too tight for anything measured.
"""

from __future__ import annotations

import argparse
import os
import sys

ZHUANG_METADATA = "demo_datasets_for_3D_analysis/cell_metadata.csv"
FIXTURES = ("aligned", "rigid", "warped")
OUT_CSV = "docs/design/spatial_3d_thresholds.csv"

#: Fields carried through to the table. Each is unit-free, so a threshold written against one is
#: comparable between a millimetre atlas and a pixel slide.
FIELDS = (
    "centroid_offset_frac",
    "iou",
    "iou_recentred",
    "containment",
    "resid_rigid",
    "local_shift_dispersion",
    "aspect_change_log2",
    "hull_area_ratio_log2",
    "axis_angle_deg",
)


def zhuang_pairs(root: str, limit: int = 0):
    """Adjacent-pair geometry over Zhuang-ABCA-1's reconstructed coordinates."""
    import pandas as pd

    from . import geometry as geom

    path = os.path.join(root, ZHUANG_METADATA)
    df = pd.read_csv(
        path,
        usecols=["brain_section_label", "x", "y", "z"],
        dtype={"brain_section_label": "category", "x": "float32", "y": "float32", "z": "float32"},
    )
    med = df.groupby("brain_section_label", observed=True)["z"].median().sort_values()
    dropped = [s for s, z in med.items() if z <= 0.0]
    keep = med[med.values > 0.0]
    order = keep.index.tolist()
    if limit:
        order = order[:limit]

    rows = []
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        A = df.loc[df.brain_section_label == a, ["x", "y"]].to_numpy(dtype=float)
        B = df.loc[df.brain_section_label == b, ["x", "y"]].to_numpy(dtype=float)
        r = geom.pair_geometry(str(a), A, str(b), B).as_row()
        r["source"] = "zhuang_abca1"
        r["known_class"] = "A"
        rows.append(r)
    return rows, dropped


def fixture_pairs(root: str):
    """Adjacent-pair geometry over the three serial-section fixtures."""
    import anndata as ad
    import numpy as np

    from . import geometry as geom

    rows = []
    for kind in FIXTURES:
        path = os.path.join(root, "test", "test_data", f"mini_serial_{kind}.h5ad")
        if not os.path.exists(path):
            print(f"  (skipping {kind}: {path} absent -- run test/smoke/make_aux_data.py)", file=sys.stderr)
            continue
        a = ad.read_h5ad(path)
        xy = np.asarray(a.obsm["spatial"], dtype=float)
        sec = a.obs["slice_id"].astype(str).to_numpy()
        order = sorted(set(sec))
        for i in range(len(order) - 1):
            r = geom.pair_geometry(order[i], xy[sec == order[i]], order[i + 1], xy[sec == order[i + 1]]).as_row()
            r["source"] = f"fixture_{kind}"
            r["known_class"] = str(a.uns["ground_truth_class"])
            rows.append(r)
    return rows


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--limit-zhuang", type=int, default=0, help="first N sections only, for a quick pass")
    ap.add_argument("--skip-zhuang", action="store_true", help="fixtures only (Zhuang is 2.9 GB and gitignored)")
    args = ap.parse_args(argv)

    rows = fixture_pairs(args.root)
    dropped: list = []
    if not args.skip_zhuang:
        zpath = os.path.join(args.root, ZHUANG_METADATA)
        if os.path.exists(zpath):
            z, dropped = zhuang_pairs(args.root, args.limit_zhuang)
            rows += z
        else:
            print(f"  (skipping Zhuang: {zpath} absent)", file=sys.stderr)

    if not rows:
        print("no pairs measured; nothing to write", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    out = os.path.join(args.root, OUT_CSV)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    summary = []
    for (source, known), grp in df.groupby(["source", "known_class"]):
        for f in FIELDS:
            v = pd.to_numeric(grp[f], errors="coerce").dropna()
            if not len(v):
                continue
            summary.append(
                {
                    "source": source,
                    "known_class": known,
                    "metric": f,
                    "n_pairs": len(v),
                    "n_refused": int(len(grp) - len(v)),
                    "min": round(float(v.min()), 5),
                    "p05": round(float(v.quantile(0.05)), 5),
                    "median": round(float(v.median()), 5),
                    "p95": round(float(v.quantile(0.95)), 5),
                    "max": round(float(v.max()), 5),
                }
            )
    s = pd.DataFrame(summary).sort_values(["metric", "source"])
    s.to_csv(out, index=False)
    print(f"wrote {out} ({len(s)} rows from {len(df)} pairs)")
    if dropped:
        print(f"excluded {len(dropped)} Zhuang sections at z <= 0 (coincident planes): {list(dropped)[:6]}")
    df.to_csv(out.replace(".csv", "_pairs.csv"), index=False)
    print(f"wrote {out.replace('.csv', '_pairs.csv')} (every pair, for auditing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
