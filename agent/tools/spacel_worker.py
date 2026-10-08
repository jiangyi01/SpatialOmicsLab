#!/usr/bin/env python
"""
spacel_worker.py

Worker script for SPACEL (Splane / Scube) spatial analysis.

- Executed inside the SPACEL conda env: /opt/conda/envs/spacel_env
- All logs and progress go to stderr.
- Stdout contains exactly one line of JSON at the end.

Tasks:
  splane  - spatial domain identification on a single slice
  scube   - 3D alignment across multiple slices

Both tasks leave out background spots (``obs['in_tissue'] == 0``; CELLxGENE Visium exports carry
every array spot) and report how many. A spot whose label is missing stops the run unless
``--drop-unlabeled`` is set, in which case it is left out and counted. Every output is written to
``<name>.partial`` and renamed into place, so a killed run never leaves a truncated file under a
published name.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from worker_utils import (
    WorkerOutput,
    build_cluster_analysis,
    describe_reduction,
    identifier_rename_note,
    identifier_rename_params,
    keep_in_tissue,
    make_names_unique_and_report,
    preflight_check,
    record_ignored,
    record_in_tissue,
    record_method,
)
from worker_utils import drop_unlabeled as _split_unlabeled  # the parameter of the same name shadows it


def log(msg: str) -> None:
    sys.stderr.write(f"[spacel-worker] {msg}\n")
    sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SPACEL worker (Splane / Scube).")
    parser.add_argument("--task", type=str, required=True, choices=["splane", "scube"])
    # Splane args
    parser.add_argument("--spatial-h5ad", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument(
        "--celltype-key",
        type=str,
        default="",
        help="obs column of per-spot cell-type labels to one-hot into Splane's proportion input. Empty: the "
        "AnnData must already carry uns['celltypes'] and one float proportion column per type.",
    )
    # Scube args
    parser.add_argument("--spatial-h5ad-paths", type=str, default="")
    parser.add_argument(
        "--cluster-key",
        type=str,
        default="",
        help="obs column present in every slice that Scube aligns on. Empty: the slices must share one of "
        "the conventional label columns, or --allow-joint-leiden must be set.",
    )
    parser.add_argument(
        "--allow-joint-leiden",
        action="store_true",
        help="When no shared label column exists, cluster all slices jointly (one Leiden over the concatenated "
        "slices) and align on those labels. Off by default: the labels then drive the alignment.",
    )
    # Both tasks
    parser.add_argument(
        "--drop-unlabeled",
        action="store_true",
        help="Leave out spots with no label (splane: an empty obs[--celltype-key] or a non-finite proportion in "
        "the uns['celltypes'] columns; scube: an empty label in the column the slices are aligned on) and "
        "report the count. Off by default: a missing label stops the run.",
    )
    return parser.parse_args()


def _write_atomically(final_path: str, write) -> None:
    """``write(<final>.partial)``, then ``os.replace`` onto the final name.

    A reader never sees a half-written file under the final name, and a write that fails takes its
    ``.partial`` with it instead of leaving it beside the outputs.
    """
    partial = final_path + ".partial"
    try:
        write(partial)
        os.replace(partial, final_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(partial)
        raise


ONEHOT_PREFIX = "ctprop__"


def _prepare_celltypes_for_splane(adata, celltype_key: str = "", drop_unlabeled: bool = False):
    """
    Make sure adata carries what Splane reads: ``uns['celltypes']`` plus one float proportion column
    per cell type in obs (``SPACEL/Splane/utils.py::generate_celltype_ad_list``).

    Two explicit sources, nothing guessed:

    * ``celltype_key`` empty -> ``uns['celltypes']`` must already exist (e.g. written by a
      deconvolution step). Every named column must be present in obs.
    * ``celltype_key`` given -> that obs column is one-hot encoded into ``ctprop__<label>`` columns
      (the prefix keeps them from overwriting an existing obs column) and ``uns['celltypes']`` is set
      to those names. A one-hot of a categorical label is a coarse stand-in for real proportions, and
      the caller is told so.

    Returns ``(adata, source)`` where ``source`` is ``"uns_celltypes"`` or ``"onehot:<key>"``.

    A spot with no label is not a cell type, and Splane cannot be told so: a NaN label used to become
    an all-zero proportion row that still entered the GCN and got a domain, and a NaN proportion gives
    its cell type a Moran's I of 0, which ``init_model`` reads as weight 0 and drops the whole type
    from the model. Either one is refused, naming the count, unless ``drop_unlabeled`` is True; then
    those spots are left out and the returned ``adata`` is smaller by that many (the caller reports it).

    The old behaviour -- try a list of label names (including ground-truth and cluster columns), then
    fall back to any categorical column with 2..50 values -- silently consumed whatever it found, so
    a labelled sample fed its own reference labels to the method and an unlabelled one was refused
    with a message that did not say what Splane needs.
    """
    import numpy as np

    if celltype_key:
        if celltype_key not in adata.obs.columns:
            raise ValueError(
                f"celltype_key={celltype_key!r} is not an obs column. Available: {list(adata.obs.columns)}"
            )
        keep, n_unlabeled = _split_unlabeled(
            adata.obs[celltype_key].values, drop_unlabeled, what=f"spots (obs[{celltype_key!r}])"
        )
        if n_unlabeled:
            log(f"Leaving out {n_unlabeled} spots with no label in obs[{celltype_key!r}] (drop_unlabeled=True)")
            adata = adata[keep].copy()
        labels = adata.obs[celltype_key].astype(str)
        categories = sorted(labels.unique().tolist())
        if len(categories) < 2:
            raise ValueError(
                f"obs[{celltype_key!r}] has {len(categories)} usable label(s); Splane needs at least two cell types."
            )
        log(f"One-hot encoding obs[{celltype_key!r}] into {len(categories)} {ONEHOT_PREFIX}* proportion columns")
        log("  WARNING: a one-hot of categorical labels stands in for per-spot cell-type proportions.")
        names = []
        for ct in categories:
            col = f"{ONEHOT_PREFIX}{ct}"
            adata.obs[col] = (labels == ct).astype(np.float64)
            names.append(col)
        adata.uns["celltypes"] = names
        return adata, f"onehot:{celltype_key}"

    if "celltypes" in adata.uns:
        names = [str(c) for c in adata.uns["celltypes"]]
        missing = [c for c in names if c not in adata.obs.columns]
        if missing:
            raise ValueError(
                f"uns['celltypes'] names {len(missing)} column(s) absent from obs: {missing[:5]}. "
                "Splane reads one float proportion column per cell type."
            )
        try:
            props = adata.obs[names].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"the uns['celltypes'] columns must hold numeric proportions; reading them as numbers failed: {exc}"
            ) from exc
        finite = np.isfinite(props).all(axis=1)
        n_unlabeled = int((~finite).sum())
        if n_unlabeled:
            if not drop_unlabeled:
                raise ValueError(
                    f"{n_unlabeled} of {adata.n_obs} spots have a missing (NaN/inf) proportion in the "
                    "uns['celltypes'] columns. Splane would give that cell type a Moran's I of 0 and leave it out "
                    "of the model. Pass drop_unlabeled=True to leave those spots out, or fill the proportions first."
                )
            log(f"Leaving out {n_unlabeled} spots with a missing proportion (drop_unlabeled=True)")
            adata = adata[finite].copy()
        log(f"Using existing uns['celltypes'] ({len(names)} cell types) as Splane's proportion input")
        return adata, "uns_celltypes"

    raise ValueError(
        "SPACEL Splane needs per-spot cell-type proportions: either uns['celltypes'] naming one float obs "
        "column per cell type (e.g. from a deconvolution step), or celltype_key naming an obs label column to "
        f"one-hot encode. Neither was found; obs columns: {list(adata.obs.columns)}"
    )


def run_splane(
    spatial_h5ad: str,
    output_dir: str,
    n_clusters: int,
    resolution: float,
    celltype_key: str = "",
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """Run SPACEL Splane for spatial domain identification."""
    import matplotlib
    import scanpy as sc

    matplotlib.use("Agg")

    preflight_check(
        inputs={"spatial_h5ad": spatial_h5ad},
        output_dir=output_dir,
        packages=["SPACEL"],
    )

    from SPACEL import Splane

    log(f"Loading spatial AnnData from {spatial_h5ad}")
    adata = sc.read_h5ad(spatial_h5ad)
    renamed = make_names_unique_and_report(adata)
    n_spots = adata.n_obs
    n_genes = adata.n_vars
    log(f"Loaded AnnData: n_spots={n_spots}, n_genes={n_genes}")

    # Background glass is not tissue. CELLxGENE Visium exports carry every array spot with
    # obs['in_tissue'] 0/1 (56-70% background on the library's four such samples); left in, those
    # spots enter the GCN graph and use up domains of their own.
    adata, _, n_off_tissue = keep_in_tissue(adata, "spots")
    if n_off_tissue:
        log(f"Left out {n_off_tissue} of {n_spots} spots with obs['in_tissue'] == 0 (background)")
    n_in_tissue = adata.n_obs

    # Cell-type proportions for Splane: an explicit source, never a guessed column. A spot with no
    # label is refused here unless drop_unlabeled, in which case it is left out and counted.
    adata, celltype_source = _prepare_celltypes_for_splane(adata, celltype_key, drop_unlabeled=drop_unlabeled)
    n_unlabeled_dropped = n_in_tissue - adata.n_obs
    n_spots_used = adata.n_obs

    # Fix sklearn KMeans compatibility: SPACEL uses algorithm='full' which
    # was renamed to 'lloyd' in scikit-learn >= 1.3.
    # Patch the SPACEL Splane utils.clustering function to fix the KMeans call.
    try:
        import SPACEL.Splane.base_model as _splane_base
        import SPACEL.Splane.utils as _splane_utils

        _orig_clustering = _splane_utils.clustering

        def _patched_clustering(cluster_obj, feature):
            # Replace algorithm='full' with 'lloyd' if needed
            if hasattr(cluster_obj, "algorithm") and cluster_obj.algorithm == "full":
                cluster_obj.algorithm = "lloyd"
            return _orig_clustering(cluster_obj, feature)

        _splane_utils.clustering = _patched_clustering
        _splane_base.clustering = _patched_clustering
        log("Patched SPACEL clustering() for sklearn KMeans compat (full -> lloyd)")
    except Exception as e:
        log(f"WARNING: Could not patch SPACEL clustering: {e}")

    log(f"Running Splane with n_clusters={n_clusters}")
    if resolution != 1.0:
        log(f"  note: resolution={resolution} has no effect; Splane clusters with KMeans(n_clusters)")
    splane_model = Splane.init_model(
        [adata],
        n_clusters=n_clusters,
    )
    # Checkpoints go to a directory this run owns. Without save_path, train() uses
    # <tmp>/Splane_models_<YYYYmmddHHMMSS> with fixed file names and deletes the previous best
    # checkpoint as it improves, so two runs started in the same second share a directory: one can
    # delete the checkpoint the other is about to torch.load, or overwrite it with its own weights.
    # The directory is removed once the domains are read; nothing in it is an output.
    checkpoint_dir = tempfile.mkdtemp(prefix="splane_checkpoints_", dir=output_dir)
    cluster_col = "splane_cluster"
    try:
        splane_model.train(save_path=checkpoint_dir)
        # train() only fits the GCN. The domain labels come from identify_spatial_domain(), which
        # clusters the latent features and writes them to obs[key]. Without this call the worker used
        # to publish obs.columns[-1] -- the one-hot indicator of the last input cell type.
        splane_model.identify_spatial_domain(key=cluster_col)
    finally:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    if cluster_col not in adata.obs.columns:
        raise RuntimeError(f"Splane.identify_spatial_domain did not write obs[{cluster_col!r}]")
    log(f"Splane wrote domain labels to obs[{cluster_col!r}]")

    clusters = adata.obs[cluster_col]
    cluster_sizes = clusters.value_counts().to_dict()
    cluster_sizes = {str(k): int(v) for k, v in cluster_sizes.items()}

    # Save outputs
    annotated_path = os.path.join(output_dir, "spacel_splane_annotated.h5ad")
    _write_atomically(annotated_path, adata.write_h5ad)
    log(f"Saved annotated AnnData to {annotated_path}")

    clusters_csv = os.path.join(output_dir, "spacel_splane_clusters.csv")
    _write_atomically(clusters_csv, adata.obs[[cluster_col]].to_csv)
    log(f"Saved clusters CSV to {clusters_csv}")

    out = WorkerOutput("spacel", task="splane")
    out.set_data(
        n_spots=int(n_spots),
        n_genes=int(n_genes),
        n_spots_used=int(n_spots_used),
        n_spots_unlabeled_dropped=int(n_unlabeled_dropped),
    )
    out.add_output_files(
        {
            "annotated_h5ad": annotated_path,
            "clusters_csv": clusters_csv,
        }
    )
    out.add_params(
        {
            "n_clusters": n_clusters,
            "resolution": resolution,
            "cluster_column": cluster_col,
            "celltype_key": celltype_key,
            "celltype_source": celltype_source,
            "drop_unlabeled": bool(drop_unlabeled),
        }
    )
    record_method(out, "SPACEL Splane (GCN latent features + KMeans via identify_spatial_domain)")
    record_ignored(
        out, "resolution", "Splane clusters its latent features with KMeans(n_clusters); it has no resolution knob."
    )
    record_in_tissue(out, n_spots, n_off_tissue)
    out.add_params(identifier_rename_params(renamed))
    if celltype_source.startswith("onehot:"):
        out.add_warning(f"Cell-type proportions were a one-hot of obs[{celltype_key!r}], not deconvolution estimates.")
    unlabeled_what = f"no label in obs[{celltype_key!r}]" if celltype_key else "a missing uns['celltypes'] proportion"
    if n_unlabeled_dropped:
        out.add_warning(
            f"{n_unlabeled_dropped} in-tissue spots with {unlabeled_what} were left out (drop_unlabeled=True); "
            "they have no domain."
        )
    cuts = []
    if n_off_tissue:
        cuts.append("the in_tissue filter (obs['in_tissue'] == 0 background)")
    if n_unlabeled_dropped:
        cuts.append(f"drop_unlabeled=True ({unlabeled_what})")
    out.set_summary(
        n_clusters=len(cluster_sizes),
        cluster_sizes=cluster_sizes,
    )
    out.set_analysis(
        build_cluster_analysis(
            cluster_sizes,
            cluster_key="spatial domain",
            total_spots=int(n_spots_used),
            n_requested=int(n_clusters),
        )
        + describe_reduction("spots", int(n_spots), int(n_spots_used), " and ".join(cuts))
        + identifier_rename_note(renamed)
    )
    return out.to_dict()


def run_scube(
    spatial_h5ad_paths: str,
    output_dir: str,
    cluster_key: str = "",
    allow_joint_leiden: bool = False,
    drop_unlabeled: bool = False,
) -> dict[str, Any]:
    """Run SPACEL Scube for 3D alignment of multiple slices."""
    import matplotlib
    import scanpy as sc

    matplotlib.use("Agg")

    paths = [p.strip() for p in spatial_h5ad_paths.split(",") if p.strip()]
    if len(paths) < 2:
        raise ValueError(
            f"Scube aligns two or more slices; got {len(paths)} path(s) in spatial_h5ad_paths. "
            "Pass a comma-separated list."
        )
    inputs = {f"slice_{i}": p for i, p in enumerate(paths)}
    preflight_check(inputs=inputs, output_dir=output_dir, packages=["SPACEL"])

    from SPACEL import Scube

    log(f"Loading {len(paths)} spatial slices for 3D alignment")
    adatas = []
    total_spots = 0
    total_off_tissue = 0
    # Summed by hand rather than with into=, so each slice's own uns records only its own rename
    # while the payload still reports the total across every slice we read.
    renamed_all = {"n_genes_renamed": 0, "n_cells_renamed": 0}
    for i, p in enumerate(paths):
        adata = sc.read_h5ad(p)
        renamed_slice = make_names_unique_and_report(adata)
        for _key, _count in renamed_slice.items():
            renamed_all[_key] = renamed_all.get(_key, 0) + int(_count)
        log(f"  Slice {i}: n_spots={adata.n_obs}, n_genes={adata.n_vars} from {p}")
        total_spots += adata.n_obs
        # Background glass (obs['in_tissue'] == 0) is not tissue to align; it would enter the
        # neighbour-label score and, on the joint-Leiden path, the clustering.
        adata, _, n_off = keep_in_tissue(adata, f"spots of slice {i}")
        if n_off:
            log(f"  Slice {i}: left out {n_off} spots with obs['in_tissue'] == 0 (background)")
        total_off_tissue += n_off
        adatas.append(adata)

    # Scube.align needs one label column shared by every slice, and its objective rewards spatial
    # neighbours that carry the SAME label id across slices. So the ids must be comparable: a
    # user-named column, one of the conventional ones present everywhere, or -- only when the
    # caller allows it -- a single Leiden over the concatenated slices. Leiden run per slice gives
    # ids that mean nothing across slices and used to drive the alignment silently.
    conventional = ["spatial_domain", "annotation", "Annotation", "cell_type", "celltype", "CellType", "cluster"]
    cluster_source = None
    if cluster_key:
        missing = [p for a, p in zip(adatas, paths) if cluster_key not in a.obs.columns]
        if missing:
            raise ValueError(f"cluster_key={cluster_key!r} is missing from {len(missing)} slice(s): {missing}")
        cluster_source = "user"
    else:
        for cand in conventional:
            if all(cand in a.obs.columns for a in adatas):
                cluster_key = cand
                cluster_source = "shared_column"
                break
    if cluster_source is None:
        if not allow_joint_leiden:
            raise ValueError(
                "No obs column is shared by every slice, so Scube has no labels to align on. Pass cluster_key "
                "naming a column present in all slices, or set allow_joint_leiden=True to cluster the slices "
                "jointly first (the alignment then depends on those clusters)."
            )
        import anndata as ad_mod

        log("No shared label column; clustering all slices jointly (Leiden on the concatenated slices)")
        joint = ad_mod.concat(
            [a.copy() for a in adatas], label="scube_slice", keys=[str(i) for i in range(len(adatas))], join="inner"
        )
        sc.pp.normalize_total(joint, target_sum=1e4)
        sc.pp.log1p(joint)
        sc.pp.pca(joint, n_comps=min(30, joint.n_obs - 1, joint.n_vars - 1))
        sc.pp.neighbors(joint)
        sc.tl.leiden(joint, resolution=0.5)
        cluster_key = "scube_joint_leiden"
        for i, a in enumerate(adatas):
            labels = joint.obs.loc[joint.obs["scube_slice"] == str(i), "leiden"].to_numpy()
            a.obs[cluster_key] = labels  # labels only; X of the written slices stays the input X
        cluster_source = "joint_leiden"
        del joint

    # A missing label is not a label. Upstream feeds the column to np.unique, which on a NaN beside
    # string labels raises a bare "'<' not supported between instances of 'float' and 'str'"; the
    # CELLxGENE samples' conventional 'annotation' column carries NaN on in-tissue spots too.
    n_unlabeled_dropped = 0
    if cluster_source != "joint_leiden":
        for i, a in enumerate(adatas):
            keep, n_missing = _split_unlabeled(
                a.obs[cluster_key].values, drop_unlabeled, what=f"spots of slice {i} in obs[{cluster_key!r}]"
            )
            if n_missing:
                log(
                    f"  Slice {i}: left out {n_missing} spots with no label in obs[{cluster_key!r}] (drop_unlabeled=True)"
                )
                adatas[i] = a[keep].copy()
                n_unlabeled_dropped += n_missing
        emptied = [p for a, p in zip(adatas, paths) if a.n_obs == 0]
        if emptied:
            raise ValueError(
                f"No spot of {len(emptied)} slice(s) carries a label in obs[{cluster_key!r}]: {emptied}. "
                "Scube has nothing to align there; pass cluster_key naming a labelled column."
            )
    total_spots_used = sum(int(a.n_obs) for a in adatas)

    log(f"Running Scube.align with cluster_key={cluster_key!r}")
    Scube.align(adatas, cluster_key=cluster_key, output_path=output_dir)

    # Save aligned slices (Scube writes aligned coords to obsm['spatial_aligned']).
    aligned_paths = []
    for i, adata in enumerate(adatas):
        out_path = os.path.join(output_dir, f"spacel_scube_aligned_slice_{i}.h5ad")
        _write_atomically(out_path, adata.write_h5ad)
        aligned_paths.append(out_path)
        log(f"  Saved aligned slice {i} to {out_path}")

    out = WorkerOutput("spacel", task="scube")
    out.set_data(
        n_slices=len(adatas),
        total_spots=int(total_spots),
        total_spots_used=int(total_spots_used),
        n_spots_unlabeled_dropped=int(n_unlabeled_dropped),
    )
    out.add_output_files(
        {
            "aligned_h5ads": aligned_paths,
            # Named so the provenance sidecar has a directory to sit in (every other key is a list).
            "output_dir": output_dir,
        }
    )
    warp_info = os.path.join(output_dir, "warp_info.npy")  # upstream Scube's per-pair rigid transforms
    if os.path.isfile(warp_info):
        out.add_output_file("warp_info_npy", warp_info)
    out.add_params(
        {
            "n_slices": len(paths),
            "input_paths": paths,
            "cluster_key": cluster_key,
            "cluster_source": cluster_source,
            "allow_joint_leiden": bool(allow_joint_leiden),
            "drop_unlabeled": bool(drop_unlabeled),
        }
    )
    if cluster_source == "joint_leiden":
        # The labels are a substitute the caller allowed (allow_joint_leiden), not ones the slices carried.
        record_method(
            out,
            "SPACEL Scube rigid pairwise alignment on joint Leiden labels",
            used_fallback=True,
            why=(
                "the slices had no shared label column, so the alignment used one joint Leiden clustering "
                f"(obs[{cluster_key!r}]) and its quality depends on those clusters"
            ),
        )
    else:
        record_method(out, "SPACEL Scube rigid pairwise alignment on shared labels")
    record_in_tissue(out, total_spots, total_off_tissue)
    out.add_params(identifier_rename_params(renamed_all))
    if n_unlabeled_dropped:
        out.add_warning(
            f"{n_unlabeled_dropped} in-tissue spots with no label in obs[{cluster_key!r}] were left out "
            "(drop_unlabeled=True); the aligned slices do not contain them."
        )
    cuts = []
    if total_off_tissue:
        cuts.append("the in_tissue filter (obs['in_tissue'] == 0 background)")
    if n_unlabeled_dropped:
        cuts.append(f"drop_unlabeled=True (no label in obs[{cluster_key!r}])")
    out.set_summary(n_slices_aligned=len(adatas), total_spots=int(total_spots_used))
    out.set_analysis(
        f"SPACEL Scube aligned {len(adatas)} spatial slices ({total_spots_used} total spots) into a 3D reconstruction "
        f"using obs[{cluster_key!r}] ({cluster_source}) as the label shared across slices."
        + describe_reduction("spots", int(total_spots), int(total_spots_used), " and ".join(cuts))
        + identifier_rename_note(renamed_all, subject="input slices")
    )
    return out.to_dict()


def main() -> None:
    args = parse_args()

    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    error_exc = None
    try:
        try:
            if args.task == "splane":
                result = run_splane(
                    spatial_h5ad=args.spatial_h5ad,
                    output_dir=args.output_dir,
                    n_clusters=args.n_clusters,
                    resolution=args.resolution,
                    celltype_key=args.celltype_key,
                    drop_unlabeled=args.drop_unlabeled,
                )
            elif args.task == "scube":
                result = run_scube(
                    spatial_h5ad_paths=args.spatial_h5ad_paths,
                    output_dir=args.output_dir,
                    cluster_key=args.cluster_key,
                    allow_joint_leiden=args.allow_joint_leiden,
                    drop_unlabeled=args.drop_unlabeled,
                )
            else:
                raise ValueError(f"Unknown task: {args.task}")
        except Exception as e:
            log(f"ERROR: {e}")
            traceback.print_exc(file=sys.stderr)
            result = None
            error_msg = str(e)
            error_exc = e
    finally:
        sys.stdout = orig_stdout

    if result is None:
        WorkerOutput.emit_error("spacel", error_msg, task=args.task, exc=error_exc)
        sys.exit(1)
    else:
        print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
