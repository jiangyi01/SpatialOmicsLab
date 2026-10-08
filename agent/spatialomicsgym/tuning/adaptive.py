"""Adaptive tuning for non-benchmark tasks.

Uses dataset characteristics, presets, and heuristic rules to select
hyperparameters when ground truth is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatialomicsgym.utils.file_io import read_h5ad_backed

logger = logging.getLogger(__name__)

#: Substrings ``profile_dataset`` looks for in the keys of ``adata.uns["spatial"]``, mapped to the
#: platform name it then reports. This is the whole platform vocabulary: a name an
#: :class:`AdaptiveRule` or a :class:`~spatialomicsgym.tuning.strategies.Preset` asks for that is not
#: a value here can never match a real dataset, so the rule silently never runs. Keeping the mapping
#: in one introspectable place is what lets a test check that correspondence.
PLATFORM_KEY_HINTS: dict[str, str] = {
    "visium": "visium",
    "merfish": "merfish",
    "xenium": "xenium",
    # Spelled out rather than a bare "slide", which would also claim a library id naming a
    # whole-slide image. A deconvolution rule and the low_density preset both target this platform.
    "slide-seq": "slide_seq",
    "slideseq": "slide_seq",
    "slide_seq": "slide_seq",
}


@dataclass
class DatasetProfile:
    """Characteristics of a dataset used to guide adaptive tuning."""

    n_spots: int = 0
    n_genes: int = 0
    n_celltypes: int | None = None  # If annotations available
    has_spatial: bool = False
    has_images: bool = False
    has_annotations: bool = False
    platform: str = "unknown"  # visium, merfish, slide_seq, xenium, etc.
    # Fraction of zeros in the expression matrix, or None when it could not be measured.
    # Not 0.0: "a perfectly dense matrix" and "the probe failed" are opposite findings, and this
    # field used to default to the former and be published as such whenever the latter happened.
    sparsity: float | None = None
    modality: str = "rna"
    estimated_memory_gb: float = 0.0

    def size_category(self) -> str:
        """Classify dataset size for preset selection."""
        if self.n_spots < 500:
            return "small"
        elif self.n_spots < 5000:
            return "medium"
        elif self.n_spots < 50000:
            return "large"
        return "very_large"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_spots": self.n_spots,
            "n_genes": self.n_genes,
            "n_celltypes": self.n_celltypes,
            "has_spatial": self.has_spatial,
            "has_images": self.has_images,
            "has_annotations": self.has_annotations,
            "platform": self.platform,
            "sparsity": None if self.sparsity is None else round(self.sparsity, 3),
            "size_category": self.size_category(),
            "estimated_memory_gb": round(self.estimated_memory_gb, 2),
        }


def _measure_sparsity(adata: Any) -> float | None:
    """Fraction of zeros in ``adata.X``, or ``None`` if it could not be measured.

    ``adata`` is open ``backed="r"``, so ``X`` is an anndata ``_CSRDataset``/``_CSCDataset`` or an
    h5py ``Dataset``. The previous implementation sampled it with an *unsorted* ``np.random.choice``
    index, which every one of those three rejects -- the backed sparse ones reject any fancy index
    at all -- so the measurement never once succeeded on a backed file.

    For a sparse matrix the answer needs no sampling: the number of stored values is the length of
    the backing ``data`` array, so ``1 - nnz/(n_obs*n_vars)`` is exact and reads no expression data
    whatsoever. (Explicitly-stored zeros would count as non-zero here, biasing the result slightly
    low; a counts matrix written by scanpy/Seurat has none.) Dense storage falls back to a bounded
    *contiguous* row window, which h5py accepts.
    """
    import numpy as np

    try:
        import scipy.sparse as sp

        X = adata.X
        total = int(adata.n_obs) * int(adata.n_vars)
        if X is None or total <= 0:
            return None

        if hasattr(X, "to_memory"):  # backed sparse dataset
            grp = getattr(X, "group", None) or getattr(X, "_group", None)
            if grp is not None and "data" in grp:
                return float(1.0 - grp["data"].shape[0] / total)
            if adata.n_obs > 50000:
                return None  # too large to materialize just to count zeros
            X = X.to_memory()

        if sp.issparse(X):
            return float(1.0 - X.nnz / total)

        sample = np.asarray(X[: min(1000, int(adata.n_obs))])
        return float(np.mean(sample == 0)) if sample.size else None
    except Exception as e:  # pragma: no cover - defensive; must stay visible, not silent
        logger.warning("Could not measure sparsity: %s", e)
        return None


def profile_dataset(dataset_path: str | None) -> DatasetProfile:
    """Profile a dataset to guide adaptive tuning decisions."""
    if not dataset_path:
        return DatasetProfile()

    path = Path(dataset_path)
    if not path.exists():
        logger.warning("Dataset not found: %s", dataset_path)
        return DatasetProfile()

    if path.suffix != ".h5ad":
        logger.warning("Unsupported format: %s", path.suffix)
        return DatasetProfile()

    try:
        # read_h5ad_backed releases the HDF5 handle on every exit from this block. Letting the
        # local go out of scope is not enough: an input with a .raw slot is cyclic, so the file
        # would stay locked until the garbage collector next ran and the next writer -- a worker
        # subprocess rewriting the same path -- would fail with errno 11.
        with read_h5ad_backed(dataset_path) as adata:
            profile = DatasetProfile(
                n_spots=adata.n_obs,
                n_genes=adata.n_vars,
                has_spatial="spatial" in adata.obsm,
            )

            # Check for images
            uns = adata.uns if adata.uns is not None else {}
            profile.has_images = "spatial" in uns or any("image" in k.lower() for k in uns)

            # Check for annotations
            annotation_keywords = ["cell_type", "celltype", "annotation", "label", "cluster"]
            for col in adata.obs.columns:
                if any(kw in col.lower() for kw in annotation_keywords):
                    profile.has_annotations = True
                    try:
                        profile.n_celltypes = int(adata.obs[col].nunique())
                    except Exception:
                        pass
                    break

            profile.sparsity = _measure_sparsity(adata)

            # Estimate memory
            profile.estimated_memory_gb = (adata.n_obs * adata.n_vars * 4) / (1024**3)

            # Detect platform from metadata
            if "spatial" in adata.uns:
                spatial_meta = adata.uns.get("spatial", {})
                if isinstance(spatial_meta, dict):
                    for key in spatial_meta:
                        key_lower = key.lower()
                        for hint, platform in PLATFORM_KEY_HINTS.items():
                            if hint in key_lower:
                                profile.platform = platform
                                break

            return profile

    except Exception as e:
        logger.warning("Failed to profile dataset: %s", e)
        return DatasetProfile()


@dataclass
class AdaptiveRule:
    """A heuristic rule for parameter adaptation."""

    condition: str  # Description of when this rule applies
    param_name: str
    adjustment: Any  # Value or callable
    reason: str


# Heuristic adaptation rules by task type
CLUSTERING_RULES: list[AdaptiveRule] = [
    AdaptiveRule(
        condition="large_dataset",
        param_name="n_neighbors",
        adjustment=20,
        reason="Larger datasets benefit from more neighbors for stable graphs",
    ),
    AdaptiveRule(
        condition="small_dataset",
        param_name="n_neighbors",
        adjustment=8,
        reason="Small datasets need fewer neighbors to avoid over-smoothing",
    ),
    AdaptiveRule(
        condition="many_celltypes",
        param_name="resolution",
        adjustment=1.0,
        reason="Higher resolution to capture more cluster diversity",
    ),
    AdaptiveRule(
        condition="few_celltypes",
        param_name="resolution",
        adjustment=0.3,
        reason="Lower resolution for datasets with few distinct types",
    ),
]

DECONVOLUTION_RULES: list[AdaptiveRule] = [
    AdaptiveRule(
        condition="slide_seq",
        param_name="n_cells_per_location",
        adjustment=5,
        reason="Slide-seqV2 beads capture ~1-5 cells",
    ),
    AdaptiveRule(
        condition="visium",
        param_name="n_cells_per_location",
        adjustment=30,
        reason="Visium spots typically contain 10-50 cells",
    ),
    AdaptiveRule(
        condition="large_dataset",
        param_name="max_epochs_map",
        adjustment=1000,
        reason="Large datasets may converge faster, reduce epochs",
    ),
]

TASK_RULES: dict[str, list[AdaptiveRule]] = {
    "spatial_clustering": CLUSTERING_RULES,
    "deconvolution": DECONVOLUTION_RULES,
}


def profile_conditions(profile: DatasetProfile) -> set[str]:
    """The condition tokens a profiled dataset satisfies.

    One vocabulary with two readers: :func:`apply_adaptive_rules` matches ``AdaptiveRule.condition``
    against it, and the agent's adaptive branch matches
    :attr:`~spatialomicsgym.tuning.strategies.Preset.suitable_for` against it to decide which preset
    a dataset actually fits. They share this definition rather than each building their own, because
    the two sets of tokens were authored independently and only partly overlap -- a divergence is
    invisible until a preset silently stops matching anything.
    """
    conditions: set[str] = {f"{profile.size_category()}_dataset"}

    # Every rule and preset written for a big dataset names large_dataset, and its reasoning only
    # gets stronger above 50,000 spots -- so the very_large band must satisfy it too, or the band
    # boundary acts as an upper bound and the biggest datasets get no adaptation at all. This is
    # the one implication among the four bands: the smaller ones are not implied by the larger.
    if profile.size_category() == "very_large":
        conditions.add("large_dataset")

    if profile.platform != "unknown":
        conditions.add(profile.platform)

    if profile.n_celltypes is not None:
        if profile.n_celltypes > 15:
            conditions.add("many_celltypes")
        elif profile.n_celltypes < 5:
            conditions.add("few_celltypes")

    # Panel breadth, on the same two numbers transcriptomics_skills uses to tell a targeted panel
    # from a whole-transcriptome assay. The 2,000-4,999 band that heuristic deliberately leaves
    # unclassified stays unclassified here too, and n_genes == 0 -- what every profile_dataset
    # failure path returns -- is an unread file, not a small panel.
    if profile.n_genes > 0:
        if profile.n_genes < 2000:
            conditions.add("few_genes")
        elif profile.n_genes >= 5000:
            conditions.add("many_genes")

    # An unmeasured sparsity is not evidence of a dense matrix, so it asserts nothing either way.
    if profile.sparsity is not None and profile.sparsity > 0.9:
        conditions.add("very_sparse")

    return conditions


def apply_adaptive_rules(
    task_type: str,
    baseline_params: dict[str, Any],
    profile: DatasetProfile,
) -> list[dict[str, Any]]:
    """Apply heuristic rules to generate adaptive parameter candidates.

    Returns a list of candidate configs (including the unchanged baseline).
    """
    rules = TASK_RULES.get(task_type, [])
    candidates: list[dict[str, Any]] = [baseline_params.copy()]
    conditions = profile_conditions(profile)

    # Apply matching rules
    adapted = baseline_params.copy()
    applied_rules: list[str] = []

    for rule in rules:
        if rule.condition in conditions and rule.param_name in adapted:
            adapted[rule.param_name] = rule.adjustment
            applied_rules.append(f"{rule.param_name}={rule.adjustment} ({rule.reason})")

    if adapted != baseline_params:
        candidates.append(adapted)
        logger.info("Applied adaptive rules: %s", "; ".join(applied_rules))

    return candidates


def should_downgrade_to_fallback(profile: DatasetProfile, task_type: str) -> tuple[bool, str]:
    """Check if adaptive tuning should downgrade to default_fallback.

    Returns (should_downgrade, reason).
    """
    # Zero spots is what every failure path in profile_dataset returns, so it has to be answered
    # before the size test -- otherwise a file that could not be opened at all is reported to the
    # user as a dataset too small to tune, and the read failure is left in a log line.
    if profile.n_spots == 0:
        return True, "Could not profile dataset"

    if profile.n_spots < 50:
        return True, f"Dataset too small ({profile.n_spots} spots) for meaningful tuning"

    if profile.estimated_memory_gb > 50:
        return True, f"Dataset too large ({profile.estimated_memory_gb:.1f}GB estimated) for tuning runs"

    return False, ""
