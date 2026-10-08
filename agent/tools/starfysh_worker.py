#!/usr/bin/env python3
"""
Starfysh worker for SpatialOmicsLab MCP (runs in /opt/conda/envs/Starfysh).

SpatialOmicsLab protocol:
- stdout: JSON only (always; success or failure)
- stderr: logs/progress/tracebacks

What runs, and what the payload says about it:
- ``params.method`` names the model that ran: Starfysh PoE (AVAE_PoE, histology-integrated) or
  standard Starfysh (AVAE). ``params.used_fallback`` is True only when PoE was requested, no H&E
  image was available, and the caller passed ``allow_poe_fallback=True``; without that switch a
  PoE request with no image is an error before any data is loaded.
- ``params.signature_source`` is ``"provided"`` (signature_csv) or ``"archetypal_analysis"``.
  There is no substitute signature set: when Archetypal Analysis fails the run fails and says so.
  (An earlier revision sliced the HVG list into ``factor_i`` chunks and labelled them signatures.)
- ``params.ignored`` lists knobs that were accepted but had no effect (``patience`` always: the
  installed ``utils.run_starfysh`` has no early stopping), and every input path the chosen
  ``input_mode`` never reads (``hires_image`` / ``scalefactors_json`` / ``counts_h5ad`` /
  ``coords_csv`` are read only in generic_counts_coords mode; ``count_h5`` / ``spatial_dir`` only in
  visium_h5_spatial; ``outs_dir`` only in spaceranger_outs). The PoE refusal says how to supply the
  image in the mode that was used.
- Spots: in generic_counts_coords mode, background spots (``in_tissue == 0`` in the counts' ``obs``,
  else in ``coords_csv``) are left out before Starfysh runs (``params.in_tissue_filter`` + a
  warning). Starfysh's own ``utils.preprocess`` then drops every spot whose ``pct_counts_mt`` is not
  below 100 -- zero-count spots (NaN) and all-mitochondrial ones; ``data.n_spots_input`` and
  ``data.n_spots_dropped_by_starfysh_preprocess`` say how many, beside ``data.n_spots`` (deconvolved).
- ``cell_type_proportions.csv`` columns are the provided signature names exactly as given (numeric
  headers included) or Archetypal Analysis's ``arch_<idx>`` names. ``arch_<i>`` is substituted only
  when ``uns['cell_types']`` cannot name the columns, with a warning and
  ``params.proportion_columns_renamed``.
- The worker deletes nothing outside its own ``<output_dir>/_starfysh_input/<sample_id>`` staging
  directory, which it rebuilds from scratch on every run that stages input (so a file staged by an
  earlier run -- an h5, an h5ad, an H&E image, a scalefactors file -- can never be read in place of
  this run's input). A Space Ranger ``outs/`` directory is read in place and never modified. (An
  earlier revision removed every ``*.h5ad`` but the newest from the *user's* outs/.)
- ``training_losses.csv`` has one row per epoch of the best restart: ``loss`` (total) plus its
  components. (An earlier revision wrote the whole loss dict into a single cell.)
- Counts: Starfysh's ``utils.preprocess`` runs normalize_total + log1p on X and fits a count model to
  it, so X must hold counts. In generic_counts_coords mode the counts h5ad goes through
  ``worker_utils.choose_counts_matrix`` after the background filter: negative or non-finite values
  (a z-scored X, as on the library's Skin FaceTemple) are refused, naming ``use_raw_counts`` when
  ``adata.raw`` holds counts; a non-integer X (log-normalised, as on Muscle Gastrocnemius) runs as
  before with a warning; ``use_raw_counts=True`` stages ``adata.raw.X``. ``params.expression_source``
  / ``params.x_matrix_kind`` say which matrix ran. The Visium modes read a 10x count matrix, which
  has no ``adata.raw`` (``use_raw_counts`` is listed in ``params.ignored`` there).
- A Visium ``spatial/`` folder without ``tissue_hires_image.png`` and ``tissue_lowres_image.png``
  (the library's VisiumHD Colon samples) made ``scanpy.read_visium`` -- which ``utils.load_adata``
  calls for a staged h5 -- raise "Could not find 'hires_image'" whatever ``use_poe`` said. The counts
  are then read with ``load_images=False`` (the images are never the model's input; PoE reads the
  H&E image through ``preprocess_img``), and ``params.visium_images_missing`` names what was absent.

Compatibility fixes included:
1) NumPy 2.x: provide np.mat alias (py_pcha uses np.mat).
2) SciPy sparse: provide .A property (Starfysh expects adata.X.A); map to toarray().
3) Starfysh AA: find_archetypal_spots(major=True) before find_markers() to populate arche_df.
4) signature_csv is read by `_read_signature_csv`, not ``utils.load_signatures``: in Starfysh 1.2
   that returns a ``(gene_sig, unique_genes)`` tuple (which the old code passed on and crashed on)
   and computes ``unique_genes`` with an ``np.unique`` that raises on any table whose columns have
   different lengths.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from worker_utils import (
    WorkerOutput,
    choose_counts_matrix,
    expression_matrix_kind,
    find_tissue_positions,
    keep_in_tissue,
    read_coords_csv,
    read_tissue_positions,
    record_expression_source,
    record_ignored,
    record_in_tissue,
    record_method,
    resolve_compute,
    safe_symlink_or_copy,
    unsupported_choice_msg,
)

#: The worker's own staging tree under output_dir. The only directory it ever deletes from.
STAGING_DIRNAME = "_starfysh_input"

#: The two ``sig_version`` spellings both dispatches of the installed ``utils.py`` implement
#: (``VisiumArguments.__init__`` and ``_update_anchors``). Anything else raises there -- after two
#: PCAs, two neighbour graphs and two UMAPs -- so it is checked here first.
SIG_VERSIONS = ("gene_score", "norm")

SIGNATURE_MODES = ("auto", "provided", "archetype")

#: The two images ``scanpy.read_visium`` opens unconditionally when it is left at ``load_images=True``,
#: as ``utils.load_adata`` leaves it for a staged ``filtered_feature_bc_matrix.h5``. Either one missing
#: raises ``OSError("Could not find 'hires_image'")`` (scanpy 1.11), in standard Starfysh too.
VISIUM_IMAGE_NAMES = ("tissue_hires_image.png", "tissue_lowres_image.png")

#: Upstream default of ``ArchetypalAnalysis.compute_archetypes(cn=...)``: the FisherS conditional
#: number that selects how many PCs feed the intrinsic-dimension estimate (the lower bound on the
#: archetype count). The portal parameter is called ``aa_r`` for historical reasons; it has always
#: been forwarded to ``cn``.
AA_CN_DEFAULT = 30


# -----------------------------
# Protocol helpers
# -----------------------------
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def emit_json(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.flush()


def _poe_image_advice(input_mode: str | None = None) -> str:
    """How a caller gets an H&E image to PoE *in this input mode*.

    ``hires_image=`` is read only in ``generic_counts_coords`` mode. The two Visium modes take the
    image from the ``spatial/`` folder they are given, so telling their callers to pass
    ``hires_image=`` sent them round the same refusal again (VisiumHD Colon's spatial/ has no
    ``tissue_hires_image.png``).
    """
    if input_mode == "visium_h5_spatial":
        return (
            "put the H&E image in spatial_dir as tissue_hires_image.png (hires_image= is read only in "
            "input_mode='generic_counts_coords' and has no effect in this mode)"
        )
    if input_mode == "spaceranger_outs":
        return (
            "use an outs_dir whose spatial/ folder holds tissue_hires_image.png (hires_image= is read only in "
            "input_mode='generic_counts_coords' and has no effect in this mode)"
        )
    if input_mode == "generic_counts_coords":
        return "pass hires_image=<path to the H&E image>"
    return (
        "pass hires_image=<path> in input_mode='generic_counts_coords', or a spatial_dir / outs_dir that contains "
        "tissue_hires_image.png"
    )


def _poe_no_image_message(input_mode: str | None = None) -> str:
    return (
        "use_poe=True was requested but no paired H&E image is available "
        "(spatial/tissue_hires_image.png missing, so preprocess_img returns img=None). "
        "PoE is Starfysh's histology-integrated model and cannot run without it. To run PoE, "
        f"{_poe_image_advice(input_mode)}; or pass use_poe=False to run standard Starfysh, or "
        "allow_poe_fallback=True to let this run degrade to standard Starfysh (the payload then records "
        "params.used_fallback=true)."
    )


def _effective_use_poe(
    requested: bool, img: Any, allow_fallback: bool = False, input_mode: str | None = None
) -> tuple[bool, str | None]:
    """Decide whether PoE can actually run, given the image `preprocess_img` returned.

    PoE is Starfysh's histology-guided mode and needs the paired H&E image.
    `VisiumPoEDataSet.__init__` starts with ``self.image = args.img.astype(np.float64)``, so a
    missing image raises `AttributeError` from library internals -- and it does so *after* archetypal
    analysis and anchor detection, minutes into the run. The library's own
    ``assert self.image is not None`` sits one line below that attribute access and can never fire.

    `preprocess_img` returns ``img=None`` whenever ``spatial/tissue_hires_image.png`` is absent,
    which is the normal case for ``generic_counts_coords`` input without ``hires_image=``. Since the
    portal defaults ``use_poe=True``, that combination is reachable without the caller asking for it.

    A missing image is an error unless ``allow_fallback`` (the portal's ``allow_poe_fallback``,
    default False) permits running standard Starfysh instead; then ``(False, warning)`` is returned
    and the caller records the substitution. Tests: `test/test_starfysh_poe_requires_an_image.py`.
    """
    if not requested:
        return False, None
    if img is None:
        if not allow_fallback:
            raise ValueError(_poe_no_image_message(input_mode))
        return False, (
            "use_poe=True was requested but no paired H&E image was loaded "
            "(spatial/tissue_hires_image.png missing, so preprocess_img returned img=None). "
            "allow_poe_fallback=True: running standard Starfysh (AVAE) without PoE image integration. "
            f"To enable PoE, {_poe_image_advice(input_mode)}; or pass use_poe=False to select this mode explicitly."
        )
    return True, None


def _require_poe_image(
    vis: VisiumLikeInput, requested: bool, allow_fallback: bool, input_mode: str | None = None
) -> None:
    """Fail a PoE request with no image *before* loading data, on the same test `preprocess_img`
    applies (``isfile(<data>/<sample>/spatial/tissue_hires_image.png)``). The decision itself is
    still taken by `_effective_use_poe` once the image has actually been loaded. The message says
    how to supply an image in *this* input mode."""
    if not requested or allow_fallback:
        return
    hires = Path(vis.data_folder) / vis.sample_id / "spatial" / "tissue_hires_image.png"
    if not hires.is_file():
        raise ValueError(_poe_no_image_message(input_mode) + f" Looked for: {hires}")


def _prepare_poe_image(img_metadata: dict) -> int:
    """Return the channel count PoE must be built for, dropping a PNG alpha channel first.

    ``AVAE_PoE`` sizes its image encoder as ``patch_r**2 * 4 * params['n_img_chan']``, and
    ``VisiumArguments`` sets ``n_img_chan = 3`` only inside ``_update_spatial_info`` -- which it
    skips whenever the counts already carry ``uns['spatial']``. ``sc.read_visium`` always writes
    that key, so every PoE run on a Visium / Space Ranger input with an RGB H&E image built a
    1-channel encoder and died in the first batch ("mat1 and mat2 shapes cannot be multiplied
    (32x3072 and 1024x256)" at patch_r=16). The count is therefore taken from the image itself.
    The PoE dataloader cuts ``(2r, 2r, 3)`` patches, so an RGBA image loses its alpha channel
    (opacity, not stain) before it is used.
    """
    img = img_metadata["img"]
    if img.ndim == 3 and img.shape[2] == 4:
        img_metadata["img"] = img = img[..., :3]
        log("[Starfysh worker] H&E image is RGBA; using its RGB channels for PoE")
    if img.ndim == 2:
        return 1
    if img.ndim == 3 and img.shape[2] == 3:
        return 3
    raise ValueError(
        f"the H&E image loaded from spatial/tissue_hires_image.png has shape {tuple(img.shape)}; PoE takes a "
        "grayscale (H, W) or RGB (H, W, 3) image."
    )


def _record_spatial_without_image(adata: Any, adata_norm: Any, img_metadata: dict, sample_id: str) -> bool:
    """Give image-less counts the ``uns['spatial']`` / ``obsm['spatial']`` Starfysh would build.

    ``VisiumArguments`` calls ``_update_spatial_info`` when the counts carry no ``uns['spatial']``
    and a scalefactors file was read, and that method normalises the image unconditionally
    (``self.img - self.img.min()``). With no image -- a counts-only h5ad in generic_counts_coords
    mode, standard Starfysh -- it raised ``AttributeError: 'NoneType' object has no attribute
    'min'`` before training. The same two entries are written here without the image, from the
    coordinates ``preprocess_img`` returned; nothing the model reads depends on them.
    """
    if img_metadata.get("img") is not None or img_metadata.get("scalefactor") is None:
        return False
    if "spatial" in adata.uns:
        return False
    coords = np.asarray(img_metadata["map_info"][["imagecol", "imagerow"]].values, dtype=float)
    for a in (adata, adata_norm):
        a.uns["spatial"] = {str(sample_id): {"images": {}, "scalefactors": img_metadata["scalefactor"]}}
        a.obsm["spatial"] = coords
    return True


def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _atomic_replace(tmp: str, final: str) -> None:
    os.replace(tmp, final)


def _write_csv_atomic(df: pd.DataFrame, path: str, **to_csv_kwargs: Any) -> None:
    tmp = path + ".partial"
    df.to_csv(tmp, **to_csv_kwargs)
    _atomic_replace(tmp, path)


# -----------------------------
# NumPy 2.x compatibility shim
# -----------------------------
# py_pcha (Starfysh AA dependency) uses np.mat, removed in NumPy 2.0.
if not hasattr(np, "mat"):
    np.mat = np.asmatrix  # type: ignore[attr-defined]


# -----------------------------
# SciPy sparse compatibility shim
# -----------------------------
# Starfysh dataloader uses `adata.X.A` for sparse matrices.
# Some SciPy builds do not expose `.A` on sparse matrices -> add it.
try:
    import scipy.sparse as sp  # type: ignore

    def _add_A_alias(cls):
        if hasattr(cls, "A"):
            return
        cls.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]

    _add_A_alias(sp.csr_matrix)
    _add_A_alias(sp.csc_matrix)
    _add_A_alias(sp.coo_matrix)
except Exception:
    pass


# -----------------------------
# Input normalization
# -----------------------------
@dataclass
class VisiumLikeInput:
    data_folder: str
    sample_id: str
    #: Spots in the counts the caller supplied, and how many of them were left out as background
    #: (``in_tissue == 0``) before staging. Set by the generic_counts_coords staging only; the
    #: Visium modes stage the caller's h5 as it is (Space Ranger's filtered matrix is in-tissue).
    n_spots_supplied: int | None = None
    n_spots_off_tissue: int = 0
    #: ``worker_utils.choose_counts_matrix``'s decision for the staged counts (generic_counts_coords
    #: mode, where the worker reads the h5ad itself); None for the Visium modes, whose 10x matrix is
    #: read by Starfysh and checked once it is loaded.
    expression_info: dict | None = None


#: The inputs each ``input_mode`` reads. Any other input the caller supplied has no effect in that
#: mode and is reported under ``params.ignored`` (``hires_image`` in visium_h5_spatial mode used to
#: vanish without a word -- while the PoE refusal told the caller to pass it).
MODE_INPUTS = {
    "visium_h5_spatial": ("count_h5", "spatial_dir"),
    "spaceranger_outs": ("outs_dir",),
    "generic_counts_coords": ("counts_h5ad", "coords_csv", "hires_image", "scalefactors_json"),
}
ALL_MODE_INPUTS = (
    "count_h5",
    "spatial_dir",
    "outs_dir",
    "counts_h5ad",
    "coords_csv",
    "hires_image",
    "scalefactors_json",
)


def _inputs_the_mode_ignores(input_mode: str, payload: dict[str, Any]) -> list[str]:
    """The input paths the caller supplied that ``input_mode`` never reads."""
    reads = MODE_INPUTS.get(input_mode, ())
    return [name for name in ALL_MODE_INPUTS if name not in reads and payload.get(name)]


def _visium_images_missing(vis: VisiumLikeInput) -> list[str]:
    """The images ``scanpy.read_visium`` would open that the sample handed to Starfysh lacks.

    ``utils.load_adata`` calls ``sc.read_visium(path, library_id=...)`` -- images included -- only
    when the sample holds ``filtered_feature_bc_matrix.h5`` (the Visium modes); an h5ad sample is read
    with ``read_h5ad`` and needs no image. Empty when nothing is missing or no h5 is read.
    """
    root = Path(vis.data_folder) / vis.sample_id
    if not (root / "filtered_feature_bc_matrix.h5").is_file():
        return []
    return [name for name in VISIUM_IMAGE_NAMES if not (root / "spatial" / name).is_file()]


class _ReadVisiumCountsOnly:
    """While active, ``scanpy.read_visium`` reads the counts without the tissue images.

    ``utils.load_adata`` gives ``read_visium`` no way to skip them, and scanpy raises
    ``OSError("Could not find 'hires_image'")`` when either PNG is absent -- so a Visium sample with no
    image (the library's VisiumHD Colon exports) crashed in standard Starfysh as well as in PoE, while
    the PoE refusal sent the caller to use_poe=False. The images are never Starfysh's input: the
    positions and scalefactors come from ``preprocess_img``, which also reads the H&E image PoE needs.
    So the counts are read with ``load_images=False``; the patch is undone on exit. Nothing is done
    when no image is missing. (A class rather than ``contextlib.contextmanager``, so the staging
    helpers can be lifted out of this module without its imports.)
    """

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        self._sc = None
        self._original = None

    def __enter__(self) -> _ReadVisiumCountsOnly:
        if not self.missing:
            return self
        import scanpy as sc

        original = getattr(sc, "read_visium", None)
        if original is None:
            return self

        def read_visium_counts_only(*args: Any, **kwargs: Any):
            kwargs["load_images"] = False
            return original(*args, **kwargs)

        log(
            f"[Starfysh worker] spatial/ has no {', '.join(self.missing)}; reading the Visium counts without "
            "images (scanpy read_visium load_images=False)"
        )
        self._sc, self._original = sc, original
        sc.read_visium = read_visium_counts_only
        return self

    def __exit__(self, *exc: Any) -> bool:
        if self._sc is not None:
            self._sc.read_visium = self._original
            self._sc = self._original = None
        return False


def _drop_empty_uns_spatial(*adatas: Any) -> None:
    """Remove the ``uns['spatial']`` stub ``read_visium(load_images=False)`` leaves: ``{library_id: {}}``.

    It holds no scalefactors and no image, yet its presence makes ``VisiumArguments`` skip
    ``_update_spatial_info`` and `_record_spatial_without_image` skip writing the coordinates. Without
    it, both behave as for any image-less h5ad.
    """
    for a in adatas:
        entry = a.uns.get("spatial") if "spatial" in a.uns else None
        if isinstance(entry, dict) and all(isinstance(v, dict) and not v for v in entry.values()):
            del a.uns["spatial"]


def _counts_kind_of_loaded(adata: Any) -> dict[str, Any]:
    """``choose_counts_matrix``'s report for counts Starfysh read itself (the Visium modes' 10x matrix).

    A 10x ``filtered_feature_bc_matrix.h5`` stores integer UMI counts and has no ``adata.raw``, so
    there is nothing to choose; the stored values are still checked, and a matrix that cannot be
    counts is refused as in generic_counts_coords mode.
    """
    kind = expression_matrix_kind(adata.X)
    if kind in ("negative", "nonfinite"):
        what = "negative values" if kind == "negative" else "NaN or infinite values"
        raise ValueError(
            f"the count matrix Starfysh read holds {what}, not counts; Starfysh's preprocessing normalises X as "
            "counts. Supply a 10x filtered_feature_bc_matrix.h5 of raw UMI counts."
        )
    warning = None
    if kind == "nonnegative_noninteger":
        warning = (
            "the count matrix Starfysh read holds non-integer values (normalised data?), and Starfysh's "
            "preprocessing normalises X as counts, so the result was computed on a matrix normalised twice."
        )
    return {"expression_source": "X", "x_matrix_kind": kind, "warning": warning}


def _make_visium_like_from_h5_and_spatial(
    output_dir: str,
    sample_id: str,
    count_h5: str,
    spatial_dir: str,
) -> VisiumLikeInput:
    """
    Create a Starfysh-readable Visium-like tree:
      <output_dir>/_starfysh_input/<sample_id>/filtered_feature_bc_matrix.h5
      <output_dir>/_starfysh_input/<sample_id>/spatial/<...>
    """
    tmp_root = Path(output_dir) / STAGING_DIRNAME
    sample_root = tmp_root / sample_id
    ensure_dir(str(tmp_root))
    _reset_staged_sample(sample_root, tmp_root, {"count_h5": count_h5, "spatial_dir": spatial_dir})
    ensure_dir(str(sample_root / "spatial"))

    safe_symlink_or_copy(count_h5, str(sample_root / "filtered_feature_bc_matrix.h5"))

    spdir = Path(spatial_dir)
    if not spdir.exists():
        raise FileNotFoundError(f"spatial_dir not found: {spatial_dir}")
    for item in spdir.iterdir():
        if item.is_file():
            safe_symlink_or_copy(str(item), str(sample_root / "spatial" / item.name))

    _normalise_tissue_positions_name(sample_root / "spatial")

    req = [
        sample_root / "filtered_feature_bc_matrix.h5",
        sample_root / "spatial" / "tissue_positions_list.csv",
        sample_root / "spatial" / "scalefactors_json.json",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Visium files: {missing}")

    return VisiumLikeInput(data_folder=str(tmp_root), sample_id=sample_id)


def _normalise_tissue_positions_name(staged_spatial: Path) -> None:
    """Give a staged Space Ranger 2.0 sample the pre-2.0 filename Starfysh reads.

    Space Ranger 2.0 renamed ``tissue_positions_list.csv`` to ``tissue_positions.csv`` and added a
    header row. Starfysh's loader asks for the old name, so a v2 sample was rejected outright. We
    write the old name -- headerless, six columns, the pre-2.0 shape -- *beside* the new one, and
    only ever inside our own ``_starfysh_input`` staging tree, never in the user's directory.
    """
    v1 = staged_spatial / "tissue_positions_list.csv"
    if v1.exists():
        return
    found = find_tissue_positions(staged_spatial)
    if found is None:
        return
    read_tissue_positions(found).to_csv(v1, header=False, index=False)
    log(f"[Starfysh worker] staged {Path(found).name} as {v1.name} (Space Ranger 2.0 layout)")


def _make_visium_like_from_spaceranger_outs(outs_dir: str, output_dir: str | None = None) -> VisiumLikeInput:
    outs = Path(outs_dir)
    if not outs.exists():
        raise FileNotFoundError(f"outs_dir not found: {outs_dir}")

    # A Space Ranger 2.0 tree is complete but spells the positions file differently, and we must not
    # write into the user's outs/. Re-stage it under output_dir instead, where the name can be fixed.
    positions = find_tissue_positions(outs / "spatial")
    if output_dir and positions is not None and not (outs / "spatial" / "tissue_positions_list.csv").exists():
        return _make_visium_like_from_h5_and_spatial(
            output_dir=output_dir,
            sample_id=outs.name if outs.name != "outs" else outs.parent.name,
            count_h5=str(outs / "filtered_feature_bc_matrix.h5"),
            spatial_dir=str(outs / "spatial"),
        )

    req = [
        outs / "filtered_feature_bc_matrix.h5",
        outs / "spatial" / "tissue_positions_list.csv",
        outs / "spatial" / "scalefactors_json.json",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError(f"outs_dir missing required files: {missing}")

    # The user's own directory is read in place. `utils.load_adata` opens filtered_feature_bc_matrix.h5
    # (required above), so any *.h5ad beside it is never consulted -- and never touched by this worker.
    return VisiumLikeInput(data_folder=str(outs.parent), sample_id=outs.name)


def _reset_staged_sample(sample_root: Path, staging_root: Path, inputs: dict[str, Any] | None = None) -> bool:
    """Empty the worker's own per-sample staging directory so this run stages from scratch.

    ``utils.load_adata`` reads ``filtered_feature_bc_matrix.h5`` in preference to any ``.h5ad`` and
    asserts exactly one ``.h5ad`` otherwise; ``preprocess_img`` reads whatever
    ``spatial/tissue_hires_image.png`` is there; and ``safe_symlink_or_copy`` leaves a real file
    already in place untouched. So a re-used output_dir served files staged by an *earlier* run --
    another sample's h5, image or scalefactors -- in place of this run's input, without a word.

    Only ``<staging_root>/<sample_id>`` is removed, and only when every guard holds:
    ``staging_root`` is named ``_starfysh_input`` and is not a symlink; ``sample_root`` is not a
    symlink and resolves strictly inside ``staging_root``; and no input the caller passed resolves
    into it (a path handed in from a previous run's staging tree would be deleted before it is
    read). Any failed guard raises before anything is deleted. ``shutil.rmtree`` unlinks the
    symlinks it finds and never follows them, so the files they point at are untouched.

    Returns True when an earlier staging directory was removed.
    """
    staging_root = Path(staging_root)
    sample_root = Path(sample_root)
    if staging_root.name != STAGING_DIRNAME or staging_root.is_symlink():
        raise RuntimeError(
            f"refusing to reset {sample_root}: its parent {staging_root} is not the worker's own "
            f"{STAGING_DIRNAME} staging directory (or is a symlink). This worker never removes files from a "
            "directory it did not create."
        )
    resolved_root = staging_root.resolve()
    resolved_sample = sample_root.resolve()
    if sample_root.is_symlink() or resolved_root not in resolved_sample.parents:
        raise ValueError(
            f"sample_id must name a directory inside {staging_root}; {sample_root} resolves to {resolved_sample}. "
            "Use a plain sample name (no '..', no absolute path)."
        )
    for name, value in (inputs or {}).items():
        if not value:
            continue
        real = Path(str(value)).resolve()
        if real == resolved_sample or resolved_sample in real.parents:
            raise ValueError(
                f"{name}={value!r} lives inside this worker's staging directory {sample_root}, which is rebuilt "
                "from scratch on every run. Pass the original file or directory instead."
            )
    if not os.path.lexists(sample_root):
        return False
    shutil.rmtree(sample_root)
    log(f"[Starfysh worker] cleared the previous staging directory {sample_root} (the worker's own copy)")
    return True


def _make_visium_like_from_counts_and_coords(
    output_dir: str,
    sample_id: str,
    counts_h5ad: str,
    coords_csv: str,
    hires_image: str | None = None,
    scalefactors_json: str | None = None,
    use_raw_counts: bool = False,
) -> VisiumLikeInput:
    """
    Generic mode:
      - counts_h5ad: AnnData with counts (obs_names=barcodes)
      - coords_csv: barcode,x,y[,in_tissue]
    We synthesize a Visium tissue_positions_list.csv and scalefactors_json.json.

    Background spots (``in_tissue == 0`` in the counts' own ``obs``, else in the coords file) are
    left out of the staged counts and counted on the returned object, and the matrix staged is the
    one ``choose_counts_matrix`` picks (``use_raw_counts``); see `_stage_counts_in_tissue`.
    """
    tmp_root = Path(output_dir) / STAGING_DIRNAME
    sample_root = tmp_root / sample_id
    spatial_root = sample_root / "spatial"
    ensure_dir(str(tmp_root))
    _reset_staged_sample(
        sample_root,
        tmp_root,
        {
            "counts_h5ad": counts_h5ad,
            "coords_csv": coords_csv,
            "hires_image": hires_image,
            "scalefactors_json": scalefactors_json,
        },
    )
    ensure_dir(str(spatial_root))

    # Starfysh is the most lenient of the four coords readers: it accepts several spellings of the
    # spot-ID column, and falls back to the first column when that column is not a coordinate.
    df = read_coords_csv(
        coords_csv,
        barcode_aliases=("spot_id", "spot", "cell_id", "index", "Unnamed: 0"),
        barcode_from_first_column=True,
    )

    n_supplied, n_off_tissue, expression_info = _stage_counts_in_tissue(
        counts_h5ad, sample_root / Path(counts_h5ad).name, df, use_raw_counts=use_raw_counts
    )

    tp = pd.DataFrame(
        {
            0: df["barcode"].astype(str).values,
            1: df["in_tissue"].astype(int).values,
            2: df["y"].astype(float).values,
            3: df["x"].astype(float).values,
            4: df["y"].astype(float).values,
            5: df["x"].astype(float).values,
        }
    )
    tp.to_csv(spatial_root / "tissue_positions_list.csv", header=False, index=False)

    sf_path = spatial_root / "scalefactors_json.json"
    if scalefactors_json:
        safe_symlink_or_copy(scalefactors_json, str(sf_path))
    else:
        sf_path.write_text(
            json.dumps(
                {
                    "tissue_hires_scalef": 1.0,
                    "tissue_lowres_scalef": 1.0,
                    "fiducial_diameter_fullres": 1.0,
                    "spot_diameter_fullres": 1.0,
                }
            )
        )

    if hires_image:
        safe_symlink_or_copy(hires_image, str(spatial_root / "tissue_hires_image.png"))

    return VisiumLikeInput(
        data_folder=str(tmp_root),
        sample_id=sample_id,
        n_spots_supplied=n_supplied,
        n_spots_off_tissue=n_off_tissue,
        expression_info=expression_info,
    )


def _stage_counts_in_tissue(
    counts_h5ad: str, dest: Path, coords: pd.DataFrame, use_raw_counts: bool = False
) -> tuple[int, int, dict[str, Any]]:
    """Stage ``counts_h5ad`` at ``dest`` for ``utils.load_adata``: in-tissue spots, and a count matrix.

    CELLxGENE Visium exports carry every array spot with ``obs['in_tissue']`` 0/1, and Starfysh's
    h5ad loader reads them all: the background glass was deconvolved as tissue. (On Muscle
    Gastrocnemius Donor1, 3396 of 4992 spots are background; they have no counts, so Starfysh's own
    preprocessing then dropped them without a word.) The flag is the counts' own
    ``obs['in_tissue']`` when present, else the ``in_tissue`` column of ``coords_csv`` matched by
    barcode (a barcode the coords file does not list is not called background). The rule is
    ``worker_utils.keep_in_tissue``'s, shared with every worker that leaves background out.

    Starfysh normalises X as counts and fits a count model to it, and those same exports hold a
    processed X with the counts in ``adata.raw`` (Muscle's X is log-normalised, Skin's z-scored). The
    in-tissue spots then go through ``worker_utils.choose_counts_matrix``: negative or non-finite
    values are refused (naming ``use_raw_counts`` when ``adata.raw`` holds counts), a non-integer X is
    staged with a warning, and ``use_raw_counts=True`` stages ``adata.raw.X``. Every stored value is
    read, so the whole file is loaded.

    With nothing left out and X chosen, the caller's file is linked in place, as before. Otherwise the
    chosen matrix of the in-tissue spots is written to ``dest`` (``.partial`` then ``os.replace``)
    inside the worker's own staging directory; the caller's file is never modified. Returns
    ``(n_supplied, n_off_tissue, expression_info)``.
    """
    import anndata as ad

    full = ad.read_h5ad(counts_h5ad)
    n_supplied = int(full.n_obs)
    if "in_tissue" in full.obs.columns:
        flag = np.asarray(full.obs["in_tissue"], dtype=object)
    else:
        by_barcode = coords.drop_duplicates("barcode").set_index("barcode")["in_tissue"]
        flag = np.asarray(pd.Series(full.obs_names.astype(str)).map(by_barcode).fillna(1), dtype=object)
    probe = ad.AnnData(obs=pd.DataFrame({"in_tissue": flag}, index=[str(i) for i in range(n_supplied)]))
    kept, _, n_off_tissue = keep_in_tissue(probe, what="spots")
    if n_off_tissue:
        rows = np.asarray(kept.obs_names, dtype=int)
        log(
            f"[Starfysh worker] leaving out {n_off_tissue} of {n_supplied} spots with in_tissue == 0 (background); "
            f"staging the {rows.size} in-tissue spots"
        )
        full = full[rows].copy()
    chosen, expression_info = choose_counts_matrix(full, use_raw_counts)
    if not n_off_tissue and expression_info["expression_source"] == "X":
        del full, chosen
        safe_symlink_or_copy(counts_h5ad, str(dest))
        return n_supplied, 0, expression_info
    if expression_info["expression_source"] != "X":
        log(f"[Starfysh worker] use_raw_counts=True: staging adata.raw.X ({chosen.n_vars} genes) as the counts")
    partial = dest.with_name(dest.name + ".partial")
    chosen.write_h5ad(str(partial))
    os.replace(str(partial), str(dest))
    return n_supplied, int(n_off_tissue), expression_info


def _count_spots_load_adata_reads(vis: VisiumLikeInput) -> int | None:
    """How many spots ``utils.load_adata`` is about to read, taken from the file it will open.

    Mirrors its precedence: ``filtered_feature_bc_matrix.h5`` (the barcode list's length), else the
    one ``*.h5ad`` (``obs`` length, read backed so ``X`` is never loaded). Used only to report what
    Starfysh's own preprocessing drops, so an unreadable file gives None instead of an error.
    """
    root = Path(vis.data_folder) / vis.sample_id
    try:
        h5 = root / "filtered_feature_bc_matrix.h5"
        if h5.is_file():
            import h5py

            with h5py.File(str(h5), "r") as fh:
                groups = ["matrix"] if "matrix" in fh else list(fh.keys())
                for name in groups:
                    grp = fh[name]
                    if isinstance(grp, h5py.Group) and "barcodes" in grp:
                        return int(grp["barcodes"].shape[0])
            return None
        h5ads = [p for p in root.iterdir() if p.name.endswith(".h5ad")]
        if len(h5ads) != 1:
            return None
        import anndata as ad

        backed = ad.read_h5ad(str(h5ads[0]), backed="r")
        try:
            return int(backed.n_obs)
        finally:
            backed.file.close()
    except Exception as exc:  # reporting only: never fail a run over a count
        log(f"[Starfysh worker] could not count the spots handed to Starfysh ({type(exc).__name__}: {exc})")
        return None


# -----------------------------
# AA compatibility helpers
# -----------------------------
def _check_aa_cn(aa_cn: int) -> int:
    """``aa_r`` is forwarded to ``compute_archetypes(cn=...)``; skdim divides by it."""
    aa_cn = int(aa_cn)
    if aa_cn <= 0:
        raise ValueError(
            f"aa_r={aa_cn} is not usable: it is passed to ArchetypalAnalysis.compute_archetypes(cn=...), the "
            "FisherS conditional number that selects the PCs for the intrinsic-dimension estimate, and skdim "
            f"divides by it (ZeroDivisionError at 0). Pass a positive integer; the upstream default is {AA_CN_DEFAULT}."
        )
    return aa_cn


def _compute_archetypes_compat(aa_model: Any, aa_param: int) -> None:
    """Call ``compute_archetypes(cn=aa_param)``; never fall back to the library defaults silently."""
    aa_param = _check_aa_cn(aa_param)
    fn = aa_model.compute_archetypes
    sig = inspect.signature(fn)
    if "cn" not in sig.parameters:
        raise RuntimeError(
            f"the installed ArchetypalAnalysis.compute_archetypes{sig} has no 'cn' parameter, so aa_r={aa_param} "
            "cannot be applied. This worker targets Starfysh 1.2 (compute_archetypes(cn=30, n_iters=20, "
            "converge=1e-3, display=False)); it does not run the library defaults in place of the requested value."
        )
    fn(cn=aa_param, display=False)


def _find_archetypal_spots_compat(aa_model: Any) -> Any:
    fn = getattr(aa_model, "find_archetypal_spots", None)
    if fn is None or not callable(fn):
        raise RuntimeError(
            "AA model missing find_archetypal_spots(); cannot populate arche_df required by find_markers()."
        )
    try:
        return fn(major=True)
    except TypeError:
        return fn()


def _read_signature_csv(signature_csv: str, adata: Any) -> tuple[pd.DataFrame, str | None]:
    """Read a marker table the way ``utils.load_signatures`` means to, without its two defects.

    Upstream (Starfysh 1.2, ``utils.py:631-658``) does ``pd.read_csv(filename, index_col=0)``, then
    ``filter_gene_sig(gene_sig, adata.to_df())``, then also returns
    ``np.unique(gene_sig.apply(lambda x: pd.unique(x[~pd.isna(x)])).values)``. When the columns hold
    different numbers of markers -- almost every real marker table -- that ``apply`` yields a ragged
    object Series and ``np.unique`` raises "The truth value of an array with more than one element
    is ambiguous". That second value is never used here, so the table is read directly and filtered
    with upstream's own rule (`_filter_gene_sig_sparse`, no densified spots x genes frame).

    ``index_col=0`` also meant the first column was never a signature, so a table laid out as the
    docs describe (one column per cell type, no index column) lost its first cell type silently.
    Here the first column is the row index when its header is blank / ``Unnamed: N`` or none of its
    values is a gene of the counts, and a signature otherwise. Every signature must share at least
    one gene with the counts; upstream raises "Empty signatures" for one that does not, but only
    after two PCAs, neighbour graphs and UMAPs.

    Returns ``(gene_sig, note)``; ``note`` says how a *named* first column was treated, else None.
    """
    if not os.path.isfile(signature_csv):
        raise FileNotFoundError(f"signature_csv not found: {signature_csv}")
    raw = pd.read_csv(signature_csv)
    if raw.shape[1] == 0:
        raise ValueError(f"signature_csv={signature_csv!r} has no columns.")
    genes = set(map(str, adata.var_names))
    first = raw.columns[0]
    header = str(first).strip()
    named = bool(header) and not header.startswith("Unnamed:")
    first_has_genes = bool(raw[first].dropna().astype(str).isin(genes).any())
    note = None
    if named and first_has_genes:
        gene_sig = raw
        note = (
            f"signature_csv has no index column: its first column {header!r} lists genes of the counts and is used "
            "as a signature (Starfysh's utils.load_signatures would have read it as the row index and dropped it)."
        )
    else:
        gene_sig = raw.set_index(first)
        if named:
            note = (
                f"signature_csv: first column {header!r} was read as the row index (none of its values is a gene of "
                "the counts), not as a signature."
            )
    if gene_sig.shape[1] == 0:
        raise ValueError(
            f"signature_csv={signature_csv!r} has no signature columns: lay it out as one column per cell type / state "
            "with its marker genes listed down it (an index column in front is optional)."
        )
    empty = [str(c) for c in gene_sig.columns if not gene_sig[c].dropna().astype(str).isin(genes).any()]
    if empty:
        raise ValueError(
            f"signature(s) {empty} in signature_csv name no gene of the counts ({len(genes)} genes after Starfysh's "
            "preprocessing). Check the gene naming (symbols vs Ensembl IDs) or drop those columns."
        )
    return _filter_gene_sig_sparse(gene_sig, adata), note


def _filter_gene_sig_sparse(gene_sig: pd.DataFrame, adata: Any) -> pd.DataFrame:
    """``utils.filter_gene_sig(gene_sig, adata.to_df())`` without densifying ``adata``.

    Upstream's rule: a signature gene whose column sum in ``adata`` is negative is replaced by the
    string ``'NaN'``. The upstream helper needs a dense spots x genes DataFrame to test that; the
    column sums come straight off the sparse (or dense) matrix here.
    """
    sums = np.asarray(adata.X.sum(axis=0)).ravel()
    negative = set(np.asarray(adata.var_names)[sums < 0].tolist())
    if not negative:
        return gene_sig
    return gene_sig.mask(gene_sig.isin(negative), "NaN")


def _derive_signatures_by_aa(AA_module: Any, adata: Any, adata_norm: Any, aa_cn: int, n_markers: int):
    """Archetypal Analysis -> archetypal spots -> Wilcoxon markers per archetype. No substitute."""
    if int(n_markers) <= 0:
        raise ValueError(f"aa_n_markers={n_markers} must be a positive integer (marker genes per archetype).")
    log("[Starfysh worker] No signatures provided; running Archetypal Analysis to derive marker sets...")
    aa_model = AA_module.ArchetypalAnalysis(adata_orig=adata_norm)
    try:
        _compute_archetypes_compat(aa_model, aa_cn)
        _find_archetypal_spots_compat(aa_model)
        gene_sig = aa_model.find_markers(n_markers=int(n_markers), display=False)
        # Upstream's own post-step, applied without densifying the spots x genes matrix.
        gene_sig = _filter_gene_sig_sparse(gene_sig, adata)
    except (ZeroDivisionError, ValueError, RuntimeError, AssertionError) as e:
        raise RuntimeError(
            f"Archetypal Analysis failed ({type(e).__name__}: {e}). No substitute signatures are generated. "
            "Supply signature_csv=<markers.csv> (signature_mode='provided'), or adjust aa_r (FisherS conditional "
            f"number, positive integer, upstream default {AA_CN_DEFAULT}), aa_n_markers and n_genes."
        ) from e
    return gene_sig


# -----------------------------
# Output helpers
# -----------------------------
def _select_device(device: str) -> str:
    """Any device spelling -> one torch.device() accepts, never CUDA on a box without it.

    Shared with every other worker (worker_utils.resolve_compute); the caller feeds the result
    straight into torch.device(), which used to raise on 'GPU'/'CPU'.
    """
    return resolve_compute(device).device


def _proportion_columns(adata, signature_source: str | None = None) -> tuple[list[str], dict[str, Any] | None]:
    """The column names ``cell_type_proportions.csv`` carries, and a report when they are not the
    names ``uns['cell_types']`` holds.

    ``VisiumArguments`` stores ``uns['cell_types'] = list(gene_sig.columns)``. For a *provided*
    signature table those are the caller's own column headers, and they are kept exactly as given
    -- numeric ones included (cluster ids '0', '1', '2' from a marker table are names too; an
    earlier revision renamed them ``arch_<i>``, so the CSV and the summary named archetypes that
    never ran while ``gene_signatures_used.csv`` and ``anchor_spots.csv`` kept the real names).
    Archetypal Analysis names its columns ``arch_<idx>`` itself. ``arch_<i>`` is substituted only
    when the stored list cannot name the columns: it has the wrong length, or (for an AA run) it
    holds bare numbers instead of names. The report then says why and maps each written name to
    the stored one, and the caller publishes it; the summary uses the same list as the CSV.
    """
    if "qc_m" not in adata.obsm:
        raise RuntimeError("Expected Starfysh composition in adata.obsm['qc_m'], but it was not found.")
    n_cols = int(adata.obsm["qc_m"].shape[1])
    cell_types = adata.uns.get("cell_types")
    ct_strs = None if cell_types is None else [str(c) for c in list(cell_types)]
    why = None
    if ct_strs is None:
        why = "adata.uns['cell_types'] is missing"
    elif len(ct_strs) != n_cols:
        why = f"adata.uns['cell_types'] lists {len(ct_strs)} names for {n_cols} proportion columns"
    elif signature_source == "provided":
        return ct_strs, None
    else:
        # Reject pollution where uns['cell_types'] holds raw integer indices instead of named
        # factors. Symptom is a downstream evaluator getting columns like ['1', '6982.0', '8346.0']
        # that map to nothing in the aggregation YAML.
        looks_numeric = all((s.replace(".", "").replace("-", "").isdigit()) for s in ct_strs if s)
        if not looks_numeric and ct_strs:
            return ct_strs, None
        why = f"adata.uns['cell_types'] holds bare numbers ({ct_strs[:5]!r}), not factor names"
    names = [f"arch_{i}" for i in range(n_cols)]
    log(f"[Starfysh worker] WARNING: {why}; using arch_* column names.")
    report: dict[str, Any] = {"why": why}
    if ct_strs is not None and len(ct_strs) == n_cols:
        report["renamed"] = dict(zip(names, ct_strs))
    elif ct_strs is not None:
        report["stored_names"] = ct_strs
    return names, report


def _proportion_column_names(adata, signature_source: str | None = None) -> list[str]:
    """The column names ``cell_type_proportions.csv`` carries (see `_proportion_columns`)."""
    return _proportion_columns(adata, signature_source)[0]


def _write_proportions_csv(
    adata, out_csv: str, signature_source: str | None = None, report: dict[str, Any] | None = None
) -> list[str]:
    """Write ``qc_m`` as spots x factors and return the column names that were written.

    ``report`` (optional dict) receives `_proportion_columns`' rename report when names were
    substituted, so the payload can say so.
    """
    names, renamed = _proportion_columns(adata, signature_source)
    if renamed is not None and report is not None:
        report.update(renamed)
    df = pd.DataFrame(np.asarray(adata.obsm["qc_m"]), index=adata.obs_names, columns=names)
    _write_csv_atomic(df, out_csv)
    return names


#: ``utils.run_starfysh`` returns the best restart's losses as ``{'tot': [...], 'reconst': [...], ...}``,
#: one value per epoch. ``tot`` is written as the ``loss`` column; the components keep their names.
_LOSS_COMPONENTS = ("reconst", "u", "z", "c", "n")


def _losses_frame(losses: Any) -> pd.DataFrame:
    """One row per training epoch: ``loss`` (total), then ``loss_<component>`` for each component.

    ``np.asarray`` of the upstream dict is a 0-d object array, so an earlier revision wrote the
    whole dict's repr into one cell of a one-row ``loss`` column.
    """
    if isinstance(losses, dict):
        if "tot" not in losses:
            raise RuntimeError(f"utils.run_starfysh returned losses without a 'tot' series: keys {sorted(losses)}")
        total = [float(v) for v in losses["tot"]]
        frame = pd.DataFrame({"loss": total})
        frame.insert(0, "epoch", np.arange(1, len(total) + 1))
        for key in _LOSS_COMPONENTS:
            series = losses.get(key)
            if series is not None and len(series) == len(total):
                frame[f"loss_{key}"] = [float(v) for v in series]
        return frame
    return pd.DataFrame({"loss": np.asarray(losses, dtype=float).reshape(-1)})


def _run_starfysh_compat(utils_module: Any, **kwargs: Any):
    """Call ``utils.run_starfysh`` with exactly the kwargs given.

    A kwarg the installed signature does not take is an error naming it -- an earlier revision
    dropped such kwargs with a stderr note, so a knob the caller set could vanish without a trace
    in the payload.
    """
    fn = utils_module.run_starfysh
    sig = inspect.signature(fn)
    allowed = set(sig.parameters.keys())
    unsupported = sorted(k for k in kwargs if k not in allowed)
    if unsupported:
        raise TypeError(
            f"the installed utils.run_starfysh{sig} does not accept {unsupported}; the worker does not drop "
            "them silently. These are knobs this Starfysh build cannot apply."
        )
    return fn(**kwargs)


# -----------------------------
# Core execution
# -----------------------------
def run_starfysh_deconvolution(payload: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()

    output_dir = ensure_dir(str(payload["output_dir"]))
    input_mode = str(payload.get("input_mode", "visium_h5_spatial"))
    sample_id = str(payload.get("sample_id") or "sample")

    device = _select_device(str(payload.get("device", "auto")))
    poe_requested = bool(payload.get("use_poe", False))
    use_poe = poe_requested
    allow_poe_fallback = bool(payload.get("allow_poe_fallback", False))
    hchannel = bool(payload.get("hchannel", False))
    use_raw_counts = bool(payload.get("use_raw_counts", False))

    # training knobs
    n_genes = int(payload.get("n_genes", 6000))
    n_repeats = int(payload.get("n_repeats", 3))
    lr = float(payload.get("lr", 1e-3))
    epochs = int(payload.get("epochs", 100))
    # `patience` is accepted for compatibility only: the installed utils.run_starfysh has no early
    # stopping, so it is reported under params.ignored rather than forwarded.
    patience_given = "patience" in payload
    if n_repeats <= 0 or epochs <= 0:
        raise ValueError(f"n_repeats={n_repeats} and epochs={epochs} must both be positive integers.")

    # signatures
    signature_mode = str(payload.get("signature_mode", "auto"))
    signature_csv = str(payload.get("signature_csv", "") or "")
    aa_cn = int(payload.get("aa_r", AA_CN_DEFAULT))
    aa_n_markers = int(payload.get("aa_n_markers", 30))
    if signature_mode not in SIGNATURE_MODES:
        raise ValueError(unsupported_choice_msg("signature_mode", signature_mode, list(SIGNATURE_MODES)))
    if signature_mode == "provided" and not signature_csv:
        raise ValueError("signature_mode=provided requires signature_csv")
    use_provided = signature_mode == "provided" or (signature_mode == "auto" and bool(signature_csv))
    # reason -> knobs it applies to; each reason becomes one params.ignored entry set + one warning.
    ignored: dict[str, list[str]] = {}
    if signature_mode == "archetype" and signature_csv:
        ignored.setdefault("signature_mode='archetype' derives signatures by Archetypal Analysis", []).append(
            "signature_csv"
        )
    if use_provided:
        ignored.setdefault("signatures were provided, so Archetypal Analysis did not run", []).extend(
            ["aa_r", "aa_n_markers"]
        )
    else:
        _check_aa_cn(aa_cn)
        if aa_n_markers <= 0:
            raise ValueError(f"aa_n_markers={aa_n_markers} must be a positive integer (marker genes per archetype).")

    # VisiumArguments knobs. Only names the installed utils.VisiumArguments keeps: its kwargs loop
    # (utils.py:75-77) drops anything not already in self.params. vlow/vhigh/z_axis belonged to the
    # legacy io.VisiumArguments, which no longer imports, so they are not read here or sent above.
    n_anchors = int(payload.get("n_anchors", 60))
    window_size = float(payload.get("window_size", 30))
    patch_r = int(payload.get("patch_r", 13))

    # sig_version: only pass if the caller provides one; an empty string defers to the library default.
    sig_version = payload.get("sig_version", None)
    if isinstance(sig_version, str) and sig_version.strip() == "":
        sig_version = None
    if sig_version is not None and str(sig_version) not in SIG_VERSIONS:
        raise ValueError(
            unsupported_choice_msg(
                "sig_version",
                sig_version,
                list(SIG_VERSIONS),
                extra="Leave it empty to use the library default (gene_score).",
            )
        )

    log(f"[Starfysh worker] input_mode={input_mode} sample_id={sample_id} device={device} use_poe={use_poe}")

    # ---- Input normalization
    if input_mode == "visium_h5_spatial":
        count_h5 = payload.get("count_h5")
        spatial_dir = payload.get("spatial_dir")
        if not count_h5 or not spatial_dir:
            raise ValueError("input_mode=visium_h5_spatial requires count_h5 and spatial_dir")
        vis = _make_visium_like_from_h5_and_spatial(output_dir, sample_id, str(count_h5), str(spatial_dir))
    elif input_mode == "spaceranger_outs":
        outs_dir = payload.get("outs_dir")
        if not outs_dir:
            raise ValueError("input_mode=spaceranger_outs requires outs_dir")
        vis = _make_visium_like_from_spaceranger_outs(str(outs_dir), output_dir=output_dir)
        sample_id = vis.sample_id
    elif input_mode == "generic_counts_coords":
        counts_h5ad = payload.get("counts_h5ad")
        coords_csv = payload.get("coords_csv")
        if not counts_h5ad or not coords_csv:
            raise ValueError("input_mode=generic_counts_coords requires counts_h5ad and coords_csv")
        hires_image = payload.get("hires_image")
        scalefactors_json = payload.get("scalefactors_json")
        vis = _make_visium_like_from_counts_and_coords(
            output_dir=output_dir,
            sample_id=sample_id,
            counts_h5ad=str(counts_h5ad),
            coords_csv=str(coords_csv),
            hires_image=str(hires_image) if hires_image else None,
            scalefactors_json=str(scalefactors_json) if scalefactors_json else None,
            use_raw_counts=use_raw_counts,
        )
    else:
        raise ValueError(
            unsupported_choice_msg(
                "input_mode", input_mode, ["visium_h5_spatial", "spaceranger_outs", "generic_counts_coords"]
            )
        )

    # Inputs this mode never reads (hires_image in visium_h5_spatial mode, ...) are reported, not dropped silently.
    mode_ignored = _inputs_the_mode_ignores(input_mode, payload)
    if mode_ignored:
        ignored.setdefault(f"input_mode={input_mode!r} reads only {', '.join(MODE_INPUTS[input_mode])}", []).extend(
            mode_ignored
        )
    if use_raw_counts and input_mode != "generic_counts_coords":
        ignored.setdefault(
            f"input_mode={input_mode!r} reads a 10x count matrix (filtered_feature_bc_matrix.h5), which holds the raw "
            "counts in its matrix and has no adata.raw; use_raw_counts is read only in generic_counts_coords mode",
            [],
        ).append("use_raw_counts")

    # A PoE request with no image is refused here, before any data is loaded, rather than minutes
    # later inside the PoE dataloader. The message says how to supply the image in this mode.
    _require_poe_image(vis, use_poe, allow_poe_fallback, input_mode)

    # Spots handed to Starfysh, counted from the file load_adata is about to open: its own
    # preprocessing drops spots (below), and the payload has to say how many.
    n_spots_to_starfysh = _count_spots_load_adata_reads(vis)
    n_spots_input = vis.n_spots_supplied if vis.n_spots_supplied is not None else n_spots_to_starfysh

    # ---- Starfysh imports (inside tool env)
    import torch
    from starfysh import AA, utils
    from starfysh import starfysh as sf_model

    # Load & preprocess. Nothing is deleted here: the staging tree was tidied when it was built, and
    # a user's directory (spaceranger_outs) is read in place. A Visium sample without its two tissue
    # PNGs is read without images (scanpy's read_visium raises on a missing one otherwise).
    images_missing = _visium_images_missing(vis)
    with _ReadVisiumCountsOnly(images_missing):
        adata, adata_norm = utils.load_adata(vis.data_folder, vis.sample_id, n_genes=n_genes, multiple_data=False)
    if images_missing:
        _drop_empty_uns_spatial(adata, adata_norm)
    # generic_counts_coords chose its matrix while staging; a Visium mode's 10x matrix is checked here.
    expression_info = vis.expression_info if vis.expression_info is not None else _counts_kind_of_loaded(adata)
    # utils.preprocess keeps a spot only when pct_counts_mt < 100 (mt_thld). A spot with no counts at
    # all has pct_counts_mt = 0/0 = NaN and one whose counts are all mitochondrial has 100, so both
    # are dropped -- upstream says nothing, and the payload used to report only the survivors.
    n_dropped_by_preprocess = None
    if n_spots_to_starfysh is not None:
        n_dropped_by_preprocess = max(int(n_spots_to_starfysh) - int(adata.n_obs), 0)
        if n_dropped_by_preprocess:
            log(
                f"[Starfysh worker] Starfysh's preprocessing dropped {n_dropped_by_preprocess} of "
                f"{n_spots_to_starfysh} spots (pct_counts_mt not < 100: zero total counts or all-mitochondrial)"
            )

    # Ensure sparse matrix is CSR (optional safety)
    try:
        import scipy.sparse as sp  # type: ignore

        if sp.issparse(adata.X) and not isinstance(adata.X, sp.csr_matrix):
            adata.X = adata.X.tocsr()
    except Exception:
        pass

    # Build signatures
    signature_note = None
    if use_provided:
        gene_sig, signature_note = _read_signature_csv(signature_csv, adata)
        signature_source = "provided"
    else:
        gene_sig = _derive_signatures_by_aa(AA, adata, adata_norm, aa_cn, aa_n_markers)
        signature_source = "archetypal_analysis"
    if signature_note:
        log(f"[Starfysh worker] NOTE: {signature_note}")

    if gene_sig is None or len(gene_sig) == 0 or gene_sig.shape[1] == 0:
        raise RuntimeError("Derived/loaded gene signatures are empty after filtering; cannot proceed.")

    # Image preprocessing
    img_metadata = utils.preprocess_img(vis.data_folder, vis.sample_id, adata_index=adata.obs_names, hchannel=hchannel)

    # PoE needs the H&E image that preprocess_img may not have found. Decide here, where both the
    # request and the image are known, rather than letting the PoE dataloader crash on img=None.
    loaded_img = img_metadata.get("img") if isinstance(img_metadata, dict) else getattr(img_metadata, "img", None)
    use_poe, poe_warning = _effective_use_poe(use_poe, loaded_img, allow_poe_fallback, input_mode)
    if poe_warning:
        log(f"[Starfysh worker] WARNING: {poe_warning}")
    n_img_chan = _prepare_poe_image(img_metadata) if use_poe else None
    if _record_spatial_without_image(adata, adata_norm, img_metadata, vis.sample_id):
        log("[Starfysh worker] no H&E image: wrote uns['spatial'] / obsm['spatial'] from the coordinates alone")
    if not use_poe:
        no_poe = "PoE image integration did not run, so the H&E image did not reach the model"
        ignored.setdefault(no_poe, []).append("patch_r")
        if hchannel:
            ignored[no_poe].append("hchannel")
        if input_mode == "generic_counts_coords" and payload.get("hires_image"):
            ignored[no_poe].append("hires_image")

    # VisiumArguments. sig_version was validated above against the spellings the library implements.
    kwargs_va = {
        "adata": adata,
        "adata_norm": adata_norm,
        "gene_sig": gene_sig,
        "img_metadata": img_metadata,
        "sample_id": vis.sample_id,
        "n_anchors": n_anchors,
        "patch_r": patch_r,
        "window_size": window_size,
    }
    if sig_version is not None:
        kwargs_va["sig_version"] = sig_version

    vis_args = utils.VisiumArguments(**kwargs_va)
    if n_img_chan is not None:
        vis_args.params["n_img_chan"] = n_img_chan
    effective_sig_version = None
    try:
        effective_sig_version = vis_args.params.get("sig_version")
    except Exception:
        effective_sig_version = sig_version

    device_t = torch.device(device)

    # Run Starfysh. Every kwarg here is one the installed run_starfysh takes; an unsupported one is an error.
    best_model, losses = _run_starfysh_compat(
        utils_module=utils,
        visium_args=vis_args,
        n_repeats=n_repeats,
        lr=lr,
        epochs=epochs,
        poe=use_poe,
        device=device_t,
        verbose=True,
    )

    # Evaluate (adds qc_m etc.)
    adata_out = vis_args.adata.copy()
    sf_model.model_eval(best_model, adata_out, vis_args, poe=use_poe, device=device_t)

    # Save outputs (each written to a .partial and renamed into place)
    out_h5ad = str(Path(output_dir) / "starfysh_annotated.h5ad")
    adata_out.write_h5ad(out_h5ad + ".partial")
    _atomic_replace(out_h5ad + ".partial", out_h5ad)

    out_props = str(Path(output_dir) / "cell_type_proportions.csv")
    column_report: dict[str, Any] = {}
    ct_names = _write_proportions_csv(adata_out, out_props, signature_source, report=column_report)

    out_sig = str(Path(output_dir) / "gene_signatures_used.csv")
    _write_csv_atomic(gene_sig, out_sig, index=False)

    out_anchors = str(Path(output_dir) / "anchor_spots.csv")
    _write_csv_atomic(vis_args.get_anchors(), out_anchors, index=False)

    out_losses = str(Path(output_dir) / "training_losses.csv")
    _write_csv_atomic(_losses_frame(losses), out_losses, index=False)

    out_model = str(Path(output_dir) / "starfysh_model_state.pt")
    torch.save(best_model.state_dict(), out_model + ".partial")
    _atomic_replace(out_model + ".partial", out_model)

    runtime = round(time.time() - t0, 3)

    model_name = "Starfysh PoE (AVAE_PoE, histology-integrated)" if use_poe else "Starfysh (AVAE)"
    method = f"{model_name}; signatures: {signature_source}"
    used_fallback = bool(poe_requested and not use_poe)

    out = WorkerOutput("starfysh", task="deconvolution")
    if signature_note:
        out.add_warning(signature_note)
    out.set_data(
        n_spots=int(adata_out.n_obs),
        n_genes=int(adata_out.n_vars),
        n_spots_input=None if n_spots_input is None else int(n_spots_input),
        n_spots_dropped_by_starfysh_preprocess=n_dropped_by_preprocess,
    )
    record_in_tissue(out, vis.n_spots_supplied, vis.n_spots_off_tissue)
    record_expression_source(out, expression_info)
    preprocess_note = ""
    if n_dropped_by_preprocess:
        preprocess_note = (
            f"{n_dropped_by_preprocess} of the {n_spots_to_starfysh} spots handed to Starfysh were dropped by its own "
            "preprocessing (utils.preprocess keeps a spot only when pct_counts_mt < 100: a spot with zero total "
            "counts has pct_counts_mt = NaN and an all-mitochondrial spot has 100); "
            f"{int(adata_out.n_obs)} spots were deconvolved."
        )
        out.add_warning(preprocess_note)
    if column_report:
        out.add_warning(
            f"cell_type_proportions.csv columns are written as arch_0..arch_{len(ct_names) - 1}: "
            f"{column_report['why']}."
            + (
                " params.proportion_columns_renamed maps each written name to the stored one."
                if "renamed" in column_report
                else ""
            )
        )
    out.add_output_files(
        {
            "starfysh_annotated_h5ad": out_h5ad,
            "cell_type_proportions_csv": out_props,
            "gene_signatures_used_csv": out_sig,
            "anchor_spots_csv": out_anchors,
            "training_losses_csv": out_losses,
            "model_state_pt": out_model,
        }
    )
    params: dict[str, Any] = {
        "sample_id": sample_id,
        "input_mode": input_mode,
        "device": device,
        "use_poe": bool(use_poe),
        "requested_use_poe": bool(poe_requested),
        "allow_poe_fallback": bool(allow_poe_fallback),
        "hchannel": bool(hchannel),
        "use_raw_counts": bool(use_raw_counts),
        "output_dir": output_dir,
        "n_genes": n_genes,
        "n_repeats": n_repeats,
        "lr": lr,
        "epochs": epochs,
        "signature_mode": signature_mode,
        "signature_source": signature_source,
        "n_signature_columns": int(gene_sig.shape[1]),
        "n_anchors": n_anchors,
        "window_size": window_size,
        "patch_r": patch_r,
        "sig_version": effective_sig_version,
    }
    if n_img_chan is not None:
        params["he_image_channels"] = n_img_chan
    if images_missing:
        params["visium_images_missing"] = list(images_missing)
    if use_provided:
        params["signature_csv"] = signature_csv
    else:
        params["aa_r"] = aa_cn
        params["aa_n_markers"] = aa_n_markers
    if column_report.get("renamed"):
        params["proportion_columns_renamed"] = column_report["renamed"]
    out.add_params(params)
    record_method(
        out,
        method=method + (" -- PoE requested, no H&E image, allow_poe_fallback=True" if used_fallback else ""),
        used_fallback=used_fallback,
        why=poe_warning or "",
    )
    if patience_given:
        record_ignored(
            out,
            ["patience"],
            "the installed utils.run_starfysh has no early stopping; every run trains for the full `epochs`",
        )
    for why, names in ignored.items():
        record_ignored(out, names, why)

    # Deconvolution-specific metrics, built from the very column names the CSV carries.
    dominant_cell_types: dict[str, int] = {}
    proportion_stats: dict[str, float] = {}
    props = np.asarray(adata_out.obsm["qc_m"])
    mean_props = props.mean(axis=0)
    dominant_idx = props.argmax(axis=1)
    for i, ct in enumerate(ct_names):
        dominant_cell_types[ct] = int((dominant_idx == i).sum())
        proportion_stats[ct] = round(float(mean_props[i]), 4)

    out.set_summary(
        cell_types=list(ct_names),
        n_cell_types=len(ct_names),
        signature_source=signature_source,
        runtime_sec=runtime,
        dominant_cell_types=dominant_cell_types,
        mean_proportions=proportion_stats,
    )
    factor_word = "cell types" if signature_source == "provided" else "archetype-derived factors"
    if column_report:
        factor_word = "factors (written as arch_<i>)"
    dropped_note = ""
    if vis.n_spots_off_tissue:
        dropped_note += (
            f" {vis.n_spots_off_tissue} of the {vis.n_spots_supplied} spots supplied have in_tissue == 0 "
            "(background) and were left out before Starfysh ran."
        )
    if preprocess_note:
        dropped_note += " " + preprocess_note
    if expression_info["expression_source"] != "X":
        dropped_note += " The counts were read from adata.raw (use_raw_counts=True)."
    elif expression_info["x_matrix_kind"] == "nonnegative_noninteger":
        dropped_note += " X is not integer counts and was normalised by Starfysh as if it were (see warnings)."
    out.set_analysis(
        f"{model_name} completed in {runtime:.1f}s (signatures: {signature_source}). "
        f"Mapped {len(ct_names)} {factor_word} across {int(adata_out.n_obs)} spots "
        f"({n_repeats} restart(s) x {epochs} epochs, lr={lr})." + dropped_note
    )
    return out.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    payload = json.loads(args.json)
    if not isinstance(payload, dict):
        raise ValueError("--json must be a JSON object")

    if payload.get("__tool__") != "starfysh_deconvolution":
        raise ValueError(f"Unsupported __tool__: {payload.get('__tool__')}")

    # stdout carries the result JSON and nothing else. skdim's FisherS (Archetypal Analysis) prints
    # "30 components are retained using conditional_number=30.00" to stdout, ahead of the JSON.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_starfysh_deconvolution(payload)
    emit_json(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[Starfysh worker] ERROR: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        WorkerOutput.emit_error("starfysh", str(e), task="deconvolution")
        sys.exit(1)
