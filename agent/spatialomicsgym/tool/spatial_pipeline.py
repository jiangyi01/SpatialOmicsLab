"""Spatial transcriptomics data diagnosis and pipeline planning.

This module acts as the intelligent entry point for spatial data analysis.
When a user provides data, the agent should call `diagnose_spatial_data` first,
which scans the input path, detects the platform/format, checks completeness,
and returns a structured plan of conversion steps needed before MCP tools can run.

Typical agent workflow:
  1. User: "Analyze my spatial data at /path/to/data"
  2. Agent calls: diagnose_spatial_data("/path/to/data")
     → returns: detected format, what's present, what's missing, conversion plan
  3. Agent calls: run_spatial_pipeline("/path/to/data", "./output.h5ad")
     → executes: auto-conversion + image embedding + validation
  4. Agent proceeds with MCP tools (clustering, SVG, deconvolution, etc.)
"""

from __future__ import annotations

import contextlib
import importlib
import json
import re
from pathlib import Path
from typing import Any

# Content-based format detection. Stdlib-only at import time (h5py is lazy inside it), so this is
# safe to import at module scope even in the minimal agent env.
from spatialomicsgym.utils.format_probe import _bundle_key, probe_format

# ---------------------------------------------------------------------------
# File signature patterns for each platform
# ---------------------------------------------------------------------------

_VISIUM_SPACERANGER_MARKERS = {
    "filtered_feature_bc_matrix.h5",
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix.h5",
}

_XENIUM_MARKERS = {
    "transcripts.csv.gz",
    "transcripts.parquet",
    "cell_feature_matrix.h5",
    "cells.csv.gz",
    "cells.parquet",
}

_XENIUM_IMAGE_MARKERS = {
    "morphology_focus.ome.tif",
    "morphology_mip.ome.tif",
    "morphology.ome.tif",
}

_MERFISH_MARKERS = {
    "cell_by_gene.csv",
    "cell_metadata.csv",
    "detected_transcripts.csv",
}

_MERFISH_IMAGE_MARKERS = {
    "mosaic_DAPI_z0.tif",
    "mosaic_PolyT_z0.tif",
    "images",
}

# Matched with `_matches_marker`, not by equality: Nanostring prepends the slide name to every file
# in an export, so `Lung5_Rep1_exprMat_file.csv` is the shape that actually ships and the bare name
# is the one that (almost) never does.
_COSMX_MARKERS = {
    "exprMat_file.csv",
    "metadata_file.csv",
    "tx_file.csv",
}

# Subdirectories a CosMx export puts its per-FOV images in. Only a flattened export has them at the
# top level, which is the only place they used to be looked for.
_COSMX_IMAGE_MARKERS = {
    "CellComposite",
    "CellLabels",
    "CompartmentLabels",
}

_STEREOSEQ_EXTENSIONS = {".gem", ".gem.gz", ".gef"}

# SAW writes a Stereo-seq chip as both a text GEM and a binary GEF, and deposited chips are often
# GEF-only. `convert_stereoseq` reads the text one and rejects the other, so the two need separate
# verdicts: saying "GEM" about a GEF promises a conversion that the converter refuses. The remedy is
# named here rather than left to the reader, because "provide a .gem" is not an instruction anyone
# can act on without knowing which tool produces one.
_STEREOSEQ_GEF_TOOL = "geftools gef2gem"
_STEREOSEQ_GEF_MISSING_STEP = f"MISSING: a text .gem -- run `{_STEREOSEQ_GEF_TOOL}` on this file to produce one"
_STEREOSEQ_GEF_SUMMARY = (
    "Stereo-seq GEF: the binary (HDF5) container, not the text GEM. Nothing here reads a GEF -- "
    f"produce the GEM with `{_STEREOSEQ_GEF_TOOL}` (SAW writes both), then convert that."
)

_SLIDESEQ_MARKERS = {
    "BeadLocationsForR.csv",
    "MappedDGEForR.csv",
    "bead_locations.csv",
}

# Resolution order for the two files `convert_slideseq` needs. The Broad's stock names come first,
# then the spelling used by deposited pucks -- `Puck_200115_08.digital_expression.txt.gz`. The
# markers carry no puck id because `_matches_marker` accepts one in front of them.
_SLIDESEQ_COUNT_MARKERS = ("MappedDGEForR.csv", "DGE.csv", "digital_expression.txt.gz", "digital_expression.txt")
_SLIDESEQ_LOCATION_MARKERS = ("BeadLocationsForR.csv", "bead_locations.csv")

_VISIUM_SPATIAL_DIR_MARKERS = {
    "tissue_positions_list.csv",
    "tissue_positions.csv",
    "scalefactors_json.json",
}

_VISIUM_IMAGE_MARKERS = {
    "tissue_hires_image.png",
    "tissue_lowres_image.png",
}

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".svs", ".ndpi"}
_OME_TIFF_EXTENSIONS = {".ome.tif", ".ome.tiff"}

#: Compression a *text* file arrives under -- the vocabulary half of the pair whose other half is
#: `file_io._TEXT_OPENERS`, which maps these same names to the decompressor that opens them. Spelled
#: here rather than imported because this module is stdlib-only at import on purpose (`file_io`
#: pulls in requests and tqdm), and pinned against the converter's tuple by a test.
_TEXT_COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz")

#: What may be under that compression. Kept in step with `spatial_data_converter.TABULAR_SUFFIXES`,
#: which is the tuple deciding whether a file reaches `_diagnose_table` at all.
_TEXT_SUFFIXES = (".csv", ".tsv", ".tab", ".txt")


def _uncompressed_name(name: str) -> str:
    """`name` lowercased, with a compression suffix removed when a delimited table is under it.

    A `.gz` on a table is an encoding, not a format: `pd.read_csv` infers the decompressor from the
    name, so `convert_merfish` reads `cell_by_gene.csv.gz` exactly as it reads `cell_by_gene.csv`
    (measured -- same 6 x 3 h5ad, `has_spatial` True). Every name test on this route compared the
    literal filename, so the gzipped half of a Vizgen export diagnosed as `unknown_tabular` and its
    directory as `unknown_directory`, summarised "Expression: NO. Coordinates: NO." over the two
    files that hold both -- and `_find_file` answered None for the sibling a plan has to name.

    Only over a text suffix. `cell_feature_matrix.h5.gz` keeps its name, because h5py does not read
    a gzipped HDF5 and matching it to the `.h5` marker would route a file to a reader that cannot
    open it. `stacker_warped.nii.gz` keeps its name for the same reason.
    """
    lower = name.lower()
    for compression in _TEXT_COMPRESSION_SUFFIXES:
        if lower.endswith(compression):
            stem = lower[: -len(compression)]
            return stem if stem.endswith(_TEXT_SUFFIXES) else lower
    return lower


def _matches_marker(name: str, marker: str) -> bool:
    """True when `name` is `marker` or a vendor-prefixed form of it.

    Nanostring's exports are named `<slide>_<marker>` -- `Lung5_Rep1_exprMat_file.csv` -- so an
    exact-name match rejects every export the instrument actually writes.

    The separator requirement is what keeps this from decaying into a substring search:
    `myexprMat_file.csv` is somebody else's file and must not pull a whole directory into a CosMx
    conversion.

    Both sides go through `_uncompressed_name`, so a gzipped export matches the plain marker and a
    gunzipped one matches a marker that carries the `.gz` (`_XENIUM_MARKERS` holds both spellings).
    """
    n, m = _uncompressed_name(name), _uncompressed_name(marker)
    return n == m or (n.endswith(m) and len(n) > len(m) and n[-len(m) - 1] in "_-.")


def _any_marker(contents: set[str], markers: set[str]) -> bool:
    """True when any directory entry matches any marker, prefix-tolerantly."""
    return any(_matches_marker(name, marker) for name in contents for marker in markers)


def _marker_names(markers: set[str]) -> set[str]:
    """A marker set spelled the way `_diagnose_directory` spells the directory's contents.

    Both sides have to be normalised, not just the files: `_XENIUM_MARKERS` carries
    `transcripts.csv.gz`, so an export somebody gunzipped is as invisible to a literal comparison as
    a gzipped Vizgen one is.
    """
    return {_uncompressed_name(m) for m in markers}


def _bundle_of(name: str, marker: str) -> str:
    """Which slide/puck `name` belongs to, asked with the table compression stripped off both sides.

    `_bundle_key` removes the marker from the end of the name and keeps the prefix. Asked with the
    literal names, `Lung5_Rep1_exprMat_file.csv.gz` does not end with `exprMat_file.csv`, so it
    falls to the empty key -- and so does every other slide in the directory, which is exactly the
    crossing `_find_marker_pair` exists to prevent.
    """
    return _bundle_key(_uncompressed_name(name), (_uncompressed_name(marker),))


def _find_marker_file(directory: Path, marker: str) -> Path | None:
    """Resolve a marker to a real file using the same rule detection matched it with.

    Detection and conversion disagreeing about which files exist is how a directory gets diagnosed
    and then refused, so both ends read this one predicate. Shortest name first, so an export that
    somehow holds both the bare and the prefixed form resolves to the bare one.
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda e: (len(e.name), e.name))
    except OSError:
        return None
    for entry in entries:
        if _matches_marker(entry.name, marker):
            return entry
    return None


def _find_marker_pair(
    directory: Path, markers_a: tuple[str, ...], markers_b: tuple[str, ...]
) -> tuple[Path | None, Path | None, bool]:
    """The two halves of one bundle, resolved together rather than one lookup each.

    Calling `_find_marker_file` twice answers two independent questions, and a directory holding
    more than one slide or puck makes the two answers disagree: `Lung5_Rep1_exprMat_file.csv` and
    `Lung5_Rep2_metadata_file.csv` were handed to `convert_cosmx` as one export. Those are the
    arguments the run actually executes with, and the two files describe different tissue -- the
    cell ids restart per slide, so the join is between two things that were never related.

    Returns ``(a, b, paired)``. ``paired`` is False only in that crossed case -- both sides matched
    and no bundle carries both halves -- so a caller need only consult it once both picks are real.
    A side that matched nothing cannot cross with anything, and the caller's existing missing-half
    branch already covers it. `_bundle_key` treats the unprefixed stock names as one bundle (the
    empty key), which keeps a plain `exprMat_file.csv`/`metadata_file.csv` export paired.
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda e: (len(e.name), e.name))
    except OSError:
        return None, None, True

    def side(markers: tuple[str, ...]) -> dict[str, Path]:
        found: dict[str, Path] = {}
        for entry in entries:
            for marker in markers:
                if _matches_marker(entry.name, marker):
                    found.setdefault(_bundle_of(entry.name, marker), entry)
                    break
        return found

    a, b = side(markers_a), side(markers_b)
    for key in a:
        if key in b:
            return a[key], b[key], True
    return (next(iter(a.values()), None), next(iter(b.values()), None), not (a and b))


def _find_marker_sibling(anchor: Path, anchor_marker: str, sibling_marker: str) -> Path | None:
    """The other half of the bundle the *named* file belongs to, or None if it is not there.

    Pointing at `Lung5_Rep2_metadata_file.csv` says which slide is meant. Resolving the partner by
    shortest name can answer with `Lung5_Rep1_exprMat_file.csv`, which is a different tissue; no
    same-slide partner is the honest answer, and the caller turns it into a refusal that says so.
    """
    key = _bundle_of(anchor.name, anchor_marker)
    try:
        entries = sorted(anchor.parent.iterdir(), key=lambda e: (len(e.name), e.name))
    except OSError:
        return None
    for entry in entries:
        if _matches_marker(entry.name, sibling_marker) and _bundle_of(entry.name, sibling_marker) == key:
            return entry
    return None


def _first_marker_file(directory: Path, markers: tuple[str, ...]) -> Path | None:
    """The first of these markers that resolves to a real file. Markers are in priority order."""
    for marker in markers:
        hit = _find_marker_file(directory, marker)
        if hit is not None:
            return hit
    return None


def _find_slideseq_files(p: Path) -> tuple[Path | None, Path | None, bool]:
    """The DGE and the bead-locations file of a Slide-seq puck, in that order, plus whether they pair.

    Deposited pucks carry the puck id in front of every file -- `Puck_200115_08_bead_locations.csv`,
    `Puck_200115_08.digital_expression.txt.gz` -- so an exact-name lookup resolves neither file in
    any published dataset. The content probe detects such a puck perfectly well, which left the run
    naming a format and then refusing to convert it.

    The plan and the run both call this, so the files the plan publishes are the files the run opens.
    """
    dge, beads, paired = _find_marker_pair(p, _SLIDESEQ_COUNT_MARKERS, _SLIDESEQ_LOCATION_MARKERS)
    return (dge or _find_file(p, ["*.tsv"])), beads, paired


def _find_cosmx_composites(p: Path, contents: set[str]) -> tuple[Path, list[str]]:
    """Locate a CosMx export's per-FOV composites, and the directory that holds them.

    A flattened export keeps `CellComposite_F001.jpg` at the top level; a stock export puts the FOV
    images in `CellComposite/`. Which directory it is belongs in the answer rather than being an
    implementation detail, because `process_cosmx_fov_images` globs inside the directory it is
    handed -- give it the export root when the images are one level down and it finds nothing.
    """
    top_level = sorted(f for f in contents if f.startswith("CellComposite_F"))
    if top_level:
        return p, top_level

    subdir_names = {m.lower() for m in _COSMX_IMAGE_MARKERS}
    for name in sorted(contents):
        if name.lower() not in subdir_names:
            continue
        sub = p / name
        try:
            found = sorted(f.name for f in sub.iterdir() if f.name.startswith("CellComposite_F"))
        except OSError:
            continue
        if found:
            return sub, found
    return p, []


def _xenium_image_rank(name: str) -> tuple[int, str]:
    """Sort key for a Xenium output's morphology images: the 2-D focus image first, the z-stack last.

    XOA v2 writes `morphology_focus/` (2-D, per stain); v1 wrote `morphology_focus.ome.tif` and
    `morphology_mip.ome.tif` beside `morphology.ome.tif`, the full 3-D z-stack -- many GB, and not
    an image a 2-D embedding wants (u22-spatial-pipeline-11).
    """
    lower = name.lower()
    for rank, stem in enumerate(("morphology_focus", "morphology_mip", "morphology.ome")):
        if lower.startswith(stem):
            return rank, lower
    return 3, lower


_MOSAIC_Z = re.compile(r"_z(\d+)\.tif$", re.IGNORECASE)


def _rank_mosaics(names: list[str]) -> list[str]:
    """A MERFISH run's mosaic TIFFs, best one to embed first: DAPI, on the middle z-plane.

    Each z-plane and each stain is its own file, and which one came first used to be string-hash
    order (u22-spatial-pipeline-11). DAPI is the nuclear stain every overlay is read against; the
    middle plane is the one most of the section is in focus on.
    """
    planes = sorted({int(m.group(1)) for n in names if (m := _MOSAIC_Z.search(n))})
    middle = planes[len(planes) // 2] if planes else 0

    def key(name: str) -> tuple[int, int, str]:
        lower = name.lower()
        stain = 0 if "dapi" in lower else 1 if "polyt" in lower else 2
        match = _MOSAIC_Z.search(name)
        return stain, abs(int(match.group(1)) - middle) if match else len(planes) + 1, lower

    return sorted(names, key=key)


_MOSAIC_STAIN = re.compile(r"mosaic_(.+?)_z\d+\.tif$", re.IGNORECASE)


def _merfish_mosaic(images: dict, p: Path) -> tuple[Path | None, str]:
    """The MERSCOPE mosaic to embed, and the stain its name says it is; ``(None, "")`` when there is none.

    The plan and the run both ask here. Mosaics at the top level are in ``images["files"]``; the
    standard MERSCOPE export keeps them in ``images/``, which detection records only as a ``path``,
    and the run read ``files`` alone -- so it answered "skipped: no image files matched" for the
    usual layout (hunt 2026-09-30, uT8-imaging-8). The stain was never passed either, so the
    metadata said DAPI whatever was embedded (uT8-imaging-7).
    """
    img_dir, names = p, list(images.get("files") or [])
    if not names and images.get("path"):
        img_dir = Path(images["path"])
        try:
            names = _rank_mosaics(
                [
                    f.name
                    for f in img_dir.iterdir()
                    if f.name.lower().startswith("mosaic_") and f.name.lower().endswith(".tif")
                ]
            )
        except OSError:
            names = []
    if not names:
        return None, ""
    match = _MOSAIC_STAIN.match(names[0])
    return img_dir / names[0], (match.group(1) if match else "unknown")


def _has_obsm_spatial(path: Path) -> bool:
    """Whether an h5ad stores obsm['spatial'], read from the HDF5 layout without loading the object."""
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            obsm = handle.get("obsm")
            return obsm is not None and "spatial" in obsm
    except Exception:
        return False


def _choose_h5ad(p: Path, names: list[str]) -> str:
    """The h5ad in a directory to diagnose: the first, in sorted order, that carries obsm['spatial'].

    A deconvolution folder routinely holds the slide and its single-cell reference side by side,
    and only one of them is a spatial dataset. When none -- or more than one -- carries coordinates,
    sorted order decides and the caller names the rest (u22-spatial-pipeline-2).
    """
    if len(names) == 1:
        return names[0]
    return next((name for name in names if _has_obsm_spatial(p / name)), names[0])


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def diagnose_spatial_data(input_path: str) -> str:
    """Scan a file or directory and diagnose spatial transcriptomics data format and completeness.

    This is the recommended first step when a user provides spatial data. It detects:
    - Platform and data format (Visium, Xenium, MERFISH, CosMx, Slide-seq, Stereo-seq, etc.)
    - Expression data presence and format
    - Spatial coordinate availability
    - Image data (H&E, fluorescence, DAPI) and format
    - What's missing for MCP tool compatibility
    - Recommended conversion pipeline steps

    Args:
        input_path: Path to a file or directory containing spatial transcriptomics data.

    Returns:
        str: JSON diagnostic report with detected format, file inventory, completeness
             assessment, and ordered pipeline steps to produce MCP-compatible h5ad.

    """
    p = Path(input_path)

    if not p.exists():
        return json.dumps(
            {
                "status": "error",
                "message": f"Path does not exist: {input_path}",
                "pipeline_steps": [],
            },
            indent=2,
        )

    # `p.exists()` can pass for an unreadable dir or a special file (FIFO/socket) — _diagnose_directory
    # then does an unguarded p.iterdir(). Keep the JSON contract instead of throwing a raw OSError.
    try:
        if p.is_file():
            return _attach_format_probe(_diagnose_file(p), p)
        if p.is_dir():
            return _attach_format_probe(_diagnose_directory(p), p)
        return json.dumps(
            {
                "status": "error",
                "message": f"Path is neither a file nor a directory: {input_path}",
                "pipeline_steps": [],
            },
            indent=2,
        )
    except OSError as e:
        return json.dumps(
            {"status": "error", "message": f"Could not read path {input_path}: {e}", "pipeline_steps": []}, indent=2
        )


def plan_spatial_pipeline(input_path: str, output_path: str = "./spatial_output.h5ad") -> str:
    """Generate an executable pipeline plan to convert spatial data to MCP-compatible h5ad.

    Runs diagnosis and returns a step-by-step plan with specific function calls,
    parameters, and expected outputs. The agent can execute these steps in order.

    Args:
        input_path: Path to spatial data (file or directory).
        output_path: Target path for the final MCP-compatible h5ad.

    Returns:
        str: JSON pipeline plan with ordered steps, each containing:
             function name, module, parameters, and description.

    """
    diagnosis = json.loads(diagnose_spatial_data(input_path))

    if diagnosis.get("status") == "error":
        return json.dumps(diagnosis, indent=2)

    steps = _build_pipeline_steps(diagnosis, input_path, output_path)

    # The diagnosis's warnings are about the very files these steps name -- "this directory holds 2
    # CosMx slides; the reader above names one of them" decides whether the plan is the right plan.
    # Only the diagnosis carried them, and the plan is the thing that gets executed.
    plan_warnings = [w for w in (diagnosis.get("warnings") or []) if w]
    produces_h5ad = any(s.get("action") in _PRODUCES_H5AD for s in steps)
    embeds_images = any(s.get("action") == "embed_images" for s in steps)

    return json.dumps(
        {
            "status": "ready",
            "input_path": input_path,
            "output_path": output_path,
            "detected_format": diagnosis.get("detected_format"),
            "platform": diagnosis.get("platform"),
            "summary": diagnosis.get("summary"),
            "warnings": plan_warnings,
            "pipeline_steps": steps,
            "estimated_outputs": {
                "h5ad_path": output_path,
                # Estimated from the plan, not from the diagnosis. All three used to answer a
                # question about the *input*: the first two were the constant True, and `has_images`
                # was `diagnosis["images"]["found"]`, i.e. "the input has an image somewhere" rather
                # than "the h5ad this plan writes will carry one". The plan for a lone histology PNG
                # -- image_only, no expression, no coordinates -- is a single `cannot_convert` step
                # that writes no h5ad at all, and all three said it would carry everything. Four more
                # branches open the image block on `found is True` and then emit no step because they
                # have no file to name. An `embed_images` step can only follow a producing one (the
                # `cannot_convert` case returns before the image block), so the two stay consistent.
                "has_expression": produces_h5ad,
                "has_spatial_coords": produces_h5ad,
                "has_images": embeds_images,
            },
        },
        indent=2,
    )


def run_spatial_pipeline(
    input_path: str,
    output_path: str = "./spatial_output.h5ad",
    skip_images: bool = False,
) -> str:
    """Execute the full spatial data pipeline: diagnose → convert → embed images → validate.

    This is the one-call solution: scans the input, converts expression data to h5ad,
    embeds available images, and validates MCP compatibility. Returns the path to a
    ready-to-use h5ad file.

    Args:
        input_path: Path to spatial data (file or directory).
        output_path: Target path for the final MCP-compatible h5ad.
        skip_images: If True, skip image embedding (faster, for expression-only analysis).

    Returns:
        str: JSON report with pipeline execution results, output path, and validation status.

    """
    import os

    os.makedirs(str(Path(output_path).parent), exist_ok=True)

    # Step 1: Diagnose
    diagnosis = json.loads(diagnose_spatial_data(input_path))
    if diagnosis.get("status") == "error":
        return json.dumps(diagnosis, indent=2)

    results: dict[str, Any] = {
        "input_path": input_path,
        "detected_format": diagnosis.get("detected_format"),
        "platform": diagnosis.get("platform"),
        "steps_executed": [],
    }

    # Which file the input is before anything is written: an already-valid h5ad named as its own
    # output is left as it is, and is still that file afterwards -- the user's own, never stamped.
    input_identity = _file_identity(input_path)

    # Step 2: Convert expression data to h5ad
    conversion_result = _execute_conversion(diagnosis, input_path, output_path)
    results["steps_executed"].append({"step": "convert_expression", "result": conversion_result})

    if conversion_result.get("status") == "error":
        results["status"] = "error"
        results["message"] = f"Conversion failed: {conversion_result.get('message', 'unknown error')}"
        return json.dumps(results, indent=2)

    h5ad_path = conversion_result.get("output_path", output_path)

    # Step 3: Embed images if available and not skipped
    if not skip_images and diagnosis.get("images", {}).get("found") is True:
        image_result = _execute_image_embedding(diagnosis, h5ad_path)
        results["steps_executed"].append({"step": "embed_images", "result": image_result})

    # The file is the input, re-encoded, and says so -- stamped here as well as by the converter
    # because two of the routes above leave it unstamped or stale: an already-valid h5ad is a byte
    # copy, and embedding images adds ``uns['spatial']`` after the converter recorded its layout.
    from spatialomicsgym.tool.conversion_record import stamp_conversion

    if input_identity is None or _file_identity(h5ad_path) != input_identity:
        stamp_conversion(h5ad_path, producer="run_spatial_pipeline", source=input_path)

    # Step 4: Validate final output
    from spatialomicsgym.tool.spatial_data_converter import validate_spatial_h5ad

    # A conversion can report success but leave a corrupt/zero-byte file; validate reads it unguarded.
    # Don't let the "one-call" pipeline throw a raw traceback — record the failure and continue.
    try:
        validation = json.loads(validate_spatial_h5ad(h5ad_path))
    except Exception as e:
        validation = {"status": "error", "message": f"validation failed: {e}"}
    results["steps_executed"].append({"step": "validate", "result": validation})

    # Build final report. A file that failed validation is a failed run, not a "warning" with an
    # output_path the agent is told to proceed from (hunt 2026-09-30, u22-spatial-pipeline-19): the
    # path is still reported, because the file is on disk and the critical issues name what to fix.
    issues = validation.get("issues") or []
    valid = validation.get("status") == "valid"
    results["status"] = "success" if valid else "error"
    results["output_path"] = h5ad_path
    results["validation"] = validation
    results["mcp_ready"] = valid
    if not valid:
        critical = [i for i in issues if "CRITICAL" in i] or [str(validation.get("message") or "validation failed")]
        results["message"] = (
            f"The converted file at {h5ad_path} is not usable by the spatial tools: {'; '.join(critical)}"
        )

    if issues:
        results["critical_issues"] = [i for i in issues if "CRITICAL" in i]

    # The diagnosis's warnings are about the file this run just read -- "extension '.csv' disagrees
    # with the content", "this directory holds 2 slides and the reader named one of them". A caller
    # of the one-call pipeline never sees the plan, so this is their only chance to be told. Two
    # things kept them out: nothing here ever read `diagnosis["warnings"]`, and `warnings` was
    # assigned only inside `if validation.get("issues")`, so a clean validation erased them.
    # Unconditional, because "no warnings" and "we did not say" are different answers.
    results["warnings"] = [w for w in (diagnosis.get("warnings") or []) if w] + [i for i in issues if "WARNING" in i]

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Internal: content-based format probe (additive cross-check on every diagnosis)
# ---------------------------------------------------------------------------

# Verdicts that mean "the name-based branches above did not work it out". Only these are eligible
# to be overridden by the probe, so a format the pipeline already identifies keeps its existing
# detected_format, platform and pipeline_steps byte-for-byte.
_UNRESOLVED_FORMATS = {"", "unknown", "unknown_tabular", "unknown_directory", "h5ad_corrupt"}

# probe format -> what to report and what to do, for the cases the probe can settle from content
# that the filename-driven branches miss.
#
# Every name under "steps" is a dotted `module.function` that must import in this environment.
# The table used to hold bare action names, and four of them -- `convert_loom_to_h5ad`,
# `convert_10x_mtx` (twice), `find_spatial_dir`, `read_zarr` -- were defined nowhere in the package.
# These names reach the agent as `pipeline_steps`, so a user handing over a `.loom`, an MTX triplet
# or a zarr store was told to call something that will never resolve: a dead end dressed as a plan,
# and worse than an admitted one because the agent spends turns on it.
#
# Half of these formats have no converter here at all. Naming the reader that does exist, plus the
# note explaining it has to be driven from `run_python_repl`, is the honest form of "unsupported";
# inventing a converter name is what produced the phantoms.
# Pinned by test_spatial_pipeline_detection_coverage.py.
_VALIDATE = "spatialomicsgym.tool.spatial_data_converter.validate_spatial_h5ad"
_CONVERTER = "spatialomicsgym.tool.spatial_data_converter"

#: Each entry also answers the two questions every report answers -- is there expression here, are
#: there coordinates -- because `_apply_probe_upgrade` rewrites the label and used to leave those
#: two keys holding the *name-based* branch's answer, or holding nothing at all. The identical bytes
#: called `filtered_feature_bc_matrix.h5` were reported as carrying expression with a note naming
#: the folder that would supply coordinates; called `counts.dat` the report said "Detected 10x
#: (counts only)" and then answered neither question.
#:
#: Only what the *format* settles is stated. A single file that is the whole dataset settles both
#: (a bare MTX has no coordinates; a PNG has no expression). A format whose halves live in separate
#: files -- Xenium, MERFISH, CosMx, Slide-seq, Space Ranger -- settles neither from its name alone,
#: so it says "unknown" and names the file that would settle it. Hardcoding `True` there is the
#: defect R61 fixed in the name-based branches, and re-adding it here would undo that. Where the
#: probe published a boolean of its own, `_apply_probe_upgrade` prefers it over these defaults.
_PROBE_UPGRADES: dict[str, dict[str, Any]] = {
    "loom": {
        "format": "loom",
        "platform": "Loom (loompy / anndata)",
        "steps": ["anndata.io.read_loom", _VALIDATE],
        "expression": {"found": True, "format": "Loom main matrix"},
        "coordinates": {"found": "unknown", "note": "Only if the writer stored (x, y) as column attributes"},
        "note": (
            "No loom converter ships here. Read it with anndata.io.read_loom in run_python_repl, "
            "put the (x, y) column attributes into obsm['spatial'], write the h5ad, then validate."
        ),
    },
    "10x_h5": {
        "format": "10x_h5_counts",
        "platform": "10x (counts only)",
        "steps": [f"{_CONVERTER}.convert_visium_h5_spatial", _VALIDATE],
        "expression": {"found": True, "format": "10x HDF5 feature-barcode matrix"},
        "coordinates": {"found": False, "note": "Need spatial/ directory for coordinates"},
        "note": (
            "Counts only. convert_visium_h5_spatial takes the Space Ranger spatial/ folder as its "
            "second argument; without tissue positions there are no coordinates to recover."
        ),
    },
    "10x_mtx": {
        "format": "10x_mtx",
        "platform": "10x (Matrix Market triplet)",
        "steps": ["scanpy.read_10x_mtx", _VALIDATE],
        "expression": {"found": True, "format": "10x Matrix Market triplet"},
        "coordinates": {"found": False, "note": "The triplet carries no coordinates"},
        "note": (
            "No MTX converter ships here, and the triplet carries no coordinates. Read it with "
            "scanpy.read_10x_mtx in run_python_repl, attach obsm['spatial'] from wherever the "
            "coordinates live, then validate."
        ),
    },
    "matrix_market": {
        "format": "matrix_market",
        "platform": "Matrix Market (MTX)",
        "steps": ["scipy.io.mmread", _VALIDATE],
        "expression": {"found": True, "format": "Matrix Market (MTX) matrix"},
        "coordinates": {"found": False, "note": "A bare matrix carries no barcodes, features or coordinates"},
        "note": (
            "A bare matrix with no barcodes, features or coordinates. Read it with scipy.io.mmread "
            "in run_python_repl, supply obs/var names and obsm['spatial'] yourself, then validate."
        ),
    },
    "visium_spaceranger": {
        "format": "visium_spaceranger",
        "platform": "10x Visium (Space Ranger)",
        "steps": [f"{_CONVERTER}.convert_visium_spaceranger", _VALIDATE],
        "expression": {"found": True, "format": "10x HDF5/MTX counts matrix"},
        "coordinates": {"found": "unknown", "note": "tissue_positions in the spatial/ directory"},
        "note": "Standard Space Ranger output; convert directly.",
    },
    "xenium_output": {
        "format": "xenium_output",
        "platform": "10x Xenium",
        "steps": [f"{_CONVERTER}.convert_xenium", _VALIDATE],
        "expression": {"found": "unknown", "note": "transcripts table or cell_feature_matrix.h5"},
        "coordinates": {"found": "unknown", "note": "x_location/y_location in transcripts, or centroids in cells"},
        "note": "Transcript-level data; the converter aggregates it to cells.",
    },
    "merfish_vizgen": {
        "format": "merfish_vizgen",
        "platform": "MERFISH/Vizgen",
        "steps": [f"{_CONVERTER}.convert_merfish", _VALIDATE],
        "expression": {"found": "unknown", "format": "cell_by_gene.csv"},
        "coordinates": {"found": "unknown", "note": "cell_metadata.csv (center_x/y) holds the only coordinates"},
        "note": "Needs both cell_by_gene.csv and cell_metadata.csv; coordinates are only in the latter.",
    },
    "cosmx_output": {
        "format": "cosmx_output",
        "platform": "Nanostring CosMx",
        "steps": [f"{_CONVERTER}.convert_cosmx", _VALIDATE],
        "expression": {"found": "unknown", "format": "exprMat_file.csv"},
        "coordinates": {
            "found": "unknown",
            "note": "metadata_file.csv (CenterX/Y_global_px) holds the only coordinates",
        },
        "note": "Needs both exprMat_file.csv and metadata_file.csv, under whatever slide prefix the export uses.",
    },
    "slideseq": {
        "format": "slideseq",
        "platform": "Slide-seq",
        "steps": [f"{_CONVERTER}.convert_slideseq", _VALIDATE],
        "expression": {"found": "unknown", "format": "DGE matrix"},
        "coordinates": {"found": "unknown", "note": "the bead-locations file holds the only coordinates"},
        "note": "Needs the DGE matrix and the bead-locations file; coordinates are only in the latter.",
    },
    "zarr": {
        "format": "zarr_store",
        "platform": "SpatialData / OME-Zarr",
        "steps": ["anndata.io.read_zarr", _VALIDATE],
        "expression": {"found": "unknown", "note": "A container, not one table; usually under tables/"},
        "coordinates": {"found": "unknown", "note": "Wherever the store's table puts them; read it to find out"},
        "note": (
            "No zarr converter ships here. A SpatialData store is a container, not one table -- "
            "the expression table is usually under tables/; read it with anndata.io.read_zarr in "
            "run_python_repl, then validate."
        ),
    },
    "parquet": {
        "format": "parquet_transcripts",
        "platform": "likely Xenium",
        "steps": [f"{_CONVERTER}.convert_xenium", _VALIDATE],
        "expression": {"found": True, "format": "transcript-level parquet"},
        "coordinates": {"found": True, "format": "embedded (x_location, y_location)"},
        "note": "Transcript-level parquet; the converter aggregates it to cells.",
    },
    "r_object": {
        "format": "r_object",
        "platform": "R (Seurat / SCE / matrix)",
        "steps": [f"{_CONVERTER}.convert_r_object", _VALIDATE],
        "expression": {"found": True, "format": "R object (Seurat / SCE / matrix)"},
        "coordinates": {"found": "unknown", "note": "Spatial coords extracted if Seurat spatial object"},
        "note": "Converted through an R subprocess, so this route needs Rscript on PATH.",
    },
    "ome_tiff": {
        "format": "image_only",
        "platform": "unknown (image only)",
        "steps": ["spatialomicsgym.tool.spatial_image_processor.embed_image_in_h5ad"],
        "expression": {"found": False, "note": "An image carries no expression matrix"},
        "coordinates": {"found": False, "note": "An image carries no spot or cell coordinates"},
        "images": {"found": True, "format": "image (identified from content, not from the filename)"},
        "note": (
            "An image, not expression data: there is nothing here to convert. embed_image_in_h5ad "
            "attaches it to an h5ad that already carries obsm['spatial']."
        ),
    },
}
# The three image verdicts differ only in how the bytes were recognised, so they share one entry.
_PROBE_UPGRADES["tiff"] = _PROBE_UPGRADES["ome_tiff"]
_PROBE_UPGRADES["image"] = _PROBE_UPGRADES["ome_tiff"]


def _attach_format_probe(report_json: str, p: Path) -> str:
    """Add the content-based probe's verdict to a diagnosis, and upgrade it if it was unresolved.

    Purely additive. The name-based branches run first and keep their answer whenever they produced
    one; the probe only fills in the cases they gave up on -- an h5ad delivered as `counts.dat`, a
    10x MTX triplet directory, a loom mis-named `.h5ad` -- and otherwise just records what the bytes
    said so a failed read reports evidence instead of a bare exception string.

    The probe is a diagnostic aid bolted onto the main ingest path, so any failure inside it is
    swallowed: a helper that can take down `diagnose_spatial_data` is a net loss however good its
    verdicts are.
    """
    try:
        result = json.loads(report_json)
    except ValueError:
        return report_json

    try:
        verdict = probe_format(p)
    except Exception as exc:
        result["format_probe"] = {"status": "probe_failed", "message": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, indent=2)

    try:
        probe = verdict.to_dict()
        result["format_probe"] = probe
        current = str(result.get("detected_format") or "")

        upgraded = current in _UNRESOLVED_FORMATS and verdict.is_known and _apply_probe_upgrade(result, verdict, p)
        if current in _UNRESOLVED_FORMATS and not upgraded:
            # Still unresolved -- either the probe could not name it, or it named something the
            # upgrade table does not carry. Both need the same thing from here: the report says what
            # was actually seen, which is the part the agent can act on. "Unrecognized file format:
            # mystery.bin" is not.
            evidence = "; ".join(probe.get("evidence", [])[:4])
            if evidence:
                result["summary"] = f"{result.get('summary', 'Format not recognised')} Probe evidence: {evidence}"
            # ...and evidence on its own was still a dead end, because `pipeline_steps` was `[]`
            # here (or `["manual_inspection_needed"]`, which names no function). See
            # :func:`_unknown_format_recovery`.
            with contextlib.suppress(Exception):
                # `is_known` is the probe's own answer to "did I commit to a format", and the route
                # branches on it. Recomputing that rule here would be a second place to keep it.
                result["unknown_format_recovery"] = _unknown_format_recovery(probe, p, identified=verdict.is_known)
                result["pipeline_steps"] = [s["action"] for s in result["unknown_format_recovery"]["steps"]]
        if probe.get("warnings"):
            result.setdefault("warnings", []).extend(probe["warnings"])
    except Exception as exc:
        result["format_probe"] = {"status": "probe_failed", "message": f"{type(exc).__name__}: {exc}"}

    return json.dumps(result, indent=2)


def _reachable_tool(*candidates: str) -> str | None:
    """The first dotted ``module.function`` that actually imports here, or ``None``.

    A recovery route that names a tool this process cannot call is the dead end it replaced, with
    more words. The web-search tools live in :mod:`spatialomicsgym.tool.literature`, which needs
    ``googlesearch``, ``bs4``, ``PyPDF2``, ``arxiv`` or ``scholarly`` depending on which one you
    call -- none of them present in the minimal agent-core env -- so whether the agent has a search
    tool is a property of the installation and has to be asked at the moment the route is written,
    not assumed.

    ``hasattr`` is not that question, and used to be asked here anyway. Those packages were once
    imported at module scope, where an absent one really did remove the attribute; they have since
    moved into the function bodies (so that one missing package stops removing all eight tools), and
    from that day ``hasattr`` answered ``True`` for every one of them. This helper was handing its
    caller ``literature.search_google`` on a box with no ``googlesearch`` -- writing the dead end its
    own docstring exists to prevent. It now asks the module that defers the import which import it
    defers.

    ``None`` is deliberate rather than a substitute tool: a step that says "search for this, and I
    have nothing here that searches" is honest and still actionable by an agent with its own
    browsing. A step naming a function that raises ``ModuleNotFoundError`` is neither.
    """
    for dotted in candidates:
        module_name, _, func_name = dotted.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except Exception:  # any import-time failure means the tool is not callable here
            continue
        if not hasattr(module, func_name):
            continue
        # The attribute exists; that is where the old check stopped and where the bug lived. A
        # module that defers its optional imports is the only thing that knows what it deferred.
        requirement = getattr(module, "requirement_for", None)
        if callable(requirement):
            try:
                if requirement(func_name) is not None:
                    continue  # importable name, uninstallable call
            except Exception:  # a module that cannot answer is not evidence against its own tool
                pass
        return dotted
    return None


#: Containers that wrap a payload rather than being one. The route for these is not "identify the
#: format" -- the format is the wrapper, and what matters is what falls out of it. ``binary``,
#: ``hdf5``, ``parquet`` and ``tiff`` also appear in ``container`` but describe the file's own
#: encoding, so they are deliberately absent here.
_WRAPPING_CONTAINERS = {
    "gzip": "<gzip.open(path, 'rb') -> write the bytes to a sibling path without the .gz suffix>",
    "zip": "<zipfile.ZipFile(path).extractall(dest) into a new directory beside the archive>",
    "tar": "<tarfile.open(path).extractall(dest) into a new directory beside the archive>",
}

#: Formats that cannot hold a count matrix and coordinates on their own. Reached only because
#: someone pointed the diagnosis at one; the honest reply is to say what it is and ask for the data,
#: not to hand back a route ending in "assemble an AnnData" from a PDF.
_NOT_A_MATRIX = {
    "pdf": "a document",
    "xml": "an XML document",
    "json": "a JSON document",
    "fasta": "a sequence file",
    "plain_text": "free text with no consistent delimiter",
    "ole2_compound": "a legacy OLE2 compound document",
}


def _unknown_format_recovery(probe: dict[str, Any], p: Path, *, identified: bool = False) -> dict[str, Any]:
    """An ordered route out of "the pipeline has no plan for this", using tools the agent already has.

    The probe knows every format it was taught and nothing else, and "point the co-scientist at my
    data" is how the system is used, so an unteachable format is a normal event rather than an edge
    case. What made it fatal was the shape of the answer: `pipeline_steps` came back `[]` for a file
    and `["manual_inspection_needed"]` -- a name no dispatcher implements -- for a directory. The
    ReAct loop needs an action, so given a verdict and no action it stops or invents a reader.

    Everything needed to break the deadlock was already present and merely unconnected: the probe
    extracts the distinguishing bytes, and the agent has `search_google` /
    `advanced_web_search_claude` to identify a signature, `extract_url_content` to read the spec it
    finds, and `run_python_repl` to try the reader that spec names.

    `identified` says whether the probe committed to a format. It has to change the route, because
    four situations arrive here and one set of steps cannot be true of all four:

    * a **wrapper** -- `.h5ad.gz`, a zipped Space Ranger run -- where the format is the wrapper and
      the only useful move is to unpack and diagnose again, not to research gzip;
    * a **named format with a reader** -- `.npy`, SQLite, a delimited text file -- where the probe
      already answered the question step 1 would have gone searching for;
    * a **named format that is not a dataset** -- a PDF, an XML sidecar -- where every conversion
      step is nonsense and the real answer is "this is not the data";
    * a genuinely **unidentified** file, which is the one the original route was written for.

    Telling the first three to "identify what wrote this file by searching for '%PDF-1.5'" is worse
    than saying nothing: it is a confident instruction contradicted by evidence in the same report.

    The dataset-bearing routes still end where every other format's does -- a validated `.h5ad` --
    because identifying a format is not the deliverable; the 88 downstream tools read one thing.

    Advice only. Nothing here executes, and `_attach_format_probe` suppresses any failure: a helper
    that can take down `diagnose_spatial_data` is a net loss however good its advice is.
    """
    details = probe.get("details") or {}
    token = str(details.get("magic_ascii") or "")
    search_terms = [str(t) for t in (details.get("search_terms") or []) if t]
    fmt = str(probe.get("format") or "")
    container = str(probe.get("container") or "")
    reader = str(probe.get("reader") or "")
    evidence = [str(e) for e in (probe.get("evidence") or []) if e]
    seen = evidence[0] if evidence else "no distinguishing signature"

    convert_tail = [
        {
            "action": "convert_to_h5ad",
            "tool": "spatialomicsgym.tool.support_tools.run_python_repl",
            "params": {"command": "<build an AnnData: X counts, obs ids, obsm['spatial'] as (x, y); write .h5ad>"},
            "description": (
                "Assemble an AnnData and write it out. `obsm['spatial']` must hold (x, y) per "
                "observation in the same row order as X -- getting that order or that orientation "
                "wrong is silent, and every downstream tool inherits it."
            ),
        },
        {
            "action": "validate_spatial_h5ad",
            "tool": "spatialomicsgym.tool.spatial_data_converter.validate_spatial_h5ad",
            "params": {"h5ad_path": "<the .h5ad written in the previous step>"},
            "description": "Confirm the result is readable by the downstream tools before analysing it.",
        },
    ]

    if identified and container in _WRAPPING_CONTAINERS:
        members = [str(m) for m in (details.get("members") or []) if m]
        if members:
            # A tar is walked only as far as the probe's byte sample reaches, so the list can be a
            # prefix. `n_members` is present only when the walk saw the archive's end marker;
            # falling back to `len(members)` would print a partial read as a total.
            total = details.get("n_members")
            noun = "member" if (total or len(members)) == 1 else "members"
            counted = f"{total} {noun}" if total else f"at least {len(members)} {noun}"
            inside = f"{counted}, first: {', '.join(members[:3])}"
        else:
            inside = f"contents the probe read as {fmt!r}"
        why = (
            f"{p.name!r} is a {container} container holding {inside}. The pipeline has no plan for "
            "it because nothing downstream reads through a wrapper -- unpack it and the contents "
            "diagnose normally."
        )
        steps = [
            {
                "action": "unpack_the_container",
                "tool": "spatialomicsgym.tool.support_tools.run_python_repl",
                "params": {"command": _WRAPPING_CONTAINERS[container]},
                "description": (
                    f"Unpack the {container} wrapper. {seen}. "
                    + (f"The probe suggests: {reader}. " if reader else "")
                    + "Unpack beside the original rather than over it -- the archive is the only copy."
                ),
            },
            {
                "action": "diagnose_spatial_data",
                "tool": "spatialomicsgym.tool.spatial_pipeline.diagnose_spatial_data",
                "params": {"input_path": "<the unpacked file or directory>"},
                "description": (
                    "Diagnose what came out and follow the plan it returns. The wrapper was the only "
                    "thing hiding the format, so this normally resolves to a real platform with a "
                    "converter. If the unpacked path is itself unresolved, its report carries this "
                    "same recovery block for the contents."
                ),
            },
        ]
    elif identified and fmt in _NOT_A_MATRIX:
        why = (
            f"{p.name!r} is {_NOT_A_MATRIX[fmt]} ({seen}), not a spatial dataset. There is no plan "
            "because there is nothing here to convert -- this is a wrong path, not a missing reader."
        )
        steps = [
            {
                "action": "identify_the_real_data_path",
                "tool": "spatialomicsgym.tool.spatial_pipeline.diagnose_spatial_data",
                "params": {"input_path": "<the file or directory holding the counts and coordinates>"},
                "description": (
                    f"Point the diagnosis at the data itself. {p.name!r} is {_NOT_A_MATRIX[fmt]}; a "
                    "spatial dataset is a count matrix plus one coordinate per observation, which "
                    "this file does not carry. If it is a sidecar, name the file holding the counts; "
                    "if it really is the matrix, say what its layout is -- no reader here infers one."
                ),
            },
        ]
    elif identified and reader:
        why = (
            f"The probe identified {p.name!r} as {fmt} ({seen}) and names a reader for it, but no "
            "converter in this pipeline turns that into a spatial h5ad. Read it directly, then "
            "assemble the h5ad by hand."
        )
        steps = [
            {
                "action": "trial_read_a_slice",
                "tool": "spatialomicsgym.tool.support_tools.run_python_repl",
                "params": {"command": f"<{reader}; print shape and dtypes>"},
                "description": (
                    f"Open the file with `{reader}` -- the probe already identified the format, so "
                    "there is nothing to look up. Print the shape, the first few row and column "
                    "labels, and any coordinate columns. Confirm the values vary: a correct shape "
                    "and plausible-looking names prove nothing on their own."
                ),
            },
            *convert_tail,
        ]
    else:
        # Nothing named, or named with no reader: the format still has to be researched. The one
        # thing that changes is what to search for -- a name when there is one, the magic bytes
        # otherwise. Searching for a filename is what the extension-based guess already failed at.
        search_tool = _reachable_tool(
            "spatialomicsgym.tool.literature.search_google",
            "spatialomicsgym.tool.literature.advanced_web_search_claude",
        )
        fetch_tool = _reachable_tool("spatialomicsgym.tool.literature.extract_url_content")
        unavailable = " No web-search tool is importable in this environment -- use your own browsing."

        if identified:
            why = (
                f"The probe identified {p.name!r} as {fmt} ({seen}), but it knows no reader for that "
                "format and no converter here handles it. Find the reader, then convert."
            )
            subject = f"the {fmt} format"
            query = f"{fmt} file format python reader spatial transcriptomics"
        else:
            why = (
                f"No signature or directory layout known to the probe matched {p.name!r}. This is a "
                "gap in what the probe was taught, not a corrupt file -- identify it, then convert."
            )
            subject = f"the {token!r} signature" if token else f"{p.name!r}"
            query = search_terms[0] if search_terms else f"{p.suffix or p.name} file format how to read"

        steps = [
            {
                "action": "identify_format_by_search",
                "tool": search_tool,
                "params": {"query": query},
                "description": (
                    f"Identify what reads this file by searching for {subject}. The probe's evidence "
                    "is the query material -- do not guess from the extension, which already failed."
                    + ("" if search_tool else unavailable)
                ),
            },
            {
                "action": "read_the_format_specification",
                "tool": fetch_tool,
                "params": {"url": "<the specification or library page found in the previous step>"},
                "description": (
                    "Read the spec or library docs far enough to learn two things: which Python "
                    "package reads it, and where that package puts the count matrix, the observation "
                    "ids and the spatial coordinates." + ("" if fetch_tool else unavailable)
                ),
            },
            {
                "action": "trial_read_a_slice",
                "tool": "spatialomicsgym.tool.support_tools.run_python_repl",
                "params": {"command": "<import the reader you found and open the file; print shape and dtypes>"},
                "description": (
                    "Open the file with that reader and print the shape, the first few row and column "
                    "labels, and the coordinate columns. Confirm the values vary -- a correct shape "
                    "and plausible-looking names prove nothing on their own."
                ),
            },
            *convert_tail,
        ]

    return {
        "why_unresolved": why,
        "distinguishing_evidence": evidence[:6],
        "magic_ascii": token,
        "search_terms": search_terms,
        "steps": [{"step": i, **step} for i, step in enumerate(steps, start=1)],
    }


#: Booleans the detectors publish on the verdict for the two questions the report answers, most
#: specific first. These are findings about the files that are actually there, so they outrank the
#: format-level defaults in `_PROBE_UPGRADES`: a Space Ranger directory is named `visium_spaceranger`
#: whether or not `spatial/` holds a tissue_positions file, and the probe says which.
_VERDICT_DETAIL_FOR = {
    "expression": ("has_expression",),
    "coordinates": ("has_tissue_positions", "has_coordinates"),
}


def _upgraded_data_fields(mapped: dict[str, Any], verdict: Any) -> dict[str, dict[str, Any]]:
    """The expression/coordinates/images entries for a report the content probe has relabelled."""
    fields = {key: dict(mapped[key]) for key in ("expression", "coordinates", "images") if key in mapped}
    details = getattr(verdict, "details", None) or {}
    for key, detail_keys in _VERDICT_DETAIL_FOR.items():
        if key not in fields:
            continue
        found = next((details[d] for d in detail_keys if isinstance(details.get(d), bool)), None)
        if found is not None:
            fields[key]["found"] = found
    return fields


def _apply_probe_upgrade(result: dict[str, Any], verdict: Any, p: Path) -> bool:
    """Replace an unresolved verdict with what the bytes say. False when the table has no entry.

    The return value is what lets the caller tell "upgraded" from "recognised, but this table does
    not carry it". Returning nothing made those the same, and the second one then fell out of both
    branches: a `.h5ad.gz` was reported as `unknown` with no steps and no recovery route, while an
    unrecognisable file got an ordered one. Knowing more about a file cannot leave the caller with
    less to do.
    """
    fmt = verdict.format

    if fmt == "h5ad":
        # Re-run the real h5ad diagnosis so the caller gets obsm/coords/QC and proper steps,
        # not just a corrected label.
        result["detected_format"] = "h5ad"
        result["platform"] = "pre-converted"
        result.update(_diagnose_h5ad(p))
        result["format_probe_upgraded"] = (
            "identified as h5ad from its HDF5 signature and AnnData root layout, not from its filename"
        )
        return True

    mapped = _PROBE_UPGRADES.get(fmt)
    if mapped is None:
        return False
    platform = mapped["platform"]
    result["detected_format"] = mapped["format"]
    result["platform"] = platform
    result["pipeline_steps"] = list(mapped["steps"])
    result.update(_upgraded_data_fields(mapped, verdict))
    result["format_probe_upgraded"] = f"identified as {fmt} from content: {'; '.join(verdict.evidence[:2])}"
    if verdict.reader:
        result["suggested_reader"] = verdict.reader
    # The note carries the part the step names cannot: which of these formats has no converter here,
    # and what has to be supplied by hand before the result is a spatial h5ad.
    result["summary"] = (
        f"Detected {platform} by content probe (confidence {verdict.confidence:.0%}). "
        f"{'; '.join(verdict.evidence[:2])}. {mapped['note']}"
    )
    return True


# ---------------------------------------------------------------------------
# Internal: File diagnosis
# ---------------------------------------------------------------------------


def _diagnose_file(p: Path) -> str:
    """Diagnose a single file."""
    suffix = p.suffix.lower()
    name = p.name.lower()

    result: dict[str, Any] = {
        "status": "diagnosed",
        "input_type": "file",
        "input_path": str(p),
        "file_name": p.name,
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 2),
    }

    # h5ad file
    if suffix == ".h5ad":
        result["detected_format"] = "h5ad"
        result["platform"] = "pre-converted"
        h5ad_diag = _diagnose_h5ad(p)
        result.update(h5ad_diag)
        return json.dumps(result, indent=2)

    # R objects (.rds, .rda, .RData)
    if suffix in (".rds", ".rda", ".rdata"):
        result["detected_format"] = "r_object"
        result["platform"] = "R (Seurat / SCE / matrix)"
        result["expression"] = {"found": True, "format": f"R object ({suffix})"}
        result["coordinates"] = {"found": "unknown", "note": "Spatial coords extracted if Seurat spatial object"}
        result["images"] = {"found": "unknown", "note": "Visium images extracted if present in Seurat object"}
        result["summary"] = (
            f"R object file ({suffix}). Will extract counts, metadata, spatial coordinates, "
            "reductions, and images via R subprocess. Supports Seurat v3/v4/v5, dgCMatrix, data.frame."
        )
        result["pipeline_steps"] = ["convert_r_object", "validate_spatial_h5ad"]
        return json.dumps(result, indent=2)

    # Stereo-seq GEF: the binary half of the format, and not the half that converts here. Calling it
    # a GEM promised `convert_stereoseq`, which refuses a `.gef` outright -- so the run named a
    # format confidently and then died on it, without ever saying how to get a file it can read.
    if suffix == ".gef":
        result["detected_format"] = "stereoseq_gef"
        result["platform"] = "Stereo-seq"
        result["expression"] = {"found": True, "format": "GEF binary container (HDF5)"}
        result["coordinates"] = {"found": True, "format": "embedded in the GEF, in chip coordinates"}
        result["images"] = {"found": False, "note": "Stereo-seq ssDNA images are not bundled in a GEF"}
        result["summary"] = _STEREOSEQ_GEF_SUMMARY
        result["pipeline_steps"] = [_STEREOSEQ_GEF_MISSING_STEP, "convert_stereoseq", "validate"]
        return json.dumps(result, indent=2)

    # Stereo-seq GEM: the text table, which is the one that converts.
    if suffix == ".gem" or str(p).endswith(".gem.gz"):
        result["detected_format"] = "stereoseq_gem"
        result["platform"] = "Stereo-seq"
        result["expression"] = {"found": True, "format": "GEM transcript table"}
        result["coordinates"] = {"found": True, "format": "embedded in GEM (x, y columns)"}
        result["images"] = {"found": False, "note": "Stereo-seq ssDNA images not bundled in GEM file"}
        result["summary"] = "Stereo-seq GEM file with transcript coordinates. Needs binning to create spot-level h5ad."
        result["pipeline_steps"] = ["convert_stereoseq", "validate"]
        return json.dumps(result, indent=2)

    # 10x H5 counts file. A Xenium output's `cell_feature_matrix.h5` matches this name test too, and
    # was told to go and find a Space Ranger spatial/ folder that a Xenium run cannot have (hunt
    # 2026-09-30, u22-spatial-pipeline-14). Beside a Xenium output's own files it is diagnosed as
    # part of that output, which names what converts and what is missing.
    if suffix == ".h5" and ("filtered" in name or "raw" in name or "feature" in name):
        rerooted = _diagnose_as_part_of_a_xenium_output(p)
        if rerooted is not None:
            return rerooted
        result["detected_format"] = "10x_h5_counts"
        result["platform"] = "10x (counts only)"
        result["expression"] = {"found": True, "format": "10x HDF5 feature-barcode matrix"}
        # Look for the sibling spatial/ folder instead of asserting it is absent. The shipped
        # data/V1_Human_Lymph_Node has filtered_feature_bc_matrix.h5 sitting next to exactly that
        # folder, and convert_visium_h5_spatial -- the step this same report goes on to recommend --
        # reads tissue_positions out of it and succeeds. Reporting "found: False" and a summary
        # reading "without spatial directory" was a claim about the filesystem made without
        # consulting it, handed to the agent as the basis for choosing what to run next.
        spatial_dir = p.parent / "spatial"
        positions = None
        if spatial_dir.is_dir():
            positions = next(
                (f.name for f in sorted(spatial_dir.iterdir()) if f.name.startswith("tissue_positions")), None
            )
        if positions:
            result["coordinates"] = {
                "found": True,
                "format": f"10x spatial/ directory ({positions})",
                "path": str(spatial_dir),
            }
            result["summary"] = f"10x counts H5 file with a sibling spatial/ directory ({positions}). Ready to convert."
        elif spatial_dir.is_dir():
            result["coordinates"] = {
                "found": False,
                "note": f"spatial/ directory present at {spatial_dir} but holds no tissue_positions file",
            }
            result["summary"] = (
                "10x counts H5 file with a spatial/ directory that has no tissue_positions file. "
                "Coordinates cannot be read until one is present."
            )
        else:
            result["coordinates"] = {"found": False, "note": "Need spatial/ directory for coordinates"}
            result["summary"] = (
                "10x counts H5 file without spatial directory. Need spatial/ folder with tissue_positions."
            )
        result["images"] = {"found": False}
        # `find_spatial_dir` used to lead this list and is defined nowhere in the package -- the same
        # phantom the probe-upgrade table carried. The conversion step looks for the sibling spatial/
        # folder itself, so there is no separate call to name.
        result["pipeline_steps"] = ["convert_visium_h5_spatial", "validate"]
        return json.dumps(result, indent=2)

    # CSV/TSV - try to detect. The suffix list is the converter's, so what we agree to diagnose as a
    # table is exactly what it can read: `Path("dge.txt.gz").suffix` is ".gz", so the `suffix in
    # (...)` test this replaces missed every compressed table except the one `.csv.gz` bolted on
    # beside it, and sent `.tsv.gz` / `.txt.gz` -- the ordinary GEO and Slide-seq DGE forms -- down
    # the container branch to be "unpacked", though pandas reads them where they lie.
    from spatialomicsgym.tool.spatial_data_converter import TABULAR_SUFFIXES

    if str(p).lower().endswith(TABULAR_SUFFIXES):
        return _diagnose_tabular_file(p, result)

    # Parquet (likely Xenium transcripts)
    if suffix == ".parquet":
        result["detected_format"] = "parquet_transcripts"
        result["platform"] = "likely Xenium"
        result["expression"] = {"found": True, "format": "transcript-level parquet"}
        result["coordinates"] = {"found": True, "format": "embedded (x_location, y_location)"}
        result["images"] = {"found": False, "note": "Look for morphology_focus.ome.tif alongside"}
        result["summary"] = "Parquet transcript file (likely Xenium). Needs cell aggregation to create h5ad."
        result["pipeline_steps"] = ["convert_xenium", "check_for_images", "validate"]
        return json.dumps(result, indent=2)

    # Image files
    if suffix in _IMAGE_EXTENSIONS or any(str(p).endswith(ext) for ext in _OME_TIFF_EXTENSIONS):
        result["detected_format"] = "image_only"
        result["platform"] = "unknown (image only)"
        result["expression"] = {"found": False}
        result["coordinates"] = {"found": False}
        result["images"] = {"found": True, "format": suffix, "path": str(p)}
        result["summary"] = (
            "Image file without expression data. Need an h5ad with spatial coordinates to embed this image."
        )
        result["pipeline_steps"] = ["need_expression_data"]
        return json.dumps(result, indent=2)

    result["detected_format"] = "unknown"
    result["platform"] = "unknown"
    result["summary"] = f"Unrecognized file format: {p.name}"
    result["pipeline_steps"] = []
    return json.dumps(result, indent=2)


def _empty_observations(p: Path, adata: Any, n_obs: int) -> dict[str, Any]:
    """How many observations hold no counts at all, as ``qc_metrics`` keys, or ``{}`` if unknowable.

    Read from the matrix where that is cheap -- a CSR row with no stored entry is empty, and
    ``indptr`` says so without reading one value -- else from ``obs['total_counts']``. The merged
    Curio ovary file SpatialBench's ovary evals use holds 198,094 such beads of 607,096 (32.6%); a
    scored run found that out only after Harmony failed on them, several steps in, and the scanpy
    clustering tool run over all of them does not finish inside the 1800 s budget (Leiden alone took
    20 minutes, measured 2026-09-25). Stated, never acted on: whether they belong in the analysis is
    the task's call.

    "At all" is the claim, and X can be a gene subset (HVGs, a panel) in which a bead with reads in
    other genes has no entry, so a row X calls empty must survive two checks. Where the file has a
    raw, ``raw/X`` must have no entry for it either; a raw that is not CSR cannot be checked from
    ``indptr``, so X does not speak for it and the count comes from the totals or not at all. Where
    ``obs['total_counts']`` exists, a row whose total is above 0 is not empty, raw or no raw. A NaN
    total is unknown, not a count, and vetoes nothing: ``ad.concat(join="outer")`` leaves one on
    every bead of a batch that had no totals, and a veto on "not 0" dropped those beads from the
    count. Reproduced before the checks: a 400-bead file with X cut to 50 of its 300 genes, every
    bead holding counts in ``.raw`` and a ``total_counts`` of at least 1, was reported as 219 beads
    (54.8%) that "hold no counts at all".

    The checks only veto. The raw never starts a count X could not give, and the source names a
    check only when it removed a row, so a file the checks agree on is diagnosed in the same bytes
    as before they existed -- this text goes verbatim into the scored prompt. Measured 2026-09-25 on
    the 10 cached SpatialBench eval files: every one is byte-identical. The two kidney files hold a
    CSC X over a CSR raw and still say nothing; a first version of the checks, which read the raw
    whenever there was one, gave them a count, moved the source of two more files to ``raw.X`` and
    added the totals clause to both ovary files, for a finding that concerns none of them.
    """
    import h5py
    import numpy as np

    def empty_rows(matrix: Any) -> Any:
        """Which rows of a CSR group have no stored entry; None for any other encoding or shape."""
        if not isinstance(matrix, h5py.Group) or matrix.attrs.get("encoding-type") != "csr_matrix":
            return None
        indptr = np.asarray(matrix["indptr"][:]) if "indptr" in matrix else np.zeros(0)
        return np.diff(indptr) == 0 if len(indptr) == n_obs + 1 else None

    with h5py.File(str(p), "r") as handle:
        empty, source = empty_rows(handle.get("X")), "X (rows with no stored entry)"
        raw = handle.get("raw")
        if empty is not None and (raw is not None or "raw.X" in handle):
            # A pre-0.7 dotted raw.X is never CSR-encoded, so it is never read and X never speaks for it.
            in_raw = empty_rows(raw.get("X") if isinstance(raw, h5py.Group) else None)
            if in_raw is None:
                empty = None
            elif (empty & ~in_raw).any():
                empty, source = empty & in_raw, "X and raw.X (rows with no stored entry in either)"
    if "total_counts" in adata.obs.columns:
        totals = np.asarray(adata.obs["total_counts"], dtype=float)
        counted = totals > 0  # a NaN total is unknown: not a count, so it vetoes nothing
        if empty is None:
            empty, source = totals == 0, "obs['total_counts']"
        elif (empty & counted).any():
            empty, source = empty & ~counted, f"{source}, except where obs['total_counts'] is above 0"
    if empty is None:
        return {}
    n_empty = int(empty.sum())
    out: dict[str, Any] = {"empty_observations": n_empty, "empty_observations_source": source}
    if n_empty:
        out["note"] = (
            f"{n_empty:,} of {n_obs:,} observations ({n_empty / max(n_obs, 1):.1%}) hold no counts at all. "
            "They carry no expression, so every expression-based step -- normalisation, gene selection, "
            "PCA, the neighbour graph, clustering -- sees them as one identical point."
        )
    return out


#: obs column names the diagnosis reads as naming a section, in the order they are tried. ``batch``
#: is last because it is the weakest: it names a section only where a section was the unit of
#: acquisition. ``ident`` and ``sample`` are absent on purpose -- Seurat's identity class is usually
#: a clustering, and a sample is as often a donor or a condition as a section -- so an object keyed
#: on either is reported as unrecognised, which is true, rather than as a stack, which may not be.
_SECTION_OBS_NAMES = (
    "slice_id",
    "slice",
    "section",
    "section_id",
    "section_label",
    "tissue_section",
    "sample_section",
    "library_id",
    "brain_section_label",
    "Bregma",
    "batch",
)


#: The names of :data:`_SECTION_OBS_NAMES` also matched in any case: the four added on 2026-09-25.
#: The seven read before that are still matched as spelled. Recognition was widened by the names
#: added and no further, and folding case on ``batch``, the weakest, would widen it where it is
#: least sure.
_SECTION_OBS_NAMES_ANY_CASE = ("slice", "section_label", "tissue_section", "sample_section")


def _section_columns(columns: Any) -> list[Any]:
    """The obs columns :data:`_SECTION_OBS_NAMES` recognises, in its order; exact spelling first."""
    by_fold: dict[str, list[Any]] = {}
    for column in columns:
        by_fold.setdefault(str(column).casefold(), []).append(column)
    found: list[Any] = []
    for name in _SECTION_OBS_NAMES:
        matches = by_fold.get(name.casefold(), [])
        if name not in _SECTION_OBS_NAMES_ANY_CASE:
            matches = [column for column in matches if str(column) == name]
        found += sorted(matches, key=lambda column: str(column) != name)
    return found


def _diagnose_h5ad(p: Path) -> dict[str, Any]:
    """Diagnose an existing h5ad file."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(str(p), backed="r")
    except Exception as e:
        return {
            "detected_format": "h5ad_corrupt",
            "summary": f"Failed to read h5ad: {e}",
            "pipeline_steps": [],
        }

    try:
        info: dict[str, Any] = {
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
        }

        # Check expression
        has_x = adata.X is not None
        info["expression"] = {"found": has_x, "format": "AnnData .X matrix"}

        # Check spatial coordinates
        has_spatial = "spatial" in adata.obsm
        coord_cols = None
        if not has_spatial:
            # Ask the converter which columns it can read, rather than keeping a second list here.
            # The hand-written one this replaces held four pairs against the converter's eight, and
            # every pair it was missing was a platform's own: Xenium's x_location/y_location,
            # Visium's pxl_col_in_fullres/pxl_row_in_fullres, CosMx's CenterX_global_px/
            # CenterY_global_px, x_bin/y_bin. So diagnosis reported "no coordinates" about files the
            # very next step reads coordinates out of -- and diagnosis is what the agent consults
            # before it picks a tool. The exclusion of array_col/array_row lives in that same
            # function, so sharing it keeps the refusal these two lists already agreed on.
            from spatialomicsgym.tool.spatial_data_converter import _detect_coordinate_columns

            coord_cols = _detect_coordinate_columns(adata.obs)

        info["coordinates"] = {
            "found": has_spatial or coord_cols is not None,
            "in_obsm": has_spatial,
            "in_obs_columns": coord_cols,
        }

        # What obsm['spatial'] holds, not only that it exists. A 1-column array or NaN rows came
        # back "MCP-compatible. Ready for analysis", the prompt then forbade the repair, and the tool
        # failed on the file (hunt 2026-09-30, u22-spatial-pipeline-4). `validate_spatial_h5ad` holds
        # the same two rules.
        n_coord_dims, n_nan_coords = 2, 0
        if has_spatial:
            try:
                import numpy as _np

                spatial_arr = _np.asarray(adata.obsm["spatial"], dtype=float)
                n_coord_dims = int(spatial_arr.shape[1]) if spatial_arr.ndim == 2 else 0
                if n_coord_dims >= 2:
                    n_nan_coords = int(_np.isnan(spatial_arr).any(axis=1).sum())
            except (TypeError, ValueError):
                n_coord_dims = 0

        # Multi-slice facts, added 2026-09-21 by Program 7 (D-063 item 5).
        #
        # These keys are ADDITIVE -- nothing above changes meaning -- and they exist because this
        # function is the mandatory entry point on every path, including the scored one, and on a
        # stack of serial sections it reported coordinates as three booleans. No column count, no
        # per-axis range, no notion that the object holds more than one section. That is the worst
        # possible answer on exactly the datasets a 3D study is about, and the agent consults this
        # before it picks a tool.
        #
        # Everything here is cheap: obsm shapes are metadata, the ranges are two reductions over
        # coordinates already in memory, and the section scan stops at the first obs column that
        # qualifies. No matrix is read.
        try:
            import numpy as _np

            obsm_shapes = {}
            for key in list(adata.obsm):
                try:
                    obsm_shapes[str(key)] = list(_np.asarray(adata.obsm[key]).shape)
                except Exception:
                    continue
            info["coordinates"]["obsm_shapes"] = obsm_shapes
            info["coordinates"]["three_column_keys"] = sorted(
                k for k, shape in obsm_shapes.items() if len(shape) == 2 and shape[1] == 3
            )

            if has_spatial:
                arr = _np.asarray(adata.obsm["spatial"], dtype=float)
                info["coordinates"]["n_dims"] = int(arr.shape[1]) if arr.ndim == 2 else 0
                finite = arr[_np.isfinite(arr).all(axis=1)] if arr.ndim == 2 else arr[:0]
                if len(finite):
                    info["coordinates"]["per_axis_range"] = [
                        [float(lo), float(hi)] for lo, hi in zip(finite.min(axis=0), finite.max(axis=0), strict=True)
                    ]

            # The section axis, if there is one. Reported, never acted on here.
            #
            # It is found by NAME, so a miss is a miss on names and not a finding about the object.
            # Until 2026-09-25 a miss was reported as section_key "", n_sections 0 and
            # looks_like_a_stack false, and in the E-03 arm that went into the scored prompt on all
            # sixteen evals -- among them the brain-aging MERFISH file, whose obs['slice'] indexes
            # each donor's sections, and a Xenium kidney file whose obs['ident'] names twelve
            # kidneys by timepoint and side. On a miss the count and the verdict are null, and the
            # note says the number of sections is unknown and why.
            #
            # No structural stand-in, because the one on offer does not separate the cases. On the
            # ten staged benchmark files the coordinate bounding boxes of cell-type, niche and
            # cluster levels overlap each other as much as those of section levels do (median IoU
            # 0.81-0.95 against 0.63-0.88), so any threshold that admits the sections admits the
            # cell types with them.
            #
            # n_sections counts the distinct values of the recognised column, which is a floor on
            # the sections rather than their number: on that same MERFISH file obs['slice'] is a
            # per-donor index, three values over 31 (donor, slice) sections.
            #
            # A recognised column that is constant divides nothing, and it is no evidence of a
            # single section either: add a constant obs['batch'] to the Xenium kidney file, whose
            # twelve kidneys are under obs['ident'], and it still holds twelve. So a constant column
            # leaves the count and the verdict as unknown as a miss does and is only named
            # (constant_section_columns); one uninformative column must not turn "unknown" into
            # "one section, not a stack".
            section_key, n_levels = "", None
            constant: list[str] = []
            for column in _section_columns(adata.obs.columns):
                # dropna first: astype(str) turns a missing label into the string 'nan', a section of its own
                levels = int(adata.obs[column].dropna().astype(str).nunique())
                if levels >= 2:
                    section_key, n_levels = str(column), levels
                    break
                constant.append(str(column))
            # Descriptive, and conditional on the task. This function is consulted before every
            # kind of analysis, not only a 3D one, so each note states a fact about the object and
            # what that fact would mean IF the question is a cross-section one. None instructs,
            # because on a clustering or an SVG run the section count is context, not a next step.
            multi: dict[str, Any] = {"section_key": section_key, "n_sections": n_levels}
            if n_levels is None:
                multi["looks_like_a_stack"] = None
                multi["section_names_checked"] = list(_SECTION_OBS_NAMES)
                multi["section_names_any_case"] = list(_SECTION_OBS_NAMES_ANY_CASE)
                if constant:
                    multi["constant_section_columns"] = constant
                why = (
                    "Every obs column recognised as naming a section is constant (constant_section_columns), "
                    "so none divides this object into sections, and"
                    if constant
                    else "No obs column is recognised as naming a section, so"
                )
                multi["note"] = (
                    f"{why} the number of sections is unknown; this is not a finding that the object holds "
                    "a single section. The section axis is read from column names only "
                    "(section_names_checked, as spelled, and in any case for section_names_any_case), and "
                    "an object that labels its sections under another name looks the same to this check "
                    "as an object holding one."
                )
            else:
                multi["looks_like_a_stack"] = True
                multi["note"] = (
                    f"obs[{section_key!r}] holds {n_levels} distinct values, so this object carries "
                    f"more than one section -- at least {n_levels}, and more if a value labels a "
                    f"section in each of several donors or samples. If the question is a "
                    f"cross-section or 3D one, whether those sections share a coordinate system is "
                    f"not established by their being in one file, and diagnose_3d_stack is what "
                    f"settles it."
                )
            info["multi_slice"] = multi
        except Exception:
            # Additive facts must never take down the diagnosis they were added to.
            pass

        # Check images
        has_images = False
        image_info: dict[str, Any] = {"found": False}
        sp = adata.uns.get("spatial")
        if isinstance(sp, dict):
            for lib_id, lib_data in sp.items():
                if not isinstance(lib_data, dict):
                    continue
                imgs = lib_data.get("images", {})
                if imgs:
                    has_images = True
                    image_info = {
                        "found": True,
                        "embedded_in_h5ad": True,
                        "library_id": lib_id,
                        "qualities": list(imgs.keys()),
                        "has_scalefactors": "scalefactors" in lib_data,
                    }
                    break
        info["images"] = image_info

        # Check QC metrics
        has_qc = "total_counts" in adata.obs.columns and "n_genes_by_counts" in adata.obs.columns
        info["qc_metrics"] = {"found": has_qc}
        try:
            info["qc_metrics"].update(_empty_observations(p, adata, info["n_obs"]))
        except Exception:
            pass  # an additive fact must never take down the diagnosis it was added to

        # Check var_names
        info["var_names_unique"] = adata.var_names.is_unique

        # Determine pipeline steps
        steps = []
        needs_repair = False

        if not has_spatial and coord_cols:
            steps.append("repair_spatial_h5ad (move coords from obs to obsm['spatial'])")
            needs_repair = True
        elif not has_spatial:
            steps.append("CRITICAL: need spatial coordinates")
        elif n_coord_dims < 2:
            steps.append(f"CRITICAL: obsm['spatial'] has {n_coord_dims} coordinate column(s); need x and y")
        elif n_nan_coords:
            steps.append("repair_spatial_h5ad (drop observations with NaN coordinates)")
            needs_repair = True

        if not has_qc:
            steps.append("repair_spatial_h5ad (compute QC metrics)")
            needs_repair = True

        if not adata.var_names.is_unique:
            steps.append("repair_spatial_h5ad (deduplicate var_names)")
            needs_repair = True

        if not has_images:
            steps.append("optional: embed_image_in_h5ad (if histology available)")

        if needs_repair:
            steps.insert(0, "repair_spatial_h5ad")
        steps.append("validate_spatial_h5ad")

        # Summary.
        #
        # Two different questions live here and they have different answers, so they get two lists.
        # "(auto-fixable)" answers *can repair_spatial_h5ad fix it* -- true of all three fixable
        # items. `blockers` answers *will the analysis tool cope if nobody repairs it*, which is the
        # question `mcp_ready` is actually read for: `_build_pipeline_steps` picks copy over repair
        # when it is True, and `enrich_prompt_with_spatial_diagnosis` then tells the turn "Do NOT
        # call run_spatial_pipeline or repair_spatial_h5ad as a preamble".
        #
        # Missing QC and duplicate var_names are tolerated on purpose: scanpy, BASS, GraphST and
        # STAGATE all recompute and deduplicate internally, so demanding a repair first would only
        # add a step. Coordinates parked in obs are not tolerated -- every spatial tool reads
        # obsm["spatial"] and raises KeyError without it -- so tagging them auto-fixable used to
        # send them down the tolerated path and produce the one outcome the tag was added to
        # prevent: the file reported ready, the repair forbidden, and the tool run anyway.
        issues = []
        blockers = []
        if not has_x:
            issues.append("missing expression matrix")
            blockers.append(issues[-1])
        if not has_spatial and not coord_cols:
            issues.append("missing spatial coordinates")
            blockers.append(issues[-1])
        elif coord_cols and not has_spatial:
            issues.append("spatial coords in obs, not obsm['spatial'] (auto-fixable)")
            blockers.append(issues[-1])
        elif n_coord_dims < 2:
            issues.append(f"obsm['spatial'] has {n_coord_dims} coordinate column(s); a spatial tool needs x and y")
            blockers.append(issues[-1])
        elif n_nan_coords:
            issues.append(
                f"{n_nan_coords} of {adata.n_obs} observations have a NaN spatial coordinate "
                "(auto-fixable: repair drops them)"
            )
            blockers.append(issues[-1])
        if not has_qc:
            issues.append("missing QC metrics (auto-fixable)")
        if not adata.var_names.is_unique:
            issues.append("non-unique var_names (auto-fixable)")

        if not issues:
            info["summary"] = "h5ad is MCP-compatible. Ready for analysis."
            if not has_images:
                info["summary"] += " No images embedded (optional for most tools)."
        elif not blockers:
            info["summary"] = (
                f"h5ad is MCP-ready. Minor auto-fixable items detected ({'; '.join(issues)}), "
                "but analysis MCP tools (scanpy, BASS, GraphST, STAGATE, etc.) compute QC and "
                "deduplicate var_names internally. Proceed directly to the analysis MCP tool. "
                "Only call repair_spatial_h5ad / run_spatial_pipeline if the analysis tool errors."
            )
        else:
            info["summary"] = f"h5ad has issues: {'; '.join(issues)}."
            if all("auto-fixable" in b for b in blockers):
                info["summary"] += " Run repair_spatial_h5ad (or run_spatial_pipeline) before the analysis tool."

        info["pipeline_steps"] = steps
        info["mcp_ready"] = not blockers
        return info
    finally:
        # backed="r" holds an HDF5 lock, and the caller's next step is routinely
        # repair_spatial_h5ad / run_spatial_pipeline rewriting this very path -- which then
        # fails with errno 11. Dropping the local is not enough: an input with a .raw slot is
        # cyclic, so it survives until the collector runs.
        with contextlib.suppress(Exception):
            adata.file.close()


def _probe_table_dialect(p: Path) -> tuple[str | None, int]:
    """The delimiter and '#' preamble length measured from the file's bytes, or ``(None, 0)``.

    Reuses :func:`~spatialomicsgym.utils.format_probe.probe_format`, which ranks the candidate
    delimiters by how consistently each one splits the sampled lines and excludes a leading '#'
    preamble from that count. There is no second heuristic here on purpose: this report already
    carries that verdict under ``format_probe``, and a delimiter the report disagreed with would be
    worse than the extension guess it replaces.

    Failure is swallowed for the same reason it is in :func:`_attach_format_probe` -- the probe is a
    diagnostic aid bolted onto ingest, and one that can take down a diagnosis which used to work
    without it is a net loss.
    """
    try:
        details = probe_format(p).details or {}
        delimiter = details.get("delimiter")
        comment_lines = int(details.get("comment_lines") or 0)
    except Exception:
        return None, 0
    if not isinstance(delimiter, str) or not delimiter:
        return None, 0
    return delimiter, max(comment_lines, 0)


def _diagnose_tabular_file(p: Path, result: dict) -> str:
    """Diagnose a CSV/TSV file to identify its spatial platform."""
    import pandas as pd

    ext_sep = "\t" if p.suffix.lower() == ".tsv" else ","
    try:
        df_head = pd.read_csv(p, nrows=5, sep=ext_sep)
    except Exception:
        try:
            df_head = pd.read_csv(p, nrows=5, sep="\t")
        except Exception as e:
            result["detected_format"] = "unknown_tabular"
            result["summary"] = f"Could not parse tabular file: {e}"
            return json.dumps(result, indent=2)

    # One column means the separator was wrong, not that the file has one column. The retry above
    # fires only when pandas *raises*, and reading a tab-delimited `.csv` with `sep=","` does not
    # raise -- it returns one very wide column -- so it never fired for the case that occurs. Every
    # column test below then looked at a single column named after the whole header line, and a
    # Xenium transcript table delivered as a tab-delimited `.csv`, or a semicolon locale export,
    # came back `unknown_tabular` / "Format not auto-detected" with nothing to dispatch on.
    #
    # The extension cannot answer this, so ask the bytes. Only on the one-column reading: a file
    # that already parsed into columns is not re-litigated, which keeps every table that resolves
    # today resolving identically.
    if df_head.shape[1] == 1:
        measured, comment_lines = _probe_table_dialect(p)
        if measured and measured != ext_sep:
            with contextlib.suppress(Exception):
                # `comment_lines` too, not just the delimiter: a Stereo-seq GEM opens with
                # `#FileFormat=GEMv0.1`, and splitting that on tabs would make the preamble the
                # header row and demote the real one to data.
                retried = pd.read_csv(p, nrows=5, sep=measured, skiprows=comment_lines)
                if retried.shape[1] > 1:
                    df_head = retried

    cols = set(df_head.columns.str.lower())

    # Xenium transcripts
    if {"cell_id", "feature_name", "x_location"}.issubset(cols):
        result["detected_format"] = "xenium_transcripts"
        result["platform"] = "10x Xenium"
        result["expression"] = {"found": True, "format": "transcript-level CSV"}
        result["coordinates"] = {"found": True, "format": "x_location/y_location columns"}
        result["images"] = {"found": False, "note": "Look for morphology_focus.ome.tif alongside"}
        result["summary"] = "Xenium transcript CSV. Needs cell aggregation to create cell-level h5ad."
        result["pipeline_steps"] = ["convert_xenium", "check_for_images", "validate"]
    # MERFISH cell_by_gene
    elif _uncompressed_name(p.name) in ("cell_by_gene.csv", "cellbygene.csv"):
        result["detected_format"] = "merfish_expression"
        result["platform"] = "MERFISH/Vizgen"
        result["expression"] = {"found": True, "format": "cell-by-gene matrix"}
        result["coordinates"] = {"found": False, "note": "Need cell_metadata.csv for coordinates"}
        result["images"] = {"found": False, "note": "Look for mosaic_DAPI_z0.tif alongside"}
        result["summary"] = "MERFISH expression matrix. Need cell_metadata.csv with coordinates."
        result["pipeline_steps"] = ["find_cell_metadata", "convert_merfish", "check_for_images", "validate"]
    # CosMx
    elif "centerx_global_px" in cols or ("fov" in cols and "cell_id" in cols):
        result["detected_format"] = "cosmx_data"
        result["platform"] = "Nanostring CosMx"
        result["expression"] = {"found": True, "format": "cell expression matrix"}
        result["coordinates"] = {"found": True, "format": "CenterX/Y_global_px columns"}
        result["images"] = {"found": False, "note": "Look for CellComposite_F*.tif FOV images"}
        result["summary"] = "CosMx expression data. Need FOV positions file for full conversion."
        result["pipeline_steps"] = ["convert_cosmx", "check_for_images", "validate"]
    # Stereo-seq GEM (tab-separated with geneID, x, y, MIDCount)
    elif {"geneid", "x", "y"}.issubset(cols) or {"gene", "x", "y"}.issubset(cols):
        result["detected_format"] = "stereoseq_gem"
        result["platform"] = "Stereo-seq"
        result["expression"] = {"found": True, "format": "GEM transcript table"}
        result["coordinates"] = {"found": True, "format": "x/y columns"}
        result["images"] = {"found": False}
        result["summary"] = "Stereo-seq GEM-format data. Needs spatial binning."
        result["pipeline_steps"] = ["convert_stereoseq", "validate"]
    # A Xenium cells table: centroids and per-cell summaries, no expression. It matched the generic
    # branch below on x_centroid/y_centroid and was reported as carrying expression, so the run
    # failed and, following the error, converted transcript_counts and cell_area as two "genes"
    # (hunt 2026-09-30, u22-spatial-pipeline-8). The Xenium directory branch already says this.
    elif _is_xenium_cells_table(cols):
        rerooted = _diagnose_as_part_of_a_xenium_output(p)
        if rerooted is not None:
            return rerooted
        result["detected_format"] = "xenium_cells"
        result["platform"] = "10x Xenium"
        result["expression"] = {"found": False, "note": "a cells table holds per-cell summaries, not gene counts"}
        result["coordinates"] = {"found": True, "format": "x_centroid/y_centroid columns"}
        result["images"] = {"found": False}
        result["summary"] = (
            "Xenium cells table: cell centroids and per-cell summaries (transcript_counts, cell_area), with no "
            "gene expression. convert_xenium reads the transcripts table (transcripts.parquet or "
            "transcripts.csv.gz) from the same Xenium output, which is not beside this file."
        )
        result["pipeline_steps"] = ["MISSING: transcripts.csv.gz or transcripts.parquet needed by convert_xenium"]
    # Generic with coordinates. The columns are recorded as the file spells them whenever that is not
    # the converter's lowercase x/y default, because the converter matches names exactly: a table
    # with X/Y or x_centroid/y_centroid was diagnosed convertible and then refused (hunt 2026-09-30,
    # u22-spatial-pipeline-8). The plan and the run pass them on as x_col/y_col.
    elif {"x", "y"}.issubset(cols) or {"x_centroid", "y_centroid"}.issubset(cols):
        spelled: dict[str, str] = {}
        for column in df_head.columns:
            spelled.setdefault(str(column).lower(), str(column))
        pair = ("x", "y") if {"x", "y"}.issubset(cols) else ("x_centroid", "y_centroid")
        columns = [spelled[pair[0]], spelled[pair[1]]]
        result["detected_format"] = "generic_csv_with_coords"
        result["platform"] = "unknown (generic tabular)"
        result["expression"] = {"found": True, "format": "tabular matrix with coordinates"}
        result["coordinates"] = {"found": True, "format": "x/y columns"}
        if columns != ["x", "y"]:
            result["coordinates"]["columns"] = columns
        result["images"] = {"found": False}
        result["summary"] = "Generic expression CSV with embedded coordinates."
        result["pipeline_steps"] = ["convert_generic_csv", "validate"]
    else:
        result["detected_format"] = "unknown_tabular"
        result["platform"] = "unknown"
        result["expression"] = {"found": "possible", "format": "tabular (needs inspection)"}
        result["coordinates"] = {"found": False, "note": "No recognized coordinate columns"}
        result["images"] = {"found": False}
        result["summary"] = f"Tabular file with columns: {list(df_head.columns)[:10]}. Format not auto-detected."
        result["pipeline_steps"] = ["manual_inspection_needed"]

    return json.dumps(result, indent=2)


#: Columns a Xenium cells table carries besides its centroids, lowercased. A table whose every
#: other column is one of these -- or a per-cell count or area -- holds no gene expression.
_XENIUM_CELL_SUMMARY_COLUMNS = {"cell_id", "segmentation_method", "z_level", "nucleus_count"}

#: A Xenium output's own files, any of which beside a file says the file is part of one.
_XENIUM_SIBLINGS = {
    "experiment.xenium",
    "transcripts.parquet",
    "transcripts.csv.gz",
    "transcripts.csv",
    "cells.parquet",
    "cells.csv.gz",
    "cells.csv",
}


def _is_xenium_cells_table(cols: set[str]) -> bool:
    """A Xenium cells table, recognised by its columns: centroids, transcript_counts and cell_area,
    and nothing else that could be a gene."""
    if not {"x_centroid", "y_centroid", "transcript_counts", "cell_area"} <= cols:
        return False
    rest = cols - {"x_centroid", "y_centroid"}
    return all(c in _XENIUM_CELL_SUMMARY_COLUMNS or c.endswith(("_counts", "_count", "_area")) for c in rest)


def _diagnose_as_part_of_a_xenium_output(p: Path) -> str | None:
    """The Xenium output directory holding ``p``, diagnosed, or None if ``p`` does not sit in one.

    Re-rooted the way ``_diagnose_outs`` re-roots a Space Ranger run: ``input_path`` stays what the
    user named, ``resolved_input_path`` is what the plan and the run read from, and the summary says
    why (u22-spatial-pipeline-8, -14).
    """
    parent = p.parent
    try:
        siblings = {f.name.lower() for f in parent.iterdir() if f.name != p.name}
    except OSError:
        return None
    if not siblings & _XENIUM_SIBLINGS:
        return None
    try:
        nested = json.loads(_diagnose_directory(parent, _allow_outs_redirect=False, _allow_bin_redirect=False))
    except (OSError, ValueError):
        return None
    if nested.get("detected_format") != "xenium_output":
        return None
    nested["input_path"] = str(p)
    nested["resolved_input_path"] = str(parent)
    nested["summary"] = (
        f"{p.name} is one file of the Xenium output in {parent.name}/, so that output was diagnosed. "
        f"{nested.get('summary', '')}"
    ).strip()
    return json.dumps(nested, indent=2)


# ---------------------------------------------------------------------------
# Internal: Directory diagnosis
# ---------------------------------------------------------------------------


def _diagnose_outs(p: Path) -> str | None:
    """Re-run detection rooted at ``p/"outs"``, or None if that is no more recognisable than ``p``.

    Space Ranger writes its deliverables one level down, into ``outs/``, and what a user hands over
    is the run directory that contains it -- so the shape the tool actually ships was the shape
    detection refused, reporting `unknown_directory` for a complete Visium run.

    One level, and only when it resolves. Walking the tree would make the verdict depend on how
    deeply someone nested their copy and could pick up a neighbouring dataset; preferring a nested
    report that is *also* unrecognised would describe files at a path the user never named, which
    is a worse answer rather than a better one.
    """
    outs = p / "outs"
    try:
        nested = json.loads(_diagnose_directory(outs, _allow_outs_redirect=False))
    except (OSError, ValueError):
        return None

    if nested.get("detected_format") in (None, "unknown_directory"):
        return None

    # The user asked about `p`, so that stays the subject of the report; `resolved_input_path` is
    # what conversion and image embedding must actually be pointed at. Only when the nested report
    # has not already resolved further: a Visium HD run redirects twice, `outs/` and then the bin,
    # and overwriting the deeper answer here would point conversion at a directory of bins.
    nested["input_path"] = str(p)
    nested["resolved_input_path"] = nested.get("resolved_input_path") or str(outs)
    nested["summary"] = (
        f"{nested.get('summary', '')} Detected from the outs/ subdirectory: the path given is the "
        "Space Ranger run directory, whose outputs live one level down in outs/."
    ).strip()
    return json.dumps(nested, indent=2)


# Space Ranger 3 writes a Visium HD run as one complete Space Ranger output per bin size, under
# `binned_outputs/square_<NNN>um/`. Each bin is a dataset in its own right, and every one of them is
# two levels below the directory a user hands over -- one further than the outs/ redirect reaches.
_HD_BIN_RE = re.compile(r"^square_(\d+)um$")
_HD_DEFAULT_BIN_UM = 8


def _visium_hd_bins(holder: Path) -> dict[int, Path]:
    """Bin size in microns -> the bin directory, for bins that hold a Space Ranger output.

    The `spatial/` check is what stops a directory merely *named* like a bin from redirecting the
    whole run somewhere with nothing readable in it.
    """
    try:
        children = sorted(holder.iterdir())
    except OSError:
        return {}
    found: dict[int, Path] = {}
    for child in children:
        match = _HD_BIN_RE.match(child.name.lower())
        if match and (child / "spatial").is_dir():
            found[int(match.group(1))] = child
    return found


def _diagnose_hd_bin(p: Path) -> str | None:
    """Re-run detection rooted at one bin of a Visium HD run, or None if this is not one.

    8um is the bin the field analyses -- roughly one cell -- so it is the one converted when it is
    present. Otherwise the largest, because it is the one that loads; choosing 2um for someone by
    default would pick the heaviest dataset in the run. The others are named in the summary rather
    than silently dropped: they are separate datasets, not parts of one.
    """
    holder = p / "binned_outputs"
    bins = _visium_hd_bins(holder if holder.is_dir() else p)
    if not bins:
        return None

    chosen = _HD_DEFAULT_BIN_UM if _HD_DEFAULT_BIN_UM in bins else max(bins)
    target = bins[chosen]
    try:
        nested = json.loads(_diagnose_directory(target, _allow_outs_redirect=False, _allow_bin_redirect=False))
    except (OSError, ValueError):
        return None
    if nested.get("detected_format") in (None, "unknown_directory"):
        return None

    others = [f"{um}um" for um in sorted(bins) if um != chosen]
    nested["input_path"] = str(p)
    nested["resolved_input_path"] = str(target)
    nested["platform"] = f"10x Visium HD ({chosen}um bin)"
    nested["visium_hd_bins"] = sorted(bins)
    nested["summary"] = (
        f"{nested.get('summary', '')} Detected as a Visium HD run: the {chosen}um bin under "
        f"{target.parent.name}/ was diagnosed."
        + (
            f" This run also holds {', '.join(others)}, which are separate datasets -- analysing "
            "them means reading each in turn."
            if others
            else ""
        )
    ).strip()
    return json.dumps(nested, indent=2)


def _diagnose_directory(p: Path, _allow_outs_redirect: bool = True, _allow_bin_redirect: bool = True) -> str:
    """Diagnose a directory containing spatial data."""
    contents = {f.name for f in p.iterdir() if not f.name.startswith(".")}
    # The exact-name marker tests below compare against this rather than the literal names, because
    # a `.gz` on a table is an encoding and every converter they plan reads either form. See
    # `_uncompressed_name`; for the `.h5` and directory markers it is the lowercased name unchanged.
    contents_lower = {_uncompressed_name(f) for f in contents}
    subdirs = {f.name for f in p.iterdir() if f.is_dir()}

    result: dict[str, Any] = {
        "status": "diagnosed",
        "input_type": "directory",
        "input_path": str(p),
        "total_files": len(contents),
    }

    expression: dict[str, Any] = {"found": False}
    coordinates: dict[str, Any] = {"found": False}
    images: dict[str, Any] = {"found": False, "files": []}
    steps: list[str] = []

    # --- Detect platform ---

    # 10x Visium Space Ranger. Prefix-tolerant, as the content probe and the converter are: 10x
    # ships every public dataset as `<sample>_filtered_feature_bc_matrix.h5` beside `spatial/`
    # (hunt 2026-09-30, u22-spatial-pipeline-1).
    has_spatial_dir = "spatial" in subdirs
    has_counts_h5 = _any_marker(contents, _VISIUM_SPACERANGER_MARKERS)
    if has_spatial_dir and has_counts_h5:
        result["detected_format"] = "visium_spaceranger"
        result["platform"] = "10x Visium (Space Ranger)"
        expression = {"found": True, "format": "10x HDF5/MTX counts matrix"}

        # List spatial/ once, here, and answer both of its questions from that listing. The
        # coordinate claim used to be hardcoded True on nothing but the subdirectory's *name*,
        # three lines above the listing that would have settled it -- so a directory holding the
        # positions table and one holding only the images and scalefactors produced byte-identical
        # reports. That second shape is ordinary (a tissue-image-only export, or a trimmed deposit)
        # and both conversion routes fail on it: `sc.read_visium` raises on the missing positions
        # table, and the MTX route's `_attach_visium_spatial` raises FileNotFoundError. This is the
        # last of three siblings -- the Xenium branch below and the `10x_h5_counts` branch above
        # were both fixed for the same shape, and the `startswith` idiom comes from the latter.
        spatial_dir = p / "spatial"
        spatial_contents = {f.name for f in spatial_dir.iterdir()} if spatial_dir.is_dir() else set()
        positions = next((n for n in sorted(spatial_contents) if n.startswith("tissue_positions")), None)
        if positions:
            coordinates = {"found": True, "format": f"tissue_positions in spatial/ directory ({positions})"}
        else:
            coordinates = {
                "found": False,
                "note": f"spatial/ directory present at {spatial_dir} but holds no tissue_positions file",
            }

        # Check for images in spatial/
        img_files = spatial_contents & _VISIUM_IMAGE_MARKERS
        sf_found = "scalefactors_json.json" in spatial_contents
        if img_files:
            images = {
                "found": True,
                "type": "H&E histology",
                "files": sorted(img_files),
                "scalefactors": sf_found,
                "path": str(spatial_dir),
            }
        steps = ["convert_visium_spaceranger"]
        if img_files:
            steps.append("process_visium_images")
        steps.append("validate_spatial_h5ad")

    # 10x Xenium
    elif contents_lower & _marker_names(_XENIUM_MARKERS):
        result["detected_format"] = "xenium_output"
        result["platform"] = "10x Xenium"
        # Checked, not assumed. Both of these were hardcoded True on a match against
        # `_XENIUM_MARKERS`, which includes `cells.csv.gz` and `cells.parquet` -- cell centroids,
        # carrying no counts at all. A directory holding only those was reported as having an
        # expression matrix that was not there and handed `convert_xenium` as its next step.
        # Only the transcripts table supplies both halves: `convert_xenium` takes `transcripts_path`
        # and derives the matrix and the coordinates from it, so its absence is what has to be said.
        has_tx = bool(contents_lower & {"transcripts.csv", "transcripts.parquet"})
        has_cell_matrix = "cell_feature_matrix.h5" in contents_lower
        has_cells = bool(contents_lower & {"cells.csv", "cells.parquet"})
        expression = {
            "found": has_tx or has_cell_matrix,
            "format": "transcripts table" if has_tx else "cell_feature_matrix.h5",
        }
        coordinates = {
            "found": has_tx or has_cells,
            "format": "x_location/y_location in transcripts" if has_tx else "centroids in cells.csv.gz/parquet",
        }
        # Check for morphology images. Ranked, not read off a set: the plan and the run are separate
        # processes, `contents` iterates in string-hash order, and files[0] is what gets embedded --
        # so one run took the 2-D focus image and another the multi-GB 3-D z-stack (hunt 2026-09-30,
        # u22-spatial-pipeline-11).
        img_files = sorted(
            f for f in contents if any(f.lower().endswith(ext) for ext in [".ome.tif", ".ome.tiff", ".tif"])
        )
        xenium_imgs = sorted(
            (f for f in contents if f.lower() in _XENIUM_IMAGE_MARKERS or f.lower().startswith("morphology")),
            key=_xenium_image_rank,
        )
        if xenium_imgs or img_files:
            images = {"found": True, "type": "fluorescence/DAPI", "files": xenium_imgs or img_files}
        steps = []
        # One file named, not "the export is incomplete": a `cell_feature_matrix.h5` next to a
        # `cells.parquet` covers both halves between them and still cannot be converted, because
        # nothing here reads either one. What has to be found is the transcripts table.
        if not has_tx:
            steps.append("MISSING: transcripts.csv.gz or transcripts.parquet needed by convert_xenium")
        steps.append("convert_xenium")
        if images["found"]:
            steps.append("process_xenium_images")
        steps.append("validate_spatial_h5ad")

    # MERFISH / Vizgen
    elif contents_lower & _marker_names(_MERFISH_MARKERS):
        result["detected_format"] = "merfish_vizgen"
        result["platform"] = "MERFISH/Vizgen"
        has_expr = "cell_by_gene.csv" in contents_lower
        has_meta = "cell_metadata.csv" in contents_lower
        expression = {"found": has_expr, "format": "cell_by_gene.csv"}
        coordinates = {"found": has_meta, "format": "cell_metadata.csv (center_x/y)"}
        # Check for mosaic images
        mosaic_files = _rank_mosaics(
            [f for f in contents if f.lower().startswith("mosaic_") and f.lower().endswith(".tif")]
        )
        img_dir = p / "images" if (p / "images").exists() else None
        if mosaic_files:
            images = {"found": True, "type": "mosaic DAPI/PolyT TIFF", "files": mosaic_files}
        elif img_dir:
            images = {"found": True, "type": "image directory", "path": str(img_dir)}
        # Both halves, not just the coordinates. `detected_transcripts.csv` alone matches
        # `_MERFISH_MARKERS`, and that export was reported as missing one file when it was missing
        # two -- `convert_merfish` takes the matrix and the metadata as required arguments, so a
        # plan naming only one of them understates what has to be found before it can run.
        if not has_expr:
            steps.append("MISSING: cell_by_gene.csv needed for expression")
        if not has_meta:
            steps.append("MISSING: cell_metadata.csv needed for coordinates")
        steps.append("convert_merfish")
        if images["found"]:
            steps.append("process_merfish_images")
        steps.append("validate_spatial_h5ad")

    # Nanostring CosMx
    elif _any_marker(contents, _COSMX_MARKERS):
        result["detected_format"] = "cosmx_output"
        result["platform"] = "Nanostring CosMx"
        # Checked, not assumed. Both of these were hardcoded True, so an export carrying only
        # `tx_file.csv` -- enough to match `_COSMX_MARKERS` -- was reported as having an expression
        # matrix *and* coordinates, naming two files that were not there. `convert_cosmx` needs
        # both and refuses without them, so the report contradicted the run it was planning. The
        # MERFISH branch above answers the same question by looking; this one now does too.
        has_expr = _any_marker(contents, {"exprMat_file.csv"})
        has_meta = _any_marker(contents, {"metadata_file.csv"})
        expression = {"found": has_expr, "format": "exprMat_file.csv"}
        coordinates = {"found": has_meta, "format": "metadata_file.csv (CenterX/Y_global_px)"}
        # Check for FOV composites, at the top level or in the subdirectory the instrument uses
        composite_dir, fov_files = _find_cosmx_composites(p, contents)
        if fov_files:
            images = {
                "found": True,
                "type": "FOV composite fluorescence",
                "files": fov_files[:5],
                "total_fovs": len(fov_files),
                "path": str(composite_dir),
            }
        # Named one file at a time: which of the two is absent decides what the user has to go and
        # find, and `convert_cosmx` needs both, so "CosMx is incomplete" would not be enough to act on.
        steps = []
        if not has_expr:
            steps.append("MISSING: exprMat_file.csv needed for expression")
        if not has_meta:
            steps.append("MISSING: metadata_file.csv needed for coordinates")
        steps.append("convert_cosmx")
        if images["found"]:
            steps.append("process_cosmx_fov_images")
        steps.append("validate_spatial_h5ad")

    # Slide-seq
    elif contents_lower & _marker_names(_SLIDESEQ_MARKERS):
        result["detected_format"] = "slideseq"
        result["platform"] = "Slide-seq"
        # Checked, not assumed, for the same reason as CosMx above: `_SLIDESEQ_MARKERS` contains
        # `BeadLocationsForR.csv`, so a puck downloaded without its DGE matched the branch and was
        # reported as having both halves -- while the probe warning in the same JSON said the
        # expression matrix was missing. Resolved through the marker tuples `convert_slideseq`
        # itself resolves its two arguments with, so the report and the run agree on what counts.
        has_counts = _any_marker(contents, set(_SLIDESEQ_COUNT_MARKERS))
        has_locs = _any_marker(contents, set(_SLIDESEQ_LOCATION_MARKERS))
        expression = {"found": has_counts, "format": "DGE matrix TSV"}
        coordinates = {"found": has_locs, "format": "bead locations CSV"}
        images = {"found": False, "note": "Slide-seq typically has no paired image"}
        steps = []
        if not has_counts:
            steps.append(
                "MISSING: a DGE matrix (MappedDGEForR.csv / *.digital_expression.txt.gz) needed for expression"
            )
        if not has_locs:
            steps.append("MISSING: BeadLocationsForR.csv or bead_locations.csv needed for coordinates")
        steps.append("convert_slideseq")
        steps.append("validate_spatial_h5ad")

    # Fallback: scan for h5ad files or recognizable patterns
    else:
        h5ad_files = sorted(f for f in contents if f.endswith(".h5ad"))
        # Split by half of the format, because only the text one converts. SAW writes both, and when
        # both are here the `.gem` is the one taken -- which is the order `_find_file` resolves in.
        gem_files = [f for f in contents if f.endswith((".gem", ".gem.gz"))]
        gef_files = [f for f in contents if f.endswith(".gef")]
        # Compression stripped here and not on the three above: their readers are h5py and the
        # GEM converter, neither of which opens a gzipped container. `.gem.gz` is already listed
        # because `convert_stereoseq` decompresses that one itself.
        csv_files = [f for f in contents if _uncompressed_name(f).endswith((".csv", ".tsv"))]
        img_files = [f for f in contents if any(f.lower().endswith(ext) for ext in _IMAGE_EXTENSIONS)]

        if h5ad_files:
            result["detected_format"] = "directory_with_h5ad"
            result["platform"] = "pre-converted"
            # Which file is THE dataset was `h5ad_files[0]` of a set, so a deconvolution folder
            # holding the slide and its single-cell reference was diagnosed -- and then copied as
            # the output -- as one or the other by string-hash seed, and the other file was never
            # named (hunt 2026-09-30, u22-spatial-pipeline-2). Sorted, the one carrying
            # obsm['spatial'] preferred, and every candidate named.
            chosen = _choose_h5ad(p, h5ad_files)
            h5ad_diag = _diagnose_h5ad(p / chosen)
            result.update(h5ad_diag)
            result["h5ad_file"] = chosen
            if len(h5ad_files) > 1:
                others = [f for f in h5ad_files if f != chosen]
                spatial = [f for f in h5ad_files if _has_obsm_spatial(p / f)]
                if spatial == [chosen]:
                    why = " because it is the only one carrying obsm['spatial']"
                elif chosen in spatial:
                    why = f" (first in name order of the {len(spatial)} carrying obsm['spatial'])"
                else:
                    why = " (first in name order; none of them carries obsm['spatial'])"
                result["h5ad_files"] = h5ad_files
                result.setdefault("warnings", []).append(
                    f"this directory holds {len(h5ad_files)} h5ad files; {chosen} was diagnosed{why}, and "
                    f"{', '.join(others)} {'was' if len(others) == 1 else 'were'} not. If another one is the "
                    "dataset to analyse, diagnose that file by name."
                )
            return json.dumps(result, indent=2)
        elif gem_files:
            result["detected_format"] = "stereoseq_directory"
            result["platform"] = "Stereo-seq"
            expression = {"found": True, "format": "GEM files", "files": gem_files}
            coordinates = {"found": True, "format": "embedded in GEM"}
            steps = ["convert_stereoseq", "validate_spatial_h5ad"]
        elif gef_files:
            # Only the binary half is here. Same refusal as a bare `.gef`, for the same reason.
            result["detected_format"] = "stereoseq_gef"
            result["platform"] = "Stereo-seq"
            expression = {"found": True, "format": "GEF binary container (HDF5)", "files": gef_files}
            coordinates = {"found": True, "format": "embedded in the GEF, in chip coordinates"}
            steps = [_STEREOSEQ_GEF_MISSING_STEP, "convert_stereoseq", "validate_spatial_h5ad"]
        else:
            # Before giving up: a Visium HD run keeps a whole Space Ranger output per bin under
            # binned_outputs/, and a Space Ranger run keeps its deliverables in outs/. HD first,
            # because it is the more specific layout; an HD run as shipped needs both hops.
            if _allow_bin_redirect:
                rerooted = _diagnose_hd_bin(p)
                if rerooted is not None:
                    return rerooted
            if _allow_outs_redirect and (p / "outs").is_dir():
                rerooted = _diagnose_outs(p)
                if rerooted is not None:
                    return rerooted
            result["detected_format"] = "unknown_directory"
            result["platform"] = "unknown"
            result["summary"] = (
                f"Could not auto-detect platform. Directory contains: "
                f"{len(csv_files)} CSV/TSV, {len(img_files)} images, "
                f"{len(h5ad_files)} h5ad files. Subdirectories: {sorted(subdirs)[:5]}"
            )
            expression = {"found": bool(csv_files), "format": "tabular (needs inspection)", "files": csv_files[:5]}
            coordinates = {"found": False, "note": "Need manual specification"}
            if img_files:
                images = {"found": True, "type": "various", "files": img_files[:5]}
            steps = ["manual_inspection_needed"]

    result["expression"] = expression
    result["coordinates"] = coordinates
    result["images"] = images
    result["pipeline_steps"] = steps

    # Build summary
    parts = [f"Detected: {result.get('platform', 'unknown')}"]
    parts.append(f"Expression: {'yes' if expression.get('found') else 'NO'}")
    parts.append(f"Coordinates: {'yes' if coordinates.get('found') else 'NO'}")
    parts.append(f"Images: {'yes (' + images.get('type', '') + ')' if images.get('found') else 'no'}")
    result["summary"] = ". ".join(parts) + "."

    return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Internal: Pipeline step building
# ---------------------------------------------------------------------------

#: Step actions that leave an h5ad on disk. `estimated_outputs` describes that file, so a plan
#: holding none of these is describing a file nothing writes.
_PRODUCES_H5AD = frozenset({"convert_expression", "repair", "copy"})

#: The converter `_execute_conversion` reaches for, for the formats that have no branch of their
#: own in `_build_pipeline_steps`. Keyed by detected_format; the value builds (function, params)
#: from the resolved input, because these converters need a sibling file the plan has to name.
_PLAN_CONVERTER_MODULE = "spatialomicsgym.tool.spatial_data_converter"


def _plan_conversion_step(fmt: str, p: Path, output_path: str, diagnosis: dict) -> dict[str, Any]:
    """Step 1 for a format with no branch of its own, decided the way `_execute_conversion` decides.

    A plan is published as ``{function, module, params}`` and read as executable -- it is what the
    tool description calls the FIRST STEP for any spatial task. The fallthrough used to publish
    ``auto_convert`` for every format without a branch, which is right for the three that
    ``auto_convert`` resolves from the file itself and wrong twice over for the rest:

    * six formats convert in the run through a converter ``auto_convert`` has no branch for. A
      Visium ``filtered_feature_bc_matrix.h5`` with its ``spatial/`` folder beside it converts, and
      the plan for it named a function whose answer is ``Could not auto-detect format``.
    * nine formats convert nowhere. The run refuses them by name and hands back a route out
      ("Read it with anndata.io.read_loom in run_python_repl, put the (x, y) column attributes
      into obsm['spatial']"); the plan replaced that with "Auto-detect and convert to h5ad".

    The refusals are phrased as the CosMx and Slide-seq branches phrase theirs -- a step that says
    BLOCKED and why -- so that a plan is never a promise the run will not keep.
    """

    def convert(function: str, params: dict[str, Any], description: str, action: str = "convert_expression"):
        return {
            "step": 1,
            "action": action,
            "function": function,
            "module": _PLAN_CONVERTER_MODULE,
            "params": params,
            "description": description,
        }

    def blocked(reason: str) -> dict[str, Any]:
        return {
            "step": 1,
            "action": "cannot_convert",
            "function": None,
            "module": None,
            "params": {},
            "description": f"BLOCKED: detected {fmt}, but it cannot be converted here -- {reason}",
        }

    if fmt in _UNIDENTIFIED_FORMATS:
        route = _UNIDENTIFIED_FORMATS[fmt] or (
            "the diagnosis carries `unknown_format_recovery`: an ordered route that identifies the bytes first"
        )
        return blocked(f"the format was never identified, so there is no converter to pick. {route}")
    if fmt in _NO_CONVERTER:
        return blocked(_NO_CONVERTER[fmt])

    if fmt == "r_object":
        return convert(
            "convert_r_object",
            {"r_file_path": str(p), "output_path": output_path},
            "Extract the Seurat/SpatialExperiment object into h5ad",
        )
    if fmt == "10x_h5_counts":
        spatial_dir = _find_spatial_dir(p)
        if spatial_dir is None:
            return blocked(
                f"no spatial/ folder with tissue positions sits beside {p.name}, and a counts matrix "
                "carries no coordinates. Point at the Space Ranger outs/ directory, or put the "
                "spatial/ folder next to this file"
            )
        return convert(
            "convert_visium_h5_spatial",
            {"counts_h5": str(p), "spatial_dir": str(spatial_dir), "output_path": output_path},
            "Join the Visium counts H5 to the tissue positions beside it",
        )
    if fmt == "merfish_expression":
        meta = _find_file(p.parent, ["cell_metadata.csv"])
        if meta is None:
            return blocked(f"no cell_metadata.csv beside {p.name}, and it holds the only coordinates")
        return convert(
            "convert_merfish",
            {"cell_by_gene_path": str(p), "cell_metadata_path": str(meta), "output_path": output_path},
            "Convert MERFISH expression + cell metadata to h5ad",
        )
    if fmt == "cosmx_data":
        # Either half of the pair can be the file the user named; the other is its slide's sibling.
        if _matches_marker(p.name, "metadata_file.csv"):
            expr_file, meta_file = _find_marker_sibling(p, "metadata_file.csv", "exprMat_file.csv"), p
        else:
            expr_file, meta_file = p, _find_marker_sibling(p, "exprMat_file.csv", "metadata_file.csv")
        if expr_file is None or meta_file is None:
            missing = "exprMat_file.csv" if expr_file is None else "metadata_file.csv"
            return blocked(
                "CosMx needs both exprMat_file.csv and metadata_file.csv from the same slide; there "
                f"is no {missing} for {p.name} in {p.parent}"
            )
        return convert(
            "convert_cosmx",
            {"expr_path": str(expr_file), "fov_positions_path": str(meta_file), "output_path": output_path},
            "Convert CosMx expression + FOV metadata to h5ad",
        )
    if fmt in ("h5ad_corrupt", "directory_with_h5ad"):
        # The run resolves the file inside the directory and then repairs-or-copies it, which is the
        # `h5ad` branch above applied to a file the user did not name. Naming `auto_convert` on the
        # *directory* was the part that could not run.
        named = diagnosis.get("h5ad_file")
        h5ad = (p / named) if (named and p.is_dir()) else p
        action, why = ("repair", "") if fmt == "h5ad_corrupt" else _h5ad_action(diagnosis)
        if action == "refuse":
            return blocked(f"{h5ad.name}: {why}. repair_spatial_h5ad cannot supply what is missing")
        if action == "repair":
            return convert(
                "repair_spatial_h5ad",
                {"h5ad_path": str(h5ad), "output_path": output_path},
                "Repair h5ad for MCP compatibility",
                action="repair",
            )
        return convert(
            "auto_convert",
            {"input_path": str(h5ad), "output_path": output_path},
            "h5ad is already MCP-compatible",
            action="copy",
        )

    return convert(
        "auto_convert",
        {"input_path": str(p), "output_path": output_path, **_generic_csv_coordinate_kwargs(diagnosis)},
        "Auto-detect and convert to h5ad",
    )


def _build_pipeline_steps(diagnosis: dict, input_path: str, output_path: str) -> list[dict[str, Any]]:
    """Build ordered pipeline steps from diagnosis."""
    steps: list[dict[str, Any]] = []
    fmt = diagnosis.get("detected_format", "unknown")
    images = diagnosis.get("images", {})
    # Detection may have re-rooted into `outs/`; the plan has to name the directory that was
    # actually diagnosed, or every path in it points at files that are one level away.
    p = Path(diagnosis.get("resolved_input_path") or input_path)

    # Step 1: Convert expression data
    if fmt == "visium_spaceranger":
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_visium_spaceranger",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {"spaceranger_dir": str(p), "output_path": output_path},
                "description": "Convert Space Ranger output to h5ad",
            }
        )
    elif fmt == "xenium_output":
        tx_file = _find_file(p, ["transcripts.parquet", "transcripts.csv.gz"])
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_xenium",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {"transcripts_path": str(tx_file) if tx_file else str(p), "output_path": output_path},
                "description": "Aggregate Xenium transcripts to cell-level h5ad",
            }
        )
    elif fmt == "merfish_vizgen":
        expr_file = _find_file(p, ["cell_by_gene.csv"])
        meta_file = _find_file(p, ["cell_metadata.csv"])
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_merfish",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {
                    "cell_by_gene_path": str(expr_file) if expr_file else "",
                    "cell_metadata_path": str(meta_file) if meta_file else "",
                    "output_path": output_path,
                },
                "description": "Convert MERFISH cell_by_gene + metadata to h5ad",
            }
        )
    elif fmt == "cosmx_output":
        expr_file, meta_file, paired = _find_marker_pair(p, ("exprMat_file.csv",), ("metadata_file.csv",))
        # A plan is copied and run. Two halves from different slides are a pair of paths that both
        # exist and convert without error into cells whose coordinates belong to other cells, so
        # the step publishes no paths at all and says which slide has to be chosen. The empty
        # string is the same "not resolved" convention the other branches use.
        description = "Convert CosMx expression + FOV metadata to h5ad"
        if expr_file and meta_file and not paired:
            description = (
                f"BLOCKED: '{expr_file.name}' and '{meta_file.name}' are from different CosMx slides "
                "and no slide here has both halves -- point at one slide's exprMat_file.csv"
            )
            expr_file = meta_file = None
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_cosmx",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {
                    "expr_path": str(expr_file) if expr_file else "",
                    "fov_positions_path": str(meta_file) if meta_file else "",
                    "output_path": output_path,
                },
                "description": description,
            }
        )
    elif fmt in ("stereoseq_gem", "stereoseq_directory"):
        gem_file = p if p.is_file() else _find_file(p, ["*.gem", "*.gem.gz", "*.gef"])
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_stereoseq",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {"gem_path": str(gem_file) if gem_file else str(p), "output_path": output_path},
                "description": "Bin Stereo-seq transcripts and create h5ad",
            }
        )
    elif fmt == "slideseq":
        dge_file, loc_file, paired = _find_slideseq_files(p)
        description = "Convert Slide-seq DGE + bead locations to h5ad"
        if dge_file and loc_file and not paired:
            description = (
                f"BLOCKED: '{dge_file.name}' and '{loc_file.name}' are from different Slide-seq pucks "
                "and no puck here has both halves -- bead barcodes do not carry across pucks"
            )
            dge_file = loc_file = None
        steps.append(
            {
                "step": 1,
                "action": "convert_expression",
                "function": "convert_slideseq",
                "module": "spatialomicsgym.tool.spatial_data_converter",
                "params": {
                    "dge_path": str(dge_file) if dge_file else "",
                    "bead_locations_path": str(loc_file) if loc_file else "",
                    "output_path": output_path,
                },
                "description": description,
            }
        )
    elif fmt == "h5ad":
        # Already h5ad: the run's own rule decides refuse / repair / copy (u22-spatial-pipeline-19).
        action, why = _h5ad_action(diagnosis)
        if action == "refuse":
            return [
                {
                    "step": 1,
                    "action": "cannot_convert",
                    "function": None,
                    "module": None,
                    "params": {},
                    "description": (
                        f"BLOCKED: detected h5ad, but it cannot be made analysis-ready here -- {why}. "
                        "repair_spatial_h5ad cannot supply what is missing"
                    ),
                }
            ]
        if action == "repair":
            steps.append(
                {
                    "step": 1,
                    "action": "repair",
                    "function": "repair_spatial_h5ad",
                    "module": "spatialomicsgym.tool.spatial_data_converter",
                    "params": {"h5ad_path": str(p), "output_path": output_path},
                    "description": "Repair h5ad for MCP compatibility",
                }
            )
        else:
            steps.append(
                {
                    "step": 1,
                    "action": "copy",
                    "function": "auto_convert",
                    "module": "spatialomicsgym.tool.spatial_data_converter",
                    "params": {"input_path": str(p), "output_path": output_path},
                    "description": "h5ad is already MCP-compatible",
                }
            )
    else:
        planned = _plan_conversion_step(fmt, p, output_path, diagnosis)
        if planned["action"] == "cannot_convert":
            # Nothing writes an h5ad, so there is no file to embed an image into and nothing for
            # `validate_spatial_h5ad` to open. A plan of one honest step beats three, two of which
            # name a path that will not exist.
            return [planned]
        steps.append(planned)

    step_num = 2

    # Step 2: Image embedding. Gate on `is True` — R-object inputs set found="unknown" (a truthy string)
    # which otherwise adds a bogus image-embed step.
    if images.get("found", False) is True:
        if fmt == "visium_spaceranger" and images.get("path"):
            steps.append(
                {
                    "step": step_num,
                    "action": "embed_images",
                    "function": "process_visium_images",
                    "module": "spatialomicsgym.tool.spatial_image_processor",
                    # The embedders no longer write over their input unless told to, and this step's
                    # whole job is to put the image in the plan's own output (hunt 2026-09-30, uT8-imaging-5).
                    "params": {"spatial_dir": images["path"], "h5ad_path": output_path, "output_path": output_path},
                    "description": "Embed Visium H&E images with scale factors",
                }
            )
        elif fmt == "xenium_output":
            img_files = images.get("files", [])
            if img_files:
                steps.append(
                    {
                        "step": step_num,
                        "action": "embed_images",
                        "function": "process_xenium_images",
                        "module": "spatialomicsgym.tool.spatial_image_processor",
                        "params": {
                            "morphology_path": str(p / img_files[0]),
                            "h5ad_path": output_path,
                            "output_path": output_path,
                            "channel": "composite",
                        },
                        "description": "Process Xenium fluorescence image and embed in h5ad",
                    }
                )
        elif fmt == "merfish_vizgen":
            mosaic, stain = _merfish_mosaic(images, p)
            if mosaic is not None:
                steps.append(
                    {
                        "step": step_num,
                        "action": "embed_images",
                        "function": "process_merfish_images",
                        "module": "spatialomicsgym.tool.spatial_image_processor",
                        "params": {
                            "mosaic_path": str(mosaic),
                            "h5ad_path": output_path,
                            "output_path": output_path,
                            "stain": stain,
                        },
                        "description": "Process MERFISH mosaic image and embed in h5ad",
                    }
                )
        elif fmt == "cosmx_output":
            steps.append(
                {
                    "step": step_num,
                    "action": "embed_images",
                    "function": "process_cosmx_fov_images",
                    "module": "spatialomicsgym.tool.spatial_image_processor",
                    # The composites are usually in `CellComposite/`; diagnosis recorded where.
                    "params": {
                        "composite_dir": images.get("path") or str(p),
                        "h5ad_path": output_path,
                        "output_path": output_path,
                    },
                    "description": "Stitch CosMx FOV images and embed in h5ad",
                }
            )
        else:
            img_files = images.get("files", [])
            if img_files:
                steps.append(
                    {
                        "step": step_num,
                        "action": "embed_images",
                        "function": "embed_image_in_h5ad",
                        "module": "spatialomicsgym.tool.spatial_image_processor",
                        "params": {
                            "h5ad_path": output_path,
                            "image_path": str(p / img_files[0]),
                            "output_path": output_path,
                        },
                        "description": "Embed image into h5ad",
                    }
                )
        step_num += 1

    # Final: Validation
    steps.append(
        {
            "step": step_num,
            "action": "validate",
            "function": "validate_spatial_h5ad",
            "module": "spatialomicsgym.tool.spatial_data_converter",
            "params": {"h5ad_path": output_path},
            "description": "Validate final h5ad for MCP tool compatibility",
        }
    )

    return steps


# ---------------------------------------------------------------------------
# Internal: Pipeline execution
# ---------------------------------------------------------------------------

# detected_format -> why nothing here can turn it into a spatial h5ad. These are *identified* and
# unsupported, which is a different answer from unidentified and leads somewhere different: stop
# looking for a converter, rather than stop guessing at the format.
_NO_CONVERTER: dict[str, str] = {
    "10x_mtx": (
        "no MTX converter ships here, and the triplet carries no coordinates. Read it with "
        "scanpy.read_10x_mtx in run_python_repl, attach obsm['spatial'] from wherever the "
        "coordinates live, write the h5ad, then validate_spatial_h5ad"
    ),
    "matrix_market": (
        "a bare matrix with no barcodes, features or coordinates. Read it with scipy.io.mmread in "
        "run_python_repl, supply obs/var names and obsm['spatial'], then validate_spatial_h5ad"
    ),
    "loom": (
        "no loom converter ships here. Read it with anndata.io.read_loom in run_python_repl, put "
        "the (x, y) column attributes into obsm['spatial'], then validate_spatial_h5ad"
    ),
    "zarr_store": (
        "no zarr converter ships here, and a SpatialData store is a container rather than one "
        "table -- the expression table is usually under tables/. Read it with anndata.io.read_zarr "
        "in run_python_repl, then validate_spatial_h5ad"
    ),
    "image_only": (
        "this is an image, not expression data, so there is nothing to convert. Convert the "
        "expression data separately, then attach this image with embed_image_in_h5ad"
    ),
    "stereoseq_gef": (
        "convert_stereoseq reads the text GEM and rejects the binary GEF. Produce the GEM with "
        f"`{_STEREOSEQ_GEF_TOOL}` (SAW writes both, so it may already be beside this file), then "
        "point convert_stereoseq at that"
    ),
    "xenium_cells": (
        "a Xenium cells table holds centroids and per-cell summaries, not gene counts. convert_xenium "
        "reads the transcripts table (transcripts.parquet or transcripts.csv.gz) from the same Xenium "
        "output; point it, or diagnose_spatial_data, at that output directory"
    ),
}

# Verdicts that mean the format was never identified, each with whatever concrete alternative exists
# for it. The diagnosis carries `unknown_format_recovery` for all three, so the refusal points there
# rather than repeating the route.
#
# None of these can convert today: `auto_convert` needs a recognised suffix, its directory branch is
# a subset of the Visium detection above it, and its tabular branch ends in `convert_generic_csv`,
# which reads coordinates only from columns literally named x and y -- a file that has those is
# detected as `generic_csv_with_coords`, never as `unknown_tabular`. So refusing here changes the
# message and not the outcome.
_UNIDENTIFIED_FORMATS = {
    "unknown": "",
    "unknown_tabular": (
        "convert_generic_csv can still read this table if its coordinates are in columns literally "
        "named x and y, or if you pass a separate coordinates file."
    ),
    "unknown_directory": "",
}

# Formats auto_convert resolves from the file itself, by suffix or by column names. Listing them is
# what keeps `auto_convert` a decision rather than a catch-all: anything else reaching the end of the
# dispatcher is refused under its own name instead of being handed to a detector that will answer
# "could not detect" about a format that was, in fact, detected.
_AUTO_CONVERTIBLE = {"xenium_transcripts", "parquet_transcripts", "generic_csv_with_coords"}


def _cannot_convert(fmt: str, reason: str) -> dict[str, Any]:
    """Refuse a conversion under the name the diagnosis just gave it.

    The fallback path answered ``Could not auto-detect format for: <path>`` -- a message that
    contradicts the diagnosis produced by the same call. Handed a format *and* its denial, the
    cheapest reading is that detection is unreliable, so the agent re-detects a format it was
    already told instead of acting on what is actually missing.
    """
    return {"status": "error", "message": f"Detected {fmt}, but it cannot be converted here: {reason}."}


def _find_spatial_dir(counts_h5: Path) -> Path | None:
    """The Space Ranger `spatial/` folder belonging to a loose counts H5, if it sits beside it.

    Only one that holds a positions file. A folder with nothing but scalefactors and images matched
    on the scalefactors, so the plan promised `convert_visium_h5_spatial` for a file whose diagnosis
    said there were no coordinates, and the run then failed in it (hunt 2026-09-30,
    u22-spatial-pipeline-20). The names are the ones `read_tissue_positions` reads.
    """
    candidate = counts_h5.parent / "spatial"
    positions = ("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet")
    if candidate.is_dir() and any((candidate / name).exists() for name in positions):
        return candidate
    return None


def _h5ad_action(diagnosis: dict) -> tuple[str, str]:
    """``("refuse", why)``, ``("repair", "")`` or ``("copy", "")`` for an h5ad already on disk.

    One rule for the plan and the run. The plan keyed on ``mcp_ready`` and the run on a "repair"
    substring in the steps, so they disagreed both ways: an h5ad with no coordinates anywhere was
    planned as a repair (which refuses) and run as a copy that then failed validation, and one
    missing only QC was planned as a copy and run as a repair (hunt 2026-09-30,
    u22-spatial-pipeline-19). What no repair can supply -- a ``CRITICAL`` step, or no expression
    matrix -- is a refusal; anything the repair fixes is repaired; only a file with nothing to fix
    is copied. Read from fields the diagnosis already publishes, so its report is unchanged.
    """
    steps = [str(s) for s in diagnosis.get("pipeline_steps") or []]
    hard = [s.split(":", 1)[1].strip() for s in steps if s.startswith("CRITICAL")]
    if (diagnosis.get("expression") or {}).get("found") is False:
        hard.insert(0, "missing expression matrix")
    if hard:
        return "refuse", "; ".join(hard)
    if any("repair" in s.lower() for s in steps):
        return "repair", ""
    return "copy", ""


def _convert_existing_h5ad(diagnosis: dict, h5ad: Path, output_path: str) -> dict[str, Any]:
    """Repair-or-copy an h5ad that is already on disk, whether it was named directly or found."""
    from spatialomicsgym.tool.spatial_data_converter import _copy_atomically, repair_spatial_h5ad

    action, why = _h5ad_action(diagnosis)
    if action == "refuse":
        return _cannot_convert(
            diagnosis.get("detected_format", "h5ad"),
            f"{h5ad.name}: {why}. repair_spatial_h5ad cannot supply what is missing, so convert the data from "
            "the original vendor output instead",
        )
    if action == "repair":
        return json.loads(repair_spatial_h5ad(str(h5ad), output_path))
    # Atomically, and not at all when the output is the input -- `shutil.copy2` raised SameFileError
    # there and wrote through a symlinked output into its target (u22-spatial-pipeline-7).
    if not _copy_atomically(h5ad, output_path):
        return {"status": "success", "output_path": output_path, "action": "none (output_path is the input)"}
    return {"status": "success", "output_path": output_path, "action": "copied (already valid)"}


def _generic_csv_coordinate_kwargs(diagnosis: dict) -> dict[str, str]:
    """The coordinate columns a generic table was diagnosed with, as convert_generic_csv's keywords.

    Only when they are not its defaults, so a plain x/y table plans and runs exactly as before
    (u22-spatial-pipeline-8).
    """
    if diagnosis.get("detected_format") != "generic_csv_with_coords":
        return {}
    columns = (diagnosis.get("coordinates") or {}).get("columns") or []
    if len(columns) != 2 or list(columns) == ["x", "y"]:
        return {}
    return {"x_col": str(columns[0]), "y_col": str(columns[1])}


def _file_identity(path: str) -> tuple[int, int] | None:
    """``(device, inode)`` of the file ``path`` names through any link, or ``None`` when there is none."""
    import os

    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino)


def _execute_conversion(diagnosis: dict, input_path: str, output_path: str) -> dict[str, Any]:
    """Execute the data conversion step.

    Detection names 23 formats; this dispatcher used to branch on 8 and hand the rest to
    `auto_convert`, whose own last line is ``Could not auto-detect format for: <path>``. So
    Slide-seq and a directory holding an h5ad -- both detected correctly, both with a working
    converter -- were refused in the language of a failed detection. Every format detection can name
    now leaves here either converted or refused *by that name*.
    """
    from spatialomicsgym.tool.spatial_data_converter import (
        auto_convert,
        convert_cosmx,
        convert_merfish,
        convert_r_object,
        convert_slideseq,
        convert_stereoseq,
        convert_visium_h5_spatial,
        convert_visium_spaceranger,
        convert_xenium,
    )

    fmt = diagnosis.get("detected_format", "unknown")
    # Same as in _build_pipeline_steps: convert from the directory detection actually read.
    p = Path(diagnosis.get("resolved_input_path") or input_path)

    if fmt in _UNIDENTIFIED_FORMATS:
        said = [
            f"Format was not identified ({fmt}), so there is no converter to pick.",
            _UNIDENTIFIED_FORMATS[fmt],
            "The diagnosis carries `unknown_format_recovery`: an ordered route that identifies the bytes first.",
        ]
        return {"status": "error", "message": " ".join(part for part in said if part)}
    if fmt in _NO_CONVERTER:
        return _cannot_convert(fmt, _NO_CONVERTER[fmt])

    try:
        if fmt == "r_object":
            return json.loads(convert_r_object(str(p), output_path))
        elif fmt == "visium_spaceranger":
            return json.loads(convert_visium_spaceranger(str(p), output_path))
        elif fmt == "xenium_output":
            tx = _find_file(p, ["transcripts.parquet", "transcripts.csv.gz"])
            return json.loads(convert_xenium(str(tx) if tx else str(p), output_path))
        elif fmt == "merfish_vizgen":
            expr = _find_file(p, ["cell_by_gene.csv"])
            meta = _find_file(p, ["cell_metadata.csv"])
            if not expr or not meta:
                return {"status": "error", "message": "MERFISH needs both cell_by_gene.csv and cell_metadata.csv"}
            return json.loads(convert_merfish(str(expr), str(meta), output_path))
        elif fmt == "cosmx_output":
            expr, meta, paired = _find_marker_pair(p, ("exprMat_file.csv",), ("metadata_file.csv",))
            if not expr or not meta:
                return {"status": "error", "message": "CosMx needs both exprMat_file.csv and metadata_file.csv"}
            if not paired:
                # Both halves are here, from different slides. Converting them would succeed and
                # produce coordinates for cells that are not in the matrix -- cell_ID restarts per
                # slide, so the join is silently wrong rather than empty. Name the slide to use.
                return _cannot_convert(
                    fmt,
                    f"'{expr.name}' and '{meta.name}' are from different CosMx slides and no slide in "
                    f"{p} has both halves. Point at one slide's exprMat_file.csv, or put its "
                    "metadata_file.csv beside it",
                )
            return json.loads(convert_cosmx(str(expr), str(meta), output_path))
        elif fmt in ("stereoseq_gem", "stereoseq_directory"):
            gem = p if p.is_file() else _find_file(p, ["*.gem", "*.gem.gz", "*.gef"])
            return json.loads(convert_stereoseq(str(gem) if gem else str(p), output_path))
        elif fmt == "slideseq":
            # Same lookups the plan published, so the run converts the files the plan named.
            dge, beads, paired = _find_slideseq_files(p)
            missing = [
                what
                for what, found in (
                    ("a DGE matrix (MappedDGEForR.csv, DGE.csv, *.digital_expression.txt.gz or *.tsv)", dge),
                    ("bead locations (BeadLocationsForR.csv or *_bead_locations.csv)", beads),
                )
                if not found
            ]
            if missing:
                # Slide-seq coordinates exist only in the bead file, so this is not recoverable by
                # trying harder -- name the file, because the user is the only one who can supply it.
                return _cannot_convert(fmt, f"{' and '.join(missing)} is missing from {p}")
            if not paired:
                # Bead barcodes are puck-specific: puck 08's beads share no barcode with puck 09's
                # DGE, so the merge drops every row and `convert_slideseq` reports an empty result
                # rather than a mismatch. Refuse while the two file names still say which pucks.
                return _cannot_convert(
                    fmt,
                    f"'{dge.name}' and '{beads.name}' are from different Slide-seq pucks and no puck in "
                    f"{p} has both halves. Bead barcodes do not carry across pucks, so this pair joins "
                    "to nothing -- convert one puck at a time",
                )
            return json.loads(convert_slideseq(str(dge), str(beads), output_path))
        elif fmt == "directory_with_h5ad":
            # Detection already opened this h5ad and reported its obsm and QC; refusing now would
            # throw that away and contradict a report saying the data is right there.
            named = diagnosis.get("h5ad_file")
            h5ad = (p / named) if named else _find_file(p, ["*.h5ad"])
            if h5ad is None or not h5ad.exists():
                return _cannot_convert(fmt, f"the h5ad reported in this directory is no longer readable at {h5ad}")
            return _convert_existing_h5ad(diagnosis, h5ad, output_path)
        elif fmt == "10x_h5_counts":
            spatial_dir = _find_spatial_dir(p)
            if spatial_dir is None:
                return _cannot_convert(
                    fmt,
                    f"no spatial/ folder with tissue positions sits beside {p.name}, and a counts "
                    "matrix carries no coordinates. Point at the Space Ranger outs/ directory, or "
                    "put the spatial/ folder next to this file",
                )
            return json.loads(convert_visium_h5_spatial(str(p), str(spatial_dir), output_path))
        elif fmt == "merfish_expression":
            # Pointing at cell_by_gene.csv is the same dataset as pointing at its directory, and
            # MERFISH coordinates are only in the metadata file beside it.
            meta = _find_file(p.parent, ["cell_metadata.csv"])
            if meta is None:
                return _cannot_convert(fmt, f"no cell_metadata.csv beside {p.name}, and it holds the only coordinates")
            return json.loads(convert_merfish(str(p), str(meta), output_path))
        elif fmt == "cosmx_data":
            # Either half of the CosMx pair can be the file the user named; the other is the sibling
            # from the same slide. Any `metadata_file.csv` in the directory is not good enough --
            # a multi-slide export holds several, and the wrong one describes different cells.
            if _matches_marker(p.name, "metadata_file.csv"):
                expr, meta = _find_marker_sibling(p, "metadata_file.csv", "exprMat_file.csv"), p
            else:
                expr, meta = p, _find_marker_sibling(p, "exprMat_file.csv", "metadata_file.csv")
            if expr is None or meta is None:
                missing = "exprMat_file.csv" if expr is None else "metadata_file.csv"
                return _cannot_convert(
                    fmt,
                    f"CosMx needs both exprMat_file.csv and metadata_file.csv from the same slide; "
                    f"there is no {missing} for {p.name} in {p.parent}",
                )
            return json.loads(convert_cosmx(str(expr), str(meta), output_path))
        elif fmt == "h5ad":
            return _convert_existing_h5ad(diagnosis, p, output_path)
        elif fmt == "h5ad_corrupt":
            # Detection reads h5ad backed; a file that fails there can still open in memory, so the
            # validate-then-repair route auto_convert already runs on an .h5ad is worth keeping --
            # what was wrong is that a *directory* holding the bad file was handed over instead.
            named = diagnosis.get("h5ad_file")
            h5ad = (p / named) if (named and p.is_dir()) else p
            try:
                return json.loads(auto_convert(str(h5ad), output_path))
            except Exception as exc:
                # It announced itself as an h5ad and then would not open. That is a different problem
                # from an unrecognised format, and only saying so points at the right fix.
                return _cannot_convert(fmt, f"{h5ad.name} could not be opened or repaired: {exc}")
        elif fmt in _AUTO_CONVERTIBLE:
            return json.loads(auto_convert(str(p), output_path, **_generic_csv_coordinate_kwargs(diagnosis)))
        else:
            # A format detection can name but nothing above claims. It should be unreachable -- the
            # coverage test walks every name the module emits -- but if a new one is added without a
            # decision, say so under that name rather than denying the diagnosis.
            return _cannot_convert(fmt, "no branch in the conversion dispatcher claims this format")
    except Exception as e:
        # Name the exception type: a bare KeyError stringifies to just its key ("'library_ids'"),
        # which a live agent relayed to the user as the entire explanation of a failed conversion.
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


def _execute_image_embedding(diagnosis: dict, h5ad_path: str) -> dict[str, Any]:
    """Execute image embedding step."""
    from spatialomicsgym.tool.spatial_image_processor import (
        embed_image_in_h5ad,
        process_cosmx_fov_images,
        process_merfish_images,
        process_visium_images,
        process_xenium_images,
    )

    fmt = diagnosis.get("detected_format", "")
    images = diagnosis.get("images", {})
    p = Path(diagnosis.get("resolved_input_path") or diagnosis.get("input_path", ""))

    # `output_path=h5ad_path` on every call: the embedders write `<stem>_image.h5ad` under the work
    # root unless told otherwise, and the pipeline reports h5ad_path as its result -- without it the
    # reported file carried no image (hunt 2026-09-30, uT8-imaging-5). Their write is atomic.
    try:
        if fmt == "visium_spaceranger" and images.get("path"):
            return json.loads(process_visium_images(images["path"], h5ad_path, output_path=h5ad_path))
        elif fmt == "xenium_output":
            img_files = images.get("files", [])
            if img_files:
                return json.loads(
                    process_xenium_images(str(p / img_files[0]), h5ad_path, channel="composite", output_path=h5ad_path)
                )
        elif fmt == "merfish_vizgen":
            mosaic, stain = _merfish_mosaic(images, p)
            if mosaic is not None:
                return json.loads(process_merfish_images(str(mosaic), h5ad_path, stain=stain, output_path=h5ad_path))
        elif fmt == "cosmx_output":
            return json.loads(process_cosmx_fov_images(images.get("path") or str(p), h5ad_path, output_path=h5ad_path))
        else:
            img_files = images.get("files", [])
            if img_files:
                return json.loads(embed_image_in_h5ad(h5ad_path, str(p / img_files[0]), output_path=h5ad_path))
        return {"status": "skipped", "reason": "no image files matched"}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


def _find_file(directory: Path, patterns: list[str]) -> Path | None:
    """Find first matching file in a directory, case-insensitively.

    Detection lowercases directory contents, but this resolver used to compare exact-case, so on a
    case-sensitive filesystem (Linux) a `Cell_by_gene.csv` was detected yet reported "file not found" at
    conversion time. Match on lowercased names so detection and conversion agree.

    Compression is matched the same way and for the same reason: `_diagnose_directory` now names a
    gzipped Vizgen export MERFISH, and the plan it publishes resolves both halves through here, so
    an exact-name lookup would answer None for `cell_metadata.csv.gz` and put an empty path in a
    step the user is meant to run. The literal name still wins where a directory holds both forms.
    """
    import fnmatch

    try:
        entries = list(directory.iterdir())
    except OSError:
        return None
    lower_map: dict[str, Path] = {}
    for e in entries:
        lower_map.setdefault(e.name.lower(), e)
    for e in entries:
        lower_map.setdefault(_uncompressed_name(e.name), e)
    for pattern in patterns:
        if "*" in pattern:
            pat = pattern.lower()
            for e in entries:
                if fnmatch.fnmatch(e.name.lower(), pat) or fnmatch.fnmatch(_uncompressed_name(e.name), pat):
                    return e
        else:
            hit = lower_map.get(pattern.lower()) or lower_map.get(_uncompressed_name(pattern))
            if hit is not None:
                return hit
    return None
