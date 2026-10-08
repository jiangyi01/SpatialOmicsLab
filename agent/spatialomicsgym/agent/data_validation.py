"""Pre-tool data readiness checking for the SpatialOmicsLab agent.

This module validates that input data is compatible with a target MCP tool
before execution.  It inspects h5ad files (or image files) against known
tool requirements and returns structured reports describing mismatches,
whether they are auto-fixable, and what actions to take.

Typical agent workflow:
    1. Agent selects a tool (e.g. ``run_cell2location``)
    2. Agent calls ``check_data_readiness(path, tool_name)``
    3. If issues are auto-fixable, agent calls ``auto_fix_data(path, issues)``
    4. Agent proceeds with tool execution on the validated/fixed file

Design notes:
    * ``anndata`` is imported lazily inside functions to avoid heavy startup cost.
    * No agent / LLM imports — this module is purely data-oriented.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
from pathlib import Path
from typing import Any

# Moved verbatim, re-exported under the same names: these feed the scored DATA-READINESS prompt, and
# test/test_obs_aliases_moved_out_of_the_agent.py pins that each name here IS the object there.
from spatialomicsgym.utils.obs_aliases import (  # noqa: F401  (re-exported)
    _CELL_TYPE_ALIASES,
    _CELL_TYPE_MAX_QUALIFIERS,
    _CELL_TYPE_NAME_CORES,
    _CELL_TYPE_NAME_QUALIFIERS,
    _COUNT_SAMPLE_BLOCKS,
    _COUNT_SAMPLE_ROWS,
    _INTEGER_TEXT_RE,
    _SPATIAL_COORD_OBS_PAIRS,
    _cell_type_name_distance,
    _count_sample_blocks,
    _find_cell_type_column,
    _find_spatial_coord_columns,
    _holds_cell_type_labels,
    _nonzero_value_sample,
    _normalise_obs_name,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Tool data requirements
# ---------------------------------------------------------------------------

TOOL_DATA_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "visualization": {
        # The loosest row in this table, deliberately. A plot DRAWS a result; it does not produce
        # one, so requiring what the producing tool required would refuse the very request the
        # plot exists to serve -- a proportion map does not need the single-cell reference the
        # deconvolution needed, and a QC report does not need coordinates at all.
        #
        # This row is also what stops a name from deciding a category: `plot_deconvolution`
        # resolved to `deconvolution` on its name alone (driven 2026-09-21) and was checked for a
        # reference it has no use for. The output-registry profile now types it, and this row is
        # what that type resolves to.
        #
        # Each plot family states its OWN prerequisites in the capability registry
        # (spatialomicsgym/viz/capabilities.py), checked against a profile of the actual dataset,
        # and a request that cannot be answered comes back naming the missing field and the call
        # that would produce it. That check is specific where this one cannot be.
        "required_format": "h5ad|csv",
        "required_obsm": [],
        "required_obs": [],
        "preferred_obs": ["cell_type", "cluster", "spatial_domain"],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "none imposed here; the plot's own prerequisites are declared per family in viz/capabilities.py",
    },
    "functional_enrichment": {
        # A ranked gene table (CSV/TSV) OR an h5ad; nothing spatial is required, so a check that
        # demands obsm["spatial"] here would refuse the one input the tool is for.
        "required_format": "csv|h5ad",
        "required_obsm": [],
        "required_obs": [],
        "preferred_obs": ["spatial_domain", "cell_type", "cluster"],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "gene symbols (not Ensembl IDs); for activity, counts are normalized automatically",
    },
    "spatial_clustering": {
        "required_format": "h5ad",
        "required_obsm": ["spatial"],
        "required_obs": [],
        "preferred_obs": ["cell_type", "cluster", "annotation"],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "basic QC and normalization if not done",
    },
    "svg_detection": {
        "required_format": "h5ad",
        "required_obsm": ["spatial"],
        "required_obs": [],
        "preferred_obs": [],
        "requires_raw_counts": True,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "raw counts preferred, tool handles normalization",
    },
    "deconvolution": {
        "required_format": "h5ad",
        "required_obsm": ["spatial"],
        "required_obs": [],
        "preferred_obs": ["cell_type"],
        "requires_raw_counts": True,
        "requires_sc_reference": True,
        "sc_reference_required_obs": [
            "CellType",
            "cell_type",
            "celltype",
            "class_label",
            "subclass_label",
            "cluster_label",
            "annotation",
            "cell_type_label",
        ],
        "requires_image": False,
        "preprocessing": "raw counts, sc reference must have cell type annotations",
    },
    "cell_communication": {
        "required_format": "h5ad",
        "required_obsm": ["spatial"],
        "required_obs": ["cell_type"],
        "preferred_obs": ["cluster"],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "must have cell type annotations in obs",
    },
    "spatial_alignment": {
        "required_format": "h5ad",
        "required_obsm": ["spatial"],
        "required_obs": [],
        "preferred_obs": [],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": False,
        "preprocessing": "multiple slices may be needed",
    },
    "image_segmentation": {
        "required_format": "image",
        "required_obsm": [],
        "required_obs": [],
        "preferred_obs": [],
        "requires_raw_counts": False,
        "requires_sc_reference": False,
        "requires_image": True,
        "preprocessing": "image must be valid TIFF/PNG",
    },
}

# ---------------------------------------------------------------------------
# 2. Tool → category mapping
# ---------------------------------------------------------------------------

#: Every key here must name a function some ``tools/*_mcp_server.py`` actually exposes -- a key that
#: matches nothing is never consulted, and the tool it was written for falls through to the
#: heuristics instead, silently. ``test/test_tool_category_map_has_not_drifted_from_the_portals.py``
#: pins that, because the drift is invisible: 28 keys had gone stale (renamed portals, tools that
#: were never shipped, duplicates of a correct entry) with nothing failing.
#:
#: Keep it *small*. This is the override, not the catalogue: :func:`resolve_category` consults
#: :mod:`spatialomicsgym.benchmarking.tool_output_registry` next, so a tool the registry already
#: describes does not belong here. Add an entry only where the registry is silent and the name-based
#: heuristics get it wrong.
TOOL_CATEGORY_MAP: dict[str, str] = {
    # spatial clustering
    "run_scanpy_spatial_domain": "spatial_clustering",
    "graphst_spatial_clustering": "spatial_clustering",
    "stagate_spatial_domains": "spatial_clustering",
    "cellcharter_cluster_spatial_domains": "spatial_clustering",
    "deepst_identify_domains": "spatial_clustering",
    "run_miso": "spatial_clustering",
    "stlearn_spatial_clustering": "spatial_clustering",
    "run_sedr": "spatial_clustering",
    "prost_pnn_domains": "spatial_clustering",
    "precast_spatial_clustering": "spatial_clustering",
    "run_bass": "spatial_clustering",
    "seurat_qc_cluster": "spatial_clustering",
    "run_spatialpca": "spatial_clustering",  # was spatialpca_clustering
    # SVG detection
    "hotspot_spatial_modules": "svg_detection",
    "somde_run": "svg_detection",
    "spatialde_run_svg": "svg_detection",
    "svgbit_run": "svg_detection",
    "spark_svg_detection": "svg_detection",
    "prost_index_svg": "svg_detection",
    "spagft_identify_svg": "svg_detection",
    # deconvolution
    "run_cell2location": "deconvolution",
    "tangram_map_sc_to_spatial": "deconvolution",
    "run_destvi": "deconvolution",
    "stride_deconvolution": "deconvolution",
    "spacexr_rctd_deconvolution": "deconvolution",
    "tacco_annotate": "deconvolution",
    "graphst_deconvolution": "deconvolution",
    "run_card": "deconvolution",
    "run_spotlight": "deconvolution",
    "run_stdeconvolve": "deconvolution",
    "ucdeconvolve_base": "deconvolution",
    "run_celldart": "deconvolution",
    "run_bulk2space": "deconvolution",
    "bayestme_deconvolution": "deconvolution",
    "starfysh_deconvolution": "deconvolution",
    "run_dstg": "deconvolution",
    "run_stdgcn": "deconvolution",
    "run_spatialscope": "deconvolution",
    "run_spatialdecon": "deconvolution",
    "spatialprompt_deconvolution": "deconvolution",
    # cell communication
    "ncem_cell_communication": "cell_communication",
    "squidpy_spatial_neighbors": "cell_communication",
    "squidpy_co_occurrence": "cell_communication",
    "squidpy_ripley": "cell_communication",
    # spatial alignment
    "paste2_partial_align": "spatial_alignment",
    "paste2_estimate_overlap": "spatial_alignment",
    "cast_align_slices": "spatial_alignment",
    "diagnose_3d_stack": "spatial_alignment",
    "inspect_3d_coordinates": "spatial_alignment",
    "list_aligner_adapters": "spatial_alignment",
    "explain_3d_contract": "spatial_alignment",
    "stalign_align_points": "spatial_alignment",
    "paste_pairwise_align": "spatial_alignment",
    "st_gears_reconstruct_3d": "spatial_alignment",  # was st_gears_alignment
    "spatialglue_integrate": "spatial_alignment",  # was spatialglue_alignment
    "spiral_integrate": "spatial_alignment",  # SPIRAL's other portal, spiral_align, the heuristics catch
    "run_spacel_scube": "spatial_alignment",  # SCube stacks slices; SPLANE, its other portal, clusters
    # image segmentation
    "run_cellpose_segmentation": "image_segmentation",
    "run_deepcell_segmentation": "image_segmentation",
    # functional enrichment (decoupler / gseapy): a gene LIST is the input, not a slide, so the
    # heuristics below have nothing to say about it and would leave it UNKNOWN.
    "run_pathway_enrichment": "functional_enrichment",
    "run_pathway_activity": "functional_enrichment",
}

#: What :func:`resolve_category` returns when nothing it consults recognises the tool.
#:
#: There used to be a default of ``spatial_clustering`` here instead. It was a guess presented as a
#: fact: 54 live portal functions reached it, and each was then checked against requirements written
#: for a different kind of tool -- four deconvolution tools never had ``requires_sc_reference``
#: checked, and every ``.rds``-consuming ``seurat_*`` function was refused for not being an h5ad.
#: :func:`check_data_readiness` treats this category as "no requirements on file", which is
#: informational and never blocks; see the branch there for why that is not the same as "ready".
UNKNOWN_CATEGORY = "unknown"

# The obs alias tables (``_CELL_TYPE_ALIASES``, ``_SPATIAL_COORD_OBS_PAIRS``), the two column finders and the
# spread-sample probe live in ``spatialomicsgym.utils.obs_aliases`` and are imported at the top of this module
# under the names they always had. They moved so the viewer can use them without importing the agent package.


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _registry_task_type(tool_name: str) -> str | None:
    """What :mod:`tool_output_registry` records that *tool_name* produces, if it knows.

    Imported lazily: the registry is a large module and the agent imports this one at load.
    """
    try:
        from spatialomicsgym.benchmarking.tool_output_registry import get_profile

        profile = get_profile(tool_name)
    except Exception:  # a missing or broken registry must not take the readiness check with it
        return None
    task_type = getattr(profile, "task_type", None) if profile is not None else None
    return str(task_type) if task_type else None


def resolve_category(tool_name: str) -> str:
    """Return the requirement category for *tool_name*.

    Four sources, in descending order of how much they actually know:

    1. :data:`TOOL_CATEGORY_MAP` -- an explicit decision someone made about this tool;
    2. :mod:`spatialomicsgym.benchmarking.tool_output_registry` -- what the tool is *recorded* as
       producing. It would cover tools created at runtime too, through ``register_dynamic_profile``,
       but nothing calls that hook yet (hunt 2026-09-30, u31-benchmarking-20), so a ``tools_user/``
       tool falls through to the heuristics;
    3. keyword heuristics on the name, which are a guess but a legible one;
    4. :data:`UNKNOWN_CATEGORY`.

    The registry sits ahead of the heuristics because a name is weak evidence and an output profile
    is strong: reading it first removed nine live contradictions, four of them deconvolution tools
    that the heuristics called ``spatial_clustering`` -- so their single-cell reference, the one
    input whose absence guarantees the run fails, was never checked.

    This is part of the public API — used by ``workflow_gates`` and other modules.
    """
    cat = TOOL_CATEGORY_MAP.get(tool_name)
    if cat is not None:
        return cat

    recorded = _registry_task_type(tool_name)
    if recorded:
        return recorded

    # Pattern-based fallback
    name = tool_name.lower()
    if "cluster" in name or "domain" in name:
        return "spatial_clustering"
    if "svg" in name or "spatially_variable" in name:
        return "svg_detection"
    if "deconv" in name or "proportion" in name or "cell2location" in name or "tangram" in name:
        return "deconvolution"
    if "commot" in name or "communication" in name or "interaction" in name:
        return "cell_communication"
    if "align" in name or "paste" in name or "moscot" in name:
        return "spatial_alignment"
    if "segment" in name or "cellpose" in name or "deepcell" in name:
        return "image_segmentation"

    return UNKNOWN_CATEGORY


def _has_raw_counts(adata: Any) -> bool:
    """Heuristic: check whether ``adata.X`` likely contains raw (integer) counts.

    Handles both in-memory and backed (``_CSRDataset``) sparse matrices. A sample with no nonzero
    value in it proves nothing either way, so it returns ``True`` rather than block: the same rule
    the size cap above follows, because ``requires_raw_counts`` is a *gate* and "we could not tell"
    must not read as "these are not counts".
    """
    import numpy as np

    try:
        values = _nonzero_value_sample(adata)
    except Exception:
        return False

    if values is None or values.size == 0:
        return True
    if np.issubdtype(values.dtype, np.integer):
        return bool(np.all(values >= 0))
    # Float matrix — check if all values are close to integers.
    try:
        return bool(np.allclose(values, np.round(values)) and np.all(values >= 0))
    except (NotImplementedError, TypeError):
        # Some sparse formats don't support allclose; fall back to dense comparison
        dense = np.asarray(values)
        return bool(np.allclose(dense, np.round(dense)) and np.all(dense >= 0))


def _validate_image_path(path: str) -> list[dict[str, str]]:
    """Return issues list for an image-based tool input."""
    p = Path(path)
    issues: list[dict[str, str]] = []
    if not p.exists():
        issues.append({"field": "file", "status": "missing", "fix": f"file not found: {path}"})
        return issues
    valid_suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    if p.suffix.lower() not in valid_suffixes:
        issues.append(
            {
                "field": "file_format",
                "status": "unsupported",
                "fix": f"expected image ({', '.join(sorted(valid_suffixes))}), got {p.suffix}",
            }
        )
    return issues


# ---------------------------------------------------------------------------
# 3. check_data_readiness
# ---------------------------------------------------------------------------


def check_data_readiness(
    h5ad_path: str,
    tool_name: str,
    sc_ref_path: str | None = None,
) -> dict[str, Any]:
    """Validate that data at *h5ad_path* is ready for *tool_name*.

    Args:
        h5ad_path: Path to the primary data file (h5ad or image).
        tool_name: The MCP tool spatialomicsgym_name (e.g. ``"run_cell2location"``).
        sc_ref_path: Optional path to a single-cell reference h5ad (for deconvolution).

    Returns:
        A dict with keys ``tool_name``, ``data_path``, ``is_ready``, ``issues``,
        ``auto_fixable``, and ``fix_actions``.
    """
    category = resolve_category(tool_name)
    reqs = TOOL_DATA_REQUIREMENTS.get(category)
    if reqs is None:
        # No requirements on file for this tool -- either nothing recognises it, or the registry
        # recorded a task type this table does not describe (``resolution``, say). That is a
        # statement about *us*, not about the data, so it must not block: ``pre_execution_gate``
        # turns ``is_ready: False`` plus a blocking issue into a refused run, and refusing a run
        # because we have no opinion is the failure this branch used to cause. The issue is still
        # reported, so the agent knows which checks did not happen.
        #
        # The one check that survives is the one that holds whatever the tool turns out to be: the
        # path has to exist. Nothing else can be asserted -- a tool we cannot categorise may well
        # take a CSV, an ``.rds`` or a directory, and the h5ad-suffix check below would reject it.
        if not Path(h5ad_path).exists():
            return {
                "tool_name": tool_name,
                "data_path": h5ad_path,
                "is_ready": False,
                "issues": [{"field": "file", "status": "missing", "fix": f"file not found: {h5ad_path}"}],
                "auto_fixable": False,
                "fix_actions": [],
            }
        return {
            "tool_name": tool_name,
            "data_path": h5ad_path,
            "is_ready": True,
            "issues": [
                {
                    "field": "category",
                    "status": "unknown_category",
                    "fix": (
                        f"no input requirements are recorded for '{tool_name}' (category '{category}'), "
                        "so its input was not checked beyond existing"
                    ),
                }
            ],
            "auto_fixable": False,
            "fix_actions": [],
        }

    issues: list[dict[str, str]] = []
    fix_actions: list[str] = []

    # --- Image tools ---
    if reqs["required_format"] == "image":
        img_issues = _validate_image_path(h5ad_path)
        return {
            "tool_name": tool_name,
            "data_path": h5ad_path,
            "is_ready": len(img_issues) == 0,
            "issues": img_issues,
            "auto_fixable": False,
            "fix_actions": [],
        }

    # --- h5ad tools ---
    data_path = Path(h5ad_path)
    if not data_path.exists():
        return {
            "tool_name": tool_name,
            "data_path": h5ad_path,
            "is_ready": False,
            "issues": [{"field": "file", "status": "missing", "fix": f"file not found: {h5ad_path}"}],
            "auto_fixable": False,
            "fix_actions": [],
        }

    if data_path.suffix.lower() not in (".h5ad",):
        issues.append(
            {
                "field": "file_format",
                "status": "unsupported",
                "fix": f"expected .h5ad file, got {data_path.suffix}",
            }
        )
        return {
            "tool_name": tool_name,
            "data_path": h5ad_path,
            "is_ready": False,
            "issues": issues,
            "auto_fixable": False,
            "fix_actions": [],
        }

    # Load in backed mode for speed. The context manager releases the HDF5 handle on every exit
    # below, exception paths included -- the previous explicit close ran only if the checks in
    # between all succeeded. An input with a .raw slot is cyclic, so an implicit release waits on
    # the garbage collector, and auto_fix_data overwrites this very path when output_path is None.
    from spatialomicsgym.utils.file_io import read_h5ad_backed

    with contextlib.ExitStack() as stack:
        try:
            adata = stack.enter_context(read_h5ad_backed(h5ad_path))
        except Exception as exc:
            return {
                "tool_name": tool_name,
                "data_path": h5ad_path,
                "is_ready": False,
                "issues": [{"field": "file", "status": "unreadable", "fix": f"cannot read h5ad: {exc}"}],
                "auto_fixable": False,
                "fix_actions": [],
            }

        obs_cols = list(adata.obs.columns)
        obsm_keys = list(adata.obsm.keys()) if adata.obsm is not None else []

        # Check required obsm keys (e.g. "spatial")
        for key in reqs.get("required_obsm", []):
            if key not in obsm_keys:
                coord_pair = _find_spatial_coord_columns(obs_cols)
                if key == "spatial" and coord_pair is not None:
                    issues.append(
                        {
                            "field": f"obsm.{key}",
                            "status": "missing_but_recoverable",
                            "fix": f"construct obsm['{key}'] from obs columns {coord_pair}",
                        }
                    )
                    fix_actions.append(f"construct obsm['{key}'] from obs columns {coord_pair[0]}, {coord_pair[1]}")
                else:
                    issues.append(
                        {
                            "field": f"obsm.{key}",
                            "status": "missing",
                            "fix": f"obsm['{key}'] is required but not found and cannot be inferred",
                        }
                    )

        # Check required obs columns
        for col in reqs.get("required_obs", []):
            if col not in obs_cols:
                # Try to find an alias
                if col == "cell_type":
                    alias = _find_cell_type_column(obs_cols)
                    if alias is not None:
                        issues.append(
                            {
                                "field": f"obs.{col}",
                                "status": "missing_but_recoverable",
                                "fix": f"rename obs column '{alias}' to '{col}'",
                            }
                        )
                        fix_actions.append(f"rename obs column '{alias}' to '{col}'")
                    else:
                        issues.append(
                            {
                                "field": f"obs.{col}",
                                "status": "missing",
                                "fix": f"obs['{col}'] is required — no suitable alias found in obs columns",
                            }
                        )
                else:
                    issues.append(
                        {
                            "field": f"obs.{col}",
                            "status": "missing",
                            "fix": f"obs['{col}'] is required but not present",
                        }
                    )

        # Check preferred obs (warnings, not blockers)
        for col in reqs.get("preferred_obs", []):
            if col not in obs_cols:
                # Only flag if no alias exists
                if col == "cell_type":
                    alias = _find_cell_type_column(obs_cols)
                    if alias is not None and alias != col:
                        issues.append(
                            {
                                "field": f"obs.{col}",
                                "status": "preferred_missing_but_alias_exists",
                                "fix": f"rename obs column '{alias}' to '{col}' (optional but recommended)",
                            }
                        )
                    elif alias is None:
                        issues.append(
                            {
                                "field": f"obs.{col}",
                                "status": "preferred_missing",
                                "fix": f"obs['{col}'] is preferred but not present (not blocking)",
                            }
                        )
                else:
                    issues.append(
                        {
                            "field": f"obs.{col}",
                            "status": "preferred_missing",
                            "fix": f"obs['{col}'] is preferred but not present (not blocking)",
                        }
                    )

        # Check raw counts requirement
        if reqs.get("requires_raw_counts", False):
            if not _has_raw_counts(adata):
                # Deconvolution tools generally handle normalized data or normalize internally,
                # so raw counts are preferred but not strictly required.
                raw_status = "preferred_missing" if category == "deconvolution" else "not_raw_counts"
                issues.append(
                    {
                        "field": "X",
                        "status": raw_status,
                        "fix": "adata.X does not appear to contain raw integer counts; check adata.raw or re-provide raw data",
                    }
                )

        # Check single-cell reference
        if reqs.get("requires_sc_reference", False):
            if sc_ref_path is None:
                issues.append(
                    {
                        "field": "sc_reference",
                        "status": "missing",
                        "fix": "a single-cell reference h5ad is required for deconvolution; provide sc_ref_path",
                    }
                )
            else:
                sc_path = Path(sc_ref_path)
                if not sc_path.exists():
                    issues.append(
                        {
                            "field": "sc_reference",
                            "status": "missing",
                            "fix": f"sc reference file not found: {sc_ref_path}",
                        }
                    )
                else:
                    # Only the read belongs inside the ``try``. Two things went wrong when the
                    # column check sat in there with it: the ``except`` reported "cannot read sc
                    # reference" for a file that had read fine, and -- because the close was the
                    # last statement of the ``try`` -- any raise above it returned while still
                    # holding an HDF5 read lock on the user's single-cell reference, which is the
                    # file the deconvolution worker opens next. ``read_h5ad_backed`` releases the
                    # handle however the block is left; see test/test_h5ad_readers_release_the_file.
                    try:
                        from spatialomicsgym.utils.file_io import read_h5ad_backed

                        with read_h5ad_backed(sc_ref_path) as sc_adata:
                            sc_obs_cols = list(sc_adata.obs.columns)
                    except Exception as exc:
                        issues.append(
                            {
                                "field": "sc_reference",
                                "status": "unreadable",
                                "fix": f"cannot read sc reference: {exc}",
                            }
                        )
                    else:
                        required_sc_obs = reqs.get("sc_reference_required_obs", [])
                        has_any = any(c in sc_obs_cols for c in required_sc_obs)
                        if not has_any and required_sc_obs:
                            issues.append(
                                {
                                    "field": "sc_reference.obs",
                                    "status": "missing_cell_type",
                                    "fix": (
                                        f"sc reference needs one of {required_sc_obs} in obs; "
                                        f"found columns: {sc_obs_cols[:20]}"
                                    ),
                                }
                            )

        # Check image requirement
        if reqs.get("requires_image", False):
            issues.append(
                {
                    "field": "image",
                    "status": "check_manually",
                    "fix": "this tool requires an image input — ensure a valid image path is provided separately",
                }
            )

        # Determine blocking issues (not just preferred/warnings)
        blocking = [
            i
            for i in issues
            if i["status"] not in ("preferred_missing", "preferred_missing_but_alias_exists", "check_manually")
        ]
        auto_fixable_statuses = {"missing_but_recoverable"}
        all_blocking_fixable = len(blocking) > 0 and all(i["status"] in auto_fixable_statuses for i in blocking)

        return {
            "tool_name": tool_name,
            "data_path": h5ad_path,
            "is_ready": len(blocking) == 0,
            "issues": issues,
            "auto_fixable": all_blocking_fixable,
            "fix_actions": fix_actions,
        }


# ---------------------------------------------------------------------------
# 4. auto_fix_data
# ---------------------------------------------------------------------------


def auto_fix_data(
    h5ad_path: str,
    issues: list[dict[str, str]],
    output_path: str | None = None,
) -> str:
    """Apply auto-fixes described in *issues* and save the corrected h5ad.

    Only fixes with status ``"missing_but_recoverable"`` are acted upon.  Other
    issues are logged as warnings.

    Args:
        h5ad_path: Path to the source h5ad file.
        issues: The ``issues`` list returned by :func:`check_data_readiness`.
        output_path: Where to save the fixed file.  Defaults to overwriting
            *h5ad_path* (a backup is created at ``<path>.bak``).

    Returns:
        The path to the fixed h5ad file.

    Raises:
        FileNotFoundError: If *h5ad_path* does not exist.
        ValueError: If the file cannot be read as h5ad.
    """
    import anndata as ad
    import numpy as np

    src = Path(h5ad_path)
    if not src.exists():
        raise FileNotFoundError(h5ad_path)

    try:
        adata = ad.read_h5ad(h5ad_path)
    except Exception as exc:
        raise ValueError(f"cannot read h5ad: {exc}") from exc

    applied: list[str] = []

    for issue in issues:
        status = issue.get("status", "")
        field = issue.get("field", "")

        if status != "missing_but_recoverable":
            if status not in ("preferred_missing", "preferred_missing_but_alias_exists", "check_manually"):
                logger.warning("Skipping non-auto-fixable issue: %s — %s", field, issue.get("fix", ""))
            continue

        # --- Fix: construct obsm['spatial'] from obs coordinate columns ---
        if field.startswith("obsm."):
            key = field.split(".", 1)[1]
            if key == "spatial":
                coord_pair = _find_spatial_coord_columns(list(adata.obs.columns))
                if coord_pair is not None:
                    cx, cy = coord_pair
                    try:
                        adata.obsm["spatial"] = np.column_stack(
                            [
                                adata.obs[cx].values.astype(float),
                                adata.obs[cy].values.astype(float),
                            ]
                        )
                    except (ValueError, TypeError) as exc:
                        # A matched coordinate column can hold non-numeric values (a stray
                        # header/label). ``astype(float)`` then raises ValueError -- downgrade
                        # to a logged "cannot fix" rather than letting it escape auto_fix_data,
                        # whose contract only raises FileNotFoundError / read-ValueError and
                        # would otherwise abort the whole fix pass.
                        logger.warning("Cannot construct obsm['spatial'] from obs['%s'], obs['%s']: %s", cx, cy, exc)
                    else:
                        applied.append(f"constructed obsm['spatial'] from obs['{cx}'], obs['{cy}']")
                else:
                    logger.warning("Cannot construct obsm['spatial'] — no coordinate columns found in obs.")

        # --- Fix: rename obs column alias → required name ---
        elif field.startswith("obs."):
            target_col = field.split(".", 1)[1]
            alias = _find_cell_type_column(list(adata.obs.columns))
            if alias is not None and alias != target_col:
                adata.obs[target_col] = adata.obs[alias]
                applied.append(f"copied obs['{alias}'] → obs['{target_col}']")
            else:
                logger.warning("Cannot auto-fix obs.%s — no suitable source column found.", target_col)

    if not applied:
        logger.info("No auto-fixes were applied.")
        return h5ad_path

    # Determine output path
    if output_path is None:
        # Create backup before overwriting
        backup = src.with_suffix(".h5ad.bak")
        shutil.copy2(src, backup)
        logger.info("Backup saved to %s", backup)
        dest = src
    else:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

    adata.write_h5ad(str(dest))
    logger.info("Fixed h5ad saved to %s (actions: %s)", dest, "; ".join(applied))
    return str(dest)


# ---------------------------------------------------------------------------
# 5. generate_data_readiness_prompt_section
# ---------------------------------------------------------------------------


def generate_data_readiness_prompt_section(
    h5ad_path: str,
    tool_name: str,
    sc_ref_path: str | None = None,
) -> str:
    """Generate a human-readable prompt section describing data readiness.

    This text can be injected into the agent's system/user prompt so the LLM
    is aware of data compatibility before generating tool-calling code.

    Args:
        h5ad_path: Path to the primary data file.
        tool_name: The MCP tool spatialomicsgym_name.
        sc_ref_path: Optional single-cell reference path.

    Returns:
        A multi-line string suitable for prompt injection.
    """
    report = check_data_readiness(h5ad_path, tool_name, sc_ref_path=sc_ref_path)

    category = resolve_category(tool_name)
    reqs = TOOL_DATA_REQUIREMENTS.get(category, {})

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append(f"DATA READINESS CHECK — tool: {tool_name} (category: {category})")
    lines.append("=" * 60)
    lines.append(f"Data path : {h5ad_path}")
    lines.append(f"Ready     : {'YES' if report['is_ready'] else 'NO'}")

    if reqs:
        lines.append("")
        lines.append("Tool requirements:")
        lines.append(f"  Format           : {reqs.get('required_format', 'n/a')}")
        lines.append(f"  Required obsm    : {reqs.get('required_obsm', [])}")
        lines.append(f"  Required obs     : {reqs.get('required_obs', [])}")
        lines.append(f"  Preferred obs    : {reqs.get('preferred_obs', [])}")
        lines.append(f"  Raw counts needed: {reqs.get('requires_raw_counts', False)}")
        lines.append(f"  SC reference     : {reqs.get('requires_sc_reference', False)}")
        lines.append(f"  Image required   : {reqs.get('requires_image', False)}")
        lines.append(f"  Preprocessing    : {reqs.get('preprocessing', 'none')}")

    if report["issues"]:
        lines.append("")
        lines.append("Issues found:")
        for i, issue in enumerate(report["issues"], 1):
            status_label = issue["status"].upper().replace("_", " ")
            lines.append(f"  {i}. [{status_label}] {issue['field']}")
            lines.append(f"     Fix: {issue['fix']}")

    if report["auto_fixable"]:
        lines.append("")
        lines.append("All blocking issues are auto-fixable.  Call auto_fix_data() before running the tool.")
    elif not report["is_ready"]:
        lines.append("")
        lines.append("Some issues require manual intervention before the tool can run.")

    if report["fix_actions"]:
        lines.append("")
        lines.append("Planned fix actions:")
        for action in report["fix_actions"]:
            lines.append(f"  - {action}")

    lines.append("=" * 60)
    return "\n".join(lines)
