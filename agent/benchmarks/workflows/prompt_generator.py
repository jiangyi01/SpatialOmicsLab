"""Generate task-appropriate prompts for SpatialOmicsLab-driven MCP smoke tests.

Two prompt modes are supported:
- **direct**: Scripted prompts that explicitly name the tool and its parameters.
- **full**: Open-ended prompts that describe the biological task without naming the tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Dataset catalogue
# ---------------------------------------------------------------------------

#: ``test/...`` entries are repository-root paths; every other entry (``benchmarks/...``) is read
#: against the agent tree -- see :func:`get_dataset_for_tool`.
_DATASET_MAP: dict[str, dict[str, str]] = {
    "default": {
        "h5ad": "test/test_data/mini_spatial.h5ad",
    },
    "deconvolution_with_ref": {
        "h5ad": "test/test_data/mini_spatial.h5ad",
        "sc_reference": "test/test_data/mini_sc_ref.h5ad",
    },
    "image_histology": {
        "h5ad": "test/test_data/mini_spatial.h5ad",
        "histology_image": "test/test_data/he_image.tif",
    },
    "image_segmentation": {
        "image": "test/test_data/synthetic_cells_2ch.tif",
    },
    "visium_format": {
        "h5_path": "test/test_data/visium_mock/filtered_feature_bc_matrix.h5",
    },
    "gem_format": {
        "gem_path": "test/test_data/mock_data.gem",
    },
    "istar_format": {
        "data_prefix": "test/test_data/istar_data",
    },
    "xfuse_format": {
        "data_dir": "test/test_data/xfuse_input",
    },
    # Slide-seqV2 datasets (bead-based; no histology images available)
    "slideseq_clustering": {
        "h5ad": "benchmarks/benchmark_data/Slide-seqV2_standardized/slide_seqV2_aorta/Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad",
    },
    "slideseq_deconvolution": {
        "h5ad": "benchmarks/benchmark_data/Slide-seqV2_standardized/slide_seqV2_aorta/Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad",
        "sc_reference": "benchmarks/benchmark_data/Slide-seqV2_standardized/scRNA_seq_vascular_tissue/Reference_data/Standard_h5ad/spatial_transcriptomics.h5ad",
    },
    "slideseq_svg": {
        "h5ad": "benchmarks/benchmark_data/Slide-seqV2_standardized/slide_seqV2_aorta/Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad",
    },
}

# ---------------------------------------------------------------------------
# Tool → dataset-category mappings
# ---------------------------------------------------------------------------

_IMAGE_TOOLS: dict[str, str] = {
    "run_miso": "image_histology",
    "stalign_align_to_image": "image_histology",
    "starfysh_deconvolution": "image_histology",
    "run_cellpose_segmentation": "image_segmentation",
    "run_deepcell_segmentation": "image_segmentation",
    "run_clustermap": "image_segmentation",
}

# Every name in these tables is a ``spatialomicsgym_name`` in MCP_server/mcp_config.yaml. Half of
# them were not -- run_tangram, run_rctd, run_spacexr, run_stride, run_tacco, run_stereoscope,
# run_istar, run_xfuse -- so the real tools fell through to "default" and their smoke prompts
# carried no reference or input data (hunt 2026-09-30, u33a-bench-runner-25).
# test/test_the_smoke_prompts_name_tools_that_exist.py keeps them pointing at declared tools.

# Tools that require histology images and therefore cannot be used with
# Slide-seqV2 data (bead-based capture with no associated H&E images).
_SLIDESEQ_INCOMPATIBLE_TOOLS: set[str] = {
    "deepst_identify_domains",
    "run_miso",  # image mode requires histology
    "istar_full_pipeline",
    "istar_check_dataset",
    "xfuse_run",
    "xfuse_spatial_analysis",
    "xfuse_build_config",
}

# Tools that need a single-cell reference, for callers that classify without the parameter list.
# With the list, :func:`_needs_reference` decides from the tool's own declaration instead.
_DECONV_REF_TOOLS: set[str] = {
    "run_cell2location",
    "tangram_map_sc_to_spatial",
    "run_destvi",
    "spacexr_rctd_deconvolution",
    "run_spotlight",
    "run_card",
    "stride_deconvolution",
    "tacco_annotate",
    "run_celldart",
    "graphst_deconvolution",
    "spatialprompt_deconvolution",
    "run_cytospace",
    "run_spatialscope",
    "run_bulk2space",
}

#: The parameter names a tool declares for its single-cell reference.
_REFERENCE_PARAMS: frozenset[str] = frozenset(
    {
        "sc_h5ad",
        "sc_h5ad_path",
        "scrna_h5ad",
        "sc_data",
        "sc_data_path",
        "sc_counts_csv",
        "sc_expr_path",
        "ref_counts_csv",
        "scrna_df_path",
    }
)

_SPECIAL_FORMAT_TOOLS: dict[str, str] = {
    "istar_full_pipeline": "istar_format",
    "istar_check_dataset": "istar_format",
    "xfuse_run": "xfuse_format",
}


def _needs_reference(tool_params: dict[str, Any] | None) -> bool:
    """Whether the tool's declared parameters include a reference it cannot run without.

    An optional one (``run_iris``'s ``ref_counts_csv`` selects a supervised mode the clustering
    benchmark does not use) does not count.
    """
    for name in _REFERENCE_PARAMS & set(tool_params or {}):
        spec = (tool_params or {}).get(name)
        if (
            not isinstance(spec, dict)
            or spec.get("required")
            or ("default" not in spec and spec.get("required") is None)
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def classify_tool(
    tool_name: str,
    tool_params: dict[str, Any] | None = None,
    *,
    dataset_name: str | None = None,
) -> str:
    """Return the dataset category appropriate for *tool_name*.

    The classification is determined by checking, in order:
    1. Dataset-specific overrides (e.g. Slide-seqV2 routing).
    2. Special-format tools (iStar, xFuse).
    3. Image tools (histology or segmentation).
    4. Tools that require a single-cell reference (by name, or by a required reference
       parameter in *tool_params*).
    5. Parameter-based heuristics (``h5_path`` -> visium, ``gem_path`` -> gem).
    6. Falls back to ``"default"``.

    Parameters
    ----------
    tool_name:
        Canonical snake_case tool name (e.g. ``"tangram_map_sc_to_spatial"``).
    tool_params:
        Optional mapping of parameter names expected by the tool.  Used for
        heuristic detection of visium / gem formats when the tool is not in
        any explicit list.
    dataset_name:
        Optional registry dataset name (e.g. ``"slide_seqv2_aorta_domain"``).
        When provided and the dataset belongs to Slide-seqV2, the function
        routes to the appropriate ``slideseq_*`` category and rejects
        image-dependent tools.

    Returns
    -------
    str
        One of the keys in :data:`_DATASET_MAP`.

    Raises
    ------
    ValueError
        If *tool_name* is incompatible with the requested Slide-seqV2 dataset
        (image-dependent tools have no histology to work with).
    """
    # ---- Slide-seqV2 dataset routing ----------------------------------------
    if dataset_name and "slide_seq" in dataset_name.lower():
        if tool_name in _SLIDESEQ_INCOMPATIBLE_TOOLS:
            raise ValueError(
                f"Tool {tool_name!r} requires histology images and is incompatible "
                f"with Slide-seqV2 dataset {dataset_name!r} (no H&E images)."
            )
        # Determine task-specific Slide-seq category
        if "deconv" in dataset_name:
            return "slideseq_deconvolution"
        if "svg" in dataset_name:
            return "slideseq_svg"
        return "slideseq_clustering"

    # ---- Original classification logic (MERFISH / Visium / generic) ---------
    if tool_name in _SPECIAL_FORMAT_TOOLS:
        return _SPECIAL_FORMAT_TOOLS[tool_name]

    if tool_name in _IMAGE_TOOLS:
        return _IMAGE_TOOLS[tool_name]

    if tool_name in _DECONV_REF_TOOLS or _needs_reference(tool_params):
        return "deconvolution_with_ref"

    if tool_params:
        if "h5_path" in tool_params:
            return "visium_format"
        if "gem_path" in tool_params:
            return "gem_format"

    return "default"


def get_dataset_for_tool(
    tool_name: str,
    tool_params: dict[str, Any] | None = None,
    project_root: str | Path | None = None,
    *,
    dataset_name: str | None = None,
) -> dict[str, str]:
    """Return a dict of fully-resolved file paths for the tool's test dataset.

    Each returned path is absolute, anchored at *project_root* (defaults to
    the agent tree this module ships in), except the ``test/`` fixtures, which
    are anchored at that tree's repository root.  An ``output_dir`` entry is
    always included.

    Parameters
    ----------
    tool_name:
        Canonical snake_case tool name.
    tool_params:
        Optional parameter mapping forwarded to :func:`classify_tool`.
    project_root:
        Agent tree (``<checkout>/agent``).  ``benchmarks/`` paths in
        :data:`_DATASET_MAP` are resolved relative to it, ``test/`` paths
        relative to its repository root.
    dataset_name:
        Optional registry dataset name forwarded to :func:`classify_tool` for
        platform-aware routing (e.g. Slide-seqV2 datasets).

    Returns
    -------
    dict[str, str]
        Mapping of parameter names to absolute path strings.
    """
    from benchmarks.config.benchmark_config import repository_root

    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    repo_root = repository_root(root)
    category = classify_tool(tool_name, tool_params, dataset_name=dataset_name)
    template = _DATASET_MAP.get(category, _DATASET_MAP["default"])

    resolved: dict[str, str] = {}
    for key, rel_path in template.items():
        base = repo_root if rel_path.startswith("test/") else root
        resolved[key] = str((base / rel_path).resolve())

    # Include dataset_name in the output path when provided, so results from
    # different datasets (e.g. Visium vs Slide-seq) do not collide.
    if dataset_name:
        resolved["output_dir"] = str(
            (root / "benchmarks" / "results" / "benchmark_smoke_test" / tool_name / dataset_name / "output").resolve()
        )
    else:
        resolved["output_dir"] = str(
            (root / "benchmarks" / "results" / "benchmark_smoke_test" / tool_name / "output").resolve()
        )
    return resolved


# ---------------------------------------------------------------------------
# Data-readiness instructions
# ---------------------------------------------------------------------------

_DATA_CHECK_PREAMBLE = """\
Before running the tool, first inspect the input data:
1. Load the h5ad file and check its obs columns, obsm keys, uns keys, and shape.
2. Verify the data has the required fields for this tool.
3. If the data is missing expected columns (e.g., 'cell_type', 'cluster', 'CellType'), \
check for alternative column names (e.g., 'annotation', 'louvain', 'leiden') and rename them as needed.
4. If the data needs preprocessing (normalization, log-transform, etc.), apply it.
5. Only proceed to run the tool after confirming data compatibility."""

# Tool-category-specific additional guidance
_CATEGORY_CHECKS: dict[str, str] = {
    "deconvolution_with_ref": (
        "- The single-cell reference h5ad must contain a 'CellType' column in obs. "
        "If absent, look for 'cell_type', 'celltype', 'annotation', or 'cluster' and rename to 'CellType'.\n"
        "- Both spatial and reference datasets should share a common gene space; "
        "verify that var_names overlap sufficiently."
    ),
    "image_histology": (
        "- Ensure the h5ad object has spatial coordinates in obsm['spatial'].\n"
        "- Verify the histology image file exists and is a valid image format (TIFF/PNG/JPEG)."
    ),
    "image_segmentation": (
        "- Verify the input image is a multi-channel TIFF with the expected number of channels.\n"
        "- Check image dtype (uint8 / uint16) and rescale if necessary."
    ),
    "slideseq_clustering": (
        "- This is Slide-seqV2 bead-based data (no histology images available).\n"
        "- Ensure the h5ad object has spatial coordinates in obsm['spatial'].\n"
        "- Expression data may be normalized (float), not raw counts; check and handle accordingly.\n"
        "- Ground truth cell-type labels are in obs['cell_type']; "
        "also check 'author_cell_type' or 'Cluster'."
    ),
    "slideseq_deconvolution": (
        "- This is Slide-seqV2 bead-based data (no histology images available).\n"
        "- The single-cell reference uses 'cell_type' (not 'CellType') as its annotation column; "
        "rename to 'CellType' if the tool requires it.\n"
        "- Both spatial and reference datasets should share a common gene space; "
        "verify that var_names overlap sufficiently.\n"
        "- Expression data may be normalized (float), not raw counts."
    ),
    "slideseq_svg": (
        "- This is Slide-seqV2 bead-based data (no histology images available).\n"
        "- Ensure the h5ad object has spatial coordinates in obsm['spatial'].\n"
        "- No curated SVG ground truth exists; evaluation uses cross-tool consensus.\n"
        "- Expression data may be normalized (float), not raw counts."
    ),
    "default": (
        "- Ensure the h5ad object has spatial coordinates in obsm['spatial'].\n"
        "- Check that obs contains clustering or cell-type annotations; "
        "look for 'cluster', 'louvain', 'leiden', 'cell_type', or 'annotation'."
    ),
}


def generate_data_check_instructions(
    tool_name: str,
    dataset_category: str,
) -> str:
    """Produce tool-specific data-readiness instructions.

    Parameters
    ----------
    tool_name:
        Canonical snake_case tool name.
    dataset_category:
        The dataset category string (a key of :data:`_DATASET_MAP`), as
        returned by :func:`classify_tool`.

    Returns
    -------
    str
        A multi-line instruction block that can be appended to any prompt.
    """
    extra = _CATEGORY_CHECKS.get(dataset_category, _CATEGORY_CHECKS["default"])
    return f"{_DATA_CHECK_PREAMBLE}\n\nTool-specific checks for `{tool_name}` ({dataset_category}):\n{extra}"


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def generate_direct_prompt(
    tool_name: str,
    description: str,
    params: dict[str, Any],
    dataset: dict[str, str],
    *,
    dataset_name: str | None = None,
) -> str:
    """Generate a scripted prompt that explicitly names the tool and parameters.

    Parameters
    ----------
    tool_name:
        Canonical tool name.
    description:
        One-line human-readable description of what the tool does.
    params:
        Tool parameter schema (names -> metadata or default values).
    dataset:
        Resolved dataset paths as returned by :func:`get_dataset_for_tool`.
    dataset_name:
        Optional registry dataset name forwarded to :func:`classify_tool`.

    Returns
    -------
    str
        A prompt string ready to be sent to the SpatialOmicsLab agent.
    """
    lines: list[str] = [
        f"Run the tool `{tool_name}` with the following parameters:",
        "",
    ]
    for key, value in dataset.items():
        lines.append(f"- {key}: {value}")

    # Include any extra params that are not already covered by the dataset
    for key, value in params.items():
        if key not in dataset:
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append(f"Tool description: {description}")
    lines.append("")

    # Data-readiness pre-check instructions
    category = classify_tool(tool_name, params, dataset_name=dataset_name)
    lines.append(generate_data_check_instructions(tool_name, category))

    return "\n".join(lines)


def generate_full_prompt(
    tool_name: str,
    description: str,
    params: dict[str, Any],
    dataset: dict[str, str],
    *,
    dataset_name: str | None = None,
) -> str:
    """Generate an open-ended prompt describing the task without naming the tool.

    The resulting prompt reads like a natural user request so the agent must
    autonomously decide which tool to invoke.

    Parameters
    ----------
    tool_name:
        Canonical tool name (used only for internal reference, not included
        verbatim in the prompt).
    description:
        One-line human-readable description of what the tool does.
    params:
        Tool parameter schema.
    dataset:
        Resolved dataset paths as returned by :func:`get_dataset_for_tool`.
    dataset_name:
        Optional registry dataset name forwarded to :func:`classify_tool`.

    Returns
    -------
    str
        A prompt string ready to be sent to the SpatialOmicsLab agent.
    """
    # Build a data-file summary
    data_lines: list[str] = []
    for key, value in dataset.items():
        if key == "output_dir":
            continue
        data_lines.append(f"  - {key}: `{value}`")

    data_section = "\n".join(data_lines) if data_lines else "  (no specific input files)"

    # Data-readiness pre-check instructions
    category = classify_tool(tool_name, params, dataset_name=dataset_name)
    data_check = generate_data_check_instructions(tool_name, category)

    prompt = (
        f"I have the following data files:\n"
        f"{data_section}\n\n"
        f"I would like to {description.rstrip('.').lower()}.\n"
        f"Please save the results to `{dataset.get('output_dir', 'output/')}`.\n\n"
        f"{data_check}"
    )
    return prompt


def generate_prompt(
    tool_name: str,
    description: str,
    params: dict[str, Any],
    dataset: dict[str, str],
    mode: str = "direct",
    *,
    dataset_name: str | None = None,
) -> str:
    """Dispatch to the appropriate prompt generator based on *mode*.

    Parameters
    ----------
    tool_name:
        Canonical tool name.
    description:
        One-line human-readable description of what the tool does.
    params:
        Tool parameter schema.
    dataset:
        Resolved dataset paths as returned by :func:`get_dataset_for_tool`.
    mode:
        Either ``"direct"`` (scripted) or ``"full"`` (open-ended).
    dataset_name:
        Optional registry dataset name forwarded to the underlying prompt
        generators for platform-aware data-check instructions.

    Returns
    -------
    str
        The generated prompt string.

    Raises
    ------
    ValueError
        If *mode* is not ``"direct"`` or ``"full"``.
    """
    if mode == "direct":
        return generate_direct_prompt(tool_name, description, params, dataset, dataset_name=dataset_name)
    if mode == "full":
        return generate_full_prompt(tool_name, description, params, dataset, dataset_name=dataset_name)
    raise ValueError(f"Unknown prompt mode {mode!r}. Expected 'direct' or 'full'.")
