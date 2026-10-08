"""MCP tool knowledge base for spatial alignment and 3D reconstruction."""

from __future__ import annotations

from typing import Any

SPATIAL_ALIGNMENT_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "paste_pairwise": {
        "task": "spatial_alignment",
        "mcp_function": "paste_pairwise_align",
        "full_name": "PASTE Pairwise Alignment",
        "description": "Align pairs of spatial transcriptomics slices using PASTE's optimal transport formulation.",
        "gpu": False,
        "priority": 1,
    },
    "paste_center": {
        "task": "spatial_alignment",
        "mcp_function": "paste_center_align",
        "full_name": "PASTE Center Alignment",
        "description": "Compute a center slice alignment for multiple spatial transcriptomics sections using PASTE.",
        "gpu": False,
        "priority": 1,
    },
    # The two branches of the Phase-2 decision tree that had no tool behind them until
    # 2026-09-22. Priority 2: PASTE stays the default for full-overlap serial sections, and these
    # are what to reach for when the diagnosis says the overlap is partial or the deformation is
    # not rigid.
    "paste2": {
        "task": "spatial_alignment",
        "mcp_function": "paste2_partial_align",
        "full_name": "PASTE2 Partial Alignment",
        "description": (
            "Align serial sections that overlap only partially, with partial optimal transport. "
            "PASTE matches every spot to something, which drags the fit when the sections do not "
            "cover the same tissue."
        ),
        "gpu": False,
        "priority": 2,
    },
    "paste2_overlap": {
        "task": "spatial_alignment",
        "mcp_function": "paste2_estimate_overlap",
        "full_name": "PASTE2 Overlap Estimate",
        "description": "Measure how much each adjacent pair overlaps. Aligns nothing; writes only the table.",
        "gpu": False,
        "priority": 3,
    },
    "cast": {
        "task": "spatial_alignment",
        "mcp_function": "cast_align_slices",
        "full_name": "CAST Non-Rigid Alignment",
        "description": (
            "Graph-neural embedding plus affine and free-form registration of serial sections at "
            "single-cell resolution. The class-C tool that consumes AnnData rather than point clouds."
        ),
        "gpu": True,
        "priority": 2,
    },
    "stalign_points": {
        "task": "spatial_alignment",
        "mcp_function": "stalign_align_points",
        "full_name": "STalign Points Alignment",
        "description": "Align two point clouds with STalign LDDMM: rasterizes each into a density image and fits an affine transform plus a diffeomorphism by image matching (no landmarks or point correspondences).",
        "gpu": False,
        "priority": 1,
    },
    "stalign_image": {
        "task": "spatial_alignment",
        "mcp_function": "stalign_align_to_image",
        "full_name": "STalign Image Alignment",
        "description": "Align spatial transcriptomics sections to a reference image using STalign.",
        "input_requirements": {"images": True},
        "gpu": False,
        "priority": 2,
    },
    "gpsa": {
        "task": "spatial_alignment",
        "mcp_function": "gpsa_align_slices",
        "full_name": "GPSA Slice Alignment",
        "description": "Align spatial transcriptomics slices using GPSA's Gaussian process spatial alignment.",
        "gpu": False,
        "priority": 2,
    },
    "spiral_align": {
        "task": "spatial_alignment",
        "mcp_function": "spiral_align",
        "full_name": "SPIRAL Alignment",
        "description": (
            "Map the second of two spatial transcriptomics sections into the first one's frame: SPIRAL "
            "integration, then a per-cluster fused Gromov-Wasserstein mapping (SPIRAL's CoordAlignment is "
            "not run; spots outside clusters both sections share are left unplaced)."
        ),
        "gpu": False,
        "priority": 2,
    },
    "spiral_integrate": {
        "task": "spatial_alignment",
        "mcp_function": "spiral_integrate",
        "full_name": "SPIRAL Integration",
        "description": (
            "Integrate two or more spatial transcriptomics slices into one batch-corrected embedding with "
            "SPIRAL; the slices need not be aligned, and their coordinates are not registered."
        ),
        "gpu": False,
        "priority": 2,
    },
    "slat": {
        "task": "spatial_alignment",
        "mcp_function": "slat_align_slices",
        "full_name": "SLAT Slice Alignment",
        "description": (
            "Align two spatial transcriptomics slices with scSLAT: a graph neural network embeds both slices and "
            "each spot of the smaller slice is paired with its most similar spot in the larger one. Writes a "
            "matching table; no coordinates are moved."
        ),
        "gpu": False,
        "priority": 2,
    },
    "stacker": {
        "task": "spatial_alignment",
        "mcp_function": "stacker_register",
        "full_name": "STACKer Registration",
        "description": "Register one tissue-section image to another with ANTsPy (affine or SyN) or a caller-supplied trained VoxelMorph network; writes a warped image and transforms in pixel space, not spot coordinates.",
        "input_requirements": {"images": True},
        "gpu": False,
        "priority": 2,
    },
    "spatrio": {
        "task": "spatial_alignment",
        "mcp_function": "spatrio_align_multiomics",
        "full_name": "SpatRio Multiomics Alignment",
        "description": (
            "Map single cells (raw RNA counts sharing gene names with the slice; optional second-modality "
            "embedding in obsm['reduction']) onto the spots of one spatial slice with SpaTrio's fused "
            "Gromov-Wasserstein optimal transport. Writes a long (spot, cell, value) transport-plan CSV, "
            "not coordinates."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    # The diagnosis that decides whether any of the above should run at all. Priority 0: on a
    # multi-section object this is the first call, because aligning a stack that is already
    # aligned moves coordinates that were right, and aligning a deformed stack with a rigid tool
    # produces a confident wrong answer.
    "spatial3d_diagnose": {
        "task": "spatial_alignment",
        "mcp_function": "diagnose_3d_stack",
        "full_name": "Serial-section alignment diagnosis",
        "description": (
            "Classify a stack of serial sections as already aligned, rigidly misaligned or "
            "non-rigidly deformed, with the measured value and calibrated threshold behind every "
            "criterion. Writes a report and modifies nothing."
        ),
        "gpu": False,
        "priority": 0,
    },
    "spatial3d_inspect": {
        "task": "spatial_alignment",
        "mcp_function": "inspect_3d_coordinates",
        "full_name": "3D coordinate inspection",
        "description": "List an object's coordinate keys, its 3D frames and the contract rules it breaks. Read-only.",
        "gpu": False,
        "priority": 0,
    },
    "spatial3d_contract": {
        "task": "spatial_alignment",
        "mcp_function": "explain_3d_contract",
        "full_name": "3D coordinate contract",
        "description": "The 3D coordinate keys, roles, units and rules, so a caller need not infer them. Read-only.",
        "gpu": False,
        "priority": 0,
    },
    "spatial3d_adapters": {
        "task": "spatial_alignment",
        "mcp_function": "list_aligner_adapters",
        "full_name": "Aligner output adapters",
        "description": (
            "Where each shipped alignment tool leaves its answer, whether the original coordinates "
            "survive its output, and whether it exposed a seed. Read-only."
        ),
        "gpu": False,
        "priority": 0,
    },
    "st_gears": {
        "task": "3d_reconstruction",
        "mcp_function": "st_gears_reconstruct_3d",
        "full_name": "ST-GEARS 3D Reconstruction",
        "description": "Reconstruct 3D tissue structures from serial spatial transcriptomics sections using ST-GEARS.",
        "gpu": False,
        "priority": 2,
    },
}
