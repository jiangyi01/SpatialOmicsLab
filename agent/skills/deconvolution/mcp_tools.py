"""MCP tool knowledge base for deconvolution and cell mapping."""

from __future__ import annotations

from typing import Any

DECONVOLUTION_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "cell2location": {
        "task": "deconvolution",
        "mcp_function": "run_cell2location",
        "full_name": "Cell2location",
        "description": "Deconvolve spatial transcriptomics spots using Cell2location's Bayesian model with single-cell reference.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 1,
    },
    "tangram": {
        "task": "deconvolution",
        "mcp_function": "tangram_map_sc_to_spatial",
        "full_name": "Tangram",
        "description": "Map single-cell RNA-seq data to spatial transcriptomics with Tangram's gradient-descent (PyTorch) mapping of reference cells or clusters to spots.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 1,
    },
    "spacexr_rctd": {
        "task": "deconvolution",
        "mcp_function": "spacexr_rctd_deconvolution",
        "full_name": "RCTD (spacexr)",
        "description": "Deconvolve cell type composition of spatial spots using RCTD's robust statistical framework.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 1,
    },
    "card": {
        "task": "deconvolution",
        "mcp_function": "run_card",
        "full_name": "CARD",
        "description": "Perform spatially informed deconvolution using CARD's conditional autoregressive model.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "destvi": {
        "task": "deconvolution",
        "mcp_function": "run_destvi",
        "full_name": "DestVI",
        "description": "Deconvolve spatial transcriptomics with continuous cell type variability using DestVI.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "stride": {
        "task": "deconvolution",
        "mcp_function": "stride_deconvolution",
        "full_name": "STRIDE",
        "description": "Deconvolve spatial transcriptomics spots using STRIDE's topic modeling approach.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "starfysh": {
        "task": "deconvolution",
        "mcp_function": "starfysh_deconvolution",
        "full_name": "Starfysh",
        "description": "Perform spatial deconvolution integrating tissue morphology with Starfysh's variational model.",
        "gpu": False,
        "priority": 2,
    },
    "bayestme": {
        "task": "deconvolution",
        "mcp_function": "bayestme_deconvolution",
        "full_name": "BayesTME",
        "description": "Deconvolve tumor microenvironment spatial transcriptomics using BayesTME's Bayesian framework.",
        "gpu": False,
        "priority": 2,
    },
    "tacco": {
        "task": "deconvolution",
        "mcp_function": "tacco_annotate",
        "full_name": "TACCO",
        "description": "Annotate and deconvolve spatial transcriptomics spots using TACCO's transfer annotation framework.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "ucdeconvolve": {
        "task": "deconvolution",
        "mcp_function": "ucdeconvolve_base",
        "full_name": "UCDeconvolve",
        "description": (
            "Deconvolve spatial spots with UCDBase, UCDeconvolve's pre-trained reference-free model, run on the "
            "UCD cloud API (needs a UCD token; the expression matrix is uploaded)."
        ),
        "gpu": False,
        "priority": 2,
    },
    "spatialprompt_deconv": {
        "task": "deconvolution",
        "mcp_function": "spatialprompt_deconvolution",
        "full_name": "SpatialPrompt Deconvolution",
        "description": (
            "Estimate per-spot cell-type proportions with SpatialPrompt's reference-guided deconvolution (spots "
            "simulated from a single-cell reference, a spatial-neighbourhood step, then k-nearest-neighbour regression)."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "celldart": {
        "task": "deconvolution",
        "mcp_function": "run_celldart",
        "full_name": "CellDART",
        "description": "Estimate cell type proportions in spatial spots using CellDART's domain adaptation approach.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "spotlight": {
        "task": "deconvolution",
        "mcp_function": "run_spotlight",
        "full_name": "SPOTlight",
        "description": "Deconvolve spatial transcriptomics data using SPOTlight's seeded NMF regression approach.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "stdeconvolve": {
        "task": "deconvolution",
        "mcp_function": "run_stdeconvolve",
        "full_name": "STdeconvolve",
        "description": "Deconvolve spatial transcriptomics spots using STdeconvolve's reference-free LDA approach.",
        "gpu": False,
        "priority": 2,
    },
    "spatialdecon": {
        "task": "deconvolution",
        "mcp_function": "run_spatialdecon",
        "full_name": "SpatialDecon",
        "description": "Deconvolve cell types in spatial transcriptomics using SpatialDecon's log-normal regression.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "stdgcn": {
        "task": "deconvolution",
        "mcp_function": "run_stdgcn",
        "full_name": "STDGCN",
        "description": "Deconvolve spatial transcriptomics using STDGCN's graph convolutional network framework.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "dstg": {
        "task": "deconvolution",
        "mcp_function": "run_dstg",
        "full_name": "DSTG-style GCN",
        "description": (
            "Deconvolve spatial spots with a DSTG-style two-layer graph convolutional network over a kNN graph of "
            "spots + reference cells (SpatialOmicsLab reimplementation; the upstream DSTG pseudo-spot pipeline is "
            "not run). Needs a labelled single-cell reference (cell_type_key)."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "smart": {
        "task": "deconvolution",
        "mcp_function": "run_smart",
        "full_name": "SMART",
        "description": (
            "Deconvolve spatial spots with SMART: a keyATM topic model whose topics are seeded by a marker gene "
            "list (one per cell type). Reference-free: needs marker_genes_csv and raw integer counts, no "
            "single-cell reference."
        ),
        "gpu": False,
        "priority": 3,
    },
    "gist": {
        "task": "deconvolution",
        "mcp_function": "run_gist",
        "full_name": "GIST",
        "description": (
            "Deconvolve spatial spots with GIST's base Bayesian model: a per-spot Stan (NUTS) regression on a "
            "scRNA-seq signature matrix. The image-guided prior is not used, so prior_lambda has no effect."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "cellpie": {
        "task": "deconvolution",
        "mcp_function": "run_cellpie",
        "full_name": "CellPie",
        "description": (
            "Reference-free CellPie intNMF topic model of a spatial slide: jointly factorises expression with "
            "histology image features in obsm['features']; without them it runs only on an explicit opt-in "
            "(allow_expression_only_fallback or allow_pca_image_fallback). Output columns are topics, not cell types."
        ),
        "gpu": False,
        "priority": 3,
    },
    "celloscope": {
        "task": "deconvolution",
        "mcp_function": "run_celloscope",
        "full_name": "Celloscope",
        "description": "Estimate per-spot cell-type proportions with Celloscope's Bayesian marker-gene model (MCMC), using marker genes derived from a single-cell reference.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "bulk2space": {
        "task": "deconvolution",
        "mcp_function": "run_bulk2space",
        "full_name": "Bulk2Space",
        "description": (
            "Estimate per-spot cell-type proportions by NNLS on the scRNA-seq reference's cell-type mean "
            "expression (upstream Bulk2Space is not run; no single-cell resolution)."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "spatialscope": {
        "task": "deconvolution",
        "mcp_function": "run_spatialscope",
        "full_name": "SpatialScope",
        "description": (
            "Spot-level cell-type proportions with SpatialScope's Stage-1 Cell-Type Identification (RCTD-style "
            "WarmStart on CPU); the single-cell Stage-2 is not run."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "scresolve": {
        "task": "deconvolution",
        "mcp_function": "run_scresolve",
        "full_name": "scResolve",
        "description": (
            "Score each spatial spot for reference cell-type marker signatures (scanpy rank_genes_groups + "
            "score_genes at the input spot resolution); upstream scResolve is not run and no resolution "
            "enhancement happens."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "redeconve": {
        "task": "deconvolution",
        "mcp_function": "redeconve_deconvolution",
        "full_name": "ReDeconve",
        "description": "Deconvolve spatial transcriptomics spots at single-cell resolution with Redeconve's quadratic programming against every reference cell.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "spacet_deconv": {
        "task": "deconvolution",
        "mcp_function": "spacet_deconvolution",
        "full_name": "SpaCET Deconvolution",
        "description": (
            "Deconvolve tumour spatial transcriptomics spots with SpaCET: the malignant fraction from the "
            "cancer type's own signatures (named in params.malignant_signature), then immune and stromal "
            "lineages and sub-lineages from built-in reference profiles; no single-cell reference."
        ),
        "gpu": False,
        "priority": 3,
    },
    "graphst_deconv": {
        "task": "deconvolution",
        "mcp_function": "graphst_deconvolution",
        "full_name": "GraphST Deconvolution",
        "description": "Deconvolve cell type proportions in spatial spots using GraphST's graph contrastive framework.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "scdot": {
        "task": "cell_spot_mapping",
        "mcp_function": "scdot_map_cells_to_spots",
        "full_name": "scDOT OT layer (entropic OT only)",
        "description": (
            "Map single cells to spatial spots by entropic optimal transport (one Sinkhorn solve on a fixed "
            "cosine cost through scDOT's OT layer); scDOT's NNLS deconvolution and joint training are not run."
        ),
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "cytospace": {
        "task": "cell_spot_mapping",
        "mcp_function": "run_cytospace",
        "full_name": "CytoSPACE",
        "description": "Optimally assign single cells to spatial spots using CytoSPACE's linear programming framework.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 2,
    },
    "celltrek": {
        "task": "cell_spot_mapping",
        "mcp_function": "celltrek_spatial_mapping",
        "full_name": "CellTrek",
        "description": "Map single cells back to spatial locations using CellTrek's cellular cartography approach.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
    "novosparc": {
        "task": "cell_spot_mapping",
        "mcp_function": "novosparc_reconstruct_spatial",
        "full_name": "novoSpaRc",
        "description": "Reconstruct spatial gene expression from single-cell data using novoSpaRc's optimal transport.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
}
