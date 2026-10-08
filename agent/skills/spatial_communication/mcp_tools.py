"""MCP tool knowledge base for spatial cell communication analysis."""

from __future__ import annotations

from typing import Any

SPATIAL_COMMUNICATION_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "commot": {
        "task": "cell_communication",
        "mcp_function": "commot_spatial_communication",
        "full_name": "COMMOT Spatial Communication",
        "description": "Infer spatially-resolved cell-cell communication using COMMOT's optimal transport framework.",
        "gpu": False,
        "priority": 1,
    },
    "ncem": {
        "task": "cell_communication",
        "mcp_function": "ncem_cell_communication",
        "full_name": "NCEM Cell Communication",
        "description": (
            "Estimate how neighbouring cell types shift gene expression with an NCEM-style linear model "
            "(sklearn Ridge on neighbour cell-type composition; the ncem package is not run)."
        ),
        "gpu": False,
        "priority": 2,
    },
    "deeplinc": {
        "task": "cell_communication",
        "mcp_function": "deeplinc_cell_interactions",
        "full_name": "DeepLinc Cell Interactions",
        "description": (
            "Score cell-type neighbourhood enrichment (observed/expected k-NN edges between cell types, 1.0 = "
            "random placement) with a label-permutation test; served under the DeepLinc name, the DeepLinc VGAE "
            "is not run and expression is not used."
        ),
        "gpu": False,
        "priority": 2,
    },
    "deeplinc_csv": {
        "task": "cell_communication",
        "mcp_function": "deeplinc_cell_interactions_csv",
        "full_name": "DeepLinc Cell Interactions (CSV)",
        "description": (
            "Same k-NN cell-type edge enrichment as deeplinc, from counts/coordinates/cell-type CSVs (the DeepLinc "
            "VGAE is not run)."
        ),
        "gpu": False,
        "priority": 3,
    },
    "mistyr": {
        "task": "cell_communication",
        "mcp_function": "mistyr_spatial_modeling",
        "full_name": "MISTy Spatial Modeling",
        "description": "Model each marker from the spot's own expression (intraview) and from neighbouring spots weighted by a Gaussian kernel (paraview) with MISTy random forests; reports per-view importances and the R2 gain.",
        "gpu": False,
        "priority": 2,
    },
    "neighborseq": {
        "task": "cell_communication",
        "mcp_function": "neighborseq_interaction_network",
        "full_name": "NeighborSeq Interaction Network",
        "description": (
            "Infer cell-cell interaction networks with Neighbor-seq: classify each cell or spot as a singlet or "
            "doublet of labelled cell types, then test each cell-type pair for enrichment."
        ),
        "gpu": False,
        "priority": 2,
    },
    "spaotsc": {
        "task": "cell_communication",
        "mcp_function": "spaotsc_run",
        "full_name": "SpaOTsc",
        "description": "Infer spatially-resolved intercellular communication using SpaOTsc's optimal transport.",
        "input_requirements": {"sc_reference": True},
        "gpu": False,
        "priority": 3,
    },
}
