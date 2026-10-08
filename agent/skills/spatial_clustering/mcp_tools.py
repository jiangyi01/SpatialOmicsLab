"""MCP tool knowledge base for spatial clustering."""

from __future__ import annotations

from typing import Any

SPATIAL_CLUSTERING_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "scanpy_spatial": {
        "task": "spatial_clustering",
        "mcp_function": "run_scanpy_spatial_domain",
        "full_name": "Scanpy Spatial Domain",
        "description": (
            "Cluster spots with Scanpy Leiden on an expression PCA kNN graph (spatial coordinates are not used "
            "in clustering); a non-spatial baseline for spatial domains."
        ),
        "gpu": False,
        "priority": 1,
    },
    "graphst": {
        "task": "spatial_clustering",
        "mcp_function": "graphst_spatial_clustering",
        "full_name": "GraphST Spatial Clustering",
        "description": "Perform graph self-supervised contrastive learning for spatial domain identification with GraphST.",
        "gpu": False,
        "priority": 2,
    },
    "stagate": {
        "task": "spatial_clustering",
        "mcp_function": "stagate_spatial_domains",
        "full_name": "STAGATE Spatial Domains",
        "description": "Identify spatial domains using STAGATE's graph attention autoencoder on spatial transcriptomics.",
        "gpu": False,
        "priority": 2,
    },
    "deepst": {
        "task": "spatial_clustering",
        "mcp_function": "deepst_identify_domains",
        "full_name": "DeepST Domain Identification",
        "description": "Identify spatial tissue domains using DeepST's deep learning graph neural network approach.",
        "gpu": False,
        "priority": 2,
    },
    "sedr": {
        "task": "spatial_clustering",
        "mcp_function": "run_sedr",
        "full_name": "SEDR",
        "description": (
            "SEDR variational-graph-autoencoder embedding of expression and a spatial kNN graph, then KMeans on it "
            "for spatial domains."
        ),
        "gpu": False,
        "priority": 2,
    },
    "miso": {
        "task": "spatial_clustering",
        "mcp_function": "run_miso",
        "full_name": "MISO",
        "description": "Cluster spots with MISO's multi-modal integration of expression and (optionally) H&E image features; without an image MISO uses no spatial coordinates, so the clusters are expression clusters.",
        "gpu": False,
        "priority": 2,
    },
    "precast": {
        "task": "spatial_clustering",
        "mcp_function": "precast_spatial_clustering",
        "full_name": "PRECAST Spatial Clustering",
        "description": "Perform embedding and clustering with PRECAST for multiple spatial transcriptomics slices jointly.",
        "gpu": False,
        "priority": 2,
    },
    "stlearn": {
        "task": "spatial_clustering",
        "mcp_function": "stlearn_spatial_clustering",
        "full_name": "stLearn Spatial Clustering",
        "description": "Cluster spatial transcriptomics data integrating morphology features using stLearn.",
        "input_requirements": {"images": True},
        "gpu": False,
        "priority": 2,
    },
    "cellcharter": {
        "task": "spatial_clustering",
        "mcp_function": "cellcharter_cluster_spatial_domains",
        "full_name": "CellCharter Spatial Domains",
        "description": "Cluster spatial domains using CellCharter's scalable spatial niche identification approach.",
        "gpu": False,
        "priority": 2,
    },
    "spaceflow": {
        "task": "spatial_clustering",
        "mcp_function": "spaceflow_spatial_domains",
        "full_name": "SpaceFlow Spatial Domains",
        "description": "Segment spatial domains with SpaceFlow's spatially regularized embedding, plus pseudo-spatial-time.",
        "gpu": False,
        "priority": 2,
    },
    "spacel_splane": {
        "task": "spatial_clustering",
        "mcp_function": "run_spacel_splane",
        "full_name": "SPACEL sPlane",
        "description": "Identify spatial domains on one slice with SPACEL's Splane, which clusters cell-type proportions (uns['celltypes'] or a one-hot of celltype_key).",
        "gpu": False,
        "priority": 3,
    },
    "iris": {
        "task": "spatial_clustering",
        "mcp_function": "run_iris",
        "full_name": "IRIS",
        "description": "Detect spatial domains with IRIS (reference-informed tissue segmentation); with a scRNA-seq reference it also writes per-spot cell-type proportions, without one it runs IRISfree on marker blocks derived from the spatial counts.",
        "gpu": False,
        "priority": 3,
    },
    "bass": {
        "task": "spatial_clustering",
        "mcp_function": "run_bass",
        "full_name": "BASS",
        "description": "Jointly perform cell type clustering and spatial domain detection using BASS Bayesian model.",
        "gpu": False,
        "priority": 3,
    },
    "stage": {
        "task": "spatial_clustering",
        "mcp_function": "stage_run",
        "full_name": "STAGE",
        "description": "Run STAGE to decode expression at new positions between measured spots (generation) or across simulated Slide-seq sections (3d_model); it does not segment domains, so cluster its output separately.",
        "gpu": False,
        "priority": 3,
    },
}
