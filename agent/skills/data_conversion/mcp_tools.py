"""MCP tool knowledge base for data format conversion."""

from __future__ import annotations

from typing import Any

DATA_CONVERSION_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "csv_to_h5ad": {
        "task": "data_conversion",
        "mcp_function": "convert_csv_to_h5ad",
        "full_name": "CSV to H5AD Converter",
        "description": "Convert CSV-formatted gene expression and metadata files to AnnData H5AD format.",
        "gpu": False,
        "priority": 1,
    },
    "h5ad_to_csv": {
        "task": "data_conversion",
        "mcp_function": "convert_h5ad_to_csv",
        "full_name": "H5AD to CSV Converter",
        "description": (
            "Export AnnData H5AD files to CSV format for gene expression matrix and metadata; cell_type_key picks "
            "the obs column written as celltypes.csv."
        ),
        "gpu": False,
        "priority": 1,
    },
    "h5ad_to_seurat": {
        "task": "data_conversion",
        "mcp_function": "convert_h5ad_to_seurat_rds",
        "full_name": "H5AD to Seurat RDS Converter",
        "description": "Convert AnnData H5AD files to Seurat RDS format for use in R-based workflows.",
        "gpu": False,
        "priority": 1,
    },
    "seurat_to_h5ad": {
        "task": "data_conversion",
        "mcp_function": "convert_seurat_rds_to_h5ad",
        "full_name": "Seurat RDS to H5AD Converter",
        "description": "Convert Seurat RDS objects to AnnData H5AD format for use in Python-based workflows.",
        "gpu": False,
        "priority": 1,
    },
}
