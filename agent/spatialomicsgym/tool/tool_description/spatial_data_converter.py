description = [
    {
        "name": "auto_convert",
        "description": "Auto-detect spatial transcriptomics data format and convert to MCP-compatible h5ad. "
        "Handles: 10x Visium (Space Ranger), Xenium, MERFISH, Slide-seq, Stereo-seq, CosMx, "
        "R objects (.rds Seurat/SCE, .rda/.RData), "
        "STARmap, seqFISH, generic CSV, and existing h5ad files. The output h5ad will have "
        "raw counts in .X, spatial coordinates in obsm['spatial'], and QC metrics in .obs. "
        "Two-file formats are routed by the keyword naming the second file: cell_metadata_path (MERFISH), "
        "fov_positions_path (CosMx), bead_locations_path (Slide-seq), starmap_coords_path (STARmap/seqFISH), "
        "spatial_dir (a 10x .h5 counts file). A keyword the chosen converter does not take is listed under "
        "ignored_parameters in the report.",
        "required_parameters": [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to input file or directory (auto-detected format)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path where the output h5ad file will be saved",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "cell_metadata_path",
                "type": "str",
                "description": "Path to cell metadata CSV (required for MERFISH format)",
                "default": None,
            },
            {
                "name": "fov_positions_path",
                "type": "str",
                "description": "Path to FOV positions CSV (required for CosMx format)",
                "default": None,
            },
            {
                "name": "bead_locations_path",
                "type": "str",
                "description": "Path to the bead locations CSV (Slide-seq; input_path is the DGE matrix)",
                "default": None,
            },
            {
                "name": "starmap_coords_path",
                "type": "str",
                "description": "Path to the cell coordinates CSV (STARmap/seqFISH; input_path is the expression "
                "matrix). A z column kept by name goes to obsm['spatial_3d']",
                "default": None,
            },
            {
                "name": "spatial_dir",
                "type": "str",
                "description": "Path to the Space Ranger spatial/ folder (when input_path is a 10x .h5 counts file "
                "kept apart from it)",
                "default": None,
            },
            {
                "name": "coords_path",
                "type": "str",
                "description": "Path to coordinates CSV (for generic CSV format)",
                "default": None,
            },
            {
                "name": "sep",
                "type": "str",
                "description": "Delimiter of a generic CSV/TSV (detected from the file when omitted)",
                "default": None,
            },
            {
                "name": "bin_size",
                "type": "int",
                "description": "Bin size for Stereo-seq spatial binning (Stereo-seq only)",
                "default": 50,
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell/spot to keep (not applied to an existing h5ad, an R "
                "object or Space Ranger output)",
                "default": 5,
            },
        ],
    },
    {
        "name": "validate_spatial_h5ad",
        "description": "Validate that an h5ad file meets MCP spatial tool requirements. "
        "Checks for: count matrix in .X, spatial coordinates in obsm['spatial'], "
        "QC metrics, and unique var_names. Returns a JSON report with issues.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to the h5ad file to validate",
                "default": None,
            }
        ],
        "optional_parameters": [],
    },
    {
        "name": "repair_spatial_h5ad",
        "description": "Repair an h5ad file to meet MCP spatial tool requirements. "
        "Fixes: moves coordinates from obs to obsm['spatial'], computes QC metrics, "
        "deduplicates var_names, ensures sparse matrix format.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to the input h5ad file to repair",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": (
                    "Output path for repaired h5ad (default: <stem>_repaired.h5ad under the work root's "
                    "repaired/ folder, never beside the input)"
                ),
                "default": None,
            }
        ],
    },
    {
        "name": "convert_visium_spaceranger",
        "description": "Convert 10x Visium Space Ranger output directory to MCP-compatible h5ad. "
        "Reads filtered_feature_bc_matrix and spatial/ directory.",
        "required_parameters": [
            {
                "name": "spaceranger_dir",
                "type": "str",
                "description": "Path to Space Ranger output directory",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [],
    },
    {
        "name": "convert_visium_h5_spatial",
        "description": "Convert 10x Visium H5 counts + spatial directory to MCP-compatible h5ad. "
        "For split Space Ranger output where counts H5 and spatial/ folder are separate.",
        "required_parameters": [
            {
                "name": "counts_h5",
                "type": "str",
                "description": "Path to filtered_feature_bc_matrix.h5",
                "default": None,
            },
            {
                "name": "spatial_dir",
                "type": "str",
                "description": "Path to the spatial/ directory",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [],
    },
    {
        "name": "convert_xenium",
        "description": "Convert 10x Xenium transcript-level data to cell-level MCP-compatible h5ad. "
        "Aggregates transcripts into cell-by-gene matrix with cell centroid coordinates.",
        "required_parameters": [
            {
                "name": "transcripts_path",
                "type": "str",
                "description": "Path to transcripts.csv.gz or transcripts.parquet",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "cell_id_col",
                "type": "str",
                "description": "Column name for cell IDs",
                "default": "cell_id",
            },
            {
                "name": "gene_col",
                "type": "str",
                "description": "Column name for gene/feature names",
                "default": "feature_name",
            },
            {
                "name": "x_col",
                "type": "str",
                "description": "Column name for x coordinates",
                "default": "x_location",
            },
            {
                "name": "y_col",
                "type": "str",
                "description": "Column name for y coordinates",
                "default": "y_location",
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell",
                "default": 5,
            },
            {
                "name": "min_qv",
                "type": "float",
                "description": "Minimum transcript decoding quality (qv) to count; 20 matches Xenium's own "
                "cell_feature_matrix. None counts every transcript",
                "default": 20,
            },
        ],
    },
    {
        "name": "convert_merfish",
        "description": "Convert MERFISH / Vizgen data (cell_by_gene.csv + cell_metadata.csv) "
        "to MCP-compatible h5ad. Removes blank/negative control probes.",
        "required_parameters": [
            {
                "name": "cell_by_gene_path",
                "type": "str",
                "description": "Path to cell_by_gene.csv (cells x genes count matrix)",
                "default": None,
            },
            {
                "name": "cell_metadata_path",
                "type": "str",
                "description": "Path to cell_metadata.csv (with cell coordinates)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "x_col",
                "type": "str",
                "description": "Column name for x coordinates",
                "default": "center_x",
            },
            {
                "name": "y_col",
                "type": "str",
                "description": "Column name for y coordinates",
                "default": "center_y",
            },
            {
                "name": "cell_id_col",
                "type": "str",
                "description": "Column in cell_metadata.csv holding the cell ID. "
                "If None, the first column (or the index) is used.",
                "default": None,
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell",
                "default": 5,
            },
        ],
    },
    {
        "name": "convert_slideseq",
        "description": "Convert Slide-seq / Slide-seqV2 data (DGE matrix + bead locations) to MCP-compatible h5ad.",
        "required_parameters": [
            {
                "name": "dge_path",
                "type": "str",
                "description": "Path to digital gene expression matrix (TSV/CSV)",
                "default": None,
            },
            {
                "name": "bead_locations_path",
                "type": "str",
                "description": "Path to bead locations file (bead_id, x, y)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per bead",
                "default": 5,
            }
        ],
    },
    {
        "name": "convert_stereoseq",
        "description": "Convert a Stereo-seq text GEM file (.gem/.gem.gz) to MCP-compatible h5ad. "
        "Bins transcripts into spatial bins and creates a count matrix. "
        "Does NOT accept the binary .gef; run `geftools gef2gem` on one first.",
        "required_parameters": [
            {
                "name": "gem_path",
                "type": "str",
                "description": "Path to .gem or .gem.gz file",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "bin_size",
                "type": "int",
                "description": "Bin size in coordinate units for spatial binning",
                "default": 50,
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per bin",
                "default": 5,
            },
        ],
    },
    {
        "name": "convert_cosmx",
        "description": "Convert Nanostring CosMx SMI data (expression matrix + FOV positions) "
        "to MCP-compatible h5ad. Removes negative control probes.",
        "required_parameters": [
            {
                "name": "expr_path",
                "type": "str",
                "description": "Path to exprMat_file.csv (cells x genes)",
                "default": None,
            },
            {
                "name": "fov_positions_path",
                "type": "str",
                "description": "Path to metadata_file.csv (cell positions and FOV info)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "x_col",
                "type": "str",
                "description": "Column name for global x pixel coordinate",
                "default": "CenterX_global_px",
            },
            {
                "name": "y_col",
                "type": "str",
                "description": "Column name for global y pixel coordinate",
                "default": "CenterY_global_px",
            },
            {
                "name": "cell_id_col",
                "type": "str",
                "description": "Column in metadata_file.csv holding the cell ID. AtoMx exports sometimes "
                "name it 'cell_id' or 'cell'.",
                "default": "cell_ID",
            },
            {
                "name": "fov_col",
                "type": "str",
                "description": "Column holding the field-of-view number, kept in obs so per-FOV batch "
                "effects can be modelled later.",
                "default": "fov",
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell",
                "default": 5,
            },
        ],
    },
    {
        "name": "convert_starmap",
        "description": "Convert STARmap data (expression matrix + 2D/3D coordinates) to MCP-compatible h5ad. "
        "Also works for seqFISH and any format with separate expression + coordinates files.",
        "required_parameters": [
            {
                "name": "expr_path",
                "type": "str",
                "description": "Path to expression matrix (cells x genes CSV)",
                "default": None,
            },
            {
                "name": "coords_path",
                "type": "str",
                "description": "Path to coordinates file (cells x [x,y] or [x,y,z] CSV)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell",
                "default": 5,
            }
        ],
    },
    {
        "name": "convert_seqfish",
        "description": "Convert seqFISH / seqFISH+ data (expression matrix + cell positions CSV) "
        "to MCP-compatible h5ad. Same interface as convert_starmap.",
        "required_parameters": [
            {
                "name": "expr_path",
                "type": "str",
                "description": "Path to expression matrix (cells x genes CSV)",
                "default": None,
            },
            {
                "name": "coords_path",
                "type": "str",
                "description": "Path to cell positions file (cells x [x,y] CSV)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell",
                "default": 5,
            }
        ],
    },
    {
        "name": "convert_generic_csv",
        "description": "Convert generic CSV/TSV expression matrix (+ optional coordinates file) "
        "to MCP-compatible h5ad. Handles any tabular spatial transcriptomics data.",
        "required_parameters": [
            {
                "name": "expr_path",
                "type": "str",
                "description": "Path to expression matrix CSV/TSV (cells x genes)",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "coords_path",
                "type": "str",
                "description": "Path to coordinates CSV (if separate from expression file)",
                "default": None,
            },
            {
                "name": "x_col",
                "type": "str",
                "description": "Column name for x coordinates",
                "default": "x",
            },
            {
                "name": "y_col",
                "type": "str",
                "description": "Column name for y coordinates",
                "default": "y",
            },
            {
                "name": "transpose",
                "type": "bool",
                "description": "Transpose matrix (genes x cells -> cells x genes)",
                "default": False,
            },
            {
                "name": "sep",
                "type": "str",
                "description": "Delimiter character",
                "default": ",",
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell/spot",
                "default": 5,
            },
        ],
    },
    {
        "name": "convert_r_object",
        "description": "Convert R objects (.rds, .rda, .RData) to MCP-compatible h5ad. "
        "Supports Seurat objects (v3/v4/v5) with automatic extraction of counts, metadata, "
        "spatial coordinates, dimensionality reductions (PCA/UMAP), and Visium images. "
        "Also handles plain dgCMatrix sparse matrices, dense matrices, and data.frames. "
        "Runs an R subprocess for extraction, then assembles the h5ad in Python. The R must have "
        "Seurat: the project's provisioned R environments are tried first, then Rscript on PATH.",
        "required_parameters": [
            {
                "name": "r_file_path",
                "type": "str",
                "description": "Path to .rds or .rda/.RData file",
                "default": None,
            },
            {
                "name": "output_path",
                "type": "str",
                "description": "Path for the output h5ad file",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "assay",
                "type": "str",
                "description": "Seurat assay to extract (default 'RNA'; auto-falls back to 'Spatial' or first available)",
                "default": "RNA",
            },
            {
                "name": "slot",
                "type": "str",
                "description": "Seurat slot/layer: 'counts', 'data', or 'scale.data'",
                "default": "counts",
            },
            {
                "name": "object_name",
                "type": "str",
                "description": "For .rda files: name of the object to extract (auto-selects Seurat if not specified)",
                "default": None,
            },
            {
                "name": "rscript_path",
                "type": "str",
                "description": "Path to an Rscript binary that has Seurat. If None, the project's "
                "provisioned R environments are tried in turn and then the bare name on PATH; set "
                "it when R lives somewhere else on this machine.",
                "default": None,
            },
            {
                "name": "timeout",
                "type": "int",
                "description": "Seconds the R extraction may run (default 300, or SOG_R_CONVERT_TIMEOUT_SECONDS); "
                "raise it for a large Seurat object",
                "default": None,
            },
        ],
    },
]
