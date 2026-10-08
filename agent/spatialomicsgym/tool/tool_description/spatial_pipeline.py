description = [
    {
        "name": "diagnose_spatial_data",
        "description": "FIRST STEP for any spatial transcriptomics task. Scans a file or directory and "
        "diagnoses the data format, platform, completeness, and MCP tool compatibility. "
        "Detects: 10x Visium, Xenium, MERFISH, CosMx, Slide-seq, Stereo-seq, generic CSV, "
        "and existing h5ad (STARmap/seqFISH expression + coordinates pairs are not detected: "
        "convert those with convert_starmap). Reports what expression data, spatial "
        "coordinates, and images are present, what is missing, and what conversion steps "
        "are needed. Always call this before running any spatial analysis MCP tools.",
        "required_parameters": [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to a file or directory containing spatial transcriptomics data",
                "default": None,
            }
        ],
        "optional_parameters": [],
    },
    {
        "name": "plan_spatial_pipeline",
        "description": "Generate a step-by-step executable pipeline plan to convert spatial data "
        "to MCP-compatible h5ad. Returns ordered steps with specific function names, "
        "parameters, and descriptions. Call diagnose_spatial_data first for a quick check, "
        "or call this for a full execution plan.",
        "required_parameters": [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to spatial data (file or directory)",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Target path for the final MCP-compatible h5ad",
                "default": "./spatial_output.h5ad",
            },
        ],
    },
    {
        "name": "run_spatial_pipeline",
        "description": "One-call solution: diagnose data, convert to h5ad, embed images, and validate. "
        "Automatically detects format, runs appropriate converter, embeds available histology "
        "or fluorescence images, and validates MCP compatibility. Returns path to a ready-to-use "
        "h5ad file that works with all spatial MCP tools (GraphST, Cell2Location, stLearn, etc.).",
        "required_parameters": [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to spatial data (file or directory)",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Target path for output h5ad",
                "default": "./spatial_output.h5ad",
            },
            {
                "name": "skip_images",
                "type": "bool",
                "description": "Skip image embedding for faster processing",
                "default": False,
            },
        ],
    },
]
