description = [
    {
        "name": "diagnose_transcriptomics_data",
        "description": "FIRST STEP for any transcriptomics task. Diagnoses ANY transcriptomics data — spatial "
        "(Visium, Xenium, MERFISH, CosMx, Slide-seq, Stereo-seq; STARmap/seqFISH expression + coordinates "
        "pairs are not detected -- convert them with convert_starmap), "
        "single-cell RNA-seq (10x Chromium, Smart-seq2, Drop-seq), and bulk RNA-seq. "
        "Detects data type, platform, format, completeness, and recommended analysis workflow. "
        "For spatial data, also reports MCP tool compatibility and conversion pipeline. "
        "For scRNA-seq data, reports suitability as a deconvolution reference. "
        "Always call this before any analysis to understand the data.",
        "required_parameters": [
            {
                "name": "input_path",
                "type": "str",
                "description": "Path to a file or directory containing transcriptomics data",
                "default": None,
            }
        ],
        "optional_parameters": [],
    },
    {
        "name": "recommend_analysis_tools",
        "description": "Given a spatial transcriptomics h5ad file and analysis goals, recommend the best MCP tools. "
        "Inspects the data (size, spatial coords, images, reference availability) and ranks "
        "30+ tools across 10 categories: spatial clustering (GraphST, STAGATE, CellCharter, DeepST, MISO, etc.), "
        "SVG detection (Hotspot, SpatialDE, SOMDE, SpaGFT, etc.), "
        "deconvolution (Cell2Location, Tangram, TACCO, STRIDE, UCDeconvolve, etc.), "
        "spatial communication (COMMOT, SPAOTSC), alignment (PASTE, MOSCOT, ST-GEARS), serial-section 3D diagnosis (diagnose_3d_stack), "
        "super-resolution (iStar, XFuse), expression denoising (SpotGF), functional enrichment "
        "(decoupler, gseapy), and comprehensive pipelines (Seurat). Returns ranked recommendations with "
        "reasons, parameter suggestions, and compatibility notes.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file (spatial or scRNA-seq)",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "analysis_goals",
                "type": "str",
                "description": "Comma-separated goals: spatial_clustering, svg_detection, deconvolution, "
                "spatial_communication, spatial_alignment, three_d_reconstruction, super_resolution, denoising, "
                "functional_enrichment, comprehensive_pipeline, or 'auto' to infer from the data ('all' "
                "means the same as 'auto'). 'auto' only ever infers the first four; spatial_alignment, "
                "three_d_reconstruction, "
                "super_resolution, denoising, functional_enrichment and comprehensive_pipeline have to "
                "be named explicitly.",
                "default": "auto",
            },
            {
                "name": "data_type",
                "type": "str",
                "description": "Data type: 'spatial', 'single_cell', 'bulk', or 'auto' to detect",
                "default": "auto",
            },
            {
                "name": "sc_reference_path",
                "type": "str",
                "description": "Path to a single-cell RNA-seq reference h5ad. REQUIRED to surface "
                "reference-based deconvolution tools (Cell2Location, RCTD, Tangram, TACCO, STRIDE, CARD, "
                "cell2location, destvi, ...); without it only reference-free deconvolution tools are recommended.",
                "default": None,
            },
        ],
    },
    {
        "name": "build_analysis_workflow",
        "description": "Build a complete, ordered analysis workflow for transcriptomics data. "
        "Produces a step-by-step plan: data preparation → QC → analysis (with specific MCP "
        "tool calls per goal) → cross-tool evaluation. Each step includes the exact function "
        "name, parameters, expected outputs, and rationale. Selects top-ranked tools per goal "
        "and includes evaluation steps for multi-tool comparison. "
        "Supports scRNA reference integration for deconvolution workflows.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "goals",
                "type": "str",
                "description": "Comma-separated goals, or the analysis_goals list returned by "
                "recommend_analysis_tools passed through unchanged: spatial_clustering, "
                "svg_detection, deconvolution, spatial_communication, spatial_alignment, "
                "three_d_reconstruction, super_resolution, denoising, functional_enrichment, "
                "comprehensive_pipeline, or 'auto' to infer from the data ('all' means the same as "
                "'auto'). 'auto' only ever infers the first four; the other six have to be named "
                "explicitly.",
                "default": "auto",
            },
            {
                "name": "reference_h5ad_path",
                "type": "str",
                "description": "Path to scRNA reference h5ad for deconvolution",
                "default": None,
            },
            {
                "name": "data_type",
                "type": "str",
                "description": "Data type: 'spatial', 'single_cell', 'bulk', or 'auto'",
                "default": "auto",
            },
        ],
    },
    {
        "name": "run_transcriptomics_qc",
        "description": "Run a standard QC pipeline on any transcriptomics h5ad. "
        "Computes total counts, genes detected, mitochondrial %, ribosomal %, hemoglobin % "
        "per cell/spot. For spatial data, also plots the spatial distribution of QC metrics. "
        "Applies adaptive filtering thresholds (based on data percentiles) and saves a "
        "QC-filtered h5ad. Generates violin plots, scatter plots, and (for spatial) spatial "
        "QC distribution plots. Returns a JSON report with metrics, thresholds, filtering "
        "results, and recommendations (e.g., warnings for high MT content or low counts).",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to input h5ad file",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to save QC results, plots, and filtered h5ad",
                "default": "./qc_results",
            },
            {
                "name": "data_type",
                "type": "str",
                "description": "Data type: 'spatial', 'single_cell', 'bulk', or 'auto'",
                "default": "auto",
            },
            {
                "name": "min_counts",
                "type": "int",
                "description": "Minimum total counts per cell/spot. Default: adaptive (from the data).",
                "default": None,
            },
            {
                "name": "min_genes",
                "type": "int",
                "description": "Minimum detected genes per cell/spot. Default: adaptive (from the data).",
                "default": None,
            },
            {
                "name": "max_pct_mt",
                "type": "float",
                "description": "Maximum mitochondrial percentage per cell/spot. Default: adaptive.",
                "default": None,
            },
        ],
    },
    {
        "name": "prepare_reference_data",
        "description": "Prepare a scRNA-seq reference h5ad for use with spatial deconvolution MCP tools "
        "(Cell2Location, Tangram, TACCO, STRIDE). Validates cell type annotations, filters "
        "rare cell types, reports composition, checks for raw counts, and outputs a ready-to-use "
        "reference. Returns a JSON report with cell type summary, QC metrics, and compatible "
        "MCP tool call parameters.",
        "required_parameters": [
            {
                "name": "sc_h5ad_path",
                "type": "str",
                "description": "Path to scRNA-seq reference h5ad file",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Path to save prepared reference h5ad",
                "default": "./reference_prepared.h5ad",
            },
            {
                "name": "labels_key",
                "type": "str",
                "description": "Column in .obs with cell type labels",
                "default": "CellType",
            },
            {
                "name": "batch_key",
                "type": "str",
                "description": "Column in .obs with batch/sample info",
                "default": "Sample",
            },
            {
                "name": "min_cells_per_type",
                "type": "int",
                "description": "Minimum cells per cell type to retain",
                "default": 10,
            },
        ],
    },
    {
        "name": "list_available_mcp_tools",
        "description": "List every MCP spatial analysis tool this agent can call, grouped by task. "
        "Tools with a curated profile carry capabilities, strengths, limitations and input "
        "requirements; the rest carry name, full name and a one-line summary. The reply also "
        "returns known_task_types, the full list of values task_type accepts. Use this to "
        "explore what tools are available before running an analysis.",
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "task_type",
                "type": "str",
                "description": "Filter by task, e.g. 'spatial_clustering', 'svg_detection', 'deconvolution', "
                "'cell_segmentation', 'spatial_analysis'. Any value listed in the reply's "
                "known_task_types is accepted; an unrecognised one is reported as unknown_task_type "
                "rather than as an empty result. Use 'all' for everything.",
                "default": "all",
            }
        ],
    },
    {
        "name": "resolve_tool_name",
        "description": (
            "Resolve a user-typed spatial transcriptomics tool name (or near-match) to a "
            "canonical MCP tool name that spatialomicsgym can actually invoke. Use this whenever the "
            "user mentions a tool by name (e.g. 'use RCTD', 'run cell2loc', 'try spacexr') "
            "and you need to determine which registered MCP tool corresponds. Returns a dict "
            "with 'name' (canonical), 'score' (1.0 for an exact, alias or run_-prefix match, else "
            "0.0), 'source' ('exact'|'alias'|'run_prefix'|'none'), and 'candidates' (top-3 "
            "alternates when nothing matched). When score < 0.7, confirm with the user before invoking."
        ),
        "required_parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "The user-typed tool name or short descriptor (e.g. 'RCTD', 'cell2loc').",
                "default": None,
            }
        ],
        "optional_parameters": [
            {
                "name": "context",
                "type": "str",
                "description": "The sentence the name appeared in (e.g. 'Use GraphST to deconvolve the "
                "spots'). For a tool whose server does several analyses, it picks the function the "
                "request is about.",
                "default": "",
            }
        ],
    },
]
