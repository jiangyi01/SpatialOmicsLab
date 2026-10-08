description = [
    {
        "name": "embed_image_in_h5ad",
        "description": "Embed a histology or fluorescence image into an h5ad file for MCP spatial tool compatibility. "
        "Creates hires/lowres versions, computes scale factors linking pixel to spatial coordinates, "
        "and stores in adata.uns['spatial'] (standard 10x Visium format). "
        "Compatible with stLearn (CNN features), Cell2Location, Starfysh, SpatialPrompt.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file (must have obsm['spatial'])",
                "default": None,
            },
            {
                "name": "image_path",
                "type": "str",
                "description": "Path to image file (PNG, TIFF, JPG, OME-TIFF)",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_image.h5ad in the session's "
                "output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {
                "name": "library_id",
                "type": "str",
                "description": "Library ID key for uns['spatial']",
                "default": "spatial_sample",
            },
            {
                "name": "hires_target_px",
                "type": "int",
                "description": "Target max dimension for hires image",
                "default": 2000,
            },
            {
                "name": "lowres_target_px",
                "type": "int",
                "description": "Target max dimension for lowres image",
                "default": 600,
            },
            {
                "name": "spot_diameter_fullres",
                "type": "float",
                "description": "Spot diameter in full-res pixels (auto-estimated if None)",
                "default": None,
            },
            {
                "name": "microns_per_pixel",
                "type": "float",
                "description": "Physical size of one full-resolution image pixel, for data whose "
                "obsm['spatial'] is in MICRONS (Xenium, MERSCOPE). None (default) means the "
                "coordinates are already in full-resolution image pixels, which is the Visium case; "
                "leaving it None for micron coordinates puts every spot in the wrong place.",
                "default": None,
            },
        ],
    },
    {
        "name": "export_image_from_h5ad",
        "description": "Export embedded images from h5ad to separate files. "
        "Useful for MCP tools like MISO that require a file path (histology_image_path) rather than embedded images.",
        "required_parameters": [
            {"name": "h5ad_path", "type": "str", "description": "Path to h5ad with embedded images", "default": None},
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to save exported image files",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "quality",
                "type": "str",
                "description": "Image quality to export: 'hires', 'lowres', or 'all'",
                "default": "hires",
            },
            {"name": "format", "type": "str", "description": "Output format: 'png' or 'tiff'", "default": "png"},
        ],
    },
    {
        "name": "process_visium_images",
        "description": "Load 10x Visium H&E images from a spatial/ directory and embed into an h5ad. "
        "Reads tissue_hires_image.png, tissue_lowres_image.png, and scalefactors_json.json.",
        "required_parameters": [
            {
                "name": "spatial_dir",
                "type": "str",
                "description": "Path to spatial/ directory from Space Ranger",
                "default": None,
            },
            {"name": "h5ad_path", "type": "str", "description": "Path to h5ad file to add images to", "default": None},
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_image.h5ad in the session's "
                "output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {
                "name": "library_id",
                "type": "str",
                "description": "Library ID for uns['spatial']",
                "default": "spatial_sample",
            },
        ],
    },
    {
        "name": "process_xenium_images",
        "description": "Process 10x Xenium morphology images (DAPI/fluorescence OME-TIFF) and embed in h5ad. "
        "Extracts specified channel or creates multi-channel RGB composite. Supports morphology_focus.ome.tif "
        "and morphology_mip.ome.tif from Xenium output.",
        "required_parameters": [
            {
                "name": "morphology_path",
                "type": "str",
                "description": "Path to Xenium morphology image (.ome.tif, .tif, .png)",
                "default": None,
            },
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file (must have obsm['spatial'])",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_image.h5ad in the session's "
                "output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {
                "name": "channel",
                "type": "int|str",
                "description": 'Channel index (an integer; a string of digits such as "1" is read as one) or '
                "'composite' for multi-channel merge. Any other string is refused.",
                "default": 0,
            },
            {
                "name": "library_id",
                "type": "str",
                "description": "Library ID for uns['spatial']",
                "default": "spatial_sample",
            },
            {
                "name": "hires_target_px",
                "type": "int",
                "description": "Target max dimension for hires image",
                "default": 2000,
            },
            {
                "name": "lowres_target_px",
                "type": "int",
                "description": "Target max dimension for lowres image",
                "default": 600,
            },
            {
                "name": "microns_per_pixel",
                "type": "float",
                "description": "Physical size of one morphology-image pixel. Xenium obsm['spatial'] "
                "is in microns, so this is what converts it to image pixels; the default 0.2125 is "
                "the documented full-resolution Xenium value. Pass the real value for a downsampled "
                "image, or None to state that the coordinates are already in this image's pixels.",
                "default": 0.2125,
            },
        ],
    },
    {
        "name": "process_merfish_images",
        "description": "Process MERFISH/Vizgen mosaic images (DAPI, PolyT, etc.) and embed in h5ad. "
        "Loads large mosaic TIFF, converts to RGB, and creates hires/lowres versions with scale factors.",
        "required_parameters": [
            {"name": "mosaic_path", "type": "str", "description": "Path to mosaic TIFF image", "default": None},
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file (must have obsm['spatial'])",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_image.h5ad in the session's "
                "output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {"name": "stain", "type": "str", "description": "Stain channel name (for metadata)", "default": "DAPI"},
            {"name": "library_id", "type": "str", "description": "Library ID", "default": "spatial_sample"},
            {
                "name": "hires_target_px",
                "type": "int",
                "description": "Target max dimension for hires",
                "default": 2000,
            },
            {
                "name": "lowres_target_px",
                "type": "int",
                "description": "Target max dimension for lowres",
                "default": 600,
            },
            {
                "name": "microns_per_pixel",
                "type": "float",
                "description": "Physical size of one mosaic pixel. None (default) reads it from the "
                "micron_to_mosaic_pixel_transform.csv sitting next to the mosaic image; pass it "
                "explicitly when that file is missing.",
                "default": None,
            },
        ],
    },
    {
        "name": "process_cosmx_fov_images",
        "description": "Process CosMx per-FOV composite images (CellComposite_F*.tif) and embed in h5ad. "
        "Handles FOV-level fluorescence composites from Nanostring CosMx/SMI platform.",
        "required_parameters": [
            {
                "name": "composite_dir",
                "type": "str",
                "description": "Directory with CellComposite_F*.tif FOV images",
                "default": None,
            },
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad file (must have obsm['spatial'])",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_image.h5ad in the session's "
                "output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {"name": "library_id", "type": "str", "description": "Library ID", "default": "spatial_sample"},
            {
                "name": "hires_target_px",
                "type": "int",
                "description": "Target max dimension for hires mosaic",
                "default": 2000,
            },
            {
                "name": "lowres_target_px",
                "type": "int",
                "description": "Target max dimension for lowres mosaic",
                "default": 600,
            },
        ],
    },
    {
        "name": "generate_tissue_mask",
        "description": "Generate a tissue mask from spatial coordinates or embedded histology image. "
        "Adds obs['in_tissue'] column for filtering. Method 'coords' detects nothing: it keeps an existing "
        "obs['in_tissue'] (e.g. Space Ranger's) or, without one, marks every spot with coordinates as in "
        "tissue. 'image' uses Otsu thresholding on the embedded image: tissue darker than the cut for H&E, "
        "brighter for fluorescence images embedded by process_xenium_images / process_merfish_images / "
        "process_cosmx_fov_images.",
        "required_parameters": [
            {"name": "h5ad_path", "type": "str", "description": "Path to h5ad file", "default": None},
        ],
        "optional_parameters": [
            {
                "name": "output_path",
                "type": "str",
                "description": "Output h5ad path. Default: a new <input name>_tissue_mask.h5ad in the "
                "session's output directory; the input file is never overwritten unless you pass its own path here",
                "default": None,
            },
            {
                "name": "method",
                "type": "str",
                "description": "Masking method: 'coords' or 'image'",
                "default": "coords",
            },
            {
                "name": "threshold",
                "type": "float",
                "description": "For method='image': a MULTIPLIER on the Otsu cut, not a pixel "
                "intensity. The cut is otsu * threshold, so 1.0 is plain Otsu, below 1.0 keeps only "
                "darker (more strongly stained) tissue and above 1.0 admits paler tissue. Must be "
                "in (0, 2]; a pixel value like 128 would select the entire slide and is refused.",
                "default": 0.8,
            },
        ],
    },
    {
        "name": "prepare_miso_image",
        "description": "Write an H&E image as a TIFF for the MISO MCP tool. run_miso takes the h5ad plus "
        "histology_image_path (this TIFF, or Space Ranger's tissue_hires_image.png directly) and reads spot "
        "coordinates from obsm['spatial'] itself; the report's run_miso_args are what to pass it. Also writes a "
        "locs.csv (in_tissue, array_row, array_col, pixel_y, pixel_x) for MISO's upstream scripts, which "
        "run_miso does not read.",
        "required_parameters": [
            {
                "name": "h5ad_path",
                "type": "str",
                "description": "Path to h5ad with spatial coordinates",
                "default": None,
            },
            {"name": "image_path", "type": "str", "description": "Path to H&E histology image", "default": None},
            {"name": "output_dir", "type": "str", "description": "Directory for prepared MISO inputs", "default": None},
        ],
        "optional_parameters": [
            {
                "name": "pixel_size_raw",
                "type": "float",
                "description": "Raw image microns/pixel (0.254 for Visium), used only for this report's "
                "computed_radius_px. Do not forward it to run_miso, which derives the scale from the "
                "scalefactors when its own pixel_size_raw is left 0",
                "default": 0.254,
            },
            {"name": "pixel_size", "type": "float", "description": "Target microns/pixel", "default": 0.5},
            {
                "name": "spot_diameter_microns",
                "type": "float",
                "description": "Physical spot diameter in microns",
                "default": 55.0,
            },
        ],
    },
    {
        "name": "convert_fluorescence_to_pseudo_he",
        "description": "Convert multi-channel fluorescence image to pseudo-H&E RGB for tools expecting histology. "
        "Maps DAPI to hematoxylin (blue-purple) and membrane/protein channel to eosin (pink) "
        "via Beer-Lambert color model. Enables Xenium/MERFISH/CosMx data to work with H&E-only tools "
        "like stLearn and MISO.",
        "required_parameters": [
            {
                "name": "image_path",
                "type": "str",
                "description": "Path to multi-channel fluorescence image",
                "default": None,
            },
            {"name": "output_path", "type": "str", "description": "Path to save pseudo-H&E RGB image", "default": None},
        ],
        "optional_parameters": [
            {
                "name": "dapi_channel",
                "type": "int",
                "description": "Channel index for DAPI/nuclear stain",
                "default": 0,
            },
            {
                "name": "membrane_channel",
                "type": "int",
                "description": "Channel index for membrane/protein marker (None to skip)",
                "default": 1,
            },
            {
                "name": "cyto_channel",
                "type": "int",
                "description": "Channel index for cytoplasm marker (None to skip)",
                "default": None,
            },
        ],
    },
]
