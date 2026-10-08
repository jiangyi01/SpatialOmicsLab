#!/usr/bin/env Rscript
#
# r_to_h5ad_converter.R
#
# Extracts data from R objects (.rds, .rda) and exports to a directory
# of flat files (MTX + CSV) that Python can assemble into h5ad.
#
# Supports:
#   - Seurat objects (v3/v4/v5)
#   - Plain dgCMatrix / matrix objects
#   - data.frame / tibble expression matrices
#   - .rda files containing any of the above
#
# Usage:
#   Rscript r_to_h5ad_converter.R --input file.rds --output-dir /tmp/out [--assay RNA] [--slot counts]
#     [--orientation auto|genes_x_cells|cells_x_genes]
#
# Output directory structure:
#   matrix.mtx.gz       - counts matrix (genes x cells, MatrixMarket)
#   barcodes.tsv.gz     - cell/spot barcodes
#   features.tsv.gz     - gene names
#   metadata.csv        - cell metadata (obs)
#   spatial_coords.csv  - spatial coordinates (if present)
#   reductions.csv      - dimensionality reduction embeddings (if present)
#   images/             - exported images (if Visium spatial)
#   scalefactors.json   - scale factors (if Visium spatial)
#   info.json           - object summary (class, dims, assays, etc.)

suppressPackageStartupMessages({
  library(Matrix)
  library(jsonlite)
})

log_msg <- function(...) {
  msg <- paste0(...)
  message(sprintf("[r-to-h5ad] %s", msg))
}

# Accepted values of --orientation. "auto" measures the layout from the identifiers; the other two
# are the vocabulary data_converter_worker.py already speaks.
ORIENTATIONS <- c("auto", "genes_x_cells", "cells_x_genes")

parse_args <- function(args) {
  opts <- list(
    input      = NULL,
    output_dir = NULL,
    assay      = "RNA",
    slot       = "counts",
    object_name = NULL,  # for .rda: which object to extract
    orientation = "auto" # for a bare matrix: which axis holds the cells
  )
  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args) && key != "--help") {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- if (i < length(args)) args[[i + 1L]] else NULL

    if (key == "--input")       { opts$input <- val; i <- i + 2L }
    else if (key == "--output-dir")  { opts$output_dir <- val; i <- i + 2L }
    else if (key == "--assay")       { opts$assay <- val; i <- i + 2L }
    else if (key == "--slot")        { opts$slot <- val; i <- i + 2L }
    else if (key == "--object-name") { opts$object_name <- val; i <- i + 2L }
    else if (key == "--orientation") {
      if (!(val %in% ORIENTATIONS)) {
        stop(sprintf("--orientation must be one of: %s", paste(ORIENTATIONS, collapse = ", ")))
      }
      opts$orientation <- val; i <- i + 2L
    }
    else if (key == "--help") {
      cat("Usage: Rscript r_to_h5ad_converter.R --input <file> --output-dir <dir> [--assay RNA] [--slot counts] [--object-name obj] [--orientation auto|genes_x_cells|cells_x_genes]\n")
      quit(status = 0)
    }
    else { stop(sprintf("Unknown argument: %s", key)) }
  }
  if (is.null(opts$input)) stop("--input is required")
  if (is.null(opts$output_dir)) stop("--output-dir is required")
  opts
}


# ---- Load R object ----

load_r_object <- function(path, object_name = NULL) {
  ext <- tolower(tools::file_ext(path))

  if (ext == "rds") {
    log_msg("Loading RDS: ", path)
    return(readRDS(path))
  }

  if (ext %in% c("rdata", "rda")) {
    log_msg("Loading RData: ", path)
    env <- new.env()
    load(path, envir = env)
    names_in_env <- ls(env)
    log_msg("Objects in RData: ", paste(names_in_env, collapse = ", "))

    if (!is.null(object_name)) {
      if (object_name %in% names_in_env) {
        return(get(object_name, envir = env))
      }
      stop(sprintf("Object '%s' not found. Available: %s", object_name, paste(names_in_env, collapse = ", ")))
    }

    # Auto-select: prefer Seurat objects, then largest matrix
    for (nm in names_in_env) {
      obj <- get(nm, envir = env)
      if (inherits(obj, "Seurat")) {
        log_msg("Auto-selected Seurat object: ", nm)
        return(obj)
      }
    }
    # Fallback: first object
    log_msg("Auto-selected first object: ", names_in_env[1])
    return(get(names_in_env[1], envir = env))
  }

  stop(sprintf("Unsupported file extension: .%s (expected .rds or .rda/.RData)", ext))
}


# ---- Extract from Seurat ----

extract_seurat <- function(obj, assay_name, slot_name, output_dir) {
  suppressPackageStartupMessages(library(Seurat))
  log_msg("Extracting Seurat object (class: ", paste(class(obj), collapse="/"), ")")

  # Determine available assays
  assays_available <- Assays(obj)
  log_msg("Available assays: ", paste(assays_available, collapse = ", "))

  # Select assay
  if (!(assay_name %in% assays_available)) {
    # Try common alternatives
    for (alt in c("RNA", "Spatial", "SCT", "integrated")) {
      if (alt %in% assays_available) {
        log_msg("Requested assay '", assay_name, "' not found, using '", alt, "'")
        assay_name <- alt
        break
      }
    }
    if (!(assay_name %in% assays_available)) {
      assay_name <- assays_available[1]
      log_msg("Using first available assay: ", assay_name)
    }
  }

  # Everything this function had to decide or could not do, for info$warnings (forwarded to the
  # payload by both converter doors).
  notes <- character(0)

  # Seurat v5 keeps one layer per sample after merge() or split() -- counts.1, counts.2 -- and then
  # no accessor below returns `slot_name` at all: all three reads fail and the export stops with
  # "Extraction failed". Join them first, as Seurat's own JoinLayers does, and say so (hunt
  # 2026-09-30, u29a-mcp-transport-3).
  assay_obj <- obj[[assay_name]]
  if (inherits(assay_obj, "Assay5")) {
    layer_names <- Layers(assay_obj)
    split_layers <- layer_names[startsWith(layer_names, paste0(slot_name, "."))]
    if (!(slot_name %in% layer_names) && length(split_layers) > 0) {
      obj[[assay_name]] <- JoinLayers(assay_obj)
      joined <- sprintf(
        "Assay '%s' held '%s' split into %d layers (%s); they were joined with JoinLayers before export.",
        assay_name, slot_name, length(split_layers), paste(split_layers, collapse = ", "))
      log_msg(joined)
      notes <- c(notes, joined)
    }
  }

  # Get count matrix
  log_msg("Extracting ", slot_name, " from assay ", assay_name)
  mat <- tryCatch({
    GetAssayData(obj, assay = assay_name, layer = slot_name)
  }, error = function(e) {
    tryCatch({
      GetAssayData(obj, assay = assay_name, slot = slot_name)
    }, error = function(e2) {
      # Seurat v5 keeps the matrix in a layer. Carry the error: a missing assay, a missing layer
      # and a Seurat version mismatch all arrive here, and the direct access below returns NULL
      # for a name that is not present -- so the run continues into the counts fallback with
      # nothing on the record about which of the three happened.
      log_msg("GetAssayData could not read '", slot_name, "' from assay '", assay_name,
              "' (", conditionMessage(e2), "); falling back to direct layer access")
      obj[[assay_name]]@layers[[slot_name]]
    })
  })

  if (is.null(mat)) {
    # Try raw counts
    mat <- tryCatch({
      GetAssayData(obj, assay = assay_name, layer = "counts")
    }, error = function(e) {
      GetAssayData(obj, assay = assay_name, slot = "counts")
    })
  }

  if (!inherits(mat, "dgCMatrix") && !inherits(mat, "matrix")) {
    mat <- as(mat, "dgCMatrix")
  }
  log_msg("Matrix: ", nrow(mat), " genes x ", ncol(mat), " cells")

  # Write matrix (genes x cells = standard 10x orientation)
  write_mtx_gz(mat, output_dir)

  # Write barcodes
  barcodes <- colnames(mat)
  write_tsv_gz(data.frame(barcode = barcodes), file.path(output_dir, "barcodes.tsv"))

  # Write features
  gene_names <- rownames(mat)
  features_df <- data.frame(gene_id = gene_names, gene_name = gene_names, feature_type = "Gene Expression")
  write_tsv_gz(features_df, file.path(output_dir, "features.tsv"))

  # Write metadata
  meta <- obj@meta.data
  meta$barcode <- rownames(meta)
  write.csv(meta, file.path(output_dir, "metadata.csv"), row.names = FALSE, quote = TRUE)

  # Write spatial coordinates. Every image is read, whatever it is called: the old gate,
  # `"spatial" %in% names(obj@images)`, matched no image Seurat's own loaders make --
  # Load10X_Spatial names it "slice1", LoadXenium/LoadNanostring "fov", and merge() keeps one per
  # slide -- so a standard Visium object came back with no coordinates, at status ok, with nothing
  # said (hunt 2026-09-30, u29a-mcp-transport-1).
  spatial_coords <- NULL
  image_names <- names(obj@images)
  per_image <- list()
  for (img_name in image_names) {
    got <- tryCatch(image_coordinates(obj, img_name), error = function(e) {
      notes <<- c(notes, sprintf("Image '%s' (%s) gave no coordinates: %s.", img_name,
                                 class(obj@images[[img_name]])[1], conditionMessage(e)))
      NULL
    })
    if (!is.null(got) && nrow(got) > 0) per_image[[img_name]] <- got
  }
  if (length(per_image) > 0) {
    axes <- colnames(per_image[[1]])
    same <- vapply(per_image, function(d) identical(colnames(d), axes), logical(1))
    if (!all(same)) {
      notes <- c(notes, sprintf(
        "Image(s) %s name their axes differently from '%s' (%s) and were left out of the coordinates.",
        paste(names(per_image)[!same], collapse = ", "), names(per_image)[1],
        paste(setdiff(axes, "barcode"), collapse = "/")))
      per_image <- per_image[same]
    }
    coords <- do.call(rbind, unname(per_image))
    if (any(duplicated(coords$barcode))) {
      notes <- c(notes, sprintf(
        "%d cell(s) appear in more than one image; the first image's coordinates were kept.",
        sum(duplicated(coords$barcode))))
      coords <- coords[!duplicated(coords$barcode), , drop = FALSE]
    }
    if (length(per_image) > 1) {
      notes <- c(notes, sprintf(paste0(
        "The object holds %d images (%s). Each cell's coordinates are in its own image's pixel frame, ",
        "so cells of different images overlap in obsm['spatial']; tell them apart by a metadata ",
        "column such as orig.ident."), length(per_image), paste(names(per_image), collapse = ", ")))
    }
    write.csv(coords, file.path(output_dir, "spatial_coords.csv"), row.names = FALSE, quote = TRUE)
    spatial_coords <- coords
    log_msg("Exported spatial coordinates: ", nrow(coords), " cells from image(s) ",
            paste(names(per_image), collapse = ", "))
  } else if (length(image_names) > 0) {
    notes <- c(notes, sprintf(
      "The object has image(s) %s but no coordinates could be read from any of them.",
      paste(image_names, collapse = ", ")))
  }

  # Scale factors and the tissue raster exist only on a Visium image; an FOV has neither slot.
  visium_images <- image_names[vapply(image_names, function(n) is_visium_image(obj@images[[n]]),
                                      logical(1))]
  if (length(visium_images) > 0) {
    img_name <- visium_images[1]
    if (length(visium_images) > 1) {
      notes <- c(notes, sprintf(
        "scalefactors.json and the tissue image are those of image '%s' only, of %d Visium images.",
        img_name, length(visium_images)))
    }

    # Export scale factors
    sf <- obj@images[[img_name]]@scale.factors
    if (!is.null(sf)) {
      sf_list <- list(
        tissue_hires_scalef = sf$hires,
        tissue_lowres_scalef = sf$lowres,
        spot_diameter_fullres = sf$spot,
        fiducial_diameter_fullres = sf$fiducial
      )
      # Remove NULLs
      sf_list <- sf_list[!sapply(sf_list, is.null)]
      write(toJSON(sf_list, auto_unbox = TRUE, digits = NA, pretty = TRUE),
            file.path(output_dir, "scalefactors.json"))
    }

    # Export images
    img_dir <- file.path(output_dir, "images")
    dir.create(img_dir, showWarnings = FALSE, recursive = TRUE)
    tryCatch({
      img_obj <- obj@images[[img_name]]
      if (!is.null(img_obj@image)) {
        # Save as PNG using R's png device
        img_array <- img_obj@image
        if (length(dim(img_array)) == 3) {
          # Name the file for the resolution it actually holds. Read10X_Image defaults to
          # `image.name = "tissue_lowres_image.png"`, so a Visium image slot carries the LOWRES
          # raster; Seurat keeps no hires raster at all. Exporting it as an unqualified
          # "tissue_image.png" is what let the Python side file it under images['hires'], pairing a
          # lowres raster with tissue_hires_scalef.
          png(file.path(img_dir, "tissue_lowres_image.png"),
              width = dim(img_array)[2], height = dim(img_array)[1])
          # finally=, because a dev.off() written after rasterImage() is skipped by the very
          # error the handler below exists to log. The device would stay open for the rest of
          # the process, and every later png() in this run would draw into it instead.
          tryCatch({
            par(mar = c(0,0,0,0))
            plot(0, 0, type = "n", xlim = c(0, 1), ylim = c(0, 1),
                 xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n")
            rasterImage(img_array, 0, 0, 1, 1)
          }, finally = dev.off())
          log_msg("Exported tissue image")
        }
      }
    }, error = function(e) {
      log_msg("Could not export image: ", e$message)
    })
  }

  # Check obsm['spatial'] equivalent
  if (is.null(spatial_coords) && "spatial" %in% names(obj@reductions)) {
    coords <- Embeddings(obj, reduction = "spatial")
    coords_df <- as.data.frame(coords)
    coords_df$barcode <- rownames(coords_df)
    write.csv(coords_df, file.path(output_dir, "spatial_coords.csv"), row.names = FALSE, quote = TRUE)
    spatial_coords <- coords_df
    log_msg("Exported spatial reduction coordinates")
  }

  # Write reductions (PCA, UMAP, etc.)
  red_names <- names(obj@reductions)
  if (length(red_names) > 0) {
    for (rn in red_names) {
      emb <- Embeddings(obj, reduction = rn)
      emb_df <- as.data.frame(emb)
      emb_df$barcode <- rownames(emb_df)
      write.csv(emb_df, file.path(output_dir, paste0("reduction_", rn, ".csv")),
                row.names = FALSE, quote = TRUE)
    }
    log_msg("Exported reductions: ", paste(red_names, collapse = ", "))
  }

  # Info JSON
  info <- list(
    class = paste(class(obj), collapse = "/"),
    n_cells = ncol(mat),
    n_genes = nrow(mat),
    assay_used = assay_name,
    slot_used = slot_name,
    assays_available = assays_available,
    reductions = red_names,
    has_spatial = !is.null(spatial_coords),
    has_images = length(image_names) > 0,
    metadata_columns = colnames(obj@meta.data)
  )
  # I() so a single note still serialises as an array, as extract_matrix does.
  if (length(notes)) info$warnings <- I(notes)
  info
}


# ---- Coordinates of one Seurat image ----

is_visium_image <- function(img) {
  inherits(img, "VisiumV1") || inherits(img, "VisiumV2")
}

# One image's cell coordinates as (axis, axis, barcode), the axes named for what they are so the
# Python side can order them by name: imagerow/imagecol for a Visium slide, x/y for an FOV.
image_coordinates <- function(obj, img_name) {
  img <- obj@images[[img_name]]
  if (is_visium_image(img)) {
    # `scale = NULL` is load-bearing. GetTissueCoordinates.VisiumV1 defaults to `scale = "lowres"`,
    # which multiplies BOTH axes by ScaleFactors(object)[["lowres"]] before returning -- while the
    # scalefactors.json written beside it stays fullres-relative (spot_diameter_fullres). Taking the
    # default therefore ships coordinates and scale factors in two different pixel frames.
    coords <- GetTissueCoordinates(obj, image = img_name, scale = NULL)
    # Seurat v5 (VisiumV2) returns columns named `x`, `y`, `cell` -- but its `x` is the *imagerow*
    # and its `y` is the *imagecol*: Read10X_Image builds the FOV from `coordinates[, c("imagerow",
    # "imagecol")]` and GetTissueCoordinates.Centroids renames those two columns positionally to
    # c("x","y"). Normalise to the v3/v4 names so the Python side has one contract to read, and so
    # that a consumer trusting the literal name `x` is not handed the row index.
    if (all(c("x", "y") %in% colnames(coords)) && !any(c("imagerow", "imagecol") %in% colnames(coords))) {
      colnames(coords)[match(c("x", "y"), colnames(coords))] <- c("imagerow", "imagecol")
    }
    return(data.frame(imagerow = coords$imagerow, imagecol = coords$imagecol,
                      barcode = rownames(coords), stringsAsFactors = FALSE))
  }
  if (inherits(img, "FOV")) {
    # LoadXenium, LoadNanostring and LoadVizgen put the instrument's own x and y on the FOV's
    # centroids (unless the loader was told flip.xy), so these keep the names x and y. full = FALSE:
    # one row per cell, never the vertices of a drawn polygon.
    if (!("centroids" %in% names(img))) {
      stop(sprintf("it has no 'centroids' boundary (it has: %s)", paste(names(img), collapse = ", ")))
    }
    coords <- GetTissueCoordinates(img, which = "centroids", full = FALSE)
    return(data.frame(x = coords$x, y = coords$y, barcode = as.character(coords$cell),
                      stringsAsFactors = FALSE))
  }
  stop(sprintf("this exporter has no coordinate reader for a %s image", class(img)[1]))
}


# ---- Orientation of a bare matrix ----
#
# A Seurat object states which axis is which. A bare matrix / dgCMatrix / data.frame states
# nothing, and whatever this file decides is final: barcodes.tsv.gz is written from colnames() and
# features.tsv.gz from rownames(), and data_converter_worker.py reads those straight into
# obs_names and var_names.
#
# Sizing the two axes against each other ("the larger one is cells") is right only for a small
# gene panel measured over many cells. On every Visium slide it is the gene axis that is larger --
# 18-32k genes against 2-5k spots -- so that rule swapped the axes of both storage orders.
#
# BARCODE_RE and BARCODE_PREFIX_RE are a deliberate mirror of
# spatialomicsgym.postanalysis.tables._BARCODE_RE / _BARCODE_PREFIX_RE. R cannot import Python, so
# test/test_a_bare_matrix_rds_is_not_oriented_by_its_shape.py pins the two spellings to each
# other: change one and the other has to change with it.
BARCODE_RE <- "^[ACGTN]{8,}(-\\d+)?$"
BARCODE_PREFIX_RE <- "^[A-Za-z0-9]{1,8}[_.\\-]"

looks_like_barcodes <- function(values) {
  if (is.null(values)) return(FALSE)
  sample_values <- trimws(as.character(values[seq_len(min(50L, length(values)))]))
  if (!length(sample_values)) return(FALSE)
  # A multi-slice object prefixes its barcodes ("s1_CATCAAACTGGCGCCC-1"); strip one such prefix.
  stripped <- sub(BARCODE_PREFIX_RE, "", sample_values, perl = TRUE)
  hits <- grepl(BARCODE_RE, sample_values, perl = TRUE, ignore.case = TRUE) |
    grepl(BARCODE_RE, stripped, perl = TRUE, ignore.case = TRUE)
  sum(hits) / length(sample_values) > 0.5
}

infer_matrix_orientation <- function(row_names, col_names, requested = "auto") {
  if (!identical(requested, "auto")) {
    return(list(orientation = requested, source = "caller", note = NULL))
  }
  rows_are_cells <- looks_like_barcodes(row_names)
  cols_are_cells <- looks_like_barcodes(col_names)
  if (cols_are_cells && !rows_are_cells) {
    return(list(orientation = "genes_x_cells", source = "measured", note = NULL))
  }
  if (rows_are_cells && !cols_are_cells) {
    return(list(orientation = "cells_x_genes", source = "measured", note = NULL))
  }
  # No evidence either way. Take the R storage convention -- features x samples, what Seurat,
  # SingleCellExperiment and the dgCMatrix branch below all use -- and say so, rather than
  # deciding it on the shape.
  reason <- if (rows_are_cells) {
    "Both axes look like barcodes"
  } else {
    "Neither axis carried recognisable barcodes"
  }
  note <- paste0(
    reason, "; the rows were taken as genes, which is the R storage convention. ",
    "If they are cells, obs and var are swapped in the result: transpose it (adata.T) and save it ",
    "again. (Run by hand, this exporter also takes --orientation cells_x_genes.)"
  )
  list(orientation = "genes_x_cells", source = "assumed", note = note)
}


# ---- Extract from plain matrix ----

extract_matrix <- function(obj, output_dir, orientation = "auto") {
  log_msg("Extracting matrix object (class: ", paste(class(obj), collapse="/"), ")")

  if (inherits(obj, "data.frame") || inherits(obj, "tbl_df")) {
    # drop = FALSE so a single numeric column keeps its name for the orientation test below.
    stored <- as.matrix(obj[, sapply(obj, is.numeric), drop = FALSE])
  } else if (inherits(obj, "dgCMatrix") || inherits(obj, "dgTMatrix") || inherits(obj, "matrix")) {
    stored <- obj
  } else {
    stop(sprintf("Cannot extract matrix from class: %s", paste(class(obj), collapse = "/")))
  }

  decided <- infer_matrix_orientation(rownames(stored), colnames(stored), orientation)
  if (identical(decided$orientation, "cells_x_genes")) {
    log_msg("Orientation (", decided$source, "): cells x genes - transposing to genes x cells")
    mat <- as(t(stored), "dgCMatrix")
  } else {
    log_msg("Orientation (", decided$source, "): genes x cells")
    mat <- as(stored, "dgCMatrix")
  }
  # Everything this function had to decide for itself, rather than read off the object. The
  # orientation is one of them; an axis with no dimnames is the other two.
  notes <- character(0)
  if (!is.null(decided$note)) {
    log_msg(decided$note)
    notes <- c(notes, decided$note)
  }

  log_msg("Matrix: ", nrow(mat), " genes x ", ncol(mat), " cells")

  write_mtx_gz(mat, output_dir)

  # Barcodes
  barcodes <- colnames(mat)
  if (is.null(barcodes)) {
    # A bare matrix need not carry cell names, but barcodes.tsv.gz is not optional -- it becomes
    # obs_names. Positional placeholders are the only thing we can write, and the caller has to be
    # told, because nothing downstream can tell them apart from real identifiers.
    barcodes <- paste0("cell_", seq_len(ncol(mat)))
    invented <- sprintf(
      paste0("The object carried no cell identifiers, so barcodes.tsv.gz was filled with positional ",
             "placeholders %s..%s. These are not the sample's barcodes: anything keyed by the real ",
             "ones -- coordinates, spot metadata, tissue_positions.csv -- cannot be joined to this object."),
      barcodes[1], barcodes[length(barcodes)])
    log_msg(invented)
    notes <- c(notes, invented)
  }
  write_tsv_gz(data.frame(barcode = barcodes), file.path(output_dir, "barcodes.tsv"))

  # Features
  genes <- rownames(mat)
  if (is.null(genes)) {
    genes <- paste0("gene_", seq_len(nrow(mat)))
    invented <- sprintf(
      paste0("The object carried no feature names, so features.tsv.gz was filled with positional ",
             "placeholders %s..%s. These are not gene symbols: marker lookups, gene-set scoring and ",
             "any join against a reference will not find them."),
      genes[1], genes[length(genes)])
    log_msg(invented)
    notes <- c(notes, invented)
  }
  write_tsv_gz(data.frame(gene_id = genes, gene_name = genes, feature_type = "Gene Expression"),
               file.path(output_dir, "features.tsv"))

  info <- list(
    class = paste(class(obj), collapse = "/"),
    n_cells = ncol(mat),
    n_genes = nrow(mat),
    orientation = decided$orientation,
    orientation_source = decided$source,
    has_spatial = FALSE,
    has_images = FALSE
  )
  # I() so a single note still serialises as an array -- the Python side accepts both, but the
  # payload should not change shape with the number of notes in it.
  if (length(notes)) info$warnings <- I(notes)
  info
}


# ---- Utility: write compressed files ----

write_mtx_gz <- function(mat, output_dir) {
  mtx_path <- file.path(output_dir, "matrix.mtx")
  writeMM(mat, mtx_path)
  system2("gzip", c("-f", mtx_path))
}

write_tsv_gz <- function(df, path) {
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE)
  system2("gzip", c("-f", path))
}


# ---- Main ----

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  obj <- tryCatch(
    load_r_object(opts$input, opts$object_name),
    error = function(e) {
      result <- list(status = "error", message = paste("Failed to load R object:", e$message))
      cat(toJSON(result, auto_unbox = TRUE, digits = NA), "\n")
      quit(status = 1)
    }
  )

  info <- tryCatch({
    if (inherits(obj, "Seurat")) {
      extract_seurat(obj, opts$assay, opts$slot, opts$output_dir)
    } else if (inherits(obj, "dgCMatrix") || inherits(obj, "dgTMatrix") ||
               inherits(obj, "matrix") || inherits(obj, "data.frame")) {
      extract_matrix(obj, opts$output_dir, opts$orientation)
    } else {
      # Try to coerce to matrix
      log_msg("Unknown class: ", paste(class(obj), collapse = "/"), " — attempting matrix extraction")
      extract_matrix(as.data.frame(obj), opts$output_dir, opts$orientation)
    }
  }, error = function(e) {
    list(status = "error", message = paste("Extraction failed:", e$message))
  })

  info$status <- if (is.null(info$status)) "ok" else info$status
  info$input <- opts$input
  info$output_dir <- opts$output_dir

  cat(toJSON(info, auto_unbox = TRUE, digits = NA, pretty = TRUE), "\n")
  # A failed extraction exits non-zero, as a failed load already does: exit 0 let a caller that
  # checks only the code assemble whatever half an export reached (hunt 2026-09-30,
  # u29a-mcp-transport-3).
  if (identical(info$status, "error")) quit(status = 1)
}

if (identical(environment(), globalenv())) {
  main()
}
