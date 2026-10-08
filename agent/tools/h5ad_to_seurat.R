#!/usr/bin/env Rscript
#
# h5ad_to_seurat.R — Assemble a Seurat object from flat files exported by Python.
#
# Input directory must contain:
#   matrix.mtx.gz       — Genes x cells sparse matrix (MatrixMarket)
#   barcodes.tsv.gz     — Cell/spot names
#   features.tsv.gz     — Gene names
#   metadata.csv        — Cell metadata (obs)
# Optional:
#   spatial_coords.csv  — Spatial coordinates (obsm['spatial'])
#   reduction_*.csv     — Dimensionality reduction embeddings
#
# Output: Seurat RDS object.
#
# STDOUT: JSON result only.
# STDERR: logging.

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(jsonlite)
})

# Inlined rather than sourced: each worker runs as its own Rscript in its own conda env, so
# there is no shared library on the path. test/test_r_workers_capture_a_real_traceback.py keeps
# the copies in step.
#
# R's traceback() reads .Traceback, which only the top-level error handler populates. Inside a
# tryCatch handler the stack has already unwound, so capturing traceback() there could only ever
# report "No traceback available" -- which is what every recorded R-worker failure carried, and
# why none of them were diagnosable. withCallingHandlers runs its handler while the stack is live.
.sog_tb <- new.env(parent = emptyenv())
.sog_tb$text <- "No traceback available"

with_r_traceback <- function(expr) {
  .sog_tb$text <- "No traceback available"
  withCallingHandlers(expr, error = function(e) {
    .sog_tb$text <- paste(utils::capture.output(print(sys.calls())), collapse = "\n")
  })
}

sog_traceback <- function() {
  .sog_tb$text
}

log_msg <- function(...) {
  message(sprintf("[h5ad-to-seurat] %s", paste0(...)))
}

parse_args <- function(args) {
  opts <- list(
    input_dir  = NULL,
    output_rds = NULL,
    project    = "SeuratProject",
    assay      = "RNA"
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[i]
    if (key == "--input-dir")  { opts$input_dir  <- args[i + 1]; i <- i + 2 }
    else if (key == "--output-rds") { opts$output_rds <- args[i + 1]; i <- i + 2 }
    else if (key == "--project")    { opts$project    <- args[i + 1]; i <- i + 2 }
    else if (key == "--assay")      { opts$assay      <- args[i + 1]; i <- i + 2 }
    else { log_msg("Unknown arg: ", key); i <- i + 1 }
  }
  opts
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  if (is.null(opts$input_dir) || is.null(opts$output_rds)) {
    result <- list(status = "error", tool = "h5ad_to_seurat",
                   error = "--input-dir and --output-rds are required")
    cat(toJSON(result, auto_unbox = TRUE, digits = NA))
    cat("\n")
    return()
  }

  tryCatch(with_r_traceback({
    indir <- opts$input_dir

    # Read sparse matrix
    mtx_file <- file.path(indir, "matrix.mtx.gz")
    bar_file <- file.path(indir, "barcodes.tsv.gz")
    feat_file <- file.path(indir, "features.tsv.gz")

    log_msg("Reading matrix from: ", mtx_file)
    mat <- readMM(mtx_file)  # genes x cells

    barcodes <- readLines(bar_file)
    features <- readLines(feat_file)

    # Ensure unique names
    if (any(duplicated(features))) {
      features <- make.unique(features)
    }
    if (any(duplicated(barcodes))) {
      barcodes <- make.unique(barcodes)
    }

    rownames(mat) <- features
    colnames(mat) <- barcodes
    mat <- as(mat, "dgCMatrix")
    log_msg("Matrix: ", nrow(mat), " genes x ", ncol(mat), " cells")

    # Create Seurat object
    seurat_obj <- CreateSeuratObject(
      counts = mat,
      project = opts$project,
      assay = opts$assay
    )
    log_msg("Seurat object created")

    # Add metadata
    meta_file <- file.path(indir, "metadata.csv")
    if (file.exists(meta_file)) {
      meta <- read.csv(meta_file, row.names = 1, check.names = FALSE)
      # Only add columns not already in seurat_obj
      existing <- colnames(seurat_obj@meta.data)
      new_cols <- setdiff(colnames(meta), existing)
      if (length(new_cols) > 0) {
        for (col in new_cols) {
          seurat_obj[[col]] <- meta[colnames(seurat_obj), col]
        }
        log_msg("Added metadata columns: ", paste(new_cols, collapse = ", "))
      }
    }

    # Add spatial coordinates
    coord_file <- file.path(indir, "spatial_coords.csv")
    if (file.exists(coord_file)) {
      coords <- read.csv(coord_file, row.names = 1, check.names = FALSE)
      log_msg("Adding spatial coordinates: ", ncol(coords), " dims")

      # No image and no FOV is built here. The staging directory this reads holds the counts,
      # metadata.csv, spatial_coords.csv and reduction_*.csv -- no raster, no scale factors, no
      # array indices -- so there is nothing to construct a Visium slice from. The coordinates
      # travel two other ways instead, and celltrek_worker.R accepts either: as x_coord/y_coord
      # metadata, and as the 'spatial' DimReduc built below.
      seurat_obj$x_coord <- coords[colnames(seurat_obj), 1]
      seurat_obj$y_coord <- coords[colnames(seurat_obj), 2]

      coord_mat <- as.matrix(coords[colnames(seurat_obj), 1:2])
      # CreateDimReducObject requires every embedding column to be the key followed by a
      # dimension number; naming them "x"/"y" makes it refuse the object, and the tryCatch
      # below would turn that into a log line and an .rds with no coordinates on it. Built
      # from the key itself, as the reduction loop further down does, so the two cannot drift.
      spatial_key <- "spatial_"
      colnames(coord_mat) <- paste0(spatial_key, seq_len(ncol(coord_mat)))
      rownames(coord_mat) <- colnames(seurat_obj)

      tryCatch({
        # Add as a cell embedding (works with all Seurat versions)
        seurat_obj[["spatial"]] <- CreateDimReducObject(
          embeddings = coord_mat,
          key = spatial_key,
          assay = opts$assay
        )
        log_msg("Spatial coordinates stored as DimReduc 'spatial'")
      }, error = function(e) {
        log_msg("Could not add spatial DimReduc: ", e$message)
      })
    }

    # Add dimensionality reductions
    red_files <- list.files(indir, pattern = "^reduction_.*\\.csv$", full.names = TRUE)
    for (rf in red_files) {
      red_name <- sub("^reduction_", "", sub("\\.csv$", "", basename(rf)))
      red_data <- read.csv(rf, row.names = 1, check.names = FALSE)
      red_mat <- as.matrix(red_data[colnames(seurat_obj), , drop = FALSE])
      colnames(red_mat) <- paste0(red_name, "_", seq_len(ncol(red_mat)))

      tryCatch({
        seurat_obj[[red_name]] <- CreateDimReducObject(
          embeddings = red_mat,
          key = paste0(red_name, "_"),
          assay = opts$assay
        )
        log_msg("Added reduction: ", red_name, " (", ncol(red_mat), " dims)")
      }, error = function(e) {
        log_msg("Could not add reduction ", red_name, ": ", e$message)
      })
    }

    # Save RDS
    out_dir <- dirname(opts$output_rds)
    if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
    # Saved beside the final name and renamed onto it, as the converter's h5ad and CSV outputs are:
    # a kill or a full disk mid-saveRDS used to leave a truncated .rds at the canonical name, which
    # exists-therefore-done retry logic trusts and readRDS later fails on (hunt 2026-09-30,
    # u29a-mcp-transport-13).
    tmp_rds <- paste0(opts$output_rds, ".partial")
    tryCatch(saveRDS(seurat_obj, tmp_rds), error = function(e) {
      unlink(tmp_rds)
      stop(e)
    })
    if (!file.rename(tmp_rds, opts$output_rds)) {
      unlink(tmp_rds)
      stop(sprintf("could not move the saved object from %s onto %s", tmp_rds, opts$output_rds))
    }
    log_msg("Saved Seurat RDS: ", opts$output_rds)

    rds_size <- file.info(opts$output_rds)$size

    result <- list(
      status = "ok",
      tool = "h5ad_to_seurat",
      data = list(
        n_cells = ncol(seurat_obj),
        n_genes = nrow(seurat_obj),
        assay = opts$assay
      ),
      output_files = list(
        seurat_rds = opts$output_rds
      ),
      summary = list(
        n_cells = ncol(seurat_obj),
        n_genes = nrow(seurat_obj),
        meta_columns = colnames(seurat_obj@meta.data),
        reductions = names(seurat_obj@reductions),
        rds_size_mb = round(rds_size / 1024 / 1024, 2)
      ),
      analysis = paste0(
        "Converted h5ad to Seurat RDS with ", ncol(seurat_obj), " cells and ",
        nrow(seurat_obj), " genes. ",
        length(names(seurat_obj@reductions)), " reductions transferred."
      )
    )

  }), error = function(e) {
    result <<- list(
      status = "error",
      tool = "h5ad_to_seurat",
      error = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(result, auto_unbox = TRUE, digits = NA))
  cat("\n")
}

main()
