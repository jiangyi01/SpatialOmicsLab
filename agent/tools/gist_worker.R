#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(GIST)
  library(Matrix)
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
  msg <- paste0(...)
  message(sprintf("[gist-worker] %s", msg))
}

# GIST() picks its Stan model by whether a per-spot prior was given --
# `if (is.null(prior_index)) model = stanmodels$GIST_base_model` -- and only the enhanced model's data
# block carries prior_lambda (the base model's standata is numGenes, numCellTypes, exprMixVec, sigMat).
# This wrapper takes no per-spot prior (the image-derived input GIST is named for), so the base model is
# the one that runs, every time, and prior_lambda never reaches Stan. It stays accepted so no caller
# breaks, and the payload lists it under params.ignored instead of echoing it as a setting.
METHOD_NAME <- "GIST base model (rstan NUTS per spot, posterior mean; no image-derived prior)"
IGNORED_PARAMS <- c("prior_lambda")
IGNORED_WHY <- paste0(
  "GIST uses prior_lambda only in its image-guided (enhanced) model, which needs a per-spot prior ",
  "(prior_values/prior_index) this tool does not take; the base model ran and has no prior_lambda term"
)
# GIST's preprocess_expr_mat reads k and d only inside `if (impute) knn_smoothing(..., k = k, d = d)`, so
# with impute_st = FALSE neither reaches anything. They stay accepted and echoed, and are listed under
# params.ignored for that run, like prior_lambda.
IMPUTE_ONLY_PARAMS <- c("impute_k", "impute_d")
IMPUTE_ONLY_WHY <- paste0(
  "impute_st=False, so no KNN smoothing ran, and GIST's preprocess_expr_mat reads k and d only inside ",
  "its smoothing step"
)
NORMALIZE_CHOICES <- c("sct", "scale", "quantile", "scale-quantile")

# GIST 0.0.0.9000's internal normalize.decon -- the only normalisation preprocess_expr_mat has -- reads
# the Seurat object back with GetAssayData(slot = ...). SeuratObject deprecated slot= in 5.0.0 and makes
# it an error three minor versions on ("The `slot` argument of `GetAssayData()` was deprecated in
# SeuratObject 5.0.0 and is now defunct"), so on SeuratObject 5.3.0 every normalize choice stopped before
# any deconvolution. This is upstream's function with slot= renamed to layer= -- the same data, read by
# the accessor SeuratObject 5 accepts -- plus one as.matrix(): Seurat 5 hands "counts" and "data" back as
# a dgCMatrix, which as.data.frame() cannot coerce ("cannot coerce class 'dgCMatrix' to a data.frame").
# The dense data frame is what GIST itself works on (it reads st_expression[, i] per spot, and the CSV
# inputs arrive dense), so the densify is the method's own, not the wrapper's. Nothing else changed. It
# replaces GIST's own only on a SeuratObject that takes layer=; the payload says so in
# params.normalize_decon_patched.
normalize_decon_layer <- function(data, doQC = FALSE, method = c("scale", "quantile", "scale-quantile", "sct"),
                                  only_hvg = TRUE) {
  method <- match.arg(method)
  if (method %in% c("scale", "scale-quantile"))
    scale.factor <- median(colSums(data))
  if (doQC) {
    data <- Seurat::CreateSeuratObject(data, min.cells = 0.1 * ncol(data), min.features = 500, assay = "RNA")
  } else {
    data <- Seurat::CreateSeuratObject(data, min.cells = 0, min.features = 0, assay = "RNA")
  }
  if (method == "scale") {
    data <- Seurat::NormalizeData(data, normalization.method = "RC", scale.factor = scale.factor)
    data <- as.data.frame(as.matrix(Seurat::GetAssayData(data, assay = "RNA", layer = "data")))
  } else if (method == "quantile") {
    data <- as.data.frame(as.matrix(Seurat::GetAssayData(data, assay = "RNA", layer = "counts")))
    data <- normalize.quantiles2(data)
  } else if (method == "scale-quantile") {
    data <- Seurat::NormalizeData(data, normalization.method = "RC", scale.factor = scale.factor)
    data <- as.data.frame(as.matrix(Seurat::GetAssayData(data, assay = "RNA", layer = "data")))
    data <- normalize.quantiles2(data)
  } else if (method == "sct") {
    if (only_hvg)
      message("SCT will return only HVG ...")
    data <- Seurat::SCTransform(data, assay = "RNA", return.only.var.genes = only_hvg)
    data <- as.data.frame(as.matrix(Seurat::GetAssayData(data, assay = "SCT", layer = "scale.data")))
  }
  # A layer read that fails comes back NULL; say which normalisation produced nothing instead of
  # letting GIST's next step die on a NULL matrix.
  if (is.null(data) || length(dim(data)) != 2L || any(dim(data) == 0L)) {
    stop(sprintf("normalize.decon (method '%s') produced no expression matrix", method))
  }
  return(data)
}

# Returns TRUE when GIST's normalize.decon was replaced by the layer= copy above.
patch_normalize_decon <- function() {
  if (utils::packageVersion("SeuratObject") < "5.0.0") {
    return(FALSE)
  }
  fn <- normalize_decon_layer
  environment(fn) <- asNamespace("GIST")
  utils::assignInNamespace("normalize.decon", fn, ns = "GIST")
  TRUE
}

# The same rule as worker_utils.drop_unlabeled: NA / "" / "nan" / "none" / "na" / "<NA>" is not a class.
# read.csv reads a pandas NaN as "" and the text NA as NA; make_signature_matrix turned the first into a
# cell type named "V1" and stopped on the second with "missing values in 'row.names' are not allowed".
# "<NA>" is what pandas writes for a missing value in a nullable string or categorical column once it
# has been cast to str, and read.csv keeps it as text -- a class called "<NA>" otherwise.
is_missing_label <- function(x) {
  chr <- trimws(as.character(x))
  out <- is.na(x) | is.na(chr) | !nzchar(chr) | tolower(chr) %in% c("nan", "none", "na", "<na>")
  out[is.na(out)] <- TRUE
  out
}

# Written next to the destination and renamed into place, so a run that dies mid-write never leaves a
# truncated file under the name a reader looks for.
write_csv_atomic <- function(df, path, row.names) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = row.names, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
  invisible(path)
}

save_rds_atomic <- function(object, path) {
  partial <- paste0(path, ".partial")
  saveRDS(object, file = partial)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
  invisible(path)
}

# as.integer("abc") is NA with only a warning, and the NA then fails far from the flag that caused it.
as_count <- function(val, key, minimum) {
  out <- suppressWarnings(as.integer(val))
  if (is.na(out) || out < minimum) {
    stop(sprintf("%s must be an integer >= %d; got '%s'", key, minimum, val))
  }
  out
}

as_number <- function(val, key) {
  out <- suppressWarnings(as.numeric(val))
  if (is.na(out)) {
    stop(sprintf("%s must be a number; got '%s'", key, val))
  }
  out
}

# Which two columns hold the coordinates is decided by the names the file carries, not by the
# order the columns happen to come in. Space Ranger writes tissue_positions.csv as barcode,
# in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres, so after read.csv
# consumes the barcodes as row names the first two columns are the tissue flag and the array row
# index, and the pixel coordinates are never read. Measured on one 120-spot slide written both
# ways: agreement with the planted domains fell from 1.00 to 0.50 -- chance on two domains -- with
# both runs returning status "ok" and nothing saying a column had been chosen by position.
#
# Most specific first. x/y comes last because it is the one pair that could be array indices,
# microns or pixels, so anything more specific has to precede it to be reachable. Files our own
# converter writes carry x/y, so they select exactly the columns position selected before.
#
# Inlined rather than sourced: each worker runs as its own Rscript in its own conda env, so there
# is no shared library on the path. test/test_the_coordinate_columns_are_the_ones_the_file_names
# .py keeps the copies in step.
resolve_coord_cols <- function(coord_names, source_path) {
  lc <- tolower(coord_names)
  for (pair in list(c("imagerow", "imagecol"),
                    c("pxl_row_in_fullres", "pxl_col_in_fullres"),
                    c("array_row", "array_col"),
                    c("row", "col"),
                    c("x", "y"))) {
    if (all(pair %in% lc)) {
      return(coord_names[match(pair, lc)])
    }
  }
  flags <- c("barcode", "barcodes", "spot", "spot_id", "spotid", "cell", "cell_id", "cellid",
             "sample", "sample_id", "index", "tissue", "in_tissue")
  rest <- coord_names[!(lc %in% flags)]
  if (length(rest) < 2) {
    stop("Need two coordinate columns in ", source_path, " and found ", length(rest),
         " once the barcode and tissue-flag columns were set aside; the file has: ",
         paste(coord_names, collapse = ", "),
         ". Name the two coordinates imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, ",
         "array_row/array_col, row/col or x/y, or give them as the only two columns besides ",
         "the flags.")
  }
  rest[seq_len(2)]
}

# Inlined rather than sourced: each worker runs as its own Rscript in its own conda env, so
# there is no shared library on the path. test/test_r_workers_report_id_mismatches.py keeps
# the copies in step.
id_mismatch_msg <- function(what, a_label, a_ids, b_label, b_ids) {
  fmt <- function(ids) {
    if (length(ids) == 0) return("<none>")
    paste0("[", paste(sprintf('"%s"', utils::head(ids, 3)), collapse = ", "),
           if (length(ids) > 3) ", ..." else "", "]")
  }
  paste0(
    "No matching ", what, " between ", a_label, " and ", b_label, ". ",
    a_label, ": ", length(a_ids), " IDs ", fmt(a_ids), "; ",
    b_label, ": ", length(b_ids), " IDs ", fmt(b_ids), ". ",
    "The two must use the same identifiers. If the IDs above look like gene names the counts ",
    "matrix is the wrong way round (both orientations were tried); if they look like row ",
    "numbers (0, 1, 2) an index was written where barcodes belong."
  )
}


# first_record_line and read_coords_csv are verbatim copies of tools/spotsweeper_worker.R's (the
# comment below is its own): each worker runs as its own Rscript in its own conda env, so there is
# no shared library on the path.
#
# Space Ranger before 2.0 writes spatial/tissue_positions_list.csv with NO header row -- every
# standard Visium folder ships it beside the headed tissue_positions.csv. read.csv(header = TRUE)
# takes its first spot as the header, so the columns come back named "1", "0", "0", "1000",
# "1000" (that spot's own values), no named pair matches, resolve_coord_cols falls through to
# the first two of those, and the in_tissue flag and array_row were read as the two axes. The
# first spot was lost to the header too. Measured on a 144-spot slide written both ways: the
# headed file gave 9 artifact spots, the headerless one 107 of 143, both at status "ok".
#
# The first line is a header unless it looks like data: an identifier that is not an
# identifier-column label, followed by nothing but numbers. A pandas frame written with integer
# column names (",0,1" or "barcode,0,1") keeps reading as a header, exactly as it did before.
# So does one whose index label is not in that list ("cell_barcode,0,1,2") when it has four or
# more fields: names that are exactly 0, 1, 2, ... are pandas' RangeIndex, and no Space Ranger row
# reads 0,1,2,3,4 (its pixel coordinates are in the thousands). Three fields stay read as a spot:
# "SPOT,0,1" is also a spot at the origin, and if the line was a header after all, reading it as
# a spot costs one coordinates row that matches no spot in the counts, where reading a spot as a
# header would lose that spot.
# A headerless file is read under Space Ranger's own column names when it has Space Ranger's six
# columns (and column 2 is a 0/1 tissue flag), as barcode,x,y when it has three, and refused
# otherwise: any other layout would be a guess at which two columns are the axes.
SPACE_RANGER_POSITION_COLS <- c("in_tissue", "array_row", "array_col",
                                "pxl_row_in_fullres", "pxl_col_in_fullres")
ID_COLUMN_LABELS <- c("", "barcode", "barcodes", "spot", "spot_id", "spotid", "cell", "cell_id",
                      "cellid", "sample", "sample_id", "index")

# The line read.csv would take as the header. read.csv skips empty lines (blank.lines.skip =
# TRUE), so a file that opens with one has always read, and the header is the first non-empty
# line; taking the literal first line instead refused such a file as "empty". A line of spaces is
# not empty to read.csv -- it becomes a one-field header and the read fails -- so it is returned
# as it is, for the caller to name.
first_record_line <- function(path) {
  con <- file(path, "r")
  on.exit(close(con))
  repeat {
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L || nzchar(line)) return(line)
  }
}

read_coords_csv <- function(path) {
  first <- first_record_line(path)
  if (length(first) == 0L) {
    stop("Coordinates file ", path, " is empty: it holds no column names and no spot.")
  }
  if (!nzchar(trimws(first))) {
    stop("Coordinates file ", path, " has a line of only spaces or tabs where its column names or ",
         "first spot belong (its first non-empty line); remove that line.")
  }
  fields <- trimws(unlist(utils::read.csv(text = first, header = FALSE, colClasses = "character",
                                          check.names = FALSE, na.strings = character(0)),
                          use.names = FALSE))
  values <- suppressWarnings(as.numeric(fields[-1]))
  range_index_names <- length(fields) >= 4L &&
    identical(fields[-1], as.character(seq_len(length(fields) - 1L) - 1L))
  headerless <- length(fields) >= 3L && !(tolower(fields[1]) %in% ID_COLUMN_LABELS) &&
    all(is.finite(values)) && !range_index_names
  if (!headerless) {
    return(list(frame = read.csv(path, row.names = 1, check.names = FALSE), header = "present"))
  }

  shown <- if (nchar(first) > 120L) paste0(substr(first, 1L, 120L), "...") else first
  rename_hint <- paste0(
    "Give the file a header row -- or, if that first line is its header, rename its columns -- so ",
    "the two coordinate columns are named imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, ",
    "array_row/array_col, row/col or x/y.")
  if (length(fields) == 6L) {
    axis_names <- SPACE_RANGER_POSITION_COLS
    header <- paste0("absent: read as Space Ranger's tissue_positions_list.csv (barcode, ",
                     paste(SPACE_RANGER_POSITION_COLS, collapse = ", "), ")")
  } else if (length(fields) == 3L) {
    axis_names <- c("x", "y")
    header <- "absent: read as barcode, x, y"
  } else {
    stop("Coordinates file ", path, " has no header row that names its columns: its first line (",
         shown, ") is an identifier followed only by numbers, so it reads as a spot, and the file ",
         "has ", length(fields), " columns, which is neither Space Ranger's six-column ",
         "tissue_positions_list.csv layout (barcode, in_tissue, array_row, array_col, ",
         "pxl_row_in_fullres, pxl_col_in_fullres) nor barcode, x, y. ", rename_hint)
  }
  frame <- read.csv(path, header = FALSE, row.names = 1, check.names = FALSE,
                    col.names = c("barcode", axis_names))
  if (length(fields) == 6L) {
    flag <- frame$in_tissue
    if (!is.numeric(flag) || !all(flag %in% c(0, 1))) {
      stop("Coordinates file ", path, " has no header row that names its columns (its first line, ",
           shown, ", is an identifier followed only by numbers, so it reads as a spot) and six ",
           "columns, but its second column holds values other than 0 and 1, so it is not Space ",
           "Ranger's in_tissue flag and the file is not tissue_positions_list.csv. ", rename_hint)
    }
  }
  log_msg("Coordinates file has no header row; ", header)
  list(frame = frame, header = header)
}

# Space Ranger's in_tissue flag, read by the rule tools/worker_utils.keep_in_tissue applies to
# obs['in_tissue'] in the Python workers: 1 / "1" / TRUE is tissue; 0, FALSE, empty and anything
# else is background glass. Returned named by `ids` (the file's spot IDs, in its row order); NULL
# when the file has no in_tissue column. Background spots are left out by default and counted --
# the rule scanpy_spatial, bsp and spagft already follow -- because a counts table converted from a
# whole-array export carries them, and analysing glass as tissue is the silent alternative.
in_tissue_flags <- function(frame, ids) {
  hit <- which(tolower(trimws(colnames(frame))) == "in_tissue")
  if (length(hit) == 0L) return(NULL)
  raw <- tolower(trimws(as.character(frame[[hit[[1L]]]])))
  raw[raw %in% "true"] <- "1"
  raw[raw %in% "false"] <- "0"
  flag <- suppressWarnings(as.numeric(raw))
  stats::setNames(!is.na(flag) & flag == 1, as.character(ids))
}

# `spots` (the counts spots matched to a coordinates row) without the ones `flags` marks as
# background. `filter` has the shape worker_utils.record_in_tissue writes to params.in_tissue_filter,
# and is NULL, like `warning`, when nothing was left out. A flag column that marks none of the spots
# as tissue is refused rather than analysed as an empty slide.
keep_in_tissue_spots <- function(spots, flags, source_path) {
  none <- list(spots = spots, n_dropped = 0L, filter = NULL, warning = NULL)
  if (is.null(flags) || length(spots) == 0L) return(none)
  on <- unname(flags[spots])
  on[is.na(on)] <- FALSE
  n <- length(spots)
  if (!any(on)) {
    stop("The in_tissue column of ", source_path, " marks none of the ", n, " spots matched to the ",
         "counts as in tissue (1); fix the column so in-tissue spots are 1, or remove it if every spot ",
         "is tissue.")
  }
  n_dropped <- sum(!on)
  if (n_dropped == 0L) return(none)
  kept <- spots[on]
  list(
    spots = kept,
    n_dropped = n_dropped,
    filter = list(n_spots_supplied = n, n_spots_off_tissue_dropped = n_dropped, n_spots_used = length(kept)),
    warning = paste0(n_dropped, " of ", n, " spots have in_tissue == 0 in ", source_path, " (background outside ",
                     "the tissue) and were left out; ", length(kept), " in-tissue spots were analysed.")
  )
}

emit_error <- function(tool, error_msg, task = NULL) {
  result <- list(
    status    = "error",
    tool      = tool,
    error     = error_msg,
    traceback = sog_traceback()
  )
  if (!is.null(task)) result$task <- task
  cat(toJSON(result, auto_unbox = TRUE, digits = NA), "\n")
}

parse_args <- function(args) {
  if (length(args) > 0 && args[[1]] %in% c("--help", "-h")) {
    cat("Usage: gist_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --ref-counts-csv PATH      scRNA-seq reference counts CSV (genes x cells) [required]\n")
    cat("  --ref-celltypes-csv PATH   Reference cell type annotation CSV [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --impute-st BOOL           Whether to KNN-smooth spatial counts (default: TRUE)\n")
    cat("  --impute-k INT             Number of neighbors for KNN smoothing (default: 5)\n")
    cat("  --impute-d INT             Number of PCs for KNN smoothing (default: 10)\n")
    cat("  --normalize STR            Normalization: sct, scale, quantile, scale-quantile (default: sct)\n")
    cat("  --prior-lambda FLOAT       Accepted but not used: only GIST's image-guided model reads it (default: 50)\n")
    cat("  --num-cores INT            Cores for parallel spot processing (default: 1)\n")
    cat("  --seed INT                 Random seed, also passed to every per-spot Stan run (default: 42)\n")
    cat("  --n-iter INT               Stan iterations per chain per spot, half of them warmup (default: 2000)\n")
    cat("  --n-chains INT             Stan chains per spot (default: 4)\n")
    cat("  --drop-unlabeled BOOL      Leave out reference cells with a missing label (default: FALSE)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    impute_st          = TRUE,
    impute_k           = 5L,
    impute_d           = 10L,
    normalize          = "sct",
    prior_lambda       = 50.0,
    num_cores          = 1L,
    seed               = 42L,
    n_iter             = 2000L,
    n_chains           = 4L,
    drop_unlabeled     = FALSE
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    if (key == "--spatial-counts-csv") {
      opts$spatial_counts_csv <- val
    } else if (key == "--spatial-coords-csv") {
      opts$spatial_coords_csv <- val
    } else if (key == "--ref-counts-csv") {
      opts$ref_counts_csv <- val
    } else if (key == "--ref-celltypes-csv") {
      opts$ref_celltypes_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--impute-st") {
      opts$impute_st <- toupper(val) %in% c("TRUE", "T", "YES", "1")
    } else if (key == "--impute-k") {
      opts$impute_k <- as_count(val, key, 1L)
    } else if (key == "--impute-d") {
      opts$impute_d <- as_count(val, key, 1L)
    } else if (key == "--normalize") {
      opts$normalize <- val
    } else if (key == "--prior-lambda") {
      opts$prior_lambda <- as_number(val, key)
    } else if (key == "--num-cores") {
      opts$num_cores <- as_count(val, key, 1L)
    } else if (key == "--seed") {
      opts$seed <- as_count(val, key, 0L)
    } else if (key == "--n-iter") {
      opts$n_iter <- as_count(val, key, 2L)
    } else if (key == "--n-chains") {
      opts$n_chains <- as_count(val, key, 1L)
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- toupper(val) %in% c("TRUE", "T", "YES", "1")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_gist <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$ref_counts_csv) || is.null(opts$ref_celltypes_csv) ||
      is.null(opts$output_dir)) {
    stop("GIST requires --spatial-counts-csv, --spatial-coords-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }
  if (!(opts$normalize %in% NORMALIZE_CHOICES)) {
    stop(sprintf("normalize must be one of %s; got '%s'",
                 paste(sprintf("'%s'", NORMALIZE_CHOICES), collapse = ", "), opts$normalize))
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv,
              opts$ref_counts_csv, opts$ref_celltypes_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  normalize_decon_patched <- patch_normalize_decon()
  if (normalize_decon_patched) {
    log_msg("GIST normalize.decon: reading the Seurat object with GetAssayData(layer =) (SeuratObject ",
            as.character(utils::packageVersion("SeuratObject")), " no longer accepts its slot argument)")
  }
  warnings <- c(paste0("ignored parameter(s) ", paste(IGNORED_PARAMS, collapse = ", "), ": ", IGNORED_WHY))
  if (!isTRUE(opts$impute_st)) {
    warnings <- c(warnings, paste0("ignored parameter(s) ", paste(IMPUTE_ONLY_PARAMS, collapse = ", "), ": ",
                                   IMPUTE_ONLY_WHY))
  }

  # --- Load reference data ---
  log_msg("Reading reference counts from: ", opts$ref_counts_csv)
  ref_counts_df <- read.csv(opts$ref_counts_csv, row.names = 1, check.names = FALSE)
  ref_counts_mat <- as.matrix(ref_counts_df)

  log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE)
  cell_types <- as.character(ref_ct_df[, 1])
  names(cell_types) <- rownames(ref_ct_df)

  # Ensure genes x cells orientation
  common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  if (length(common_cells) == 0) {
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(cell_types)))}

  n_ref_cells_supplied <- ncol(ref_counts_mat)
  n_ref_cells_without_label_row <- n_ref_cells_supplied - length(common_cells)
  if (n_ref_cells_without_label_row > 0) {
    warnings <- c(warnings, paste0(
      n_ref_cells_without_label_row, " of ", n_ref_cells_supplied, " reference cells have no row in ",
      "ref_celltypes_csv and were left out of the signature matrix"
    ))
  }
  ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]
  cell_types <- cell_types[common_cells]

  # --- Missing labels are not a class ---
  missing <- is_missing_label(cell_types)
  n_ref_cells_dropped_unlabeled <- 0L
  if (any(missing)) {
    if (!isTRUE(opts$drop_unlabeled)) {
      stop(sprintf(paste0(
        "%d of %d reference cells have no label (NA/empty) in ref_celltypes_csv. Pass ",
        "drop_unlabeled=True to leave them out, or label them first; a missing label is not a class."
      ), sum(missing), length(cell_types)))
    }
    n_ref_cells_dropped_unlabeled <- sum(missing)
    warnings <- c(warnings, paste0(
      "drop_unlabeled=True: ", n_ref_cells_dropped_unlabeled, " reference cells with no label were left out"
    ))
    cell_types <- cell_types[!missing]
    ref_counts_mat <- ref_counts_mat[, names(cell_types), drop = FALSE]
    if (length(cell_types) == 0) {
      stop("No reference cell is left once the unlabelled ones are dropped (drop_unlabeled=True).")
    }
  }
  common_cells <- names(cell_types)

  n_ref_cells <- length(common_cells)
  n_ref_types <- length(unique(cell_types))
  n_genes_ref <- nrow(ref_counts_mat)
  log_msg("Reference: ", n_ref_cells, " cells, ", n_ref_types, " cell types")

  # --- Build label data frame for make_signature_matrix ---
  sc_labels <- data.frame(
    cell = common_cells,
    bio_celltype = cell_types,
    stringsAsFactors = FALSE
  )

  # --- Preprocess reference counts ---
  log_msg("Preprocessing reference counts (normalize = ", opts$normalize, ")...")
  ref_preprocessed <- preprocess_expr_mat(
    ref_counts_mat,
    impute    = FALSE,
    normalize = opts$normalize
  )

  # --- Build signature matrix ---
  log_msg("Building signature matrix...")
  sig_mat <- make_signature_matrix(ref_preprocessed, sc_labels)

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  sp_counts_mat <- as.matrix(sp_counts_df)

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Space Ranger 1's tissue_positions_list.csv has no header row; read.csv(header = TRUE) took its first
  # spot as the header, lost that spot and named the columns after its values, so the resolver read the
  # in_tissue flag as both axes. read_coords_csv decides from the first line whether it is a header.
  coords_read <- read_coords_csv(opts$spatial_coords_csv)
  sp_coords_df <- coords_read$frame

  # Ensure genes x spots orientation
  common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  if (length(common_spots) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(sp_counts_mat), "coordinates", rownames(sp_coords_df)))}

  n_spots_supplied <- ncol(sp_counts_mat)
  n_spots_without_coords <- n_spots_supplied - length(common_spots)
  if (n_spots_without_coords > 0) {
    warnings <- c(warnings, paste0(
      n_spots_without_coords, " of ", n_spots_supplied, " spots in spatial_counts_csv have no row in ",
      "spatial_coords_csv and were not deconvolved"
    ))
  }

  # Background spots (in_tissue == 0 in the coordinates file) are left out and counted.
  tissue <- keep_in_tissue_spots(common_spots, in_tissue_flags(sp_coords_df, rownames(sp_coords_df)),
                                 opts$spatial_coords_csv)
  common_spots <- tissue$spots
  n_spots_off_tissue_dropped <- tissue$n_dropped
  if (!is.null(tissue$warning)) {
    warnings <- c(warnings, tissue$warning)
    log_msg("WARNING: ", tissue$warning)
  }

  sp_counts_mat <- sp_counts_mat[, common_spots, drop = FALSE]
  coord_cols <- resolve_coord_cols(colnames(sp_coords_df), opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(sp_coords_df), collapse = ", "), ")")
  sp_coords_df <- sp_coords_df[common_spots, coord_cols, drop = FALSE]
  colnames(sp_coords_df) <- c("x", "y")

  n_genes <- nrow(sp_counts_mat)
  n_spots <- ncol(sp_counts_mat)
  log_msg("Spatial: ", n_genes, " genes x ", n_spots, " spots")

  # --- Preprocess spatial counts ---
  log_msg("Preprocessing spatial counts (impute = ", opts$impute_st,
          ", k = ", opts$impute_k, ", d = ", opts$impute_d, ")...")
  st_preprocessed <- preprocess_expr_mat(
    sp_counts_mat,
    impute    = opts$impute_st,
    k         = opts$impute_k,
    d         = opts$impute_d,
    seed      = opts$seed,
    normalize = opts$normalize
  )

  # GIST() deconvolves on intersect(rownames(sig_mat), rownames(st_expression)). With normalize = "sct"
  # preprocess_expr_mat keeps only SCTransform's variable genes (return.only.var.genes = TRUE), chosen
  # separately in the reference and in the spatial data, so the genes used can be far fewer than either
  # file supplied.
  n_genes_ref_after_normalize <- nrow(sig_mat)
  n_genes_spatial_after_normalize <- nrow(st_preprocessed)
  genes_used <- intersect(rownames(sig_mat), rownames(st_preprocessed))
  n_genes_used <- length(genes_used)
  gene_selection <- if (identical(opts$normalize, "sct")) {
    "SCTransform variable genes in each dataset, then the genes both share"
  } else {
    "all genes both datasets share"
  }
  if (n_genes_used == 0) {
    stop(sprintf(paste0(
      "No gene is shared between the reference signature (%d genes after normalize='%s', of %d supplied) ",
      "and the spatial data (%d genes after normalize='%s', of %d supplied). The two must use the same ",
      "gene identifiers%s."
    ), n_genes_ref_after_normalize, opts$normalize, n_genes_ref, n_genes_spatial_after_normalize,
    opts$normalize, n_genes,
    if (identical(opts$normalize, "sct")) "; normalize='sct' keeps only each dataset's SCTransform variable genes, and normalize='scale' keeps them all" else ""))
  }
  log_msg("Genes used by GIST: ", n_genes_used, " (reference ", n_genes_ref_after_normalize, " of ", n_genes_ref,
          ", spatial ", n_genes_spatial_after_normalize, " of ", n_genes, " after normalize = ", opts$normalize, ")")

  # --- Run GIST deconvolution ---
  # seed, iter and chains reach rstan::sampling through GIST's `...`. Without the seed each spot's Stan run
  # drew its own from the session RNG, which the parallel workers of num_cores > 1 never seeded.
  log_msg("Running GIST base model (num_cores = ", opts$num_cores, ", chains = ", opts$n_chains,
          ", iter = ", opts$n_iter, ", seed = ", opts$seed, ")...")
  proportions <- GIST(
    st_expression = st_preprocessed,
    sig_mat       = sig_mat,
    num_cores     = opts$num_cores,
    seed          = opts$seed,
    iter          = opts$n_iter,
    chains        = opts$n_chains
  )
  n_spots_deconvolved <- nrow(proportions)

  # --- Format results ---
  prop_df <- as.data.frame(proportions)
  prop_df$spot <- rownames(prop_df)

  # --- Save outputs ---
  prop_path <- file.path(opts$output_dir, "gist_proportions.csv")
  write_csv_atomic(prop_df, prop_path, row.names = FALSE)

  sig_path <- file.path(opts$output_dir, "gist_signature_matrix.csv")
  write_csv_atomic(sig_mat, sig_path, row.names = TRUE)

  rds_path <- file.path(opts$output_dir, "gist_result.rds")
  save_rds_atomic(list(
    proportions   = proportions,
    signature_mat = sig_mat,
    coords        = sp_coords_df
  ), rds_path)

  log_msg("Saved proportions to: ", prop_path)
  log_msg("Saved signature matrix to: ", sig_path)
  log_msg("Saved RDS object to: ", rds_path)

  # --- Summary ---
  cell_types_found <- colnames(proportions)
  n_celltypes <- length(cell_types_found)

  # Determine dominant cell type per spot
  dominant <- apply(proportions, 1, function(row) {
    colnames(proportions)[which.max(row)]
  })
  dominant_counts <- as.list(table(dominant))

  gene_text <- if (identical(opts$normalize, "sct")) {
    paste0("normalize='sct' kept only the genes SCTransform returned (its variable genes, among the genes it ",
           "could model), ", n_genes_ref_after_normalize,
           " of ", n_genes_ref, " in the reference and ", n_genes_spatial_after_normalize, " of ", n_genes,
           " in the spatial data, and GIST used the ", n_genes_used, " genes both share")
  } else {
    paste0("GIST used the ", n_genes_used, " genes the reference (", n_genes_ref, ") and the spatial data (",
           n_genes, ") share after normalize='", opts$normalize, "'")
  }

  result <- list(
    status       = "ok",
    tool         = "gist",
    task         = "deconvolution",
    data         = list(
      n_spots                          = n_spots,
      n_spots_supplied                 = n_spots_supplied,
      n_spots_without_coords           = n_spots_without_coords,
      n_spots_off_tissue_dropped       = n_spots_off_tissue_dropped,
      n_spots_deconvolved              = n_spots_deconvolved,
      n_genes_spatial                  = n_genes,
      n_genes_spatial_after_normalize  = n_genes_spatial_after_normalize,
      n_genes_ref                      = n_genes_ref,
      n_genes_ref_after_normalize      = n_genes_ref_after_normalize,
      n_genes_used                     = n_genes_used,
      n_ref_cells                      = n_ref_cells,
      n_ref_cells_supplied             = n_ref_cells_supplied,
      n_ref_cells_without_label_row    = n_ref_cells_without_label_row,
      n_ref_cells_dropped_unlabeled    = n_ref_cells_dropped_unlabeled,
      n_ref_types                      = n_ref_types
    ),
    output_files = list(
      proportions_csv  = prop_path,
      signature_csv    = sig_path,
      result_rds       = rds_path
    ),
    params       = list(
      impute_st        = opts$impute_st,
      impute_k         = opts$impute_k,
      impute_d         = opts$impute_d,
      normalize        = opts$normalize,
      prior_lambda     = opts$prior_lambda,
      num_cores        = opts$num_cores,
      seed             = opts$seed,
      n_iter           = opts$n_iter,
      n_chains         = opts$n_chains,
      drop_unlabeled   = isTRUE(opts$drop_unlabeled),
      method           = METHOD_NAME,
      used_fallback    = FALSE,
      ignored          = I(IGNORED_PARAMS),
      stan_model       = "GIST_base_model",
      normalize_decon_patched = normalize_decon_patched,
      gene_selection   = gene_selection
    ),
    summary      = list(
      n_cell_types     = n_celltypes,
      cell_types_found = cell_types_found,
      dominant_counts  = dominant_counts
    ),
    analysis     = paste0(
      "GIST deconvolved ", n_spots_deconvolved, " spatial spots into ", n_celltypes,
      " cell types with its base model (no image-derived prior, so prior_lambda was not used): the ",
      "posterior mean of rstan NUTS sampling, ", opts$n_chains, " chain(s) x ", opts$n_iter,
      " iterations per spot (half warmup), seed ", opts$seed, ". ", gene_text, ". ",
      if (n_spots_without_coords > 0) paste0(n_spots_without_coords, " of ", n_spots_supplied,
                                             " spots had no coordinates and were not deconvolved. ") else "",
      if (n_spots_off_tissue_dropped > 0) paste0(n_spots_off_tissue_dropped, " spots marked in_tissue == 0 ",
                                                 "(background) were left out. ") else "",
      if (!isTRUE(opts$impute_st)) "No KNN smoothing ran (impute_st=False), so impute_k and impute_d were not used. " else "",
      if (n_ref_cells_dropped_unlabeled > 0) paste0(n_ref_cells_dropped_unlabeled,
                                                    " unlabelled reference cells were left out (drop_unlabeled). ") else "",
      "Dominant cell type: ", names(which.max(unlist(dominant_counts))),
      " (", max(unlist(dominant_counts)), " spots)."
    ),
    warnings     = I(warnings)
  )
  # prior_lambda is never used; impute_k and impute_d only when impute_st is on.
  if (!isTRUE(opts$impute_st)) result$params$ignored <- I(c(IGNORED_PARAMS, IMPUTE_ONLY_PARAMS))
  if (!is.null(tissue$filter)) result$params$in_tissue_filter <- tissue$filter
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args runs inside the handler: an unknown flag or a bad number must still leave a JSON payload.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    set.seed(opts$seed)
    sink(stderr())
    result <- run_gist(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "gist",
      task      = "deconvolution",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
