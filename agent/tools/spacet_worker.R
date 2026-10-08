#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SpaCET)
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
  msg <- paste0(...)
  message(sprintf("[spacet-worker] %s", msg))
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

# first_record_line and read_coords_csv are verbatim copies of tools/spotsweeper_worker.R's (the
# comment below is its own): each worker runs as its own Rscript in its own conda env, so there is
# no shared library on the path. load_csv reads coords_csv with them; the Space Ranger folder
# branch keeps read_positions, which also reads the .parquet layout.
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

# The six columns of a Space Ranger tissue-positions file, in the order the headerless (pre-2.0)
# layout writes them. A file with a header is read by these names instead of by position.
SPACERANGER_POSITION_COLUMNS <- c("barcode", "in_tissue", "array_row", "array_col",
                                  "pxl_row_in_fullres", "pxl_col_in_fullres")

# What SpaCET 1.4.0's inferMal_cor raises when an explicitly requested signature marks no cluster
# malignant ("No malignant cells detected in this tumor ST data set.") or does not exist for the
# cancer type ("SpaCET does not include <type> <CNA|expr> signature."). Only these two move the
# worker on to the next of the requested type's own signatures; any other error is re-raised, so a
# change in upstream's wording fails loudly instead of letting a substitute signature in.
NO_MALIGNANT_SIGNAL <- "No malignant cells detected|SpaCET does not include"

# SpaCET draws random numbers in one place on this path -- the clustering inside inferMal_cor --
# and seeds it itself with set.seed(123) immediately before. SpatialDeconv and
# SpaCET.CCI.colocalization draw none. So --seed cannot change the result: it is still accepted,
# set and echoed, and listed under params.ignored so the payload does not present it as a setting
# the run depended on.
IGNORED_PARAMS <- c("seed")

# Accepts both spellings the callers use: a bare switch ("--run-cci") and a switch with an explicit
# value ("--run-cci true"). Returns the value and how many arguments it consumed.
switch_value <- function(args, i) {
  if (i < length(args)) {
    nxt <- tolower(args[[i + 1L]])
    if (nxt %in% c("true", "t", "1", "yes")) return(list(value = TRUE, step = 2L))
    if (nxt %in% c("false", "f", "0", "no")) return(list(value = FALSE, step = 2L))
  }
  list(value = TRUE, step = 1L)
}

parse_int <- function(key, val, minimum = NULL) {
  out <- suppressWarnings(as.integer(val))
  if (is.na(out) || (!is.null(minimum) && out < minimum)) {
    stop(sprintf("%s must be an integer%s; got '%s'", key,
                 if (is.null(minimum)) "" else sprintf(" >= %d", minimum), val))
  }
  out
}

parse_args <- function(args) {
  opts <- list(
    counts_csv               = NULL,
    coords_csv               = NULL,
    visium_path              = NULL,
    matrix_dir               = NULL,
    output_dir               = NULL,
    cancer_type              = "BRCA",
    platform                 = "Visium",
    organism                 = "human",
    core_no                  = 1L,
    run_cci                  = FALSE,
    allow_signature_fallback = FALSE,
    seed                     = 0L
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--run-cci") {
      sv <- switch_value(args, i)
      opts$run_cci <- sv$value
      i <- i + sv$step
      next
    }
    if (key == "--allow-signature-fallback") {
      sv <- switch_value(args, i)
      opts$allow_signature_fallback <- sv$value
      i <- i + sv$step
      next
    }
    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    if (key == "--counts-csv") {
      opts$counts_csv <- val
    } else if (key == "--coords-csv") {
      opts$coords_csv <- val
    } else if (key == "--visium-path") {
      opts$visium_path <- val
    } else if (key == "--matrix-dir") {
      opts$matrix_dir <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--cancer-type") {
      opts$cancer_type <- val
    } else if (key == "--platform") {
      opts$platform <- val
    } else if (key == "--organism") {
      opts$organism <- val
    } else if (key == "--core-no") {
      opts$core_no <- parse_int(key, val, minimum = 1L)
    } else if (key == "--seed") {
      opts$seed <- parse_int(key, val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# A reader that opens the file while it is being written sees either nothing or the whole table,
# never a truncated one (rename(2) is atomic on one filesystem).
write_csv_atomic <- function(df, path, row.names = TRUE) {
  partial <- paste0(path, ".partial")
  utils::write.csv(df, partial, row.names = row.names, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
  invisible(path)
}

save_rds_atomic <- function(obj, path) {
  partial <- paste0(path, ".partial")
  saveRDS(obj, file = partial)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
  invisible(path)
}

show_ids <- function(ids) {
  if (length(ids) == 0) return("<none>")
  paste0("[", paste(sprintf('"%s"', utils::head(ids, 3)), collapse = ", "),
         if (length(ids) > 3) ", ..." else "", "]")
}

# Both sides' identifiers, so the caller can see whether it passed barcodes with a different
# suffix, row numbers, or a matrix the wrong way round.
spot_mismatch_msg <- function(a_label, a_ids, b_label, b_ids, hint = "") {
  paste0(
    "None of the ", length(a_ids), " spots in ", a_label, " has a row in ", b_label, ". ",
    a_label, ": ", show_ids(a_ids), "; ", b_label, ": ", length(b_ids), " IDs ", show_ids(b_ids),
    ". The two must name spots with the same identifiers.", hint
  )
}

# --- counts from a Matrix Market folder (Space Ranger's filtered_feature_bc_matrix/) ---
read_mtx_dir <- function(dir) {
  pick <- function(candidates) {
    for (name in candidates) {
      path <- file.path(dir, name)
      if (file.exists(path)) return(path)
    }
    NA_character_
  }
  mtx_path  <- pick(c("matrix.mtx.gz", "matrix.mtx"))
  feat_path <- pick(c("features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv"))
  bc_path   <- pick(c("barcodes.tsv.gz", "barcodes.tsv"))
  found <- c(matrix = mtx_path, features = feat_path, barcodes = bc_path)
  if (anyNA(found)) {
    stop(dir, " is not a complete Matrix Market folder: it has no ",
         paste(names(found)[is.na(found)], collapse = " or "), " file (expected matrix.mtx[.gz], ",
         "features.tsv[.gz] or genes.tsv[.gz], and barcodes.tsv[.gz])")
  }
  mat <- methods::as(Matrix::readMM(mtx_path), "CsparseMatrix")
  feats <- utils::read.delim(feat_path, header = FALSE, sep = "\t", quote = "",
                             colClasses = "character", comment.char = "")
  bcs <- utils::read.delim(bc_path, header = FALSE, sep = "\t", quote = "",
                           colClasses = "character", comment.char = "")
  if (nrow(feats) != nrow(mat) || nrow(bcs) != ncol(mat)) {
    stop(mtx_path, " is ", nrow(mat), " x ", ncol(mat), " but ", feat_path, " lists ", nrow(feats),
         " features and ", bc_path, " lists ", nrow(bcs), " barcodes")
  }
  # Gene symbols, as SpaCET's reference profiles are keyed; column 1 is the Ensembl ID.
  rownames(mat) <- if (ncol(feats) >= 2) feats[[2]] else feats[[1]]
  colnames(mat) <- bcs[[1]]

  n_other <- 0L
  other_types <- character(0)
  if (ncol(feats) >= 3) {
    keep <- feats[[3]] == "Gene Expression"
    if (!any(keep)) {
      stop(feat_path, " lists no 'Gene Expression' feature (types: ",
           paste(sort(unique(feats[[3]])), collapse = ", "), ")")
    }
    if (!all(keep)) {
      n_other <- sum(!keep)
      other_types <- sort(unique(feats[[3]][!keep]))
      mat <- mat[keep, , drop = FALSE]
    }
  }
  list(counts = mat, path = mtx_path, n_other_features = n_other, other_feature_types = other_types)
}

find_positions_file <- function(visium_path) {
  for (dir in c(file.path(visium_path, "spatial"), visium_path)) {
    for (name in c("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet")) {
      path <- file.path(dir, name)
      if (file.exists(path)) return(path)
    }
  }
  NA_character_
}

# Whether the first row holds column names is decided from the row, not the file name: the
# Space Ranger 2.0 tissue_positions.csv has a header, the pre-2.0 tissue_positions_list.csv does
# not -- except where a re-export wrote one into the _list file (the VisiumHD colon and Xenium
# tonsil samples do), which SpaCET's own reader, header = FALSE, turns into character columns.
# in_tissue, array_row and array_col are integers in every data row and words in every header.
read_positions <- function(path) {
  if (grepl("[.]parquet$", path)) {
    pos <- as.data.frame(arrow::read_parquet(path))
    has_header <- TRUE
  } else {
    first <- utils::read.csv(path, header = FALSE, nrows = 1, colClasses = "character")
    has_header <- ncol(first) >= 4 &&
      anyNA(suppressWarnings(as.numeric(unlist(first[1, 2:4], use.names = FALSE))))
    pos <- utils::read.csv(path, header = has_header, check.names = FALSE,
                           colClasses = "character")
  }
  if (has_header) {
    missing <- setdiff(SPACERANGER_POSITION_COLUMNS, colnames(pos))
    if (length(missing) > 0) {
      stop(path, " has a header row without the Space Ranger column(s) ",
           paste(missing, collapse = ", "), "; its columns are ",
           paste(colnames(pos), collapse = ", "))
    }
    pos <- pos[, SPACERANGER_POSITION_COLUMNS, drop = FALSE]
  } else {
    if (ncol(pos) < length(SPACERANGER_POSITION_COLUMNS)) {
      stop(path, " has ", ncol(pos), " column(s); a headerless Space Ranger tissue-positions file has ",
           "six: ", paste(SPACERANGER_POSITION_COLUMNS, collapse = ", "))
    }
    pos <- pos[, seq_along(SPACERANGER_POSITION_COLUMNS), drop = FALSE]
    colnames(pos) <- SPACERANGER_POSITION_COLUMNS
  }
  pos$barcode <- as.character(pos$barcode)
  for (col in SPACERANGER_POSITION_COLUMNS[-1]) {
    values <- suppressWarnings(as.numeric(pos[[col]]))
    if (anyNA(values)) {
      stop(path, ": column ", col, " has ", sum(is.na(values)), " non-numeric value(s), e.g. '",
           pos[[col]][is.na(values)][1], "'")
    }
    pos[[col]] <- values
  }
  pos
}

H5_ONLY_MSG <- paste0(
  " holds its counts only as filtered_feature_bc_matrix.h5, and this R cannot read HDF5: SpaCET ",
  "reads that file with Seurat::Read10X_h5, which needs the R packages Seurat and hdf5r, and neither ",
  "is installed in the spacet environment. Call spacet_deconvolution through its MCP portal, which ",
  "converts the .h5 to a Matrix Market folder before R starts, or pass counts_csv + coords_csv ",
  "(convert_h5ad_to_csv writes both)."
)

# A Space Ranger folder, read here rather than by create.SpaCET.object.10X: upstream needs Seurat +
# hdf5r for the .h5 (absent in this env), reads tissue_positions_list.csv with header = FALSE,
# requires scalefactors_json.json, and takes no platform. Spots keep their barcodes, so the
# proportions table joins back onto the matrix and the h5ad it came from.
load_visium <- function(opts) {
  vp <- opts$visium_path
  if (!dir.exists(vp)) {
    stop("--visium-path ", vp, " is not a directory. It should be a Space Ranger output folder: ",
         "filtered_feature_bc_matrix/ (or filtered_feature_bc_matrix.h5) beside a spatial/ folder.")
  }
  matrix_dir <- opts$matrix_dir
  if (is.null(matrix_dir)) {
    candidate <- file.path(vp, "filtered_feature_bc_matrix")
    if (any(file.exists(file.path(candidate, c("matrix.mtx.gz", "matrix.mtx"))))) {
      matrix_dir <- candidate
    } else if (file.exists(file.path(vp, "filtered_feature_bc_matrix.h5"))) {
      stop("--visium-path ", vp, H5_ONLY_MSG)
    } else {
      stop("--visium-path ", vp, " has neither filtered_feature_bc_matrix/matrix.mtx[.gz] nor ",
           "filtered_feature_bc_matrix.h5, so there are no counts to read.")
    }
  } else if (!dir.exists(matrix_dir)) {
    stop("--matrix-dir ", matrix_dir, " is not a directory")
  }
  log_msg("Loading counts from: ", matrix_dir)
  mtx <- read_mtx_dir(matrix_dir)
  counts <- mtx$counts

  pos_path <- find_positions_file(vp)
  if (is.na(pos_path)) {
    stop("--visium-path ", vp, " has no spatial/tissue_positions.csv, spatial/tissue_positions_list.csv ",
         "or spatial/tissue_positions.parquet, so the spots cannot be placed")
  }
  log_msg("Loading spot positions from: ", pos_path)
  pos <- read_positions(pos_path)

  barcodes <- colnames(counts)
  on_tissue <- pos$barcode[pos$in_tissue == 1]
  n_off_tissue <- sum(barcodes %in% pos$barcode[pos$in_tissue != 1])
  n_unplaced <- sum(!(barcodes %in% pos$barcode))
  spots <- barcodes[barcodes %in% on_tissue]
  if (length(spots) == 0) {
    stop(spot_mismatch_msg("the counts matrix", barcodes, paste0(pos_path, " (in_tissue == 1)"), on_tissue))
  }
  counts <- counts[, spots, drop = FALSE]
  placed <- pos[match(spots, pos$barcode), , drop = FALSE]

  # Pixel positions on the tissue image, as upstream stores them, when there is an image to put them
  # on; otherwise full-resolution pixels and no image.
  spatial_dir <- dirname(pos_path)
  image_path <- NA
  image_res <- "none"
  scale <- 1
  sf_path <- file.path(spatial_dir, "scalefactors_json.json")
  if (file.exists(sf_path) && grepl("visium", tolower(opts$platform))) {
    sf <- jsonlite::fromJSON(sf_path)
    for (res in c("lowres", "hires")) {
      img <- file.path(spatial_dir, paste0("tissue_", res, "_image.png"))
      key <- paste0("tissue_", res, "_scalef")
      if (file.exists(img) && !is.null(sf[[key]])) {
        image_path <- img
        image_res <- res
        scale <- as.numeric(sf[[key]])
        break
      }
    }
  }
  spot_coords <- data.frame(
    pixel_row = round(placed$pxl_row_in_fullres * scale, 3),
    pixel_col = round(placed$pxl_col_in_fullres * scale, 3),
    array_row = placed$array_row,
    array_col = placed$array_col,
    row.names = spots
  )
  # Upstream's micrometre grid for a Visium lattice (100 um centre to centre).
  spot_coords$coordinate_x_um <- spot_coords$array_col * 0.5 * 100
  spot_coords$coordinate_y_um <- spot_coords$array_row * 0.5 * sqrt(3) * 100
  spot_coords$coordinate_y_um <- max(spot_coords$coordinate_y_um) - spot_coords$coordinate_y_um

  notes <- character(0)
  if (n_off_tissue > 0) {
    notes <- c(notes, paste0(n_off_tissue, " matrix barcode(s) sit outside the tissue (in_tissue == 0 in ",
                             basename(pos_path), ") and were not deconvolved"))
  }
  if (n_unplaced > 0) {
    notes <- c(notes, paste0(n_unplaced, " matrix barcode(s) have no row in ", basename(pos_path),
                             " and were not deconvolved, e.g. ", show_ids(setdiff(barcodes, pos$barcode))))
  }
  if (mtx$n_other_features > 0) {
    notes <- c(notes, paste0(mtx$n_other_features, " non-gene-expression feature(s) (",
                             paste(mtx$other_feature_types, collapse = ", "),
                             ") were left out of the counts"))
  }

  n_gene_rows <- nrow(counts)
  obj <- create.SpaCET.object(
    counts = counts,
    spotCoordinates = spot_coords,
    imagePath = image_path,
    platform = opts$platform,
    organism = opts$organism
  )
  n_placed <- length(barcodes) - n_unplaced
  in_tissue_filter <- if (n_off_tissue > 0) {
    list(n_spots_supplied = n_placed, n_spots_off_tissue_dropped = n_off_tissue, n_spots_used = length(spots))
  } else {
    NULL
  }
  list(
    obj = obj,
    notes = notes,
    n_gene_rows = n_gene_rows,
    in_tissue_filter = in_tissue_filter,
    data = list(
      input_mode                = "visium_path",
      counts_source             = mtx$path,
      positions_source          = pos_path,
      image                     = image_res,
      n_matrix_spots            = length(barcodes),
      n_spots_off_tissue        = n_off_tissue,
      n_spots_without_position  = n_unplaced,
      n_non_expression_features = mtx$n_other_features
    )
  )
}

# Read a counts CSV straight into a sparse dgCMatrix, one block of rows at a time. A copy of
# tools/spacexr_worker.R's read_counts_csv_sparse (each R worker runs in its own env, so each carries
# its own), with this tool's name in the refusals.
#
# load_csv used read.csv -> as.matrix: a dense data.frame and then a dense double matrix of the whole
# table, which create.SpaCET.object immediately re-sparsifies. For the library's VisiumHD Colon slide as
# a genes x spots CSV (18,085 x 507,684) that is ~73 GB per copy, ~147 GB at the as.matrix step -- so on a
# box with less than that R was OOM-killed with no JSON, before check_stage1_memory could refuse the run
# with its numbers. Here only one block (~block_bytes of doubles) is ever dense; what accumulates is the
# non-zero triplets, and the stage-1 check runs before SpaCET starts on both input paths.
#
# Same reading rules as the read.csv(row.names = 1, check.names = FALSE) it replaces: first field of every
# row is the row name, header names are kept verbatim, "NA" is missing, a header with one field fewer than
# the rows names only the data columns. file() opens .gz/.bz2/.xz as well.
read_counts_csv_sparse <- function(path, what = "counts", block_bytes = 256 * 1024^2) {
  con <- file(path, open = "r")
  on.exit(close(con), add = TRUE)
  header <- scan(con, what = "", sep = ",", quote = "\"", nlines = 1L, quiet = TRUE,
                 na.strings = character(0))
  if (length(header) == 0L) {
    stop(what, " CSV ", path, " is empty")
  }
  first <- readLines(con, n = 1L, warn = FALSE)
  if (length(first) == 0L) {
    stop(what, " CSV ", path, " has a header but no rows")
  }
  n_fields <- length(scan(text = first, what = "", sep = ",", quote = "\"", quiet = TRUE,
                          na.strings = character(0)))
  pushBack(first, con)
  if (n_fields == length(header)) {
    col_names <- header[-1L]
  } else if (n_fields == length(header) + 1L) {
    col_names <- header
  } else {
    stop(sprintf("%s CSV %s: the header has %d fields but the first row has %d", what, path,
                 length(header), n_fields))
  }
  n_cols <- length(col_names)
  if (n_cols == 0L) {
    stop(what, " CSV ", path, " has row names but no data columns")
  }
  block_rows <- max(1L, as.integer(floor(block_bytes / (8 * n_cols))))
  row_types <- c(list(""), rep(list(0), n_cols))

  i_parts <- list(); j_parts <- list(); x_parts <- list(); rn_parts <- list()
  n_rows <- 0L
  n_missing <- 0
  n_negative <- 0
  k <- 0L
  repeat {
    blk <- scan(con, what = row_types, sep = ",", quote = "\"", nmax = block_rows,
                quiet = TRUE, multi.line = FALSE, na.strings = "NA")
    nr <- length(blk[[1L]])
    if (nr == 0L) break
    m <- matrix(unlist(blk[-1L], use.names = FALSE), nrow = nr, ncol = n_cols)
    n_missing <- n_missing + sum(is.na(m))
    idx <- which(!is.na(m) & m != 0)
    x <- m[idx]
    n_negative <- n_negative + sum(x < 0)
    k <- k + 1L
    i_parts[[k]] <- as.integer((idx - 1) %% nr) + n_rows + 1L
    j_parts[[k]] <- as.integer((idx - 1) %/% nr) + 1L
    x_parts[[k]] <- x
    rn_parts[[k]] <- blk[[1L]]
    n_rows <- n_rows + nr
    rm(m, idx, x, blk)
  }
  if (n_missing > 0) {
    stop(sprintf("%s CSV %s holds %.0f missing (NA/empty) values; SpaCET needs a count in every cell",
                 what, path, n_missing))
  }
  if (n_negative > 0) {
    stop(sprintf("%s CSV %s holds %.0f negative values; SpaCET models counts, which are never negative",
                 what, path, n_negative))
  }
  row_names <- unlist(rn_parts, use.names = FALSE)
  if (anyNA(row_names)) {
    stop(what, " CSV ", path, ": a row has no name (the first field is missing)")
  }
  if (anyDuplicated(row_names)) {
    dup <- unique(row_names[duplicated(row_names)])
    stop(sprintf("%s CSV %s: duplicate row names are not allowed (%d repeated, e.g. %s)", what, path,
                 length(dup), paste(utils::head(dup, 3), collapse = ", ")))
  }
  Matrix::sparseMatrix(
    i = unlist(i_parts, use.names = FALSE),
    j = unlist(j_parts, use.names = FALSE),
    x = as.numeric(unlist(x_parts, use.names = FALSE)),
    dims = c(n_rows, n_cols),
    dimnames = list(row_names, col_names)
  )
}

# counts_csv (genes x spots) + coords_csv. The coordinates are aligned to the counts' spots by
# name: Space Ranger's tissue_positions.csv lists every array spot in array order, the counts only
# the in-tissue ones in their own order, and create.SpaCET.object refuses anything but identical
# spot IDs in identical order.
load_csv <- function(opts) {
  log_msg("Loading counts from: ", opts$counts_csv)
  counts <- read_counts_csv_sparse(opts$counts_csv, "counts_csv")
  dup <- colnames(counts)[duplicated(colnames(counts))]
  if (length(dup) > 0) {
    stop(opts$counts_csv, " names ", length(dup), " spot(s) more than once, e.g. ", show_ids(unique(dup)))
  }

  log_msg("Loading coordinates from: ", opts$coords_csv)
  # Space Ranger 1's tissue_positions_list.csv has no header row; read.csv(header = TRUE) took its first
  # spot as the header -- that spot was lost and the columns were named after its values, so the
  # resolver took the in_tissue flag as both axes. read_coords_csv decides from the first line.
  coords <- read_coords_csv(opts$coords_csv)$frame
  coord_cols <- resolve_coord_cols(colnames(coords), opts$coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords), collapse = ", "), ")")

  all_spots <- colnames(counts)
  spots <- all_spots[all_spots %in% rownames(coords)]
  if (length(spots) == 0) {
    hint <- if (any(rownames(counts) %in% rownames(coords))) {
      " The counts' ROW names match the coordinates, so the counts table is spots x genes; SpaCET wants genes as rows and spots as columns."
    } else {
      ""
    }
    stop(spot_mismatch_msg(opts$counts_csv, all_spots, opts$coords_csv, rownames(coords), hint))
  }
  n_unplaced <- length(all_spots) - length(spots)
  n_unused <- nrow(coords) - length(spots)
  # Counts spots the coordinates file marks in_tissue == 0 (background glass) are left out and counted.
  matched <- spots
  tissue <- keep_in_tissue_spots(matched, in_tissue_flags(coords, rownames(coords)), opts$coords_csv)
  spots <- tissue$spots
  counts <- counts[, spots, drop = FALSE]
  coords <- coords[spots, coord_cols, drop = FALSE]
  colnames(coords) <- c("imagerow", "imagecol")

  notes <- character(0)
  # SpaCET normalises its input itself (counts per million, then log2), so a table that is already
  # normalised or log-transformed runs, but is normalised a second time. Said, not refused.
  n_fractional <- sum(counts@x != round(counts@x))
  if (n_fractional > 0) {
    notes <- c(notes, paste0("counts_csv holds ", n_fractional, " non-integer value(s) (of ", length(counts@x),
                             " non-zero values in the spots analysed); SpaCET treats its input as raw counts and ",
                             "normalises it itself, so normalised or log-transformed values are normalised twice"))
  }
  if (n_unplaced > 0) {
    notes <- c(notes, paste0(n_unplaced, " of the ", length(all_spots), " spots in counts_csv have no row in ",
                             "coords_csv and were not deconvolved, e.g. ",
                             show_ids(setdiff(all_spots, matched))))
  }
  if (!is.null(tissue$warning)) {
    notes <- c(notes, sub("[.]$", "", tissue$warning))
  }

  n_gene_rows <- nrow(counts)
  obj <- create.SpaCET.object(
    counts = counts,
    spotCoordinates = coords,
    platform = opts$platform,
    organism = opts$organism
  )
  list(
    obj = obj,
    notes = notes,
    n_gene_rows = n_gene_rows,
    in_tissue_filter = tissue$filter,
    data = list(
      input_mode                  = "counts_csv",
      counts_source               = opts$counts_csv,
      positions_source            = opts$coords_csv,
      coordinate_columns          = I(coord_cols),
      n_matrix_spots              = length(all_spots),
      n_spots_off_tissue          = tissue$n_dropped,
      n_spots_without_position    = n_unplaced,
      n_coordinate_rows_unused    = n_unused
    )
  )
}

# Malignant reference, SpaCET's first stage. Left alone (signatureType = NULL) SpaCET walks
# CNA:<type> -> expr:<type> -> expr:PANCAN and, when nothing matches, declares the 5% of spots with
# the most detected genes malignant ("seq_depth" / "current_sample") -- the likely outcome on
# non-tumour tissue, and one the payload used to hide behind "for BRCA cancer type". By default only
# the requested type's own two signatures are tried, each through SpaCET's own signatureType
# switch; the rest of the cascade runs only when the caller allows it.
run_deconvolution <- function(spacet_obj, opts) {
  requested <- opts$cancer_type
  check_large_slide(ncol(spacet_obj@input$counts), requested)
  n_stage1_genes <- stage1_gene_count(spacet_obj@input$counts, requested, spacet_obj@input$organism)
  need <- check_stage1_memory(n_stage1_genes, ncol(spacet_obj@input$counts))
  if (!is.na(need)) {
    log_msg(sprintf("Stage 1 densifies %d genes x %d spots: at least %.2f GB", as.integer(n_stage1_genes),
                    ncol(spacet_obj@input$counts), need / 1e9))
  }
  if (isTRUE(opts$allow_signature_fallback)) {
    log_msg("Running deconvolution (cancer type: ", requested, "; SpaCET's full signature cascade allowed)...")
    obj <- SpaCET.deconvolution(spacet_obj, cancerType = requested, coreNo = opts$core_no)
    return(list(obj = obj, tried = passed_over_by_cascade(requested, obj@results$deconvolution$malRes$sig)))
  }
  tried <- character(0)
  for (signature_type in c("CNA", "expr")) {
    log_msg("Running deconvolution (cancer type: ", requested, ", ", signature_type, " signature)...")
    attempt <- tryCatch(
      SpaCET.deconvolution(spacet_obj, cancerType = requested, signatureType = signature_type,
                           coreNo = opts$core_no),
      error = function(e) e
    )
    if (!inherits(attempt, "error")) {
      return(list(obj = attempt, tried = tried))
    }
    reason <- conditionMessage(attempt)
    if (!grepl(NO_MALIGNANT_SIGNAL, reason)) {
      stop(attempt)
    }
    log_msg("The ", signature_type, ":", requested, " signature marked no cluster malignant: ", reason)
    tried <- c(tried, paste0(signature_type, ":", requested, " (", trimws(reason), ")"))
  }
  stop(
    "SpaCET found no malignant cluster for cancer_type='", requested, "': ",
    paste(tried, collapse = "; "), ". Left to its own cascade SpaCET would go on to the pan-cancer ",
    "expression signature and then, if that matched nothing either, take the 5% of spots with the ",
    "most detected genes as its malignant reference -- which on non-tumour tissue calls healthy spots ",
    "malignant. Set allow_signature_fallback=True to let it (the payload then names the signature ",
    "that ran)",
    if (!identical(requested, "PANCAN")) ", or cancer_type='PANCAN' to ask for the pan-cancer expression signature directly" else "",
    "."
  )
}

# On the allowed path SpaCET walks its cascade itself and reports only where it stopped. The steps
# before that one found no malignant cluster, or SpaCET has no such signature (upstream skips a
# missing one without saying so), so they are listed with both possibilities rather than as tried.
passed_over_by_cascade <- function(requested, sig) {
  sig <- as.character(sig)
  if (length(sig) != 2) return(character(0))
  steps <- if (identical(requested, "PANCAN")) {
    "expr:PANCAN"
  } else {
    c(paste0("CNA:", requested), paste0("expr:", requested), "expr:PANCAN")
  }
  steps <- c(steps, "seq_depth:current_sample")
  at <- match(paste(sig, collapse = ":"), steps)
  if (is.na(at) || at == 1L) return(character(0))
  paste0(steps[seq_len(at - 1L)],
         " (passed over by SpaCET's cascade: no malignant cluster, or no such signature)")
}

# From 20,000 spots SpaCET's inferMal_cor neither clusters nor cascades: it takes the requested
# type's copy-number (CNA) signature directly, whatever signatureType says. SpaCET has none for some
# types it otherwise accepts (PANCAN among them), and indexing the empty match then fails as a bare
# "subscript out of bounds". Say so before the run, with the types that do work.
LARGE_SLIDE_SPOTS <- 20000L

cna_signature_types <- function() {
  dict <- new.env(parent = emptyenv())
  load(system.file("extdata", "cancerDictionary.rda", package = "SpaCET"), envir = dict)
  names(dict$cancerDictionary$CNA)
}

check_large_slide <- function(n_spots, cancer_type) {
  if (n_spots < LARGE_SLIDE_SPOTS) return(invisible(TRUE))
  available <- cna_signature_types()
  if (!any(grepl(cancer_type, available))) {
    stop(
      "cancer_type='", cancer_type, "' has no copy-number (CNA) signature in SpaCET, and on ", n_spots,
      " spots (", LARGE_SLIDE_SPOTS, " or more) SpaCET uses only that signature: it skips the clustering ",
      "and the signature cascade, so neither the expression signature nor allow_signature_fallback ",
      "applies. Cancer types with a CNA signature: ",
      paste(sub("^[^_]*_", "", available), collapse = ", "), "."
    )
  }
  invisible(TRUE)
}

# Copied from tools/celltrek_worker.R. Free memory by the rule of tools/worker_utils.py
# available_memory_bytes(), which an R worker cannot
# import: the smaller of the host's MemAvailable and the room under the cgroup memory limit (v2 first,
# then v1). memory.current / memory.usage_in_bytes count the page cache, and a memory-limited container
# sits at its limit on cache alone after reading its inputs; the kernel reclaims both file LRU lists
# (active_file, inactive_file in memory.stat) before it OOM-kills anything, so the working set is usage
# minus those and the room is the limit minus the working set. When the file LRU counters cannot be
# read the cache cannot be told apart, so usage is not subtracted at all and the room is the limit.
# NA when nothing can be read.
CGROUP_MEMORY_FILES <- list(
  list(limit = "memory.max", usage = "memory.current", stat = "memory.stat", prefix = ""),
  list(
    limit = file.path("memory", "memory.limit_in_bytes"), usage = file.path("memory", "memory.usage_in_bytes"),
    stat = file.path("memory", "memory.stat"), prefix = "total_"
  )
)
# cgroup v1 reports "no limit" as a page-rounded LONG_MAX; anything this large is no limit.
CGROUP_NO_LIMIT <- 2^60

read_kernel_lines <- function(path) {
  if (!file.exists(path)) {
    return(character(0))
  }
  tryCatch(suppressWarnings(readLines(path, warn = FALSE)), error = function(e) character(0))
}

read_kernel_count <- function(path) {
  txt <- trimws(paste(read_kernel_lines(path), collapse = "\n"))
  if (grepl("^[0-9]+$", txt)) as.numeric(txt) else NA_real_
}

cgroup_memory_room_bytes <- function(cgroup_dir = "/sys/fs/cgroup") {
  for (layout in CGROUP_MEMORY_FILES) {
    limit <- read_kernel_count(file.path(cgroup_dir, layout$limit))
    if (is.na(limit) || !(limit > 0 && limit < CGROUP_NO_LIMIT)) {
      next # absent, "max", or v1's no-limit value: no limit at this level
    }
    stat <- list()
    for (line in read_kernel_lines(file.path(cgroup_dir, layout$stat))) {
      parts <- strsplit(trimws(line), "[[:space:]]+")[[1]]
      if (length(parts) == 2L && grepl("^[0-9]+$", parts[2])) {
        stat[[parts[1]]] <- as.numeric(parts[2])
      }
    }
    file_lru <- vapply(c("active_file", "inactive_file"), function(k) {
      v <- stat[[paste0(layout$prefix, k)]]
      if (is.null(v)) v <- stat[[k]]
      if (is.null(v)) NA_real_ else v
    }, numeric(1))
    used <- read_kernel_count(file.path(cgroup_dir, layout$usage))
    if (is.na(used) || all(is.na(file_lru))) {
      return(limit)
    }
    working_set <- max(used - sum(file_lru, na.rm = TRUE), 0)
    return(max(limit - working_set, 0))
  }
  NA_real_
}

available_memory_bytes <- function(meminfo = "/proc/meminfo", cgroup_dir = "/sys/fs/cgroup") {
  host <- NA_real_
  hit <- grep("^MemAvailable:", read_kernel_lines(meminfo), value = TRUE)
  if (length(hit) > 0) {
    parts <- strsplit(trimws(hit[1]), "[[:space:]]+")[[1]]
    if (length(parts) >= 2L) host <- suppressWarnings(as.numeric(parts[2]) * 1024)
  }
  found <- c(host, cgroup_memory_room_bytes(cgroup_dir))
  found <- found[is.finite(found)]
  if (length(found) == 0L) NA_real_ else min(found)
}

# SpaCET's stage 1 (inferMal_cor) makes the genes x spots matrix dense at EVERY slide size, before its
# < 20,000-spot branch: sweep() of the dgCMatrix by its column sums and then `- rowMeans()` each return a
# dense dgeMatrix (checked with Matrix 1.7.5), base::sweep builds a dense STATS array of the same size
# beside the first, and `is.na()` and `log2(@x + 1)` make more full-size temporaries in between. Only the gene axis narrows on a large slide: from 20,001 spots SpaCET.deconvolution
# keeps the genes in its reference profiles -- 17,050 of the library VisiumHD Colon slide's 18,085, ~69 GB
# per copy on its 507,684 bins. Measured peak of those lines on a 2,000 x 25,000 sparse matrix: 4.2 dense
# copies. The check uses 3, a lower bound, so a run it refuses could not have finished; the alternative is
# the kernel's OOM killer and no JSON at all. No parameter of this tool makes the matrix smaller.
STAGE1_DENSE_COPIES <- 3L

# The genes stage 1 densifies, counted the way SpaCET.deconvolution selects them: expressed in some spot,
# mapped to human symbols on a mouse slide (SpaCET's own Mouse2Human table), and from 20,001 spots only
# those in its reference profiles (with the LIHC/CHOL normal-liver profiles, as upstream adds them). NA
# when one of SpaCET's tables cannot be found; the check is then skipped rather than guessed.
stage1_gene_count <- function(counts, cancer_type, organism) {
  genes <- rownames(counts)[Matrix::rowSums(counts) > 0]
  if (identical(tolower(organism), "mouse")) {
    m2h_path <- system.file("extdata", "Mouse2Human_filter.csv", package = "SpaCET")
    if (!nzchar(m2h_path)) return(NA_integer_)
    m2h <- utils::read.csv(m2h_path, row.names = 1)
    idx <- match(genes, m2h$mouse)
    genes <- unique(m2h$human[idx[!is.na(idx)]])
  }
  if (ncol(counts) > LARGE_SLIDE_SPOTS) {
    ref_path <- system.file("extdata", "combRef_0.5.rda", package = "SpaCET")
    if (!nzchar(ref_path)) return(NA_integer_)
    ref_env <- new.env(parent = emptyenv())
    load(ref_path, envir = ref_env)
    ref_genes <- rownames(ref_env$Ref$refProfiles)
    if (cancer_type %in% c("LIHC", "CHOL")) {
      normal_path <- system.file("extdata", "Ref_Normal_LIHC.rda", package = "SpaCET")
      if (!nzchar(normal_path)) return(NA_integer_)
      load(normal_path, envir = ref_env)
      ref_genes <- intersect(ref_genes, rownames(ref_env$Ref_Normal$refProfiles))
    }
    genes <- genes[genes %in% ref_genes]
  }
  length(genes)
}

check_stage1_memory <- function(n_genes, n_spots, avail = available_memory_bytes()) {
  if (is.na(n_genes)) return(NA_real_)
  one <- 8 * as.numeric(n_genes) * as.numeric(n_spots)
  need <- STAGE1_DENSE_COPIES * one
  if (!is.na(avail) && need > avail) {
    stop(sprintf(paste0(
      "SpaCET's first stage (inferMal_cor) holds its %.0f genes x %.0f spots matrix dense, at every slide ",
      "size: %.2f GB per copy, and at least %d copies are alive at once, so at least %.2f GB, while %.2f GB ",
      "is available here (the smaller of MemAvailable and the room under the cgroup memory limit, page ",
      "cache counted as free). The dense matrix is intrinsic to SpaCET and no parameter of this tool makes ",
      "it smaller; run it where at least %.2f GB is free."),
      n_genes, n_spots, one / 1e9, STAGE1_DENSE_COPIES, need / 1e9, avail / 1e9, need / 1e9))
  }
  need
}

# SpaCET's propMat is two levels in one table: the major lineages (Malignant, the names of
# Ref$lineageTree, Unidentifiable), which sum to 1 in every spot, and the sub-lineages that split some
# of them (B cell, T CD4, T CD8, cDC, Macrophage), each set summing to its parent. Measured on a
# 2,812-spot slide: per-spot column sums of the whole table ran from 1.00 to 1.99. A reader that
# treats every row as its own cell type counts those cells twice, so the payload names the levels.
# Without a lineage tree (never on SpaCET's own path) the levels are not claimed.
lineage_levels <- function(cell_types, lineage_tree) {
  if (is.null(lineage_tree) || length(lineage_tree) == 0) {
    return(list(major = cell_types, sub = character(0), split = character(0)))
  }
  major <- cell_types[cell_types %in% c("Malignant", names(lineage_tree), "Unidentifiable")]
  split <- names(lineage_tree)[vapply(names(lineage_tree), function(p) {
    any(lineage_tree[[p]] != p)
  }, logical(1))]
  list(major = major, sub = setdiff(cell_types, major), split = intersect(split, cell_types))
}

describe_signature <- function(sig) {
  if (length(sig) != 2) return("not reported by SpaCET")
  if (identical(sig[[1]], "seq_depth")) {
    return(paste0("the 5% of spots with the most detected genes, SpaCET's ", sig[[1]], "/", sig[[2]],
                  " fallback when no cancer signature matches"))
  }
  kind <- if (identical(sig[[1]], "CNA")) "copy-number (CNA)" else "expression"
  paste0("the ", sig[[2]], " ", kind, " signature")
}

run_spacet <- function(opts) {
  if (is.null(opts$output_dir)) {
    stop("--output-dir is required")
  }
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  ignored <- IGNORED_PARAMS
  warnings <- paste0(
    "ignored parameter(s) seed: SpaCET seeds its only random step (the clustering inside ",
    "inferMal_cor) with set.seed(123) itself and draws no other random numbers, so seed=",
    opts$seed, " has no effect on the result."
  )

  # --- Create SpaCET object ---
  if (!is.null(opts$visium_path)) {
    loaded <- load_visium(opts)
  } else if (!is.null(opts$counts_csv) && !is.null(opts$coords_csv)) {
    loaded <- load_csv(opts)
  } else {
    stop("Either --visium-path or both --counts-csv and --coords-csv are required")
  }
  spacet_obj <- loaded$obj
  warnings <- c(warnings, loaded$notes)

  n_genes <- nrow(spacet_obj@input$counts)
  n_spots_in <- ncol(spacet_obj@input$counts)
  log_msg("SpaCET object: ", n_genes, " genes x ", n_spots_in, " spots")
  # create.SpaCET.object keeps one row per gene symbol (rm_duplicates: the copy with the most counts).
  n_duplicate_genes <- loaded$n_gene_rows - n_genes
  if (n_duplicate_genes > 0) {
    loaded$notes <- c(loaded$notes, paste0(
      n_duplicate_genes, " gene row(s) repeat a symbol already present; SpaCET kept the copy with the ",
      "most counts of each (", loaded$n_gene_rows, " rows in, ", n_genes, " genes used)"
    ))
    warnings <- c(warnings, loaded$notes[length(loaded$notes)])
  }

  # --- QC: SpaCET drops spots with no detected gene ---
  log_msg("Running quality control...")
  spacet_obj <- SpaCET.quality.control(spacet_obj)
  n_spots_qc <- ncol(spacet_obj@input$counts)
  n_qc_removed <- n_spots_in - n_spots_qc
  if (isTRUE(opts$allow_signature_fallback) && n_spots_qc >= LARGE_SLIDE_SPOTS) {
    ignored <- c(ignored, "allow_signature_fallback")
    warnings <- c(warnings, paste0(
      "ignored parameter(s) allow_signature_fallback: on ", n_spots_qc, " spots (", LARGE_SLIDE_SPOTS,
      " or more) SpaCET takes the ", opts$cancer_type, " CNA signature directly and has no cascade to ",
      "fall back through."
    ))
  }

  # --- Deconvolution ---
  physical <- suppressWarnings(parallel::detectCores(logical = FALSE))
  core_no_effective <- if (!is.na(physical) && physical < opts$core_no) physical - 1L else opts$core_no
  decon <- run_deconvolution(spacet_obj, opts)
  spacet_obj <- decon$obj

  # --- Extract deconvolution results ---
  deconv <- spacet_obj@results$deconvolution
  prop_mat <- deconv$propMat
  cell_types <- rownames(prop_mat)
  lineage <- lineage_levels(cell_types, deconv$Ref$lineageTree)
  n_spots <- ncol(prop_mat)
  log_msg("Deconvolution complete: ", length(cell_types), " cell types across ", n_spots, " spots")

  sig <- as.character(deconv$malRes$sig)
  signature <- if (length(sig) == 2) paste(sig, collapse = ":") else "unknown"
  used_fallback <- length(sig) == 2 && !identical(sig[[2]], opts$cancer_type)
  signature_text <- describe_signature(sig)
  method <- paste0("SpaCET.deconvolution (malignant reference: ", signature_text, ")")
  if (used_fallback) {
    warnings <- c(warnings, paste0(
      "fallback ran: ", method, " -- no cluster matched the ", opts$cancer_type, " signatures, and ",
      "allow_signature_fallback=True let SpaCET's cascade go on to ", signature
    ))
  }

  # Save proportion matrix
  prop_path <- file.path(opts$output_dir, "spacet_proportions.csv")
  write_csv_atomic(prop_mat, prop_path)
  # The lineage levels beside the table, for a reader that never sees this payload: the benchmark
  # standardizer scores SpaCET's major lineages only and refuses a hierarchical table whose levels it
  # cannot name (hunt 2026-09-30, u31-benchmarking-10).
  levels_path <- file.path(opts$output_dir, "spacet_lineage_levels.json")
  levels_partial <- paste0(levels_path, ".partial")
  writeLines(toJSON(list(major_lineages = I(lineage$major), sub_lineages = I(lineage$sub)), auto_unbox = TRUE, digits = NA),
             levels_partial)
  if (!file.rename(levels_partial, levels_path)) stop("could not move ", levels_partial, " into place at ", levels_path)

  # --- CCI (optional) ---
  cci_result_keys <- character(0)
  cci_error <- NULL
  cci_path <- NULL
  cci_pairs <- 0L
  if (opts$run_cci) {
    log_msg("Running CCI colocalization...")
    attempt <- tryCatch(SpaCET.CCI.colocalization(spacet_obj), error = function(e) e)
    if (inherits(attempt, "error")) {
      cci_error <- conditionMessage(attempt)
      log_msg("CCI colocalization failed: ", cci_error)
      warnings <- c(warnings, paste0(
        "run_cci=True but SpaCET.CCI.colocalization failed: ", cci_error,
        " The deconvolution is complete; no colocalization table was written."
      ))
    } else {
      spacet_obj <- attempt
      if ("CCI" %in% names(spacet_obj@results)) {
        cci_result_keys <- names(spacet_obj@results$CCI)
      }
      coloc <- spacet_obj@results$CCI$colocalization
      if (!is.null(coloc)) {
        cci_path <- file.path(opts$output_dir, "spacet_cci_colocalization.csv")
        write_csv_atomic(coloc, cci_path, row.names = FALSE)
        cci_pairs <- nrow(coloc)
      }
    }
  }

  # Save RDS
  rds_path <- file.path(opts$output_dir, "spacet_object.rds")
  save_rds_atomic(spacet_obj, rds_path)

  metrics <- spacet_obj@results$metrics
  summary <- list(
    qc_metrics       = I(if (!is.null(metrics)) rownames(metrics) else character(0)),
    cci_available    = length(cci_result_keys) > 0,
    cci_keys         = I(cci_result_keys)
  )
  if (!is.null(cci_error)) summary$cci_error <- cci_error
  if (!is.null(cci_path)) summary$n_cci_pairs <- cci_pairs

  data <- c(
    list(
      n_spots               = n_spots,
      n_spots_in            = n_spots_in,
      n_spots_removed_by_qc = n_qc_removed,
      n_genes               = n_genes,
      n_genes_in            = loaded$n_gene_rows,
      n_duplicate_gene_rows_removed = n_duplicate_genes,
      n_cell_types          = length(cell_types),
      cell_types            = I(cell_types),
      major_lineages        = I(lineage$major),
      sub_lineages          = I(lineage$sub)
    ),
    loaded$data
  )

  qc_note <- if (n_qc_removed > 0) {
    paste0(" (", n_spots_in, " were handed to it; its quality control removed ", n_qc_removed,
           " with no detected gene)")
  } else {
    ""
  }
  fallback_note <- if (used_fallback) {
    paste0(" No cluster matched the ", opts$cancer_type, " signatures, so allow_signature_fallback=True ",
           "let SpaCET fall back to ", signature, "; the Malignant row is not a ", opts$cancer_type,
           " tumour call.")
  } else {
    ""
  }
  tried_note <- if (length(decon$tried) > 0) {
    paste0(" Before it: ", paste(decon$tried, collapse = "; "), ".")
  } else {
    ""
  }
  input_note <- if (length(loaded$notes) > 0) paste0(" ", paste(loaded$notes, collapse = "; "), ".") else ""
  large_note <- if (n_spots_qc >= LARGE_SLIDE_SPOTS) {
    paste0(" From 20,000 spots SpaCET takes the ", opts$cancer_type, " CNA signature without clustering ",
           "or cascading", if (n_spots_qc > LARGE_SLIDE_SPOTS) ", and keeps only the genes in its reference profiles" else "",
           ".")
  } else {
    ""
  }
  cci_note <- if (!opts$run_cci) {
    ""
  } else if (!is.null(cci_error)) {
    paste0(" CCI colocalization failed: ", cci_error)
  } else if (!is.null(cci_path)) {
    paste0(" CCI colocalization scored ", cci_pairs, " cell-type pairs (spacet_cci_colocalization.csv).")
  } else {
    " CCI colocalization returned no table."
  }

  levels_note <- if (length(lineage$sub) > 0) {
    paste0(" The table is hierarchical: ", length(lineage$major), " major lineages (data.major_lineages) sum ",
           "to 1 in each spot, and ", length(lineage$sub), " sub-lineages (data.sub_lineages) split ",
           paste(lineage$split, collapse = ", "), ", each set summing to its parent -- do not add the ",
           "two levels together.")
  } else {
    ""
  }

  payload <- list(
    status       = "ok",
    tool         = "SpaCET",
    task         = "deconvolution",
    data         = data,
    output_files = list(
      proportions_csv  = prop_path,
      lineage_levels_json = levels_path,
      spacet_rds       = rds_path
    ),
    params       = list(
      cancer_type                = opts$cancer_type,
      platform                   = spacet_obj@input$platform,
      organism                   = spacet_obj@input$organism,
      core_no                    = opts$core_no,
      core_no_effective          = core_no_effective,
      run_cci                    = opts$run_cci,
      allow_signature_fallback   = opts$allow_signature_fallback,
      seed                       = opts$seed,
      malignant_signature        = signature,
      malignant_signatures_tried = I(decon$tried),
      method                     = method,
      used_fallback              = used_fallback,
      ignored                    = I(ignored)
    ),
    summary      = summary,
    warnings     = I(warnings),
    analysis     = paste0(
      "SpaCET deconvolved ", length(cell_types), " cell types across ", n_spots, " spots", qc_note,
      ".", levels_note, " Malignant reference: ", signature_text, ".", fallback_note, tried_note, input_note,
      large_note, cci_note
    )
  )
  if (!is.null(cci_path)) payload$output_files$cci_colocalization_csv <- cci_path
  if (!is.null(loaded$in_tissue_filter)) payload$params$in_tissue_filter <- loaded$in_tissue_filter
  payload
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # Parsing sits inside the handler: a bad value ("--core-no abc") is refused in the JSON payload
  # the portal reads, not as a bare R error with nothing on stdout.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    set.seed(opts$seed)
    run_spacet(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "SpaCET",
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
