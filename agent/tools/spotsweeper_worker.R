#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SpotSweeper)
  library(SpatialExperiment)
  library(scuttle)
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
  message(sprintf("[spotsweeper-worker] %s", msg))
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

# Write to <path>.partial and move it into place, so a run killed mid-write never leaves a
# truncated table under the name a reader trusts.
write_atomically <- function(path, writer) {
  tmp <- paste0(path, ".partial")
  done <- FALSE
  on.exit(if (!done && file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) stop("Could not move ", tmp, " into place at ", path)
  done <- TRUE
  invisible(path)
}


# Read a counts CSV straight into a sparse dgCMatrix, one block of rows at a time. A copy of
# tools/spacexr_worker.R's read_counts_csv_sparse (each R worker runs in its own env, so each carries
# its own), with this tool's name in the refusals. The counts used to be read with read.csv -> as.matrix
# -- a dense data.frame and a dense double matrix of the whole table -- only to be made sparse again for
# the SpatialExperiment; nothing SpotSweeper computes (per-spot sums, neighbour statistics on them) needs
# them dense.
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
    stop(sprintf("%s CSV %s holds %.0f missing (NA/empty) values; SpotSweeper needs a count in every cell",
                 what, path, n_missing))
  }
  if (n_negative > 0) {
    stop(sprintf(paste0("%s CSV %s holds %.0f negative values; SpotSweeper's QC metrics are sums of counts ",
                        "(library size, detected genes, mitochondrial share), which are never negative"),
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
  sparseMatrix(
    i = unlist(i_parts, use.names = FALSE),
    j = unlist(j_parts, use.names = FALSE),
    x = as.numeric(unlist(x_parts, use.names = FALSE)),
    dims = c(n_rows, n_cols),
    dimnames = list(row_names, col_names)
  )
}

# Copied from tools/spacet_worker.R (and tools/spacexr_worker.R): each R worker runs in its own env.
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

# The array-index column pairs a coordinates file can name, row first -- the pairs resolve_coord_cols
# ranks after the pixel positions. Copied, with array_index_lattice below, from tools/spotclean_worker.R.
ARRAY_INDEX_PAIRS <- list(c("array_row", "array_col"), c("row", "col"))

# Which lattice a pair of array indices lays the spots on. Visium numbers its hexagonal lattice so
# that array_row + array_col has one parity on every spot (the column index steps by 2 along a row
# and each row is offset by 1); in index space a spot's six neighbours then sit at 1.41 (four) and
# 2.0 (two) where on the slide all six are one pitch away. Measured on the library's Heart Fetal12W
# sample (4,992 spots): parity 0 on every spot; index-space neighbour distances 1.41 x 4, 2.0 x 2;
# after imagerow = array_row * sqrt(3), all six at 2.0. A square lattice (both parities present) is
# already isotropic, so its indices are distances up to one scale, which SpotClean divides out.
array_index_lattice <- function(row_idx, col_idx) {
  ok <- is.finite(row_idx) & is.finite(col_idx)
  r <- row_idx[ok]
  cc <- col_idx[ok]
  if (length(r) < 2L || any(r != round(r)) || any(cc != round(cc))) {
    return("not integer")
  }
  parity <- (round(r) + round(cc)) %% 2
  if (length(unique(parity)) == 1L && length(unique(r)) > 1L && length(unique(cc)) > 1L) {
    "hexagonal"
  } else {
    "square"
  }
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
    cat("Usage: spotsweeper_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --n-neighbors INT          Number of nearest neighbors for outlier detection (default: 36)\n")
    cat("  --threshold FLOAT          Z-score cutoff for outlier detection (default: 3.0)\n")
    cat("  --allow-array-index-fallback BOOL\n")
    cat("                             With only Visium hexagonal array indices (array_row/array_col) for\n")
    cat("                             positions, read them as the lattice's physical layout (default: FALSE)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    output_dir         = NULL,
    n_neighbors        = 36L,
    threshold          = 3.0,
    allow_array_index_fallback = FALSE
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
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-neighbors") {
      n <- suppressWarnings(as.numeric(val))
      if (is.na(n) || !is.finite(n) || n < 1 || n != round(n)) {
        stop(sprintf("--n-neighbors must be a positive whole number of neighbours, got '%s'", val))
      }
      opts$n_neighbors <- as.integer(n)
    } else if (key == "--threshold") {
      cutoff <- suppressWarnings(as.numeric(val))
      # A NaN cutoff makes every comparison NA, which SpotSweeper stores and sum(na.rm = TRUE)
      # then counts as zero outliers at status "ok".
      if (is.na(cutoff) || !is.finite(cutoff)) {
        stop(sprintf("--threshold must be a finite modified z-score cutoff, got '%s'", val))
      }
      opts$threshold <- cutoff
    } else if (key == "--allow-array-index-fallback") {
      opts$allow_array_index_fallback <- tolower(val) %in% c("true", "t", "1", "yes")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_spotsweeper <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$output_dir)) {
    stop("SpotSweeper requires --spatial-counts-csv, --spatial-coords-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  counts_mat <- read_counts_csv_sparse(opts$spatial_counts_csv, "spatial_counts_csv")

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  coords_read <- read_coords_csv(opts$spatial_coords_csv)
  coords_df <- coords_read$frame
  warnings <- character(0)

  # Ensure genes x spots orientation: try to match colnames to coord rownames
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  if (length(common_spots) == 0) {
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}

  # A spot in the counts with no row in the coordinates file cannot be placed, so it is left out
  # -- and counted, rather than vanishing from n_spots unannounced. (Rows of the coordinates file
  # with no counts, such as Space Ranger's in_tissue = 0 spots, are not the caller's data.)
  n_spots_in_counts <- ncol(counts_mat)
  no_coords <- setdiff(colnames(counts_mat), common_spots)
  if (length(no_coords) > 0) {
    warnings <- c(warnings, paste0(
      length(no_coords), " of ", n_spots_in_counts, " spots in the counts have no row in ",
      opts$spatial_coords_csv, " and were left out of the analysis (e.g. ",
      paste(utils::head(no_coords, 3), collapse = ", "), ")."))
    log_msg("WARNING: ", utils::tail(warnings, 1))
  }

  # Background glass (in_tissue == 0) is left out and counted, by the rule of
  # worker_utils.keep_in_tissue -- SpotSweeper's own vignette keeps spe$in_tissue == 1 before its QC. A
  # counts table converted from a whole-array export carries the background (3,009 of the 4,992 spots of
  # the library's Heart Fetal12W sample; every background spot of the Muscle and Skin samples has under
  # 100 counts), and scored beside the tissue it pulls every edge spot's neighbourhood towards glass and
  # puts glass into findArtifacts' two k-means clusters. The converter's metadata.csv carries the flag
  # for exactly this; a file without an in_tissue column is taken as all tissue.
  n_spots_matched <- length(common_spots)
  tissue <- keep_in_tissue_spots(common_spots, in_tissue_flags(coords_df, rownames(coords_df)),
                                 opts$spatial_coords_csv)
  if (!is.null(tissue$warning)) {
    warnings <- c(warnings, tissue$warning)
    log_msg("WARNING: ", tissue$warning)
  }
  common_spots <- tissue$spots

  counts_mat <- counts_mat[, common_spots, drop = FALSE]
  coords_df  <- coords_df[common_spots, , drop = FALSE]

  # SpotSweeper's metrics are sums of counts; a normalised or log-transformed table still runs, but its
  # "library size" is then not one. Said, not refused.
  n_fractional <- sum(counts_mat@x != round(counts_mat@x))
  if (n_fractional > 0) {
    warnings <- c(warnings, paste0(
      "spatial_counts_csv holds ", n_fractional, " non-integer value(s) (of ", length(counts_mat@x),
      " non-zero values in the spots analysed); SpotSweeper's library size and detected-gene metrics are ",
      "meant for raw counts, so on normalised or log-transformed values they are not what their names say."))
    log_msg("WARNING: ", utils::tail(warnings, 1))
  }

  coord_cols <- resolve_coord_cols(colnames(coords_df), opts$spatial_coords_csv)
  # The resolver matches names case-blind and takes the first match, so two columns that share a
  # name -- or one column picked for both axes -- cannot be told apart. That is what a data row
  # read as a header looks like; refuse it rather than analyse a degenerate slide.
  dup_lc <- unique(tolower(colnames(coords_df))[duplicated(tolower(colnames(coords_df)))])
  if (anyDuplicated(coord_cols) > 0L || any(tolower(coord_cols) %in% dup_lc)) {
    stop("The coordinate columns chosen in ", opts$spatial_coords_csv, " (",
         paste(coord_cols, collapse = ", "), ") are not two distinct columns: the file names ",
         "more than one column ", paste(intersect(tolower(coord_cols), dup_lc), collapse = ", "),
         " (columns: ", paste(colnames(coords_df), collapse = ", "), "). Give the file a header ",
         "row with one unique name per column.")
  }
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_df), collapse = ", "), ")")
  if (!all(vapply(coords_df[, coord_cols, drop = FALSE], is.numeric, logical(1)))) {
    stop("Coordinate columns ", paste(coord_cols, collapse = ", "), " in ",
         opts$spatial_coords_csv, " are not numeric.")
  }

  # What the two columns are, and whether they are positions. SpotSweeper finds each spot's neighbours by
  # Euclidean k-nearest-neighbour search on spatialCoords, and resolve_coord_cols (the fleet's shared copy)
  # ranks Visium's array_row/array_col above x/y. On Visium's hexagonal lattice those indices are not a
  # distance: every spot has the same parity of row + col, and in index space a spot's six neighbours sit
  # at 1.41 (four) and 2.0 (two) -- where two second-ring spots also sit -- so only 82% of each spot's 6
  # nearest index-space neighbours are its physical ones (83.5% of 36; measured on the library's Heart
  # Fetal12W sample). The converter's metadata.csv for a CELLxGENE Visium object carries exactly that pair
  # and no pixel columns. So: an x/y pair the file also carries is used instead, as tools/spotclean_worker.R
  # ranks it; hexagonal indices alone are refused unless allow_array_index_fallback converts them
  # (row x sqrt(3), which puts all six neighbours one pitch apart); a square lattice's are used as given.
  used_fallback <- FALSE
  coordinate_kind <- paste0("positions named '", coord_cols[1], "'/'", coord_cols[2], "', read as distances")
  index_pair <- any(vapply(ARRAY_INDEX_PAIRS, function(pair) identical(tolower(coord_cols), pair), logical(1)))
  if (index_pair) {
    lc_all <- tolower(colnames(coords_df))
    xy_cols <- colnames(coords_df)[match(c("x", "y"), lc_all)]
    xy_usable <- !anyNA(xy_cols) && sum(lc_all %in% c("x", "y")) == 2L &&
      all(vapply(coords_df[, xy_cols, drop = FALSE], is.numeric, logical(1)))
    if (xy_usable) {
      coordinate_kind <- paste0("positions named '", xy_cols[1], "'/'", xy_cols[2], "', read as distances in ",
                                "preference to the array indices '", coord_cols[1], "'/'", coord_cols[2],
                                "' the file also carries")
      coord_cols <- xy_cols
      log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "), " (x/y preferred to the array indices)")
    } else {
      lattice <- array_index_lattice(as.numeric(coords_df[[coord_cols[1]]]), as.numeric(coords_df[[coord_cols[2]]]))
      if (lattice == "hexagonal" && !isTRUE(opts$allow_array_index_fallback)) {
        stop("The coordinates file ", opts$spatial_coords_csv, " has no pixel columns (imagerow/imagecol, ",
             "pxl_row_in_fullres/pxl_col_in_fullres or x/y), and the pair it does have, '", coord_cols[1], "'/'",
             coord_cols[2], "', is Visium's hexagonal array index: every spot has the same parity of row + col. ",
             "In index space a spot's six neighbours lie at 1.41 and 2.0 where on the slide they are all one pitch ",
             "away, so SpotSweeper's nearest-neighbour search would pick the wrong neighbourhoods. Give pixel ",
             "coordinates -- Space Ranger's tissue_positions.csv as it is -- or pass allow_array_index_fallback=True ",
             "to read the indices as the lattice's physical layout (the row axis is ", coord_cols[1], " x sqrt(3), ",
             "the column axis is ", coord_cols[2], ").")
      }
      if (lattice == "hexagonal") {
        used_fallback <- TRUE
        coordinate_kind <- paste0(
          "Visium hexagonal array indices, converted to the lattice's physical layout (the row axis is ",
          coord_cols[1], " x sqrt(3), the column axis is ", coord_cols[2], ") because allow_array_index_fallback=True")
      } else if (lattice == "square") {
        coordinate_kind <- paste0(
          "array indices on a square lattice, read as positions (a square lattice's indices are distances up ",
          "to one scale, and nearest-neighbour ranks do not depend on the scale)")
      } else {
        coordinate_kind <- "named like array indices but not whole numbers, so read as positions"
      }
      warnings <- c(warnings, paste0(
        "No pixel coordinates in the coordinates file: '", coord_cols[1], "'/'", coord_cols[2], "' are ",
        coordinate_kind, "."))
      log_msg("WARNING: ", utils::tail(warnings, 1))
    }
  }
  spatial_coords <- as.matrix(coords_df[, coord_cols, drop = FALSE])
  if (used_fallback) {
    storage.mode(spatial_coords) <- "double"
    spatial_coords[, 1] <- spatial_coords[, 1] * sqrt(3)
  }
  colnames(spatial_coords) <- c("x", "y")

  n_genes  <- nrow(counts_mat)
  n_spots  <- ncol(counts_mat)
  log_msg("Input: ", n_genes, " genes x ", n_spots, " spots")

  # BiocNeighbors::findKNN caps k at n - 1 with a warning on stderr, so on a small slide the
  # neighbourhood SpotSweeper used is not the one that was asked for.
  n_neighbors_effective <- as.integer(min(opts$n_neighbors, n_spots - 1L))
  if (n_neighbors_effective < opts$n_neighbors) {
    warnings <- c(warnings, paste0(
      "n_neighbors=", opts$n_neighbors, " exceeds the ", n_spots, " spots analysed; SpotSweeper's ",
      "nearest-neighbour search caps it at ", n_neighbors_effective, " (every other spot)."))
    log_msg("WARNING: ", utils::tail(warnings, 1))
  }

  # --- Build SpatialExperiment ---
  log_msg("Building SpatialExperiment object...")
  spe <- SpatialExperiment(
    assays  = list(counts = as(counts_mat, "dgCMatrix")),
    spatialCoords = spatial_coords
  )

  # Add a sample_id column (required by SpotSweeper)
  spe$sample_id <- rep("sample1", ncol(spe))

  # --- Compute QC metrics with scuttle ---
  log_msg("Computing QC metrics via scuttle::addPerCellQCMetrics...")

  # Identify mitochondrial genes (MT- prefix for human, mt- for mouse)
  is_mito <- grepl("^(MT-|mt-)", rownames(spe))
  n_mito  <- sum(is_mito)
  log_msg("Mitochondrial genes found: ", n_mito, " / ", nrow(spe))

  if (n_mito > 0) {
    spe <- addPerCellQCMetrics(spe, subsets = list(Mito = is_mito))
  } else {
    spe <- addPerCellQCMetrics(spe)
    log_msg("WARNING: No mitochondrial genes found (MT-/mt- prefix). ",
            "Artifact detection will be skipped.")
  }

  qc_cols_before <- colnames(colData(spe))
  log_msg("QC columns after scuttle: ", paste(qc_cols_before, collapse = ", "))

  # --- Run localOutliers for multiple metrics ---
  log_msg("Running localOutliers (n_neighbors=", opts$n_neighbors,
          ", cutoff=", opts$threshold, ")...")

  # Library size outliers (lower = bad)
  spe <- localOutliers(spe,
                       metric      = "sum",
                       direction   = "lower",
                       n_neighbors = opts$n_neighbors,
                       samples     = "sample_id",
                       log         = TRUE,
                       cutoff      = opts$threshold)

  # Unique genes outliers (lower = bad)
  spe <- localOutliers(spe,
                       metric      = "detected",
                       direction   = "lower",
                       n_neighbors = opts$n_neighbors,
                       samples     = "sample_id",
                       log         = TRUE,
                       cutoff      = opts$threshold)

  # Mitochondrial ratio outliers (higher = bad) -- only if mito genes present
  has_mito_qc <- "subsets_Mito_percent" %in% colnames(colData(spe))
  if (has_mito_qc) {
    spe <- localOutliers(spe,
                         metric      = "subsets_Mito_percent",
                         direction   = "higher",
                         n_neighbors = opts$n_neighbors,
                         samples     = "sample_id",
                         log         = TRUE,
                         cutoff      = opts$threshold)
  }

  # --- Run findArtifacts (requires mito metrics) ---
  # Two different reasons leave no artifact call -- no mitochondrial genes, or findArtifacts
  # raising (its rlm/prcomp steps fail when every spot's mito ratio is the same, e.g. all zero) --
  # and the payload used to give the first reason for both, with the error only on stderr.
  run_artifacts <- has_mito_qc
  artifact_status <- if (has_mito_qc) "ran" else "skipped"
  artifact_error <- NULL
  if (run_artifacts) {
    log_msg("Running findArtifacts...")
    tryCatch({
      spe <- findArtifacts(spe,
                           mito_percent = "subsets_Mito_percent",
                           mito_sum     = "subsets_Mito_sum",
                           samples      = "sample_id",
                           n_rings      = 5,
                           log          = TRUE,
                           name         = "artifact")
      log_msg("findArtifacts completed")
    }, error = function(e) {
      log_msg("WARNING: findArtifacts failed: ", conditionMessage(e))
      run_artifacts <<- FALSE
      artifact_status <<- "failed"
      artifact_error <<- conditionMessage(e)
    })
  } else {
    log_msg("Skipping findArtifacts (no mitochondrial QC metrics available)")
  }
  if (identical(artifact_status, "failed")) {
    warnings <- c(warnings, paste0(
      "findArtifacts failed, so no artifact spots were called (the local outlier results are ",
      "complete): ", artifact_error))
  } else if (identical(artifact_status, "skipped")) {
    warnings <- c(warnings, paste0(
      "No gene name starts with MT- or mt- (Ensembl IDs are not recognised as mitochondrial), so ",
      "the mitochondrial-ratio outlier test and findArtifacts were not run."))
  }
  # findArtifacts needs 90 neighbours for its fifth ring (3 * i * (i + 1)).
  if (run_artifacts && n_spots - 1L < 90L) {
    warnings <- c(warnings, paste0(
      "findArtifacts' outer rings ask for up to 90 neighbours; with ", n_spots, " spots they were ",
      "capped at ", n_spots - 1L, "."))
  }

  # --- Extract results ---
  qc_results <- as.data.frame(colData(spe))
  qc_cols_after <- colnames(qc_results)
  new_cols <- setdiff(qc_cols_after, qc_cols_before)

  log_msg("New QC columns from SpotSweeper: ", paste(new_cols, collapse = ", "))

  # Count outliers
  outlier_cols <- grep("_outliers$", qc_cols_after, value = TRUE)
  outlier_counts <- list()
  for (col in outlier_cols) {
    n_outliers <- sum(qc_results[[col]], na.rm = TRUE)
    outlier_counts[[col]] <- n_outliers
    log_msg("  ", col, ": ", n_outliers, " outliers")
  }

  # Count artifacts. findArtifacts() writes a LOGICAL column -- FALSE for every spot, TRUE for the
  # ones it puts in the artifact cluster -- so the count is a sum, exactly like the outlier columns
  # above. Testing it against "" instead makes R coerce both operands to character, and
  # as.character(FALSE) is "FALSE", which is not "": every non-NA spot passed and the published
  # count was always the whole slide. as.logical() is identity on that column, so it changes no
  # answer, and it cannot raise on an unexpected type the way sum() would -- this line runs before
  # the CSV and RDS are written, and an error here would throw away statistics already computed.
  n_artifacts <- 0
  if ("artifact" %in% qc_cols_after) {
    n_artifacts <- sum(as.logical(qc_results$artifact), na.rm = TRUE)
    log_msg("Artifacts detected: ", n_artifacts)
  }

  # --- Save outputs ---
  qc_csv_path <- file.path(opts$output_dir, "spotsweeper_qc_results.csv")
  write_atomically(qc_csv_path, function(tmp) write.csv(qc_results, tmp, quote = TRUE))
  log_msg("Saved QC results to: ", qc_csv_path)

  rds_path <- file.path(opts$output_dir, "spotsweeper_spe.rds")
  write_atomically(rds_path, function(tmp) saveRDS(spe, file = tmp))
  log_msg("Saved SpatialExperiment RDS to: ", rds_path)

  # --- Build result ---
  artifact_summary <- list(
    artifacts_detected = n_artifacts,
    ran_findArtifacts  = run_artifacts,
    status             = artifact_status
  )
  if (!is.null(artifact_error)) artifact_summary$findArtifacts_error <- artifact_error
  if (identical(artifact_status, "skipped")) {
    artifact_summary$skipped_reason <- "no gene name starts with MT- or mt-"
  }

  metrics_run <- if (has_mito_qc) "sum, detected and subsets_Mito_percent" else "sum and detected"
  method <- paste0(
    "SpotSweeper localOutliers on ", metrics_run,
    switch(artifact_status,
           ran = "; findArtifacts (n_rings = 5)",
           failed = "; findArtifacts (n_rings = 5) failed",
           skipped = "; findArtifacts not run (no mitochondrial genes)"),
    if (used_fallback) {
      "; neighbours found on Visium hexagonal array indices converted to the lattice layout (allow_array_index_fallback)"
    } else {
      ""
    })

  artifact_sentence <- switch(
    artifact_status,
    ran = paste0(
      paste0("Artifact detection found ", n_artifacts, " artifact spots."),
      " findArtifacts always splits the slide into two k-means clusters on local mitochondrial ",
      "variance and labels one of them artifact, so this count is never zero; check where those ",
      "spots lie before removing them."),
    failed = paste0(
      "Artifact detection failed and called no artifact spots; findArtifacts raised: ",
      artifact_error),
    skipped = paste0(
      "Artifact detection and the mitochondrial-ratio outlier test were skipped: no gene name ",
      "starts with MT- or mt- (Ensembl IDs are not recognised as mitochondrial)."))

  payload <- list(
    status       = "ok",
    tool         = "spotsweeper",
    task         = "spatial_qc",
    data         = list(
      n_genes           = n_genes,
      n_spots           = n_spots,
      n_spots_in_counts = n_spots_in_counts,
      n_spots_without_coords = length(no_coords),
      n_spots_off_tissue = tissue$n_dropped,
      n_mito_genes      = n_mito,
      n_neighbors       = opts$n_neighbors,
      n_neighbors_effective = n_neighbors_effective,
      threshold         = opts$threshold
    ),
    params       = list(
      n_neighbors           = opts$n_neighbors,
      n_neighbors_effective = n_neighbors_effective,
      threshold             = opts$threshold,
      coordinate_columns    = I(coord_cols),
      coordinates_header    = coords_read$header,
      coordinate_kind       = coordinate_kind,
      allow_array_index_fallback = isTRUE(opts$allow_array_index_fallback),
      method                = method,
      used_fallback         = used_fallback,
      ignored               = I(character(0))
    ),
    output_files = list(
      qc_results_csv    = qc_csv_path,
      spe_rds           = rds_path
    ),
    outlier_summary = outlier_counts,
    artifact_summary = artifact_summary,
    qc_columns_added = new_cols,
    warnings     = I(warnings),
    analysis     = paste0(
      "SpotSweeper analyzed ", n_spots, " spots x ", n_genes, " genes",
      if (length(no_coords) > 0) {
        paste0(" (", length(no_coords), " of ", n_spots_in_counts,
               " counts spots had no coordinates and were left out)")
      } else {
        ""
      },
      if (tissue$n_dropped > 0) {
        paste0(" (", tissue$n_dropped, " of the ", n_spots_matched, " spots with coordinates have in_tissue == 0 ",
               "and were left out as background)")
      } else {
        ""
      },
      ". ",
      if (used_fallback) paste0("Neighbours were found on ", coordinate_kind, ". ") else "",
      "Local outlier detection (n_neighbors=", n_neighbors_effective,
      if (n_neighbors_effective < opts$n_neighbors) paste0(", capped from ", opts$n_neighbors) else "",
      ", cutoff=", opts$threshold, ") identified: ",
      paste(sapply(names(outlier_counts), function(col) {
        paste0(col, "=", outlier_counts[[col]])
      }), collapse = ", "),
      ". ",
      artifact_sentence
    )
  )
  if (!is.null(tissue$filter)) payload$params$in_tissue_filter <- tissue$filter
  payload
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args inside the handler: a bad flag used to abort R with nothing on stdout.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    sink(stderr())
    result <- run_spotsweeper(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "spotsweeper",
      task      = "spatial_qc",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
