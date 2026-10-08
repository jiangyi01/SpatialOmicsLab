#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(BASS)
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
  message(sprintf("[bass-worker] %s", msg))
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


# Inlined rather than sourced, copied from tools/spotsweeper_worker.R (first_record_line and
# read_coords_csv), which carries the measurements behind it: each worker runs as its own Rscript
# in its own conda env, so there is no shared library on the path.
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

# Background spots. Space Ranger's positions files flag every array spot in_tissue 0/1, and a count
# matrix that keeps the whole array (a raw matrix, or a CELLxGENE export) carries the background
# spots too, so they reached the model as if they were tissue. worker_utils.keep_in_tissue leaves
# them out of the Python workers that analyse spots -- background is left out by default and
# reported -- and this is the same rule for a worker whose flag arrives in the coordinates file. No
# in_tissue column, or one that is 1 for every spot analysed, changes nothing; TRUE/"1"/1 count as
# in tissue and any other value as background; a column that marks none of the spots as tissue is
# refused rather than analysed as an empty slide.
keep_in_tissue_spots <- function(coords_df, spots, source_path) {
  n <- length(spots)
  unchanged <- list(spots = spots, n_supplied = n, n_dropped = 0L)
  if (!("in_tissue" %in% colnames(coords_df))) return(unchanged)
  raw <- coords_df[spots, "in_tissue"]
  flag <- tolower(trimws(as.character(raw)))
  flag[flag %in% "true"] <- "1"
  flag[flag %in% "false"] <- "0"
  value <- suppressWarnings(as.numeric(flag))
  keep <- !is.na(value) & value == 1
  n_keep <- sum(keep)
  if (n_keep == n) return(unchanged)
  if (n_keep == 0L) {
    seen <- utils::head(sort(unique(as.character(raw))), 8)
    stop("The in_tissue column of ", source_path, " marks none of the ", n, " spots that have ",
         "counts as in tissue (values seen: ", paste(seen, collapse = ", "), "); fix the column so ",
         "in-tissue spots are 1, or remove it if every spot is tissue.")
  }
  list(spots = spots[keep], n_supplied = n, n_dropped = n - n_keep)
}

# The payload shape worker_utils.record_in_tissue writes for the Python workers.
in_tissue_params <- function(tissue) {
  list(n_spots_supplied           = tissue$n_supplied,
       n_spots_off_tissue_dropped = tissue$n_dropped,
       n_spots_used               = tissue$n_supplied - tissue$n_dropped)
}

in_tissue_warning <- function(tissue, source_path) {
  paste0(tissue$n_dropped, " of ", tissue$n_supplied, " spots with counts and coordinates have ",
         "in_tissue == 0 in ", source_path, " (background outside the tissue) and were left out; ",
         tissue$n_supplied - tissue$n_dropped, " in-tissue spots were kept.")
}

# The resolver matches names case-blind and takes the first match, so two columns that share a name
# -- or one column picked for both axes -- cannot be told apart. That is what a data row read as a
# header looks like; it is refused rather than analysed as a degenerate slide (the same check as
# tools/spotsweeper_worker.R).
check_coord_cols <- function(coords_df, coord_cols, source_path) {
  lc <- tolower(colnames(coords_df))
  dup_lc <- unique(lc[duplicated(lc)])
  if (anyDuplicated(coord_cols) > 0L || any(tolower(coord_cols) %in% dup_lc)) {
    stop("The coordinate columns chosen in ", source_path, " (",
         paste(coord_cols, collapse = ", "), ") are not two distinct columns: the file names ",
         "more than one column ", paste(intersect(tolower(coord_cols), dup_lc), collapse = ", "),
         " (columns: ", paste(colnames(coords_df), collapse = ", "), "). Give the file a header ",
         "row with one unique name per column.")
  }
  if (!all(vapply(coords_df[, coord_cols, drop = FALSE], is.numeric, logical(1)))) {
    stop("Coordinate columns ", paste(coord_cols, collapse = ", "), " in ", source_path,
         " are not numeric.")
  }
  invisible(coord_cols)
}

# Every spot at one position is not a layout. Reading the in_tissue flag as both axes put every
# in-tissue spot at (1, 1), and the spatial model built its neighbourhood on that single point while
# the run reported status ok.
check_not_one_point <- function(xy, coord_cols, source_path) {
  if (nrow(xy) >= 2L && length(unique(paste(xy[[1]], xy[[2]]))) == 1L) {
    stop("All ", nrow(xy), " spots sit at one position (", coord_cols[1], " = ", xy[[1]][1], ", ",
         coord_cols[2], " = ", xy[[2]][1], ") in ", source_path, ", so the coordinates describe ",
         "no layout and the spatial model would see a single point. Pass a coordinates file whose ",
         "two axis columns hold each spot's position.")
  }
  invisible(xy)
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

# What runs. This worker has one path -- the four upstream calls in run_bass() -- so there is no
# substitute to fall back to, and params.used_fallback is always FALSE.
METHOD_NAME <- "BASS (createBASSObject -> BASS.preprocess -> BASS.run -> BASS.postprocess)"

# BASS.preprocess's own defaults, passed to it by name so the payload reports the values that ran
# rather than values it assumes. With more than BASS_N_SE genes in the file BASS keeps the
# BASS_N_SE most significant SPARK-X spatially expressed genes (the whole panel otherwise), drops
# any gene with no count, and reduces what is left to BASS_N_PC principal components: the MCMC
# never sees the genes, only those components.
BASS_GENE_SELECT <- "sparkx"
BASS_N_SE <- 3000L
BASS_N_PC <- 20L

# A reader that opens the file while it is being written sees either the previous version or the
# whole new one, never a truncated table (rename(2) is atomic on one filesystem).
write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = FALSE, quote = TRUE)
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

# as.integer("abc") is NA with only a warning, and an NA domain or iteration count used to travel on
# into kmeans or the sampler and die there with a message about something else.
parse_int_at_least <- function(val, key, lowest) {
  parsed <- suppressWarnings(as.integer(val))
  if (is.na(parsed) || parsed < lowest) {
    stop(sprintf("%s expects an integer >= %d, got '%s'", key, lowest, val))
  }
  parsed
}

# Values that are not whole numbers, counted a block of columns at a time so a whole-transcriptome
# slide does not need a second full-size matrix just to be checked.
count_non_integer <- function(m, chunk = 512L) {
  n <- 0
  if (ncol(m) == 0) return(n)
  for (start in seq(1L, ncol(m), by = chunk)) {
    block <- m[, start:min(ncol(m), start + chunk - 1L), drop = FALSE]
    n <- n + sum(block != round(block))
  }
  n
}

# The requested count is an input; the occupied count is what the run produced. When they agree the
# number is still the caller's, so it is not presented as something the data showed.
count_note <- function(noun, found, requested, param, bass_arg) {
  if (found == requested) {
    paste0(" NOTE: ", requested, " ", noun, "s were supplied by the caller (", param, ", BASS's ",
           bass_arg, "), so that count is an input and not a finding; this run does not show that ",
           "the data supports ", found, ".")
  } else {
    paste0(" NOTE: ", requested, " ", noun, "s were requested (", param, ", BASS's ", bass_arg,
           ") but ", found, " are occupied after BASS's post-processing; treat ", found,
           " as the method's own answer.")
  }
}

parse_args <- function(args) {
  if (length(args) > 0 && args[[1]] %in% c("--help", "-h")) {
    cat("Usage: bass_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --n-clusters INT           Number of spatial domains, BASS's R (default: 7)\n")
    cat("  --n-cell-types INT         Number of cell types, BASS's C (default: 5)\n")
    cat("  --burn-in INT              MCMC burn-in iterations (default: 500)\n")
    cat("  --n-samples INT            Posterior samples after burn-in (default: 1000)\n")
    cat("  --seed INT                 Random seed (default: 42)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    output_dir         = NULL,
    n_clusters         = 7L,
    n_cell_types       = 5L,
    burn_in            = 500L,
    n_samples          = 1000L,
    seed               = 42L
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
    } else if (key == "--n-clusters") {
      opts$n_clusters <- parse_int_at_least(val, key, 1L)
    } else if (key == "--n-cell-types") {
      opts$n_cell_types <- parse_int_at_least(val, key, 1L)
    } else if (key == "--burn-in") {
      opts$burn_in <- parse_int_at_least(val, key, 0L)
    } else if (key == "--n-samples") {
      opts$n_samples <- parse_int_at_least(val, key, 1L)
    } else if (key == "--seed") {
      opts$seed <- parse_int_at_least(val, key, -.Machine$integer.max)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_bass <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$output_dir)) {
    stop("BASS requires --spatial-counts-csv, --spatial-coords-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  warnings <- character(0)

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  counts_mat <- as.matrix(counts_df)
  rm(counts_df)

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Header-aware: Space Ranger's headerless tissue_positions_list.csv used to lose its first spot to
  # the header and have its in_tissue flag read as both axes, so every in-tissue spot sat at (1, 1).
  coords_read <- read_coords_csv(opts$spatial_coords_csv)
  coords_df <- coords_read$frame

  # Ensure genes x spots orientation
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  if (length(common_spots) == 0) {
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}

  # A spot in the counts with no coordinates cannot be placed on the slide, and a coordinate row
  # with no counts has nothing to cluster. The intersect above drops both; they are counted here so
  # the payload can say so instead of reporting a smaller slide as if it were the whole one.
  n_spots_input <- ncol(counts_mat)
  n_spots_without_coords <- n_spots_input - length(common_spots)
  n_coords_without_counts <- nrow(coords_df) - length(common_spots)
  if (n_spots_without_coords > 0) {
    warnings <- c(warnings, paste0(
      n_spots_without_coords, " of ", n_spots_input, " spots in the counts have no row in the ",
      "coordinates file and were left out: BASS needs a position for every spot it models."))
  }

  # Background spots (in_tissue == 0) are left out and reported, as the Python workers do.
  tissue <- keep_in_tissue_spots(coords_df, common_spots, opts$spatial_coords_csv)
  n_spots_off_tissue <- tissue$n_dropped
  if (n_spots_off_tissue > 0) {
    common_spots <- tissue$spots
    warnings <- c(warnings, in_tissue_warning(tissue, opts$spatial_coords_csv))
    log_msg("WARNING: ", utils::tail(warnings, 1))
  }

  counts_mat <- counts_mat[, common_spots, drop = FALSE]
  coord_cols <- resolve_coord_cols(colnames(coords_df), opts$spatial_coords_csv)
  check_coord_cols(coords_df, coord_cols, opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_df), collapse = ", "), ")")
  coords_df <- coords_df[common_spots, coord_cols, drop = FALSE]
  colnames(coords_df) <- c("x", "y")
  check_not_one_point(coords_df, coord_cols, opts$spatial_coords_csv)

  # --- What the values are ---
  # BASS.preprocess log-normalises by each spot's total, so a missing or negative value either
  # stops it with a message about size factors or turns into NaN inside the PCA. Named here instead.
  if (!is.numeric(counts_mat)) {
    stop("The counts CSV holds non-numeric values; BASS needs a numeric genes x spots count matrix ",
         "with gene names in the first column.")
  }
  if (anyNA(counts_mat)) {
    stop("The counts CSV has ", sum(is.na(counts_mat)), " missing value(s). BASS log-normalises ",
         "each spot by its total count, which a missing value leaves undefined. Write 0 for a gene ",
         "that was not detected, or remove the genes or spots that carry the gaps.")
  }
  min_value <- min(counts_mat)
  if (min_value < 0) {
    stop("The counts CSV has ", sum(counts_mat < 0), " negative value(s) (minimum ", min_value,
         "). BASS log-normalises counts, and a negative value means this matrix is scaled or ",
         "centred expression rather than counts. Pass the raw counts.")
  }
  # Not refused: BASS runs on a normalised matrix and always has. It is said out loud because
  # BASS.preprocess log-normalises whatever it is given and SPARK-X models counts, so a CELLxGENE X
  # (already log-normalised) is transformed twice.
  n_values_non_integer <- count_non_integer(counts_mat)
  if (n_values_non_integer > 0) {
    warnings <- c(warnings, paste0(
      n_values_non_integer, " value(s) in the counts are not whole numbers (maximum ",
      signif(max(counts_mat), 4), "), so the matrix looks normalised rather than raw counts. ",
      "BASS log-normalises its input and SPARK-X assumes counts, so a normalised matrix is ",
      "transformed a second time; pass the raw counts for the model BASS describes."))
  }

  # --- Spots with no counts ---
  # scater::normalizeCounts, the first thing BASS.preprocess calls, stops with "size factors should
  # be positive" on a spot whose total count is zero. CELLxGENE Visium exports keep off-tissue spots
  # as all-zero columns (3,396 of 4,992 on the Gastrocnemius donor 1 slide), so every such run
  # failed. An empty spot carries no expression for BASS to assign; it is left out here, where it
  # can be counted, and bass_domains.csv lists exactly the spots that were modelled.
  spot_totals <- colSums(counts_mat)
  empty_spots <- spot_totals == 0
  n_spots_empty <- sum(empty_spots)
  if (n_spots_empty == length(spot_totals)) {
    stop("All ", length(spot_totals), " spots that have coordinates have zero total counts, so ",
         "there is nothing for BASS to model.")
  }
  if (n_spots_empty > 0) {
    log_msg("Leaving out ", n_spots_empty, " spot(s) with zero total counts")
    counts_mat <- counts_mat[, !empty_spots, drop = FALSE]
    coords_df <- coords_df[!empty_spots, , drop = FALSE]
    common_spots <- common_spots[!empty_spots]
    warnings <- c(warnings, paste0(
      n_spots_empty, " of ", length(spot_totals), " spots have zero total counts (typically ",
      "off-tissue spots) and were left out: BASS normalises each spot by its total, which is ",
      "undefined for an empty spot. bass_domains.csv lists the ", ncol(counts_mat),
      " spots that were modelled."))
  }

  n_genes <- nrow(counts_mat)
  n_spots <- ncol(counts_mat)

  # --- What BASS.preprocess will do with the genes ---
  # The same rule it applies: SPARK-X selection when the file has more than BASS_N_SE rows (BASS@P
  # counts every row, detected or not), then every gene with no count is dropped (SPARK-X drops those
  # itself). The payload used to report only n_genes -- the rows in the file -- as though the model
  # had seen all of them.
  n_genes_with_counts <- sum(rowSums(counts_mat) > 0)
  use_sparkx <- n_genes > BASS_N_SE
  n_genes_used <- if (use_sparkx) min(BASS_N_SE, n_genes_with_counts) else n_genes_with_counts
  # prcomp returns min(spots, genes) components and BASS.preprocess keeps columns 1:BASS_N_PC, so a
  # smaller input stops there with "subscript out of bounds".
  if (min(n_genes_used, n_spots) < BASS_N_PC) {
    stop("BASS reduces the expression to ", BASS_N_PC, " principal components, which needs at ",
         "least ", BASS_N_PC, " genes with counts and ", BASS_N_PC, " spots; this input has ",
         n_genes_used, " gene(s) with counts and ", n_spots, " spot(s) with counts and coordinates.")
  }

  log_msg("Input: ", n_genes, " genes x ", n_spots, " spots (", n_genes_with_counts,
          " genes with counts)")

  # --- Prepare BASS inputs ---
  # BASS expects a list of count matrices (one per sample) and a list of
  # coordinate data frames (one per sample). For a single-sample run we
  # wrap them in length-1 lists.
  log_msg("Creating BASS object (C=", opts$n_cell_types, " cell types, R=", opts$n_clusters,
          " spatial domains)...")

  set.seed(opts$seed)

  # BASS expects counts as genes x spots matrices and coords as data.frames
  cnts_list <- list(counts_mat)
  coords_list <- list(data.frame(x = coords_df$x, y = coords_df$y,
                                 row.names = rownames(coords_df)))

  # In BASS, C is the number of cell types and R the number of spatial domains
  # (createBASSObject.Rd). BASS.run seeds init_c from kmeans(centers = C) and init_z from
  # kmeans(centers = R), and results$z -- read below as the domain column -- is the R-component
  # labelling. The two used to be passed the other way round, so the domain column came back with
  # up to n_cell_types labels and the cell_type column with up to n_clusters.
  bass_obj <- createBASSObject(
    X = cnts_list,
    xy = coords_list,
    C = opts$n_cell_types,
    R = opts$n_clusters,
    init_method = "kmeans",
    beta_method = "SW",
    burnin  = opts$burn_in,
    nsample = opts$n_samples
  )

  # --- Preprocess ---
  log_msg("Preprocessing (log-normalization, ",
          if (use_sparkx) paste0("top-", BASS_N_SE, " SPARK-X gene selection, ") else "",
          BASS_N_PC, "-component PCA)...")
  bass_obj <- BASS.preprocess(bass_obj, doLogNormalize = TRUE, geneSelect = BASS_GENE_SELECT,
                              doPCA = TRUE, scaleFeature = TRUE, nSE = BASS_N_SE, nPC = BASS_N_PC)

  # --- Run BASS ---
  log_msg("Running BASS MCMC (burnin=", opts$burn_in, ", nsample=", opts$n_samples, ")...")
  bass_obj <- BASS.run(bass_obj)

  # --- Post-process ---
  log_msg("Post-processing BASS results...")
  bass_obj <- BASS.postprocess(bass_obj)

  # --- Extract results ---
  log_msg("Extracting domain labels and cell type clusters...")

  # Spatial domain labels (z): list of vectors, one per sample
  domain_labels <- bass_obj@results$z[[1]]
  # Cell type labels (c): list of vectors, one per sample
  celltype_labels <- bass_obj@results$c[[1]]

  # Build output data frame
  results_df <- data.frame(
    spot_id     = common_spots,
    domain      = domain_labels,
    cell_type   = celltype_labels,
    x           = coords_df$x,
    y           = coords_df$y,
    stringsAsFactors = FALSE
  )

  # --- Save outputs ---
  domain_path <- file.path(opts$output_dir, "bass_domains.csv")
  write_csv_atomic(results_df, domain_path)

  rds_path <- file.path(opts$output_dir, "bass_result.rds")
  save_rds_atomic(bass_obj, rds_path)

  log_msg("Saved domain labels to: ", domain_path)
  log_msg("Saved BASS object to: ", rds_path)

  # --- Summary statistics ---
  domain_counts <- as.list(table(domain_labels))
  celltype_counts <- as.list(table(celltype_labels))
  n_domains_found <- length(unique(domain_labels))
  n_cell_types_found <- length(unique(celltype_labels))

  gene_sentence <- if (use_sparkx) {
    paste0(" BASS log-normalised the counts, kept the ", n_genes_used, " most significant SPARK-X ",
           "spatially expressed genes of the ", n_genes_with_counts, " with counts (", n_genes,
           " genes supplied), and reduced them to ", BASS_N_PC, " principal components.")
  } else {
    paste0(" BASS log-normalised the counts and reduced the ", n_genes_used, " genes with counts (",
           n_genes, " supplied) to ", BASS_N_PC, " principal components; the file has no more than ",
           BASS_N_SE, " genes, so no SPARK-X gene selection ran.")
  }
  analysis <- paste0(
    "BASS assigned ", n_spots, " spots to ", n_domains_found, " spatial domains and ",
    n_cell_types_found, " cell-type clusters (Bayesian multi-scale model; ", opts$burn_in,
    " burn-in and ", opts$n_samples, " posterior MCMC samples, seed ", opts$seed, ").",
    gene_sentence,
    count_note("spatial domain", n_domains_found, opts$n_clusters, "n_clusters", "R"),
    count_note("cell-type cluster", n_cell_types_found, opts$n_cell_types, "n_cell_types", "C")
  )
  if (n_spots_empty > 0) {
    analysis <- paste0(analysis, " NOTE: ", n_spots_empty, " of ", n_spots_empty + n_spots,
                       if (n_spots_off_tissue > 0) " in-tissue" else "",
                       " spots with coordinates had zero total counts and were left out, so the ",
                       "domains above describe the ", n_spots, " spots with counts, not the whole ",
                       "array.")
  }
  if (n_spots_off_tissue > 0) {
    analysis <- paste0(analysis, " NOTE: ", n_spots_off_tissue, " of ", tissue$n_supplied,
                       " spots with counts and coordinates are marked in_tissue == 0 (background ",
                       "outside the tissue) in the coordinates file and were left out.")
  }
  if (n_spots_without_coords > 0) {
    analysis <- paste0(analysis, " NOTE: ", n_spots_without_coords, " of ", n_spots_input,
                       " spots in the counts had no coordinates and were left out.")
  }
  if (n_values_non_integer > 0) {
    analysis <- paste0(analysis, " NOTE: the input is not integer counts (", n_values_non_integer,
                       " non-integer values), so it was normalised a second time by BASS.")
  }

  params <- list(
    n_clusters       = opts$n_clusters,
    n_cell_types     = opts$n_cell_types,
    burn_in          = opts$burn_in,
    n_samples        = opts$n_samples,
    seed             = opts$seed,
    method           = METHOD_NAME,
    used_fallback    = FALSE,
    bass_R           = opts$n_clusters,
    bass_C           = opts$n_cell_types,
    init_method      = "kmeans",
    beta_method      = "SW",
    gene_selection   = if (use_sparkx) BASS_GENE_SELECT else "none",
    n_se_genes       = BASS_N_SE,
    n_pcs            = BASS_N_PC,
    coordinate_columns = I(coord_cols),
    coordinates_header = coords_read$header
  )
  if (n_spots_off_tissue > 0) params$in_tissue_filter <- in_tissue_params(tissue)

  result <- list(
    status       = "ok",
    tool         = "bass",
    task         = "clustering",
    data         = list(
      n_genes                 = n_genes,
      n_genes_with_counts     = n_genes_with_counts,
      n_genes_used            = n_genes_used,
      n_pcs                   = BASS_N_PC,
      n_spots                 = n_spots,
      n_spots_input           = n_spots_input,
      n_spots_without_coords  = n_spots_without_coords,
      n_coords_without_counts = n_coords_without_counts,
      n_spots_empty           = n_spots_empty,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_values_non_integer    = n_values_non_integer,
      n_clusters              = n_domains_found,
      n_cell_types            = n_cell_types_found,
      n_domains_found         = n_domains_found,
      n_cell_types_found      = n_cell_types_found
    ),
    output_files = list(
      domains_csv      = domain_path,
      result_rds       = rds_path
    ),
    params       = params,
    summary      = list(
      n_domains_found    = n_domains_found,
      n_cell_types_found = n_cell_types_found,
      domain_counts      = domain_counts,
      celltype_counts    = celltype_counts
    ),
    analysis     = analysis
  )
  if (length(warnings) > 0) result$warnings <- I(warnings)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  res <- tryCatch(with_r_traceback({
    # Inside the handler, so a bad flag ("--n-clusters abc") comes back as an error payload on
    # stdout instead of an R abort that leaves the portal no JSON to read.
    opts <- parse_args(args)
    # Redirect R-level stdout to stderr. Note: BASS internals (C/Rcpp)
    # may still print to fd 1 bypassing R's sink(); the Python wrapper's
    # _parse_result scans from the last stdout line for JSON, so this
    # noise is harmless.
    sink(stderr())
    result <- run_bass(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "bass",
      task      = "clustering",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
