#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(mistyR)
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
  message(sprintf("[mistyr-worker] %s", msg))
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

# Copied from tools/spotsweeper_worker.R (first_record_line + read_coords_csv), the reader the R workers
# share for a coordinates file; inlined for the same reason as the helpers above.
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

# The same rule as worker_utils.keep_in_tissue: a coordinates file that carries Space Ranger's
# in_tissue flag (tissue_positions*.csv, or the converter's metadata.csv of a CELLxGENE export, which
# lists every array spot) marks the background glass, and those spots are left out and counted rather
# than modelled as tissue. TRUE/"1"/1 is in tissue; anything else is not. A flag that marks no spot as
# in tissue is refused. Returns the ids kept.
in_tissue_ids <- function(coords_df, ids, path) {
  if (!("in_tissue" %in% colnames(coords_df))) return(ids)
  raw <- tolower(trimws(as.character(coords_df[ids, "in_tissue"])))
  raw[raw == "true"] <- "1"
  raw[raw == "false"] <- "0"
  flag <- suppressWarnings(as.numeric(raw))
  keep <- !is.na(flag) & flag == 1
  if (!any(keep)) {
    stop("The in_tissue column of ", path, " marks none of the ", length(ids), " spots analysed as in ",
         "tissue (values seen: ", paste(utils::head(sort(unique(raw)), 8), collapse = ", "), "); fix the ",
         "column so in-tissue spots are 1, or remove it if every spot is tissue.")
  }
  ids[keep]
}

# Each table is written to <path>.partial and moved into place, so a reader that opens it while it is
# being written sees the old file or the whole new one, never a truncated table under the name it
# trusts (rename(2) is atomic on one filesystem).
move_into_place <- function(partial, path) {
  if (!file.rename(partial, path)) stop("Could not move ", partial, " into place at ", path)
  invisible(path)
}

# The paraview weighs every other spot by mistyR's Gaussian kernel, exp(-d^2 / l^2), with d in the
# units the coordinates carry. The portal's old default l = 10 was mistyR's vignette number, and
# there it is 10 grid spacings: the vignette's synthetic slide (data("synthetic"), add_paraview(pos,
# l = 10)) sits on a unit grid whose median nearest-neighbour distance is exactly 1. Applied to
# full-resolution pixel coordinates -- what Space Ranger's pxl_*_in_fullres carry, and the
# converter's x/y whenever obsm['spatial'] does -- 10 is a fraction of one spot: at a 273 px pitch
# the weight of the nearest spot is exp(-(273/10)^2), which is 0 in double precision, so the
# paraview was identically zero while the run still reported paraview importances and an R2 gain
# (measured on the FFPE normal prostate slide, 290 px: all 400 para.10 importances were 0).
#
# So an unset l is measured in spot spacings, not coordinate units: 10 x the median distance from a
# spot to its nearest other spot, which is the vignette's own setting on the vignette's own grid and
# does not move when the same slide arrives in pixels, microns or array indices. An explicit l is
# used as given; if even the closest pair of spots gets a weight of exactly 0 the paraview cannot be
# anything but zero and the run stops, and an l below one spacing is reported in the payload.
PARAVIEW_SPACINGS <- 10

nearest_spot_distances <- function(coords) {
  m <- as.matrix(coords)
  storage.mode(m) <- "double"
  n <- nrow(m)
  if (n < 2) stop("mistyR needs at least two spots to build a paraview; the coordinates have ", n, ".")
  bad <- sum(!is.finite(m))
  if (bad > 0) {
    stop(bad, " coordinate value(s) are missing or not numbers, so spot distances (and the paraview) ",
         "cannot be computed. Fix or remove those spots in the coordinates file.")
  }
  nn <- distances::nearest_neighbor_search(distances::distances(m), k = 2L)
  # Column i holds spot i and its nearest other spot, in either order when two spots share a
  # position; either way the distance read is the one to the nearest spot that is not i itself.
  other <- ifelse(nn[1, ] == seq_len(n), nn[2, ], nn[1, ])
  sqrt(rowSums((m - m[other, , drop = FALSE])^2))
}

resolve_paraview_l <- function(l, nn_dist) {
  positive <- nn_dist[nn_dist > 0]
  if (length(positive) == 0) {
    stop("Every spot shares its position with another, so there is no spot spacing to set the paraview ",
         "bandwidth from and no distance for its kernel to weigh. Check the coordinates file.")
  }
  spacing <- stats::median(positive)
  auto <- is.null(l)
  if (auto) {
    l <- signif(PARAVIEW_SPACINGS * spacing, 3)
  } else if (length(l) != 1 || !is.finite(l) || l <= 0) {
    stop("l = ", paste(l, collapse = ", "), " is not a paraview bandwidth: give a positive number in the ",
         "units of the coordinates, or leave l unset for ", PARAVIEW_SPACINGS, "x the median spot spacing.")
  }
  closest <- min(nn_dist)
  if (exp(-(closest / l)^2) == 0) {
    stop("With l = ", signif(l, 4), " the paraview would be identically zero. Its Gaussian weight is ",
         "exp(-d^2 / l^2) with d in coordinate units, and even the closest pair of spots (",
         signif(closest, 4), " units apart; median spacing ", signif(spacing, 4), ") gets exp(-(",
         signif(closest, 4), "/", signif(l, 4), ")^2), which is 0 in double precision, so every paraview ",
         "feature would be 0 and MISTy would report importances for an empty view. Give l in the units of ",
         "the coordinates (for example l = ", signif(PARAVIEW_SPACINGS * spacing, 3), ", ", PARAVIEW_SPACINGS,
         "x the median spacing), or leave l unset to use that.")
  }
  note <- NULL
  if (!auto && l < spacing) {
    note <- paste0(
      "l = ", signif(l, 4), " is below the median spot spacing (", signif(spacing, 4), " coordinate units): ",
      "the paraview weight on a spot's typical nearest neighbour is exp(-(", signif(spacing, 4), "/",
      signif(l, 4), ")^2) = ", signif(exp(-(spacing / l)^2), 3), ", so the paraview reads little beyond ",
      "the nearest spots. Leaving l unset uses ", PARAVIEW_SPACINGS, "x the median spacing (l = ",
      signif(PARAVIEW_SPACINGS * spacing, 3), ")."
    )
  }
  list(l = l, auto = auto, spacing = spacing, closest = closest, note = note)
}

# mistyR::run_misty(seed = 42, ...) draws the CV folds with withr::with_seed(seed, ...) and its
# random_forest_model hands the same seed to ranger::ranger, whose documentation reads "Set to 0 to
# ignore the R seed": a seed of 0 makes ranger draw one from the system's random device, so every run
# fits different forests and reports different importances and R2 gains. 0 is the portal's default,
# so 0 runs as run_misty's own default seed, 42, and the payload reports both the seed asked for
# (params.seed) and the one used (params.seed_used). Any other seed is passed through unchanged.
MISTY_DEFAULT_SEED <- 42L

misty_seed_for <- function(seed) {
  if (isTRUE(seed == 0L)) MISTY_DEFAULT_SEED else seed
}


# Values are checked where they are parsed, and parsing runs inside main()'s handler: a bad value used
# to become NA with a coercion warning -- as.integer("3000000000") is NA, set.seed(NA) then aborted
# before any JSON was printed and the portal read an empty stdout -- or to be taken at face value:
# head(x, -3) keeps all but three rows, so --n-top -3 published nearly every interaction as the "top"
# ones while params.n_top said -3.
parse_int <- function(val, key, min_value = NULL) {
  parsed <- suppressWarnings(as.integer(val))
  if (is.na(parsed) || (!is.null(min_value) && parsed < min_value)) {
    stop(sprintf("%s expects an integer%s, got '%s'", key,
                 if (is.null(min_value)) " (within R's 32-bit integer range)" else sprintf(" >= %d", min_value),
                 val))
  }
  parsed
}

parse_num <- function(val, key) {
  parsed <- suppressWarnings(as.numeric(val))
  if (is.na(parsed) || !is.finite(parsed)) stop(sprintf("%s expects a finite number, got '%s'", key, val))
  parsed
}

parse_args <- function(args) {
  opts <- list(
    expression_csv = NULL,
    coords_csv     = NULL,
    output_dir     = NULL,
    l              = NULL,
    n_top          = 20L,
    bypass_intra   = FALSE,
    seed           = 0L
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]

    # Boolean flags
    if (key == "--bypass-intra") {
      opts$bypass_intra <- TRUE
      i <- i + 1L
      next
    }

    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    if (key == "--expression-csv") {
      opts$expression_csv <- val
    } else if (key == "--coords-csv") {
      opts$coords_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--l") {
      opts$l <- parse_num(val, "--l")
    } else if (key == "--n-top") {
      opts$n_top <- parse_int(val, "--n-top", min_value = 0L)
    } else if (key == "--seed") {
      opts$seed <- parse_int(val, "--seed")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_misty_worker <- function(opts) {
  if (is.null(opts$expression_csv) || is.null(opts$coords_csv) || is.null(opts$output_dir)) {
    stop("mistyR requires --expression-csv, --coords-csv, and --output-dir")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  log_msg("Reading expression data from: ", opts$expression_csv)
  expr_df <- read.csv(opts$expression_csv, row.names = 1, check.names = FALSE)

  log_msg("Reading coordinates from: ", opts$coords_csv)
  # Not read.csv(header = TRUE): Space Ranger's pre-2.0 tissue_positions_list.csv has no header, and
  # its first spot became the column names ('0','0','0',pxl,pxl), so x = y = in_tissue and the run was
  # refused as a slide whose spots all share one position.
  coords_read <- read_coords_csv(opts$coords_csv)
  coords_df <- coords_read$frame

  # Ensure matching IDs. mistyR wants markers in columns and spots in rows, but an expression CSV
  # arrives either way round; try the other orientation before aborting, as every other R worker
  # here does -- the abort below tells the reader both were tried. t() on a data.frame goes via a
  # matrix, so wrap it back up: mistyR's create_initial_view() takes a table, not a matrix.
  common_ids <- intersect(rownames(expr_df), rownames(coords_df))
  if (length(common_ids) == 0) {
    log_msg("0 common IDs, transposing expression table")
    expr_df <- as.data.frame(t(expr_df))
    common_ids <- intersect(rownames(expr_df), rownames(coords_df))
  }
  if (length(common_ids) == 0) {
    stop(id_mismatch_msg("IDs", "expression", rownames(expr_df), "coordinates data", rownames(coords_df)))}

  notes <- character(0)
  # Background spots (in_tissue == 0) are left out, counted among the spots both files share.
  n_spots_supplied <- length(common_ids)
  common_ids <- in_tissue_ids(coords_df, common_ids, opts$coords_csv)
  n_spots_off_tissue <- n_spots_supplied - length(common_ids)
  if (n_spots_off_tissue > 0) {
    notes <- c(notes, paste0(
      n_spots_off_tissue, " of ", n_spots_supplied, " spots have in_tissue == 0 in ", basename(opts$coords_csv),
      " (background outside the tissue) and were left out; ", length(common_ids), " in-tissue spots were analysed."
    ))
    log_msg(notes[length(notes)])
  }

  expr_df <- expr_df[common_ids, , drop = FALSE]

  # MISTy models every column as a target, and mistyR::run_misty refuses a target with zero variance
  # ("Targets ... have zero variance (they are noninformative). Remove them to proceed.") -- but only
  # after add_paraview's O(spots^2 x features) pass has run. convert_h5ad_to_csv, the documented way in,
  # writes every gene, and on the library's Visium samples a third of them are all-zero (11,678 of
  # 36,601 on V1_Breast_Cancer_Block_A_Section_1); leaving the background spots out above makes more of
  # them constant. So the features that are constant over the spots analysed are left out here, before
  # any view is built, and counted and named in data, params, the warnings and the analysis. A constant
  # feature carries nothing for MISTy to model or to predict from. A column that is not numeric, or
  # that has missing values, is refused by name: a random forest can do nothing with either.
  n_features_supplied <- ncol(expr_df)
  if (n_features_supplied == 0) {
    stop("The expression table ", opts$expression_csv, " has no feature columns besides the spot IDs.")
  }
  not_numeric <- colnames(expr_df)[!vapply(expr_df, is.numeric, logical(1))]
  if (length(not_numeric) > 0) {
    stop(length(not_numeric), " column(s) of ", opts$expression_csv, " are not numeric (",
         paste(utils::head(not_numeric, 5), collapse = ", "), if (length(not_numeric) > 5) ", ..." else "",
         "); every column after the spot ID must be a feature's expression.")
  }
  has_na <- colnames(expr_df)[vapply(expr_df, anyNA, logical(1))]
  if (length(has_na) > 0) {
    stop(length(has_na), " feature(s) of ", opts$expression_csv, " have missing values among the ",
         nrow(expr_df), " spots analysed (", paste(utils::head(has_na, 5), collapse = ", "),
         if (length(has_na) > 5) ", ..." else "", "); MISTy's random forests cannot use a missing value. ",
         "Fill or remove them.")
  }
  if (nrow(expr_df) < 2) {
    stop("mistyR needs at least two spots to model; ", nrow(expr_df), " spot(s) are left to analyse.")
  }
  feature_sd <- vapply(expr_df, stats::sd, numeric(1))
  varies <- is.finite(feature_sd) & feature_sd > 0
  zero_variance <- colnames(expr_df)[!varies]
  if (!any(varies)) {
    stop("All ", n_features_supplied, " features of ", opts$expression_csv, " are constant across the ",
         nrow(expr_df), " spots analysed, so MISTy has no target to model.")
  }
  if (length(zero_variance) > 0) {
    expr_df <- expr_df[, varies, drop = FALSE]
    notes <- c(notes, paste0(
      length(zero_variance), " of ", n_features_supplied, " features are constant across the ", nrow(expr_df),
      " spots analysed (zero variance, e.g. ", paste(utils::head(zero_variance, 5), collapse = ", "),
      if (length(zero_variance) > 5) ", ..." else "", ") and were left out: MISTy cannot model a constant ",
      "target, and a constant predictor carries no information. ", ncol(expr_df), " features were modelled."
    ))
    log_msg(notes[length(notes)])
  }

  coord_cols <- resolve_coord_cols(colnames(coords_df), opts$coords_csv)
  # The resolver matches names case-blind and takes the first match, so two columns that share a name
  # cannot be told apart; refuse that rather than model a degenerate slide.
  dup_lc <- unique(tolower(colnames(coords_df))[duplicated(tolower(colnames(coords_df)))])
  if (anyDuplicated(coord_cols) > 0L || any(tolower(coord_cols) %in% dup_lc)) {
    stop("The coordinate columns chosen in ", opts$coords_csv, " (", paste(coord_cols, collapse = ", "),
         ") are not two distinct columns: the file names more than one column ",
         paste(intersect(tolower(coord_cols), dup_lc), collapse = ", "), " (columns: ",
         paste(colnames(coords_df), collapse = ", "), "). Give the file a header row with one unique name ",
         "per column.")
  }
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_df), collapse = ", "), ")")
  coords_df <- coords_df[common_ids, coord_cols, drop = FALSE]
  colnames(coords_df) <- c("x", "y")

  # Sanitize column names for ranger formula interface:
  # replace hyphens, dots, colons etc with underscores
  original_names <- colnames(expr_df)
  safe_names <- make.names(original_names, unique = TRUE)
  colnames(expr_df) <- safe_names

  # make.names() is a one-way mangle, and mistyR keys every table it returns on the names it was
  # handed. HLA-DRA arrives back as HLA.DRA, and from the output alone that is indistinguishable
  # from a gene whose symbol really does contain a dot, so the caller cannot undo it. unique=TRUE
  # makes safe_names a bijection with original_names, so match() inverts it exactly -- including
  # the collision suffix it adds when two symbols mangle to the same string. Anything that is not
  # one of our feature names (view labels like para.10, "intercept", NA) misses and is left alone.
  restore_gene_names <- function(df, cols) {
    for (col in cols) {
      if (col %in% colnames(df)) {
        vals <- as.character(df[[col]])
        hit <- match(vals, safe_names)
        vals[!is.na(hit)] <- original_names[hit[!is.na(hit)]]
        df[[col]] <- vals
      }
    }
    df
  }

  log_msg("N spots = ", nrow(expr_df), ", N features = ", ncol(expr_df))

  paraview <- resolve_paraview_l(opts$l, nearest_spot_distances(coords_df))
  log_msg("Median spot spacing = ", signif(paraview$spacing, 4), " coordinate units; paraview l = ",
          paraview$l, if (paraview$auto) paste0(" (", PARAVIEW_SPACINGS, "x the spacing, l unset)") else " (given)")
  if (!is.null(paraview$note)) {
    notes <- c(notes, paraview$note)
    log_msg(paraview$note)
  }

  # Build MISTy views
  log_msg("Creating MISTy views (l = ", paraview$l, ")...")
  views <- create_initial_view(expr_df)
  views <- views %>% add_paraview(coords_df, l = paraview$l)

  if (isTRUE(opts$bypass_intra)) {
    log_msg("Bypassing intraview")
  }

  # Run MISTy
  misty_out <- file.path(opts$output_dir, "misty_results")
  dir.create(misty_out, showWarnings = FALSE, recursive = TRUE)
  # run_misty's own seed (default 42) is the one that draws the CV folds and seeds ranger; the global
  # set.seed() in main() reaches neither, so without it every run was seed 42 whatever was asked. It is
  # never handed a raw 0, which ranger reads as "no seed" (see misty_seed_for).
  misty_seed <- misty_seed_for(opts$seed)
  if (!identical(as.integer(misty_seed), as.integer(opts$seed))) {
    notes <- c(notes, paste0(
      "seed = ", opts$seed, " ran as mistyR's default seed ", misty_seed, " (params.seed_used): ranger, which ",
      "fits MISTy's random forests, reads a seed of 0 as 'no seed' and would draw a different one on every ",
      "run, so the importances and R2 gains could not be reproduced. Give any non-zero seed to use that seed."
    ))
    log_msg(notes[length(notes)])
  }
  log_msg("Running MISTy model (seed = ", misty_seed, ")...")
  mistyR::run_misty(views, results.folder = misty_out, seed = misty_seed, bypass.intra = opts$bypass_intra)

  # Collect results
  log_msg("Collecting MISTy results...")
  misty_results <- collect_results(misty_out)

  # Extract and save performance metrics
  performance_df <- misty_results$improvements
  performance_df <- restore_gene_names(performance_df, c("target"))
  perf_path <- file.path(opts$output_dir, "mistyr_performance.csv")
  perf_path_partial <- paste0(perf_path, ".partial")
  write.csv(performance_df, perf_path_partial, row.names = FALSE, quote = TRUE)
  move_into_place(perf_path_partial, perf_path)

  # Extract and save importances
  # Repaired before the write, and before top_interactions is sliced out of it below, so the
  # headline table inherits the user's own gene names too.
  importances_df <- misty_results$importances
  importances_df <- restore_gene_names(importances_df, c("Predictor", "Target"))
  imp_path <- file.path(opts$output_dir, "mistyr_importances.csv")
  imp_path_partial <- paste0(imp_path, ".partial")
  write.csv(importances_df, imp_path_partial, row.names = FALSE, quote = TRUE)
  move_into_place(imp_path_partial, imp_path)

  # Extract contributions
  contributions_df <- misty_results$contributions
  contributions_df <- restore_gene_names(contributions_df, c("target"))
  contrib_path <- file.path(opts$output_dir, "mistyr_contributions.csv")
  contrib_path_partial <- paste0(contrib_path, ".partial")
  write.csv(contributions_df, contrib_path_partial, row.names = FALSE, quote = TRUE)
  move_into_place(contrib_path_partial, contrib_path)

  # Identify top interactions
  if ("Importance" %in% colnames(importances_df)) {
    top_interactions <- head(
      importances_df[order(-importances_df$Importance), ],
      opts$n_top
    )
  } else {
    top_interactions <- head(importances_df, opts$n_top)
  }
  top_path <- file.path(opts$output_dir, "mistyr_top_interactions.csv")
  top_path_partial <- paste0(top_path, ".partial")
  write.csv(top_interactions, top_path_partial, row.names = FALSE, quote = TRUE)
  move_into_place(top_path_partial, top_path)

  # Mean R2 improvement. collect_results() returns `improvements` in long form --
  # one row per target x measure, the number in `value` -- so gain.R2 is one of the
  # values of `measure` and never a column. The colnames branch below is kept for a
  # shape that does report it as a column; it is the long form mistyR emits today.
  mean_r2_gain <- NA_real_
  if (all(c("measure", "value") %in% colnames(performance_df))) {
    gain_rows <- performance_df[performance_df$measure == "gain.R2", , drop = FALSE]
    if (nrow(gain_rows) > 0) {
      mean_r2_gain <- mean(gain_rows$value, na.rm = TRUE)
    }
  } else if ("gain.R2" %in% colnames(performance_df)) {
    mean_r2_gain <- mean(performance_df$gain.R2, na.rm = TRUE)
  }
  if (!is.finite(mean_r2_gain)) {
    notes <- c(notes, paste0(
      "No gain.R2 rows were found in the MISTy performance table, so the mean R2 gain ",
      "could not be computed. The per-target numbers are in ", basename(perf_path), "."
    ))
    log_msg(notes[length(notes)])
  }
  # Never round NA into the sentence: round(NA, 4) renders as the literal word "NA".
  gain_phrase <- if (is.finite(mean_r2_gain)) {
    paste0("mean R2 gain of ", round(mean_r2_gain, 4))
  } else {
    "no readable mean R2 gain"
  }

  l_phrase <- paste0(
    "paraview bandwidth l = ", paraview$l, " coordinate units (",
    signif(paraview$l / paraview$spacing, 3), "x the median spot spacing of ", signif(paraview$spacing, 4),
    if (paraview$auto) "; l was unset, so it was set from the spacing)" else ")"
  )

  payload <- list(
    status       = "ok",
    tool         = "mistyr",
    task         = "spatial_modeling",
    data         = list(
      n_spots            = nrow(expr_df),
      n_features         = ncol(expr_df),
      n_targets          = length(unique(importances_df$Target)),
      spot_spacing       = paraview$spacing,
      n_spots_supplied   = n_spots_supplied,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_features_supplied = n_features_supplied,
      n_features_dropped_zero_variance = length(zero_variance)
    ),
    output_files = list(
      performance_csv    = perf_path,
      importances_csv    = imp_path,
      contributions_csv  = contrib_path,
      top_interactions   = top_path,
      results_folder     = misty_out
    ),
    params       = list(
      method             = "mistyR::run_misty (intraview + Gaussian paraview, ranger random forests)",
      used_fallback      = FALSE,
      l_param            = paraview$l,
      l_auto             = paraview$auto,
      l_in_spot_spacings = paraview$l / paraview$spacing,
      paraview_family    = "gaussian",
      bypass_intra       = opts$bypass_intra,
      n_top              = opts$n_top,
      seed               = opts$seed,
      seed_used          = misty_seed,
      coord_columns_used = paste(coord_cols, collapse = ","),
      coords_header      = coords_read$header,
      n_features_dropped_zero_variance = length(zero_variance),
      zero_variance_features_dropped = I(utils::head(zero_variance, 20))
    ),
    summary      = list(
      mean_r2_gain       = mean_r2_gain
    ),
    analysis     = paste0(
      "MISTy spatial modeling analyzed ", length(unique(importances_df$Target)),
      " targets across ", nrow(expr_df), " spots with ", gain_phrase, "; ", l_phrase, ".",
      if (n_spots_off_tissue > 0) paste0(
        " ", n_spots_off_tissue, " of the ", n_spots_supplied, " spots supplied have in_tissue == 0 in the ",
        "coordinates file (background) and were left out before the views were built.") else "",
      if (length(zero_variance) > 0) paste0(
        " ", length(zero_variance), " of the ", n_features_supplied, " features supplied are constant across ",
        "the spots analysed (zero variance) and were left out; ", ncol(expr_df), " were modelled.") else ""
    )
  )
  if (n_spots_off_tissue > 0) {
    payload$params$in_tissue_filter <- list(
      n_spots_supplied           = n_spots_supplied,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_used               = n_spots_supplied - n_spots_off_tissue
    )
  }
  if (length(notes) > 0) {
    payload$warnings <- I(notes)
  }
  payload
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # Parsing sits inside the handler, as in neighborseq_worker.R: a bad value ("--seed 3000000000",
  # "--n-top -3") is refused in the JSON payload the portal reads, not as a bare R error with nothing on
  # stdout.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    set.seed(opts$seed)
    run_misty_worker(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "mistyr",
      task      = "spatial_modeling",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
