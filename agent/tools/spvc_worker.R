#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(spVC)
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
  message(sprintf("[spvc-worker] %s", msg))
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


parse_args <- function(args) {
  opts <- list(
    counts_csv   = NULL,
    coords_csv   = NULL,
    output_dir   = NULL,
    n_top        = 100L,
    pval_cutoff  = 0.05,
    seed         = 0L,
    # No cap unless the caller sets one: every spot handed in is analysed. The flag's value is
    # checked in run_spvc, inside the error handler, so a bad value comes back as a payload.
    max_spots    = NA_integer_,
    max_genes    = 1000L
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    if (key == "--counts-csv") {
      opts$counts_csv <- val
    } else if (key == "--coords-csv") {
      opts$coords_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-top") {
      # Checked in run_spvc (as_whole_number), inside the error handler: as.integer() alone
      # truncated 2.7 to 2 and passed -5 through to head(), which reads it as "all but the last 5".
      opts$n_top <- val
    } else if (key == "--pval-cutoff") {
      opts$pval_cutoff <- val
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else if (key == "--max-spots") {
      opts$max_spots <- val
    } else if (key == "--max-genes") {
      opts$max_genes <- val
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# The spot cap is opt-in. NA (the default, or an explicit "NA"/"none") means every spot is
# analysed; anything else has to be a whole number of at least 1. as.integer() alone turned a
# typo into NA -- which now means "no cap" -- with nothing but a coercion warning on stderr.
as_spot_cap <- function(val) {
  if (is.null(val) || (length(val) == 1 && is.na(val))) return(NA_integer_)
  txt <- trimws(as.character(val))
  if (length(txt) == 1 && toupper(txt) %in% c("NA", "NONE", "NULL")) return(NA_integer_)
  num <- suppressWarnings(as.numeric(txt))
  if (length(num) != 1 || !is.finite(num) || num < 1 || num != floor(num)) {
    stop("max_spots must be a whole number of at least 1, or omitted to analyse every spot; got '",
         paste(txt, collapse = " "), "'.")
  }
  as.integer(num)
}

# A count option: a whole number of at least `minimum`, or the run stops naming the option. Checked
# in run_spvc so the refusal comes back as a payload. as.integer() alone took n_top = -5 to head(),
# which returns every gene but the last five -- published as the top SVGs at status "ok" -- and
# max_genes = 0 to order(...)[1:0], which is one gene, reported as "the 0 highest-variance genes".
as_whole_number <- function(val, name, minimum, what) {
  txt <- trimws(as.character(val))
  num <- suppressWarnings(as.numeric(txt))
  if (length(num) != 1 || !is.finite(num) || num < minimum || num != floor(num)) {
    stop(name, " must be a whole number of at least ", minimum, " (", what, "); got '",
         paste(txt, collapse = " "), "'.")
  }
  as.integer(num)
}

# The BH cutoff for spvc_significant_svgs.csv. A value that is not a number made every comparison
# NA, and R's `[` returns an all-NA row for each NA index, so the significant table filled with
# empty rows that n_significant then counted.
as_pval_cutoff <- function(val) {
  txt <- trimws(as.character(val))
  num <- suppressWarnings(as.numeric(txt))
  if (length(num) != 1 || !is.finite(num)) {
    stop("pval_cutoff must be a number (the BH-adjusted p-value cutoff, e.g. 0.05); got '",
         paste(txt, collapse = " "), "'.")
  }
  num
}

# Copied from tools/spotsweeper_worker.R (first_record_line verbatim; read_coords_csv with the one
# difference named below), so a headerless Space Ranger v1 tissue_positions_list.csv is read as
# coordinates. read.csv(header = TRUE) took its first spot as the header, so the columns came back
# named "0", "0", "0", <pixel>, <pixel> -- that spot's own values -- the resolver returned "0" for
# both axes, coords_raw[["0"]] is the in_tissue flag, and spVC was fitted with every spot at
# x == y in {0, 1}; the first spot was lost to the header too.
#
# The first line is a header unless it looks like data: an identifier that is not an
# identifier-column label, followed by nothing but numbers. A pandas frame written with integer
# column names (",0,1" or "barcode,0,1") keeps reading as a header. So does one whose index label
# is not in that list when it has four or more fields named exactly 0, 1, 2, ... (pandas'
# RangeIndex). A headerless file is read under Space Ranger's own column names when it has Space
# Ranger's six columns (and column 2 is a 0/1 tissue flag), as barcode,x,y when it has three, and
# refused otherwise: any other layout would be a guess at which two columns are the axes.
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

# The one difference from spotsweeper's copy: the frame keeps the spot identifiers as its first
# column and every value as text (colClasses = "character"), the shape this worker has always read.
# read.csv(row.names = 1) type-converts the identifiers first, so "0001" became "1" and no longer
# matched the counts header, which keeps its names as text.
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
    return(list(frame = read.csv(path, check.names = FALSE, colClasses = "character"),
                header = "present"))
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
  frame <- read.csv(path, header = FALSE, check.names = FALSE, colClasses = "character",
                    col.names = c("barcode", axis_names))
  if (length(fields) == 6L) {
    flag <- suppressWarnings(as.numeric(frame$in_tissue))
    if (!all(!is.na(flag) & flag %in% c(0, 1))) {
      stop("Coordinates file ", path, " has no header row that names its columns (its first line, ",
           shown, ", is an identifier followed only by numbers, so it reads as a spot) and six ",
           "columns, but its second column holds values other than 0 and 1, so it is not Space ",
           "Ranger's in_tissue flag and the file is not tissue_positions_list.csv. ", rename_hint)
    }
  }
  log_msg("Coordinates file has no header row; ", header)
  list(frame = frame, header = header)
}

# worker_utils.keep_in_tissue's rule, by hand: a spot is tissue when its in_tissue flag reads 1 (or
# TRUE); anything else -- 0, FALSE, a blank -- is background. Returns the logical keep vector.
in_tissue_keep <- function(flag) {
  flag <- tolower(trimws(as.character(flag)))
  flag[flag == "true"] <- "1"
  flag[flag == "false"] <- "0"
  value <- suppressWarnings(as.numeric(flag))
  !is.na(value) & value == 1
}

# Values that are not whole numbers, counted a block of columns at a time so a whole-transcriptome
# slide does not need a second full-size matrix just to be checked. Copied from tools/bass_worker.R.
count_non_integer <- function(m, chunk = 512L) {
  n <- 0
  if (ncol(m) == 0) return(n)
  for (start in seq(1L, ncol(m), by = chunk)) {
    block <- m[, start:min(ncol(m), start + chunk - 1L), drop = FALSE]
    n <- n + sum(block != round(block))
  }
  n
}

# worker_utils.choose_counts_matrix's rule for a counts CSV, by hand. spVC fits a quasi-Poisson GAM
# per gene, i.e. models the values as counts: a negative value is not a count (mgcv's quasipoisson
# initialize stops on it, for every gene) and is refused here, before anything runs. Non-integer
# values are fitted as before -- quasi-Poisson accepts them -- and counted, so the caller is told the
# matrix looks normalised. Returns the number of non-integer values.
check_counts_values <- function(m, source_path) {
  if (length(m) == 0) return(0)
  min_value <- min(m)
  if (min_value < 0) {
    stop("The counts in ", source_path, " hold ", sum(m < 0), " negative value(s) (minimum ",
         signif(min_value, 4), ") over the ", ncol(m), " analysed spots, so they are scaled or centred ",
         "expression, not counts. spVC fits a quasi-Poisson model per gene, which cannot take a negative ",
         "value; nothing was fitted. Export the raw counts instead: a CELLxGENE-style h5ad keeps them in ",
         "adata.raw when its X is scaled.")
  }
  count_non_integer(m)
}

# Write to <path>.partial and move it into place, so a run killed mid-write never leaves a
# truncated table under the name a reader trusts. Copied from tools/spotsweeper_worker.R.
write_atomically <- function(path, writer) {
  tmp <- paste0(path, ".partial")
  done <- FALSE
  on.exit(if (!done && file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) stop("Could not move ", tmp, " into place at ", path)
  done <- TRUE
  invisible(path)
}

# test.spVC's own row and column filters, passed explicitly below. The worker applies neither
# itself, so they are named once here and counted before the call: test.spVC reports what they
# removed only with cat() on stdout, which base_mcp does not return.
SPVC_FILTER_SPOT_COUNTS <- 5
SPVC_FILTER_MIN_NONZERO <- 5
# spVC:::varying.test returns max(2e-17, p): every stronger signal comes back as exactly this
# value, so genes at the floor are tied and their order is not something spVC measured. The more
# spots a slide has, the more genes reach it.
SPVC_PVALUE_FLOOR <- 2e-17

run_spvc <- function(opts) {
  if (is.null(opts$counts_csv) || is.null(opts$coords_csv) || is.null(opts$output_dir)) {
    stop("spVC requires --counts-csv, --coords-csv, and --output-dir")
  }
  opts$max_spots <- as_spot_cap(opts$max_spots)
  opts$n_top <- as_whole_number(opts$n_top, "n_top", 0L,
                                "how many of the ranked genes go to predicted_genes.json")
  opts$max_genes <- as_whole_number(opts$max_genes, "max_genes", 1L,
                                    "how many of the highest-variance genes spVC is given")
  opts$pval_cutoff <- as_pval_cutoff(opts$pval_cutoff)

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  log_msg("Reading counts from: ", opts$counts_csv)
  counts_header <- read.csv(opts$counts_csv, nrows = 1, check.names = FALSE)
  counts_df <- read.csv(opts$counts_csv, row.names = 1, check.names = FALSE,
                        colClasses = c("character", rep("numeric", ncol(counts_header) - 1)))
  counts_mat <- as.matrix(counts_df)
  # The data.frame is a second full copy of the matrix and nothing reads it again. With no spot
  # cap the whole slide stays in memory, so it goes now rather than at the end of the run.
  rm(counts_df)
  # anyNA/range allocate nothing the size of the matrix; the count is taken only on failure.
  if (length(counts_mat) > 0 &&
      (anyNA(counts_mat) || !all(is.finite(range(counts_mat, na.rm = TRUE))))) {
    n_missing_counts <- sum(!is.finite(counts_mat))
    stop(n_missing_counts, " of the ", length(counts_mat), " values in ", opts$counts_csv,
         " are empty or not finite. spVC needs a complete counts matrix; no output was written.")
  }

  log_msg("Reading coordinates from: ", opts$coords_csv)
  coords_read <- read_coords_csv(opts$coords_csv)
  coords_raw <- coords_read$frame
  rownames_col <- coords_raw[, 1]
  coord_cols <- resolve_coord_cols(colnames(coords_raw)[-1], opts$coords_csv)
  # The resolver matches names case-blind and takes the first match, so two columns that share a
  # name -- or one column picked for both axes -- cannot be told apart; refuse rather than fit a
  # degenerate slide.
  dup_lc <- unique(tolower(colnames(coords_raw))[duplicated(tolower(colnames(coords_raw)))])
  if (anyDuplicated(coord_cols) > 0L || any(tolower(coord_cols) %in% dup_lc)) {
    stop("The coordinate columns chosen in ", opts$coords_csv, " (",
         paste(coord_cols, collapse = ", "), ") are not two distinct columns: the file names ",
         "more than one column ", paste(intersect(tolower(coord_cols), dup_lc), collapse = ", "),
         " (columns: ", paste(colnames(coords_raw), collapse = ", "), "). Give the file a header ",
         "row with one unique name per column.")
  }
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_raw), collapse = ", "), ")")
  coords_df <- data.frame(
    x = suppressWarnings(as.numeric(coords_raw[[coord_cols[1]]])),
    y = suppressWarnings(as.numeric(coords_raw[[coord_cols[2]]])),
    row.names = rownames_col
  )
  # Space Ranger's tissue flag, when the file carries one (tissue_positions.csv names it; the
  # headerless tissue_positions_list.csv is read under that name above).
  tissue_col <- colnames(coords_raw)[-1][tolower(colnames(coords_raw)[-1]) == "in_tissue"]
  in_tissue_flag <- NULL
  if (length(tissue_col) > 0) {
    in_tissue_flag <- coords_raw[[tissue_col[1]]]
    names(in_tissue_flag) <- rownames_col
  }

  # Ensure matching spot IDs (counts may be genes x spots or spots x genes)
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  if (length(common_spots) == 0) {
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot/cell IDs", "counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}

  # A spot the counts carry but the coordinates do not cannot be placed, so it is left out --
  # and n_spots_supplied below is counted after that, so the loss is stated here.
  n_spots_without_coords <- ncol(counts_mat) - length(common_spots)
  n_spots_supplied <- length(common_spots)

  # Background next: a spot whose in_tissue flag is 0 lies outside the tissue (a
  # raw_feature_bc_matrix, or a CELLxGENE export, carries every array spot) and is left out, by
  # worker_utils.keep_in_tissue's rule. No flag, or a flag that is 1 everywhere, keeps every spot.
  n_spots_off_tissue_dropped <- 0L
  if (!is.null(in_tissue_flag)) {
    keep_tissue <- in_tissue_keep(in_tissue_flag[common_spots])
    n_spots_off_tissue_dropped <- sum(!keep_tissue)
    if (n_spots_off_tissue_dropped == n_spots_supplied) {
      stop("The in_tissue column of ", opts$coords_csv, " marks none of the ", n_spots_supplied,
           " spots as in tissue (values seen: ",
           paste(utils::head(sort(unique(as.character(in_tissue_flag[common_spots]))), 8), collapse = ", "),
           "); fix the column so in-tissue spots are 1, or remove it if every spot is tissue.")
    }
    common_spots <- common_spots[keep_tissue]
  }

  counts_mat <- counts_mat[, common_spots, drop = FALSE]
  # Counts or refuse, over the spots analysed (after the background is left out).
  n_values_non_integer <- check_counts_values(counts_mat, opts$counts_csv)
  coords_df <- coords_df[common_spots, 1:2, drop = FALSE]
  colnames(coords_df) <- c("x", "y")
  bad_xy <- !is.finite(coords_df$x) | !is.finite(coords_df$y)
  if (any(bad_xy)) {
    stop(sum(bad_xy), " of the ", nrow(coords_df), " spots have a missing or non-numeric coordinate in ",
         "columns ", paste(coord_cols, collapse = ", "), " of ", opts$coords_csv, " (first: ",
         rownames(coords_df)[which(bad_xy)[1]], "). spVC needs a position for every spot.")
  }

  # Both caps below shrink the problem before spVC sees it, and both used to announce
  # themselves only through log_msg -- i.e. on stderr, which base_mcp discards when the
  # worker exits 0. Collect the same facts here so they ride out on the payload.
  notes <- character(0)
  n_spots_input <- ncol(counts_mat)
  if (n_spots_without_coords > 0) {
    notes <- c(notes, paste0(
      n_spots_without_coords, " of the ", n_spots_supplied + n_spots_without_coords, " spots in ",
      "counts_csv have no row in coords_csv and were left out; n_spots_supplied counts the ",
      n_spots_supplied, " that have both."
    ))
  }
  if (n_values_non_integer > 0) {
    notes <- c(notes, paste0(
      n_values_non_integer, " value(s) in the counts are not whole numbers (maximum ",
      signif(max(counts_mat), 4), "), so the matrix looks normalised rather than raw counts. spVC fits ",
      "a quasi-Poisson model of counts and ran on these values as supplied; export the raw counts ",
      "(a CELLxGENE-style h5ad keeps them in adata.raw) for the model spVC describes."
    ))
  }
  if (n_spots_off_tissue_dropped > 0) {
    # The sentence worker_utils.record_in_tissue writes for the Python workers.
    notes <- c(notes, paste0(
      n_spots_off_tissue_dropped, " of ", n_spots_supplied, " spots have in_tissue == 0 in coords_csv ",
      "(background outside the tissue) and were left out; ", n_spots_input, " in-tissue spots were analysed."
    ))
  }

  # Every spot is analysed unless the caller set max_spots. Only an explicit cap draws a
  # random subset, and the payload says how much of the slide it left out.
  if (!is.na(opts$max_spots)) {
    if (ncol(counts_mat) > opts$max_spots) {
      set.seed(opts$seed)
      keep_idx <- sample(ncol(counts_mat), opts$max_spots)
      counts_mat <- counts_mat[, keep_idx, drop = FALSE]
      coords_df <- coords_df[keep_idx, , drop = FALSE]
      log_msg("Subsampled from ", n_spots_input, " to ", opts$max_spots, " spots")
      notes <- c(notes, paste0(
        "Analysed a random ", opts$max_spots, " of ", n_spots_input, " spots (",
        round(100 * opts$max_spots / n_spots_input, 1), "% of the slide) because max_spots was ",
        "set to ", opts$max_spots, "; the other ", n_spots_input - opts$max_spots, " spots were ",
        "not tested. Omit max_spots to analyse every spot."
      ))
    }
  }

  log_msg("N genes = ", nrow(counts_mat), ", N spots = ", ncol(counts_mat))

  # Prepare coordinate matrix (spots x 2)
  coords_mat <- as.matrix(coords_df)

  # test.spVC expects Y as (gene x spot). Manual reference at
  # /workspace/hands_by_myself/runners/spvc_svg_detection.R passes
  # counts (gene x spot) directly and recovers gene names from names(rc).
  # Earlier orientation t(counts_mat) made names(rc) == spot barcodes,
  # so all "predicted SVGs" were spot IDs and F1 was 0.
  Y <- counts_mat  # genes x spots — rownames(Y) = gene names

  # Filter zero-variance genes (variance across spots → per-gene → rowVars). In blocks of rows:
  # apply() over the whole matrix first transposes a full copy of it, which on a whole slide
  # doubles the largest allocation the worker makes. The values are the same var() per row.
  gene_var <- numeric(nrow(Y))
  names(gene_var) <- rownames(Y)
  for (block in seq_len(ceiling(nrow(Y) / 2000))) {
    rows <- ((block - 1L) * 2000L + 1L):min(nrow(Y), block * 2000L)
    gene_var[rows] <- apply(Y[rows, , drop = FALSE], 1, var)
  }
  keep_genes <- gene_var > 0
  Y <- Y[keep_genes, , drop = FALSE]
  log_msg("After filtering zero-variance: ", nrow(Y), " genes remain")

  # spVC can crash with large gene counts due to BPST basis matrix size.
  # Select top genes by variance to keep computation tractable.
  # Manual reference uses max_genes=1000 (HVG-1000 by variance).
  max_genes <- opts$max_genes
  if (nrow(Y) > max_genes) {
    n_genes_before_cap <- nrow(Y)
    top_idx <- order(gene_var[keep_genes], decreasing = TRUE)[1:max_genes]
    Y <- Y[top_idx, , drop = FALSE]
    log_msg("Reduced to top ", max_genes, " genes by variance for spVC")
    notes <- c(notes, paste0(
      "Tested the ", max_genes, " highest-variance genes of ", n_genes_before_cap,
      ": the max_genes cap was reached. The other ", n_genes_before_cap - max_genes,
      " were never given to spVC, so they are absent from every output file rather than ",
      "ranked last in it."
    ))
  }

  # test.spVC keeps only spots whose counts over the submitted genes sum to at least
  # SPVC_FILTER_SPOT_COUNTS (every kept spot lies inside the bounding-rectangle mesh built below,
  # so this is the whole spot filter), then tests only genes non-zero in more than
  # SPVC_FILTER_MIN_NONZERO of those spots. Counted here with the same rules, because the
  # package announces both only through cat() on stdout.
  spots_fitted <- colSums(Y) >= SPVC_FILTER_SPOT_COUNTS
  n_spots_fitted <- sum(spots_fitted)
  n_genes_testable <- if (n_spots_fitted > 0) {
    sum(rowSums(Y[, spots_fitted, drop = FALSE] != 0) > SPVC_FILTER_MIN_NONZERO)
  } else {
    0L
  }
  log_msg("spVC's filters keep ", n_spots_fitted, " of ", ncol(Y), " spots and ",
          n_genes_testable, " of ", nrow(Y), " genes")
  if (n_spots_fitted < ncol(Y)) {
    notes <- c(notes, paste0(
      "spVC fitted ", n_spots_fitted, " of the ", ncol(Y), " spots it was given: ",
      ncol(Y) - n_spots_fitted, " had fewer than ", SPVC_FILTER_SPOT_COUNTS, " counts summed over ",
      "the ", nrow(Y), " genes submitted, and test.spVC drops such spots (filter.spot.counts). ",
      "No p-value draws on them."
    ))
  }
  if (n_genes_testable == 0) {
    stop(
      "spVC has nothing to test: no gene of the ", nrow(Y), " submitted is non-zero in more than ",
      SPVC_FILTER_MIN_NONZERO, " of the ", n_spots_fitted, " spots it keeps (spots need at least ",
      SPVC_FILTER_SPOT_COUNTS, " counts summed over those genes), so every gene would be skipped ",
      "by test.spVC's filter.min.nonzero. No output was written."
    )
  }
  if (n_genes_testable < nrow(Y)) {
    notes <- c(notes, paste0(
      "spVC skipped ", nrow(Y) - n_genes_testable, " of the ", nrow(Y), " genes submitted as ",
      "non-zero in ", SPVC_FILTER_MIN_NONZERO, " or fewer of the ", n_spots_fitted, " fitted spots ",
      "(test.spVC's filter.min.nonzero), so it tested ", n_genes_testable, ". The skipped genes ",
      "are absent from every output file rather than ranked last in it."
    ))
  }

  # Build a simple rectangular boundary mesh for spVC
  # spVC requires V (boundary vertices) and Tr (triangulation) as a COARSE mesh
  # covering the spatial domain. A simple bounding rectangle with 2 triangles works.
  log_msg("Building boundary mesh for spVC...")
  # Bound here, not in the handler below. `x <<- v` from a handler searches the enclosing
  # frames and, finding nothing, creates x in globalenv -- where the code after the
  # tryCatch would not see it.
  results <- NULL
  spvc_failure <- NULL
  n_genes_fitted <- NA_integer_
  # mgcv's own warnings from the per-gene fits ("Fitting terminated with step failure - check
  # results carefully", Davies-to-Liu p-value approximation, round-off) went only to stderr.
  fit_warnings <- character(0)
  tryCatch({
    library(geometry)

    # Create bounding rectangle with small margin
    x_range <- range(coords_mat[, 1])
    y_range <- range(coords_mat[, 2])
    margin_x <- diff(x_range) * 0.01
    margin_y <- diff(y_range) * 0.01

    V <- rbind(
      c(x_range[1] - margin_x, y_range[1] - margin_y),
      c(x_range[2] + margin_x, y_range[1] - margin_y),
      c(x_range[2] + margin_x, y_range[2] + margin_y),
      c(x_range[1] - margin_x, y_range[2] + margin_y)
    )
    Tr <- rbind(c(1L, 2L, 3L), c(1L, 3L, 4L))

    # Normalize coordinates to [0,1] range (spVC works better with normalized coords)
    S_norm <- coords_mat
    S_norm[, 1] <- (S_norm[, 1] - x_range[1]) / diff(x_range)
    S_norm[, 2] <- (S_norm[, 2] - y_range[1]) / diff(y_range)
    V_norm <- V
    V_norm[, 1] <- (V_norm[, 1] - x_range[1]) / diff(x_range)
    V_norm[, 2] <- (V_norm[, 2] - y_range[1]) / diff(y_range)

    log_msg("V: ", nrow(V_norm), " vertices, Tr: ", nrow(Tr), " triangles")

    # Run spVC batch test on all genes (Y is gene x spot)
    log_msg("Running spVC test on ", nrow(Y), " genes...")
    # test.spVC reports progress with cat() on stdout, which carries only this worker's JSON: the
    # call runs with its output sunk to stderr (undone on the way out, error or not).
    sink(stderr())
    res <- tryCatch(
      withCallingHandlers(
        test.spVC(
          Y = Y,
          S = S_norm,
          V = V_norm,
          Tr = Tr,
          para.cores = 1,
          reduced.only = TRUE,
          filter.min.nonzero = SPVC_FILTER_MIN_NONZERO,
          filter.spot.counts = SPVC_FILTER_SPOT_COUNTS
        ),
        warning = function(w) {
          fit_warnings <<- c(fit_warnings, conditionMessage(w))
        }
      ),
      finally = sink()
    )

    # res$results.constant is a per-gene named list. Each element r has
    # r$p.value, a named vector with entries like "(Intercept)", "gamma_0".
    # The spatial-effect p-value is r$p.value["gamma_0"] (per the package
    # convention; same extraction used by the manual SpVC SVG runner).
    rc <- res$results.constant
    n_genes_fitted <- length(rc)
    gene_names_rc <- names(rc)
    if (is.null(gene_names_rc) || length(gene_names_rc) == 0) {
      gene_names_rc <- as.character(seq_along(rc))
    }
    pv_per_gene <- sapply(rc, function(r) {
      pv <- r$p.value
      if (is.null(pv)) return(NA_real_)
      if ("gamma_0" %in% names(pv)) return(as.numeric(pv["gamma_0"]))
      if (length(pv) >= 2) return(as.numeric(pv[2]))
      if (length(pv) >= 1) return(as.numeric(pv[1]))
      NA_real_
    })
    stat_per_gene <- sapply(rc, function(r) {
      if (!is.null(r$Deviance) && length(r$Deviance) >= 1) return(as.numeric(r$Deviance[1]))
      if (!is.null(r$LRT) && length(r$LRT) >= 1)           return(as.numeric(r$LRT[1]))
      NA_real_
    })

    results <- data.frame(
      gene      = gene_names_rc,
      pvalue    = pv_per_gene,
      statistic = stat_per_gene,
      stringsAsFactors = FALSE
    )
    # Drop genes whose fit returned a non-finite p-value before ranking. Genes test.spVC's own
    # filter.min.nonzero skipped are not in rc at all; they were counted before the call.
    results <- results[is.finite(results$pvalue), , drop = FALSE]
    log_msg("spVC completed. Genes returned: ", length(rc),
            ", with finite p-values: ", nrow(results))
  }, error = function(e) {
    log_msg("spVC test failed: ", conditionMessage(e))
    spvc_failure <<- conditionMessage(e)
  })

  # A crash is a failed run, not a run that measured nothing. Substituting an empty
  # results table here used to carry on to a status-ok payload reporting "0 significant
  # SVGs out of 0 tested" -- and the evaluator scores the empty predicted_genes.json as
  # a real F1 of 0. The reason reached nobody: base_mcp drops a worker's stderr when it
  # exits 0, so log_msg above is the one place the error is recorded and the one place
  # nothing reads. Stop instead, before any file is written; main()'s handler publishes
  # this message at status error.
  if (!is.null(spvc_failure)) {
    # mgcv's quasipoisson stop on a negative value is about the input, not memory or geometry;
    # check_counts_values refuses such input up front, so this branch is a guard, not a path.
    advice <- if (grepl("negative values not allowed", spvc_failure, fixed = TRUE)) {
      paste0("The counts hold negative values, which a quasi-Poisson model cannot fit; export the raw ",
             "counts (a CELLxGENE-style h5ad keeps them in adata.raw).")
    } else {
      paste0("spVC fits a bivariate spline over the whole spot set; the usual causes are too many ",
             "genes for the available memory (lower --max-genes, currently ", opts$max_genes,
             ") or spot coordinates that collapse to a line or a point.")
    }
    stop(
      "spVC's spatial test did not run, so no gene was tested: ", spvc_failure,
      ". No output was written. ", advice
    )
  }

  if (length(fit_warnings) > 0) {
    kinds <- sort(table(fit_warnings), decreasing = TRUE)
    shown <- utils::head(kinds, 8)
    notes <- c(notes, paste0(
      "test.spVC raised ", length(fit_warnings), " warning(s) while fitting the ", n_genes_fitted,
      " genes, most frequent first: ",
      paste0(substr(names(shown), 1, 160), " (x", as.integer(shown), ")", collapse = "; "),
      if (length(kinds) > 8) paste0("; and ", length(kinds) - 8, " other kind(s)") else "",
      ". spVC does not say which genes they came from."
    ))
  }

  n_genes_nonfinite <- n_genes_fitted - nrow(results)
  if (n_genes_nonfinite > 0 && nrow(results) > 0) {
    notes <- c(notes, paste0(
      n_genes_nonfinite, " of the ", n_genes_fitted, " genes spVC fitted returned a non-finite ",
      "p-value and were left out of every output file."
    ))
  }

  if (nrow(results) == 0) {
    notes <- c(notes, paste0(
      "spVC ran but returned no gene with a finite p-value, so every output file is empty. ",
      "Nothing was ranked -- this is not a measurement that the ", nrow(counts_mat),
      " input genes are non-spatial."
    ))
  }

  # Adjust p-values, then rank. Ties -- above all the genes clamped to SPVC_PVALUE_FLOOR -- are
  # broken by count variance, highest first: the same order the max_genes cap selects by, now
  # stated rather than inherited from whichever order the rows happened to arrive in.
  n_genes_at_pvalue_floor <- 0L
  if (nrow(results) > 0) {
    results$adjusted_pvalue <- p.adjust(results$pvalue, method = "BH")
    tie_var <- unname(gene_var[results$gene])
    results <- results[order(results$pvalue, -tie_var), ]
    n_genes_at_pvalue_floor <- sum(results$pvalue <= SPVC_PVALUE_FLOOR)
  }
  if (n_genes_at_pvalue_floor > 1) {
    notes <- c(notes, paste0(
      n_genes_at_pvalue_floor, " of the ", nrow(results), " genes tested reached spVC's p-value ",
      "floor of ", SPVC_PVALUE_FLOOR, " (the package clamps smaller p-values to it), so spVC does ",
      "not rank them against each other; they are ordered by count variance, highest first.",
      if (n_genes_at_pvalue_floor > opts$n_top) paste0(
        " That is more than n_top = ", opts$n_top, ", so which of them are in the top ",
        opts$n_top, " was decided by that variance order, not by spVC."
      ) else ""
    ))
  }

  # Save full results. Every file goes through <name>.partial and a rename (write_atomically).
  results_path <- file.path(opts$output_dir, "spvc_results.csv")
  write_atomically(results_path, function(p) write.csv(results, p, row.names = FALSE, quote = TRUE))

  # Significant SVGs
  sig_genes <- results[results$adjusted_pvalue < opts$pval_cutoff, ]
  sig_path <- file.path(opts$output_dir, "spvc_significant_svgs.csv")
  write_atomically(sig_path, function(p) write.csv(sig_genes, p, row.names = FALSE, quote = TRUE))

  # Top N (n_top >= 0 was checked above: head() reads a negative n as "all but the last n")
  top_genes <- head(results, opts$n_top)
  top_path <- file.path(opts$output_dir, "spvc_top_svgs.csv")
  write_atomically(top_path, function(p) write.csv(top_genes, p, row.names = FALSE, quote = TRUE))

  # predicted_genes.json — standardised SVG-prediction artifact for benchmark
  # evaluator. Matches manual /workspace/hands_by_myself/runners/spvc_svg_detection.R
  # (lines 165-175: rank by spatial p-value ascending, keep top 10% — for HVG-1000
  # that is 100 genes, exactly n_top). Without this file, the evaluator finds no
  # predictions and scores F1=0.
  pg_genes <- as.character(top_genes$gene)
  pg_path <- file.path(opts$output_dir, "predicted_genes.json")
  write_atomically(pg_path, function(p) {
    writeLines(jsonlite::toJSON(list(predicted_genes = pg_genes), auto_unbox = FALSE, digits = NA), p)
  })
  log_msg("wrote ", pg_path, " with ", length(pg_genes), " genes")

  payload <- list(
    status       = "ok",
    tool         = "spvc",
    task         = "svg_detection",
    data         = list(
      # Both sides of each cap. n_spots/n_genes_input alone were measured on opposite
      # sides of them -- n_spots after the subsample, n_genes_input before the filter --
      # so read as a pair they described a matrix that was never analysed.
      n_spots           = ncol(counts_mat),
      n_spots_input     = n_spots_input,
      n_spots_without_coords = n_spots_without_coords,
      # Spots with both counts and coordinates, and how many of them in_tissue == 0 left out;
      # n_spots_input is what remains.
      n_spots_supplied  = n_spots_supplied,
      n_spots_off_tissue_dropped = n_spots_off_tissue_dropped,
      # What test.spVC itself kept of n_spots / n_genes_submitted (its spot and gene filters).
      n_spots_fitted    = n_spots_fitted,
      n_genes_input     = nrow(counts_mat),
      n_genes_submitted = nrow(Y),
      n_genes_fitted    = n_genes_fitted,
      n_genes_nonfinite = n_genes_nonfinite,
      n_genes_at_pvalue_floor = n_genes_at_pvalue_floor,
      n_fit_warnings    = length(fit_warnings),
      # Values of the analysed counts that are not whole numbers (0 for raw counts).
      n_values_non_integer = n_values_non_integer
    ),
    output_files = list(
      results_csv       = results_path,
      significant_csv   = sig_path,
      top_svgs_csv      = top_path
    ),
    params       = list(
      method            = paste0("spVC test.spVC, reduced model (constant intercept + spatially varying ",
                                 "intercept, no covariates; gamma_0 p-value), bounding-rectangle BPST ",
                                 "mesh of 2 triangles"),
      used_fallback     = FALSE,
      pval_cutoff       = opts$pval_cutoff,
      pval_adjustment   = "BH",
      n_top             = opts$n_top,
      # null when no cap was set: every spot was analysed.
      max_spots         = if (is.na(opts$max_spots)) NA else opts$max_spots,
      max_genes         = opts$max_genes,
      filter_spot_counts = SPVC_FILTER_SPOT_COUNTS,
      filter_min_nonzero = SPVC_FILTER_MIN_NONZERO,
      tie_break         = "count variance, highest first",
      coordinate_columns = I(coord_cols),
      coordinates_header = coords_read$header,
      seed              = opts$seed
    ),
    summary      = list(
      n_genes_tested    = nrow(results),
      n_significant     = nrow(sig_genes),
      top_genes         = head(results$gene, min(20, nrow(results)))
    ),
    analysis     = paste0(
      "spVC identified ", nrow(sig_genes), " significant spatially variable genes ",
      "out of ", nrow(results), " tested (BH-adjusted p < ", opts$pval_cutoff, "), fitting ",
      n_spots_fitted, " of the ", n_spots_input, " spots in the input.",
      if (n_spots_off_tissue_dropped > 0) paste0(
        " Those are the in-tissue spots: ", n_spots_off_tissue_dropped, " of the ", n_spots_supplied,
        " spots supplied have in_tissue == 0 (background outside the tissue) and were left out first."
      ) else "",
      if (n_values_non_integer > 0) paste0(
        " NOTE: the input is not integer counts (", n_values_non_integer, " non-integer values), so ",
        "spVC's quasi-Poisson count model was fitted to normalised values."
      ) else ""
    )
  )
  # The shape worker_utils.record_in_tissue gives the Python workers, only when a spot was dropped.
  if (n_spots_off_tissue_dropped > 0) {
    payload$params$in_tissue_filter <- list(
      n_spots_supplied = n_spots_supplied,
      n_spots_off_tissue_dropped = n_spots_off_tissue_dropped,
      n_spots_used = n_spots_input
    )
  }
  # I() keeps a one-element vector an array in the JSON, matching how the Python
  # workers' WorkerOutput.add_warnings serialises the same key.
  if (length(notes) > 0) payload$warnings <- I(notes)
  payload
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  set.seed(opts$seed)

  res <- tryCatch(with_r_traceback({
    run_spvc(opts)
  }), error = function(e) {
    # Pop any stdout redirect still open (run_spvc sinks to stderr around chatty calls), so the
    # error payload below reaches stdout, where run_worker_cli reads it.
    while (sink.number() > 0) sink()
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "spvc",
      task      = "svg_detection",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
