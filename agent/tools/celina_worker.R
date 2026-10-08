#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(CELINA)
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
  message(sprintf("[celina-worker] %s", msg))
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

# Space Ranger writes the spot table twice, and only one of the two carries a header line:
# outs/spatial/tissue_positions.csv (Space Ranger 2+) has one, tissue_positions_list.csv (Space
# Ranger 1) has none -- barcode, tissue flag, array row, array col, pixel row, pixel col straight
# into the first line. read.csv defaults to header = TRUE, so on the older file the first spot's own
# values became the column names. Measured on the SpinalCord slide: the columns came back named
# 0, 0, 0, 613, 701, resolve_coord_cols matched none of them and fell through to the first two --
# "0" and "0", the tissue flag twice -- and Create_Celina_Object's scale() turned that constant pair
# into NaN for every spot, with the first spot gone from the table.
#
# Decide from the first line itself: if every field after the barcode parses as a number, it is data
# and not a header. Header fields are text, so a file that does have one reads exactly as it read
# before. On the six-column layout the recovered names are the ones SpotClean's read10xSlide()
# imposes on this same file, and imagerow/imagecol is the pair the resolver prefers; any other width
# keeps the spot and leaves the columns to the resolver's fall-through.
#
# Verbatim copy of tools/spotclean_worker.R's reader: each worker runs as its own Rscript in its
# own conda env, so there is no shared library on the path.
read_coords_csv <- function(path) {
  first <- read.csv(path, header = FALSE, nrows = 1, check.names = FALSE, stringsAsFactors = FALSE)
  headerless <- ncol(first) > 1 && all(vapply(first[-1], is.numeric, logical(1)))
  if (!headerless) {
    return(read.csv(path, row.names = 1, check.names = FALSE))
  }
  SPACE_RANGER_V1 <- c("barcode", "tissue", "row", "col", "imagerow", "imagecol")
  named <- if (ncol(first) == length(SPACE_RANGER_V1)) {
    SPACE_RANGER_V1
  } else {
    c("barcode", paste0("V", seq_len(ncol(first) - 1L)))
  }
  df <- read.csv(path, header = FALSE, col.names = named, row.names = 1, check.names = FALSE)
  log_msg("Coordinates file has no header line; read as Space Ranger's tissue_positions_list.csv (",
          nrow(df), " spots, columns ", paste(colnames(df), collapse = ", "), ")")
  df
}

# resolve_coord_cols is the same text in fourteen workers and a test keeps the copies in step, so
# the check that its answer is usable sits beside this one caller rather than inside it. Two answers
# are not usable. A name that is a bare number is a spot's value read as the header -- the first
# spot is gone and the "column" is whatever that spot held. A name the file carries more than once
# cannot pick one column: read.csv keeps the repeat, and data.frame subsetting hands back the first
# match for both, so on the headerless SpinalCord file the pair came back as "0" and "0" -- the
# tissue flag twice. Only the two chosen names are checked: a repeated or numeric name the resolver
# did not pick changes nothing, and refusing it would refuse a file that reads correctly.
check_coord_cols <- function(coord_cols, coord_names, source_path) {
  same <- length(coord_cols) == 2L && coord_cols[1] == coord_cols[2]
  numeric_names <- coord_cols[!is.na(suppressWarnings(as.numeric(coord_cols)))]
  repeated <- unique(coord_cols[coord_cols %in% coord_names[duplicated(coord_names)]])
  if (same || length(numeric_names) > 0 || length(repeated) > 0) {
    stop("The coordinate columns chosen from ", source_path, " are \"", coord_cols[1], "\" and \"",
         coord_cols[2], "\" (the file has: ", paste(coord_names, collapse = ", "), "). ",
         if (same) "They are the same column, so every spot would sit on the diagonal x == y. " else "",
         if (length(numeric_names) > 0) paste0(
           "A column named after a number (", paste(sprintf('"%s"', numeric_names), collapse = ", "),
           ") is a spot's line read as the header line, and that spot is lost. ") else "",
         if (length(repeated) > 0) paste0(
           "The name(s) ", paste(sprintf('"%s"', repeated), collapse = ", "),
           " appear more than once in the file, so which column is meant is ambiguous. ") else "",
         "Add a header line naming the barcode and the two coordinates, each exactly once ",
         "(imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, array_row/array_col, row/col ",
         "or x/y), or pass Space Ranger's tissue_positions_list.csv unchanged -- a file with no ",
         "header line at all is recognised.")
  }
  invisible(coord_cols)
}

# CELINA scales each axis on its own (Create_Celina_Object, scaling_method = "separate"), and
# scale() of a constant column is NaN for every spot, so a constant or non-numeric axis is refused
# here, by the name the file gave it: downstream every kernel would be NaN and the run would either
# die inside CELINA or report p-values computed on nothing.
check_coord_values <- function(coords_df, source_path) {
  for (col in colnames(coords_df)) {
    v <- suppressWarnings(as.numeric(as.character(coords_df[[col]])))
    if (anyNA(v) || !all(is.finite(v))) {
      bad <- sum(is.na(v) | !is.finite(v))
      stop("Coordinate column \"", col, "\" of ", source_path, " has ", bad,
           " non-numeric, missing or infinite values among the ", length(v),
           " spots in the analysis. Every spot needs two finite coordinates.")
    }
    if (length(v) > 1 && stats::sd(v) == 0) {
      stop("Coordinate column \"", col, "\" of ", source_path, " is constant: every one of the ",
           length(v), " spots has ", v[1], ". CELINA scales each axis, and a constant axis scales ",
           "to NaN, so the run cannot proceed. Check that the two columns read (",
           paste(colnames(coords_df), collapse = ", "), ") are the spot positions.")
    }
    coords_df[[col]] <- v
  }
  coords_df
}

# CARD writes card_proportions.csv with the cell types as columns and the spot identifiers as a
# trailing "spot" column (row.names = FALSE); the converter and pandas write them as an unnamed
# first column. read.csv(row.names = 1) took CARD's first cell type as the row names -- a
# "duplicate 'row.names' are not allowed" abort, or no spot overlap -- and kept "spot" as a
# character "cell type" that made the whole matrix character. SPOTlight's spotlight_proportions.csv
# has the same layout. Find the identifier column by name wherever it sits ("row.names" is what
# read.csv calls the first column of a file whose header is one field short), fall back to the first
# column, and refuse anything left that is not numeric.
PROP_ID_COLS <- c("", "row.names", "spot", "spots", "spot_id", "spotid", "barcode", "barcodes",
                  "cell", "cell_id", "cellid", "index")

read_proportions_csv <- function(path) {
  raw <- read.csv(path, row.names = NULL, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(raw) < 2) {
    stop("The proportions file ", path, " has ", ncol(raw), " column(s); it needs a spot ",
         "identifier column and at least one cell type column.")
  }
  lc <- tolower(trimws(colnames(raw)))
  # By the list's order, not the file's: CARD's trailing "spot" must win over a cell type that
  # happens to be called "cell" earlier in the row.
  named_at <- which(lc %in% PROP_ID_COLS)
  named_at <- named_at[order(match(lc[named_at], PROP_ID_COLS))]
  id_at <- if (length(named_at) > 0) named_at[1] else 1L
  id_name <- colnames(raw)[id_at]
  id_label <- if (nzchar(id_name)) sprintf('"%s"', id_name) else "1 (unnamed)"
  ids <- as.character(raw[[id_at]])
  dup <- unique(ids[duplicated(ids)])
  if (length(dup) > 0) {
    stop("The spot identifiers taken from column ", id_label, " of ", path, " are not unique (",
         length(dup), " repeated, e.g. ", paste(sprintf('"%s"', utils::head(dup, 3)), collapse = ", "),
         "). If the identifiers are in another column, name it spot or barcode.")
  }
  props <- raw[, -id_at, drop = FALSE]
  not_numeric <- colnames(props)[!vapply(props, is.numeric, logical(1))]
  if (length(not_numeric) > 0) {
    stop("These columns of ", path, " are not numeric: ",
         paste(sprintf('"%s"', not_numeric), collapse = ", "),
         ". Every column besides the spot identifier (column ", id_label,
         ") must hold one cell type's proportions.")
  }
  mat <- as.matrix(props)
  rownames(mat) <- ids
  list(mat = mat, id_column = id_name, id_column_named = length(named_at) > 0)
}

# The converter writes celltypes.csv with one label column and metadata.csv with every obs column.
# Taking column 1 of whichever file arrived made metadata.csv's first column -- Patient, Age -- the
# cell type labels, and only the later intersection with the proportion columns (usually empty,
# sometimes partial) stood between that and a result. One column is unambiguous; more than one
# needs the caller to name it.
read_sc_labels_csv <- function(path, column) {
  df <- read.csv(path, row.names = 1, check.names = FALSE, stringsAsFactors = FALSE)
  if (ncol(df) < 1) {
    stop(paste("The CSV given to --sc-celltype-labels-csv has no label column. It must hold cell",
               "barcodes in the first (index) column and one cell type label column."))
  }
  if (!is.null(column) && nzchar(column)) {
    if (!(column %in% colnames(df))) {
      stop("--sc-celltype-column \"", column, "\" is not a column of ", path,
           "; its columns are: ", paste(colnames(df), collapse = ", "), ".")
    }
    chosen <- column
  } else if (ncol(df) == 1L) {
    chosen <- colnames(df)[1]
  } else {
    stop("The CSV given to --sc-celltype-labels-csv (", path, ") has ", ncol(df),
         " columns besides the cell barcodes (",
         paste(utils::head(colnames(df), 12), collapse = ", "),
         if (ncol(df) > 12) ", ..." else "",
         ") and none was named. Pass sc_celltype_column (--sc-celltype-column) with the one that ",
         "holds the cell type labels; the first column is not assumed, because in a metadata ",
         "table it is usually not the cell type.")
  }
  labels <- as.character(df[[chosen]])
  names(labels) <- rownames(df)
  list(labels = labels, column = chosen)
}

# A missing label is not a class. CELINA's get_scRNA_info averages each tested cell type over the
# reference cells with `labels == cell_type`; an NA label makes that index NA for EVERY cell type,
# which puts a column of NAs into every mean, and a label spelled "nan"/"None" that the proportions
# also carry as a column would be tested as a cell type. A blank label that names no tested cell
# type is inert -- no `==` ever selects it -- and a run with some is exactly as correct as before, so
# it is kept and counted rather than refused. drop_unlabeled = TRUE leaves out every missing label
# (NA, "", "NA", "nan", "None") and says how many; the missing-label vocabulary is
# tools/worker_utils.py's drop_unlabeled().
LABEL_MISSING <- c("", "na", "nan", "none")

resolve_unlabeled <- function(labels, cell_type_names, drop_unlabeled, column, path) {
  missing <- is.na(labels) | tolower(trimws(labels)) %in% LABEL_MISSING
  n_missing <- sum(missing)
  if (n_missing == 0L) {
    return(list(keep = rep(TRUE, length(labels)), n_dropped = 0L, n_unlabeled = 0L))
  }
  if (isTRUE(drop_unlabeled)) {
    log_msg("Dropping ", n_missing, " of ", length(labels),
            " reference cells with no label (drop_unlabeled=TRUE)")
    return(list(keep = !missing, n_dropped = n_missing, n_unlabeled = 0L))
  }
  n_na <- sum(is.na(labels))
  as_class <- intersect(unique(labels[missing & !is.na(labels)]), cell_type_names)
  if (n_na > 0L || length(as_class) > 0L) {
    stop(n_missing, " of ", length(labels), " reference cells have no label in column \"", column,
         "\" of ", path, " (NA/empty/nan). ",
         if (n_na > 0L) paste0(n_na, " are NA, and CELINA's `labels == cell_type` turns an NA into ",
                               "an NA column in every cell type's reference mean. ") else "",
         if (length(as_class) > 0L) paste0(
           "The proportions table has a column named ", paste(sprintf('"%s"', as_class), collapse = ", "),
           ", so the missing label would be tested as a cell type. ") else "",
         "Pass drop_unlabeled=TRUE to leave these cells out, or label them first; a missing label ",
         "is not a cell type.")
  }
  log_msg(n_missing, " of ", length(labels), " reference cells have a blank label; no tested cell ",
          "type selects them, so they are inert (drop_unlabeled=TRUE would leave them out)")
  list(keep = rep(TRUE, length(labels)), n_dropped = 0L, n_unlabeled = n_missing)
}

# A tissue/background flag holds 0/1 (or TRUE/FALSE, in any case, or those as text). Returns the
# column as integer 0/1 with NA where a spot has no value, or NULL when the column is not a flag at
# all -- a label such as 'thymus', a code such as 2, or nothing but blanks.
#
# Verbatim copy of tools/spotclean_worker.R's as_tissue_flag: each worker runs as its own Rscript in
# its own conda env, so there is no shared library on the path.
as_tissue_flag <- function(values) {
  if (is.factor(values)) values <- as.character(values)
  if (is.logical(values)) {
    return(if (all(is.na(values))) NULL else as.integer(values))
  }
  if (is.numeric(values)) {
    present <- values[!is.na(values)]
    if (length(present) == 0L || !all(present %in% c(0, 1))) return(NULL)
    return(as.integer(values))
  }
  text <- tolower(trimws(as.character(values)))
  text[text %in% c("", "na", "nan")] <- NA_character_
  present <- text[!is.na(text)]
  if (length(present) == 0L || !all(present %in% c("0", "1", "true", "false"))) return(NULL)
  as.integer(text %in% c("1", "true")) + ifelse(is.na(text), NA_integer_, 0L)
}

# Space Ranger's tissue-positions files list every array spot and say which are on the tissue:
# in_tissue in tissue_positions.csv, the second field of the headerless tissue_positions_list.csv
# (read_coords_csv names it "tissue"). A spot flagged 0 is background glass. When the counts carry
# such spots too (a raw matrix, or a CELLxGENE export, which keeps all 4,992 array spots), CELINA
# would test them as tissue, and their ambient counts turn the tissue edge into a "pattern". The
# rule is tools/worker_utils.py's keep_in_tissue(): only a flag of 1 is in the tissue. A column of
# either name that is not a 0/1 flag (CELLxGENE's text 'tissue' label) is not used, and is named in
# the log. Returns list(flag = integer vector aligned to the rows, column = the name used or NA).
TISSUE_FLAG_COLS <- c("in_tissue", "tissue")

read_tissue_flag <- function(coords_df) {
  lc <- tolower(colnames(coords_df))
  for (hit in TISSUE_FLAG_COLS[TISSUE_FLAG_COLS %in% lc]) {
    at <- match(hit, lc)
    parsed <- as_tissue_flag(coords_df[[at]])
    if (is.null(parsed)) {
      seen <- utils::head(unique(as.character(coords_df[[at]])), 3)
      log_msg("Column '", colnames(coords_df)[at], "' of the coordinates file is not a 0/1 tissue flag ",
              "(values ", paste0("'", seen, "'", collapse = ", "), "); no spot is left out by it")
      next
    }
    return(list(flag = parsed, column = colnames(coords_df)[at]))
  }
  list(flag = NULL, column = NA_character_)
}

# Verbatim copy of tools/spark_worker.R's reduction_note (tools/worker_utils.py describe_reduction):
# one sentence naming both counts whenever part of the input was dropped before the method ran.
reduction_note <- function(noun, n_supplied, n_used, reason = "") {
  if (n_supplied <= 0 || n_used >= n_supplied) return("")
  pct <- 100 * n_used / n_supplied
  because <- if (nzchar(reason)) paste0(" by ", reason) else ""
  paste0(" NOTE: of the ", n_supplied, " ", noun, " supplied, ", n_used, " (", sprintf("%.1f", pct),
         "%) were analysed; ", n_supplied - n_used, " were dropped before the method ran", because,
         ". The results below describe the ", n_used, " analysed ", noun, ", not the full input.")
}

# Written beside the target and renamed into place, so a reader that arrives early sees the
# previous file or none, never a half-written one.
write_atomically <- function(path, writer) {
  partial <- paste0(path, ".partial")
  writer(partial)
  if (!file.rename(partial, path)) {
    stop("Could not move ", partial, " into place as ", path)
  }
  invisible(path)
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
  # Handle --help
  if (length(args) > 0 && args[[1]] %in% c("--help", "-h")) {
    cat("Usage: celina_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH    Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH    Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --cell-proportions-csv PATH  Cell type proportions CSV (spots x cell types) [required]\n")
    cat("  --sc-counts-csv PATH         Single-cell reference counts CSV (genes x cells) [required]\n")
    cat("  --sc-celltype-labels-csv PATH  Single-cell cell type labels CSV (cells x label column(s)) [required]\n")
    cat("  --sc-celltype-column NAME    Column of the labels CSV holding the cell types; required when\n")
    cat("                               the file has more than one column besides the barcodes\n")
    cat("  --drop-unlabeled true|false  Leave out reference cells with a missing label (NA, empty,\n")
    cat("                               nan, None). Default false: an NA label, or one the\n")
    cat("                               proportions name as a cell type, is refused; a blank label\n")
    cat("                               no tested cell type selects is kept and counted\n")
    cat("  --output-dir PATH            Output directory [required]\n")
    cat("  --num-cores INT              Cores for the interaction tests (default: 1)\n")
    cat("  --help                       Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv     = NULL,
    spatial_coords_csv     = NULL,
    cell_proportions_csv   = NULL,
    sc_counts_csv          = NULL,
    sc_celltype_labels_csv = NULL,
    sc_celltype_column     = "",
    drop_unlabeled         = FALSE,
    output_dir             = NULL,
    num_cores              = 1L
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
    } else if (key == "--cell-proportions-csv") {
      opts$cell_proportions_csv <- val
    } else if (key == "--sc-counts-csv") {
      opts$sc_counts_csv <- val
    } else if (key == "--sc-celltype-labels-csv") {
      opts$sc_celltype_labels_csv <- val
    } else if (key == "--sc-celltype-column") {
      opts$sc_celltype_column <- val
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- tolower(val) %in% c("true", "1", "yes")
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--num-cores") {
      opts$num_cores <- as.integer(val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_celina <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$cell_proportions_csv) || is.null(opts$sc_counts_csv) ||
      is.null(opts$sc_celltype_labels_csv) || is.null(opts$output_dir)) {
    stop(paste("CELINA requires --spatial-counts-csv, --spatial-coords-csv, --cell-proportions-csv,",
               "--sc-counts-csv, --sc-celltype-labels-csv, and --output-dir. The single-cell",
               "reference is not optional: CELINA derives the per-cell-type gene list from it."))
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv,
              opts$cell_proportions_csv, opts$sc_counts_csv,
              opts$sc_celltype_labels_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial expression ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  sp_counts_mat <- as.matrix(sp_counts_df)

  # --- Load spatial coordinates ---
  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  sp_coords_df <- read_coords_csv(opts$spatial_coords_csv)

  # --- Load cell type proportions ---
  log_msg("Reading cell type proportions from: ", opts$cell_proportions_csv)
  prop_read <- read_proportions_csv(opts$cell_proportions_csv)
  prop_mat <- prop_read$mat
  log_msg("Spot identifiers of the proportions table: column ",
          if (nzchar(prop_read$id_column)) prop_read$id_column else "1 (unnamed)",
          if (prop_read$id_column_named) "" else " (first column; none was named spot/barcode)")

  # --- Align spot IDs across all inputs ---
  # Expression may be genes x spots or spots x genes; detect orientation
  spots_in_coords <- rownames(sp_coords_df)
  spots_in_props  <- rownames(prop_mat)

  # Try colnames of counts first (genes x spots)
  common_spots <- Reduce(intersect, list(colnames(sp_counts_mat), spots_in_coords, spots_in_props))
  if (length(common_spots) == 0) {
    # Try rownames of counts (spots x genes)
    sp_counts_mat <- t(sp_counts_mat)
    common_spots <- Reduce(intersect, list(colnames(sp_counts_mat), spots_in_coords, spots_in_props))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(sp_counts_mat), "coordinates and proportions", Reduce(intersect, list(spots_in_coords, spots_in_props))))}

  # --- Which counts spots are analysed, and why the others are not ---
  # CELINA can only test a spot that has coordinates, a proportion row and tissue under it. The
  # intersection above used to be the whole story, and nothing said how many counts spots it cost:
  # RCTD (UMI_min) and CARD (its spot QC) leave low-count spots out of their proportions, so those
  # spots vanished without a word. Each counts spot is now accounted for, in this order: no
  # coordinate row; flagged off the tissue by the coordinates file; no proportion row.
  warnings <- character(0)
  counts_spots <- colnames(sp_counts_mat)
  n_spots_in_counts <- length(counts_spots)
  has_coords <- counts_spots %in% spots_in_coords
  tissue <- read_tissue_flag(sp_coords_df)
  off_tissue <- rep(FALSE, n_spots_in_counts)
  if (!is.null(tissue$flag)) {
    flag_of_spot <- tissue$flag[match(counts_spots, spots_in_coords)]
    off_tissue <- has_coords & !(flag_of_spot %in% 1L)
  }
  has_props <- counts_spots %in% spots_in_props
  n_spots_without_coords <- sum(!has_coords)
  n_spots_off_tissue <- sum(off_tissue)
  n_spots_without_props <- sum(has_coords & !off_tissue & !has_props)
  common_spots <- counts_spots[has_coords & !off_tissue & has_props]
  if (length(common_spots) == 0) {
    stop("Every one of the ", n_spots_in_counts, " spots in ", opts$spatial_counts_csv, " that has a row ",
         "in the coordinates file and the proportions table is marked off the tissue (column \"",
         tissue$column, "\" of ", opts$spatial_coords_csv, " is 0 for them; ", n_spots_off_tissue,
         " spots), so no tissue spot is left to test. Check the flag column, or pass a coordinates ",
         "file without it if every spot is tissue.")
  }
  if (n_spots_without_coords > 0) {
    warnings <- c(warnings, paste0(
      n_spots_without_coords, " of the ", n_spots_in_counts, " spots in the counts CSV have no row in ",
      "the coordinates file and were left out."))
  }
  if (n_spots_off_tissue > 0) {
    warnings <- c(warnings, paste0(
      n_spots_off_tissue, " of the ", n_spots_in_counts, " spots in the counts CSV have ", tissue$column,
      " == 0 in the coordinates file (background outside the tissue) and were left out; ",
      n_spots_in_counts - n_spots_off_tissue, " spots were not flagged off the tissue."))
  }
  if (n_spots_without_props > 0) {
    warnings <- c(warnings, paste0(
      n_spots_without_props, " of the ", n_spots_in_counts, " spots in the counts CSV have no row in ",
      "the proportions table and were left out (deconvolution tools such as RCTD and CARD leave ",
      "low-count spots out of their proportions)."))
  }
  dropped_why <- c(
    if (n_spots_without_coords > 0) paste0(n_spots_without_coords, " with no coordinate row"),
    if (n_spots_off_tissue > 0) paste0(n_spots_off_tissue, " flagged off the tissue (", tissue$column,
                                       " == 0)"),
    if (n_spots_without_props > 0) paste0(n_spots_without_props, " with no row in the proportions table")
  )
  spot_note <- reduction_note(
    "spots", n_spots_in_counts, length(common_spots),
    paste0("matching the counts CSV to the coordinates and the proportions (",
           paste(dropped_why, collapse = "; "), ")"))

  sp_counts_mat <- sp_counts_mat[, common_spots, drop = FALSE]
  coord_cols <- resolve_coord_cols(colnames(sp_coords_df), opts$spatial_coords_csv)
  check_coord_cols(coord_cols, colnames(sp_coords_df), opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(sp_coords_df), collapse = ", "), ")")
  sp_coords_df  <- sp_coords_df[common_spots, coord_cols, drop = FALSE]
  sp_coords_df  <- check_coord_values(sp_coords_df, opts$spatial_coords_csv)
  colnames(sp_coords_df) <- c("x", "y")
  prop_mat <- prop_mat[common_spots, , drop = FALSE]

  n_spots <- length(common_spots)
  n_genes <- nrow(sp_counts_mat)
  n_celltypes <- ncol(prop_mat)
  cell_type_names <- colnames(prop_mat)

  log_msg("Aligned data: ", n_spots, " of the ", n_spots_in_counts, " counts spots (",
          n_spots_without_coords, " without coordinates, ", n_spots_off_tissue, " off the tissue, ",
          n_spots_without_props, " without proportions), ", n_genes, " genes, ", n_celltypes, " cell types")

  # --- Load the single-cell reference ---
  # CELINA's preprocess_input() derives each cell type's marker gene list from a
  # single-cell reference; it has no default for any of the three arguments.
  log_msg("Reading single-cell reference counts from: ", opts$sc_counts_csv)
  sc_counts_mat <- as.matrix(read.csv(opts$sc_counts_csv, row.names = 1, check.names = FALSE))

  log_msg("Reading single-cell cell type labels from: ", opts$sc_celltype_labels_csv)
  sc_read <- read_sc_labels_csv(opts$sc_celltype_labels_csv, opts$sc_celltype_column)
  sc_labels <- sc_read$labels
  log_msg("Cell type label column: ", sc_read$column)

  # Reference cells may be listed as rows instead of columns.
  common_cells <- intersect(colnames(sc_counts_mat), names(sc_labels))
  if (length(common_cells) == 0 && length(intersect(rownames(sc_counts_mat), names(sc_labels))) > 0) {
    sc_counts_mat <- t(sc_counts_mat)
    common_cells <- intersect(colnames(sc_counts_mat), names(sc_labels))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "--sc-counts-csv", colnames(sc_counts_mat),
                         "--sc-celltype-labels-csv", names(sc_labels)))
  }
  # A reference cell with no row in the labels file has no cell type and cannot be used; say how
  # many, rather than report only the cells that were matched.
  n_reference_cells_in_counts <- ncol(sc_counts_mat)
  n_reference_cells_without_label_row <- n_reference_cells_in_counts - length(common_cells)
  if (n_reference_cells_without_label_row > 0) {
    warnings <- c(warnings, paste0(
      n_reference_cells_without_label_row, " of the ", n_reference_cells_in_counts, " cells in the ",
      "reference counts CSV have no row in the labels CSV and were left out of the reference."))
  }
  # Missing labels are judged on the cells that are actually used: a label row with no counts
  # column behind it never reaches CELINA.
  unlabeled <- resolve_unlabeled(unname(sc_labels[common_cells]), cell_type_names,
                                 opts$drop_unlabeled, sc_read$column, opts$sc_celltype_labels_csv)
  common_cells <- common_cells[unlabeled$keep]
  if (length(common_cells) == 0) {
    stop("No reference cell is left once the unlabelled ones are dropped (drop_unlabeled=TRUE).")
  }
  sc_counts_mat <- sc_counts_mat[, common_cells, drop = FALSE]
  sc_labels <- unname(sc_labels[common_cells])

  # Both matrices' orientations are already pinned -- the slide's by the spot IDs, the reference's
  # by the cell IDs -- so an empty gene intersection here is a naming mismatch, not a transposed
  # table. Deliberately not the shared id-mismatch helper: its transpose hint is the wrong advice.
  common_genes <- intersect(rownames(sp_counts_mat), rownames(sc_counts_mat))
  if (length(common_genes) == 0) {
    fmt_ids <- function(ids) {
      if (length(ids) == 0) return("<none>")
      paste0("[", paste(sprintf('"%s"', utils::head(ids, 3)), collapse = ", "),
             if (length(ids) > 3) ", ..." else "", "]")
    }
    stop(paste0(
      "No gene IDs are shared between --sc-counts-csv and --spatial-counts-csv. ",
      "--sc-counts-csv: ", nrow(sc_counts_mat), " genes ", fmt_ids(rownames(sc_counts_mat)), "; ",
      "--spatial-counts-csv: ", nrow(sp_counts_mat), " genes ", fmt_ids(rownames(sp_counts_mat)), ". ",
      "Both tables were already oriented by their spot and cell IDs, so this is a gene naming ",
      "mismatch rather than a transposed matrix -- typically Ensembl IDs on one side and gene ",
      "symbols on the other. Convert one side to the other's nomenclature and pass it again."
    ))
  }
  sc_counts_mat <- sc_counts_mat[common_genes, , drop = FALSE]
  log_msg("Single-cell reference: ", length(common_cells), " cells, ",
          length(common_genes), " genes shared with the slide")

  # Only cell types the reference can characterise can be tested.
  cell_types_to_test <- intersect(cell_type_names, unique(sc_labels))
  if (length(cell_types_to_test) == 0) {
    stop(paste0("None of the cell types in --cell-proportions-csv (",
                paste(utils::head(cell_type_names, 8), collapse = ", "),
                ") appear in --sc-celltype-labels-csv (",
                paste(utils::head(unique(sc_labels), 8), collapse = ", "),
                "). The two files must use the same cell type names."))
  }
  if (length(cell_types_to_test) < n_celltypes) {
    log_msg("Testing ", length(cell_types_to_test), " of ", n_celltypes,
            " cell types; the rest are absent from the reference labels")
  }

  # CELINA keeps a spot for marker derivation by rowSums(proportions) >= 0.8, and a missing
  # proportion makes that test NA, which indexes an NA location name and aborts inside the library
  # with "subscript out of bounds"; an infinite one turns the later per-spot renormalisation into
  # NaN. CELINA reads only the tested cell types' columns -- get_cell_type_gene_list takes
  # celltype_proportion[, cell_types_to_test] and Testing_interaction_all takes
  # celltype_mat[names(genes_list), ] -- so the check covers exactly those columns, on the spots
  # that are analysed. A non-finite value in a column with no reference cells is never read, and a
  # run carrying one finished before this check existed: it is counted and logged, not refused.
  bad_prop <- !is.finite(prop_mat[, cell_types_to_test, drop = FALSE])
  if (any(bad_prop)) {
    bad_cols <- cell_types_to_test[colSums(bad_prop) > 0]
    stop(sum(bad_prop), " proportion values in ", opts$cell_proportions_csv, " are missing or not ",
         "finite on the ", length(common_spots), " spots shared with the counts and coordinates, in ",
         "tested column(s) ", paste(sprintf('"%s"', utils::head(bad_cols, 8)), collapse = ", "),
         if (length(bad_cols) > 8) ", ..." else "", ". Every analysed spot needs a proportion for ",
         "every tested cell type.")
  }
  untested_cols <- setdiff(cell_type_names, cell_types_to_test)
  n_untested_not_finite <- if (length(untested_cols) > 0) {
    sum(!is.finite(prop_mat[, untested_cols, drop = FALSE]))
  } else 0L
  if (n_untested_not_finite > 0) {
    log_msg(n_untested_not_finite, " proportion values are missing or not finite in column(s) ",
            "with no reference cells, which are not tested and which CELINA does not read")
  }

  # --- Create CELINA object ---
  # Upstream wants cell types x spots, the transpose of how the proportions CSV is stored.
  log_msg("Creating CELINA object...")
  celina_obj <- Create_Celina_Object(
    celltype_mat        = t(prop_mat),
    gene_expression_mat = sp_counts_mat,
    location            = as.matrix(sp_coords_df)
  )

  # --- Preprocess ---
  log_msg("Preprocessing input (deriving marker genes for ",
          length(cell_types_to_test), " cell types)...")
  celina_obj <- preprocess_input(
    celina_obj,
    cell_types_to_test  = cell_types_to_test,
    scRNA_count         = sc_counts_mat,
    sc_cell_type_labels = sc_labels
  )

  # --- Calculate kernel ---
  log_msg("Calculating spatial kernel...")
  celina_obj <- Calculate_Kernel(celina_obj)

  # --- Test interactions ---
  log_msg("Testing cell type-gene interactions (num_cores = ", opts$num_cores, ")...")
  celina_obj <- Testing_interaction_all(celina_obj, num_cores = opts$num_cores)

  # --- Extract results ---
  log_msg("Extracting results...")

  # @result is a NAMED LIST, one data.frame per tested cell type, each genes x 12 with columns
  # Gaussian1..5, Matern1..5, Spline and CombinedPvals. Flatten it to one long table so the
  # cell type and gene each become a column the caller can read.
  per_celltype <- celina_obj@result
  if (!is.list(per_celltype) || length(per_celltype) == 0) {
    stop("CELINA returned no interaction results for any cell type.")
  }
  interaction_result <- do.call(rbind, lapply(names(per_celltype), function(ct) {
    tab <- per_celltype[[ct]]
    if (is.null(tab) || nrow(tab) == 0) return(NULL)
    data.frame(cell_type = ct, gene = rownames(tab), tab,
               row.names = NULL, check.names = FALSE, stringsAsFactors = FALSE)
  }))
  if (is.null(interaction_result) || nrow(interaction_result) == 0) {
    stop(paste("CELINA tested", length(per_celltype),
               "cell types and returned an empty table for every one of them."))
  }

  # Save interaction results
  results_path <- file.path(opts$output_dir, "celina_interactions.csv")
  write_atomically(results_path, function(p) write.csv(interaction_result, p, row.names = FALSE, quote = TRUE))

  # Save full CELINA object
  rds_path <- file.path(opts$output_dir, "celina_result.rds")
  write_atomically(rds_path, function(p) saveRDS(celina_obj, file = p))

  log_msg("Saved interaction results to: ", results_path)
  log_msg("Saved RDS to: ", rds_path)

  # --- Build summary ---
  # CELINA reports significance in CombinedPvals, which pools its Gaussian, Matern and Spline tests.
  n_tests <- nrow(interaction_result)
  # What was sent to testing is not always what came back: a cell type whose marker list comes out
  # empty returns no rows and is left out of the table. The analysis counts the ones in the table.
  cell_types_with_results <- unique(as.character(interaction_result$cell_type))
  pval_col <- if ("CombinedPvals" %in% colnames(interaction_result)) "CombinedPvals" else NA
  if (is.na(pval_col)) {
    stop(paste0("CELINA's result table has no CombinedPvals column (columns: ",
                paste(colnames(interaction_result), collapse = ", "),
                "). This build of CELINA reports significance differently."))
  }
  n_sig <- sum(interaction_result[[pval_col]] < 0.05, na.rm = TRUE)

  ordered_idx <- order(interaction_result[[pval_col]], na.last = NA)
  top_genes <- unique(as.character(interaction_result$gene[ordered_idx]))
  top_genes <- utils::head(top_genes[!is.na(top_genes)], 10)

  sig_pct <- if (n_tests > 0) round(n_sig / n_tests * 100, 1) else NA

  list(
    status       = "ok",
    tool         = "celina",
    task         = "interaction",
    data         = list(
      n_spots            = n_spots,
      # Every counts spot is accounted for: n_spots_in_counts = n_spots + the three below.
      n_spots_in_counts           = n_spots_in_counts,
      n_spots_without_coordinates = n_spots_without_coords,
      n_spots_off_tissue_dropped  = n_spots_off_tissue,
      n_spots_without_proportions = n_spots_without_props,
      n_genes            = n_genes,
      n_cell_types       = n_celltypes,
      # as.list: main() serialises with auto_unbox, which turns a one-element vector into a bare
      # string, so a run with a single cell type would publish a string where others publish a list.
      cell_types         = as.list(cell_type_names),
      cell_types_tested  = as.list(cell_types_to_test),
      cell_types_with_results = as.list(cell_types_with_results),
      n_reference_cells  = length(common_cells),
      n_reference_cells_in_counts         = n_reference_cells_in_counts,
      n_reference_cells_without_label_row = n_reference_cells_without_label_row,
      n_reference_cells_dropped   = unlabeled$n_dropped,
      n_reference_cells_unlabeled = unlabeled$n_unlabeled,
      n_reference_genes  = length(common_genes),
      # Missing or non-finite proportions in the untested columns (no reference cells): never read.
      n_untested_proportion_values_not_finite = as.integer(n_untested_not_finite),
      # How the inputs were read, so a caller can check it: the two coordinate columns CELINA was
      # given, and which column of the proportions table held the spot identifiers.
      coord_columns         = as.list(coord_cols),
      proportions_id_column = if (nzchar(prop_read$id_column)) prop_read$id_column else "<first column, unnamed>"
    ),
    output_files = list(
      interactions_csv = results_path,
      result_rds       = rds_path
    ),
    params       = c(list(
      num_cores          = opts$num_cores,
      pvalue_column      = pval_col,
      sc_celltype_column = sc_read$column,
      drop_unlabeled     = isTRUE(opts$drop_unlabeled),
      # The coordinates column read as the tissue flag ("" when the file has none).
      tissue_flag_column = if (is.na(tissue$column)) "" else tissue$column
    ),
    # The same record tools/worker_utils.py record_in_tissue() writes, present only when spots were cut.
    if (n_spots_off_tissue > 0) list(in_tissue_filter = list(
      n_spots_supplied           = n_spots_in_counts,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_used               = n_spots_in_counts - n_spots_off_tissue
    ))),
    summary      = list(
      n_tests          = n_tests,
      n_significant    = n_sig,
      significant_pct  = sig_pct,
      top_genes        = as.list(top_genes)
    ),
    warnings     = I(warnings),
    analysis     = paste0(
      "CELINA tested cell type-gene spatial interactions across ",
      n_spots, " spots for ", length(cell_types_with_results), " of ", n_celltypes, " cell types",
      if (length(cell_types_with_results) < length(cell_types_to_test)) paste0(
        " (", length(cell_types_to_test), " were sent to testing; ",
        paste(setdiff(cell_types_to_test, cell_types_with_results), collapse = ", "),
        " returned no results)") else "",
      if (length(cell_types_to_test) < n_celltypes) paste0(
        "; ", n_celltypes - length(cell_types_to_test),
        " of the proportion columns have no cells in the reference labels",
        if (n_untested_not_finite > 0) paste0(
          " (their ", n_untested_not_finite, " missing or non-finite values were not read)") else "") else "",
      ". ",
      if (!is.na(n_sig)) paste0(
        n_sig, " significant interactions found out of ", n_tests,
        " tests (", sig_pct, "% at CombinedPvals < 0.05, not corrected for multiple testing)."
      ) else paste0(n_tests, " interaction tests completed."),
      " Reference: ", length(common_cells), " cells labelled by column \"", sc_read$column, "\"",
      if (unlabeled$n_dropped > 0) paste0(", after dropping ", unlabeled$n_dropped,
                                          " with no label") else "",
      if (unlabeled$n_unlabeled > 0) paste0(", of which ", unlabeled$n_unlabeled,
                                            " have a blank label and belong to no tested cell type") else "",
      if (n_reference_cells_without_label_row > 0) paste0(
        "; ", n_reference_cells_without_label_row, " of the ", n_reference_cells_in_counts,
        " cells in the reference counts have no row in the labels CSV and were not used") else "",
      ".",
      spot_note
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  res <- tryCatch(with_r_traceback({
    sink(stderr())
    result <- run_celina(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "celina",
      task      = "interaction",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
