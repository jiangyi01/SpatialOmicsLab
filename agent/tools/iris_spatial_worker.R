#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(IRIS)
  library(Matrix)
})

# ---------------------------------------------------------------------------
# Monkey-patch IRIS_spatial to fix dgeMatrix bug.
#
# IRIS_spatial normalises counts with sweep(), which converts dgCMatrix to
# dgeMatrix (dense).  The C++ iteration functions (IRIS_ref_iter /
# IRIS_Marker_iter) only accept dgCMatrix, so the very first iteration call
# crashes with "dgeMatrix is not supported".  The original code re-coerces
# inside the for-loop (subsequent iterations) but misses the first call.
#
# The patch below wraps the original function and, just before it hands the
# IRIS object to the C++ layer, converts all count matrices back to
# dgCMatrix.  It does this by temporarily replacing the internal iteration
# helpers with wrappers that coerce their first argument.
# ---------------------------------------------------------------------------
# What IRIS's model actually ran on. IRIS_spatial cuts the QC'd counts down to its informative genes
# and drops every spot with no counts on them before the C++ iterations, and keeps neither number.
# The iteration wrappers below are the one place that final matrix passes through, so they note its
# size here and the payload reports it -- not the size of the file that was read, which is what
# data.n_spots / data.n_genes used to carry (4,992 spots reported for a Skin slide on which IRIS
# labelled about 685).
.iris_model_input <- new.env(parent = emptyenv())

note_model_input <- function(countList) {
  .iris_model_input$n_genes <- NROW(countList[[1]])
  .iris_model_input$n_spots <- NCOL(countList[[1]])
  invisible(NULL)
}

local({
  orig_spatial <- IRIS::IRIS_spatial
  orig_ref_iter <- IRIS::IRIS_ref_iter
  orig_marker_iter <- IRIS::IRIS_Marker_iter

  # Wrapper that coerces countList entries to CsparseMatrix before calling C++
  safe_ref_iter <- function(countList, ...) {
    countList <- lapply(countList, function(x) {
      if (!inherits(x, "dgCMatrix")) as(x, "CsparseMatrix") else x
    })
    note_model_input(countList)
    orig_ref_iter(countList, ...)
  }

  safe_marker_iter <- function(countList, ...) {
    countList <- lapply(countList, function(x) {
      if (!inherits(x, "dgCMatrix")) as(x, "CsparseMatrix") else x
    })
    note_model_input(countList)
    orig_marker_iter(countList, ...)
  }

  patched_spatial <- function(IRIS_object, ...) {
    # Temporarily replace the iteration helpers in the IRIS namespace
    ns <- asNamespace("IRIS")
    unlockBinding("IRIS_ref_iter", ns)
    unlockBinding("IRIS_Marker_iter", ns)
    on.exit({
      assign("IRIS_ref_iter", orig_ref_iter, envir = ns)
      assign("IRIS_Marker_iter", orig_marker_iter, envir = ns)
      lockBinding("IRIS_ref_iter", ns)
      lockBinding("IRIS_Marker_iter", ns)
    }, add = TRUE)
    assign("IRIS_ref_iter", safe_ref_iter, envir = ns)
    assign("IRIS_Marker_iter", safe_marker_iter, envir = ns)

    orig_spatial(IRIS_object, ...)
  }

  # Override in our global environment so run_iris() picks it up
  assign("IRIS_spatial", patched_spatial, envir = globalenv())
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
  message(sprintf("[iris-worker] %s", msg))
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

# Write to <path>.partial and move it into place, so a run killed mid-write never leaves a truncated
# table under the name a reader trusts (the same helper as tools/spotsweeper_worker.R).
write_atomically <- function(path, writer) {
  tmp <- paste0(path, ".partial")
  done <- FALSE
  on.exit(if (!done && file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) stop("Could not move ", tmp, " into place at ", path)
  done <- TRUE
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

# --- Reference annotation columns -------------------------------------------------------------
#
# Which column of the reference annotation holds the cell types is decided by its name, never by
# its position. The old rule grepped five loose patterns in column order and otherwise took the
# first column, and the payload never said which one it used: a Colon VisiumHD reference (columns
# Patient, BC, QCFilter, Level1, Level2, ...) ran with the patient ID as the cell type, and a tonsil
# annotation carrying PrelimCellType before CellType ran on the preliminary labels, because a
# case-insensitive "celltype" pattern matched it first. Exact names now win over names that merely
# contain a cell-type word; a contained match has to be the only one; a lone column is used as it
# is; anything else stops and lists the columns, so the caller names one with --ct-varname.
# Tried in this order; the first rule with exactly one hit decides. A "contains" rule with several
# hits stops rather than picking the first, because the columns are equally good candidates.
CT_RULES <- list(
  list(kind = "exact",    names = c("celltype", "cell_type", "cell.type", "celltypes", "cell_types")),
  list(kind = "contains", names = c("celltype", "cell_type", "cell.type")),
  list(kind = "exact",    names = c("type")),
  list(kind = "exact",    names = c("cluster")),
  list(kind = "contains", names = c("cluster"))
)
SAMPLE_PATTERNS <- c("sample", "batch", "donor")

# The spellings worker_utils.drop_unlabeled treats as "no label". pandas writes NaN as an empty
# field, which read.csv keeps as "" -- a class IRIS would otherwise fit a profile for.
MISSING_LABELS <- c("", "nan", "none", "na")

is_missing_label <- function(x) {
  is.na(x) | tolower(trimws(as.character(x))) %in% MISSING_LABELS
}

resolve_ct_column <- function(meta_cols, requested, source_path) {
  if (!is.null(requested)) {
    if (!(requested %in% meta_cols)) {
      stop("ct_varname '", requested, "' is not a column of ", source_path, "; its columns are: ",
           paste(meta_cols, collapse = ", "), ".")
    }
    return(list(column = requested, how = "named by the caller with ct_varname"))
  }
  lc <- tolower(meta_cols)
  for (rule in CT_RULES) {
    for (nm in rule$names) {
      hit <- if (rule$kind == "exact") which(lc == nm) else which(grepl(nm, lc, fixed = TRUE))
      if (length(hit) == 1) {
        how <- if (rule$kind == "exact") {
          paste0("matched by its name ('", nm, "')")
        } else {
          paste0("the only column whose name contains '", nm, "'")
        }
        return(list(column = meta_cols[hit], how = how))
      }
      if (length(hit) > 1) {
        stop("Several columns of ", source_path, " could hold the cell types (",
             paste(meta_cols[hit], collapse = ", "), ": each is ",
             if (rule$kind == "exact") "named" else "a name containing", " '", nm, "'), and IRIS ",
             "would fit a different reference for each. Pass ct_varname (--ct-varname) to name the ",
             "one to use.")
      }
    }
  }
  if (length(meta_cols) == 1) {
    return(list(column = meta_cols, how = "the annotation's only column"))
  }
  stop("No column of ", source_path, " is named as a cell type (celltype, cell_type, type or ",
       "cluster), and it has ", length(meta_cols), " columns: ", paste(meta_cols, collapse = ", "),
       ". Pass ct_varname (--ct-varname) naming the column that holds the cell types; the first ",
       "column is not taken on trust, since it is as likely a patient or barcode as a label.")
}

resolve_sample_column <- function(meta_cols, requested, ct_varname, source_path) {
  if (!is.null(requested)) {
    if (!(requested %in% meta_cols)) {
      stop("sample_varname '", requested, "' is not a column of ", source_path, "; its columns are: ",
           paste(meta_cols, collapse = ", "), ".")
    }
    return(list(column = requested, how = "named by the caller with sample_varname"))
  }
  candidates <- setdiff(meta_cols, ct_varname)
  for (pat in SAMPLE_PATTERNS) {
    hit <- grep(pat, candidates, ignore.case = TRUE)
    if (length(hit) > 0) {
      how <- paste0("first column whose name contains '", pat, "'")
      # createscRef averages each cell type's profile over the samples this column defines, so a
      # different candidate gives a different reference. The first is used, and the caller is told
      # which others there were rather than finding it only inside params.
      warning_text <- NULL
      if (length(hit) > 1) {
        how <- paste0(how, " (of ", paste(candidates[hit], collapse = ", "), ")")
        warning_text <- paste0(
          length(hit), " columns of ", source_path, " could hold the sample (",
          paste(candidates[hit], collapse = ", "), ": each contains '", pat, "'); '",
          candidates[hit[1]], "' was used because it comes first. IRIS averages each cell type's ",
          "reference profile over these samples, so pass sample_varname (--sample-varname) to ",
          "choose another."
        )
      }
      return(list(column = candidates[hit[1]], how = how, warning = warning_text))
    }
  }
  list(column = NULL, how = "none found; every reference cell was treated as one sample", warning = NULL)
}

# A column name the annotation does not already use. The one-sample stand-in used to be written as
# meta_df$sample unconditionally, which replaced the cell types whenever the cell-type column was
# itself called "sample" (resolve_sample_column never offers the cell-type column as the sample).
unused_column_name <- function(existing, base = "sample") {
  name <- base
  i <- 1L
  while (name %in% existing) {
    name <- paste0(base, "_", i)
    i <- i + 1L
  }
  name
}

# --- Arguments --------------------------------------------------------------------------------

parse_count <- function(key, val, min_value = NULL) {
  n <- suppressWarnings(as.integer(val))
  if (is.na(n) || (!is.null(min_value) && n < min_value)) {
    floor_text <- if (is.null(min_value)) "" else sprintf(" >= %d", min_value)
    stop(sprintf("%s must be an integer%s, not '%s'", key, floor_text, val))
  }
  n
}

parse_switch <- function(key, val) {
  v <- tolower(trimws(val))
  if (v %in% c("true", "1", "yes")) return(TRUE)
  if (v %in% c("false", "0", "no")) return(FALSE)
  stop(sprintf("%s takes true or false, not '%s'", key, val))
}

parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    ct_varname         = NULL,
    sample_varname     = NULL,
    output_dir         = NULL,
    n_clusters         = 7L,
    seed               = 42L,
    drop_unlabeled     = FALSE,
    # createIRISObject's own defaults (minCountGene = 100, minCountSpot = 5), passed by name.
    min_spot_counts    = 100L,
    min_gene_spots     = 5L
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
    } else if (key == "--ct-varname") {
      if (nzchar(val)) opts$ct_varname <- val
    } else if (key == "--sample-varname") {
      if (nzchar(val)) opts$sample_varname <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-clusters") {
      opts$n_clusters <- parse_count(key, val, 1L)
    } else if (key == "--seed") {
      opts$seed <- parse_count(key, val)
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- parse_switch(key, val)
    } else if (key == "--min-spot-counts") {
      opts$min_spot_counts <- parse_count(key, val, 0L)
    } else if (key == "--min-gene-spots") {
      opts$min_gene_spots <- parse_count(key, val, 0L)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# IRIS reseeds every random step it takes: set.seed(islice) before the Dirichlet start of the
# reference model and before the LIGER start of IRISfree, set.seed(12345678) inside both k-means
# helpers (kmeansFunc_Initialize, kmeansFunc_Iter), and rliger's optimizeALS runs on its own
# rand.seed = 1. Nothing else in the run draws a random number, so the result is the same for every
# seed. The seed is still set and echoed, and listed under params.ignored, so the payload does not
# present it as a setting the result depended on.
SEED_IGNORED_WHY <- paste0(
  "IRIS reseeds every random step itself (set.seed(islice) before its Dirichlet/LIGER start, ",
  "set.seed(12345678) in both k-means helpers, rliger optimizeALS rand.seed = 1), so the run is ",
  "deterministic and this value has no effect on the result."
)

run_iris <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$output_dir)) {
    stop("IRIS requires --spatial-counts-csv, --spatial-coords-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  warnings <- character(0)
  ignored <- "seed"
  warnings <- c(warnings, paste0("ignored parameter(s) seed: seed=", opts$seed, " was set, but ",
                                 SEED_IGNORED_WHY))

  # --- Load spatial counts ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  counts_mat <- as.matrix(counts_df)

  # --- Load coordinates ---
  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Header-aware: Space Ranger's headerless tissue_positions_list.csv used to lose its first spot to
  # the header and have its in_tissue flag read as both axes, so IRIS built its spatial neighbourhood
  # on every in-tissue spot sitting at (1, 1).
  coords_read <- read_coords_csv(opts$spatial_coords_csv)
  coords_df <- coords_read$frame

  # Determine orientation: match spots between counts columns and coords rows
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  if (length(common_spots) == 0) {
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}

  # What was read, before anything is left out. A spot with no coordinates cannot be placed on the
  # neighbour graph, so it is set aside here -- and counted, rather than lost in the intersect.
  n_genes_input <- nrow(counts_mat)
  n_spots_input <- ncol(counts_mat)
  n_spots_matched <- length(common_spots)
  n_spots_without_coords <- n_spots_input - n_spots_matched

  # Background spots (in_tissue == 0) are left out and reported, as the Python workers do.
  tissue <- keep_in_tissue_spots(coords_df, common_spots, opts$spatial_coords_csv)
  n_spots_off_tissue <- tissue$n_dropped
  if (n_spots_off_tissue > 0) {
    common_spots <- tissue$spots
    warnings <- c(warnings, in_tissue_warning(tissue, opts$spatial_coords_csv))
    log_msg("WARNING: ", utils::tail(warnings, 1))
  }
  n_spots_in_tissue <- length(common_spots)

  # IRIS expects genes x spots (sparse matrix) and spots x 2 location
  counts_mat <- counts_mat[, common_spots, drop = FALSE]
  coord_cols <- resolve_coord_cols(colnames(coords_df), opts$spatial_coords_csv)
  check_coord_cols(coords_df, coord_cols, opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_df), collapse = ", "), ")")
  coords_df  <- coords_df[common_spots, coord_cols, drop = FALSE]
  colnames(coords_df) <- c("x", "y")
  check_not_one_point(coords_df, coord_cols, opts$spatial_coords_csv)

  log_msg("Data: ", n_genes_input, " genes x ", n_spots_input, " spots (", n_spots_matched,
          " with coordinates, ", n_spots_in_tissue, " of them in tissue)")

  # Convert to sparse matrix
  counts_sparse <- as(counts_mat, "sparseMatrix")

  # createIRISObject's QC, measured before it runs so an empty result stops with the numbers
  # instead of inside IRIS: a gene is kept when it is detected in more than min_gene_spots spots,
  # then a spot is kept when its total over those genes reaches min_spot_counts.
  qc_genes <- Matrix::rowSums(counts_sparse > 0) > opts$min_gene_spots
  qc_totals <- Matrix::colSums(counts_sparse[qc_genes, , drop = FALSE])
  n_spots_pass_qc <- sum(qc_totals >= opts$min_spot_counts)
  if (n_spots_pass_qc < max(2L, opts$n_clusters)) {
    stop("IRIS's QC (createIRISObject) would keep ", n_spots_pass_qc, " of ", n_spots_in_tissue,
         " spots: a spot is kept only with at least min_spot_counts=", opts$min_spot_counts,
         " total counts over the ", sum(qc_genes), " genes detected in more than min_gene_spots=",
         opts$min_gene_spots, " spots, and the median spot here has ", stats::median(qc_totals),
         ". ", opts$n_clusters, " domains need at least that many spots. Lower min_spot_counts ",
         "(or min_gene_spots) to keep more of them.")
  }

  # --- Determine IRIS mode ---
  # file.exists() decides whether to complain here, not which method to run. Selecting the mode
  # from it meant a mistyped --ref-counts-csv silently ran IRISfree instead, and the two modes do
  # not agree: on a 100-spot fixture the reference run split 43/27/30 and the reference-free one
  # 44/24/32, with payloads that were otherwise identical. Same convention as the required-input
  # check above.
  use_reference <- !is.null(opts$ref_counts_csv) || !is.null(opts$ref_celltypes_csv)
  if (use_reference) {
    if (is.null(opts$ref_counts_csv) || is.null(opts$ref_celltypes_csv)) {
      stop("IRIS needs --ref-counts-csv and --ref-celltypes-csv together; only one was given. ",
           "Pass both to run against a reference, or neither to run reference-free.")
    }
    for (f in c(opts$ref_counts_csv, opts$ref_celltypes_csv)) {
      if (!file.exists(f)) stop(sprintf("Reference file not found: %s", f))
    }
  }

  set.seed(opts$seed)

  ref_params <- list()
  ref_data <- list()
  ref_summary <- list()
  ref_sentence <- ""

  if (use_reference) {
    # --- IRIS mode with scRNA reference ---
    log_msg("Using IRIS mode with scRNA-seq reference")
    marker_source <- "scRNA-seq reference cell types"
    method_name <- "IRIS (reference-based: createIRISObject version = 'IRIS' -> IRIS_spatial)"

    log_msg("Reading reference counts from: ", opts$ref_counts_csv)
    ref_df <- read.csv(opts$ref_counts_csv, row.names = 1, check.names = FALSE)
    ref_mat <- as.matrix(ref_df)

    log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
    meta_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE)

    # IRIS wants genes x cells. Which axis holds the cells is read off the annotation's own row
    # names, the way the spot orientation above is read off the coordinates -- not off which axis
    # is longer. That shape rule is right only where the cells outnumber the genes, and the
    # reference this repo stages is 32,397 genes x 113,304 cells, the other way round.
    common_cells <- intersect(colnames(ref_mat), rownames(meta_df))
    if (length(common_cells) == 0) {
      ref_mat <- t(ref_mat)
      common_cells <- intersect(colnames(ref_mat), rownames(meta_df))
    }
    if (length(common_cells) == 0) {
      stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_mat),
                           "cell type annotations", rownames(meta_df)))
    }

    ct <- resolve_ct_column(colnames(meta_df), opts$ct_varname, opts$ref_celltypes_csv)
    ct_varname <- ct$column

    # The two files are lined up by cell ID. IRIS's sc_QC compares rownames(sc_meta) with
    # colnames(sc_count) position by position and stops when they differ, so an annotation in
    # another order, or one with rows for cells the counts do not have, used to end the run with
    # "Cell name in sc_count count data does not match with the rownames of sc_meta".
    n_ref_cells_input <- ncol(ref_mat)
    n_ref_annotation_rows_unmatched <- nrow(meta_df) - length(common_cells)
    n_ref_cells_unannotated <- n_ref_cells_input - length(common_cells)
    meta_df <- meta_df[common_cells, , drop = FALSE]
    ref_mat <- ref_mat[, common_cells, drop = FALSE]

    # A cell with no label is not a class. IRIS's sc_QC silently drops an NA label and fits a
    # profile for "" (what pandas writes for NaN), so both are counted here and refused unless the
    # caller allows them to be left out.
    unlabeled <- is_missing_label(meta_df[[ct_varname]])
    n_ref_cells_unlabeled <- n_ref_cells_unannotated + sum(unlabeled)
    if (n_ref_cells_unlabeled > 0 && !opts$drop_unlabeled) {
      stop(n_ref_cells_unlabeled, " of ", n_ref_cells_input, " reference cells have no label in column '",
           ct_varname, "' (", n_ref_cells_unannotated, " have no row in ", opts$ref_celltypes_csv,
           ", ", sum(unlabeled), " are NA or empty). Pass drop_unlabeled=True to leave them out, or ",
           "label them first; a missing label is not a class.")
    }
    if (any(unlabeled)) {
      meta_df <- meta_df[!unlabeled, , drop = FALSE]
      ref_mat <- ref_mat[, !unlabeled, drop = FALSE]
    }
    if (n_ref_cells_unlabeled > 0) {
      warnings <- c(warnings, paste0(
        n_ref_cells_unlabeled, " of ", n_ref_cells_input, " reference cells had no label in column '",
        ct_varname, "' and were left out (drop_unlabeled=True)."
      ))
    }
    if (n_ref_annotation_rows_unmatched > 0) {
      warnings <- c(warnings, paste0(
        n_ref_annotation_rows_unmatched, " rows of ", opts$ref_celltypes_csv, " name cells that ",
        opts$ref_counts_csv, " does not have; they were not used."
      ))
    }

    smp <- resolve_sample_column(colnames(meta_df), opts$sample_varname, ct_varname,
                                 opts$ref_celltypes_csv)
    sample_varname <- smp$column
    if (!is.null(smp$warning)) warnings <- c(warnings, smp$warning)
    if (is.null(sample_varname)) {
      # One sample for every cell, under a name no existing column has.
      sample_varname <- unused_column_name(colnames(meta_df))
      meta_df[[sample_varname]] <- "sample1"
    }

    # IRIS's sc_QC keeps only cells with counts and cell types with more than one cell, and
    # selectInfo contrasts each type with the mean of the others -- which with two types is a
    # single column, where it fails with "'x' must be an array of at least two dimensions".
    ref_labels <- as.character(meta_df[[ct_varname]])
    ref_has_counts <- colSums(ref_mat) > 0
    type_sizes <- table(ref_labels[ref_has_counts])
    usable_types <- names(type_sizes)[type_sizes > 1]
    singleton_types <- names(type_sizes)[type_sizes <= 1]
    if (length(usable_types) < 3) {
      stop("IRIS's reference mode needs at least 3 cell types with two or more cells each, and column '",
           ct_varname, "' of ", opts$ref_celltypes_csv, " (", ct$how, ") has ", length(usable_types),
           if (length(type_sizes) > 0) paste0(" (", paste(names(type_sizes), collapse = ", "), ")") else "",
           ". Pass ct_varname naming the column that holds the cell types.")
    }
    if (length(singleton_types) > 0) {
      warnings <- c(warnings, paste0(
        "IRIS's sc_QC leaves out cell types with a single cell, so ", length(singleton_types),
        " reference type(s) were not modelled: ", paste(singleton_types, collapse = ", "), "."
      ))
    }

    log_msg("Cell type column: ", ct_varname, " (", ct$how, "), Sample column: ", sample_varname,
            " (", smp$how, ")")
    log_msg("Reference: ", ncol(ref_mat), " cells, ", length(usable_types), " cell types")

    iris_obj <- createIRISObject(
      spatial_countMat_list = list(Slice1 = counts_sparse),
      spatial_location_list = list(Slice1 = coords_df),
      sc_count              = ref_mat,
      sc_meta               = meta_df,
      ct.varname            = ct_varname,
      sample.varname        = sample_varname,
      version               = "IRIS",
      minCountGene          = opts$min_spot_counts,
      minCountSpot          = opts$min_gene_spots
    )

    # Read back from IRIS itself, before IRIS_spatial discards the single-cell object.
    ref_types_used <- as.character(iris_obj@internal_info$ct.select)
    n_ref_cells_used <- ncol(iris_obj@internal_info$sc_eset)

    ref_params <- list(
      ct_varname            = ct_varname,
      ct_varname_source     = ct$how,
      sample_varname        = sample_varname,
      sample_varname_source = smp$how,
      drop_unlabeled        = opts$drop_unlabeled
    )
    ref_data <- list(
      n_ref_cells_input               = n_ref_cells_input,
      n_ref_cells_used                = n_ref_cells_used,
      n_ref_cells_unlabeled           = n_ref_cells_unlabeled,
      n_ref_annotation_rows_unmatched = n_ref_annotation_rows_unmatched,
      n_ref_cell_types                = length(ref_types_used)
    )
    ref_summary <- list(
      ref_cell_types         = I(ref_types_used),
      ref_cell_types_dropped = I(singleton_types)
    )
    ref_sentence <- paste0(
      "Cell types came from column '", ct_varname, "' of the scRNA-seq reference, ", ct$how, ": ",
      length(ref_types_used), " types over ", n_ref_cells_used, " cells. Sample column: '",
      sample_varname, "', ", smp$how, "."
    )
  } else {
    # --- IRISfree mode (no reference) ---
    # IRISfree requires markerList with at least 2 cell types.
    # Generate a simple marker list from top variable genes.
    log_msg("Using IRISfree mode (no scRNA-seq reference provided)")
    method_name <- paste0("IRISfree (reference-free: createIRISObject version = 'IRISfree' -> ",
                          "IRIS_spatial, on marker groups this worker derives from the spatial counts)")

    gene_vars <- apply(counts_mat, 1, var)
    top_genes <- names(sort(gene_vars, decreasing = TRUE))[1:min(100, n_genes_input)]

    # Split into n_clusters groups as pseudo-markers
    n_per_group <- max(2L, length(top_genes) %/% max(2L, opts$n_clusters))
    marker_list <- list()
    for (k in seq_len(min(opts$n_clusters, length(top_genes) %/% n_per_group))) {
      start_idx <- (k - 1L) * n_per_group + 1L
      end_idx <- min(k * n_per_group, length(top_genes))
      marker_list[[paste0("type_", k)]] <- top_genes[start_idx:end_idx]
    }
    # Need at least 2 marker groups
    if (length(marker_list) < 2) {
      half <- length(top_genes) %/% 2
      marker_list <- list(
        type_1 = top_genes[1:half],
        type_2 = top_genes[(half + 1):length(top_genes)]
      )
    }

    log_msg("Generated ", length(marker_list), " marker groups with ",
            sum(sapply(marker_list, length)), " total genes")

    marker_source <- paste0("derived from the spatial counts: top ", length(top_genes),
                            " genes by variance, split into ", length(marker_list),
                            " contiguous blocks")
    # log_msg writes to stderr, and base_mcp returns a status-ok payload without the stderr
    # tail, so the payload is the only place a caller can learn these groups are ours.
    warnings <- c(warnings, paste0(
      "No scRNA-seq reference was given, so IRISfree ran on ", length(marker_list),
      " marker groups this worker built from the spatial counts themselves (", marker_source,
      "). They are not curated cell-type markers. IRIS labels them CT1..CT", length(marker_list),
      " in iris_proportions.csv, where they read as cell types but are gene blocks. Pass ",
      "--ref-counts-csv and --ref-celltypes-csv to run against real annotations instead."
    ))

    # The reference-only settings have nothing to act on without a reference.
    unused <- c(if (!is.null(opts$ct_varname)) "ct_varname",
                if (!is.null(opts$sample_varname)) "sample_varname",
                if (isTRUE(opts$drop_unlabeled)) "drop_unlabeled")
    if (length(unused) > 0) {
      ignored <- c(ignored, unused)
      warnings <- c(warnings, paste0(
        "ignored parameter(s) ", paste(unused, collapse = ", "), ": they describe the scRNA-seq ",
        "reference, and no reference was given, so IRISfree ran without one."
      ))
    }

    iris_obj <- createIRISObject(
      spatial_countMat_list = list(Slice1 = counts_sparse),
      spatial_location_list = list(Slice1 = coords_df),
      sc_count              = NULL,
      sc_meta               = NULL,
      ct.varname            = NULL,
      sample.varname        = NULL,
      markerList            = marker_list,
      version               = "IRISfree",
      minCountGene          = opts$min_spot_counts,
      minCountSpot          = opts$min_gene_spots
    )

    # createIRISObject prints "STOP! The average number of unique marker genes for each cell type
    # is less than 10" to stdout and carries on. Say it where the caller reads.
    n_markers_kept <- length(iris_obj@internal_info$marker)
    if (n_markers_kept < 10L * length(marker_list)) {
      warnings <- c(warnings, paste0(
        "Only ", n_markers_kept, " of the marker genes survived IRIS's gene QC for ",
        length(marker_list), " marker groups -- fewer than the 10 per group IRIS asks for ",
        "(createIRISObject prints 'STOP!' for this and continues)."
      ))
    }
  }

  n_genes_after_qc <- NROW(iris_obj@countList[[1]])
  n_spots_after_qc <- NCOL(iris_obj@countList[[1]])

  # --- Run IRIS spatial domain identification ---
  log_msg("Running IRIS spatial domain identification with ", opts$n_clusters, " clusters...")
  .iris_model_input$n_genes <- NULL
  .iris_model_input$n_spots <- NULL
  iris_obj <- IRIS_spatial(
    iris_obj,
    numCluster = opts$n_clusters
  )
  n_genes_used <- .iris_model_input$n_genes
  if (is.null(n_genes_used)) {
    stop("IRIS_spatial returned without reaching its iteration step, so the genes it modelled ",
         "are unknown; the iteration wrappers at the top of this worker were not installed.")
  }

  # --- Extract domain labels ---
  log_msg("Extracting domain labels...")
  domains <- iris_obj@spatialDomain
  if (is.null(domains) || (is.data.frame(domains) && nrow(domains) == 0)) {
    stop("Could not extract domain labels from IRIS result")
  }

  # spatialDomain is a data.frame with columns:
  #   Slice, spotName, x, y, <version>_domain (e.g. IRISfree_domain or IRIS_domain)
  domain_col <- grep("_domain$", colnames(domains), value = TRUE)
  if (length(domain_col) == 0) {
    # Fallback: last column is typically the domain label
    domain_col <- colnames(domains)[ncol(domains)]
  } else {
    domain_col <- domain_col[1]
  }
  domain_labels <- domains[[domain_col]]

  # --- Save outputs ---
  # The spatialDomain data.frame already contains spot names and coordinates
  domain_df <- data.frame(
    spot   = domains$spotName,
    domain = as.integer(domain_labels),
    stringsAsFactors = FALSE
  )
  domain_path <- file.path(opts$output_dir, "iris_domains.csv")
  write_atomically(domain_path, function(tmp) write.csv(domain_df, tmp, row.names = FALSE, quote = TRUE))

  # Also save the full spatialDomain table (includes coordinates)
  full_domain_path <- file.path(opts$output_dir, "iris_spatial_domains.csv")
  write_atomically(full_domain_path, function(tmp) write.csv(domains, tmp, row.names = FALSE, quote = TRUE))
  log_msg("Saved full domain table to: ", full_domain_path)

  # Also save proportions if available
  prop_path <- NULL
  prop_out <- iris_obj@IRIS_Prop
  if (!is.null(prop_out) && is.data.frame(prop_out) && nrow(prop_out) > 0) {
    prop_path <- file.path(opts$output_dir, "iris_proportions.csv")
    write_atomically(prop_path, function(tmp) write.csv(prop_out, tmp, row.names = FALSE, quote = TRUE))
    log_msg("Saved proportions to: ", prop_path)
  }

  rds_path <- file.path(opts$output_dir, "iris_result.rds")
  write_atomically(rds_path, function(tmp) saveRDS(iris_obj, file = tmp))

  log_msg("Saved domains to: ", domain_path)
  log_msg("Saved IRIS object to: ", rds_path)

  # --- Summary ---
  domain_counts <- as.list(table(domain_labels))
  n_domains_found <- length(unique(domain_labels))

  # The spots that got a domain are the rows written above. Every spot of the counts file that is
  # not among them is accounted for by one of three steps, each counted where it happened.
  n_spots_used <- nrow(domain_df)
  n_spots_dropped <- n_spots_input - n_spots_used
  n_spots_below_qc <- n_spots_in_tissue - n_spots_after_qc
  n_spots_no_model_counts <- n_spots_after_qc - n_spots_used
  drop_sentence <- ""
  if (n_spots_dropped > 0) {
    reasons <- character(0)
    if (n_spots_without_coords > 0) {
      reasons <- c(reasons, paste0(n_spots_without_coords, " have no row in ", opts$spatial_coords_csv))
    }
    if (n_spots_off_tissue > 0) {
      reasons <- c(reasons, paste0(n_spots_off_tissue, " are marked in_tissue == 0 (background outside ",
                                   "the tissue) in ", opts$spatial_coords_csv))
    }
    if (n_spots_below_qc > 0) {
      reasons <- c(reasons, paste0(
        n_spots_below_qc, " fell below IRIS's QC (createIRISObject keeps a spot only with at least ",
        "min_spot_counts=", opts$min_spot_counts, " total counts over the genes detected in more ",
        "than min_gene_spots=", opts$min_gene_spots, " spots)"
      ))
    }
    if (n_spots_no_model_counts > 0) {
      reasons <- c(reasons, paste0(
        n_spots_no_model_counts, " have no counts on the ", n_genes_used,
        " genes IRIS modelled (IRIS_spatial drops them)"
      ))
    }
    drop_sentence <- paste0(
      n_spots_dropped, " of the ", n_spots_input, " spots in the counts have no domain and no row in ",
      "iris_domains.csv: ", paste(reasons, collapse = "; "), ".",
      if (n_spots_below_qc > 0) " Lower min_spot_counts to keep low-count spots." else ""
    )
    warnings <- c(warnings, drop_sentence)
    drop_sentence <- paste0(" ", drop_sentence)
  }

  k_sentence <- ""
  if (n_domains_found != opts$n_clusters) {
    k_sentence <- paste0(" n_clusters=", opts$n_clusters, " was requested; IRIS's k-means returned ",
                         n_domains_found, " non-empty domains.")
    warnings <- c(warnings, trimws(k_sentence))
  }

  output_files <- list(
    domains_csv         = domain_path,
    spatial_domains_csv = full_domain_path,
    result_rds          = rds_path
  )
  # In reference mode this matrix is the per-spot cell-type composition -- the deconvolution
  # result. base_mcp returns a status-ok payload as-is, so output_files is the only channel that
  # tells the caller the file exists. IRISfree can leave IRIS_Prop empty, so advertise it only
  # under the same guard that wrote it.
  if (!is.null(prop_path)) output_files$proportions_csv <- prop_path

  result <- list(
    status       = "ok",
    tool         = "iris_spatial",
    task         = "clustering",
    data         = list(
      n_spots                 = n_spots_used,
      n_genes                 = n_genes_used,
      n_domains               = n_domains_found,
      n_spots_used            = n_spots_used,
      n_genes_used            = n_genes_used,
      n_spots_input           = n_spots_input,
      n_spots_without_coords  = n_spots_without_coords,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_after_qc        = n_spots_after_qc,
      n_spots_dropped         = n_spots_dropped,
      n_genes_input           = n_genes_input,
      n_genes_after_qc        = n_genes_after_qc
    ),
    output_files = output_files,
    params       = list(
      n_clusters       = opts$n_clusters,
      seed             = opts$seed,
      mode             = if (use_reference) "IRIS" else "IRISfree",
      method           = method_name,
      used_fallback    = FALSE,
      marker_source    = marker_source,
      min_spot_counts  = opts$min_spot_counts,
      min_gene_spots   = opts$min_gene_spots,
      coordinate_columns = I(coord_cols),
      coordinates_header = coords_read$header,
      ignored          = I(ignored)
    ),
    warnings     = I(warnings),
    summary      = list(
      n_domains_found  = n_domains_found,
      domain_counts    = domain_counts
    ),
    analysis     = paste0(
      "IRIS identified ", n_domains_found, " spatial domains across ",
      n_spots_used, " spots using ", n_genes_used, " informative genes (of ", n_genes_input,
      " in the counts).", drop_sentence, k_sentence, " ",
      "Domain sizes range from ", min(unlist(domain_counts)),
      " to ", max(unlist(domain_counts)), " spots. ",
      if (use_reference) {
        ref_sentence
      } else {
        paste0("This was a reference-free (IRISfree) run: the marker sets IRISfree requires ",
               "were derived from these spatial counts by gene variance, not from any ",
               "annotation, so any per-type proportions are unnamed gene blocks rather than ",
               "cell types.")
      }
    )
  )
  result$params <- c(result$params, ref_params)
  if (n_spots_off_tissue > 0) result$params$in_tissue_filter <- in_tissue_params(tissue)
  result$data <- c(result$data, ref_data)
  result$summary <- c(result$summary, ref_summary)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args runs inside the handler: a bad flag or value has to come back as an error payload,
  # not as an R abort that leaves nothing on stdout for the portal to read.
  res <- tryCatch(with_r_traceback({
    sink(stderr())
    opts <- parse_args(args)
    result <- run_iris(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "iris_spatial",
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
