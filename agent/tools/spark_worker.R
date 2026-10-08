#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(SPARK)
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
  msg <- paste0(...)
  message(sprintf("[spark-worker] %s", msg))
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

# Space Ranger writes the spot table twice, and only one of the two carries a header line:
# outs/spatial/tissue_positions.csv (Space Ranger 2+) has one, tissue_positions_list.csv (Space
# Ranger 1) has none -- barcode, tissue flag, array row, array col, pixel row, pixel col straight
# into the first line. read.csv defaults to header = TRUE, so on the older file the first spot's
# own values became the column names and that spot was dropped. Measured on the SpinalCord
# Targeted slide's tissue_positions_list.csv: the columns came back named ACGCCTGACACGCGCT-1, 0,
# 0, 0, 613, 701; numbers match nothing in the resolver's ladder, so its fall-through took the
# first two non-flag names -- "0" and "0", the SAME column -- and every one of the 2812 in-tissue
# spots was placed at (1, 1). SPARK then died inside its kernel code on "'from' must be a finite
# number", which says nothing about the file that caused it.
#
# Decide from the first line itself: if the first field is a barcode (not empty) and every field
# after it parses as a number, it is data and not a header. Header fields are text, so a file that
# does have one reads exactly as it read before -- for Space Ranger's own header and for every
# vocabulary this portal advertises. The first field must be non-empty because pandas writes an
# unnamed index as an empty header cell: DataFrame(obsm["spatial"], index=obs_names).to_csv()
# starts ",0,1", a header whose names happen to be numbers, and the resolver's fall-through has
# always read that file correctly. The same frame written with a NAMED index (obs_names.name set,
# as many h5ad files have it) starts "barcode,0,1": the first field is text and the rest are
# numbers, so a line whose later fields are exactly pandas' default column labels 0, 1, ..., k-1
# is also a header (read as before this probe existed). On the six-column layout a data line
# would have to be an out-of-tissue spot at array (1, 2) and pixel (3, 4); on an x/y file, a first
# spot at exactly (0, 1) -- which, read as a header, leaves the counts with a spot that has no
# coordinate row, and that is reported in warnings and the analysis rather than lost in silence.
#
# On the six-column layout the recovered names are the ones
# SpotClean's read10xSlide() imposes on this same file, and the resolver's imagerow/imagecol rung
# then picks the pixel coordinates. Any other width keeps the spot and leaves the columns to the
# resolver's fall-through.
#
# The barcode stays as column 1 of the frame (not row names) because the caller offers
# colnames(coords_raw)[-1] to the resolver and reads coords_raw[, 1] as the identifiers. Every
# column is read as character so R cannot mangle a long numeric identifier; the two chosen
# coordinate columns are converted with as.numeric afterwards.
read_coords_csv <- function(path) {
  first <- read.csv(path, header = FALSE, nrows = 1, check.names = FALSE, stringsAsFactors = FALSE)
  # An empty first cell comes back NA, and nzchar(NA) is TRUE, so NA is tested for explicitly.
  lead <- first[[1]][1]
  rest <- first[-1]
  numeric_rest <- ncol(first) > 1 && all(vapply(rest, is.numeric, logical(1)))
  pandas_labels <- numeric_rest &&
    identical(as.numeric(unlist(rest, use.names = FALSE)), as.numeric(seq_len(ncol(rest)) - 1))
  headerless <- numeric_rest && !pandas_labels && !is.na(lead) && nzchar(trimws(as.character(lead)))
  if (!headerless) {
    return(read.csv(path, check.names = FALSE, colClasses = "character"))
  }
  SPACE_RANGER_V1 <- c("barcode", "tissue", "row", "col", "imagerow", "imagecol")
  named <- if (ncol(first) == length(SPACE_RANGER_V1)) {
    SPACE_RANGER_V1
  } else {
    c("barcode", paste0("V", seq_len(ncol(first) - 1L)))
  }
  df <- read.csv(path, header = FALSE, col.names = named, check.names = FALSE,
                 colClasses = "character")
  log_msg("Coordinates file has no header line; read as ",
          if (identical(named, SPACE_RANGER_V1)) "Space Ranger's tissue_positions_list.csv" else "a headerless table",
          " (", nrow(df), " spots, columns ", paste(colnames(df), collapse = ", "), ")")
  # SpotClean's name for Space Ranger's in_tissue flag is "tissue"; a HEADED file's "tissue" column
  # could be anything (a tissue label, say), so only this reader may vouch for it being the flag.
  attr(df, "space_ranger_v1") <- identical(named, SPACE_RANGER_V1)
  df
}

# The column of the coordinates frame holding Space Ranger's in_tissue flag, or NULL. A headed
# file names it in_tissue; the headerless tissue_positions_list.csv was read under SpotClean's
# names above, where it is "tissue".
in_tissue_column <- function(coord_names, space_ranger_v1) {
  hit <- coord_names[tolower(coord_names) == "in_tissue"]
  if (length(hit) > 0) return(hit[1])
  if (isTRUE(space_ranger_v1) && "tissue" %in% coord_names) return("tissue")
  NULL
}

# worker_utils.keep_in_tissue's rule, by hand: a spot is tissue when its flag reads 1 (or TRUE);
# anything else -- 0, FALSE, a blank -- is background. Returns the logical keep vector.
in_tissue_keep <- function(flag) {
  flag <- tolower(trimws(as.character(flag)))
  flag[flag == "true"] <- "1"
  flag[flag == "false"] <- "0"
  value <- suppressWarnings(as.numeric(flag))
  !is.na(value) & value == 1
}

# Non-finite, negative and non-integer values over EVERY value of the counts matrix, a block of
# spots at a time so the logical temporaries stay small beside a whole-transcriptome table.
audit_counts <- function(m, tol = 1e-6, block_values = 1e7) {
  res <- list(n_checked = as.numeric(length(m)), n_non_finite = 0, n_negative = 0,
              n_non_integer = 0, example = NA_real_)
  n_col <- ncol(m)
  step <- max(1L, as.integer(block_values %/% max(1L, nrow(m))))
  start <- 1L
  while (start <= n_col) {
    last <- min(n_col, start + step - 1L)
    b <- m[, start:last, drop = FALSE]
    fin <- is.finite(b)
    res$n_non_finite <- res$n_non_finite + as.numeric(sum(!fin))
    v <- b[fin]
    res$n_negative <- res$n_negative + as.numeric(sum(v < 0))
    off <- abs(v - round(v)) > tol
    k <- as.numeric(sum(off))
    if (k > 0 && is.na(res$example)) res$example <- v[off][1]
    res$n_non_integer <- res$n_non_integer + k
    start <- last + 1L
  }
  res
}

# resolve_coord_cols is a byte-for-byte copy shared by fourteen R workers (a test keeps them in
# step), so the check that its answer is usable lives here, beside the one caller. An answer that
# names a column the file carries more than once is not usable: read.csv keeps duplicate names,
# and [[ returns the first match every time, so two columns both called "0" -- which is what the
# headerless file used to produce -- gave x == y on every spot. When those values vary, SPARK runs
# to status "ok" on a diagonal line that is not the slide; when they do not, it dies far from the
# cause. Either way the message belongs here, naming the file's columns.
#
# A name that is a bare number is NOT refused here. read_coords_csv already reads a first line of
# numbers as data, so a numeric name that reaches this point is a real header cell -- pandas
# writes one for every integer-labelled frame (",0,1"), and that file has always read correctly.
check_coord_cols <- function(coord_cols, coord_names, source_path) {
  same <- coord_cols[1] == coord_cols[2]
  repeated <- coord_cols[coord_cols %in% coord_names[duplicated(coord_names)]]
  if (same || length(repeated) > 0) {
    stop("The coordinate columns chosen from ", source_path, " are '", coord_cols[1], "' and '",
         coord_cols[2], "' (the file has: ", paste(coord_names, collapse = ", "), "). ",
         if (same) "They are the same column, so every spot would sit on the diagonal x == y. " else "",
         if (length(repeated) > 0) paste0("The name(s) ", paste(unique(repeated), collapse = ", "),
                                          " appear more than once in the file, so which column is meant is ambiguous. ") else "",
         "Name the two coordinates imagerow/imagecol, pxl_row_in_fullres/pxl_col_in_fullres, ",
         "array_row/array_col, row/col or x/y, each exactly once, or hand over the file with no ",
         "header line at all (Space Ranger's tissue_positions_list.csv layout is recognised).")
  }
  invisible(coord_cols)
}

# One sentence naming both counts, for a filter that dropped part of the user's input. Mirrors
# worker_utils.describe_reduction, which the Python workers use: empty when nothing was dropped,
# so a run that used the whole input reads exactly as it did before. Deliberately free of the
# substring "Spatial:", which some analysis lines use as a field marker.
reduction_note <- function(noun, n_supplied, n_used, reason = "") {
  if (n_supplied <= 0 || n_used >= n_supplied) return("")
  pct <- 100 * n_used / n_supplied
  because <- if (nzchar(reason)) paste0(" by ", reason) else ""
  paste0(" NOTE: of the ", n_supplied, " ", noun, " supplied, ", n_used, " (", sprintf("%.1f", pct),
         "%) were analysed; ", n_supplied - n_used, " were dropped before the method ran", because,
         ". The results below describe the ", n_used, " analysed ", noun, ", not the full input.")
}

# Atomic write: a reader that globs *spark*.csv while the worker is still writing sees either
# the finished file or nothing, never a truncated table. The output names themselves are unchanged.
write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = FALSE, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place as ", path)
  }
  invisible(path)
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
    percentage   = 0.1,
    min_total_counts = 10L,
    seed         = 0L
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
      opts$n_top <- as.integer(val)
    } else if (key == "--pval-cutoff") {
      opts$pval_cutoff <- as.numeric(val)
    } else if (key == "--percentage") {
      opts$percentage <- as.numeric(val)
    } else if (key == "--min-total-counts") {
      opts$min_total_counts <- as.integer(val)
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_spark <- function(opts) {
  if (is.null(opts$counts_csv) || is.null(opts$coords_csv) || is.null(opts$output_dir)) {
    stop("SPARK requires --counts-csv, --coords-csv, and --output-dir")
  }

  warnings <- character(0)

  # SPARK's variance-component fit (spark.vc) and kernel tests (spark.test) call no random-number
  # generator anywhere in the package namespace, so the seed cannot change the result. It is
  # still set, and still echoed in params, but it is listed under params.ignored so the payload
  # does not present it as a setting the run depended on.
  warnings <- c(warnings, paste0(
    "ignored parameter(s) seed: SPARK's variance-component fit and kernel tests are deterministic ",
    "(no random-number generator is used), so seed=", opts$seed, " has no effect on the result."
  ))

  log_msg("Reading counts from: ", opts$counts_csv)
  # Read header to determine column names (may be long numeric IDs)
  counts_header <- read.csv(opts$counts_csv, nrows = 1, check.names = FALSE)
  counts_df <- read.csv(opts$counts_csv, row.names = 1, check.names = FALSE,
                        colClasses = c("character", rep("numeric", ncol(counts_header) - 1)))
  counts_mat <- as.matrix(counts_df)

  log_msg("Reading coordinates from: ", opts$coords_csv)
  coords_raw <- read_coords_csv(opts$coords_csv)
  flag_col <- in_tissue_column(colnames(coords_raw)[-1], attr(coords_raw, "space_ranger_v1"))
  # First column is spot ID (index), rest are coordinates and flags
  rownames_col <- coords_raw[, 1]
  coord_cols <- resolve_coord_cols(colnames(coords_raw)[-1], opts$coords_csv)
  check_coord_cols(coord_cols, colnames(coords_raw), opts$coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(coords_raw), collapse = ", "), ")")
  coords_df <- data.frame(
    x = as.numeric(coords_raw[[coord_cols[1]]]),
    y = as.numeric(coords_raw[[coord_cols[2]]]),
    row.names = rownames_col
  )

  # Ensure matching spot/cell IDs. The portal advertises genes x spots; a spots x genes table is
  # recognised by its column names matching no coordinate row, transposed, and SAID so -- in the
  # log, in params.counts_orientation and in warnings -- rather than silently.
  # The retry sits directly above the abort, whose message says both orientations were tried
  # (test/test_r_workers_try_the_other_orientation_before_the_abort_that_claims_it.py); the
  # transpose is reported only once the abort has been passed, i.e. when it actually matched.
  n_spots_in_counts <- ncol(counts_mat)
  counts_orientation <- "genes_x_spots"
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  transposed <- FALSE
  if (length(common_spots) == 0) {
    # Try transposing counts (cells may be rows)
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
    transposed <- TRUE
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot/cell IDs", "counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}
  if (transposed) {
    n_spots_in_counts <- ncol(counts_mat)
    counts_orientation <- "spots_x_genes (transposed to genes x spots)"
    log_msg("Counts CSV is spots x genes (no column name matched a coordinate row); ",
            "transposed to genes x spots")
    warnings <- c(warnings, paste0(
      "The counts CSV was spots x genes -- none of its column names matched a coordinate row -- ",
      "so it was transposed to genes x spots before SPARK ran (params.counts_orientation)."
    ))
  }

  counts_mat <- counts_mat[, common_spots, drop = FALSE]
  coords_df <- coords_df[common_spots, , drop = FALSE]
  # Background spots: a coordinates file that carries Space Ranger's in_tissue flag (a
  # tissue_positions file, beside a raw matrix of every array spot) says which matched spots are
  # glass. Their ambient counts would make the tissue/background edge a "spatial pattern" in every
  # gene, so they are left out -- worker_utils.keep_in_tissue's rule -- and reported.
  n_spots_matched <- length(common_spots)
  n_spots_off_tissue <- 0L
  if (!is.null(flag_col)) {
    flag_by_spot <- stats::setNames(coords_raw[[flag_col]], rownames_col)
    keep <- in_tissue_keep(flag_by_spot[common_spots])
    if (!any(keep)) {
      stop("Column '", flag_col, "' of ", opts$coords_csv, " marks none of the ", n_spots_matched,
           " spots matched to the counts as in tissue (values seen: ",
           paste(utils::head(sort(unique(as.character(flag_by_spot[common_spots]))), 8), collapse = ", "),
           "); fix the column so in-tissue spots are 1, or remove it if every spot is tissue.")
    }
    if (!all(keep)) {
      n_spots_off_tissue <- sum(!keep)
      counts_mat <- counts_mat[, keep, drop = FALSE]
      coords_df <- coords_df[keep, , drop = FALSE]
      log_msg("Left out ", n_spots_off_tissue, " of ", n_spots_matched, " spots with ", flag_col,
              " == 0 (background); ", sum(keep), " in-tissue spots are analysed")
      warnings <- c(warnings, paste0(
        n_spots_off_tissue, " of ", n_spots_matched, " spots have Space Ranger's in_tissue flag at 0 (column '",
        flag_col, "' of ", opts$coords_csv, "; background outside the tissue) and were left out; ", sum(keep),
        " in-tissue spots were analysed."
      ))
    }
  }
  tissue_note <- reduction_note(
    "spots", n_spots_matched, ncol(counts_mat),
    "leaving out the background spots the coordinates file flags in_tissue == 0"
  )
  # Checked on the matched spots only: a coordinates file may carry rows for spots the counts do
  # not have (every array spot, say), and a blank there has never mattered. A blank on a spot
  # that IS analysed would reach SPARK's distance matrix as NA and fail far from the cause.
  for (axis in c("x", "y")) {
    n_bad <- sum(!is.finite(coords_df[[axis]]))
    if (n_bad > 0) {
      stop("Coordinate column '", coord_cols[match(axis, c("x", "y"))], "' of ", opts$coords_csv,
           " has ", n_bad, " value(s) that are not numbers on the ", nrow(coords_df), " spots matched ",
           "to the counts (first: ", paste(utils::head(rownames(coords_df)[!is.finite(coords_df[[axis]])], 3),
           collapse = ", "), "). A coordinate must be numeric; the file carries the columns: ",
           paste(colnames(coords_raw), collapse = ", "))
    }
  }
  # A counts spot with no coordinate row cannot enter a spatial test; it is left out, and the
  # count is said in warnings AND in the analysis, which is the text a reader actually quotes.
  match_note <- reduction_note(
    "spots", n_spots_in_counts, length(common_spots),
    "matching the counts CSV to the coordinates file (a spot with no coordinate row cannot be tested)"
  )
  if (nzchar(match_note)) {
    warnings <- c(warnings, paste0(
      n_spots_in_counts - length(common_spots), " of the ", n_spots_in_counts,
      " spots in the counts CSV have no row in the coordinates file and were left out; ",
      length(common_spots), " spots were matched."
    ))
  }

  n_spots <- ncol(counts_mat)
  n_genes_input <- nrow(counts_mat)
  log_msg("N genes = ", n_genes_input, ", N spots = ", n_spots, " (counts ", counts_orientation, ")")

  # SPARK fits a Poisson count model (spark.vc's glm(family = poisson)), and R's glm only WARNS on
  # non-integer y -- so a log-normalised table ran to status "ok" and its p-values were published as
  # spatially variable genes. convert_h5ad_to_csv writes adata.X, which on CELLxGENE files is often
  # normalised. Every value of the analysed table is checked; nothing is rounded.
  audit <- audit_counts(counts_mat)
  whole <- function(x) format(x, scientific = FALSE, trim = TRUE)
  if (audit$n_non_finite > 0 || audit$n_negative > 0) {
    stop("The counts CSV ", opts$counts_csv, " holds ", whole(audit$n_non_finite), " missing/NaN/infinite and ",
         whole(audit$n_negative), " negative values of the ", whole(audit$n_checked), " analysed (", n_genes_input,
         " genes x ", n_spots, " spots; a blank cell reads as missing). SPARK fits a Poisson count model, ",
         "so these are not counts it can model -- this looks like scaled or centred data. Pass a table of raw ",
         "integer counts: convert_h5ad_to_csv writes adata.X, so export from an h5ad whose X holds the raw ",
         "counts (e.g. its adata.raw or layers['counts']).")
  }
  if (audit$n_non_integer > 0) {
    stop(whole(audit$n_non_integer), " of the ", whole(audit$n_checked), " analysed values in the counts CSV ",
         opts$counts_csv, " are not integers (e.g. ", format(audit$example, digits = 6), "): this looks ",
         "like normalised data. SPARK fits a Poisson count model, and R's glm only warns on non-integer ",
         "counts, so the run would return p-values for values that are not counts. Pass a table of raw ",
         "integer counts: convert_h5ad_to_csv writes adata.X, so export from an h5ad whose X holds the raw ",
         "counts (e.g. its adata.raw or layers['counts']). Nothing was rounded.")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  location <- as.data.frame(coords_df[, 1:2])
  colnames(location) <- c("x", "y")

  # CreateSPARKObject applies two filters of its own, in this order: a GENE is kept when it is
  # non-zero in at least floor(percentage * n_spots) spots, then a SPOT is kept when its total
  # count over the kept genes is strictly greater than min_total_counts. The spot filter is the
  # one the docs used to describe as a per-gene minimum; both are read back from the object so
  # the payload reports what SPARK analysed rather than what it was handed.
  log_msg("Creating SPARK object...")
  spark_obj <- CreateSPARKObject(
    counts     = counts_mat,
    location   = location,
    percentage = opts$percentage,
    min_total_counts = opts$min_total_counts
  )
  n_spots_used <- nrow(spark_obj@location)
  n_genes_used <- nrow(spark_obj@counts)
  log_msg("After SPARK's filters: ", n_genes_used, " genes (percentage >= ", opts$percentage,
          "), ", n_spots_used, " spots (total counts > ", opts$min_total_counts, ")")
  if (n_genes_used == 0) {
    stop("SPARK's percentage filter (percentage=", opts$percentage, ": a gene must be non-zero in at ",
         "least ", floor(opts$percentage * n_spots), " of ", n_spots, " spots) kept none of the ",
         n_genes_input, " genes. Lower percentage.")
  }
  if (n_spots_used == 0) {
    stop("SPARK's min_total_counts filter (min_total_counts=", opts$min_total_counts, ": a spot must ",
         "have a total count over the ", n_genes_used, " retained genes greater than that) kept ",
         "none of the ", n_spots, " spots. Lower min_total_counts, or lower percentage so more ",
         "genes count towards each spot's total.")
  }
  gene_note <- reduction_note(
    "genes", n_genes_input, n_genes_used,
    paste0("SPARK's percentage filter (percentage=", opts$percentage, ": a gene is tested only when ",
           "it is non-zero in at least ", floor(opts$percentage * n_spots), " of ", n_spots, " spots)")
  )
  spot_note <- reduction_note(
    "spots", n_spots, n_spots_used,
    paste0("SPARK's min_total_counts filter (min_total_counts=", opts$min_total_counts,
           ": a spot is kept only when its total count over the ", n_genes_used,
           " retained genes is greater than ", opts$min_total_counts, ")")
  )
  if (nzchar(spot_note)) {
    warnings <- c(warnings, trimws(spot_note))
  }

  # R warnings raised inside SPARK's fit (glm convergence, fitted rates of 0, ...) used to reach
  # stderr only, which a successful run's payload never carries. They are collected and said.
  r_warnings <- character(0)
  collect_warning <- function(w) {
    r_warnings <<- c(r_warnings, conditionMessage(w))
    invokeRestart("muffleWarning")
  }

  log_msg("Fitting variance components (spark.vc)...")
  spark_obj@lib_size <- apply(spark_obj@counts, 2, sum)
  spark_obj <- withCallingHandlers(spark.vc(
    spark_obj,
    covariates = NULL,
    lib_size   = spark_obj@lib_size,
    num_core   = 1L,
    verbose    = FALSE
  ), warning = collect_warning)

  log_msg("Running SPARK test (spark.test)...")
  spark_obj <- withCallingHandlers(spark.test(
    spark_obj,
    check_positive = TRUE,
    verbose = FALSE
  ), warning = collect_warning)

  if (length(r_warnings) > 0) {
    distinct <- unique(r_warnings)
    log_msg("SPARK raised ", length(r_warnings), " R warning(s); first distinct: ",
            paste(utils::head(distinct, 3), collapse = " | "))
    warnings <- c(warnings, paste0(
      "SPARK raised ", length(r_warnings), " R warning(s) while fitting (", length(distinct),
      " distinct; first: ", distinct[1], ")."
    ))
  }

  # Extract results
  results_df <- spark_obj@res_mtest
  results_df$gene <- rownames(results_df)

  # Sort by combined p-value
  if ("combined_pvalue" %in% colnames(results_df)) {
    results_df <- results_df[order(results_df$combined_pvalue), ]
  } else if ("adjusted_pvalue" %in% colnames(results_df)) {
    results_df <- results_df[order(results_df$adjusted_pvalue), ]
  }

  # Save full results
  results_path <- file.path(opts$output_dir, "spark_results.csv")
  write_csv_atomic(results_df, results_path)

  # Identify significant SVGs. SPARK's adjusted_pvalue is its Benjamini-Yekutieli correction of
  # the Cauchy-combined kernel p-value; the column the cutoff was applied to is named in params
  # and in the analysis, because "p < 0.05" reads as the raw p-value and it never was.
  if ("adjusted_pvalue" %in% colnames(results_df)) {
    pvalue_column <- "adjusted_pvalue"
    pvalue_adjustment <- "Benjamini-Yekutieli (SPARK's own p.adjust(method = 'BY') of combined_pvalue)"
  } else if ("combined_pvalue" %in% colnames(results_df)) {
    pvalue_column <- "combined_pvalue"
    pvalue_adjustment <- "none (SPARK returned no adjusted_pvalue column)"
  } else {
    stop("SPARK's res_mtest carries neither adjusted_pvalue nor combined_pvalue (columns: ",
         paste(colnames(results_df), collapse = ", "), "), so no significance cutoff can be applied.")
  }
  sig_genes <- results_df[which(results_df[[pvalue_column]] < opts$pval_cutoff), , drop = FALSE]

  sig_path <- file.path(opts$output_dir, "spark_significant_svgs.csv")
  write_csv_atomic(sig_genes, sig_path)

  # Top N genes
  top_genes <- head(results_df, opts$n_top)
  top_path <- file.path(opts$output_dir, "spark_top_svgs.csv")
  write_csv_atomic(top_genes, top_path)

  result <- list(
    status       = "ok",
    tool         = "spark",
    task         = "svg_detection",
    data         = list(
      n_spots           = n_spots,
      n_genes_input     = n_genes_input,
      n_spots_in_counts = n_spots_in_counts,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_used      = n_spots_used,
      n_genes_used      = n_genes_used
    ),
    output_files = list(
      results_csv       = results_path,
      significant_csv   = sig_path,
      top_svgs_csv      = top_path
    ),
    params       = list(
      pval_cutoff       = opts$pval_cutoff,
      percentage        = opts$percentage,
      min_total_counts  = opts$min_total_counts,
      n_top             = opts$n_top,
      seed              = opts$seed,
      method            = "SPARK (CreateSPARKObject -> spark.vc -> spark.test)",
      used_fallback     = FALSE,
      counts_orientation = counts_orientation,
      pvalue_column     = pvalue_column,
      pvalue_adjustment = pvalue_adjustment,
      ignored           = I("seed")
    ),
    summary      = list(
      n_genes_tested    = nrow(results_df),
      n_significant     = nrow(sig_genes),
      top_genes         = head(results_df$gene, min(20, nrow(results_df)))
    ),
    warnings     = I(warnings),
    analysis     = paste0(
      "SPARK identified ", nrow(sig_genes), " significant spatially variable genes ",
      "out of ", nrow(results_df), " tested (", pvalue_column, " < ", opts$pval_cutoff,
      "; adjustment: ", pvalue_adjustment, ") on ", n_spots_used, " spots.",
      match_note, tissue_note, gene_note, spot_note
    )
  )
  # worker_utils.record_in_tissue's shape, present only when a background spot was left out.
  if (n_spots_off_tissue > 0) {
    result$params$in_tissue_filter <- list(
      n_spots_supplied           = n_spots_matched,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_used               = n_spots_matched - n_spots_off_tissue
    )
  }
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  set.seed(opts$seed)

  res <- tryCatch(with_r_traceback({
    run_spark(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "spark",
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
