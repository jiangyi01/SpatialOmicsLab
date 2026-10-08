#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SpatialPCA)
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
  message(sprintf("[spatialpca-worker] %s", msg))
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


# How many genes a SpatialPCA object's @normalized_expr actually holds.
#
# Measured against SpatialPCA 1.3.0: the constructor slices the scaled matrix with
# `[na.omit(match(SVGnames, rownames(...))), ]` and no drop = FALSE, so a selection of one gene
# comes back as a dimensionless numeric vector rather than a 1-row matrix. nrow() of that is NULL,
# and NULL in a `<` comparison yields logical(0), which is the "argument is of length zero" error.
# Reading the length instead answers the question the comparison is really asking.
selected_gene_count <- function(expr) {
  d <- dim(expr)
  if (length(d) < 2L) {
    if (length(expr) == 0L) 0L else 1L
  } else {
    d[1]
  }
}

# SpatialPCA reseeds the only random steps a run takes: Seurat's SCTransform (which
# CreateSpatialPCAObject calls without seed.use, so Seurat's own default seed.use = 1448145 applies)
# and SpatialPCA_EstimateLoading, whose body opens with set.seed(1234). SPARK's spark() draws no R
# random numbers (its R code has none, and its compiled code imports none of R's RNG), and the
# kernel build and SpatialPCs are deterministic. So --seed changes nothing in the result; it is
# still set and echoed, and listed under params.ignored so the payload does not present it as a
# setting the result depended on.
SEED_IGNORED_WHY <- paste0(
  "SpatialPCA reseeds every random step itself (SCTransform runs with Seurat's seed.use = 1448145, ",
  "and SpatialPCA_EstimateLoading opens with set.seed(1234)); SPARK and the kernel steps draw no ",
  "random numbers, so this value has no effect on the result."
)

# Seurat's CreateSeuratObject(min.cells = 20, min.features = 20), as CreateSpatialPCAObject calls it
# below: a spot with fewer than 20 detected genes is dropped before gene selection.
MIN_FEATURES_PER_SPOT <- 20L

parse_flag <- function(val) {
  tolower(trimws(val)) %in% c("true", "t", "1", "yes", "y")
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

parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    output_dir         = NULL,
    n_components       = 20L,
    seed               = 42L,
    allow_hvg_fallback = FALSE
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
    } else if (key == "--n-components") {
      opts$n_components <- as.integer(val)
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else if (key == "--allow-hvg-fallback") {
      opts$allow_hvg_fallback <- parse_flag(val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  # as.integer() of a non-number is NA, which would otherwise surface much later as "missing value
  # where TRUE/FALSE needed" from the component cap.
  if (is.na(opts$n_components)) stop("--n-components must be an integer")
  if (is.na(opts$seed)) stop("--seed must be an integer")

  opts
}

run_spatialpca <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$output_dir)) {
    stop("SpatialPCA requires --spatial-counts-csv, --spatial-coords-csv, and --output-dir")
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

  # Ensure genes x spots orientation
  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Header-aware: Space Ranger's headerless tissue_positions_list.csv used to lose its first spot to
  # the header and have its in_tissue flag read as both axes, so the SpatialPCA kernel was built on
  # every in-tissue spot sitting at (1, 1).
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

  # What was read, before anything is left out. A spot with no coordinates cannot be placed in the
  # kernel, so it is set aside here -- and counted, rather than lost in the intersect.
  n_spots_input <- ncol(counts_mat)
  n_spots_without_coords <- n_spots_input - length(common_spots)

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
  coords_df  <- coords_df[common_spots, coord_cols, drop = FALSE]
  colnames(coords_df) <- c("x", "y")
  check_not_one_point(coords_df, coord_cols, opts$spatial_coords_csv)
  coords_mat <- as.matrix(coords_df)

  n_genes <- nrow(counts_mat)
  n_spots <- ncol(counts_mat)
  log_msg("Data: ", n_genes, " genes x ", n_spots, " spots")

  # --- Create SpatialPCA object ---
  log_msg("Creating SpatialPCA object...")
  set.seed(opts$seed)

  # Spatial gene selection with SPARK, as SpatialPCA is published. When SPARK's selection is
  # unusable the highly-variable-gene selection runs only if allow_hvg_fallback allows it, and which
  # of the two ran is recorded, because they are not the same method: PCA over highly variable genes
  # with a spatial kernel bolted on is a different analysis that must not be reported in the same words.
  gene_selection <- "spatial"
  gene_selection_reason <- NA_character_
  # Set inside the tryCatch expression (which evaluates in this frame) once SPARK has returned; still
  # NA in the handler means SPARK itself failed rather than selecting too few genes.
  n_sel <- NA_integer_
  spca <- tryCatch({
    obj <- CreateSpatialPCAObject(
      counts    = counts_mat,
      location  = coords_mat,
      project   = "SpatialPCA",
      gene.type = "spatial",
      sparkversion = "spark",
      numCores_spark = 1,
      gene.number = min(3000, n_genes),
      customGenelist = NULL,
      min.loctions = 20,
      min.features = MIN_FEATURES_PER_SPOT
    )
    # Deparsed from SpatialPCA 1.3.0, CreateSpatialPCAObject always returns its object -- it has no
    # return(NULL) at all -- so a too-small selection arrives here as a live object, not as an error.
    # It arrives undersized in a shape R will not measure with nrow(): SVGnames is taken with
    # `[1:significant_gene_number]`, and 1:0 is c(1, 0), so zero significant genes and one
    # significant gene both yield a one-name selection (the library prints "Identified 1 spatial
    # genes" for both). selected_gene_count() reads that shape; see its comment.
    n_sel <- selected_gene_count(obj@normalized_expr)
    if (n_sel < opts$n_components) {
      found <- if (n_sel == 1L) {
        # A count of 1 here is always the dropped-vector shape, in which SPARK's ranking cannot
        # separate none from one, so name the bound rather than asserting either count. A count of
        # 0 comes from a real 0-row matrix (no SVG name matched) and is exact.
        "at most 1 spatially variable gene"
      } else {
        paste0(n_sel, " spatially variable gene(s)")
      }
      stop("SPARK selected ", found, ", fewer than the ",
           opts$n_components, " components requested")
    }
    obj
  }, error = function(e) {
    # The highly-variable-gene selection is a substitute, not SpatialPCA as published, so it runs
    # only when the caller allowed it. Otherwise the run stops with SPARK's finding and the knob.
    if (!isTRUE(opts$allow_hvg_fallback)) {
      remedy <- if (is.na(n_sel)) {
        ""
      } else {
        paste0(" A SPARK selection is used as it is when n_components (", opts$n_components,
               ") is at most its size; SpatialPCA needs at least 3 genes.")
      }
      stop("Spatial gene selection unusable: ", conditionMessage(e), ". SpatialPCA as published ",
           "selects its genes with SPARK; running it over highly variable genes instead is a ",
           "different analysis, done only with allow_hvg_fallback=True.", remedy)
    }
    gene_selection <<- "hvg"
    gene_selection_reason <<- conditionMessage(e)
    log_msg("Spatial gene selection unusable (", conditionMessage(e),
            "); falling back to highly variable genes (allow_hvg_fallback=True)")
    CreateSpatialPCAObject(
      counts    = counts_mat,
      location  = coords_mat,
      project   = "SpatialPCA",
      gene.type = "hvg",
      sparkversion = "spark",
      numCores_spark = 1,
      gene.number = min(3000, n_genes),
      customGenelist = NULL,
      min.loctions = 20,
      min.features = MIN_FEATURES_PER_SPOT
    )
  })

  # Cap n_components to the number of available genes
  n_available_genes <- selected_gene_count(spca@normalized_expr)

  # SpatialPCA cannot decompose fewer than three genes, and asking for fewer components does not
  # help. Measured against SpatialPCA 1.3.0: at two surviving genes SpatialPCA_EstimateLoading dies
  # with "$ operator is invalid for atomic vectors" -- try(optim(...), silent = TRUE) returns a
  # character try-error and the next line reads $par off it, so the real optim failure is already
  # gone -- and RSpectra::eigs_sym independently refuses any matrix smaller than 3x3 for every k.
  # Three genes runs to completion. Neither library message names a gene count or a selection
  # method, and this is the last point at which we still know both.
  if (n_available_genes < 3L) {
    why <- if (is.na(gene_selection_reason)) {
      ""
    } else {
      paste0(" The spatial selection was abandoned because: ", gene_selection_reason)
    }
    stop("Gene selection (", gene_selection, ") left ", n_available_genes,
         " gene(s); SpatialPCA needs at least 3 to estimate loadings.", why)
  }

  # CreateSpatialPCAObject keeps only the spots Seurat's QC kept (min.features); count them before
  # the component cap, which is bounded by the spots actually modelled.
  n_spots_dropped_by_qc <- n_spots - ncol(spca@normalized_expr)
  n_spots <- ncol(spca@normalized_expr)

  actual_components <- min(opts$n_components, n_available_genes - 1L, n_spots - 1L)
  if (actual_components < 1L) actual_components <- 1L
  if (actual_components != opts$n_components) {
    log_msg("Adjusted n_components from ", opts$n_components, " to ", actual_components,
            " (available genes: ", n_available_genes, ")")
    warnings <- c(warnings, paste0(
      "n_components=", opts$n_components, " was requested; ", actual_components,
      " were computed, the most that ", n_available_genes, " selected genes and ", n_spots,
      " spots allow (at most genes - 1 and spots - 1). params.n_components is the number computed."
    ))
  }

  # --- Build kernel ---
  log_msg("Building spatial kernel...")
  spca <- SpatialPCA_buildKernel(
    spca,
    bandwidthtype = "SJ",
    bandwidth.set.by.user = NULL
  )

  # --- Estimate loadings ---
  log_msg("Estimating loadings with ", actual_components, " components...")
  spca <- SpatialPCA_EstimateLoading(
    spca,
    fast    = FALSE,
    SpatialPCnum = actual_components
  )

  # --- Compute spatial PCs ---
  log_msg("Computing spatial PCs...")
  spca <- SpatialPCA_SpatialPCs(spca, fast = FALSE)

  # --- Extract results ---
  spatial_pcs <- spca@SpatialPCs  # n_components x n_spots
  loadings    <- spca@W           # n_genes x n_components

  # --- Save outputs ---
  pcs_df <- as.data.frame(t(spatial_pcs))
  colnames(pcs_df) <- paste0("SpatialPC_", seq_len(nrow(spatial_pcs)))
  # Use spot names from the SpatialPCA location matrix (some spots may be
  # filtered out during CreateSpatialPCAObject by min.loctions/min.features)
  actual_spots <- colnames(spca@normalized_expr)
  if (is.null(actual_spots) || length(actual_spots) != nrow(pcs_df)) {
    actual_spots <- rownames(spca@location)
  }
  if (is.null(actual_spots) || length(actual_spots) != nrow(pcs_df)) {
    actual_spots <- paste0("spot_", seq_len(nrow(pcs_df)))
  }
  pcs_df$spot <- actual_spots
  n_spots <- nrow(pcs_df)  # the spots that got a row in spatialpca_pcs.csv
  pcs_path <- file.path(opts$output_dir, "spatialpca_pcs.csv")
  write_atomically(pcs_path, function(tmp) write.csv(pcs_df, tmp, row.names = FALSE, quote = TRUE))

  loadings_df <- as.data.frame(loadings)
  colnames(loadings_df) <- paste0("SpatialPC_", seq_len(ncol(loadings)))
  loadings_df$gene <- rownames(loadings)
  loadings_path <- file.path(opts$output_dir, "spatialpca_loadings.csv")
  write_atomically(loadings_path, function(tmp) write.csv(loadings_df, tmp, row.names = FALSE, quote = TRUE))

  rds_path <- file.path(opts$output_dir, "spatialpca_result.rds")
  write_atomically(rds_path, function(tmp) saveRDS(spca, file = tmp))

  log_msg("Saved spatial PCs to: ", pcs_path)
  log_msg("Saved loadings to: ", loadings_path)

  # --- Summary ---
  n_pcs_computed <- nrow(spatial_pcs)
  # Measured from the loadings matrix just written, not from the input. Gene selection keeps a
  # small fraction of the genes supplied (a 400-gene input has produced a 44-gene analysis), and
  # publishing nrow(counts_mat) put a number in the payload that its own loadings CSV refuted.
  n_genes_used <- nrow(loadings)
  var_explained <- apply(spatial_pcs, 1, var)
  top_var_pcs <- head(order(var_explained, decreasing = TRUE), 5)

  selection_label <- if (identical(gene_selection, "hvg")) {
    "highly variable genes"
  } else {
    "spatially variable genes (SPARK)"
  }
  fallback_note <- if (identical(gene_selection, "hvg")) {
    paste0(
      " SPARK's spatial gene selection was not usable on this input (", gene_selection_reason,
      "), so the components are over highly variable genes rather than spatially variable ones",
      " (allow_hvg_fallback=True)."
    )
  } else {
    ""
  }
  used_fallback <- identical(gene_selection, "hvg")
  method_name <- if (used_fallback) {
    "SpatialPCA over highly variable genes (SPARK selection unusable; allowed by allow_hvg_fallback)"
  } else {
    "SpatialPCA (SPARK spatially variable genes)"
  }
  if (used_fallback) {
    warnings <- c(warnings, paste0("fallback ran: ", method_name, " -- ", gene_selection_reason))
  }

  # Where every spot read went: the ones without coordinates and the ones Seurat's QC dropped are
  # not in spatialpca_pcs.csv, and the sentence says so rather than reporting the modelled count alone.
  spots_note <- if (n_spots_input > n_spots) {
    paste0(
      " NOTE: of the ", n_spots_input, " spots in the counts, ", n_spots, " were analysed; ",
      n_spots_without_coords, " had no coordinates, ",
      if (n_spots_off_tissue > 0) {
        paste0(n_spots_off_tissue, " are marked in_tissue == 0 (background outside the tissue) ",
               "in the coordinates file, ")
      } else {
        ""
      },
      "and ", n_spots_dropped_by_qc,
      " were dropped by SpatialPCA's QC (fewer than ", MIN_FEATURES_PER_SPOT, " detected genes)."
    )
  } else {
    ""
  }
  component_note <- if (n_pcs_computed != opts$n_components) {
    paste0(" ", n_pcs_computed, " components were computed where ", opts$n_components,
           " were requested: at most (selected genes - 1) and (spots - 1) can be.")
  } else {
    ""
  }

  result <- list(
    status       = "ok",
    tool         = "spatialpca",
    task         = "dim_reduction",
    data         = list(
      n_spots               = n_spots,
      n_spots_input         = n_spots_input,
      n_spots_without_coords = n_spots_without_coords,
      n_spots_off_tissue_dropped = n_spots_off_tissue,
      n_spots_dropped_by_qc = n_spots_dropped_by_qc,
      n_genes               = n_genes_used,
      n_genes_read          = n_genes,
      gene_selection        = gene_selection,
      gene_selection_reason = gene_selection_reason,
      n_components          = n_pcs_computed
    ),
    output_files = list(
      spatial_pcs_csv  = pcs_path,
      loadings_csv     = loadings_path,
      result_rds       = rds_path
    ),
    params       = list(
      n_components           = n_pcs_computed,
      n_components_requested = opts$n_components,
      seed                   = opts$seed,
      allow_hvg_fallback     = isTRUE(opts$allow_hvg_fallback),
      gene_selection         = gene_selection,
      method                 = method_name,
      used_fallback          = used_fallback,
      coordinate_columns     = I(coord_cols),
      coordinates_header     = coords_read$header,
      ignored                = I(ignored)
    ),
    warnings     = I(warnings),
    summary      = list(
      n_pcs_computed   = n_pcs_computed,
      top_variance_pcs = top_var_pcs
    ),
    analysis     = paste0(
      "SpatialPCA computed ", n_pcs_computed, " spatially-aware principal ",
      "components across ", n_spots, " spots using ", n_genes_used, " of the ",
      n_genes, " genes supplied (", selection_label, "). ",
      "The spatial kernel was built with Sheather-Jones bandwidth selection.",
      fallback_note, component_note, spots_note
    )
  )
  if (n_spots_off_tissue > 0) result$params$in_tissue_filter <- in_tissue_params(tissue)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args runs inside the handler, so a bad flag still ends in a JSON error payload.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    sink(stderr())
    result <- run_spatialpca(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "spatialpca",
      task      = "dim_reduction",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
