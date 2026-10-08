#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SpotClean)
  library(SummarizedExperiment)
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
  message(sprintf("[spotclean-worker] %s", msg))
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

# A cutoff that is not a whole number used to become NA, and the NA filter mask then died inside the
# subset with "logical subscript contains NAs", naming neither the flag nor the value.
parse_cutoff <- function(key, val) {
  out <- suppressWarnings(as.numeric(val))
  if (is.na(out) || !is.finite(out) || out < 0 || out != round(out)) {
    stop(sprintf("%s must be a non-negative whole number of counts, got '%s'", key, val))
  }
  as.integer(out)
}

parse_args <- function(args) {
  if (length(args) > 0 && args[[1]] %in% c("--help", "-h")) {
    cat("Usage: spotclean_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --gene-cutoff INT          Minimum total count per gene (default: 10)\n")
    cat("  --spot-cutoff INT          Minimum total count per spot (default: 100)\n")
    cat("  --verbose BOOL             Verbose output (default: TRUE)\n")
    cat("  --allow-array-index-fallback BOOL\n")
    cat("                             With no pixel columns, read Visium hexagonal array indices\n")
    cat("                             (array_row/array_col) as the lattice's physical layout (default: FALSE)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    output_dir         = NULL,
    gene_cutoff        = 10L,
    spot_cutoff        = 100L,
    verbose            = TRUE,
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
    } else if (key == "--gene-cutoff") {
      opts$gene_cutoff <- parse_cutoff(key, val)
    } else if (key == "--spot-cutoff") {
      opts$spot_cutoff <- parse_cutoff(key, val)
    } else if (key == "--verbose") {
      opts$verbose <- toupper(val) == "TRUE"
    } else if (key == "--allow-array-index-fallback") {
      opts$allow_array_index_fallback <- toupper(val) == "TRUE"
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# SpotClean takes its one length scale from the array indices, not from the image coordinates:
# .SpotClean() regresses imagecol on col and uses that slope as the Gaussian bandwidth behind
# every contamination weight. Our coordinate CSVs carry only x and y (data_converter_worker.py
# writes index,x,y), so the indices have to be reconstructed from the geometry. Measured on the
# real Space Ranger slide SpotClean ships with itself (V1_Adult_Mouse_Brain, 4992 spots): half
# the median nearest-neighbour distance is 3.4958 against a true array-column step of 3.5113,
# and indices derived that way correlate with imagecol at 0.99997 -- close enough that SpotClean
# takes its intended branch and recovers the spacing we measured for it.
median_nn_distance <- function(x, y, probe_cap = 1000L) {
  n <- length(x)
  if (n < 2L) return(NA_real_)

  # Probe a stride through the coordinate-sorted spots rather than all of them: the full pairwise
  # sweep costs ~20s at n = 20000 and the median settles long before that. Sorting by coordinate
  # is what makes the answer independent of the order the CSV happened to list the spots in.
  ord <- order(x, y)
  probe <- if (n <= probe_cap) ord else ord[unique(round(seq(1, n, length.out = probe_cap)))]

  nn <- numeric(length(probe))
  chunk <- max(1L, min(length(probe), as.integer(2e6 %/% n)))
  for (s0 in seq(1L, length(probe), by = chunk)) {
    sel <- s0:min(s0 + chunk - 1L, length(probe))
    idx <- probe[sel]
    d2 <- outer(x[idx], x, "-")^2 + outer(y[idx], y, "-")^2
    d2[cbind(seq_along(idx), idx)] <- Inf
    nn[sel] <- sqrt(apply(d2, 1, min))
  }

  # A pair of coincident spots measures 0 and a lone spot measures Inf; neither scales anything.
  nn <- nn[is.finite(nn) & nn > 0]
  if (length(nn) == 0L) return(NA_real_)
  median(nn)
}

# Space Ranger writes the spot table twice, and only one of the two carries a header line:
# outs/spatial/tissue_positions.csv (Space Ranger 2+) has one, tissue_positions_list.csv (Space
# Ranger 1) has none -- barcode, tissue flag, array row, array col, pixel row, pixel col straight
# into the first line. read.csv defaults to header = TRUE, so on the older file the first spot's own
# values became the column names and that spot was dropped: measured on a six-spot slide, five rows
# survived and the columns came back named 1, 0, 0, 100, 100. Numbers match no name in the flag
# lookup or the coordinate ladder below, so the flag and the array row index were then taken as the
# image coordinates, and the run still finished at status "ok" -- on one 400-spot slide, 399 spots,
# no decontamination at all, and a median spot spacing of 0.1029 against a true 98.228.
#
# Decide from the first line itself: if every field after the barcode parses as a number, it is data
# and not a header. Header fields are text, so a file that does have one reads exactly as it read
# before -- for Space Ranger's own header and for every vocabulary this portal advertises. On the
# six-column layout the recovered names are the ones SpotClean's own read10xSlide() imposes on this
# same file; any other width keeps the spot and leaves the columns to the fallback further down.
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

# A tissue/background flag holds 0/1 (or TRUE/FALSE, in any case, or those as text). Returns the
# column as integer 0/1 with NA where a spot has no value, or NULL when the column is not a flag at
# all -- a label such as 'thymus', a code such as 2, or nothing but blanks.
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

# Write to <path>.partial and move it into place, so a run killed mid-write never leaves a
# truncated table under the name a reader trusts. Same helper as tools/spotsweeper_worker.R; each R
# worker runs in its own env, so each carries its own copy.
write_atomically <- function(path, writer) {
  tmp <- paste0(path, ".partial")
  done <- FALSE
  on.exit(if (!done && file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) stop("Could not move ", tmp, " into place at ", path)
  done <- TRUE
  invisible(path)
}

# The array-index column pairs a coordinates file can name, row first. Tried only after every pixel
# spelling (see the ladder in run_spotclean), because indices are not distances on every lattice.
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

run_spotclean <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$output_dir)) {
    stop("SpotClean requires --spatial-counts-csv, --spatial-coords-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  counts_mat <- as.matrix(counts_df)

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  coords_df <- read_coords_csv(opts$spatial_coords_csv)

  # Ensure genes x spots orientation: try to match colnames to coord rownames
  common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  if (length(common_spots) == 0) {
    counts_mat <- t(counts_mat)
    common_spots <- intersect(colnames(counts_mat), rownames(coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "counts", colnames(counts_mat), "coordinates", rownames(coords_df)))}

  counts_mat <- counts_mat[, common_spots, drop = FALSE]

  # Preserve the tissue/background flag if the file carries it, before subsetting coords.
  # SpotClean's own read10xSlide() calls that flag 'tissue'. Space Ranger calls it 'in_tissue', in
  # outs/spatial/tissue_positions.csv -- the only file in a Visium run that says which spots are
  # background, which is the split SpotClean exists to model. Reading only the first spelling meant
  # the file a user actually has could not be used: measured on one 400-spot slide, 256 of them
  # background, the Space Ranger header found no flag, fell through to in_tissue/array_row as the
  # coordinates, and died in the spacing guard below telling the user their spots were coincident.
  # 'tissue' is tried first, so a file carrying both spellings resolves as it resolves today. The
  # column found is renamed to 'tissue' so everything downstream reads one spelling; flag_name
  # keeps what the file called it, for the log and the payload.
  #
  # A name is not enough: the column also has to hold a 0/1 flag. CELLxGENE Visium objects carry an
  # integer in_tissue AND a text 'tissue' column ('thymus', 'outflow tract myocardium', ...), and a
  # coordinates file exported from their obs -- the converter's metadata.csv is exactly that -- put
  # the text first. as.integer() made it NA, n_background became NA, and the run died on
  # `if (n_background == 0)` with "missing value where TRUE/FALSE needed", naming no column, while
  # the real flag was left to be read as a coordinate. A named column that is not a flag is now set
  # aside -- dropped from the frame, so it cannot be a coordinate either -- and the payload says so.
  TISSUE_COLS <- c("tissue", "in_tissue")
  flag_hits <- TISSUE_COLS[TISSUE_COLS %in% tolower(colnames(coords_df))]
  flag_name <- NA_character_
  flag_notes <- character(0)
  for (hit in flag_hits) {
    at <- match(hit, tolower(colnames(coords_df)))
    parsed <- as_tissue_flag(coords_df[[at]])
    if (is.null(parsed)) {
      seen <- utils::head(unique(as.character(coords_df[[at]])), 3)
      flag_notes <- c(flag_notes, paste0(
        "Column '", colnames(coords_df)[at], "' of the coordinates file is not a 0/1 tissue flag (values ",
        paste0("'", seen, "'", collapse = ", "), "), so it was neither used as the tissue/background ",
        "flag nor read as a coordinate."))
      log_msg(flag_notes[length(flag_notes)])
      coords_df[[at]] <- NULL
      next
    }
    flag_name <- colnames(coords_df)[at]
    coords_df[[at]] <- parsed
    colnames(coords_df)[at] <- "tissue"
    break
  }
  has_tissue <- !is.na(flag_name)
  tissue_col <- if (has_tissue) coords_df[common_spots, "tissue"] else NULL

  # Where the background went, for the no-background explanation below. The flag column names the
  # background over the WHOLE coordinates file, and two things can keep every one of those spots from
  # reaching SpotClean while the flag was read correctly: the counts may not carry them at all (Space
  # Ranger's filtered_feature_bc_matrix, and an h5ad of the in-tissue spots, hold only the tissue -- every
  # Space Ranger sample in the library ships that way, e.g. V1_Human_Heart: 745 background rows in
  # tissue_positions.csv, none in the matrix), or spot_cutoff may remove them (every background spot of
  # the CELLxGENE Muscle and Skin samples has fewer than 100 counts). Counted with which() rather than a
  # bare sum() so neither can be mistaken for the tissue/background tally further down.
  # Logical NA, not NA_integer_: jsonlite writes a numeric NA as the string "NA" and only a logical
  # one as null, which is what a run with no flag column has for these two.
  n_coord_rows <- nrow(coords_df)
  n_background_in_coords <- NA
  n_background_not_in_counts <- NA
  if (has_tissue) {
    background_ids <- rownames(coords_df)[which(coords_df[["tissue"]] == 0L)]
    n_background_in_coords <- length(background_ids)
    n_background_not_in_counts <- length(setdiff(background_ids, common_spots))
  }

  # The flag is read out of this frame by name, so it must not also be read as a coordinate.
  # SpotClean's own read10xSlide() names Space Ranger's tissue-positions file as
  # barcode, tissue, row, col, imagerow, imagecol -- the flag FIRST -- so on the canonical input
  # columns 1:2 were the flag and the array row index, and the pixel coordinates were never read.
  # Measured on one 400-spot slide written x,y,tissue and again tissue,x,y, both runs status "ok":
  # median_spot_spacing 98.228 -> 0.1037 and mean_contamination_pct 22.65 -> 5.02, with 17913 of
  # 28800 cleaned values differing by more than 1 and no field in either payload saying so.
  #
  # Take the columns the file names -- SpotClean's own imagerow/imagecol first, then Space Ranger's
  # pxl_*_in_fullres, then the x/y this portal advertises -- and otherwise the first two that are
  # not the flag. Most specific first: x/y stays last because it is the one pair that could be
  # array indices, microns or pixels, so anything more specific has to precede it to be reachable.
  # Every input that reads correctly today selects the same two columns it selected before: a bare
  # x,y or x,y,tissue file matches on x/y as it did, and an unnamed pair still falls through to the
  # first two columns.
  #
  # Array indices come after all of those, by name, row first. On a file with no pixel columns --
  # the converter's metadata.csv for a CELLxGENE Visium object carries in_tissue, array_row,
  # array_col and no pixel position -- the unnamed fall-through used to take array_row/array_col
  # (its first two numeric columns) and hand them to SpotClean's Euclidean kernel as if they were
  # pixels. On Visium's hexagonal lattice that is not a distance (see array_index_lattice), so it is
  # checked below and refused unless the caller allows the lattice conversion.
  coord_names <- colnames(coords_df)
  lc_names <- tolower(coord_names)
  coord_cols <- NULL
  for (pair in list(c("imagerow", "imagecol"), c("pxl_row_in_fullres", "pxl_col_in_fullres"),
                    c("x", "y"))) {
    if (all(pair %in% lc_names)) {
      coord_cols <- coord_names[match(pair, lc_names)]
      break
    }
  }
  index_pair <- FALSE
  if (is.null(coord_cols)) {
    for (pair in ARRAY_INDEX_PAIRS) {
      if (all(pair %in% lc_names)) {
        coord_cols <- coord_names[match(pair, lc_names)]
        index_pair <- TRUE
        break
      }
    }
  }
  if (is.null(coord_cols)) {
    # Only a numeric column can be a coordinate, and neither flag spelling is one: when both are
    # present only one of them became 'tissue' above, and the other must not become imagerow.
    numeric_names <- coord_names[vapply(coords_df, is.numeric, logical(1))]
    coord_cols <- numeric_names[!(tolower(numeric_names) %in% c("tissue", TISSUE_COLS))]
  }
  if (length(coord_cols) < 2) {
    stop("Need two coordinate columns in ", opts$spatial_coords_csv, " and found ",
         length(coord_cols), " numeric one(s) after setting aside the tissue/background flag; the file has: ",
         paste(coord_names, collapse = ", "),
         ". Name the two image coordinates imagerow/imagecol, pxl_row_in_fullres/",
         "pxl_col_in_fullres or x/y, or give them as the only two numeric columns besides the flag.")
  }
  coord_cols <- coord_cols[seq_len(2)]
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(coord_names, collapse = ", "), ")")

  coords_df <- coords_df[common_spots, coord_cols, drop = FALSE]
  colnames(coords_df) <- c("x", "y")

  # What the two columns are, and whether they can be read as distances. SpotClean's kernel is an
  # isotropic Gaussian on the Euclidean distance between (imagerow, imagecol), so pixel positions are
  # used as they are and so are the indices of a square lattice (one scale, which the spacing
  # measured below divides out). Visium's hexagonal indices are not: read as they are, the kernel is
  # stretched by up to sqrt(2) and the contamination estimate still comes back "ok".
  coordinate_notes <- character(0)
  used_fallback <- FALSE
  coordinate_kind <- if (tolower(coord_cols[1]) %in% c("imagerow", "pxl_row_in_fullres", "x")) {
    paste0("positions named '", coord_cols[1], "'/'", coord_cols[2], "', read as distances")
  } else {
    paste0("the first two numeric columns ('", coord_cols[1], "'/'", coord_cols[2], "'), which carry no ",
           "recognised coordinate name, read as distances")
  }
  if (index_pair) {
    lattice <- array_index_lattice(suppressWarnings(as.numeric(coords_df$x)), suppressWarnings(as.numeric(coords_df$y)))
    if (lattice == "hexagonal" && !isTRUE(opts$allow_array_index_fallback)) {
      stop("The coordinates file ", opts$spatial_coords_csv, " has no pixel columns (imagerow/imagecol, ",
           "pxl_row_in_fullres/pxl_col_in_fullres or x/y), and the pair it does have, '", coord_cols[1], "'/'",
           coord_cols[2], "', is Visium's hexagonal array index: every spot has the same parity of row + col. ",
           "In index space a spot's six neighbours lie at 1.41 and 2.0 where on the slide they are all one pitch ",
           "away, so SpotClean's distance kernel would be distorted by up to sqrt(2). Give pixel coordinates -- ",
           "Space Ranger's tissue_positions.csv as it is, or the converter's x/y coordinates with the ",
           "in_tissue column beside them -- or pass allow_array_index_fallback=True to read the indices as the ",
           "lattice's physical layout (imagerow = ", coord_cols[1], " x sqrt(3), imagecol = ", coord_cols[2], ").")
    }
    if (lattice == "hexagonal") {
      coords_df$x <- as.numeric(coords_df$x) * sqrt(3)
      used_fallback <- TRUE
      coordinate_kind <- paste0(
        "Visium hexagonal array indices, converted to the lattice's physical layout (imagerow = ",
        coord_cols[1], " x sqrt(3), imagecol = ", coord_cols[2], ") because allow_array_index_fallback=True"
      )
    } else if (lattice == "square") {
      coordinate_kind <- paste0(
        "array indices on a square lattice, read as positions (a square lattice's indices are distances ",
        "up to one scale, which the measured spot spacing divides out)"
      )
    } else {
      coordinate_kind <- "named like array indices but not whole numbers, so read as positions"
    }
    coordinate_notes <- c(coordinate_notes, paste0(
      "No pixel coordinates in the coordinates file: '", coord_cols[1], "'/'", coord_cols[2], "' are ",
      coordinate_kind, "."
    ))
    log_msg(coordinate_notes[length(coordinate_notes)])
  }

  n_genes_before <- nrow(counts_mat)
  n_spots_before <- ncol(counts_mat)
  total_counts_before <- sum(counts_mat)

  log_msg("Input: ", n_genes_before, " genes x ", n_spots_before, " spots")

  # --- Filter genes and spots ---
  gene_sums <- rowSums(counts_mat)
  spot_sums <- colSums(counts_mat)
  keep_genes <- gene_sums >= opts$gene_cutoff
  keep_spots <- spot_sums >= opts$spot_cutoff

  counts_mat <- counts_mat[keep_genes, keep_spots, drop = FALSE]
  coords_df <- coords_df[colnames(counts_mat), , drop = FALSE]

  # The total the cutoffs left behind. Without it there is only one subtraction available at the
  # end -- total_counts_before minus the post-SpotClean total -- and it spans both this filter and
  # decontamination, so neither can be reported on its own.
  total_counts_kept <- sum(counts_mat)

  # tissue_col was taken over every common spot, before the line above dropped the low-count
  # ones, and it is the only vector the filter does not realign. counts_mat and tissue_col were
  # both indexed by common_spots in the same order, so keep_spots -- the mask that just filtered
  # counts_mat -- is what puts them back in step. Without this the data.frame below is handed a
  # tissue column as long as the unfiltered slide beside a barcode column as long as the
  # filtered one, and R stops with "arguments imply differing number of rows". That is not a
  # rare input: a tissue column is what gives SpotClean any background to model, and background
  # spots are the low-count ones the cutoff removes.
  n_background_removed_by_cutoff <- 0L
  if (!is.null(tissue_col)) {
    n_background_removed_by_cutoff <- length(which(tissue_col == 0L & !keep_spots))
    tissue_col <- tissue_col[keep_spots]
  }

  n_genes_filtered <- sum(!keep_genes)
  n_spots_filtered <- sum(!keep_spots)

  log_msg("After filtering: ", nrow(counts_mat), " genes x ", ncol(counts_mat),
          " spots (removed ", n_genes_filtered, " genes, ", n_spots_filtered, " spots)")

  # --- Build slide_info for SpotClean ---
  log_msg("Creating SpotClean slide object via createSlide()...")

  # SpotClean expects slide_info with columns: barcode, tissue, imagerow, imagecol
  # It also uses row/col internally for spot distance estimation.
  # If the coords CSV contained a 'tissue' column use it; otherwise mark all as
  # in-tissue (tissue=1).
  if (!is.null(tissue_col)) {
    tissue_vec <- as.integer(tissue_col)
    # as_tissue_flag() accepted the column because every value it has is 0/1; a spot with no value
    # at all is still unknown, and counting it as either side would move it between the ambient
    # profile and the output. Refuse, naming the spot, as the coordinate check below does.
    no_flag <- which(is.na(tissue_vec))
    if (length(no_flag) > 0L) {
      stop(length(no_flag), " of the ", length(tissue_vec), " spots have no value in the '", flag_name,
           "' column of ", opts$spatial_coords_csv, " (first: '", colnames(counts_mat)[no_flag[1]],
           "'), so SpotClean cannot tell whether they are tissue or background. Fill them in with 0 or 1.")
    }
    log_msg("Using column '", flag_name, "' from the coordinates file as the tissue/background flag")
  } else {
    tissue_vec <- rep(1L, ncol(counts_mat))
    log_msg("No tissue/background flag found (looked for 'tissue' and 'in_tissue'); ",
            "assuming all spots are in-tissue")
  }

  image_row <- as.numeric(coords_df[, 1])
  image_col <- as.numeric(coords_df[, 2])

  # A blank cell, an "NA" string or any other non-numeric text becomes NA the moment as.numeric()
  # runs above, and median_nn_distance() measures every distance against the whole coordinate
  # vector -- so one NA anywhere puts an NA in every row of the distance matrix, every
  # nearest-neighbour distance is NA, the finite filter empties, and the function returns NA_real_.
  # Measured live on a 60-spot 6x10 lattice with a single imagecol blanked: the run died on the
  # spacing message below, which tells the user there are "fewer than two spots, or every spot at
  # the same position" about a slide with 60 spots at 60 distinct positions. Both halves of that
  # sentence are false there, and it names no row, so there is nothing in it to act on.
  #
  # Catching it here makes that older message true again: past this point every coordinate is a
  # finite number, so an NA spacing really does mean too few spots or coincident ones. Refuse
  # rather than drop the spots -- the counts matrix is already aligned to coords_df, and silently
  # analysing a subset would publish a contamination estimate for a slide the user did not supply.
  bad_coord <- which(!is.finite(image_row) | !is.finite(image_col))
  if (length(bad_coord) > 0L) {
    first <- bad_coord[1]
    stop(length(bad_coord), " of the ", length(image_row), " spots in ", opts$spatial_coords_csv,
         " have no usable coordinate. A blank, missing or non-numeric cell reads as NA, and one NA ",
         "makes every nearest-neighbour distance NA, so the spot spacing SpotClean sizes its ",
         "contamination kernel from cannot be measured at all. First affected spot: '",
         rownames(coords_df)[first], "' with ", coord_cols[1], "='", coords_df[first, 1], "', ",
         coord_cols[2], "='", coords_df[first, 2], "'. Drop those spots or fill in their coordinates.")
  }

  # createSlide() wants only barcode/tissue/imagerow/imagecol, but .SpotClean() reads slide$col to
  # size its contamination kernel (see median_nn_distance above). Numbering the spots in CSV order
  # made that bandwidth a regression on the row order instead: 0.0954 on the reference slide and
  # 0.000616 after a reshuffle, against a true 3.5113. At that scale every off-diagonal Gaussian
  # weight underflows to zero while the self weight stays 1, so the kernel becomes the identity
  # and the spatial half of the method stops contributing anything.
  nn_dist <- median_nn_distance(image_row, image_col)
  if (!is.finite(nn_dist)) {
    stop("Could not measure the spacing between spots: the ", ncol(counts_mat),
         " coordinate(s) in ", opts$spatial_coords_csv, " give no positive nearest-neighbour ",
         "distance (fewer than two spots, or every spot at the same position). SpotClean sizes ",
         "its contamination kernel from that spacing.")
  }
  array_step <- nn_dist / 2
  array_row <- round((image_row - min(image_row)) / array_step)
  array_col <- round((image_col - min(image_col)) / array_step)
  log_msg("Median spot spacing: ", signif(nn_dist, 6), " (array index step ",
          signif(array_step, 6), ")")

  # .SpotClean() picks its kernel bandwidth by regression, and picks the regression with
  # `if (abs(cor(slide$imagecol, slide$col)) > 0.99)`. cor() is NA the moment either argument has
  # no variance, and `if (abs(NA) > 0.99)` is an error, so the run dies inside the library with
  # "missing value where TRUE/FALSE needed" -- naming neither the file nor the column. slide$col is
  # the index built just above, so it holds one value for every slide whose spread along the second
  # coordinate is under half the measured spacing. Measured live on 60 spots at a true spacing of
  # 100: a second coordinate holding a single value, alternating over 3 px, and ramping over 23.6 px
  # all died that way. median_nn_distance() measures a 2-D distance and is finite on all three, so
  # nothing above this point sees them. A flat coordinate implies a flat index, so checking the
  # index alone covers both halves of cor()'s NA condition and fires on exactly those inputs.
  #
  # There is no repair to make here, only an honest refusal. With the coordinate flat, cor() is NA
  # whatever index it is handed; and when the coordinate varies below the step, which branch
  # SpotClean takes turns on sub-step noise -- taking the imagecol one on the ramp above scales the
  # kernel by 0.2 against a true 100, which is the collapse these indices were introduced to stop.
  if (length(unique(array_col)) < 2L) {
    stop("The coordinates in ", opts$spatial_coords_csv, " are effectively one-dimensional: '",
         coord_cols[2], "' spans ", signif(diff(range(image_col)), 6), " across ", length(image_col),
         " spots, against a measured spot spacing of ", signif(nn_dist, 6), ". SpotClean sizes its ",
         "contamination kernel by regressing that coordinate on an array index derived from it, and ",
         "on a slide this narrow every spot lands on the same index, so the regression is undefined. ",
         "Supply coordinates that vary along both axes.")
  }

  slide_info <- data.frame(
    barcode  = colnames(counts_mat),
    tissue   = tissue_vec,
    row      = array_row,
    col      = array_col,
    imagerow = image_row,
    imagecol = image_col,
    stringsAsFactors = FALSE
  )

  slide_obj <- createSlide(
    count_mat  = counts_mat,
    slide_info = slide_info,
    gene_cutoff = 0,
    verbose    = opts$verbose
  )

  # --- Check that background spots exist ---
  n_background <- sum(tissue_vec == 0)
  # The other side of the same mask. spotclean() estimates the ambient profile from the background
  # spots and returns the tissue ones only, so n_spots_after is narrower than the input for a reason
  # that has nothing to do with the cutoffs -- measured on a 400-spot fixture at --spot-cutoff 0:
  # 400 in, spots_filtered 0, 144 out, and the missing 256 were exactly the background spots. Both
  # counts are published below so that ledger closes. Counted as "not background" rather than
  # "== 1"; as_tissue_flag() has already reduced the column to 0/1, so the two sum to the kept spots.
  n_tissue_spots <- sum(tissue_vec != 0)
  log_msg("Background spots: ", n_background, " / ", ncol(counts_mat))

  if (n_background == 0) {
    log_msg("WARNING: No background (tissue=0) spots found. ",
            "SpotClean requires background spots to model ambient RNA. ",
            "Returning original counts without decontamination.")

    # Save raw counts as-is
    cleaned_counts <- counts_mat
    cleaned_df <- as.data.frame(cleaned_counts)
    n_genes_after  <- nrow(cleaned_counts)
    n_spots_after  <- ncol(cleaned_counts)
    total_counts_after <- sum(cleaned_counts)

    cleaned_path <- file.path(opts$output_dir, "spotclean_cleaned_counts.csv")
    write_atomically(cleaned_path, function(p) write.csv(cleaned_df, p, quote = TRUE))
    log_msg("Saved original counts to: ", cleaned_path)

    # Why none reached it, in the order the run lost them. The advice to supply a flag column is only
    # true when there was none: on a run that read 'in_tissue' and found background in it, that advice
    # sent the caller back to the file they had already given.
    no_background_cause <- if (!has_tissue) {
      paste0(
        "The coordinates file has no 0/1 'tissue' or 'in_tissue' column, so every spot was taken as tissue. ",
        "Provide one (1 = tissue, 0 = background) -- Space Ranger writes it as 'in_tissue' in ",
        "outs/spatial/tissue_positions.csv -- beside counts that include the background spots."
      )
    } else if (n_background_in_coords == 0L) {
      paste0(
        "The '", flag_name, "' column of ", opts$spatial_coords_csv, " marks all ", n_coord_rows, " of its spots ",
        "as tissue (1), so there is no background to model. SpotClean needs the whole capture area: Space ",
        "Ranger's tissue_positions.csv lists the background spots with in_tissue = 0, and its ",
        "raw_feature_bc_matrix holds their counts."
      )
    } else if (n_background_removed_by_cutoff > 0L) {
      paste0(
        "spot_cutoff=", opts$spot_cutoff, " removed all ", n_background_removed_by_cutoff, " background spots ",
        "the counts carry: each has fewer than ", opts$spot_cutoff, " counts in total, as background spots ",
        "usually do. Lower spot_cutoff (0 keeps every spot) so SpotClean can use them.",
        if (n_background_not_in_counts > 0L) {
          paste0(" A further ", n_background_not_in_counts, " spots the '", flag_name, "' column flags as ",
                 "background are not in the counts at all.")
        } else {
          ""
        }
      )
    } else {
      paste0(
        "The '", flag_name, "' column of ", opts$spatial_coords_csv, " flags ", n_background_in_coords,
        " spots as background (0), but the counts carry none of them: the matrix holds only the tissue ",
        "spots, as Space Ranger's filtered_feature_bc_matrix and an h5ad of the in-tissue spots do. ",
        "SpotClean needs the raw matrix (raw_feature_bc_matrix), which also holds the background spots."
      )
    }
    no_background_warning <- paste0(
      "No background spots (tissue or in_tissue = 0) reached SpotClean, which estimates the ambient RNA ",
      "contamination from them, so the original counts are returned without decontamination. ",
      no_background_cause
    )
    return(list(
      status       = "ok",
      tool         = "spotclean",
      task         = "qc_cleanup",
      warning      = no_background_warning,
      data         = list(
        n_genes_before     = n_genes_before,
        n_spots_before     = n_spots_before,
        n_genes_after      = n_genes_after,
        n_spots_after      = n_spots_after,
        # Zero background by definition on this branch: none reached SpotClean. The three counts
        # below say why -- no flag, a flag that marks no background, background the counts do not
        # carry, or background the spot cutoff removed -- and the warning says the same in words.
        n_tissue_spots     = n_tissue_spots,
        n_background_spots = n_background,
        # Where the background went (null without a flag column): flagged in the coordinates file,
        # flagged but absent from the counts, and present but removed by spot_cutoff.
        n_background_spots_in_coordinates    = n_background_in_coords,
        n_background_spots_not_in_counts     = n_background_not_in_counts,
        n_background_spots_removed_by_cutoff = n_background_removed_by_cutoff
      ),
      output_files = list(
        cleaned_counts_csv = cleaned_path
      ),
      # The list a reader of warnings looks in; the singular 'warning' above is kept for callers
      # that already read it, but on its own it left this branch's main fact out of 'warnings'.
      warnings     = as.list(c(no_background_warning, flag_notes, coordinate_notes)),
      params       = list(
        gene_cutoff  = opts$gene_cutoff,
        spot_cutoff  = opts$spot_cutoff,
        # SpotClean itself did not run on this branch: the counts are returned as read.
        method       = "none: no background spots, so SpotClean did not run and the counts are returned as read",
        used_fallback = used_fallback,
        allow_array_index_fallback = isTRUE(opts$allow_array_index_fallback),
        # Which two columns of the coordinates file this run read as the slide geometry. See the
        # note beside the same field on the decontaminated payload below.
        coordinate_columns = coord_cols,
        coordinate_kind = coordinate_kind,
        # Which column was read as the tissue/background flag (null when none was). A flag-named
        # column set aside because it does not hold 0/1 is listed under warnings.
        tissue_flag_column = flag_name,
        verbose      = opts$verbose
      ),
      summary      = list(
        genes_filtered  = n_genes_filtered,
        spots_filtered  = n_spots_filtered,
        # Measured above, before this branch was taken, and just as worth checking here: see the
        # note beside the same two fields on the decontaminated payload below.
        median_spot_spacing = signif(nn_dist, 6),
        array_index_step    = signif(array_step, 6),
        decontaminated  = FALSE
      ),
      analysis     = paste0(
        "SpotClean could not decontaminate because no background spots ",
        "(tissue=0) reached it. ", no_background_cause, " Original counts for ", n_spots_after,
        " spots x ", n_genes_after, " genes returned unchanged."
      )
    ))
  }

  # --- Run SpotClean ---
  log_msg("Running SpotClean decontamination...")
  slide_clean <- spotclean(slide_obj, verbose = opts$verbose)

  # --- Extract cleaned counts ---
  log_msg("Extracting cleaned counts...")
  cleaned_counts <- as.matrix(assay(slide_clean, "decont"))
  cleaned_df <- as.data.frame(cleaned_counts)

  n_genes_after <- nrow(cleaned_counts)
  n_spots_after <- ncol(cleaned_counts)
  total_counts_after <- sum(cleaned_counts)

  # --- Save outputs ---
  cleaned_path <- file.path(opts$output_dir, "spotclean_cleaned_counts.csv")
  write_atomically(cleaned_path, function(p) write.csv(cleaned_df, p, quote = TRUE))

  rds_path <- file.path(opts$output_dir, "spotclean_result.rds")
  write_atomically(rds_path, function(p) saveRDS(slide_clean, file = p))

  log_msg("Saved cleaned counts to: ", cleaned_path)
  log_msg("Saved SpotClean result RDS to: ", rds_path)

  # --- QC metrics ---
  # SpotClean does not discard counts, it moves them between spots, so the difference between the
  # input and output totals is never a contamination figure. Its EM runs over the genes
  # keepHighGene() selected, and SpotClean:::.SpotClean then rescales the genes it skipped by
  # rowSums(raw)/rowSums(raw_tissue) and rbinds them back, restoring each skipped gene's total
  # exactly. Measured on a 300 x 400 fixture in spotclean_env: sum(decont) equalled the filtered
  # input to the count, and every gene matched to 9e-13.
  #
  # What total_counts_before - total_counts_after actually measures is the gene and spot cutoffs
  # above. Running this worker twice on that fixture, changing only --spot-cutoff, it read 0% at
  # cutoff 0 and 3.28% at cutoff 100 -- and the 3.28% was the 11520 counts the cutoff dropped.
  # Publishing it as "contamination removed" therefore reported ambient RNA that had been removed
  # in a run where nothing was removed and no contamination was measured.
  counts_dropped_by_filter <- total_counts_before - total_counts_kept
  pct_dropped_by_filter <- if (total_counts_before > 0) {
    round(100 * counts_dropped_by_filter / total_counts_before, 2)
  } else {
    0
  }

  # SpotClean's own answer to "how contaminated was this slide". .calculate_cont_rate returns
  # received_counts/fitted_total_counts per tissue spot -- the share of a spot's observed counts
  # that arrived from its neighbours. NA_real_ rather than NULL when the field is absent, because
  # jsonlite renders NULL as {} and only NA as null.
  contamination_rate <- metadata(slide_clean)$contamination_rate
  mean_contamination_pct <- if (length(contamination_rate) > 0) {
    round(100 * mean(contamination_rate, na.rm = TRUE), 2)
  } else {
    NA_real_
  }

  # Background spots the cutoff removed are not in the ambient profile SpotClean estimated.
  cutoff_notes <- character(0)
  if (n_background_removed_by_cutoff > 0L) {
    cutoff_notes <- paste0(
      "spot_cutoff=", opts$spot_cutoff, " removed ", n_background_removed_by_cutoff, " of the ",
      n_background_removed_by_cutoff + n_background, " background spots the counts carry; SpotClean ",
      "estimated the ambient profile from the other ", n_background, ". Lower spot_cutoff (0 keeps every ",
      "spot) to let it use them all."
    )
  }

  list(
    status       = "ok",
    tool         = "spotclean",
    task         = "qc_cleanup",
    data         = list(
      n_genes_before     = n_genes_before,
      n_spots_before     = n_spots_before,
      n_genes_after      = n_genes_after,
      n_spots_after      = n_spots_after,
      # The term that closes the ledger. n_spots_before - spots_filtered is the number of spots that
      # reached SpotClean, and it splits into these two; n_spots_after is the tissue half. Without
      # them a run reads as 400 in, 0 filtered, 144 out, which invites raising --spot-cutoff to
      # chase a filter that took nothing.
      n_tissue_spots     = n_tissue_spots,
      n_background_spots = n_background,
      n_background_spots_in_coordinates    = n_background_in_coords,
      n_background_spots_not_in_counts     = n_background_not_in_counts,
      n_background_spots_removed_by_cutoff = n_background_removed_by_cutoff
    ),
    output_files = list(
      cleaned_counts_csv = cleaned_path,
      result_rds         = rds_path
    ),
    warnings     = as.list(c(flag_notes, coordinate_notes, cutoff_notes)),
    params       = list(
      gene_cutoff        = opts$gene_cutoff,
      spot_cutoff        = opts$spot_cutoff,
      method             = "SpotClean (spotclean())",
      # TRUE only when Visium's hexagonal array indices stood in for pixel coordinates, which
      # allow_array_index_fallback has to permit; coordinate_kind says what the two columns were.
      used_fallback      = used_fallback,
      allow_array_index_fallback = isTRUE(opts$allow_array_index_fallback),
      coordinate_kind    = coordinate_kind,
      # The column read as the tissue/background flag -- the split SpotClean's ambient profile is
      # estimated from. A flag-named column that did not hold 0/1 is listed under warnings instead.
      tissue_flag_column = flag_name,
      # Which two columns of the coordinates file were read as the slide geometry. The caller
      # supplied the header, but not the rule that picks from it, and the two are not the same
      # question: the same three columns in two orders gave 22.65% and 5.02% contamination and
      # every other field of this payload was unchanged. log_msg goes to stderr and the portal
      # returns stdout, so this is the only place the choice is visible to a caller.
      coordinate_columns = coord_cols,
      verbose            = opts$verbose
    ),
    summary      = list(
      genes_filtered     = n_genes_filtered,
      spots_filtered     = n_spots_filtered,
      total_counts_before = total_counts_before,
      total_counts_kept   = total_counts_kept,
      total_counts_after  = total_counts_after,
      counts_dropped_by_filter = counts_dropped_by_filter,
      pct_dropped_by_filter    = pct_dropped_by_filter,
      mean_contamination_pct   = mean_contamination_pct,
      # The bandwidth this run actually used. .SpotClean takes it from the slope of an image
      # coordinate on slide_info$col, so a spacing measured wrong silently turns the contamination
      # kernel into the identity -- 0.0954 against a true 3.5113 on the slide SpotClean ships --
      # and the run still returns status "ok" with every other field unchanged. log_msg goes to
      # stderr and the portal returns stdout, so without these two the caller has no way to
      # compare what was measured against the pitch their platform is known to have.
      median_spot_spacing      = signif(nn_dist, 6),
      array_index_step         = signif(array_step, 6)
    ),
    analysis     = paste0(
      "SpotClean decontaminated ", n_spots_after, " spots x ", n_genes_after,
      " genes. Of the ", n_spots_before, " spots read in, ", n_spots_filtered,
      " were dropped by the cutoffs; of the ", n_tissue_spots + n_background,
      " that reached SpotClean, ", n_tissue_spots, " were in tissue and ", n_background,
      " were background. SpotClean estimates the ambient profile from the background spots and ",
      "returns the tissue ones only, so the output is narrower than the input by design and not ",
      "by filtering. SpotClean redistributes counts between spots rather than discarding them, so the ",
      "total is conserved (", total_counts_kept, " in, ", total_counts_after, " out); its own ",
      "estimate is that ", mean_contamination_pct, "% of the counts observed at an average ",
      "tissue spot came from elsewhere on the slide. Separately, the gene and spot cutoffs ",
      "applied before SpotClean ran dropped ", counts_dropped_by_filter, " counts (",
      pct_dropped_by_filter, "% of the input)."
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args inside the handler: a bad flag used to abort R with nothing on stdout.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    sink(stderr())
    result <- run_spotclean(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "spotclean",
      task      = "qc_cleanup",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
