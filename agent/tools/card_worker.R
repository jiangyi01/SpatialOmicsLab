#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(CARD)
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
  message(sprintf("[card-worker] %s", msg))
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
# into the first line. read.csv defaults to header = TRUE, so on the older file the first spot's
# own values became the column names and that spot was dropped. Measured on the Glioblastoma
# slide's tissue_positions_list.csv: the columns came back named CGAGGATATTCAGAGC-1, 0, 0, 0,
# 43551, 15100; numbers match nothing in the resolver's ladder, so its fall-through took "0" and
# "0" -- the SAME column, the tissue flag -- and every in-tissue spot sat at (1, 1). CARD then
# normalises the coordinates by max(x range, y range) = 0, and the run still returned status "ok".
#
# Decide from the first line itself: if the first field is a barcode (not empty) and every field
# after it parses as a number, it is data and not a header. Header fields are text, so a file that
# does have one reads exactly as it read before. The first field must be non-empty because pandas
# writes an unnamed index as an empty header cell (",0,1"), and that file has always read
# correctly. On the six-column layout the recovered names are the ones SpotClean's read10xSlide()
# imposes on this same file, and the resolver's imagerow/imagecol rung then picks the pixel
# coordinates. Any other width keeps the spot and leaves the columns to the resolver's fall-through.
#
# Verbatim copy of tools/spark_worker.R's reader (the by-name family keeps the barcode as column 1
# of the frame, not row names, and reads every column as character so a long numeric identifier is
# not mangled): each worker runs as its own Rscript in its own conda env, so there is no shared
# library on the path.
read_coords_csv <- function(path) {
  first <- read.csv(path, header = FALSE, nrows = 1, check.names = FALSE, stringsAsFactors = FALSE)
  # An empty first cell comes back NA, and nzchar(NA) is TRUE, so NA is tested for explicitly.
  lead <- first[[1]][1]
  headerless <- ncol(first) > 1 && !is.na(lead) && nzchar(trimws(as.character(lead))) &&
    all(vapply(first[-1], is.numeric, logical(1)))
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
  df
}

# resolve_coord_cols is a byte-for-byte copy shared by fourteen R workers (a test keeps them in
# step), so the check that its answer is usable lives here, beside the one caller. An answer that
# names a column the file carries more than once is not usable: read.csv keeps duplicate names,
# and [[ returns the first match every time, so two columns both called "0" -- which is what the
# headerless file used to produce -- gave x == y on every spot. Copied from tools/spark_worker.R.
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
# so a run that used the whole input reads exactly as it did before. Copied from
# tools/spark_worker.R.
reduction_note <- function(noun, n_supplied, n_used, reason = "") {
  if (n_supplied <= 0 || n_used >= n_supplied) return("")
  pct <- 100 * n_used / n_supplied
  because <- if (nzchar(reason)) paste0(" by ", reason) else ""
  paste0(" NOTE: of the ", n_supplied, " ", noun, " supplied, ", n_used, " (", sprintf("%.1f", pct),
         "%) were analysed; ", n_supplied - n_used, " were dropped before the method ran", because,
         ". The results below describe the ", n_used, " analysed ", noun, ", not the full input.")
}

# A label that names nothing -- the same set tools/worker_utils.drop_unlabeled treats as missing.
# pandas writes a NaN label as an empty field, and read.csv keeps an empty character field as "".
LABEL_MISSING <- c("", "na", "nan", "none")

is_unlabeled <- function(labels) {
  is.na(labels) | tolower(trimws(labels)) %in% LABEL_MISSING
}

# When the annotation file has more than one column, the label column is chosen by its name, in
# tiers: a cell-type name first (celltype, cell_type, cell.type), then annotation/annot, then label,
# then cluster; within a tier the first in file order wins, as it always did. It used to be the first
# matching column in FILE order across all of them, so a metadata.csv that carries "cluster" before
# "cell_type" was deconvolved into cluster numbers. Every candidate is returned so the caller can say
# which others it passed over.
CELLTYPE_COLUMN_TIERS <- list(c("celltype", "cell_type", "cell.type"), c("annotation", "annot"),
                              "label", "cluster")

pick_celltype_column <- function(column_names, source_path) {
  lc <- tolower(column_names)
  candidates <- column_names[lc %in% unlist(CELLTYPE_COLUMN_TIERS)]
  for (tier in CELLTYPE_COLUMN_TIERS) {
    hit <- which(lc %in% tier)
    if (length(hit) > 0) {
      return(list(index = hit[[1L]], candidates = candidates))
    }
  }
  stop(sprintf(paste0(
    "ref_celltypes_csv (%s) has %d columns and none is named as a cell type label (%s, any case); ",
    "columns were: %s. Write a file whose one column besides the cell barcodes is the cell type."),
    source_path, length(column_names), paste(unlist(CELLTYPE_COLUMN_TIERS), collapse = ", "),
    paste(column_names, collapse = ", ")))
}

# Space Ranger's in_tissue flag, read by the rule tools/worker_utils.keep_in_tissue applies to
# obs['in_tissue'] in the Python workers: 1 / "1" / TRUE is tissue; 0, FALSE, empty and anything
# else is background glass. Returned named by `ids` (the file's spot IDs, in its row order); NULL
# when the file has no in_tissue column. Background spots are left out by default and counted --
# the rule scanpy_spatial, bsp and spagft already follow -- because a counts table converted from a
# whole-array export carries them, and analysing glass as tissue is the silent alternative.
# Copied from tools/spacexr_worker.R.
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
# as tissue is refused rather than analysed as an empty slide. Copied from tools/spacexr_worker.R.
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

# read_coords_csv names the six columns of a headerless file (Space Ranger 1's
# tissue_positions_list.csv) as SpotClean's read10xSlide() does: barcode, tissue, row, col, imagerow,
# imagecol. That "tissue" column IS Space Ranger's in_tissue flag. A column called "tissue" in any
# other file is not -- a CELLxGENE metadata.csv carries tissue = "heart" -- so the name is mapped to
# in_tissue only for exactly that layout.
SPACE_RANGER_V1_COLUMNS <- c("barcode", "tissue", "row", "col", "imagerow", "imagecol")

tissue_flag_frame <- function(frame) {
  if (identical(colnames(frame), SPACE_RANGER_V1_COLUMNS)) colnames(frame)[2] <- "in_tissue"
  frame
}

# What a count matrix holds, in the classes tools/worker_utils.expression_matrix_kind uses: "counts"
# (finite, non-negative, integer-valued), "nonnegative_noninteger" (normalised or log data),
# "negative" (scaled / z-scored data), "nonfinite" (NA, NaN or Inf -- an empty CSV field reads as NA),
# "nonnumeric" (a text column), or "empty". CARD models raw counts and normalises them itself: a
# negative or missing value used to reach createCARDObject, whose filters and reference basis fail on
# it without naming the file, and normalised data ran as counts without a word. Read a block of
# columns at a time, so no full-size temporary is built beside the largest object in the process.
count_matrix_kind <- function(m, block_cells = 2^22) {
  if (!is.numeric(m)) return("nonnumeric")
  if (length(m) == 0L) return("empty")
  n_col <- ncol(m)
  step <- max(1L, as.integer(floor(block_cells / max(1L, nrow(m)))))
  kind <- "counts"
  for (start in seq.int(1L, n_col, by = step)) {
    v <- as.vector(m[, start:min(n_col, start + step - 1L), drop = FALSE])
    if (anyNA(v) || any(is.infinite(v))) return("nonfinite")
    if (any(v < 0)) return("negative")
    if (identical(kind, "counts") && any(abs(v - round(v)) > 1e-3)) kind <- "nonnegative_noninteger"
  }
  kind
}

# Stop on a matrix CARD cannot model; return a warning (or "") for normalised data, which runs as before.
check_count_matrix <- function(kind, knob, path) {
  advice <- paste0(" CARD models raw counts and normalises them itself; write the raw counts (for an h5ad ",
                   "whose X is processed, the counts in adata.raw) to ", knob, ".")
  if (kind %in% c("nonnumeric", "nonfinite", "negative")) {
    what <- switch(kind,
                   nonnumeric = "non-numeric values (a text column besides the row names?)",
                   nonfinite = "missing (NA/empty), NaN or infinite values",
                   negative = "negative values (scaled or z-scored data)")
    stop(knob, " (", path, ") holds ", what, ", not counts.", advice)
  }
  if (identical(kind, "nonnegative_noninteger")) {
    return(paste0(knob, " holds non-integer values (normalised or log-transformed data?), and CARD normalises ",
                  "its input as counts, so the result was computed on data normalised twice.", advice))
  }
  ""
}

# Free memory by the rule of tools/worker_utils.py available_memory_bytes(), which an R worker cannot
# import: the smaller of the host's MemAvailable and the room under the cgroup memory limit (v2 first,
# then v1). memory.current / memory.usage_in_bytes count the page cache, and a memory-limited container
# sits at its limit on cache alone after reading its inputs; the kernel reclaims both file LRU lists
# (active_file, inactive_file in memory.stat) before it OOM-kills anything, so the working set is usage
# minus those and the room is the limit minus the working set. When the file LRU counters cannot be
# read the cache cannot be told apart, so usage is not subtracted at all and the room is the limit.
# NA when nothing can be read. Copied from tools/spacexr_worker.R (itself from celltrek_worker.R).
# This worker used to take min(MemAvailable, cgroup LIMIT) -- blind to what the cgroup already holds,
# CARD's own dense count matrices included -- so check_kernel_memory passed kernels the container
# could not hold, and CARD_deconvolution was OOM-killed with no JSON.
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

# CARD_deconvolution builds its spatial kernel as a dense spots x spots matrix of doubles: rdist()
# of the normalised coordinates, then exp(-ED^2 / (2 * 0.1^2)), and CARDref copies it again into
# Armadillo. That is intrinsic to the method -- the conditional autoregressive prior couples every
# pair of spots -- so it is checked against what is available before the fit starts, with the lower
# bound only (two copies), so a run this refuses could not have finished. The alternative was the
# kernel's OOM killer and no JSON at all.
check_kernel_memory <- function(n_spots, avail = available_memory_bytes()) {
  one <- 8 * as.numeric(n_spots)^2
  need <- 2 * one
  if (!is.na(avail) && need > avail) {
    stop(sprintf(paste0(
      "CARD's spatial kernel is a dense %.0f x %.0f matrix of doubles (%.2f GB), and CARD_deconvolution ",
      "holds at least two copies of it: at least %.2f GB, while %.2f GB is available here ",
      "(MemAvailable / room under the cgroup limit). The kernel is intrinsic to CARD's conditional autoregressive ",
      "model and no parameter of this tool makes it smaller; run it where at least %.2f GB is free."),
      n_spots, n_spots, one / 1e9, need / 1e9, avail / 1e9, need / 1e9))
  }
  need
}

# New files appear whole or not at all: written beside the target, then renamed over it. The
# output names themselves are unchanged.
write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = FALSE, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place as ", path)
  }
  invisible(path)
}

save_rds_atomic <- function(object, path) {
  partial <- paste0(path, ".partial")
  saveRDS(object, file = partial)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place as ", path)
  }
  invisible(path)
}

# CARD's own spatial QC, as createCARDObject (CARD 1.1) applies it, in this order: a GENE is kept
# when it is non-zero in MORE THAN minCountSpot spots; then a SPOT is kept when its total count over
# the kept genes is at least minCountGene and at most 1e6. The names read backwards -- minCountGene
# filters spots and minCountSpot filters genes -- and the portal's docs used to describe them the
# other way round. Computed here only to say WHICH spots go and why, and to stop with the knob's
# name before CARD would die on an empty (or one-row) matrix; the counts the payload reports are
# read back from the CARD object afterwards.
CARD_MAX_SPOT_TOTAL <- 1e6

preview_card_spatial_qc <- function(counts, min_count_gene, min_count_spot) {
  genes_kept <- Matrix::rowSums(counts > 0) > min_count_spot
  totals <- Matrix::colSums(counts[genes_kept, , drop = FALSE])
  list(
    n_genes_kept = sum(genes_kept),
    below = colnames(counts)[totals < min_count_gene],
    above = colnames(counts)[totals > CARD_MAX_SPOT_TOTAL],
    n_spots_kept = sum(totals >= min_count_gene & totals <= CARD_MAX_SPOT_TOTAL)
  )
}

first_ids <- function(ids, n = 5L) {
  paste0(paste(utils::head(ids, n), collapse = ", "), if (length(ids) > n) ", ..." else "")
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
    cat("Usage: card_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --ref-counts-csv PATH      scRNA-seq reference counts CSV (genes x cells) [required]\n")
    cat("  --ref-celltypes-csv PATH   Reference cell type annotation CSV [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --min-count-gene INT       CARD's minCountGene, a SPOT filter despite its name: a spot whose\n")
    cat("                             total count (over the genes min-count-spot keeps) is below this\n")
    cat("                             is dropped, as is any spot above 1e6 (default: 100)\n")
    cat("  --min-count-spot INT       CARD's minCountSpot, a GENE filter despite its name: a gene non-zero\n")
    cat("                             in this many spots or fewer is dropped (default: 5)\n")
    cat("  --drop-unlabeled BOOL      Leave out reference cells with no label (NA, empty, nan, none)\n")
    cat("                             instead of stopping (default: false)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    min_count_gene     = 100L,
    min_count_spot     = 5L,
    drop_unlabeled     = FALSE
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
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--min-count-gene") {
      opts$min_count_gene <- as.integer(val)
    } else if (key == "--min-count-spot") {
      opts$min_count_spot <- as.integer(val)
    } else if (key == "--drop-unlabeled") {
      flag <- tolower(trimws(val))
      if (!(flag %in% c("true", "false", "1", "0", "yes", "no"))) {
        stop(sprintf("--drop-unlabeled takes true or false, not '%s'", val))
      }
      opts$drop_unlabeled <- flag %in% c("true", "1", "yes")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_card <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$ref_counts_csv) || is.null(opts$ref_celltypes_csv) ||
      is.null(opts$output_dir)) {
    stop("CARD requires --spatial-counts-csv, --spatial-coords-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv,
              opts$ref_counts_csv, opts$ref_celltypes_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  for (knob in c("min_count_gene", "min_count_spot")) {
    v <- opts[[knob]]
    if (length(v) != 1L || is.na(v) || v < 0) {
      stop(knob, " must be a non-negative integer; got ", paste(format(v), collapse = ", "), ".")
    }
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  warnings <- character(0)

  # --- Load reference data ---
  log_msg("Reading reference counts from: ", opts$ref_counts_csv)
  ref_counts_df <- read.csv(opts$ref_counts_csv, row.names = 1, check.names = FALSE)
  ref_counts_mat <- as.matrix(ref_counts_df)

  log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE,
                        stringsAsFactors = FALSE)
  # One column is the label. With more (STCoscientist may pass the converter's full metadata.csv),
  # the label column is chosen by name, most specific first, and the choice is reported.
  if (ncol(ref_ct_df) < 1L) {
    stop("ref_celltypes_csv (", opts$ref_celltypes_csv, ") has no column besides the cell barcodes; ",
         "it needs one holding the cell type label.")
  }
  if (ncol(ref_ct_df) > 1L) {
    picked <- pick_celltype_column(colnames(ref_ct_df), opts$ref_celltypes_csv)
    sel <- picked$index
    log_msg("metadata CSV has ", ncol(ref_ct_df),
            " columns; auto-selected '", colnames(ref_ct_df)[sel], "' as celltype")
    passed_over <- setdiff(picked$candidates, colnames(ref_ct_df)[sel])
    if (length(passed_over) > 0) {
      warnings <- c(warnings, paste0(
        "ref_celltypes_csv has ", ncol(ref_ct_df), " columns; the cell type labels were read from '",
        colnames(ref_ct_df)[sel], "' (params.ref_celltype_column), not from ",
        paste(sprintf("'%s'", passed_over), collapse = ", "),
        ". Pass a one-column file to choose another."))
    }
  } else {
    sel <- 1L
  }
  ref_celltype_column <- colnames(ref_ct_df)[sel]
  cell_types <- as.character(ref_ct_df[, sel])
  names(cell_types) <- rownames(ref_ct_df)

  # Ensure genes x cells orientation
  common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  if (length(common_cells) == 0) {
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(cell_types)))}

  # A reference cell with no row in the annotation file has no label to deconvolve into; it is left
  # out, as it always was, and now the count is said (data.n_ref_cells_in_counts and a warning)
  # rather than data.n_ref_cells silently describing only the cells that matched.
  n_ref_cells_in_counts <- ncol(ref_counts_mat)
  if (length(common_cells) < n_ref_cells_in_counts) {
    warnings <- c(warnings, paste0(
      n_ref_cells_in_counts - length(common_cells), " of the ", n_ref_cells_in_counts,
      " cells in the reference counts CSV have no row in ref_celltypes_csv and were left out of the ",
      "reference (first: ", first_ids(setdiff(colnames(ref_counts_mat), common_cells)), "); ",
      length(common_cells), " cells were matched."))
  }

  ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]
  cell_types <- cell_types[common_cells]
  n_ref_cells <- length(common_cells)

  # A missing label is not a class. CARD makes every distinct label a column of the proportions: a
  # "nan" or "None" label became a cell type called "nan", an empty label (how pandas writes NaN)
  # crashed CARD's reference build with "subscript out of bounds", and an NA label was dropped by
  # CARD's sc_QC while data.n_ref_cells still counted it. drop_unlabeled leaves them out and says
  # how many; without it the run stops and names the knob (tools/worker_utils.drop_unlabeled).
  unlabeled <- is_unlabeled(cell_types)
  n_ref_cells_unlabeled_dropped <- 0L
  if (any(unlabeled)) {
    if (!isTRUE(opts$drop_unlabeled)) {
      stop(sprintf(paste0(
        "%d of %d reference cells have no label in column '%s' of %s (NA/empty/nan/none; first: %s). ",
        "Pass drop_unlabeled=True to leave them out, or label them first; a missing label is not a class."),
        sum(unlabeled), length(cell_types), ref_celltype_column, opts$ref_celltypes_csv,
        first_ids(names(cell_types)[unlabeled])))
    }
    n_ref_cells_unlabeled_dropped <- sum(unlabeled)
    log_msg("drop_unlabeled=true: leaving out ", n_ref_cells_unlabeled_dropped, " of ",
            length(cell_types), " reference cells with no label")
    common_cells <- common_cells[!unlabeled]
    ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]
    cell_types <- cell_types[common_cells]
    warnings <- c(warnings, paste0(
      "drop_unlabeled=true: ", n_ref_cells_unlabeled_dropped, " of ", n_ref_cells, " reference cells ",
      "had no label in column '", ref_celltype_column, "' and were left out of the reference."))
  }
  if (length(common_cells) == 0) {
    stop("No reference cell is left once the unlabelled ones are dropped (drop_unlabeled=true).")
  }
  if (length(unique(cell_types)) < 2L) {
    stop("CARD needs at least two reference cell types to deconvolve into; column '", ref_celltype_column,
         "' of ", opts$ref_celltypes_csv, " has ", length(unique(cell_types)), ": ",
         paste(unique(cell_types), collapse = ", "), ".")
  }

  log_msg("Reference: ", length(common_cells), " cells, ",
          length(unique(cell_types)), " cell types (column '", ref_celltype_column, "')")

  # Checked on the cells that enter CARD: a value CARD cannot model stops the run by name, and
  # normalised data runs as before with a warning (params.ref_counts_kind).
  ref_counts_kind <- count_matrix_kind(ref_counts_mat)
  ref_kind_warning <- check_count_matrix(ref_counts_kind, "ref_counts_csv", opts$ref_counts_csv)
  if (nzchar(ref_kind_warning)) {
    warnings <- c(warnings, ref_kind_warning)
  }

  # Build reference meta data frame for CARD
  ref_meta <- data.frame(
    cellID   = common_cells,
    cellType = cell_types,
    row.names = common_cells,
    stringsAsFactors = FALSE
  )

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  sp_counts_mat <- as.matrix(sp_counts_df)
  rm(sp_counts_df)

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Every column is read as character, so R cannot mangle a long numeric spot ID; a file with no
  # header line (Space Ranger's tissue_positions_list.csv) is recognised by read_coords_csv.
  sp_coords_raw <- read_coords_csv(opts$spatial_coords_csv)
  coord_cols <- resolve_coord_cols(colnames(sp_coords_raw)[-1], opts$spatial_coords_csv)
  check_coord_cols(coord_cols, colnames(sp_coords_raw), opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(sp_coords_raw), collapse = ", "), ")")
  sp_coords_df <- data.frame(
    x = suppressWarnings(as.numeric(sp_coords_raw[[coord_cols[1]]])),
    y = suppressWarnings(as.numeric(sp_coords_raw[[coord_cols[2]]])),
    row.names = sp_coords_raw[, 1]
  )

  # Ensure genes x spots orientation
  counts_orientation <- "genes_x_spots"
  common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  if (length(common_spots) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
    counts_orientation <- "spots_x_genes (transposed to genes x spots)"
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(sp_counts_mat), "coordinates", rownames(sp_coords_df)))}
  if (counts_orientation != "genes_x_spots") {
    log_msg("Spatial counts CSV is spots x genes (no column name matched a coordinate row); transposed")
  }

  # A counts spot with no coordinate row cannot enter CARD's spatial prior; it is left out, and the
  # count is said in warnings AND in the analysis.
  n_spots_in_counts <- ncol(sp_counts_mat)
  match_note <- reduction_note(
    "spots", n_spots_in_counts, length(common_spots),
    "matching the counts CSV to the coordinates file (a spot with no coordinate row has no place in CARD's spatial prior)"
  )
  if (nzchar(match_note)) {
    warnings <- c(warnings, paste0(
      n_spots_in_counts - length(common_spots), " of the ", n_spots_in_counts,
      " spots in the spatial counts CSV have no row in the coordinates file and were left out (first: ",
      first_ids(setdiff(colnames(sp_counts_mat), common_spots)), "); ", length(common_spots),
      " spots were matched."))
  }

  # Background spots are not tissue. A coordinates file that carries Space Ranger's in_tissue flag
  # (tissue_positions.csv, the converter's metadata.csv -- which its own warning says to pass for
  # exactly this -- or the headerless tissue_positions_list.csv, whose flag column read_coords_csv
  # names "tissue") says which spots are glass. They used to be deconvolved as tissue: on a CELLxGENE
  # export, ~3000 of 4992 spots, most of them above min_count_gene. For CARD it matters beyond those
  # rows: its conditional autoregressive prior couples every pair of spots, so the background shaped
  # the in-tissue estimates too, and grew the dense kernel. They are left out and counted, as
  # tools/worker_utils.keep_in_tissue does for the Python workers.
  n_spots_matched <- length(common_spots)
  tissue <- keep_in_tissue_spots(common_spots, in_tissue_flags(tissue_flag_frame(sp_coords_raw), sp_coords_raw[, 1]),
                                 opts$spatial_coords_csv)
  common_spots <- tissue$spots
  n_spots_off_tissue_dropped <- tissue$n_dropped
  if (!is.null(tissue$warning)) {
    warnings <- c(warnings, tissue$warning)
    log_msg("WARNING: ", tissue$warning)
  }
  tissue_note <- reduction_note(
    "spots", n_spots_matched, length(common_spots),
    "leaving out the in_tissue == 0 background spots the coordinates file names (they have no row in card_proportions.csv)"
  )

  sp_counts_mat <- sp_counts_mat[, common_spots, drop = FALSE]
  sp_coords_df <- sp_coords_df[common_spots, 1:2, drop = FALSE]
  colnames(sp_coords_df) <- c("x", "y")

  sp_counts_kind <- count_matrix_kind(sp_counts_mat)
  sp_kind_warning <- check_count_matrix(sp_counts_kind, "spatial_counts_csv", opts$spatial_counts_csv)
  if (nzchar(sp_kind_warning)) {
    warnings <- c(warnings, sp_kind_warning)
  }

  # Checked on the matched spots only: a positions file carries rows for every array spot, and a
  # blank on a spot the counts do not have has never mattered.
  for (axis in c("x", "y")) {
    bad <- !is.finite(sp_coords_df[[axis]])
    if (any(bad)) {
      stop("Coordinate column '", coord_cols[match(axis, c("x", "y"))], "' of ", opts$spatial_coords_csv,
           " has ", sum(bad), " value(s) that are not numbers on the ", nrow(sp_coords_df),
           " spots matched to the counts (first: ", first_ids(rownames(sp_coords_df)[bad], 3L),
           "). A coordinate must be numeric; the file carries the columns: ",
           paste(colnames(sp_coords_raw), collapse = ", "))
    }
  }
  # CARD scales both axes by max(x range, y range); when that is 0 every normalised coordinate is
  # 0/0 = NaN, and the spatial kernel is computed on nothing.
  if (nrow(sp_coords_df) > 1 &&
      max(diff(range(sp_coords_df$x)), diff(range(sp_coords_df$y))) == 0) {
    stop("Every one of the ", nrow(sp_coords_df), " matched spots sits at the same position (",
         sp_coords_df$x[1], ", ", sp_coords_df$y[1], ") in columns '", coord_cols[1], "' and '",
         coord_cols[2], "' of ", opts$spatial_coords_csv, ". CARD normalises the coordinates by ",
         "their range, which is 0, so its spatial kernel would be NaN. Check that these two columns ",
         "are the spot positions (the file carries: ", paste(colnames(sp_coords_raw), collapse = ", "), ").")
  }

  n_genes <- nrow(sp_counts_mat)
  n_spots <- ncol(sp_counts_mat)

  log_msg("Spatial: ", n_genes, " genes x ", n_spots, " spots (counts ", counts_orientation, ")")

  # createCARDObject turns a dense matrix into this same sparse one (as(as.matrix(x),
  # "sparseMatrix")); doing it here once lets the QC preview below run without a dense logical copy.
  sp_counts_mat <- as(sp_counts_mat, "sparseMatrix")

  qc <- preview_card_spatial_qc(sp_counts_mat, opts$min_count_gene, opts$min_count_spot)
  if (qc$n_genes_kept < 2) {
    stop("CARD's gene filter (min_count_spot=", opts$min_count_spot, ": a gene is kept only when it is ",
         "non-zero in more than ", opts$min_count_spot, " spots) keeps ", qc$n_genes_kept, " of the ",
         n_genes, " spatial genes, and CARD needs at least two. Lower min_count_spot.")
  }
  if (qc$n_spots_kept < 2) {
    stop("CARD's spot filter (min_count_gene=", opts$min_count_gene, ": a spot is kept only when its ",
         "total count over the ", qc$n_genes_kept, " genes min_count_spot keeps is at least ",
         opts$min_count_gene, ", and at most 1e6) keeps ", qc$n_spots_kept, " of the ", n_spots,
         " spots, and CARD needs at least two. Lower min_count_gene (",
         length(qc$below), " spots fall below it).")
  }

  # --- Create CARD object ---
  log_msg("Creating CARD object...")
  card_obj <- createCARDObject(
    sc_count   = ref_counts_mat,
    sc_meta    = ref_meta,
    spatial_count = sp_counts_mat,
    spatial_location = sp_coords_df,
    ct.varname = "cellType",
    ct.select  = unique(cell_types),
    sample.varname = NULL,
    minCountGene = opts$min_count_gene,
    minCountSpot = opts$min_count_spot
  )
  n_genes_after_qc <- nrow(card_obj@spatial_countMat)
  n_spots_after_qc <- ncol(card_obj@spatial_countMat)
  n_ref_cells_used <- ncol(card_obj@sc_eset)
  log_msg("After CARD's QC: ", n_genes_after_qc, " genes (non-zero in more than ", opts$min_count_spot,
          " spots), ", n_spots_after_qc, " spots (total count in [", opts$min_count_gene, ", 1e6]); ",
          n_ref_cells_used, " reference cells")

  check_kernel_memory(n_spots_after_qc)

  # --- Run CARD deconvolution ---
  log_msg("Running CARD deconvolution...")
  card_obj <- CARD_deconvolution(card_obj)

  # --- Extract proportions ---
  log_msg("Extracting cell type proportions...")
  prop_mat <- card_obj@Proportion_CARD
  n_spots_used <- nrow(prop_mat)
  n_informative_genes <- nrow(card_obj@algorithm_matrix$B)
  prop_df <- as.data.frame(prop_mat)
  prop_df$spot <- rownames(prop_df)

  # --- Say which spots CARD left out, and why ---
  # createCARDObject's spot filter, then CARD_deconvolution's own: a spot with no count on any of
  # CARD's informative genes (Xinput[, colSums(Xinput) > 0]). card_proportions.csv has no row for
  # any of them.
  n_below <- length(qc$below)
  n_above <- length(qc$above)
  no_informative <- setdiff(colnames(card_obj@spatial_countMat), rownames(prop_mat))
  n_no_informative <- length(no_informative)
  if (n_below > 0) {
    warnings <- c(warnings, paste0(
      n_below, " of ", n_spots, " spots have a total count below min_count_gene=", opts$min_count_gene,
      " (over the ", n_genes_after_qc, " genes min_count_spot=", opts$min_count_spot, " keeps) and ",
      "were dropped by CARD's QC; card_proportions.csv has no row for them (first: ",
      first_ids(qc$below), "). min_count_gene=0 turns this filter off."))
  }
  if (n_above > 0) {
    warnings <- c(warnings, paste0(
      n_above, " of ", n_spots, " spots have a total count above CARD's fixed ceiling of 1e6 and were ",
      "dropped by CARD's QC; card_proportions.csv has no row for them (first: ", first_ids(qc$above), ")."))
  }
  if (n_no_informative > 0) {
    warnings <- c(warnings, paste0(
      n_no_informative, " of ", n_spots_after_qc, " spots that passed CARD's QC have no count on any of ",
      "the ", n_informative_genes, " informative genes CARD selected from the reference and were ",
      "dropped by CARD_deconvolution; card_proportions.csv has no row for them (first: ",
      first_ids(no_informative), ")."))
  }
  if (n_spots_after_qc != n_spots - n_below - n_above) {
    warnings <- c(warnings, paste0(
      "CARD kept ", n_spots_after_qc, " spots after its QC where its documented rule predicts ",
      n_spots - n_below - n_above, "; the counts in data are read from the CARD object."))
  }
  spot_note <- reduction_note(
    "spots", n_spots, n_spots_used,
    paste0("CARD's own filters (", n_below, " below min_count_gene=", opts$min_count_gene, ", ", n_above,
           " above 1e6, ", n_no_informative, " with no count on its ", n_informative_genes,
           " informative genes)")
  )
  ref_note <- if (n_ref_cells_used < length(common_cells)) {
    paste0(" CARD's reference QC left out ", length(common_cells) - n_ref_cells_used, " of ",
           length(common_cells), " reference cells with no count.")
  } else {
    ""
  }
  if (nzchar(ref_note)) {
    warnings <- c(warnings, trimws(ref_note))
  }

  # --- Save outputs ---
  prop_path <- file.path(opts$output_dir, "card_proportions.csv")
  write_csv_atomic(prop_df, prop_path)

  rds_path <- file.path(opts$output_dir, "card_result.rds")
  save_rds_atomic(card_obj, rds_path)

  log_msg("Saved proportions to: ", prop_path)
  log_msg("Saved CARD object to: ", rds_path)

  # --- Summary ---
  cell_types_found <- colnames(prop_mat)
  n_celltypes <- length(cell_types_found)

  # Determine dominant cell type per spot
  dominant <- apply(prop_mat, 1, function(row) colnames(prop_mat)[which.max(row)])
  dominant_counts <- as.list(table(dominant))

  params <- list(
    min_count_gene   = opts$min_count_gene,
    min_count_spot   = opts$min_count_spot,
    drop_unlabeled   = isTRUE(opts$drop_unlabeled),
    method           = "CARD (createCARDObject -> CARD_deconvolution)",
    used_fallback    = FALSE,
    ref_celltype_column = ref_celltype_column,
    counts_orientation = counts_orientation,
    spatial_counts_kind = sp_counts_kind,
    ref_counts_kind  = ref_counts_kind
  )
  # The shape tools/worker_utils.record_in_tissue writes, and only when a spot was left out.
  if (!is.null(tissue$filter)) {
    params$in_tissue_filter <- tissue$filter
  }

  list(
    status       = "ok",
    tool         = "card",
    task         = "deconvolution",
    data         = list(
      n_spots          = n_spots,
      n_spots_used     = n_spots_used,
      n_spots_in_counts = n_spots_in_counts,
      n_spots_off_tissue_dropped = n_spots_off_tissue_dropped,
      n_spots_below_min_count_gene = n_below,
      n_spots_above_max_count      = n_above,
      n_spots_without_informative_counts = n_no_informative,
      n_genes_spatial  = n_genes,
      n_genes_after_qc = n_genes_after_qc,
      n_informative_genes = n_informative_genes,
      n_ref_cells      = n_ref_cells,
      n_ref_cells_in_counts = n_ref_cells_in_counts,
      n_ref_cells_unlabeled_dropped = n_ref_cells_unlabeled_dropped,
      n_ref_cells_used = n_ref_cells_used,
      n_ref_types      = length(unique(cell_types)),
      coord_columns    = coord_cols
    ),
    output_files = list(
      proportions_csv  = prop_path,
      result_rds       = rds_path
    ),
    params       = params,
    summary      = list(
      n_cell_types     = n_celltypes,
      cell_types_found = cell_types_found,
      dominant_counts  = dominant_counts
    ),
    warnings     = I(warnings),
    analysis     = paste0(
      "CARD deconvolution mapped ", n_celltypes, " cell types across ",
      n_spots_used, " spatial spots using spatially-informed conditional ",
      "autoregressive modeling. Dominant cell type: ",
      names(which.max(unlist(dominant_counts))),
      " (", max(unlist(dominant_counts)), " spots).",
      match_note, tissue_note, spot_note, ref_note,
      if (n_ref_cells_unlabeled_dropped > 0) paste0(
        " ", n_ref_cells_unlabeled_dropped, " unlabelled reference cells were left out (drop_unlabeled).") else ""
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  res <- tryCatch(with_r_traceback({
    sink(stderr())
    result <- run_card(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "card",
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
