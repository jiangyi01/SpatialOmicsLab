#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(spacexr)
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
  message(sprintf("[spacexr-worker] %s", msg))
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


# first_record_line is a verbatim copy of tools/spotsweeper_worker.R's, and read_coords_by_name decides
# the layout by the rule of that file's read_coords_csv (the comment below is its own): each worker runs as
# its own Rscript in its own conda env, so there is no shared library on the path.
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

# The coordinates file as a frame whose column 1 is the spot IDs and whose every column is character, so R
# never parses a spot ID. spotsweeper's read_coords_csv cannot be used here: it reads the whole file with
# read.csv(row.names = 1) and default column classes, so a numeric spot ID of 16 or more digits becomes a
# double, neighbouring IDs collapse to one string and the read stops on "duplicate 'row.names' are not
# allowed" (spacexr read every coordinates file as character before the headerless layout was handled).
# The layout is decided from the first non-empty line alone, by read_coords_csv's own rule (the comment
# above), with its messages, and the file is read once, in that layout.
read_coords_by_name <- function(path) {
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
    return(read.csv(path, check.names = FALSE, colClasses = "character"))
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
    # read_coords_csv's test on the numeric column read.csv would have made: an entry that is not a
    # number (NA, empty, TRUE) is not 0/1 either.
    flag <- suppressWarnings(as.numeric(frame$in_tissue))
    if (!all(flag %in% c(0, 1))) {
      stop("Coordinates file ", path, " has no header row that names its columns (its first line, ",
           shown, ", is an identifier followed only by numbers, so it reads as a spot) and six ",
           "columns, but its second column holds values other than 0 and 1, so it is not Space ",
           "Ranger's in_tissue flag and the file is not tissue_positions_list.csv. ", rename_hint)
    }
  }
  log_msg("Coordinates file has no header row; ", header)
  frame
}

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

# The three modes run.RCTD accepts. Checked before any file is read: an unknown mode used to be
# rejected by run.RCTD only after the reference and the slide had been loaded and create.RCTD had
# selected its genes.
VALID_MODES <- c("doublet", "full", "multi")

# RCTD's own floor (create.RCTD's CELL_MIN_INSTANCE): process_cell_type_info refuses a reference
# with fewer cells of any one type, so a smaller type has to leave before RCTD sees it.
CELL_MIN_INSTANCE <- 25L

# Read a counts CSV straight into a sparse dgCMatrix, one block of rows at a time.
#
# read.csv -> as.matrix -> as(, "CsparseMatrix") held the whole table as a data.frame and then
# as a dense double matrix, only to re-sparsify it: 73 GB of doubles for the VisiumHD Colon slide
# (507,684 x 18,085), 26 GB for the Xenium tonsil, before RCTD had started. Here only one block
# (~block_bytes of doubles) is ever dense; what accumulates is the non-zero triplets.
#
# Same reading rules as the read.csv(row.names = 1, check.names = FALSE) it replaces: first field
# of every row is the row name, header names are kept verbatim, "NA" is missing, a header with
# one field fewer than the rows names only the data columns. file() opens .gz/.bz2/.xz as well.
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
    stop(sprintf("%s CSV %s holds %.0f missing (NA/empty) values; RCTD needs a count in every cell",
                 what, path, n_missing))
  }
  if (n_negative > 0) {
    stop(sprintf("%s CSV %s holds %.0f negative values; RCTD models counts, which are never negative",
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

# spacexr's get_cell_type_info -- per cell type, the mean over its cells of counts / nUMI -- with
# the same result and without its memory. Upstream computes it with base::sweep, which builds a
# dense genes x cells array per type (measured: ~4x the dense size at peak). That is why Reference()
# caps every type at 10,000 cells by default; with the cap gone (n_max_cells = 0) a 63,900-cell
# type on 18,082 genes would need ~37 GB there. Handed to create.RCTD as cell_type_profiles, which
# it then uses exactly as it would its own.
cell_type_profiles_sparse <- function(counts, cell_types, nUMI) {
  cell_types <- droplevels(cell_types)
  types <- levels(cell_types)
  scaled <- counts %*% Diagonal(x = 1 / as.numeric(nUMI))
  membership <- sparseMatrix(i = seq_along(cell_types), j = as.integer(cell_types), x = 1,
                             dims = c(length(cell_types), length(types)))
  sums <- as.matrix(scaled %*% membership)
  means <- sweep(sums, 2, as.numeric(table(cell_types)[types]), "/")
  dimnames(means) <- list(rownames(counts), types)
  as.data.frame(means, optional = TRUE)
}

# Copied from tools/celltrek_worker.R. Free memory by the rule of tools/worker_utils.py
# available_memory_bytes(), which an R worker cannot
# import: the smaller of the host's MemAvailable and the room under the cgroup memory limit (v2 first,
# then v1). memory.current / memory.usage_in_bytes count the page cache, and a memory-limited container
# sits at its limit on cache alone after reading its inputs; the kernel reclaims both file LRU lists
# (active_file, inactive_file in memory.stat) before it OOM-kills anything, so the working set is usage
# minus those and the room is the limit minus the working set. When the file LRU counters cannot be
# read the cache cannot be told apart, so usage is not subtracted at all and the room is the limit.
# NA when nothing can be read.
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

# RCTD's per-pixel fit (fitPixels, all three modes) holds t(as.matrix(counts[gene_list_reg, ])) --
# spots x regression genes, dense doubles -- in this process, and its PSOCK cluster exports that
# matrix to every worker. That is intrinsic to RCTD. Checked against what is available before the
# fit starts, with the lower bound only (one copy here plus one per worker), so a run this refuses
# could not have finished; the alternative was the kernel's OOM killer and no JSON at all.
check_fit_memory <- function(n_spots, n_genes, n_workers, avail = available_memory_bytes()) {
  need <- 8 * as.numeric(n_spots) * as.numeric(n_genes) * (1 + n_workers)
  if (!is.na(avail) && need > avail) {
    stop(sprintf(paste0(
      "RCTD's per-pixel fit holds a dense %.0f spots x %.0f regression genes matrix (%.2f GB) here ",
      "and one more copy in each of its %d parallel workers: at least %.2f GB, and %.2f GB is ",
      "available. Lower max_cores (each worker holds its own copy), or raise gene_cutoff_reg and ",
      "fc_cutoff_reg so RCTD selects fewer regression genes."),
      n_spots, n_genes, 8 * as.numeric(n_spots) * as.numeric(n_genes) / 1e9, n_workers,
      need / 1e9, avail / 1e9))
  }
  need
}

# New files appear whole or not at all: written beside the target, then renamed over it.
write_csv_atomic <- function(df, path, row.names) {
  tmp <- paste0(path, ".partial")
  write.csv(df, tmp, row.names = row.names, quote = TRUE)
  if (!file.rename(tmp, path)) stop("could not move ", tmp, " to ", path)
  invisible(path)
}

save_rds_atomic <- function(object, path) {
  tmp <- paste0(path, ".partial")
  saveRDS(object, file = tmp)
  if (!file.rename(tmp, path)) stop("could not move ", tmp, " to ", path)
  invisible(path)
}

# A label that names nothing. The same set tools/worker_utils.drop_unlabeled treats as missing,
# "<na>" included: it is what pandas writes for a missing value once a column is cast to str.
is_unlabeled <- function(labels) {
  is.na(labels) | tolower(trimws(labels)) %in% c("", "nan", "none", "na", "<na>")
}

# spot x cell type matrix from RCTD's doublet call. A singlet is its first_type at 1; a doublet
# (certain or uncertain) splits between first_type and second_type by weights_doublet, which
# decompose_sparse already normalises to 1; a reject has no call and stays NA, as RCTD intends.
doublet_matrix <- function(results, cell_type_names) {
  rdf <- results$results_df
  wd <- as.matrix(results$weights_doublet)
  out <- matrix(0, nrow = nrow(rdf), ncol = length(cell_type_names),
                dimnames = list(rownames(rdf), cell_type_names))
  cls <- as.character(rdf$spot_class)
  ft <- match(as.character(rdf$first_type), cell_type_names)
  st <- match(as.character(rdf$second_type), cell_type_names)
  sgl <- which(cls == "singlet")
  out[cbind(sgl, ft[sgl])] <- 1
  dbl <- which(cls %in% c("doublet_certain", "doublet_uncertain"))
  out[cbind(dbl, ft[dbl])] <- wd[dbl, 1L]
  out[cbind(dbl, st[dbl])] <- wd[dbl, 2L]
  out[which(cls == "reject"), ] <- NA
  out
}

# multi mode leaves RCTD@results as one unnamed list per spot (process_beads_multi), with no
# $weights: the old `as.matrix(rctd@results$weights)` was as.matrix(NULL) and every multi run died
# there, after the whole fit. all_weights is the same unconstrained fit doublet mode keeps in
# $weights; sub_weights is the multi decomposition over the chosen cell_type_list.
multi_matrices <- function(results, barcodes, cell_type_names) {
  n <- length(results)
  all_w <- matrix(0, nrow = n, ncol = length(cell_type_names), dimnames = list(barcodes, cell_type_names))
  sub_w <- all_w
  types <- character(n)
  confident <- character(n)
  n_types <- integer(n)
  for (i in seq_len(n)) {
    r <- results[[i]]
    aw <- r$all_weights
    if (is.null(names(aw))) all_w[i, ] <- as.numeric(aw) else all_w[i, names(aw)] <- as.numeric(aw)
    chosen <- as.character(r$cell_type_list)
    sw <- r$sub_weights
    if (length(chosen) > 0L) {
      nm <- if (is.null(names(sw))) chosen else names(sw)
      sub_w[i, nm] <- as.numeric(sw)
    }
    types[i] <- paste(chosen, collapse = ";")
    confident[i] <- paste(names(r$conf_list)[as.logical(r$conf_list)], collapse = ";")
    n_types[i] <- length(chosen)
  }
  list(all = all_w, sub = sub_w,
       calls = data.frame(spot = barcodes, cell_types = types, n_types = n_types,
                          confident_types = confident, stringsAsFactors = FALSE))
}


parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    mode               = "doublet",
    max_cores          = 4L,
    gene_cutoff        = 0.000125,
    fc_cutoff          = 0.5,
    gene_cutoff_reg    = 0.0002,
    fc_cutoff_reg      = 0.75,
    UMI_min            = 100L,
    UMI_min_sigma      = 100L,
    n_max_cells        = 0L,
    ref_UMI_min        = 100L,
    seed               = 0L
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
    } else if (key == "--mode") {
      opts$mode <- val
    } else if (key == "--max-cores") {
      opts$max_cores <- as.integer(val)
    } else if (key == "--gene-cutoff") {
      opts$gene_cutoff <- as.numeric(val)
    } else if (key == "--fc-cutoff") {
      opts$fc_cutoff <- as.numeric(val)
    } else if (key == "--gene-cutoff-reg") {
      opts$gene_cutoff_reg <- as.numeric(val)
    } else if (key == "--fc-cutoff-reg") {
      opts$fc_cutoff_reg <- as.numeric(val)
    } else if (key == "--umi-min") {
      opts$UMI_min <- as.integer(val)
    } else if (key == "--umi-min-sigma") {
      opts$UMI_min_sigma <- as.integer(val)
    } else if (key == "--n-max-cells") {
      opts$n_max_cells <- as.integer(val)
    } else if (key == "--ref-umi-min") {
      opts$ref_UMI_min <- as.integer(val)
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_rctd <- function(opts) {
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$ref_counts_csv) || is.null(opts$ref_celltypes_csv) ||
      is.null(opts$output_dir)) {
    stop("spacexr requires --spatial-counts-csv, --spatial-coords-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }
  if (!(opts$mode %in% VALID_MODES)) {
    stop(sprintf("mode = '%s' is not an RCTD mode; use one of: %s", opts$mode,
                 paste(VALID_MODES, collapse = ", ")))
  }
  if (is.na(opts$n_max_cells) || opts$n_max_cells < 0L ||
      (opts$n_max_cells > 0L && opts$n_max_cells < CELL_MIN_INSTANCE)) {
    stop(sprintf(paste0("n_max_cells = %s: use 0 (keep every reference cell, the default) or a cap ",
                        "of at least %d cells per type, RCTD's minimum per cell type"),
                 opts$n_max_cells, CELL_MIN_INSTANCE))
  }
  if (is.na(opts$ref_UMI_min) || opts$ref_UMI_min < 0L) {
    stop(sprintf("ref_UMI_min = %s: use a non-negative UMI count", opts$ref_UMI_min))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  warn_msgs <- character(0)
  add_warning <- function(...) {
    msg <- paste0(...)
    log_msg("WARNING: ", msg)
    warn_msgs <<- c(warn_msgs, msg)
  }

  # --- Load reference data (sparse + integer; mirrors manual runner) ---
  log_msg("STEP-1 Reading reference counts from: ", opts$ref_counts_csv)
  # Read straight into a sparse dgCMatrix, so RCTD's check_counts also skips the dense->sparse
  # coercion path that fails on >2^31-element matrices.
  ref_counts_mat <- read_counts_csv_sparse(opts$ref_counts_csv, "reference counts")
  log_msg("STEP-1 ref_counts_mat as dgCMatrix: ", nrow(ref_counts_mat), " x ", ncol(ref_counts_mat))
  # RCTD requires integer counts — ceiling fractional values (preserves UMI semantics). Counted,
  # because a normalised or log matrix passes through this line looking like counts.
  n_ref_values_non_integer <- sum(ref_counts_mat@x %% 1 != 0)
  ref_counts_mat@x <- as.numeric(ceiling(ref_counts_mat@x))

  log_msg("STEP-2 Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE)
  if (ncol(ref_ct_df) == 0L) {
    stop("ref_celltypes_csv ", opts$ref_celltypes_csv, " has cell IDs but no label column")
  }
  # Auto-detect celltype column when given a multi-column metadata file.
  # Without this, STCoscientist sometimes passes the full obs metadata CSV — the first
  # column then becomes Age/Donor/etc. and RCTD silently labels every cell
  # with junk like "60-64y", producing meaningless deconvolution weights.
  if (ncol(ref_ct_df) > 1L) {
    pat <- "^(celltype|cell_type|cell\\.type|annotation|annot|cluster|label)$"
    hits <- grep(pat, colnames(ref_ct_df), ignore.case = TRUE)
    if (length(hits) >= 1L) {
      sel <- hits[[1L]]
      log_msg("STEP-2 metadata CSV has ", ncol(ref_ct_df),
              " columns; auto-selected '", colnames(ref_ct_df)[sel],
              "' (col index ", sel, ") as celltype")
    } else {
      stop(sprintf(
        "ref_celltypes_csv has %d columns and no recognizable celltype column (%s); columns were: %s",
        ncol(ref_ct_df), pat, paste(colnames(ref_ct_df), collapse = ", ")))
    }
  } else {
    sel <- 1L
  }
  celltype_column <- colnames(ref_ct_df)[sel]
  raw_ct <- as.character(ref_ct_df[, sel])
  names(raw_ct) <- rownames(ref_ct_df)

  # Ensure matching cell IDs
  common_cells <- intersect(colnames(ref_counts_mat), names(raw_ct))
  if (length(common_cells) == 0) {
    log_msg("STEP-2 0 common cells, transposing ref_counts_mat")
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(raw_ct))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(raw_ct)))}
  n_ref_cells_in_counts <- ncol(ref_counts_mat)
  n_ref_cells_matched <- length(common_cells)
  if (n_ref_cells_matched < n_ref_cells_in_counts) {
    add_warning(n_ref_cells_in_counts - n_ref_cells_matched, " of ", n_ref_cells_in_counts,
                " reference cells in the counts have no row in ref_celltypes_csv and were left out.")
  }
  ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]
  raw_ct <- raw_ct[common_cells]

  # Drop NA / blank / "nan" labels (would crash get_de_genes during create.RCTD). A missing label
  # is not a class, so these cells leave the reference -- and the payload says how many.
  unlabeled <- is_unlabeled(raw_ct)
  n_ref_cells_unlabeled <- sum(unlabeled)
  if (n_ref_cells_unlabeled > 0) {
    log_msg("STEP-2 dropping ", n_ref_cells_unlabeled, " cells with NA/blank cell_type")
    add_warning(n_ref_cells_unlabeled, " of ", n_ref_cells_matched, " reference cells have no label in '",
                celltype_column, "' (NA, empty, 'nan', 'none', 'na' or '<NA>') and were left out.")
    ref_counts_mat <- ref_counts_mat[, !unlabeled, drop = FALSE]
    raw_ct <- raw_ct[!unlabeled]
  }
  if (length(raw_ct) == 0) {
    stop("every matched reference cell is unlabeled in '", celltype_column, "' of ", opts$ref_celltypes_csv)
  }

  # RCTD's Reference() rejects '/' in a cell type name, and get_de_genes is unsafe with
  # whitespace, so both become '_'. The columns of every output use the new names; the payload
  # maps them back, and two labels that would become one name stop the run instead of merging.
  clean_ct <- gsub("[/[:space:]]+", "_", raw_ct)
  name_map <- unique(data.frame(from = unname(raw_ct), to = unname(clean_ct), stringsAsFactors = FALSE))
  merged <- unique(name_map$to[duplicated(name_map$to)])
  if (length(merged) > 0) {
    clash <- name_map[name_map$to %in% merged, , drop = FALSE]
    stop(sprintf(paste0("cell type labels %s would all become '%s' once '/' and whitespace are ",
                        "replaced with '_' (RCTD rejects '/'); rename them so they stay distinct"),
                 paste(sprintf("'%s'", clash$from[clash$to == merged[1]]), collapse = ", "), merged[1]))
  }
  name_map <- name_map[name_map$from != name_map$to, , drop = FALSE]
  renamed_cell_types <- stats::setNames(as.list(name_map$to), name_map$from)
  if (nrow(name_map) > 0) {
    add_warning(nrow(name_map), " cell type label(s) contain '/' or whitespace and appear in the ",
                "outputs with '_' instead (summary.renamed_cell_types), e.g. '", name_map$from[1],
                "' -> '", name_map$to[1], "'.")
  }
  cell_types <- factor(unname(clean_ct))
  names(cell_types) <- names(raw_ct)

  # --- Load spatial data (sparse + integer) ---
  log_msg("STEP-3 Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_mat <- read_counts_csv_sparse(opts$spatial_counts_csv, "spatial counts")
  log_msg("STEP-3 sp_counts_mat as dgCMatrix: ", nrow(sp_counts_mat), " x ", ncol(sp_counts_mat))
  n_sp_values_non_integer <- sum(sp_counts_mat@x %% 1 != 0)
  sp_counts_mat@x <- as.numeric(ceiling(sp_counts_mat@x))

  log_msg("STEP-4 Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Space Ranger 1's tissue_positions_list.csv has no header row: read.csv(header = TRUE) took its first
  # spot as the header -- that spot was lost, and the columns were named after its values, so the
  # resolver's fall-through read the in_tissue flag as both axes.
  sp_coords_raw <- read_coords_by_name(opts$spatial_coords_csv)
  coord_cols <- resolve_coord_cols(colnames(sp_coords_raw)[-1], opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(sp_coords_raw), collapse = ", "), ")")
  sp_coords_df <- data.frame(
    x = as.numeric(sp_coords_raw[[coord_cols[1]]]),
    y = as.numeric(sp_coords_raw[[coord_cols[2]]]),
    row.names = sp_coords_raw[, 1]
  )

  # Ensure matching spot IDs
  common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  if (length(common_spots) == 0) {
    log_msg("STEP-4 0 common spots, transposing sp_counts_mat")
    sp_counts_mat <- t(sp_counts_mat)
    common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(sp_counts_mat), "coordinates", rownames(sp_coords_df)))}
  # A spot in the counts with no coordinates never reaches RCTD, and so is in neither n_spots nor
  # n_spots_in; counted here so the gap between the counts file and the result is accounted for.
  n_spots_supplied <- ncol(sp_counts_mat)
  n_spots_without_coords <- n_spots_supplied - length(common_spots)
  n_coords_without_counts <- nrow(sp_coords_df) - length(common_spots)
  if (n_spots_without_coords > 0) {
    add_warning(n_spots_without_coords, " of ", n_spots_supplied, " spots in the spatial counts have no ",
                "row in the coordinates file and were not analysed.")
  }
  # Background spots (in_tissue == 0 in the coordinates file) are left out and counted.
  tissue <- keep_in_tissue_spots(common_spots, in_tissue_flags(sp_coords_raw, sp_coords_raw[, 1]),
                                 opts$spatial_coords_csv)
  common_spots <- tissue$spots
  n_spots_off_tissue_dropped <- tissue$n_dropped
  if (!is.null(tissue$warning)) add_warning(tissue$warning)
  sp_counts_mat <- sp_counts_mat[, common_spots, drop = FALSE]
  sp_coords_df <- sp_coords_df[common_spots, 1:2, drop = FALSE]
  colnames(sp_coords_df) <- c("x", "y")

  # --- Restrict sp + sc to gene intersection (mirrors manual; prevents
  # RCTD internal subscript-out-of-bounds when get_de_genes returns a gene
  # that exists in sc reference but not in spatial counts). ---
  n_genes_spatial_in <- nrow(sp_counts_mat)
  n_genes_ref_in <- nrow(ref_counts_mat)
  gene_intersection <- intersect(rownames(sp_counts_mat), rownames(ref_counts_mat))
  if (length(gene_intersection) == 0) {
    stop("STEP-5 no shared genes between spatial and reference")
  }
  log_msg("STEP-5 gene intersection: ", length(gene_intersection),
          " (sp had ", nrow(sp_counts_mat), ", ref had ", nrow(ref_counts_mat), ")")
  sp_counts_mat  <- sp_counts_mat[gene_intersection, , drop = FALSE]
  ref_counts_mat <- ref_counts_mat[gene_intersection, , drop = FALSE]

  # --- Reference cell filters, applied here so each is counted ---
  # Reference() drops every cell with nUMI < min_UMI (its default 100) with nothing but an R
  # warning on stderr. nUMI is the cell's total over the shared genes, as it always was here.
  ref_nUMI <- colSums(ref_counts_mat)
  low_umi <- ref_nUMI < opts$ref_UMI_min
  n_ref_cells_below_umi <- sum(low_umi)
  if (n_ref_cells_below_umi > 0) {
    if (n_ref_cells_below_umi == length(ref_nUMI)) {
      stop(sprintf(paste0("every one of the %d labelled reference cells has fewer than ref_UMI_min = %d ",
                          "UMIs over the %d genes it shares with the spatial data (largest: %.0f); lower ",
                          "ref_UMI_min"),
                   length(ref_nUMI), opts$ref_UMI_min, length(gene_intersection), max(ref_nUMI)))
    }
    add_warning(n_ref_cells_below_umi, " of ", length(ref_nUMI), " labelled reference cells have fewer than ",
                "ref_UMI_min = ", opts$ref_UMI_min, " UMIs over the ", length(gene_intersection),
                " genes shared with the spatial data and were left out; lower ref_UMI_min to keep them.")
    ref_counts_mat <- ref_counts_mat[, !low_umi, drop = FALSE]
    cell_types <- cell_types[!low_umi]
    ref_nUMI <- ref_nUMI[!low_umi]
  }
  cell_types <- droplevels(cell_types)

  # Filter out cell types with <25 cells (RCTD minimum requirement), counted after the UMI filter
  # so a type the filter thins below RCTD's floor leaves here instead of crashing
  # process_cell_type_info.
  ct_counts <- table(cell_types)
  small_cts <- names(ct_counts[ct_counts < CELL_MIN_INSTANCE])
  n_ref_cells_small_types <- 0L
  if (length(small_cts) > 0) {
    log_msg("STEP-5 dropping ", length(small_cts), " cell type(s) with <25 cells: ", paste(small_cts, collapse=", "))
    n_ref_cells_small_types <- as.integer(sum(ct_counts[small_cts]))
    add_warning(length(small_cts), " cell type(s) have fewer than ", CELL_MIN_INSTANCE,
                " usable reference cells, RCTD's minimum, and are absent from every output: ",
                paste(sprintf("%s (%d)", small_cts, as.integer(ct_counts[small_cts])), collapse = ", "), ".")
    keep <- !(cell_types %in% small_cts)
    cell_types <- droplevels(cell_types[keep])
    ref_counts_mat <- ref_counts_mat[, keep, drop = FALSE]
    ref_nUMI <- ref_nUMI[keep]
  }
  if (nlevels(cell_types) < 2L) {
    stop(sprintf(paste0("RCTD needs at least 2 cell types with %d or more reference cells; after the ",
                        "filters %d remain (per type before the %d-cell floor: %s)"),
                 CELL_MIN_INSTANCE, nlevels(cell_types), CELL_MIN_INSTANCE,
                 paste(sprintf("%s=%d", names(ct_counts), as.integer(ct_counts)), collapse = ", ")))
  }

  n_ref_cells_before_cap <- ncol(ref_counts_mat)
  # Reference() also draws at most n_max_cells cells per type at random (its default: 10,000) and
  # says so only in an R warning. That cap is opt-in now: 0 hands it the largest type's size, so
  # nothing is drawn. The profiles below are computed sparse, which is what makes that affordable.
  cap <- if (opts$n_max_cells > 0L) opts$n_max_cells else max(as.integer(table(cell_types)))
  log_msg("STEP-6 Creating Reference object: ", length(cell_types), " cells, ",
          length(levels(cell_types)), " cell types (n_max_cells = ",
          if (opts$n_max_cells > 0L) opts$n_max_cells else "none", ")")
  # Pass nUMI + require_int=FALSE explicitly: avoids check_counts integer
  # modulo over the entire sparse matrix (slow + has overflow edge cases).
  reference <- Reference(
    counts      = ref_counts_mat,
    cell_types  = cell_types,
    nUMI        = ref_nUMI,
    require_int = FALSE,
    n_max_cells = cap,
    min_UMI     = opts$ref_UMI_min
  )
  n_ref_cells <- ncol(reference@counts)
  n_ref_cells_downsampled <- n_ref_cells_before_cap - n_ref_cells
  if (n_ref_cells_downsampled > 0) {
    add_warning("n_max_cells = ", opts$n_max_cells, ": RCTD's Reference drew ", opts$n_max_cells,
                " cells at random (seed ", opts$seed, ") from every cell type larger than that, so ",
                n_ref_cells, " of ", n_ref_cells_before_cap, " reference cells were used; n_max_cells = 0 ",
                "uses every cell.")
  }
  rm(ref_counts_mat); gc(verbose = FALSE)
  log_msg("STEP-6 cell type profiles from ", n_ref_cells, " reference cells (sparse)")
  profiles <- cell_type_profiles_sparse(reference@counts, reference@cell_types, reference@nUMI)
  ref_types_used <- colnames(profiles)
  ref_type_counts <- as.list(table(reference@cell_types)[ref_types_used])

  log_msg("STEP-7 Creating SpatialRNA object: ", ncol(sp_counts_mat), " spots, ",
          nrow(sp_counts_mat), " genes")
  spatial_rna <- SpatialRNA(
    coords      = sp_coords_df,
    counts      = sp_counts_mat,
    nUMI        = colSums(sp_counts_mat),
    require_int = FALSE
  )

  # --- Run RCTD ---
  log_msg("STEP-8 Creating RCTD object (mode = ", opts$mode, ")...")
  rctd <- create.RCTD(
    spatial_rna,
    reference,
    max_cores          = opts$max_cores,
    gene_cutoff        = opts$gene_cutoff,
    fc_cutoff          = opts$fc_cutoff,
    gene_cutoff_reg    = opts$gene_cutoff_reg,
    fc_cutoff_reg      = opts$fc_cutoff_reg,
    UMI_min            = opts$UMI_min,
    UMI_min_sigma      = opts$UMI_min_sigma,
    cell_type_profiles = profiles
  )
  rm(reference); gc(verbose = FALSE)
  n_genes_reg  <- length(rctd@internal_vars$gene_list_reg)
  n_genes_bulk <- length(rctd@internal_vars$gene_list_bulk)
  n_fit_spots  <- ncol(rctd@spatialRNA@counts)
  n_workers <- if (opts$max_cores > 1L) min(opts$max_cores, parallel::detectCores()) else 0L
  check_fit_memory(n_fit_spots, n_genes_reg, n_workers)

  log_msg("Running RCTD deconvolution...")
  rctd <- run.RCTD(rctd, doublet_mode = opts$mode)

  # The fit is the expensive part: it is on disk before anything below can fail.
  rds_path <- file.path(opts$output_dir, "spacexr_rctd.rds")
  save_rds_atomic(rctd, rds_path)

  # --- Extract results ---
  log_msg("Extracting deconvolution results...")
  cell_type_names <- rctd@cell_type_info$renorm[[2]]
  sidecar_path <- NULL
  spot_class_counts <- NULL
  if (opts$mode == "multi") {
    mm <- multi_matrices(rctd@results, colnames(rctd@spatialRNA@counts), cell_type_names)
    weights_raw <- mm$all
  } else {
    # doublet mode keeps the unconstrained fit (every spot's all_weights) in $weights -- the very
    # fit full mode returns -- and the doublet call in results_df / weights_doublet.
    weights_raw <- as.matrix(rctd@results$weights)
  }
  # Normalize per-row to proportions — same convention as the
  # manual baseline runner so downstream eval reads identical schema.
  row_sums <- rowSums(weights_raw)
  n_spots_zero_weight <- sum(row_sums == 0)
  if (n_spots_zero_weight > 0) {
    add_warning(n_spots_zero_weight, " spot(s) got zero weight for every cell type and are all-zero rows ",
                "in proportions.csv.")
  }
  row_sums[row_sums == 0] <- 1
  prop <- sweep(weights_raw, 1, row_sums, FUN = "/")
  prop_df <- as.data.frame(prop, optional = TRUE)

  prop_path <- file.path(opts$output_dir, "proportions.csv")
  write_csv_atomic(prop_df, prop_path, row.names = TRUE)
  log_msg("STEP-9 wrote proportions.csv: ", nrow(prop_df), " x ", ncol(prop_df))

  # Keep the legacy spacexr_weights.csv for any external consumer still expecting it.
  if (opts$mode == "full") {
    weights_df <- prop_df
    weights_df$spot <- rownames(weights_df)
  } else if (opts$mode == "multi") {
    weights_df <- mm$calls
    sidecar_path <- file.path(opts$output_dir, "spacexr_rctd_multi_weights.csv")
    write_csv_atomic(as.data.frame(mm$sub, optional = TRUE), sidecar_path, row.names = TRUE)
  } else {
    results_df <- rctd@results$results_df
    weights_df <- data.frame(
      spot = rownames(results_df),
      first_type = as.character(results_df$first_type),
      second_type = as.character(results_df$second_type),
      spot_class = as.character(results_df$spot_class),
      stringsAsFactors = FALSE
    )
    classes <- factor(as.character(results_df$spot_class),
                      levels = c("singlet", "doublet_certain", "doublet_uncertain", "reject"))
    spot_class_counts <- as.list(table(classes))
    sidecar_path <- file.path(opts$output_dir, "spacexr_rctd_doublet_weights.csv")
    write_csv_atomic(as.data.frame(doublet_matrix(rctd@results, cell_type_names), optional = TRUE),
                     sidecar_path, row.names = TRUE)
  }
  weights_path <- file.path(opts$output_dir, "spacexr_weights.csv")
  write_csv_atomic(weights_df, weights_path, row.names = FALSE)

  cell_types_found <- colnames(prop_df)

  # RCTD drops spots its internal filters reject (e.g. nUMI < UMI_min), so the deliverable can
  # hold fewer rows than the matrix we fed it. n_spots describes proportions.csv -- the thing a
  # reader will open -- and n_spots_in keeps the input count. Live round 2: the old summary said
  # "across 4035 spots" over a 4033-row file.
  n_spots_mapped <- nrow(prop_df)
  n_spots_in     <- ncol(sp_counts_mat)
  n_dropped      <- n_spots_in - n_spots_mapped

  # What proportions.csv is, per mode. In doublet mode it is NOT the doublet call: RCTD keeps its
  # unconstrained fit in results$weights (gather_results: weights[i, ] = all_weights), the same
  # decompose_full(constrain = FALSE) fit full mode makes, so any number of types can be non-zero.
  proportions_source <- if (opts$mode == "full") {
    "RCTD full-mode weights, row-normalised"
  } else {
    "RCTD unconstrained full-model weights (all_weights, the fit full mode returns), row-normalised"
  }
  mode_note <- if (opts$mode == "full") {
    "proportions.csv holds RCTD's full-mode weights, normalised per spot."
  } else if (opts$mode == "doublet") {
    paste0(
      "proportions.csv holds RCTD's unconstrained full-model weights (results$weights, the same fit ",
      "full mode returns), normalised per spot, so any number of types can be non-zero in a spot. ",
      "The doublet call -- at most 2 types per spot -- is spacexr_rctd_doublet_weights.csv (singlet = ",
      "1 for first_type; reject = NA), with each spot's class in spacexr_weights.csv: ",
      spot_class_counts$singlet, " singlet, ", spot_class_counts$doublet_certain, " doublet_certain, ",
      spot_class_counts$doublet_uncertain, " doublet_uncertain, ", spot_class_counts$reject, " reject."
    )
  } else {
    paste0(
      "proportions.csv holds RCTD's unconstrained full-model weights (each spot's all_weights), ",
      "normalised per spot. The multi-type decomposition -- at most ", rctd@config$MAX_MULTI_TYPES,
      " types per spot -- is spacexr_rctd_multi_weights.csv, with each spot's chosen types in ",
      "spacexr_weights.csv."
    )
  }
  ref_losses <- c(
    if (n_ref_cells_matched < n_ref_cells_in_counts) paste0(n_ref_cells_in_counts - n_ref_cells_matched, " without a label row"),
    if (n_ref_cells_unlabeled > 0) paste0(n_ref_cells_unlabeled, " unlabeled"),
    if (n_ref_cells_below_umi > 0) paste0(n_ref_cells_below_umi, " below ref_UMI_min = ", opts$ref_UMI_min),
    if (n_ref_cells_small_types > 0) paste0(n_ref_cells_small_types, " in types under ", CELL_MIN_INSTANCE, " cells"),
    if (n_ref_cells_downsampled > 0) paste0(n_ref_cells_downsampled, " not drawn under n_max_cells = ", opts$n_max_cells)
  )
  non_integer_note <- if (n_ref_values_non_integer + n_sp_values_non_integer > 0) {
    paste0(" NOTE: ", n_sp_values_non_integer, " spatial and ", n_ref_values_non_integer,
           " reference values were not integers and were rounded up, because RCTD models counts; ",
           "if either matrix is normalised or log-transformed this fit is not meaningful.")
  } else {
    ""
  }
  if (n_ref_values_non_integer + n_sp_values_non_integer > 0) {
    add_warning(n_sp_values_non_integer, " spatial and ", n_ref_values_non_integer,
                " reference count values were not integers and were rounded up (ceiling) for RCTD.")
  }

  output_files <- list(
    proportions_csv  = prop_path,
    weights_csv      = weights_path,
    rctd_rds         = rds_path
  )
  if (opts$mode == "doublet") output_files$doublet_weights_csv <- sidecar_path
  if (opts$mode == "multi") output_files$multi_weights_csv <- sidecar_path
  result <- list(
    status       = "ok",
    tool         = "spacexr",
    task         = "deconvolution",
    data         = list(
      n_spots          = n_spots_mapped,
      n_spots_in       = n_spots_in,
      n_cell_types     = length(cell_types_found),
      n_genes_spatial  = nrow(sp_counts_mat),
      n_genes_spatial_in       = n_genes_spatial_in,
      n_genes_ref_in           = n_genes_ref_in,
      n_genes_shared           = length(gene_intersection),
      n_genes_reg              = n_genes_reg,
      n_genes_bulk             = n_genes_bulk,
      n_spots_supplied         = n_spots_supplied,
      n_spots_without_coords   = n_spots_without_coords,
      n_spots_off_tissue_dropped = n_spots_off_tissue_dropped,
      n_coords_without_counts  = n_coords_without_counts,
      n_spots_zero_weight      = n_spots_zero_weight,
      n_ref_cells_in_counts    = n_ref_cells_in_counts,
      n_ref_cells_matched      = n_ref_cells_matched,
      n_ref_cells_unlabeled    = n_ref_cells_unlabeled,
      n_ref_cells_below_umi    = n_ref_cells_below_umi,
      n_ref_cells_small_types  = n_ref_cells_small_types,
      n_ref_cells_downsampled  = n_ref_cells_downsampled,
      n_ref_cells              = n_ref_cells,
      n_values_non_integer_spatial = n_sp_values_non_integer,
      n_values_non_integer_ref     = n_ref_values_non_integer,
      celltype_column          = celltype_column,
      proportions_source       = proportions_source
    ),
    output_files = output_files,
    params       = list(
      mode             = opts$mode,
      max_cores        = opts$max_cores,
      gene_cutoff      = opts$gene_cutoff,
      fc_cutoff        = opts$fc_cutoff,
      gene_cutoff_reg  = opts$gene_cutoff_reg,
      fc_cutoff_reg    = opts$fc_cutoff_reg,
      UMI_min          = opts$UMI_min,
      UMI_min_sigma    = opts$UMI_min_sigma,
      n_max_cells      = opts$n_max_cells,
      ref_UMI_min      = opts$ref_UMI_min,
      seed             = opts$seed,
      method           = sprintf("RCTD (spacexr %s), %s mode", as.character(utils::packageVersion("spacexr")), opts$mode),
      used_fallback    = FALSE
    ),
    summary      = c(
      list(
        # The cells RCTD's reference was actually built from, after every filter below; it used
        # to be the count matched to a label, before any of them.
        n_ref_cells      = n_ref_cells,
        n_ref_types      = length(ref_types_used),
        ref_cells_per_type = ref_type_counts,
        dropped_cell_types = I(small_cts),
        renamed_cell_types = if (length(renamed_cell_types) > 0) renamed_cell_types else structure(list(), names = character(0)),
        cell_types_found = cell_types_found
      ),
      if (!is.null(spot_class_counts)) list(spot_class_counts = spot_class_counts)
    ),
    analysis     = paste0(
      "RCTD mapped ", length(cell_types_found), " cell types across ",
      n_spots_mapped,
      if (n_dropped > 0) paste0(" of ", n_spots_in) else "",
      " spots in ", opts$mode, " mode.",
      if (n_dropped > 0) {
        paste0(
          " ", n_dropped, " input spot", if (n_dropped == 1) "" else "s",
          " fell below RCTD's internal filters (e.g. UMI_min = ", opts$UMI_min,
          ") and are absent from proportions.csv."
        )
      } else {
        ""
      },
      if (n_spots_without_coords > 0) {
        paste0(" ", n_spots_without_coords, " spots in the counts had no coordinates and were not analysed.")
      } else {
        ""
      },
      if (n_spots_off_tissue_dropped > 0) {
        paste0(" ", n_spots_off_tissue_dropped, " spots marked in_tissue == 0 (background) were left out.")
      } else {
        ""
      },
      " ", mode_note,
      " The reference was built from ", n_ref_cells, " of ", n_ref_cells_in_counts, " cells in ",
      length(ref_types_used), " types",
      if (length(ref_losses) > 0) paste0(" (left out: ", paste(ref_losses, collapse = "; "), ")") else "",
      if (length(small_cts) > 0) paste0("; types dropped: ", paste(small_cts, collapse = ", ")) else "",
      ". RCTD fitted each spot on ", n_genes_reg, " regression genes (", n_genes_bulk,
      " platform-effect genes) selected from the ", length(gene_intersection),
      " genes the slide and the reference share.",
      non_integer_note
    )
  )
  if (!is.null(tissue$filter)) result$params$in_tissue_filter <- tissue$filter
  if (length(warn_msgs) > 0) {
    result$warnings <- I(warn_msgs)
  }
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args runs inside the handler: an unknown flag or a missing value must still leave a JSON
  # payload on stdout, not a bare R error with nothing for the portal to read.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    set.seed(opts$seed)
    run_rctd(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "spacexr",
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
