#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(PRECAST)
  library(Seurat)
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
  message(sprintf("[precast-worker] %s", msg))
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


# Space Ranger before 2.0 writes spatial/tissue_positions_list.csv with NO header row -- every
# standard Visium folder ships it beside the headed tissue_positions.csv. read.csv(header = TRUE)
# takes its first spot as the header, so the columns come back named after that spot's own values,
# no named pair matches, resolve_coord_cols falls through to the tissue flag and the array row, and
# the first spot is lost to the header.
#
# first_record_line() and read_coords_csv() are copied from tools/spotsweeper_worker.R, the fleet's
# one reader of this file (each R worker runs as its own Rscript in its own env, so nothing can be
# sourced). The reader this worker carried before took ANY first line whose fields after the
# barcode were numbers as a spot and named the columns V1, V2, ... for any width other than six, so a
# headerless four- or five-column file had its first two numeric columns -- whatever they were --
# read as the axes, silently.
#
# The first line is a header unless it looks like data: an identifier that is not an
# identifier-column label, followed by nothing but numbers. A pandas frame written with integer
# column names (",0,1" or "barcode,0,1") keeps reading as a header. So does one whose index label is
# not in that list ("cell_barcode,0,1,2") when it has four or more fields: names that are exactly
# 0, 1, 2, ... are pandas' RangeIndex, and no Space Ranger row reads 0,1,2,3,4. Three fields stay
# read as a spot. A headerless file is read under Space Ranger's own column names when it has Space
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

# resolve_coord_cols is the same text in fourteen workers and a test keeps the copies in step, so
# the check that its answer is usable sits beside the callers rather than inside it. A name that is
# a bare number is a spot's value read as the header; a name the file carries more than once cannot
# pick one column (data.frame subsetting hands back the first match for both). Names that are
# exactly 0, 1, ... are pandas' RangeIndex (DataFrame(obsm['spatial']).to_csv() writes ",0,1"),
# which read_coords_csv reads as the header it is, so they are the file's own names, not a spot.
check_coord_cols <- function(coord_cols, coord_names, source_path) {
  same <- length(coord_cols) == 2L && coord_cols[1] == coord_cols[2]
  range_index <- length(coord_names) >= 2L &&
    identical(as.character(coord_names), as.character(seq_along(coord_names) - 1L))
  numeric_names <- if (range_index) character(0) else
    coord_cols[!is.na(suppressWarnings(as.numeric(coord_cols)))]
  repeated <- unique(coord_cols[coord_cols %in% coord_names[duplicated(coord_names)]])
  if (same || length(numeric_names) > 0 || length(repeated) > 0) {
    stop("The coordinate columns chosen from ", source_path, " are \"", coord_cols[1], "\" and \"",
         coord_cols[2], "\" (the file has: ", paste(coord_names, collapse = ", "), "). ",
         if (same) "They are the same column. " else "",
         if (length(numeric_names) > 0) paste0(
           "A column named after a number (", paste(sprintf('"%s"', numeric_names), collapse = ", "),
           ") is a spot's line read as the header line. ") else "",
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

# The two chosen columns as a numeric n x 2 matrix, refusing what PRECAST cannot place.
coord_matrix <- function(coords_df, source_path) {
  out <- matrix(NA_real_, nrow = nrow(coords_df), ncol = 2L)
  for (k in 1:2) {
    v <- suppressWarnings(as.numeric(as.character(coords_df[[k]])))
    bad <- sum(is.na(v) | !is.finite(v))
    if (bad > 0) {
      stop("Coordinate column \"", colnames(coords_df)[k], "\" of ", source_path, " has ", bad,
           " non-numeric, missing or infinite values among the ", length(v),
           " spots in the analysis. Every spot needs two finite coordinates.")
    }
    out[, k] <- v
  }
  out
}

# How PRECAST finds each spot's neighbours. Upstream (PRECAST 1.8 AddAdjList):
#   'ST' / 'Visium'  find_neighbors: a spot's neighbours are the spots at EXACT offsets of the
#                    (row, col) pair -- ST (0, +-1) / (+-1, 0), Visium (col +-2, row) and
#                    (col +-1, row +-1). These are array-index offsets; on pixel coordinates (spots
#                    ~100-300 px apart) or on a hexagonal lattice read with the ST offsets, nothing
#                    ever matches.
#   'Other_SRT'      DR.SC getAdj_auto: a radius search on whatever the coordinates are.
# The shipped default was 'ST', and the worker preferred the pixel columns of a Space Ranger file, so
# every Visium input -- the converter's x/y, tissue_positions.csv, the h5ad's array coordinates --
# got an adjacency with no edges. PRECAST then printed "Neighbors were identified for 0 out of N
# spots" to stderr and ran as a purely non-spatial clustering, and the payload said status ok.
PLATFORMS <- c("auto", "ST", "Visium", "Other_SRT")
LATTICE_PLATFORMS <- c("ST", "Visium")

canonical_platform <- function(platform) {
  hit <- match(tolower(platform), tolower(PLATFORMS))
  if (is.na(hit)) {
    stop("platform must be one of ", paste(sprintf("'%s'", PLATFORMS), collapse = ", "),
         "; got '", platform, "'. 'auto' reads the coordinates file and picks the neighbour ",
         "definition that fits it.")
  }
  PLATFORMS[hit]
}

# A lattice platform needs the array indices, not the pixel positions resolve_coord_cols prefers.
lattice_coord_cols <- function(coord_names) {
  lc <- tolower(coord_names)
  pair <- c("array_row", "array_col")
  if (all(pair %in% lc)) coord_names[match(pair, lc)] else NULL
}

# Neighbours per spot under upstream's own offsets, counted sparsely (upstream builds an n x n
# dense matrix to do this; here it only decides and checks, so an O(n) lookup is enough).
lattice_neighbour_counts <- function(row, col, platform) {
  if (tolower(platform) == "visium") {
    dx <- c(-2, 2, -1, 1, -1, 1)
    dy <- c(0, 0, -1, -1, 1, 1)
  } else {
    dx <- c(0, 1, 0, -1)
    dy <- c(-1, 0, 1, 0)
  }
  key <- paste(col, row, sep = "_")
  n <- integer(length(row))
  for (k in seq_along(dx)) {
    n <- n + as.integer(paste(col + dx[k], row + dy[k], sep = "_") %in% key)
  }
  n
}

is_integer_valued <- function(v) {
  v <- suppressWarnings(as.numeric(as.character(v)))
  length(v) > 0 && all(is.finite(v)) && all(abs(v - round(v)) < 1e-8)
}

# Which neighbour definition runs, and on which two columns of each sample's coordinates file.
# Decided once for all samples: AddAdjList takes one platform.
#   ST / Visium  the array indices when the file names array_row/array_col, else the resolved pair.
#   Other_SRT    the resolved pair (pixel positions first), as before.
#   auto         the array indices (or the resolved pair when it is itself an integer grid) if every
#                sample has neighbours under the ST offsets (a square grid) or else under the Visium
#                offsets (a hexagonal grid); otherwise Other_SRT on the resolved pair.
choose_platform <- function(requested, coords_list, resolved, lattice) {
  n <- length(coords_list)
  cand <- lapply(seq_len(n), function(s) if (!is.null(lattice[[s]])) lattice[[s]] else resolved[[s]])
  if (requested %in% LATTICE_PLATFORMS) {
    return(list(platform = requested, coord_cols = cand,
                why = paste0("platform='", requested, "' was requested")))
  }
  if (requested == "Other_SRT") {
    return(list(platform = "Other_SRT", coord_cols = resolved,
                why = "platform='Other_SRT' was requested"))
  }
  on_grid <- all(vapply(seq_len(n), function(s) {
    is_integer_valued(coords_list[[s]][[cand[[s]][1]]]) &&
      is_integer_valued(coords_list[[s]][[cand[[s]][2]]])
  }, logical(1)))
  if (on_grid) {
    for (platform in LATTICE_PLATFORMS) {
      med <- vapply(seq_len(n), function(s) {
        df <- coords_list[[s]]
        stats::median(lattice_neighbour_counts(as.numeric(df[[cand[[s]][1]]]),
                                               as.numeric(df[[cand[[s]][2]]]), platform))
      }, numeric(1))
      if (all(med > 0)) {
        shape <- if (platform == "ST") "a square grid" else "a hexagonal (Visium) grid"
        return(list(platform = platform, coord_cols = cand,
                    why = paste0("platform='auto': ", paste(cand[[1]], collapse = "/"), " form ",
                                 shape, " (median ", paste(med, collapse = ", "),
                                 " neighbours per spot under the ", platform, " offsets)")))
      }
    }
  }
  list(platform = "Other_SRT", coord_cols = resolved,
       why = paste0("platform='auto': ", paste(resolved[[1]], collapse = "/"),
                    " are not array indices on a square or hexagonal grid, so neighbours are ",
                    "found by a radius search (Other_SRT)"))
}

no_neighbour_msg <- function(platform, s, n_with, n_spots, coord_cols, source_path) {
  head <- paste0("platform='", platform, "' finds a neighbour for ", n_with, " of ", n_spots,
                 " spots in sample ", s, " (coordinates ", paste(coord_cols, collapse = "/"), " of ",
                 source_path, "), so PRECAST would run as a non-spatial clustering. ")
  if (!(platform %in% LATTICE_PLATFORMS)) {
    return(paste0(head, "PRECAST's radius search ended with a median of 0 neighbours per spot on these ",
                  "coordinates. Check that the two columns read are the spot positions, or give a ",
                  "coordinates file carrying array_row/array_col so the ST or Visium grid can be used."))
  }
  paste0(head, "'", platform,
         "' matches neighbours at exact array-index offsets (ST: +-1 on one axis; Visium: col +-2, ",
         "or col +-1 and row +-1), which pixel positions and the wrong grid never hit. Pass ",
         "platform='auto' to let the coordinates decide, platform='Other_SRT' for a radius search ",
         "on these coordinates, or a coordinates file carrying array_row/array_col.")
}

# Upstream find_neighbors allocates an n x n double matrix per sample and then a logical mask of it
# (about 12 bytes per spot pair). That is intrinsic to PRECAST's ST/Visium path, so the size is
# checked against the memory this process can still allocate before it is asked for.
#
# The same reading as worker_utils.available_memory_bytes (Python workers call that one): the smaller
# of MemAvailable and the room left under the cgroup memory limit, with the cgroup's page cache
# (active_file + inactive_file in memory.stat) counted as reclaimable. MemAvailable alone is the
# host's memory seen from inside a memory-limited container, so the refusal passed there and the run
# was OOM-killed with no payload.
read_small_text <- function(path) {
  tryCatch(suppressWarnings(readLines(path, warn = FALSE)), error = function(e) character(0))
}

meminfo_available_bytes <- function(path = "/proc/meminfo") {
  line <- grep("^MemAvailable:", read_small_text(path), value = TRUE)
  if (length(line) == 0) return(NA_real_)
  value <- suppressWarnings(as.numeric(strsplit(trimws(line[1]), "[[:space:]]+")[[1]][2]))
  if (is.na(value)) NA_real_ else value * 1024
}

CGROUP_MEMORY_FILES <- list(
  c("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", ""),
  c("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    "/sys/fs/cgroup/memory/memory.stat", "total_")
)

cgroup_memory_room_bytes <- function(files = CGROUP_MEMORY_FILES) {
  for (f in files) {
    raw <- trimws(paste(read_small_text(f[1]), collapse = ""))
    if (!grepl("^[0-9]+$", raw)) next  # absent, or "max": no limit at this level
    limit <- as.numeric(raw)
    if (!(limit > 0 && limit < 2^60)) next  # cgroup v1 reports "no limit" as a huge number
    stat <- list()
    for (parts in strsplit(trimws(read_small_text(f[3])), "[[:space:]]+")) {
      if (length(parts) == 2L && grepl("^[0-9]+$", parts[2])) stat[[parts[1]]] <- as.numeric(parts[2])
    }
    lru <- vapply(c("active_file", "inactive_file"), function(k) {
      v <- stat[[paste0(f[4], k)]]
      if (is.null(v)) v <- stat[[k]]
      if (is.null(v)) NA_real_ else v
    }, numeric(1))
    used <- trimws(paste(read_small_text(f[2]), collapse = ""))
    if (!grepl("^[0-9]+$", used) || all(is.na(lru))) return(limit)
    working_set <- max(as.numeric(used) - sum(lru, na.rm = TRUE), 0)
    return(max(limit - working_set, 0))
  }
  NA_real_
}

mem_available_bytes <- function() {
  found <- c(meminfo_available_bytes(), cgroup_memory_room_bytes())
  found <- found[!is.na(found)]
  if (length(found) == 0) NA_real_ else min(found)
}

check_lattice_memory <- function(n_spots, platform) {
  n_max <- max(n_spots)
  need <- 12 * as.numeric(n_max)^2
  avail <- mem_available_bytes()
  if (!is.na(avail) && need > avail) {
    stop("platform='", platform, "' builds its neighbour graph with PRECAST's find_neighbors, which ",
         "allocates a dense spots x spots matrix: for the largest sample (", n_max, " spots) that ",
         "is about ", round(need / 1024^3, 1), " GB, and ", round(avail / 1024^3, 1),
         " GB is available (MemAvailable, or the room left under the cgroup memory limit). ",
         "platform='Other_SRT' builds the graph sparsely by radius on the same slide.")
  }
  invisible(need)
}

# Spots with at least one neighbour, and the median neighbour count, per sample of the graph
# PRECAST will actually use.
neighbour_stats <- function(adj_list) {
  per <- lapply(adj_list, function(adj) {
    k <- as.numeric(Matrix::rowSums(adj != 0)) - as.numeric(Matrix::diag(adj) != 0)
    c(with = sum(k > 0), median = stats::median(k))
  })
  list(with = vapply(per, function(p) as.integer(p[["with"]]), integer(1)),
       median = vapply(per, function(p) as.numeric(p[["median"]]), numeric(1)))
}

# Which of `spots` a coordinates file marks as tissue. Space Ranger's positions files list every
# spot on the array with in_tissue 0/1, and CELLxGENE Visium exports carry the background spots in
# their counts too (56-70% of the spots on the library's four such samples), so without this the
# glass around the section was clustered as tissue. The rule is worker_utils.keep_in_tissue's:
# no in_tissue column keeps every spot; 1 / TRUE / "1" / "true" is tissue, anything else is not; a
# column that marks none of the spots as tissue is refused rather than clustered as empty.
in_tissue_mask <- function(coords, spots, source_path) {
  hit <- match("in_tissue", tolower(colnames(coords)))
  if (is.na(hit)) return(rep(TRUE, length(spots)))
  raw <- coords[spots, hit]
  flag <- tolower(trimws(as.character(raw)))
  flag[flag == "true"] <- "1"
  flag[flag == "false"] <- "0"
  keep <- suppressWarnings(as.numeric(flag)) %in% 1
  if (!any(keep)) {
    stop("The in_tissue column of ", source_path, " marks none of the ", length(spots),
         " spots shared with the counts as in tissue (values seen: ",
         paste(utils::head(sort(unique(as.character(raw))), 8), collapse = ", "),
         "); fix the column so in-tissue spots are 1, or remove it if every spot is tissue.")
  }
  keep
}

write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = FALSE, quote = TRUE)
  if (!file.rename(partial, path)) stop("Could not move ", partial, " into place at ", path)
  invisible(path)
}

save_rds_atomic <- function(obj, path) {
  partial <- paste0(path, ".partial")
  saveRDS(obj, file = partial)
  if (!file.rename(partial, path)) stop("Could not move ", partial, " into place at ", path)
  invisible(path)
}


parse_args <- function(args) {
  opts <- list(
    counts_csvs    = character(0),
    coords_csvs    = character(0),
    output_dir     = NULL,
    K              = 7L,
    platform       = "auto",
    gene_number    = 2000L,
    core_num       = 1L,
    max_iter       = 50L,
    sigma_equal    = FALSE,
    seed           = 0L,
    allow_fixed_number_fallback = FALSE
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key == "--sigma-equal") {
      opts$sigma_equal <- TRUE
      i <- i + 1L
      next
    }
    if (key == "--allow-fixed-number-fallback") {
      opts$allow_fixed_number_fallback <- TRUE
      i <- i + 1L
      next
    }
    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    if (key == "--counts-csv") {
      opts$counts_csvs <- c(opts$counts_csvs, val)
    } else if (key == "--coords-csv") {
      opts$coords_csvs <- c(opts$coords_csvs, val)
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--K") {
      opts$K <- as.integer(val)
    } else if (key == "--platform") {
      opts$platform <- val
    } else if (key == "--gene-number") {
      opts$gene_number <- as.integer(val)
    } else if (key == "--core-num") {
      opts$core_num <- as.integer(val)
    } else if (key == "--max-iter") {
      opts$max_iter <- as.integer(val)
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_precast <- function(opts) {
  if (is.null(opts$output_dir)) {
    stop("--output-dir is required")
  }
  if (length(opts$counts_csvs) == 0 || length(opts$coords_csvs) == 0) {
    stop("At least one --counts-csv and one --coords-csv are required")
  }
  if (length(opts$counts_csvs) != length(opts$coords_csvs)) {
    stop("Number of --counts-csv and --coords-csv must match")
  }
  platform_requested <- canonical_platform(opts$platform)

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  n_samples <- length(opts$counts_csvs)
  seuList <- list()
  warns <- character(0)

  # --- Coordinates first: the neighbour definition is one choice for every sample ---
  coords_list <- list()
  resolved <- list()
  lattice <- list()
  coords_header <- character(n_samples)
  for (s in seq_len(n_samples)) {
    log_msg("Reading sample ", s, " coordinates from: ", opts$coords_csvs[s])
    coords_read <- read_coords_csv(opts$coords_csvs[s])
    coords_list[[s]] <- coords_read$frame
    coords_header[s] <- coords_read$header
    resolved[[s]] <- resolve_coord_cols(colnames(coords_list[[s]]), opts$coords_csvs[s])
    check_coord_cols(resolved[[s]], colnames(coords_list[[s]]), opts$coords_csvs[s])
    lattice[s] <- list(lattice_coord_cols(colnames(coords_list[[s]])))
  }
  choice <- choose_platform(platform_requested, coords_list, resolved, lattice)
  platform <- choice$platform
  log_msg("Neighbour definition: ", platform, " (", choice$why, ")")

  n_spots_in_counts <- integer(n_samples)
  n_spots_without_coords <- integer(n_samples)
  n_spots_supplied <- integer(n_samples)
  n_spots_off_tissue <- integer(n_samples)
  for (s in seq_len(n_samples)) {
    log_msg("Reading sample ", s, " counts from: ", opts$counts_csvs[s])
    counts_df <- read.csv(opts$counts_csvs[s], row.names = 1, check.names = FALSE)
    counts <- as.matrix(counts_df)
    coords <- coords_list[[s]]

    # Ensure matching spot IDs
    common_spots <- intersect(colnames(counts), rownames(coords))
    if (length(common_spots) == 0) {
      counts <- t(counts)
      common_spots <- intersect(colnames(counts), rownames(coords))
    }
    if (length(common_spots) == 0) {
      stop(id_mismatch_msg("spot IDs", "counts", colnames(counts), "coordinates", rownames(coords)), paste0(" (sample ", s, ")"))
    }

    # A spot in the counts with no row in the coordinates file cannot be placed, so it is left out --
    # and counted, rather than vanishing from every count in the payload. (Rows of the coordinates
    # file with no counts, such as the in_tissue = 0 spots of a filtered matrix, are not the
    # caller's data.)
    n_spots_in_counts[s] <- ncol(counts)
    no_coords <- setdiff(colnames(counts), common_spots)
    n_spots_without_coords[s] <- length(no_coords)
    if (length(no_coords) > 0) {
      warns <- c(warns, paste0(
        length(no_coords), " of ", ncol(counts), " spots in the counts of sample ", s, " have no row in ",
        opts$coords_csvs[s], " and were left out (e.g. ",
        paste(utils::head(no_coords, 3), collapse = ", "), ")."))
      log_msg("WARNING: ", utils::tail(warns, 1))
    }

    # Background spots (in_tissue = 0) are not tissue: left out and reported.
    n_spots_supplied[s] <- length(common_spots)
    on_tissue <- in_tissue_mask(coords, common_spots, opts$coords_csvs[s])
    n_spots_off_tissue[s] <- sum(!on_tissue)
    if (n_spots_off_tissue[s] > 0) {
      common_spots <- common_spots[on_tissue]
      warns <- c(warns, paste0(
        n_spots_off_tissue[s], " of ", n_spots_supplied[s], " spots of sample ", s, " have in_tissue == 0 ",
        "in ", opts$coords_csvs[s], " (background outside the tissue) and were left out; ",
        length(common_spots), " in-tissue spots were analysed."))
      log_msg("WARNING: ", utils::tail(warns, 1))
    }

    counts <- counts[, common_spots, drop = FALSE]
    coord_cols <- choice$coord_cols[[s]]
    check_coord_cols(coord_cols, colnames(coords), opts$coords_csvs[s])
    log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
            " (of ", paste(colnames(coords), collapse = ", "), ")")
    coords <- coords[common_spots, coord_cols, drop = FALSE]
    xy <- coord_matrix(coords, opts$coords_csvs[s])

    # Refuse an empty lattice graph here, before SPARK-X and the dense upstream neighbour search.
    if (platform %in% LATTICE_PLATFORMS) {
      nb <- lattice_neighbour_counts(xy[, 1], xy[, 2], platform)
      if (stats::median(nb) == 0) {
        stop(no_neighbour_msg(platform, s, sum(nb > 0), length(nb), coord_cols, opts$coords_csvs[s]))
      }
    }

    # Sanitize feature names (Seurat does not allow underscores)
    rownames(counts) <- gsub("_", "-", rownames(counts))

    meta <- data.frame(row.names = common_spots, row = xy[, 1], col = xy[, 2])
    seuList[[s]] <- CreateSeuratObject(counts = counts, meta.data = meta)
    log_msg("Sample ", s, ": ", ncol(counts), " spots, ", nrow(counts), " genes")
  }

  # --- Create PRECAST object ---
  # CreatePRECASTObject selects gene_number spatially variable genes per sample with SPARK-X, keeps
  # the ones shared across samples, and filters: genes seen in fewer than 20 spots and spots with
  # fewer than 20 genes before the selection, 15 / 15 after it. When a sample has fewer SVGs than
  # gene_number it LOWERS gene_number, saying so only in an R warning. params.gene_number is the
  # request; what the fit used is read back off preobj@seulist and published beside it.
  log_msg("Creating PRECAST object (gene_number=", opts$gene_number, ")...")
  lowered <- character(0)
  preobj <- withCallingHandlers(
    CreatePRECASTObject(
      seuList,
      gene.number = opts$gene_number,
      rawData.preserve = FALSE
    ),
    warning = function(w) {
      msg <- conditionMessage(w)
      if (grepl("set minimum number of variable genes", msg, fixed = TRUE)) {
        lowered <<- c(lowered, msg)
        log_msg("CreatePRECASTObject: ", msg)
        invokeRestart("muffleWarning")
      }
    }
  )
  n_genes_read <- vapply(seuList, function(x) as.integer(nrow(x)), integer(1))
  n_genes_used <- vapply(preobj@seulist, function(x) as.integer(nrow(x)), integer(1))
  gene_number_selected <- opts$gene_number
  if (length(lowered) > 0) {
    lowered_to <- suppressWarnings(as.integer(sub(".*gene\\.number=([0-9]+).*", "\\1", lowered[1])))
    if (!is.na(lowered_to)) gene_number_selected <- lowered_to
    warns <- c(warns, paste0(
      "PRECAST lowered gene_number from ", opts$gene_number, " to ", gene_number_selected,
      ": SPARK-X found fewer spatially variable genes than requested in at least one sample ",
      "(CreatePRECASTObject: \"", lowered[1], "\"). The fit used ",
      paste(n_genes_used, collapse = ", "), " genes (params.gene_number_used)."))
  }

  if (platform %in% LATTICE_PLATFORMS) {
    check_lattice_memory(vapply(preobj@seulist, function(x) as.numeric(ncol(x)), numeric(1)), platform)
  }

  # DR.SC getAdj_auto (Other_SRT) stops with "The radius.upper is too smaller ..." when a radius
  # from a random 100-spot sample does not reach a median of 4 neighbours. The 6-nearest-neighbour
  # graph PRECAST also offers is a different neighbourhood definition, so it runs only when the
  # caller allowed it, and the payload says it ran.
  log_msg("Adding adjacency list (platform: ", platform, ")...")
  adjacency_type <- "fixed_distance"
  radius_error <- NULL
  adj_obj <- tryCatch({
    AddAdjList(preobj, platform = platform)
  }, error = function(e) {
    if (!grepl("radius.upper", conditionMessage(e), fixed = TRUE)) {
      stop(e)
    }
    radius_error <<- conditionMessage(e)
    NULL
  })
  if (is.null(adj_obj)) {
    if (!isTRUE(opts$allow_fixed_number_fallback)) {
      stop("PRECAST's radius search (platform='", platform, "') found no radius giving a median of ",
           "4 neighbours per spot: ", radius_error, " Pass allow_fixed_number_fallback=True to ",
           "build the graph from each spot's 6 nearest neighbours instead (reported as ",
           "params.adjacency_type='fixed_number'), or give a coordinates file carrying ",
           "array_row/array_col so the ST or Visium grid can be used.")
    }
    log_msg("Radius search failed (", radius_error, "); allow_fixed_number_fallback is set, so ",
            "building a 6-nearest-neighbour adjacency instead")
    adj_obj <- AddAdjList(preobj, platform = platform, type = "fixed_number")
    adjacency_type <- "fixed_number"
    warns <- c(warns, paste0("fallback ran: PRECAST with a 6-nearest-neighbour adjacency ",
                             "(type='fixed_number') -- the radius search failed: ", radius_error))
  }
  preobj <- adj_obj
  used_fallback <- adjacency_type == "fixed_number"

  adj_stats <- neighbour_stats(preobj@AdjList)
  for (s in seq_len(n_samples)) {
    log_msg("Sample ", s, ": ", adj_stats$with[s], " of ", ncol(preobj@seulist[[s]]),
            " spots have a neighbour; median ", adj_stats$median[s], " neighbours per spot")
  }
  empty <- which(adj_stats$median == 0)
  if (length(empty) > 0) {
    s <- empty[1]
    stop(no_neighbour_msg(platform, s, adj_stats$with[s], ncol(preobj@seulist[[s]]),
                          choice$coord_cols[[s]], opts$coords_csvs[s]))
  }

  # PRECAST's ICM.EM calls set.seed(parameterList$seed) right before its initialisation, and
  # model_set defaults that seed to 1 -- so without seed= here, set.seed(opts$seed) in main() never
  # reached the fit and every seed gave the seed-1 answer.
  log_msg("Setting parameters...")
  preobj <- AddParSetting(
    preobj,
    Sigma_equal = opts$sigma_equal,
    coreNum     = opts$core_num,
    maxIter     = opts$max_iter,
    seed        = opts$seed,
    verbose     = FALSE
  )

  # --- Run PRECAST ---
  log_msg("Running PRECAST with K=", opts$K, "...")
  preobj <- PRECAST(preobj, K = opts$K)

  # With one K, SelectModel only reshapes the fit into resList$cluster. An error here is PRECAST's
  # own and is reported as such: the old handler looked for a $bic no PRECAST fit carries, returned
  # the object unselected, and the run then died on a missing cluster vector far from the cause.
  log_msg("Selecting best model...")
  preobj <- SelectModel(preobj)

  # --- Extract results ---
  # The cluster vector is indexed against preobj@seulist and nothing else: inside PRECAST(),
  # XList <- lapply(PRECASTObj@seulist, get_norm_data) is what the model is fitted on.
  # @seulist is NOT the seuList we handed in. CreatePRECASTObject runs filter_spot four times
  # (premin.features / postmin.features), and filter_spot is a logical mask on nFeature, so the
  # spots it drops sit scattered through the ordering rather than in one trailing block. Naming
  # the labels off the unfiltered list and cutting it to length would therefore shift every
  # label from the first dropped spot onward -- a full-size, plausible clustering CSV with the
  # right barcodes and the right cluster values, wrongly paired. The two slots differ only in
  # the case of one letter, which is what makes the wrong one easy to reach for.
  cluster_list <- preobj@resList$cluster
  fitted_list <- preobj@seulist
  if (is.null(fitted_list) || length(fitted_list) != n_samples) {
    stop("PRECAST kept ", length(fitted_list), " filtered sample(s) for ", n_samples,
         " input sample(s), so its cluster labels cannot be matched to spot barcodes.")
  }
  all_spots <- character(0)
  all_clusters <- integer(0)
  all_samples <- character(0)
  cluster_summaries <- list()
  n_spots_clustered <- integer(n_samples)

  for (s in seq_len(n_samples)) {
    clusters_s <- cluster_list[[s]]
    spots_s <- colnames(fitted_list[[s]])

    n_clust <- length(clusters_s)
    n_spots <- length(spots_s)
    if (n_clust != n_spots) {
      stop("Sample ", s, ": PRECAST returned ", n_clust, " cluster labels for the ", n_spots,
           " spots it kept. Pairing them in order would mislabel spots, so no clustering is",
           " written.")
    }
    n_spots_clustered[s] <- n_spots
    if (n_spots < ncol(seuList[[s]])) {
      warns <- c(warns, paste0(
        "PRECAST's own quality filter (CreatePRECASTObject: spots with fewer than 20 detected genes ",
        "before gene selection, or fewer than 15 of the selected genes after it) left out ",
        ncol(seuList[[s]]) - n_spots, " of ", ncol(seuList[[s]]), " spots of sample ", s,
        "; they are not in precast_clusters.csv."))
    }

    all_spots <- c(all_spots, spots_s)
    all_clusters <- c(all_clusters, clusters_s)
    all_samples <- c(all_samples, rep(paste0("sample_", s), length(spots_s)))
    cluster_summaries[[paste0("sample_", s)]] <- as.list(table(clusters_s))
  }

  # Save cluster assignments
  result_df <- data.frame(
    sample  = all_samples,
    spot    = all_spots,
    cluster = all_clusters,
    stringsAsFactors = FALSE
  )
  clusters_path <- file.path(opts$output_dir, "precast_clusters.csv")
  write_csv_atomic(result_df, clusters_path)

  # Save PRECAST object
  rds_path <- file.path(opts$output_dir, "precast_object.rds")
  save_rds_atomic(preobj, rds_path)

  n_found <- length(unique(all_clusters))
  res <- list(
    status       = "ok",
    tool         = "PRECAST",
    task         = "spatial_clustering_and_integration",
    data         = list(
      n_samples        = n_samples,
      # The spots that are actually in precast_clusters.csv. seuList still holds every spot we
      # read in, including the ones CreatePRECASTObject filtered out and never clustered.
      n_spots          = n_spots_clustered,
      n_spots_read     = sapply(seuList, ncol),
      # Spots of the counts with no coordinates row, and in-tissue spots removed by PRECAST's QC.
      n_spots_in_counts           = n_spots_in_counts,
      n_spots_without_coordinates = n_spots_without_coords,
      n_spots_dropped_by_qc       = sapply(seuList, ncol) - n_spots_clustered,
      # Genes per sample: read from the counts, and fitted on (after SPARK-X selection and QC).
      n_genes_read     = n_genes_read,
      n_genes_used     = n_genes_used,
      K                = opts$K,
      n_clusters_found = n_found,
      # The graph the spatial prior was fitted on, per sample.
      n_spots_with_neighbors = adj_stats$with,
      median_neighbors       = adj_stats$median
    ),
    output_files = list(
      clusters_csv     = clusters_path,
      precast_rds      = rds_path
    ),
    params       = list(
      method           = if (used_fallback) "PRECAST (6-nearest-neighbour adjacency)" else "PRECAST",
      used_fallback    = used_fallback,
      platform         = platform,
      platform_requested = platform_requested,
      coord_columns    = choice$coord_cols,
      adjacency_type   = adjacency_type,
      allow_fixed_number_fallback = isTRUE(opts$allow_fixed_number_fallback),
      gene_number      = opts$gene_number,
      gene_number_used = n_genes_used,
      gene_selection   = "SPARK-X (CreatePRECASTObject), genes shared across samples",
      coords_header    = coords_header,
      core_num         = opts$core_num,
      max_iter         = opts$max_iter,
      sigma_equal      = opts$sigma_equal,
      seed             = opts$seed
    ),
    summary      = list(
      cluster_distribution = cluster_summaries
    ),
    analysis     = paste0(
      "PRECAST ", if (n_samples == 1) "clustered 1 sample" else paste0("integrated ", n_samples, " samples"),
      " (", sum(n_spots_clustered), " spots) and identified ", n_found,
      " spatial domains with K=", opts$K, ". Neighbours: platform ", platform, " on ",
      paste(unique(vapply(choice$coord_cols, paste, character(1), collapse = "/")), collapse = "; "),
      if (adjacency_type == "fixed_number") ", 6 nearest neighbours (radius search failed)" else "",
      ", median ", paste(adj_stats$median, collapse = ", "), " neighbours per spot (",
      choice$why, "). Genes: SPARK-X selected up to ", gene_number_selected,
      " spatially variable genes per sample",
      if (gene_number_selected != opts$gene_number) paste0(" (gene_number=", opts$gene_number, " requested)") else "",
      ", and PRECAST fitted on ",
      paste(sprintf("%d of %d", n_genes_used, n_genes_read), collapse = ", "), " genes read",
      if (n_samples > 1) " (per sample)" else "", ".",
      if (any(n_spots_off_tissue > 0)) paste0(
        " ", sum(n_spots_off_tissue), " of ", sum(n_spots_supplied),
        " spots had in_tissue == 0 (background) and were left out.") else "",
      if (any(sapply(seuList, ncol) > n_spots_clustered)) paste0(
        " PRECAST's quality filter left out ", sum(sapply(seuList, ncol) - n_spots_clustered), " of ",
        sum(sapply(seuList, ncol)), " spots it was handed.") else ""
    )
  )
  if (any(n_spots_off_tissue > 0)) {
    res$params$in_tissue_filter <- list(
      n_spots_supplied            = n_spots_supplied,
      n_spots_off_tissue_dropped  = n_spots_off_tissue,
      n_spots_used                = n_spots_supplied - n_spots_off_tissue
    )
  }
  if (length(warns) > 0) res$warnings <- as.list(warns)
  res
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  set.seed(opts$seed)

  res <- tryCatch(with_r_traceback({
    run_precast(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "PRECAST",
      task      = "spatial_clustering_and_integration",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
