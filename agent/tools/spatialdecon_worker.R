#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SpatialDecon)
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
  message(sprintf("[spatialdecon-worker] %s", msg))
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

# A reference cell whose label is NA, empty, or a spelling of "missing" is not a cell type. With the
# annotations read as character, the text NA arrives as NA and the empty field pandas writes for NaN as
# "", and either one killed the profile build with a bare "subscript out of bounds" from rowMeans.
# The spellings match worker_utils.drop_unlabeled.
is_missing_label <- function(x) {
  s <- trimws(tolower(as.character(x)))
  is.na(x) | is.na(s) | s %in% c("", "nan", "none", "na", "<na>")
}

# SpatialDecon itself takes a genes x cell-types profile matrix, and a profile matrix is what this
# portal once advertised for ref_counts_csv. This worker builds the profile from cell-level counts, so
# a profile matrix handed in fails the cell-ID match. When its columns (or rows) are the annotation's
# cell-type names rather than its barcodes, say that, instead of the generic mismatch message.
refuse_profile_matrix <- function(ref_mat, cell_types) {
  type_names <- unique(as.character(cell_types[!is_missing_label(cell_types)]))
  for (axis in list(list(ids = colnames(ref_mat), name = "columns"), list(ids = rownames(ref_mat), name = "rows"))) {
    hits <- intersect(axis$ids, type_names)
    if (length(hits) > 0 && length(hits) >= length(axis$ids) / 2) {
      stop(paste0(
        "ref_counts_csv looks like a cell-type profile matrix: ", length(hits), " of its ",
        length(axis$ids), " ", axis$name, " are cell-type names from ref_celltypes_csv (e.g. \"",
        hits[[1]], "\"), not cell barcodes. This tool needs cell-level reference counts ",
        "(genes x cells, one column per cell barcode listed in ref_celltypes_csv) and builds the ",
        "profile itself as the per-cell-type mean."
      ))
    }
  }
  invisible(NULL)
}

# Read a counts CSV straight into a sparse dgCMatrix, one block of rows at a time. A copy of
# tools/spacexr_worker.R's read_counts_csv_sparse (each R worker runs in its own env, so each carries
# its own), with this tool's name in the refusals.
#
# read.csv -> as.matrix held the whole table as a data.frame and then as a dense double matrix, and the
# column subset and (for a cells x genes file) the transpose made more full-size copies, all alive at
# once -- ~40 GB each for the library's Colon reference (279,609 cells x 18,082 genes) -- only to take one
# per-cell-type mean. Here only one block (~block_bytes of doubles) is ever dense; what accumulates is the
# non-zero triplets.
#
# Same reading rules as the read.csv(row.names = 1, check.names = FALSE) it replaces: first field of every
# row is the row name, header names are kept verbatim, "NA" is missing, a header with one field fewer than
# the rows names only the data columns. file() opens .gz/.bz2/.xz as well.
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
    stop(sprintf("%s CSV %s holds %.0f missing (NA/empty) values; SpatialDecon needs a value in every cell",
                 what, path, n_missing))
  }
  if (n_negative > 0) {
    stop(sprintf(paste0("%s CSV %s holds %.0f negative values; SpatialDecon's log-normal model and the ",
                        "per-cell-type mean profile take non-negative expression (counts or normalised ",
                        "counts), never scaled or centred data"), what, path, n_negative))
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

# The per-cell-type mean of the reference cells' counts, as the genes x cell-types profile matrix
# spatialdecon() takes as X -- the numbers sapply(types, function(t) rowMeans(ref[, cells of t])) gave,
# computed as one sparse product (counts %*% membership / cells per type) so the reference is never
# dense. `cell_types` is named by cell barcode; every name is a column of `counts`. Columns in the order
# of `types`.
cell_type_mean_profile <- function(counts, cell_types, types) {
  membership <- sparseMatrix(
    i = match(names(cell_types), colnames(counts)),
    j = match(cell_types, types),
    x = 1,
    dims = c(ncol(counts), length(types))
  )
  n_cells <- as.numeric(table(factor(cell_types, levels = types)))
  profile <- as.matrix(counts %*% membership)
  profile <- sweep(profile, 2, n_cells, "/")
  dimnames(profile) <- list(rownames(counts), types)
  profile
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

# spatialdecon() takes the spots as a base (dense) matrix -- it stops on anything else -- and fits them
# with its log-normal regression, so the shared-genes x spots matrix is dense by necessity. Alive at once
# while it fits: this worker's counts, normalised and background matrices, and inside deconLNR the
# rbind(Y, bg, weights) it applies over the spots, three more of the same size. Six copies is a lower
# bound (spatialdecon's own gene subsets and its residual matrices come on top), so a run it refuses could
# not have finished; the alternative is the kernel's OOM killer and no JSON at all.
SPATIAL_DENSE_COPIES <- 6L

check_spatial_memory <- function(n_genes, n_spots, avail = available_memory_bytes()) {
  one <- 8 * as.numeric(n_genes) * as.numeric(n_spots)
  need <- SPATIAL_DENSE_COPIES * one
  if (!is.na(avail) && need > avail) {
    stop(sprintf(paste0(
      "SpatialDecon fits the %.0f shared genes x %.0f spots matrix dense (spatialdecon() takes a base ",
      "matrix, and rbind()s it with the background and the weights): %.2f GB per copy, and at least %d ",
      "copies are alive at once, so at least %.2f GB, while %.2f GB is available here (the smaller of ",
      "MemAvailable and the room under the cgroup memory limit, page cache counted as free). The dense ",
      "matrix is intrinsic to spatialdecon() and no parameter of this tool makes it smaller; run it where ",
      "at least %.2f GB is free."),
      n_genes, n_spots, one / 1e9, SPATIAL_DENSE_COPIES, need / 1e9, avail / 1e9, need / 1e9))
  }
  need
}

# Write to <path>.partial and move it into place, so a run killed mid-write never leaves a truncated
# table under the name a reader trusts (rename(2) is atomic on one filesystem).
write_atomically <- function(path, writer) {
  tmp <- paste0(path, ".partial")
  done <- FALSE
  on.exit(if (!done && file.exists(tmp)) unlink(tmp), add = TRUE)
  writer(tmp)
  if (!file.rename(tmp, path)) stop("Could not move ", tmp, " into place at ", path)
  done <- TRUE
  invisible(path)
}

parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    normalize          = TRUE,
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
    } else if (key == "--ref-counts-csv") {
      opts$ref_counts_csv <- val
    } else if (key == "--ref-celltypes-csv") {
      opts$ref_celltypes_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--normalize") {
      opts$normalize <- tolower(val) %in% c("true", "1", "yes")
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- tolower(val) %in% c("true", "1", "yes")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_spatialdecon <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$ref_counts_csv) ||
      is.null(opts$ref_celltypes_csv) || is.null(opts$output_dir)) {
    stop("SpatialDecon requires --spatial-counts-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$ref_counts_csv, opts$ref_celltypes_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial counts ---
  # Sparse, like the reference below. spatialdecon() needs the spots dense, but only on the genes it
  # shares with the reference, so the matrix is made dense once those are known and not before.
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_mat <- read_counts_csv_sparse(opts$spatial_counts_csv, "spatial_counts_csv")

  # SpatialDecon reads genes x spots. Which axis holds the genes is settled below, once the
  # reference profile's own gene names exist to compare against -- the same evidence the cell
  # orientation uses. Neither the first letter of a row name nor the larger of the two dimensions
  # answers it: a Visium barcode starts with a letter too, and the shape rule is right only where
  # the spots outnumber the genes. The MERFISH panel this repo stages is 649 genes x 78,329 cells.

  # --- Load reference and build profile matrix ---
  # spatialdecon() never sees the reference cells: it takes the genes x cell-types profile X. So the
  # cells are read sparse and reduced to that profile without ever being dense.
  log_msg("Reading reference counts from: ", opts$ref_counts_csv)
  ref_counts_mat <- read_counts_csv_sparse(opts$ref_counts_csv, "ref_counts_csv")

  log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_raw <- read.csv(opts$ref_celltypes_csv, check.names = FALSE, colClasses = "character")
  ref_ct_df <- data.frame(celltype = ref_ct_raw[, 2], row.names = ref_ct_raw[, 1])
  cell_types <- as.character(ref_ct_df[, 1])
  names(cell_types) <- rownames(ref_ct_df)

  # Ensure genes x cells orientation
  common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  if (length(common_cells) == 0) {
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  }
  if (length(common_cells) == 0) {
    # Transposed back, so "columns" and "rows" in the message are the file's own.
    refuse_profile_matrix(t(ref_counts_mat), cell_types)
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(cell_types)))}

  cell_types <- cell_types[common_cells]

  warnings <- character(0)
  unlabeled <- is_missing_label(cell_types)
  n_unlabeled <- sum(unlabeled)
  if (n_unlabeled > 0) {
    if (!isTRUE(opts$drop_unlabeled)) {
      stop(sprintf(paste0("%d of %d reference cells have no cell-type label (NA/empty) in ref_celltypes_csv. ",
                          "Pass drop_unlabeled=True to leave them out, or label them first; a missing label ",
                          "is not a class."), n_unlabeled, length(cell_types)))
    }
    cell_types <- cell_types[!unlabeled]
    warnings <- c(warnings, sprintf(paste0("%d of %d reference cells had no cell-type label and were left out ",
                                           "(drop_unlabeled=True); %d were used."),
                                    n_unlabeled, length(common_cells), length(cell_types)))
    if (length(cell_types) == 0) stop("Every reference cell is unlabeled; there is nothing to build a profile from.")
  }

  log_msg("Reference: ", length(cell_types), " cells, ",
          length(unique(cell_types)), " cell types")

  # Build profile matrix: average expression per cell type (genes x cell_types), as one sparse
  # product over the labelled cells; the cells are then released.
  log_msg("Building reference profile matrix...")
  unique_types <- sort(unique(cell_types))
  profile_mat <- cell_type_mean_profile(ref_counts_mat, cell_types, unique_types)
  rm(ref_counts_mat)

  # --- Orient the spatial matrix against the reference's genes, then intersect ---
  common_genes <- intersect(rownames(sp_counts_mat), rownames(profile_mat))
  if (length(common_genes) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_genes <- intersect(rownames(sp_counts_mat), rownames(profile_mat))
  }
  if (length(common_genes) == 0) {
    stop(id_mismatch_msg("gene IDs", "spatial counts", rownames(sp_counts_mat),
                         "reference profile", rownames(profile_mat)))
  }

  n_genes_spatial <- nrow(sp_counts_mat)
  n_spots <- ncol(sp_counts_mat)
  log_msg("Spatial data: ", n_genes_spatial, " genes x ", n_spots, " spots")

  # A handful of shared genes is a different complaint from a wrong axis: the two identifier sets
  # do meet, there is just too little overlap to regress on.
  if (length(common_genes) < 10) {
    stop(sprintf("Only %d common genes between spatial and reference data", length(common_genes)))
  }
  log_msg("Common genes: ", length(common_genes))

  # The one dense copy of the spots this worker makes, on the shared genes only; checked against the
  # memory spatialdecon() will need before it is allocated.
  need <- check_spatial_memory(length(common_genes), n_spots)
  log_msg(sprintf("SpatialDecon fits %d genes x %d spots dense: at least %.2f GB", length(common_genes),
                  n_spots, need / 1e9))
  sp_counts_mat <- as.matrix(sp_counts_mat[common_genes, , drop = FALSE])
  profile_mat   <- profile_mat[common_genes, , drop = FALSE]

  # A spot with no counts on the shared genes carries no information, but spatialdecon() still fits
  # it -- to its own lower threshold -- and returns a near-uniform composition, which the proportions
  # CSV would present like any other. Counted and named rather than passed off as an estimate.
  empty_spots <- colnames(sp_counts_mat)[colSums(sp_counts_mat) == 0]
  n_empty_spots <- length(empty_spots)
  if (n_empty_spots > 0) {
    warnings <- c(warnings, sprintf(paste0("%d of %d spots have no counts on the %d shared genes (e.g. \"%s\"); ",
                                           "SpatialDecon still fits them, to its lower threshold, so their rows ",
                                           "in spatialdecon_proportions.csv are not informed by any data."),
                                    n_empty_spots, ncol(sp_counts_mat), length(common_genes), empty_spots[[1]]))
  }

  # --- Normalize spatial counts if requested ---
  if (opts$normalize) {
    log_msg("Normalizing spatial counts (library size normalization)...")
    lib_sizes <- colSums(sp_counts_mat)
    lib_sizes[lib_sizes == 0] <- 1
    norm_mat <- sweep(sp_counts_mat, 2, lib_sizes, "/") * median(lib_sizes)
  } else {
    norm_mat <- sp_counts_mat
  }

  # --- Background ---
  # A constant background, always. The worker used to try derive_GeoMx_background(negnames =
  # character(0)) first, which stops on every input ("probe pool didn't have any negprobes
  # specified") because only GeoMx data carries negative probes, and then logged a "fallback" to
  # this same constant. It was never a fallback: it is the only background this wrapper has.
  background <- 0.1
  log_msg("Using a constant background of ", background, " (no negative probes outside GeoMx)")
  bg <- matrix(background, nrow = nrow(norm_mat), ncol = ncol(norm_mat),
               dimnames = list(rownames(norm_mat), colnames(norm_mat)))

  # --- Run SpatialDecon ---
  log_msg("Running SpatialDecon...")
  decon_result <- spatialdecon(
    norm        = norm_mat,
    bg          = bg,
    X           = profile_mat,
    align_genes = TRUE
  )

  # --- Extract proportions ---
  log_msg("Extracting cell type proportions...")
  # beta contains cell type abundances (cell_types x spots)
  beta_mat <- decon_result$beta
  # Normalize to proportions. A spot whose every abundance came back zero has no composition to
  # normalise; it keeps a row of zeros in the CSV, and the payload counts those rows rather than
  # letting them pass for proportions.
  col_sums <- colSums(beta_mat)
  zero_spots <- colnames(beta_mat)[col_sums == 0]
  n_zero_spots <- length(zero_spots)
  if (n_zero_spots > 0) {
    warnings <- c(warnings, sprintf(paste0("%d of %d spots got no estimate from SpatialDecon (every cell-type ",
                                           "abundance was zero; e.g. \"%s\"); their rows in ",
                                           "spatialdecon_proportions.csv are all zeros, not a composition."),
                                    n_zero_spots, ncol(beta_mat), zero_spots[[1]]))
  }
  col_sums[col_sums == 0] <- 1
  prop_mat <- sweep(beta_mat, 2, col_sums, "/")
  prop_df <- as.data.frame(t(prop_mat))
  prop_df$spot <- rownames(prop_df)

  # --- Save outputs ---
  prop_path <- file.path(opts$output_dir, "spatialdecon_proportions.csv")
  write_atomically(prop_path, function(tmp) write.csv(prop_df, tmp, row.names = FALSE, quote = TRUE))

  rds_path <- file.path(opts$output_dir, "spatialdecon_result.rds")
  write_atomically(rds_path, function(tmp) saveRDS(decon_result, file = tmp))

  log_msg("Saved proportions to: ", prop_path)
  log_msg("Saved result object to: ", rds_path)

  # --- Summary ---
  # prop_mat is cell_types (rows) x spots (cols)
  cell_types_found <- rownames(prop_mat)
  n_celltypes <- length(cell_types_found)

  # Determine dominant cell type per spot (iterate over columns = spots)
  dominant <- apply(prop_mat, 2, function(col) rownames(prop_mat)[which.max(col)])
  dominant_counts <- as.list(table(dominant))

  result <- list(
    status       = "ok",
    tool         = "spatialdecon",
    task         = "deconvolution",
    data         = list(
      n_spots          = n_spots,
      n_genes_spatial  = n_genes_spatial,
      n_common_genes   = length(common_genes),
      n_ref_cells      = length(common_cells),
      n_ref_cells_used = length(cell_types),
      n_ref_cells_unlabeled = n_unlabeled,
      n_ref_types      = length(unique_types),
      n_spots_without_counts   = n_empty_spots,
      n_spots_without_estimate = n_zero_spots
    ),
    output_files = list(
      proportions_csv  = prop_path,
      result_rds       = rds_path
    ),
    params       = list(
      normalize        = opts$normalize,
      drop_unlabeled   = isTRUE(opts$drop_unlabeled),
      background       = background,
      method           = paste0("SpatialDecon spatialdecon() constrained log-normal regression against ",
                                "per-cell-type mean counts of the reference cells, constant background ",
                                background),
      used_fallback    = FALSE
    ),
    summary      = list(
      n_cell_types     = n_celltypes,
      cell_types_found = cell_types_found,
      dominant_counts  = dominant_counts
    ),
    analysis     = paste0(
      "SpatialDecon estimated proportions for ", n_celltypes,
      " cell types across ", n_spots, " spatial spots using ",
      "constrained log-normal regression with ", length(common_genes),
      " genes, against a profile built from ", length(cell_types),
      " reference cells, with a constant background of ", background, ".",
      if (n_unlabeled > 0) sprintf(" %d unlabeled reference cells were left out (drop_unlabeled=True).",
                                   n_unlabeled) else "",
      if (n_empty_spots > 0) sprintf(paste0(" %d of %d spots have no counts on the shared genes; their ",
                                            "proportions are not informed by any data."),
                                     n_empty_spots, n_spots) else "",
      if (n_zero_spots > 0) sprintf(paste0(" %d of %d spots got no estimate (every abundance was zero) and are ",
                                           "all-zero rows in spatialdecon_proportions.csv."),
                                    n_zero_spots, n_spots) else ""
    )
  )
  # Only when there is something to say, as WorkerOutput does: a warning on a clean run trains the
  # reader to skip the one that matters.
  if (length(warnings) > 0) result$warnings <- I(warnings)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args inside the handler: a bad flag used to abort R with nothing on stdout.
  res <- tryCatch(with_r_traceback({
    sink(stderr())
    opts <- parse_args(args)
    result <- run_spatialdecon(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "spatialdecon",
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
