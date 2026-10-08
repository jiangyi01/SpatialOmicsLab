#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Neighborseq)
  library(Matrix)
  library(Seurat)
  library(dplyr)
  library(rlang)
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
  message(sprintf("[neighborseq-worker] %s", msg))
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

# ---------------------------------------------------------------------------------------------
# What runs, and what nothing here can change.
#
# Neighborseq 0.1.0 exposes neighborseq(cell.mat, celltypes, sample, iter, exclude,
# multiplet.degree, n.am, f, nsim, do.mroc, homotypic) and nothing else. Its classifier is
# multiplet_rf(), which hard-codes xgboost(nround = 1, max_depth = 20, num_parallel_tree = 200);
# no argument of neighborseq() reaches the tree count, and no step applies a p-value cutoff (every
# tested pair is returned with its pval and padj). So this tool's n_trees and pval_cutoff are
# accepted, echoed, and listed under params.ignored -- they used to be echoed as if they had run.
# ---------------------------------------------------------------------------------------------
METHOD_NAME <- paste0(
  "Neighbor-seq (Neighborseq::neighborseq: xgboost random-forest classifier trained on artificial ",
  "doublets, then a multiplet-enrichment Wilcoxon test)"
)
IGNORED_PARAMS <- c("n_trees", "pval_cutoff")
IGNORED_WHY <- paste0(
  "Neighborseq takes neither -- multiplet_rf fixes the forest at xgboost num_parallel_tree = 200 ",
  "with one boosting round, and neighborseq() applies no p-value cutoff (every tested pair is ",
  "written with its pval and padj)"
)
UPSTREAM_NUM_PARALLEL_TREE <- 200L
UPSTREAM_NROUND <- 1L
UPSTREAM_MAX_DEPTH <- 20L
UPSTREAM_TRAIN_FRACTION <- 0.8
UPSTREAM_MULTIPLET_DEGREE <- 2L

# Passed to neighborseq() by this worker (its own defaults, stated rather than implied).
NEIGHBORSEQ_ITER <- 1L
NEIGHBORSEQ_N_AM <- 100L
NEIGHBORSEQ_NSIM <- 100L

# Marker panel. FindAllMarkers runs on every cell: the worker used to pass
# max.cells.per.ident = 200 (upstream prep_cell_mat's max.cells default), which drew a random 200
# cells per cluster before choosing the genes and said so nowhere. No cell is subsampled now.
MARKER_TOP_N <- 50L
MARKER_LOGFC_THRESHOLD <- 0.25
MARKER_MIN_PCT <- 0.1
HVG_PANEL_SIZE <- 100L
MIN_PANEL_GENES <- 10L
HVG_PADDING <- 50L

# Dense copies of the artificial-multiplet table held at once (the per-row list, its rbindlist,
# the train/test split and the numeric training matrix xgboost is handed).
AM_DENSE_COPIES <- 4

parse_flag <- function(val, key) {
  low <- tolower(val)
  if (low %in% c("true", "1", "yes")) return(TRUE)
  if (low %in% c("false", "0", "no")) return(FALSE)
  stop(sprintf("%s expects true or false, got '%s'", key, val))
}

parse_int <- function(val, key, min_value = NULL) {
  parsed <- suppressWarnings(as.integer(val))
  if (is.na(parsed) || (!is.null(min_value) && parsed < min_value)) {
    stop(sprintf("%s expects an integer%s, got '%s'", key,
                 if (is.null(min_value)) "" else sprintf(" >= %d", min_value), val))
  }
  parsed
}

parse_num <- function(val, key) {
  parsed <- suppressWarnings(as.numeric(val))
  if (is.na(parsed)) stop(sprintf("%s expects a number, got '%s'", key, val))
  parsed
}

parse_args <- function(args) {
  opts <- list(
    counts_csv               = NULL,
    clusters_csv             = NULL,
    output_dir               = NULL,
    n_trees                  = 500L,
    n_top                    = 20L,
    pval_cutoff              = 0.05,
    seed                     = 0L,
    allow_hvg_panel_fallback = FALSE,
    drop_unlabeled           = FALSE,
    drop_singleton_clusters  = FALSE
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
    } else if (key == "--clusters-csv") {
      opts$clusters_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-trees") {
      opts$n_trees <- parse_int(val, key)
    } else if (key == "--n-top") {
      opts$n_top <- parse_int(val, key, min_value = 0L)
    } else if (key == "--pval-cutoff") {
      opts$pval_cutoff <- parse_num(val, key)
    } else if (key == "--seed") {
      opts$seed <- parse_int(val, key)
    } else if (key == "--allow-hvg-panel-fallback") {
      opts$allow_hvg_panel_fallback <- parse_flag(val, key)
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- parse_flag(val, key)
    } else if (key == "--drop-singleton-clusters") {
      opts$drop_singleton_clusters <- parse_flag(val, key)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# A reader that opens the file while it is being written sees either the previous version or the
# whole new one, never a truncated table (rename(2) is atomic on one filesystem).
write_csv_atomic <- function(df, path, row.names = TRUE) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = row.names, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
  invisible(path)
}

# The same rule as worker_utils.drop_unlabeled: NA / "" / "nan" / "none" / "na" is not a class.
is_missing_label <- function(x) {
  chr <- trimws(as.character(x))
  out <- is.na(x) | is.na(chr) | !nzchar(chr) | tolower(chr) %in% c("nan", "none", "na")
  out[is.na(out)] <- TRUE
  out
}

# Neighborseq names its classes by pasting cell types with "_" and takes them apart again with
# tidyr::separate (default sep "[^[:alnum:]]+") and stringr::str_which (a regex substring match).
# A label with a space, underscore or any punctuation -- every CELLxGENE label: "T cell",
# "cardiac muscle cell" -- is split into fragments ("T", "cell"), and wilcox.test then dies on
# mu = NA; a numeric label is found inside another ("1" inside "10_11"), which inflates the edge
# totals the enrichment score divides by. So Neighborseq never sees the caller's labels: each
# becomes a fixed-width, letters-only token ("ctaa", "ctab", ...). Equal length and no separator
# means no token can be split, and none can match inside another. Every output is mapped back to
# the original names, and the mapping is written to neighborseq_label_map.csv.
label_tokens <- function(n) {
  width <- 2L
  while (26^width < n) width <- width + 1L
  vapply(seq_len(n) - 1L, function(i) {
    chars <- character(width)
    for (p in width:1) {
      chars[p] <- letters[(i %% 26L) + 1L]
      i <- i %/% 26L
    }
    paste0("ct", paste(chars, collapse = ""))
  }, character(1))
}

# "ctaa" -> "B cell"; "ctaa_ctab" -> "B cell_T cell" (the shape Neighborseq's own names had).
decode_class_names <- function(x, tok2lab) {
  vapply(strsplit(as.character(x), "_", fixed = TRUE), function(parts) {
    lab <- tok2lab[parts]
    if (anyNA(lab)) {
      stop("Neighborseq returned a class name that is not one of this run's label tokens: ",
           paste(parts, collapse = "_"))
    }
    paste(unname(lab), collapse = "_")
  }, character(1))
}

# Memory this process can still allocate, in bytes, or NA when nothing can be read: the smaller of
# MemAvailable and the room under the cgroup memory limit, with the cgroup's page cache counted as
# reclaimable -- the rule of worker_utils.available_memory_bytes, which the Python workers share. This
# used to take the cgroup LIMIT itself as the room, so a container already using most of its limit
# passed the preflight and was OOM-killed building the table the preflight exists to budget.
CGROUP_MEMORY_FILES <- list(
  c("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", ""),
  c("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    "/sys/fs/cgroup/memory/memory.stat", "total_")
)

available_memory_bytes <- function(meminfo_path = "/proc/meminfo", cgroup_files = CGROUP_MEMORY_FILES) {
  found <- numeric(0)
  read_first <- function(path, n = -1L) {
    if (!file.exists(path)) return(character(0))
    tryCatch(suppressWarnings(readLines(path, n = n, warn = FALSE)), error = function(e) character(0))
  }
  meminfo <- read_first(meminfo_path)
  line <- grep("^MemAvailable:", meminfo, value = TRUE)
  if (length(line) > 0) {
    kb <- suppressWarnings(as.numeric(strsplit(trimws(sub("^MemAvailable:", "", line[[1]])), "\\s+")[[1]][[1]]))
    if (!is.na(kb)) found <- c(found, kb * 1024)
  }
  for (f in cgroup_files) {
    raw <- trimws(read_first(f[[1]], n = 1L))
    if (length(raw) != 1L || !grepl("^[0-9]+$", raw)) next  # absent, or "max": no limit at this level
    limit <- as.numeric(raw)
    if (!(limit > 0 && limit < 2^60)) next
    stat <- strsplit(trimws(read_first(f[[3]])), "\\s+")
    stat <- stat[lengths(stat) == 2L]
    values <- stats::setNames(suppressWarnings(as.numeric(vapply(stat, `[[`, "", 2L))), vapply(stat, `[[`, "", 1L))
    lru <- vapply(c("active_file", "inactive_file"), function(k) {
      v <- values[paste0(f[[4]], k)]
      if (is.na(v)) v <- values[k]
      unname(v)
    }, numeric(1))
    used <- trimws(read_first(f[[2]], n = 1L))
    if (length(used) != 1L || !grepl("^[0-9]+$", used) || all(is.na(lru))) {
      found <- c(found, limit)
    } else {
      working_set <- max(as.numeric(used) - sum(lru, na.rm = TRUE), 0)
      found <- c(found, max(limit - working_set, 0))
    }
    break
  }
  if (length(found) == 0) NA_real_ else min(found)
}

# The counts CSV is read in blocks of rows straight into a sparse matrix. read.csv + as.matrix held
# the whole table dense twice before Matrix() made it sparse -- about 73 GB per copy for a converter
# export of Visium HD (18,085 x 507,684) -- when Neighborseq itself only ever needs the marker-panel
# rows dense. Only one block (about COUNTS_BLOCK_VALUES values) is dense at a time; what is kept is
# the non-zero entries. The table must be comma-separated with a header row, as read.csv read it: the
# header may or may not carry a label for the row-name column.
COUNTS_BLOCK_VALUES <- 1e7

csv_fields <- function(lines) {
  scan(text = lines, what = "", sep = ",", quote = "\"", quiet = TRUE, na.strings = character(0),
       strip.white = FALSE, blank.lines.skip = TRUE, comment.char = "")
}

read_counts_sparse <- function(path, block_values = COUNTS_BLOCK_VALUES) {
  con <- file(path, "r")
  on.exit(close(con))
  header <- character(0)
  repeat {  # the first non-empty line, as read.csv takes it
    line <- readLines(con, n = 1L, warn = FALSE)
    if (length(line) == 0L) stop("counts_csv ", path, " is empty: it holds no header and no row.")
    if (nzchar(line)) {
      header <- csv_fields(line)
      break
    }
  }
  n_fields <- NA_integer_
  col_names <- NULL
  block_rows <- max(1L, as.integer(block_values %/% max(length(header), 1L)))
  row_names <- list()
  i_parts <- list()
  j_parts <- list()
  x_parts <- list()
  n_rows <- 0L
  repeat {
    lines <- readLines(con, n = block_rows, warn = FALSE)
    if (length(lines) == 0L) break
    lines <- lines[nzchar(lines)]
    if (length(lines) == 0L) next
    if (is.na(n_fields)) {
      n_fields <- length(csv_fields(lines[[1]]))
      if (n_fields == length(header)) {
        col_names <- header[-1]
      } else if (n_fields == length(header) + 1L) {
        col_names <- header
      } else {
        stop("counts_csv ", path, ": its first row has ", n_fields, " fields and its header ", length(header),
             "; a counts table has one row-name field and then one value per header column.")
      }
      if (n_fields < 2L) stop("counts_csv ", path, " has no value column after its row names.")
    }
    fields <- csv_fields(lines)
    if (length(fields) != length(lines) * n_fields) {
      counts <- utils::count.fields(textConnection(lines), sep = ",", quote = "\"", blank.lines.skip = TRUE,
                                    comment.char = "")
      bad <- which(counts != n_fields)[1]
      stop("counts_csv ", path, ": row ", n_rows + bad, " has ", counts[bad], " fields where the first row has ",
           n_fields, "; every row needs the same number.")
    }
    block <- matrix(fields, nrow = length(lines), ncol = n_fields, byrow = TRUE)
    rm(fields)
    row_names[[length(row_names) + 1L]] <- block[, 1]
    raw <- block[, -1, drop = FALSE]
    rm(block)
    values <- suppressWarnings(as.numeric(raw))
    na_at <- which(is.na(values))
    if (length(na_at) > 0L) {
      # "NA" and an empty field are missing values, as read.csv read them; anything else is not a number.
      unparsed <- na_at[!(raw[na_at] %in% c("NA", ""))]
      if (length(unparsed) > 0L) {
        hit <- unparsed[[1]]
        stop("counts_csv ", path, ": the value '", raw[hit], "' in row ", n_rows + ((hit - 1L) %% nrow(raw)) + 1L,
             " is not a number; a counts table holds numbers only after its row names.")
      }
    }
    nz <- which(is.na(values) | values != 0)
    i_parts[[length(i_parts) + 1L]] <- as.integer((nz - 1L) %% nrow(raw) + 1L + n_rows)
    j_parts[[length(j_parts) + 1L]] <- as.integer((nz - 1L) %/% nrow(raw) + 1L)
    x_parts[[length(x_parts) + 1L]] <- values[nz]
    n_rows <- n_rows + nrow(raw)
    rm(raw, values, nz)
  }
  if (is.na(n_fields)) stop("counts_csv ", path, " has a header but no row.")
  rn <- unlist(row_names, use.names = FALSE)
  dup <- unique(rn[duplicated(rn)])
  if (length(dup) > 0) {
    stop("counts_csv ", path, " names some rows more than once (e.g. ",
         paste(sprintf('"%s"', utils::head(dup, 3)), collapse = ", "), "); each row name must be unique.")
  }
  Matrix::sparseMatrix(
    i = unlist(i_parts, use.names = FALSE), j = unlist(j_parts, use.names = FALSE),
    x = unlist(x_parts, use.names = FALSE), dims = c(n_rows, n_fields - 1L), dimnames = list(rn, col_names)
  )
}

# Neighborseq builds its training set as a dense table: n.am artificial profiles for every
# singlet type and every unordered pair (k + k(k+1)/2 classes for k clusters), one column per
# panel gene. That table is intrinsic to the method, so it is estimated before it is built and the
# run refuses with the numbers when it cannot fit, instead of dying half-way through.
am_memory_estimate <- function(n_clusters, n_genes) {
  n_classes <- n_clusters + n_clusters * (n_clusters + 1) / 2
  rows <- n_classes * NEIGHBORSEQ_N_AM
  list(n_classes = n_classes, rows = rows, bytes = rows * n_genes * 8 * AM_DENSE_COPIES)
}

# The clusters CSV: barcode, label. Both columns are read as text. A bare read.csv parsed an all-digit
# barcode column (Zhuang/ABC-atlas MERFISH ids, many CosMx/Xenium exports) as double, so its row names
# became "1.06105955553685e+38" and matched none of the counts' ids, which read_counts_sparse keeps as
# text (NSEQ-1). "NA" is still a missing label, as read.csv read it.
read_cluster_labels <- function(path) {
  clusters_df <- read.csv(path, check.names = FALSE, colClasses = "character", na.strings = "NA")
  if (ncol(clusters_df) < 2) stop("clusters CSV needs two columns: barcode, cluster label")
  labels <- clusters_df[[2]]
  names(labels) <- clusters_df[[1]]
  if (anyDuplicated(names(labels))) stop("clusters CSV has duplicate barcodes in its first column")
  attr(labels, "label_column") <- colnames(clusters_df)[2]
  labels
}

run_neighborseq <- function(opts) {
  if (is.null(opts$counts_csv) || is.null(opts$clusters_csv) || is.null(opts$output_dir)) {
    stop("Neighborseq requires --counts-csv, --clusters-csv, and --output-dir")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  warnings <- character(0)
  warnings <- c(warnings, paste0(
    "ignored parameter(s) ", paste(IGNORED_PARAMS, collapse = ", "), ": ", IGNORED_WHY
  ))

  # --- Load data (sparse from the first block; the whole table is never dense) ---
  log_msg("Reading counts from: ", opts$counts_csv, " (in row blocks, into a sparse matrix)")
  counts_mat <- read_counts_sparse(opts$counts_csv)
  log_msg("Counts: ", nrow(counts_mat), " rows x ", ncol(counts_mat), " columns, ",
          length(counts_mat@x), " non-zero values")

  log_msg("Reading cluster assignments from: ", opts$clusters_csv)
  labels <- read_cluster_labels(opts$clusters_csv)
  label_column <- attr(labels, "label_column")
  attr(labels, "label_column") <- NULL

  # Ensure matching cell IDs (counts may be genes x cells or cells x genes)
  orientation <- "genes x cells"
  common_cells <- intersect(colnames(counts_mat), names(labels))
  if (length(common_cells) == 0) {
    counts_mat <- t(counts_mat)
    orientation <- "cells x genes (transposed)"
    common_cells <- intersect(colnames(counts_mat), names(labels))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "counts", colnames(counts_mat), "cluster assignments", names(labels)))}

  n_cells_in_counts <- ncol(counts_mat)
  n_label_rows <- length(labels)
  n_cells_without_label_row <- n_cells_in_counts - length(common_cells)
  if (n_cells_without_label_row > 0) {
    warnings <- c(warnings, paste0(
      n_cells_without_label_row, " of ", n_cells_in_counts, " cells in counts_csv have no row in ",
      "clusters_csv and were not analysed"
    ))
  }

  labels <- labels[common_cells]

  # --- Missing labels are not a class ---
  missing <- is_missing_label(labels)
  n_unlabeled <- sum(missing)
  n_unlabeled_dropped <- 0L
  if (n_unlabeled > 0) {
    if (!isTRUE(opts$drop_unlabeled)) {
      stop(sprintf(paste0(
        "%d of %d cells have no label (NA/empty) in column '%s' of clusters_csv. Pass ",
        "drop_unlabeled=True to leave them out, or label them first; a missing label is not a class."
      ), n_unlabeled, length(labels), label_column))
    }
    n_unlabeled_dropped <- n_unlabeled
    labels <- labels[!missing]
    warnings <- c(warnings, paste0(
      "drop_unlabeled=True: ", n_unlabeled, " cells with no label were left out of the analysis"
    ))
  }

  clusters <- droplevels(as.factor(labels))
  names(clusters) <- names(labels)

  # --- A one-cell cluster cannot be sampled correctly ---
  # artificial_multiplets() draws each cell type's members with sample(idx, ...). When a type has
  # exactly one cell, idx is a single integer and base::sample(k, n) draws from 1:k instead -- the
  # singleton's "profiles" become random other cells, of any type, and nothing says so.
  sizes <- table(clusters)
  singletons <- names(sizes)[sizes == 1L]
  dropped_singleton_clusters <- character(0)
  if (length(singletons) > 0) {
    if (!isTRUE(opts$drop_singleton_clusters)) {
      stop(paste0(
        length(singletons), " cluster(s) have exactly one cell: ",
        paste(sprintf('"%s"', singletons), collapse = ", "), ". Neighborseq samples a cluster's ",
        "cells with base::sample(), which on a single index draws from 1:index -- the cluster would ",
        "be trained on random other cells. Pass drop_singleton_clusters=True to leave those cells ",
        "out, or merge them into another cluster in clusters_csv."
      ))
    }
    dropped_singleton_clusters <- singletons
    keep <- !(as.character(clusters) %in% singletons)
    clusters <- droplevels(clusters[keep])
    warnings <- c(warnings, paste0(
      "drop_singleton_clusters=True: left out ", length(singletons), " one-cell cluster(s) (",
      paste(singletons, collapse = ", "), ")"
    ))
  }

  if (nlevels(clusters) < 2L) {
    stop(sprintf(paste0(
      "Neighborseq needs at least 2 clusters to model interactions between them; %d remain after ",
      "matching counts to clusters_csv and removing unlabelled or one-cell clusters."
    ), nlevels(clusters)))
  }

  counts_mat <- counts_mat[, names(clusters), drop = FALSE]

  # --- Label tokens Neighborseq cannot mis-split ---
  tokens <- label_tokens(nlevels(clusters))
  lab2tok <- stats::setNames(tokens, levels(clusters))
  tok2lab <- stats::setNames(levels(clusters), tokens)
  celltypes_tok <- factor(unname(lab2tok[as.character(clusters)]), levels = tokens)
  names(celltypes_tok) <- names(clusters)
  sizes <- table(clusters)
  label_map <- data.frame(
    token   = tokens,
    label   = levels(clusters),
    n_cells = as.integer(sizes[levels(clusters)]),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )

  n_cells <- ncol(counts_mat)
  n_genes <- nrow(counts_mat)
  log_msg("N genes = ", n_genes, ", N cells = ", n_cells,
          ", N clusters = ", nlevels(clusters), " (", orientation, ")")

  # --- Prepare cell matrix (manual, Seurat v5-compatible) ---
  log_msg("Preparing cell matrix with marker gene selection (all cells; no per-cluster cap)...")
  sparse_counts <- as(counts_mat, "CsparseMatrix")
  rm(counts_mat)
  invisible(gc())

  suppressWarnings({
    sobj <- Seurat::CreateSeuratObject(counts = sparse_counts)
    sobj <- Seurat::NormalizeData(sobj, verbose = FALSE)
    sobj <- Seurat::FindVariableFeatures(sobj, nfeatures = min(5000, nrow(sparse_counts)), verbose = FALSE)
    Seurat::Idents(sobj) <- clusters

    markers <- Seurat::FindAllMarkers(sobj, logfc.threshold = MARKER_LOGFC_THRESHOLD,
                                      only.pos = TRUE, min.pct = MARKER_MIN_PCT,
                                      verbose = FALSE)
  })

  used_fallback <- FALSE
  n_marker_genes_found <- 0L
  clusters_without_markers <- levels(clusters)
  if (nrow(markers) > 0) {
    wt_col <- if ("avg_log2FC" %in% colnames(markers)) "avg_log2FC" else "avg_logFC"
    top_markers <- markers %>%
      dplyr::group_by(cluster) %>%
      dplyr::top_n(n = MARKER_TOP_N, wt = !!rlang::sym(wt_col))
    marker_genes <- unique(top_markers$gene)
    n_marker_genes_found <- length(marker_genes)
    gene_selection <- sprintf("FindAllMarkers top %d per cluster", MARKER_TOP_N)
    # Seurat skips a cluster under 3 cells, or one with no gene past the thresholds, and the
    # suppressWarnings above hides that it did; the panel then speaks for fewer clusters than ran.
    clusters_without_markers <- setdiff(levels(clusters), as.character(unique(markers$cluster)))
    if (length(clusters_without_markers) > 0) {
      warnings <- c(warnings, paste0(
        "FindAllMarkers returned no marker gene for ", length(clusters_without_markers), " cluster(s) (",
        paste(clusters_without_markers, collapse = ", "), "); the panel carries no gene chosen for them"
      ))
    }
  } else {
    if (!isTRUE(opts$allow_hvg_panel_fallback)) {
      stop(paste0(
        "FindAllMarkers found no positive marker gene for any of the ", nlevels(clusters),
        " clusters (logfc.threshold = ", MARKER_LOGFC_THRESHOLD, ", min.pct = ", MARKER_MIN_PCT,
        "), so there is no marker panel to train Neighborseq's classifier on. Check that the ",
        "labels in clusters_csv describe the expression in counts_csv, or pass ",
        "allow_hvg_panel_fallback=True to train on the top ", HVG_PANEL_SIZE,
        " variable features instead."
      ))
    }
    marker_genes <- head(Seurat::VariableFeatures(sobj), HVG_PANEL_SIZE)
    used_fallback <- TRUE
    gene_selection <- sprintf("top %d variable features (allow_hvg_panel_fallback)", HVG_PANEL_SIZE)
  }
  # A panel under MIN_PANEL_GENES genes is padded with variable features -- reported, not silent.
  n_padding_genes <- 0L
  if (length(marker_genes) < MIN_PANEL_GENES) {
    extra <- head(Seurat::VariableFeatures(sobj), HVG_PADDING)
    n_padding_genes <- length(setdiff(extra, marker_genes))
    marker_genes <- unique(c(marker_genes, extra))
    gene_selection <- sprintf("%s + %d variable features (panel had fewer than %d genes)",
                              gene_selection, n_padding_genes, MIN_PANEL_GENES)
    warnings <- c(warnings, paste0(
      "the marker panel had ", length(marker_genes) - n_padding_genes, " gene(s), fewer than ",
      MIN_PANEL_GENES, "; ", n_padding_genes, " top variable features were added to it"
    ))
  }
  method <- if (used_fallback) {
    paste0(METHOD_NAME, " on the top ", HVG_PANEL_SIZE, " variable features (no marker genes found)")
  } else {
    METHOD_NAME
  }
  if (used_fallback) {
    warnings <- c(warnings, paste0(
      "fallback ran: ", method, " -- FindAllMarkers returned no marker gene and ",
      "allow_hvg_panel_fallback=True"
    ))
  }

  # Use raw counts for the marker genes (Seurat v5 compatible access)
  cell_mat <- Seurat::GetAssayData(sobj, layer = "counts")[marker_genes, , drop = FALSE]
  log_msg("Cell matrix after prep: ", nrow(cell_mat), " genes x ", ncol(cell_mat), " cells (",
          gene_selection, ")")

  # --- Memory preflight for the dense artificial-multiplet table ---
  est <- am_memory_estimate(nlevels(clusters), nrow(cell_mat))
  avail <- available_memory_bytes()
  if (!is.na(avail) && est$bytes > avail) {
    stop(sprintf(paste0(
      "Neighborseq's training table is dense and would not fit: %d clusters make %d classes ",
      "(every singlet and every pair), x %d artificial profiles each = %d rows x %d panel genes, ",
      "about %.1f GiB held in %d copies, but about %.1f GiB is available here (MemAvailable / cgroup ",
      "limit). The class count grows with the square of the cluster count, so run it with fewer, ",
      "coarser labels in clusters_csv, or where that much memory is available. No cell is subsampled."
    ), nlevels(clusters), as.integer(est$n_classes), NEIGHBORSEQ_N_AM, as.integer(est$rows),
    nrow(cell_mat), est$bytes / 2^30, as.integer(AM_DENSE_COPIES), avail / 2^30))
  }

  # --- Run Neighborseq ---
  log_msg("Running Neighborseq interaction analysis on ", nlevels(clusters), " label tokens...")
  ns_result <- neighborseq(
    cell.mat = cell_mat,
    celltypes = celltypes_tok,
    iter = NEIGHBORSEQ_ITER,
    n.am = NEIGHBORSEQ_N_AM,
    nsim = NEIGHBORSEQ_NSIM
  )

  # --- Extract and save results (labels decoded back to the caller's names) ---
  label_map_path <- file.path(opts$output_dir, "neighborseq_label_map.csv")
  write_csv_atomic(label_map, label_map_path, row.names = FALSE)

  interactions <- NULL
  interaction_path <- NULL
  if (!is.null(ns_result$result) && is.data.frame(ns_result$result)) {
    interactions <- as.data.frame(ns_result$result)
    for (col in intersect(c("Cell_1", "Cell_2"), colnames(interactions))) {
      interactions[[col]] <- decode_class_names(interactions[[col]], tok2lab)
    }
    interaction_path <- file.path(opts$output_dir, "neighborseq_interactions.csv")
    write_csv_atomic(interactions, interaction_path)
  }

  # Save predictions: one row per analysed cell, named by its barcode (xgpred returns them in
  # cell.mat's column order but without names, so the table used to be keyed 1..n).
  pred_path <- NULL
  if (!is.null(ns_result$pred)) {
    pred_df <- as.data.frame(ns_result$pred, check.names = FALSE)
    colnames(pred_df) <- decode_class_names(colnames(pred_df), tok2lab)
    if (nrow(pred_df) == ncol(cell_mat)) rownames(pred_df) <- colnames(cell_mat)
    pred_path <- file.path(opts$output_dir, "neighborseq_predictions.csv")
    write_csv_atomic(pred_df, pred_path)
  }

  # Top interactions
  top_path <- NULL
  if (!is.null(interactions) && nrow(interactions) > 0) {
    if ("padj" %in% colnames(interactions)) {
      top_int <- head(interactions[order(interactions$padj), ], opts$n_top)
    } else if ("pvalue" %in% colnames(interactions)) {
      top_int <- head(interactions[order(interactions$pvalue), ], opts$n_top)
    } else {
      top_int <- head(interactions, opts$n_top)
    }
    top_path <- file.path(opts$output_dir, "neighborseq_top_interactions.csv")
    write_csv_atomic(top_int, top_path, row.names = FALSE)
  }

  n_interactions_tested <- if (is.null(interactions)) 0L else nrow(interactions)
  dropped_note <- paste0(
    if (n_unlabeled_dropped > 0) paste0(" ", n_unlabeled_dropped, " unlabelled cells were left out (drop_unlabeled).") else "",
    if (length(dropped_singleton_clusters) > 0) paste0(
      " One-cell cluster(s) left out (drop_singleton_clusters): ",
      paste(dropped_singleton_clusters, collapse = ", "), ".") else "",
    if (n_cells_without_label_row > 0) paste0(
      " ", n_cells_without_label_row, " cells in counts_csv had no row in clusters_csv.") else ""
  )

  payload <- list(
    status       = "ok",
    tool         = "neighborseq",
    task         = "interaction_network",
    data         = list(
      n_cells                    = n_cells,
      n_genes                    = n_genes,
      n_clusters                 = nlevels(clusters),
      n_cells_in_counts          = n_cells_in_counts,
      n_label_rows               = n_label_rows,
      n_cells_without_label_row  = n_cells_without_label_row,
      n_unlabeled_dropped        = n_unlabeled_dropped,
      dropped_singleton_clusters = I(dropped_singleton_clusters),
      counts_orientation         = orientation,
      n_interactions_tested      = n_interactions_tested
    ),
    output_files = list(
      interactions_csv = interaction_path,
      predictions_csv  = pred_path,
      top_interactions = top_path,
      label_map_csv    = label_map_path
    ),
    params       = list(
      n_trees                   = opts$n_trees,
      n_top                     = opts$n_top,
      pval_cutoff               = opts$pval_cutoff,
      seed                      = opts$seed,
      allow_hvg_panel_fallback  = isTRUE(opts$allow_hvg_panel_fallback),
      drop_unlabeled            = isTRUE(opts$drop_unlabeled),
      drop_singleton_clusters   = isTRUE(opts$drop_singleton_clusters),
      method                    = method,
      used_fallback             = used_fallback,
      ignored                   = I(IGNORED_PARAMS),
      num_parallel_tree         = UPSTREAM_NUM_PARALLEL_TREE,
      xgboost_nround            = UPSTREAM_NROUND,
      xgboost_max_depth         = UPSTREAM_MAX_DEPTH,
      train_fraction            = UPSTREAM_TRAIN_FRACTION,
      multiplet_degree          = UPSTREAM_MULTIPLET_DEGREE,
      iter                      = NEIGHBORSEQ_ITER,
      n_artificial_multiplets   = NEIGHBORSEQ_N_AM,
      nsim                      = NEIGHBORSEQ_NSIM,
      gene_selection            = gene_selection,
      marker_logfc_threshold    = MARKER_LOGFC_THRESHOLD,
      marker_min_pct            = MARKER_MIN_PCT,
      marker_top_n              = MARKER_TOP_N,
      marker_max_cells_per_ident = "all (no per-cluster downsampling)",
      label_encoding            = "letters-only tokens (see neighborseq_label_map.csv); outputs decoded to the original labels"
    ),
    summary      = list(
      n_marker_genes       = nrow(cell_mat),
      n_marker_genes_found = n_marker_genes_found,
      n_padding_genes      = n_padding_genes,
      clusters_without_markers = I(clusters_without_markers),
      gene_selection       = gene_selection,
      cluster_labels       = levels(clusters)
    ),
    analysis     = paste0(
      "Neighborseq analyzed cell-cell interactions across ",
      nlevels(clusters), " clusters using ", n_cells,
      " cells and ", nrow(cell_mat), " marker genes (", gene_selection, ", chosen on all cells). ",
      "Method: ", method, "; the forest is fixed upstream at ", UPSTREAM_NUM_PARALLEL_TREE,
      " trees, so n_trees and pval_cutoff were not used (params.ignored). ",
      n_interactions_tested, " cell-type pairs were tested; each carries its own pval and padj.",
      dropped_note
    )
  )
  payload$warnings <- I(warnings)
  payload
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # Parsing sits inside the handler: a bad value ("--n-top abc") is refused in the JSON payload the
  # portal reads, not as a bare R error with nothing on stdout.
  res <- tryCatch(with_r_traceback({
    opts <- parse_args(args)
    set.seed(opts$seed)
    run_neighborseq(opts)
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "neighborseq",
      task      = "interaction_network",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
