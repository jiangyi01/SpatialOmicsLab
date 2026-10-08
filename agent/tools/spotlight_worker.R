#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SingleCellExperiment)
  library(SPOTlight)
  # scran is used when it is installed (spotlight_env had none on 2026-09-29); see marker_table().
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
  message(sprintf("[spotlight-worker] %s", msg))
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

# Space Ranger before 2.0 writes spatial/tissue_positions_list.csv with NO header row -- every
# standard Visium folder ships it beside the headed tissue_positions.csv. read.csv(header = TRUE)
# takes its first spot as the header: that spot was lost (and then reported as a spot with "no row"
# in the coordinates file), and the columns came back named after its own values, so nothing named
# the pixel columns. In 17 of the library's 58 such files the lost spot is an in-tissue spot of the
# slide, left out of the deconvolution.
#
# first_record_line() and read_coords_csv() are copied from tools/spotsweeper_worker.R, which
# explains the rule: the first line is a header unless it looks like data -- an identifier that is
# not an identifier-column label, followed by nothing but numbers (pandas' 0, 1, 2, ... column
# labels excepted). A headerless file is read under Space Ranger's own column names when it has
# Space Ranger's six columns (and column 2 is a 0/1 tissue flag), as barcode,x,y when it has three,
# and refused otherwise. One change for this worker's by-name family: the identifiers stay as
# column 1 of the frame (not row names) and every column is read as character, as this worker has
# always read the file, so a long numeric spot ID is not mangled and the caller offers
# colnames(frame)[-1] to the resolver.
SPACE_RANGER_POSITION_COLS <- c("in_tissue", "array_row", "array_col",
                                "pxl_row_in_fullres", "pxl_col_in_fullres")
ID_COLUMN_LABELS <- c("", "barcode", "barcodes", "spot", "spot_id", "spotid", "cell", "cell_id",
                      "cellid", "sample", "sample_id", "index")

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
    return(list(frame = read.csv(path, check.names = FALSE, colClasses = "character"),
                header = "present"))
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
    flag <- trimws(frame$in_tissue)
    if (!all(flag %in% c("0", "1"))) {
      stop("Coordinates file ", path, " has no header row that names its columns (its first line, ",
           shown, ", is an identifier followed only by numbers, so it reads as a spot) and six ",
           "columns, but its second column holds values other than 0 and 1, so it is not Space ",
           "Ranger's in_tissue flag and the file is not tissue_positions_list.csv. ", rename_hint)
    }
  }
  log_msg("Coordinates file has no header row; ", header)
  list(frame = frame, header = header)
}

# Space Ranger's in_tissue flag, as a number per spot: 1 in tissue, 0 background, NA unreadable.
# The same reading as tools/worker_utils.py's keep_in_tissue(): TRUE/FALSE and 1/0, any case.
# Only a column named in_tissue is a flag; a text "tissue" column (CELLxGENE: 'thymus') is not.
in_tissue_values <- function(x) {
  v <- tolower(trimws(as.character(x)))
  v[v == "true"] <- "1"
  v[v == "false"] <- "0"
  suppressWarnings(as.numeric(v))
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


# What runs. SPOTlight() is one path -- trainNMF (seeded NMF, model "ns") then runDeconvolution
# (NNLS per spot) -- so there is no substitute to fall back to and params.used_fallback is always
# FALSE. The version is read at run time so the payload names the package that actually ran.
METHOD_NAME <- "SPOTlight (trainNMF seeded nsNMF -> runDeconvolution NNLS)"

# Marker selection, the SPOTlight vignette's recipe: scran::scoreMarkers, keep mean.AUC > 0.8.
# mean.AUC is the mean over every OTHER cell type of the pairwise AUC (the probability that a cell
# of this type out-expresses a cell of that one, ties counted half). A type with no gene above the
# threshold would make SPOTlight stop (`groups %in% mgs$cluster`), so it is seeded with its
# MARKER_TOP_BY_RANK best genes by the same statistic instead -- and the payload names every such
# type, because its topic rests on weaker markers than the rest.
MARKER_AUC_MIN <- 0.8
MARKER_TOP_BY_RANK <- 25L

# A missing label is not a class. read.csv keeps a blank label as "" -- which became a cell type
# of its own, written as column "V1" of the proportions -- and reads "NA" as NA, which reached
# nnls as "Arguments `A' and `b' have incompatible dimensions". The vocabulary is
# tools/worker_utils.py's drop_unlabeled(): NA, "", "na", "nan", "none" (any case, trimmed).
LABEL_MISSING <- c("", "na", "nan", "none")

unlabeled_mask <- function(labels) {
  is.na(labels) | tolower(trimws(labels)) %in% LABEL_MISSING
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

# mean.AUC as scran::scoreMarkers defines it: for each cell type, the mean over every other type
# of the pairwise Mann-Whitney AUC. The AUC is rank-based, so it is unchanged by the log base or the
# size-factor scaling of the log-counts it is computed on. This replaced a one-vs-rest AUC (each
# type against all other cells pooled) that the code labelled mean.AUC and described as scran's:
# a gene shared by two types scores ~0.5 against its partner, which the pairwise mean keeps and the
# pooled comparison dilutes -- on the 3-type test fixture a gene high in two types read 0.895 pooled
# and 0.789 pairwise, and crossed the 0.8 marker threshold only under the wrong statistic.
pairwise_mean_auc <- function(lc, grp) {
  grp <- as.character(grp)
  groups <- unique(grp)
  n_groups <- length(groups)
  if (n_groups < 2L) {
    stop("The reference has ", n_groups, " cell type (", paste(groups, collapse = ", "),
         "); SPOTlight needs at least two to deconvolve a spot.")
  }
  idx <- lapply(groups, function(g) which(grp == g))
  sums <- matrix(0, nrow = nrow(lc), ncol = n_groups, dimnames = list(rownames(lc), groups))
  for (a in seq_len(n_groups - 1L)) {
    ia <- idx[[a]]
    na <- length(ia)
    for (b in seq.int(a + 1L, n_groups)) {
      ib <- idx[[b]]
      nb <- length(ib)
      r <- matrixStats::rowRanks(lc[, c(ia, ib), drop = FALSE], ties.method = "average")
      auc_ab <- (rowSums(r[, seq_len(na), drop = FALSE]) - na * (na + 1) / 2) / (na * nb)
      sums[, a] <- sums[, a] + auc_ab
      sums[, b] <- sums[, b] + (1 - auc_ab)
    }
  }
  sums / (n_groups - 1L)
}

# Returns the marker table SPOTlight is seeded with, plus what was done to build it.
marker_table <- function(ref_sce, lc, grp) {
  if (requireNamespace("scran", quietly = TRUE)) {
    log_msg("Scoring markers with scran::scoreMarkers (mean.AUC)...")
    score_res <- scran::scoreMarkers(ref_sce, groups = grp)
    rows <- list()
    for (g in names(score_res)) {
      d <- as.data.frame(score_res[[g]])
      if (!"mean.AUC" %in% colnames(d)) {
        # The former code put mean.logFC.cohen here under the AUC's name and applied the AUC's 0.8
        # threshold to it -- a different statistic on a different scale, silently.
        stop("scran::scoreMarkers returned no mean.AUC column (it has: ",
             paste(colnames(d), collapse = ", "), "); this worker selects markers by mean.AUC > ",
             MARKER_AUC_MIN, ". Update scran.")
      }
      rows[[g]] <- data.frame(gene = rownames(d), cluster = g, mean.AUC = as.numeric(d$mean.AUC),
                              stringsAsFactors = FALSE)
    }
    mgs <- do.call(rbind, rows)
    method <- "scran::scoreMarkers mean.AUC (mean of pairwise AUCs against every other cell type)"
  } else {
    log_msg("scran is not installed: computing scoreMarkers' mean.AUC (mean of pairwise AUCs) in base R")
    log_msg("  ", nrow(lc), " genes x ", ncol(lc), " cells, ", length(unique(grp)), " cell types")
    auc <- pairwise_mean_auc(lc, grp)
    mgs <- data.frame(
      gene     = rep(rownames(auc), times = ncol(auc)),
      cluster  = rep(colnames(auc), each = nrow(auc)),
      mean.AUC = as.vector(auc),
      stringsAsFactors = FALSE
    )
    method <- paste0("mean of pairwise Mann-Whitney AUCs against every other cell type, computed in ",
                     "base R (the statistic scran::scoreMarkers reports as mean.AUC; scran is not ",
                     "installed in this environment)")
  }
  rownames(mgs) <- NULL

  all_groups <- unique(as.character(grp))
  keep <- mgs[!is.na(mgs$mean.AUC) & mgs$mean.AUC > MARKER_AUC_MIN, , drop = FALSE]
  by_rank <- setdiff(all_groups, unique(as.character(keep$cluster)))
  for (g in by_rank) {
    sub <- mgs[as.character(mgs$cluster) == g & !is.na(mgs$mean.AUC), , drop = FALSE]
    sub <- sub[order(-sub$mean.AUC), , drop = FALSE]
    keep <- rbind(keep, utils::head(sub, MARKER_TOP_BY_RANK))
  }
  if (length(by_rank) > 0L) {
    log_msg("  ", length(by_rank), " cell type(s) had no gene with mean.AUC > ", MARKER_AUC_MIN,
            " and were seeded with their top ", MARKER_TOP_BY_RANK, " genes by mean.AUC: ",
            paste(by_rank, collapse = ", "))
  }
  rownames(keep) <- NULL
  list(mgs = keep, method = method, by_rank = by_rank)
}


# The genes each cell type's NMF topic is seeded with -- what SPOTlight 1.6.7's .init_nmf does to the
# marker table, by name. trainNMF keeps the markers that are genes of the model (in both matrices,
# non-zero in both); .init_nmf then keeps each type's top n_top of those by weight (order(...,
# decreasing = TRUE): ties stay in table order) and removes every gene that also sits in another
# type's top-n_top list. A type left with no gene gets an all-1e-12 W column: its topic starts
# unseeded. Counting only "has a marker among the model genes" missed the second cut, so a type
# whose markers were all shared with a sibling (fine subtypes: B_mem_1..4) was fitted unseeded
# while the payload listed no unseeded type.
seed_genes_per_type <- function(mgs_df, model_genes, n_top, types) {
  usable <- mgs_df[mgs_df$gene %in% model_genes, , drop = FALSE]
  top <- lapply(types, function(k) {
    df <- usable[as.character(usable$cluster) == k, , drop = FALSE]
    df <- df[order(df$mean.AUC, decreasing = TRUE), , drop = FALSE]
    as.character(utils::head(df$gene, n_top))
  })
  names(top) <- types
  seeds <- lapply(types, function(k) {
    others <- unlist(top[names(top) != k], use.names = FALSE)
    top[[k]][!top[[k]] %in% others]
  })
  names(seeds) <- types
  list(top = top, seeds = seeds)
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
    cat("Usage: spotlight_worker.R [options]\n")
    cat("Options:\n")
    cat("  --spatial-counts-csv PATH  Spatial gene expression counts CSV (genes x spots) [required]\n")
    cat("  --spatial-coords-csv PATH  Spatial coordinates CSV (spots x coords) [required]\n")
    cat("  --ref-counts-csv PATH      scRNA-seq reference counts CSV (genes x cells) [required]\n")
    cat("  --ref-celltypes-csv PATH   Reference cell type annotation CSV [required]\n")
    cat("  --output-dir PATH          Output directory [required]\n")
    cat("  --n-top INT                Top marker genes per cell type that seed the NMF (default: 100)\n")
    cat("  --min-cont FLOAT           Per-spot proportion below which a cell type is set to zero before\n")
    cat("                             the rest are rescaled to sum to 1; SPOTlight's min_prop (default: 0.09)\n")
    cat("  --drop-unlabeled true|false  Leave out reference cells with a missing label (NA, empty,\n")
    cat("                             nan, none) instead of refusing the run (default: false)\n")
    cat("  --help                     Show this help message\n")
    quit(status = 0)
  }

  opts <- list(
    spatial_counts_csv = NULL,
    spatial_coords_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    n_top              = 100L,
    min_cont           = 0.09,
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
    } else if (key == "--n-top") {
      opts$n_top <- suppressWarnings(as.numeric(val))
    } else if (key == "--min-cont") {
      opts$min_cont <- suppressWarnings(as.numeric(val))
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- tolower(val) %in% c("true", "1", "yes")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  # Checked here, by the names the caller used, rather than left to SPOTlight's stopifnot().
  if (length(opts$n_top) != 1L || is.na(opts$n_top) || opts$n_top < 1 || opts$n_top != round(opts$n_top)) {
    stop("n_top must be a whole number of marker genes per cell type, at least 1; got ", opts$n_top, ".")
  }
  opts$n_top <- as.integer(opts$n_top)
  if (length(opts$min_cont) != 1L || is.na(opts$min_cont) || opts$min_cont < 0 || opts$min_cont > 1) {
    stop("min_cont is a proportion and must lie in [0, 1]; got ", opts$min_cont, ".")
  }

  opts
}

run_spotlight <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$spatial_coords_csv) ||
      is.null(opts$ref_counts_csv) || is.null(opts$ref_celltypes_csv) ||
      is.null(opts$output_dir)) {
    stop("SPOTlight requires --spatial-counts-csv, --spatial-coords-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$spatial_coords_csv,
              opts$ref_counts_csv, opts$ref_celltypes_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  warns <- character(0)

  # --- Load reference data ---
  log_msg("Reading reference counts from: ", opts$ref_counts_csv)
  ref_counts_df <- read.csv(opts$ref_counts_csv, row.names = 1, check.names = FALSE)
  ref_counts_mat <- as.matrix(ref_counts_df)

  log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE)
  # Auto-detect celltype column if STCoscientist passed full obs metadata (>1 column).
  if (ncol(ref_ct_df) > 1L) {
    pat <- "^(celltype|cell_type|cell\\.type|annotation|annot|cluster|label)$"
    hits <- grep(pat, colnames(ref_ct_df), ignore.case = TRUE)
    if (length(hits) >= 1L) {
      sel <- hits[[1L]]
      log_msg("metadata CSV has ", ncol(ref_ct_df),
              " columns; auto-selected '", colnames(ref_ct_df)[sel], "' as celltype")
    } else {
      stop(sprintf(
        "ref_celltypes_csv has %d columns and no recognizable celltype column; columns were: %s",
        ncol(ref_ct_df), paste(colnames(ref_ct_df), collapse = ", ")))
    }
    raw_ct <- as.character(ref_ct_df[, sel])
  } else {
    raw_ct <- as.character(ref_ct_df[, 1L])
  }
  names(raw_ct) <- rownames(ref_ct_df)
  n_ref_annotated <- length(raw_ct)

  missing_lab <- unlabeled_mask(raw_ct)
  n_unlabeled <- sum(missing_lab)
  if (n_unlabeled > 0L) {
    if (!isTRUE(opts$drop_unlabeled)) {
      stop(n_unlabeled, " of ", length(raw_ct), " reference cells in ", opts$ref_celltypes_csv,
           " have no cell type label (NA, empty, nan or none). Pass drop_unlabeled=True to leave ",
           "them out, or label them first; a missing label is not a cell type.")
    }
    log_msg("Dropping ", n_unlabeled, " of ", length(raw_ct),
            " reference cells with no label (drop_unlabeled=TRUE)")
    warns <- c(warns, sprintf(
      "%d of %d reference cells had no cell type label and were left out (drop_unlabeled=True).",
      n_unlabeled, length(raw_ct)))
    raw_ct <- raw_ct[!missing_lab]
  }

  # Sanitize cell type names: replace / and spaces (SPOTlight prohibits special chars). The
  # proportions columns carry the sanitized names, so the payload says which labels were rewritten,
  # and two labels the rewrite would merge into one are refused rather than pooled.
  cell_types <- gsub("/", "_", raw_ct)
  cell_types <- gsub(" ", "_", cell_types)
  names(cell_types) <- names(raw_ct)
  label_pairs <- unique(data.frame(from = unname(raw_ct), to = unname(cell_types), stringsAsFactors = FALSE))
  merged <- unique(label_pairs$to[duplicated(label_pairs$to)])
  if (length(merged) > 0L) {
    clash <- label_pairs[label_pairs$to %in% merged, , drop = FALSE]
    stop("Cell type labels ", paste(sprintf('"%s"', clash$from), collapse = ", "),
         " all become ", paste(sprintf('"%s"', merged), collapse = ", "),
         " once '/' and ' ' are replaced by '_' for SPOTlight, which would pool distinct types. ",
         "Rename them in ", opts$ref_celltypes_csv, " so they stay distinct.")
  }
  renamed <- label_pairs[label_pairs$from != label_pairs$to, , drop = FALSE]
  cell_type_renames <- if (nrow(renamed) > 0L) as.list(stats::setNames(renamed$to, renamed$from)) else NULL

  # Ensure genes x cells orientation: try to match colnames to cell type names
  common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  if (length(common_cells) == 0) {
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(cell_types)))}

  n_ref_in_counts <- ncol(ref_counts_mat)
  n_ref_counts_unannotated <- length(setdiff(colnames(ref_counts_mat), rownames(ref_ct_df)))
  if (n_ref_counts_unannotated > 0L) {
    warns <- c(warns, sprintf(
      "%d of %d reference cells in the counts matrix have no row in %s and were left out.",
      n_ref_counts_unannotated, n_ref_in_counts, opts$ref_celltypes_csv))
  }

  cell_types <- cell_types[common_cells]

  # SPOTlight 1.6.7 matches topics to cell types by POSITION in two lists that are in different
  # orders unless the reference's cells come grouped in sorted cell-type order. trainNMF numbers the
  # topics in first-appearance order (unique(groups)); .init_nmf removes shared seed genes with
  # mgs[ks != k] -- a logical index built in that order, applied to a list split() put in sorted
  # order -- and .topic_profiles / runDeconvolution label the proportions in sorted order. With
  # cells ordered B, C, A, pure-A spots came back 0.53 "B" / 0.46 "A" and B got no share anywhere;
  # with B, A, C two topics lost every seed. The benchmark reference (mini_visium_sc_ref, 44 types)
  # is not in sorted order. The cells are therefore handed over grouped by type in the order
  # factor() sorts the labels -- the same order split() uses -- which moves no value: NMF on
  # permuted columns is the same factorisation, and the marker AUCs are rank statistics.
  type_levels <- levels(factor(cell_types))
  type_pos <- match(cell_types, type_levels)
  reference_reordered <- is.unsorted(type_pos)
  if (reference_reordered) {
    cell_order <- order(type_pos)
    common_cells <- common_cells[cell_order]
    cell_types <- cell_types[cell_order]
    log_msg("Reference cells regrouped by cell type in sorted order (SPOTlight pairs topics and ",
            "types by position)")
  }
  ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]

  log_msg("Building reference SingleCellExperiment: ", length(common_cells),
          " cells, ", length(unique(cell_types)), " cell types")

  # SPOTlight expects a SingleCellExperiment with counts and colData$type
  ref_sce <- SingleCellExperiment(
    assays = list(counts = ref_counts_mat),
    colData = data.frame(type = factor(cell_types), row.names = common_cells)
  )
  # Compute logcounts for marker gene detection
  # logNormCounts equivalent: normalize by library size then log1p
  lib_sizes <- colSums(counts(ref_sce))
  lib_sizes[lib_sizes == 0] <- 1
  norm_counts <- t(t(counts(ref_sce)) / lib_sizes) * mean(lib_sizes)
  logcounts(ref_sce) <- log1p(norm_counts)

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  sp_counts_mat <- as.matrix(sp_counts_df)

  log_msg("Reading spatial coordinates from: ", opts$spatial_coords_csv)
  # Every column is read as character (see read_coords_csv) so R cannot mangle a long numeric spot
  # ID; a headerless Space Ranger v1 tissue_positions_list.csv is recognised rather than losing its
  # first spot to the header.
  coords_read <- read_coords_csv(opts$spatial_coords_csv)
  sp_coords_raw <- coords_read$frame
  coord_cols <- resolve_coord_cols(colnames(sp_coords_raw)[-1], opts$spatial_coords_csv)
  log_msg("Coordinate columns: ", paste(coord_cols, collapse = ", "),
          " (of ", paste(colnames(sp_coords_raw), collapse = ", "), ")")
  sp_coords_df <- data.frame(
    x = as.numeric(sp_coords_raw[[coord_cols[1]]]),
    y = as.numeric(sp_coords_raw[[coord_cols[2]]]),
    row.names = sp_coords_raw[, 1]
  )

  # Ensure genes x spots orientation
  common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  if (length(common_spots) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_spots <- intersect(colnames(sp_counts_mat), rownames(sp_coords_df))
  }
  if (length(common_spots) == 0) {
    stop(id_mismatch_msg("spot IDs", "spatial counts", colnames(sp_counts_mat), "coordinates", rownames(sp_coords_df)))}

  n_spots_in_counts <- ncol(sp_counts_mat)
  if (length(common_spots) < n_spots_in_counts) {
    warns <- c(warns, sprintf(
      "%d of %d spots in the spatial counts have no row in %s and were left out.",
      n_spots_in_counts - length(common_spots), n_spots_in_counts, opts$spatial_coords_csv))
  }

  # Background spots are not tissue. A coordinates file that carries Space Ranger's in_tissue flag
  # (tissue_positions.csv, or the headerless tissue_positions_list.csv) says which spots are glass;
  # when the counts include them (a raw_feature_bc_matrix, a CELLxGENE export) they are left out and
  # counted, as tools/worker_utils.py's keep_in_tissue() does for the Python workers. Deconvolving
  # them fitted every background spot with a composition of cell types that are not there.
  n_spots_before_tissue_filter <- length(common_spots)
  n_spots_off_tissue <- 0L
  flag_col <- colnames(sp_coords_raw)[-1][tolower(colnames(sp_coords_raw)[-1]) == "in_tissue"]
  if (length(flag_col) > 0L) {
    flag <- in_tissue_values(sp_coords_raw[[flag_col[1]]])
    names(flag) <- sp_coords_raw[, 1]
    on_tissue <- flag[common_spots]
    on_tissue <- !is.na(on_tissue) & on_tissue == 1
    if (!any(on_tissue)) {
      seen <- utils::head(sort(unique(as.character(sp_coords_raw[[flag_col[1]]]))), 8)
      stop("Column '", flag_col[1], "' of ", opts$spatial_coords_csv, " marks none of the ",
           length(common_spots), " spots in the counts as in tissue (values seen: ",
           paste(seen, collapse = ", "), "); fix the column so in-tissue spots are 1, or remove it ",
           "if every spot is tissue.")
    }
    n_spots_off_tissue <- sum(!on_tissue)
    if (n_spots_off_tissue > 0L) {
      common_spots <- common_spots[on_tissue]
      warns <- c(warns, sprintf(
        "%d of %d spots have %s == 0 in %s (background outside the tissue) and were left out; %d in-tissue spots were deconvolved.",
        n_spots_off_tissue, n_spots_before_tissue_filter, flag_col[1], opts$spatial_coords_csv,
        length(common_spots)))
      log_msg("Left out ", n_spots_off_tissue, " of ", n_spots_before_tissue_filter,
              " spots with ", flag_col[1], " == 0")
    }
  }

  sp_counts_mat <- sp_counts_mat[, common_spots, drop = FALSE]
  sp_coords_df <- sp_coords_df[common_spots, 1:2, drop = FALSE]
  colnames(sp_coords_df) <- c("x", "y")

  log_msg("Building spatial SingleCellExperiment: ", ncol(sp_counts_mat),
          " spots, ", nrow(sp_counts_mat), " genes")

  sp_sce <- SingleCellExperiment(
    assays = list(counts = sp_counts_mat),
    colData = data.frame(row.names = common_spots)
  )

  # --- Find marker genes (mean.AUC > 0.8, the SPOTlight vignette's recipe) ---
  markers <- marker_table(ref_sce, as.matrix(logcounts(ref_sce)), as.character(ref_sce$type))
  mgs_df <- markers$mgs
  log_msg("Selected ", nrow(mgs_df), " markers across ",
          length(unique(as.character(mgs_df$cluster))), " cell types")

  # The genes trainNMF will factorise: markers present, and non-zero, in both matrices -- the same
  # intersect() and .filter() SPOTlight applies before it seeds the model.
  genes_nonzero <- intersect(rownames(ref_counts_mat)[rowSums(ref_counts_mat) > 0],
                             rownames(sp_counts_mat)[rowSums(sp_counts_mat) > 0])
  model_genes <- intersect(unique(mgs_df$gene), genes_nonzero)
  n_markers_per_type <- as.list(table(factor(as.character(mgs_df$cluster),
                                             levels = sort(unique(as.character(cell_types))))))
  seeding <- seed_genes_per_type(mgs_df, model_genes, opts$n_top, type_levels)
  n_seeds_per_type <- lapply(seeding$seeds, length)
  no_usable <- type_levels[vapply(seeding$top, length, integer(1)) == 0L]
  all_shared <- setdiff(type_levels[vapply(seeding$seeds, length, integer(1)) == 0L], no_usable)
  unseeded <- c(no_usable, all_shared)
  n_seed_genes <- length(unique(unlist(seeding$seeds, use.names = FALSE)))
  if (length(markers$by_rank) > 0L) {
    warns <- c(warns, sprintf(
      "%d cell type(s) had no gene with mean.AUC > %s and were seeded with their top %d genes by mean.AUC: %s.",
      length(markers$by_rank), MARKER_AUC_MIN, MARKER_TOP_BY_RANK, paste(markers$by_rank, collapse = ", ")))
  }
  if (length(no_usable) > 0L) {
    warns <- c(warns, sprintf(
      "%d cell type(s) have no marker gene that is expressed in both the reference and the spatial data, so their NMF topic starts unseeded: %s.",
      length(no_usable), paste(no_usable, collapse = ", ")))
  }
  if (length(all_shared) > 0L) {
    warns <- c(warns, sprintf(
      "%d cell type(s) start with an unseeded NMF topic because every one of their top n_top=%d markers is also among another cell type's top %d, and SPOTlight seeds a topic only with genes no other type's list holds: %s.",
      length(all_shared), opts$n_top, opts$n_top, paste(all_shared, collapse = ", ")))
  }
  # The marker table as SPOTlight is handed it: cluster as a factor over every reference type, so
  # .init_nmf's split() keeps a type left with no marker gene as an empty entry and its positional
  # mgs[ks != k] still lines up with the type order above.
  mgs_df$cluster <- factor(as.character(mgs_df$cluster), levels = type_levels)

  # --- Run SPOTlight ---
  # min_cont is SPOTlight's min_prop: in each spot a cell type whose NNLS share is below it is set
  # to zero and the rest are rescaled to sum to 1. SPOTlight 1.x has no min_cont argument; passing
  # one sent it through `...` to NMF::nmf, which refused it, and a tryCatch keyed on that message
  # re-ran without it -- so every run used min_prop's 0.01 while the payload implied 0.09. n_top was
  # parsed and never passed, so SPOTlight seeded each topic with every marker instead.
  spotlight_version <- tryCatch(as.character(utils::packageVersion("SPOTlight")),
                                error = function(e) "unknown")
  log_msg("Running SPOTlight ", spotlight_version, " (n_top = ", opts$n_top, ", min_prop = ",
          opts$min_cont, ", weight_id = 'mean.AUC')...")
  res <- SPOTlight(
    x = ref_sce, y = sp_sce, groups = ref_sce$type,
    mgs = mgs_df, weight_id = "mean.AUC", group_id = "cluster",
    gene_id = "gene", n_top = opts$n_top, min_prop = opts$min_cont
  )

  # --- Extract proportions ---
  log_msg("Extracting deconvolution proportions...")
  mat_prop <- res$mat
  prop_df <- as.data.frame(mat_prop)
  prop_df$spot <- rownames(prop_df)

  # A spot in which every cell type fell below min_cont has nothing left to rescale, and
  # SPOTlight writes it as 0/0 = NaN (NA in the CSV). It is counted, not filled in.
  n_unassigned <- sum(rowSums(is.na(as.matrix(mat_prop))) > 0)
  if (n_unassigned > 0L) {
    warns <- c(warns, sprintf(
      "%d of %d spots have no proportions (NA): every cell type in them fell below min_cont=%s. Lower min_cont to keep them.",
      n_unassigned, nrow(mat_prop), format(opts$min_cont)))
  }

  # Save proportions CSV
  prop_path <- file.path(opts$output_dir, "spotlight_proportions.csv")
  write_atomically(prop_path, function(p) write.csv(prop_df, p, row.names = FALSE, quote = TRUE))

  # Save NMF model object
  rds_path <- file.path(opts$output_dir, "spotlight_result.rds")
  write_atomically(rds_path, function(p) saveRDS(res, file = p))

  log_msg("Saved proportions to: ", prop_path)
  log_msg("Saved RDS to: ", rds_path)

  # --- Build summary ---
  n_spots <- nrow(mat_prop)
  n_celltypes <- ncol(mat_prop)

  analysis <- sprintf(
    paste0("SPOTlight mapped %d cell types across %d spots. Proportions below min_cont=%s were set to ",
           "zero and each spot rescaled to sum to 1; each NMF topic was seeded with up to %d markers ",
           "(%d markers in all, mean.AUC > %s; %d genes in the model; %d seed genes once SPOTlight ",
           "drops the genes two types' top lists share)."),
    n_celltypes, n_spots, format(opts$min_cont), opts$n_top, nrow(mgs_df), MARKER_AUC_MIN,
    length(model_genes), n_seed_genes)
  if (length(unseeded) > 0L) {
    analysis <- paste0(analysis, sprintf(" %d cell type(s) started with an unseeded topic: %s.",
                                         length(unseeded), paste(unseeded, collapse = ", ")))
  }
  if (n_spots_off_tissue > 0L) {
    analysis <- paste0(analysis, sprintf(
      " %d of %d spots were background (in_tissue == 0) and were left out.",
      n_spots_off_tissue, n_spots_before_tissue_filter))
  }
  if (length(markers$by_rank) > 0L) {
    analysis <- paste0(analysis, sprintf(" %d cell type(s) had no marker above the threshold and were seeded by rank.",
                                         length(markers$by_rank)))
  }
  if (n_unassigned > 0L) {
    analysis <- paste0(analysis, sprintf(" %d spots had every cell type below min_cont and carry NA.",
                                         n_unassigned))
  }

  params <- list(
    n_top                = opts$n_top,
    min_cont             = opts$min_cont,
    min_prop             = opts$min_cont,
    drop_unlabeled       = isTRUE(opts$drop_unlabeled),
    method               = paste0(METHOD_NAME, ", SPOTlight ", spotlight_version),
    used_fallback        = FALSE,
    marker_method        = markers$method,
    marker_auc_min       = MARKER_AUC_MIN,
    marker_top_by_rank   = MARKER_TOP_BY_RANK,
    weight_id            = "mean.AUC",
    coordinates_header   = coords_read$header,
    coordinate_columns   = I(as.character(coord_cols)),
    reference_cells_regrouped_by_type = reference_reordered
  )
  if (n_spots_off_tissue > 0L) {
    params$in_tissue_filter <- list(
      n_spots_supplied            = n_spots_before_tissue_filter,
      n_spots_off_tissue_dropped  = n_spots_off_tissue,
      n_spots_used                = n_spots_before_tissue_filter - n_spots_off_tissue
    )
  }

  summary <- list(
    n_cell_types                   = n_celltypes,
    n_markers                      = nrow(mgs_df),
    n_markers_per_cell_type        = n_markers_per_type,
    cell_types_seeded_by_rank      = I(as.character(markers$by_rank)),
    cell_types_without_usable_markers = I(as.character(unseeded)),
    cell_types_whose_seeds_were_all_shared = I(as.character(all_shared)),
    n_seed_genes                   = n_seed_genes,
    n_seed_genes_per_cell_type     = n_seeds_per_type,
    n_spots_without_proportions    = n_unassigned
  )
  if (!is.null(cell_type_renames)) summary$cell_type_renames <- cell_type_renames

  result <- list(
    status       = "ok",
    tool         = "spotlight",
    task         = "deconvolution",
    data         = list(
      n_spots                   = n_spots,
      n_cell_types              = n_celltypes,
      n_spots_in_counts         = n_spots_in_counts,
      n_ref_cells               = length(common_cells),
      n_ref_cells_in_counts     = n_ref_in_counts,
      n_ref_cells_annotated     = n_ref_annotated,
      n_ref_cells_unlabeled_dropped = n_unlabeled,
      n_genes_spatial           = nrow(sp_counts_mat),
      n_genes_reference         = nrow(ref_counts_mat),
      n_genes_in_model          = length(model_genes)
    ),
    output_files = list(proportions_csv = prop_path, result_rds = rds_path),
    params       = params,
    summary      = summary,
    analysis     = analysis
  )
  if (length(warns) > 0L) result$warnings <- I(warns)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 0 && args[[1]] %in% c("--help", "-h")) parse_args(args)  # prints usage to stdout, quits

  # parse_args() runs inside the handler so that a bad n_top / min_cont / unknown flag comes back
  # as a JSON error the portal can read, not as bare R output on stderr.
  res <- tryCatch(with_r_traceback({
    sink(stderr())
    opts <- parse_args(args)
    result <- run_spotlight(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "spotlight",
      task      = "deconvolution",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA, force = TRUE), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
