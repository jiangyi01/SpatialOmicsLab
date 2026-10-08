#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(Redeconve)
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
  message(sprintf("[redeconve-worker] %s", msg))
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


# A reference cell whose label is NA, empty, or a spelling of "missing" is not a cell type. read.csv
# turns the text NA into NA, which came out as a fourth cell type called NA (null in the JSON) at
# status ok; the empty field pandas writes for NaN became a type called "" and the run died on a bare
# "subscript out of bounds". The spellings match worker_utils.drop_unlabeled.
is_missing_label <- function(x) {
  s <- trimws(tolower(as.character(x)))
  is.na(x) | is.na(s) | s %in% c("", "nan", "none", "na", "<na>")
}

# The n_top genes (of `genes`) whose log1p(CPM) varies most across the reference cells. Library sizes
# are each cell's total over every reference gene; ties are broken by gene name so the pick is stable.
top_variable_genes <- function(ref_mat, genes, n_top) {
  lib <- colSums(ref_mat)
  lib[lib == 0] <- 1
  x <- log1p(sweep(ref_mat[genes, , drop = FALSE], 2, lib, "/") * 1e6)
  n <- ncol(x)
  v <- if (n > 1) (rowSums(x^2) - rowSums(x)^2 / n) / (n - 1) else rep(0, nrow(x))
  v[!is.finite(v)] <- 0
  ord <- order(-v, genes)
  genes[ord[seq_len(n_top)]]
}

parse_args <- function(args) {
  # n_top_genes 0 = every gene the reference and the slide share, which is what Redeconve's own
  # genemode = "default" uses and what this worker always ran. The old default of 2000 was echoed
  # in params and never applied.
  opts <- list(
    spatial_counts_csv = NULL,
    ref_counts_csv     = NULL,
    ref_celltypes_csv  = NULL,
    output_dir         = NULL,
    n_top_genes        = 0L,
    seed               = 0L,
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
    } else if (key == "--n-top-genes") {
      opts$n_top_genes <- suppressWarnings(as.integer(val))
      if (is.na(opts$n_top_genes) || opts$n_top_genes < 0L) {
        stop(sprintf("--n-top-genes must be a whole number >= 0 (0 = every shared gene); got '%s'", val))
      }
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else if (key == "--drop-unlabeled") {
      opts$drop_unlabeled <- tolower(val) %in% c("true", "1", "yes")
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_redeconve <- function(opts) {
  if (is.null(opts$spatial_counts_csv) || is.null(opts$ref_counts_csv) ||
      is.null(opts$ref_celltypes_csv) || is.null(opts$output_dir)) {
    stop("Redeconve requires --spatial-counts-csv, --ref-counts-csv, --ref-celltypes-csv, and --output-dir")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load reference data ---
  log_msg("Reading reference counts from: ", opts$ref_counts_csv)
  ref_counts_df <- read.csv(opts$ref_counts_csv, row.names = 1, check.names = FALSE)
  ref_counts_mat <- as.matrix(ref_counts_df)

  log_msg("Reading reference cell types from: ", opts$ref_celltypes_csv)
  ref_ct_df <- read.csv(opts$ref_celltypes_csv, row.names = 1, check.names = FALSE)
  cell_types <- ref_ct_df[, 1]
  names(cell_types) <- rownames(ref_ct_df)

  # Ensure matching cell IDs
  common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  if (length(common_cells) == 0) {
    ref_counts_mat <- t(ref_counts_mat)
    common_cells <- intersect(colnames(ref_counts_mat), names(cell_types))
  }
  if (length(common_cells) == 0) {
    stop(id_mismatch_msg("cell IDs", "reference counts", colnames(ref_counts_mat), "cell type annotations", names(cell_types)))}

  ref_counts_mat <- ref_counts_mat[, common_cells, drop = FALSE]
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
    ref_counts_mat <- ref_counts_mat[, !unlabeled, drop = FALSE]
    cell_types <- cell_types[!unlabeled]
    warnings <- c(warnings, sprintf(paste0("%d of %d reference cells had no cell-type label and were left out ",
                                           "(drop_unlabeled=True); %d were used."),
                                    n_unlabeled, length(common_cells), length(cell_types)))
    if (length(cell_types) == 0) stop("Every reference cell is unlabeled; there is nothing to deconvolve against.")
  }
  cell_types <- stats::setNames(as.character(cell_types), names(cell_types))

  log_msg("Reference: ", length(cell_types), " cells, ",
          length(unique(cell_types)), " cell types")

  # --- Load spatial data ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  sp_counts_mat <- as.matrix(sp_counts_df)

  # Redeconve wants genes on the rows of both matrices. Which axis holds them is read off the
  # reference's own gene names, the way the cell orientation above is read off the annotations --
  # not off which axis is longer. That shape rule is right only where the spots outnumber the
  # genes, and the MERFISH panel this repo stages is 649 genes x 78,329 cells, the other way round.
  common_genes <- intersect(rownames(sp_counts_mat), rownames(ref_counts_mat))
  if (length(common_genes) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_genes <- intersect(rownames(sp_counts_mat), rownames(ref_counts_mat))
  }
  if (length(common_genes) == 0) {
    stop(id_mismatch_msg("gene IDs", "spatial counts", rownames(sp_counts_mat),
                         "reference counts", rownames(ref_counts_mat)))
  }

  log_msg("Spatial: ", ncol(sp_counts_mat), " spots, ", nrow(sp_counts_mat), " genes")

  unique_types <- unique(cell_types)

  # --- Run Redeconve ---
  log_msg("Running Redeconve deconvolution...")

  # n_top_genes was echoed in params for every run and never reached deconvoluting(), which used
  # every shared gene (genemode = "default"). A positive value below the shared count now selects;
  # 0, or a value at or above the shared count, keeps every shared gene exactly as before.
  n_common <- length(common_genes)
  if (opts$n_top_genes > 0L && opts$n_top_genes < n_common) {
    genes_used <- top_variable_genes(ref_counts_mat, common_genes, opts$n_top_genes)
    genemode <- "customized"
    gene_selection <- sprintf(paste0("top %d of %d shared genes by variance of log1p(CPM) across the ",
                                     "reference cells (n_top_genes=%d)"),
                              length(genes_used), n_common, opts$n_top_genes)
  } else {
    genes_used <- common_genes
    genemode <- "default"
    gene_selection <- if (opts$n_top_genes > 0L) {
      sprintf("all %d shared genes (n_top_genes=%d is not below the shared count)", n_common, opts$n_top_genes)
    } else {
      sprintf("all %d shared genes (n_top_genes=0)", n_common)
    }
  }

  sp_subset <- sp_counts_mat[genes_used, , drop = FALSE]
  ref_subset <- ref_counts_mat[genes_used, , drop = FALSE]

  log_msg("Using ", length(genes_used), " genes (", gene_selection, "), ",
          ncol(ref_subset), " ref cells, ", ncol(sp_subset), " spots")

  result <- deconvoluting(
    ref       = ref_subset,
    st        = sp_subset,
    genemode  = genemode,
    gene.list = genes_used,
    hpmode    = "default",
    dopar     = FALSE,
    ncores    = 1L
  )

  # Extract cell-to-spot weight matrix (cells x spots)
  weight_mat <- as.matrix(result)

  # Save raw weight matrix
  weight_path <- file.path(opts$output_dir, "redeconve_weights.csv")
  write.csv(weight_mat, weight_path, quote = TRUE)

  # Aggregate weights by cell type to get proportions (cell types x spots)
  proportions <- matrix(0, nrow = length(unique_types), ncol = ncol(weight_mat))
  rownames(proportions) <- unique_types
  colnames(proportions) <- colnames(weight_mat)
  for (ct in unique_types) {
    ct_cells <- names(cell_types[cell_types == ct])
    ct_cells_present <- intersect(ct_cells, rownames(weight_mat))
    if (length(ct_cells_present) > 0) {
      proportions[ct, ] <- colSums(weight_mat[ct_cells_present, , drop = FALSE])
    }
  }
  # Normalize columns to sum to 1. A spot whose every reference-cell abundance came back zero has no
  # composition to normalise; it keeps a row of zeros in the CSV, and the payload counts those rows
  # rather than letting them pass for proportions.
  col_sums <- colSums(proportions)
  zero_spots <- colnames(proportions)[col_sums == 0]
  n_zero_spots <- length(zero_spots)
  col_sums[col_sums == 0] <- 1
  proportions <- sweep(proportions, 2, col_sums, "/")
  proportions <- as.data.frame(t(proportions))  # spots x cell_types

  # Save proportions
  prop_path <- file.path(opts$output_dir, "redeconve_proportions.csv")
  write.csv(proportions, prop_path, quote = TRUE)

  # Summary statistics per cell type
  mean_props <- colMeans(proportions, na.rm = TRUE)
  summary_df <- data.frame(
    cell_type       = names(mean_props),
    mean_proportion = as.numeric(mean_props),
    stringsAsFactors = FALSE
  )
  summary_path <- file.path(opts$output_dir, "redeconve_summary.csv")
  write.csv(summary_df, summary_path, row.names = FALSE, quote = TRUE)

  # Redeconve's quadratic program and the gene pick above draw no random numbers, so the seed is set
  # and changes nothing. Said on the payload rather than echoed as though it were a setting.
  ignored <- "seed"
  warnings <- c(warnings, paste0("ignored parameter(s) seed: seed=", opts$seed, " was set, but Redeconve's ",
                                 "quadratic programming and the gene selection draw no random numbers, so ",
                                 "every seed gives the same result."))
  unlabeled_note <- if (n_unlabeled > 0) {
    sprintf(" %d unlabeled reference cells were left out (drop_unlabeled=True).", n_unlabeled)
  } else {
    ""
  }
  zero_note <- ""
  if (n_zero_spots > 0) {
    zero_note <- sprintf(paste0(" %d of %d spots got no estimate (every reference-cell abundance was zero) ",
                                "and are all-zero rows in redeconve_proportions.csv."),
                         n_zero_spots, ncol(sp_subset))
    warnings <- c(warnings, sprintf(paste0("%d of %d spots got no estimate from Redeconve (every reference-cell ",
                                           "abundance was zero; e.g. \"%s\"); their rows in ",
                                           "redeconve_proportions.csv are all zeros, not a composition."),
                                    n_zero_spots, ncol(sp_subset), zero_spots[[1]]))
  }

  list(
    status       = "ok",
    tool         = "redeconve",
    task         = "deconvolution",
    data         = list(
      n_spots             = ncol(sp_subset),
      n_cell_types        = length(unique_types),
      n_common_genes      = n_common,
      n_genes_used        = length(genes_used),
      n_ref_cells         = length(common_cells),
      n_ref_cells_used    = length(cell_types),
      n_ref_cells_unlabeled = n_unlabeled,
      n_spots_without_estimate = n_zero_spots
    ),
    output_files = list(
      proportions_csv  = prop_path,
      weights_csv      = weight_path,
      summary_csv      = summary_path
    ),
    params       = list(
      n_top_genes      = opts$n_top_genes,
      seed             = opts$seed,
      drop_unlabeled   = isTRUE(opts$drop_unlabeled),
      gene_selection   = gene_selection,
      genemode         = genemode,
      method           = "Redeconve deconvoluting() quadratic programming (hpmode = 'default', normalize = TRUE)",
      used_fallback    = FALSE,
      ignored          = I(ignored)
    ),
    summary      = list(
      n_ref_cells      = length(common_cells),
      n_ref_types      = length(unique_types),
      n_genes_used     = length(genes_used),
      cell_types       = unique_types
    ),
    warnings     = I(warnings),
    analysis     = paste0(
      "Redeconve deconvolved ", ncol(sp_subset), " spots into ",
      length(unique_types), " cell types against ", length(cell_types),
      " reference cells using ", length(genes_used), " genes: ", gene_selection, ".",
      unlabeled_note, zero_note
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args inside the handler: a bad flag used to abort R with nothing on stdout. Everything
  # before the payload goes to stderr: deconvoluting() print()s its progress and draws a progress bar
  # on stdout, which put eight lines of chatter in front of the JSON the portal reads.
  res <- tryCatch(with_r_traceback({
    sink(stderr())
    opts <- parse_args(args)
    set.seed(opts$seed)
    result <- run_redeconve(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", e$message)
    list(
      status    = "error",
      tool      = "redeconve",
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
