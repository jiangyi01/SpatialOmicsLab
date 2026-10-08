#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(STdeconvolve)
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
  message(sprintf("[stdeconvolve-worker] %s", msg))
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

# Which axis of the counts CSV holds the genes. STdeconvolve takes one matrix and no second
# input to intersect it with, so the answer is read off the identifiers themselves: Ensembl IDs
# and gene-symbol-shaped names are gene evidence; ACGT barcodes (anywhere in the name, so a
# sample prefix does not hide them), gem-group suffixes on all-lowercase IDs (Xenium
# "aaaaadoa-1"), Visium HD bins ("s_008um_00301_00321-1"), bare or dotted integers, "10x12" grid
# positions and spot_1/cell_1 style names are spot evidence. A name that looks like a spot never
# counts as a gene, so a symbol-shaped Visium barcode cannot vote twice. The previous rule
# transposed only when the ROW names were Visium-style ACGT barcodes, so a spots x genes table
# from Xenium, Visium HD or a sample-prefixed atlas was fit with the genes as the documents.
COUNTS_ORIENTATIONS <- c("auto", "genes_x_spots", "spots_x_genes")
ORIENTATION_EVIDENCE_CUT <- 0.6

looks_like_spot_id <- function(x) {
  grepl("[ACGT]{10,}", x) |
    grepl("^[0-9]+([_.-][0-9]+)*$", x) |
    grepl("^-?[0-9]+(\\.[0-9]+)?x-?[0-9]+(\\.[0-9]+)?$", x) |
    grepl("^[a-z]+-[0-9]+$", x) |
    grepl("^s_[0-9]+um_[0-9]+_[0-9]+(-[0-9]+)?$", x) |
    grepl("^(spot|cell|bin|bead|barcode|pixel|loc|location)[_.-]?[0-9]+$", x, ignore.case = TRUE)
}

looks_like_gene_id <- function(x) {
  !looks_like_spot_id(x) & (
    grepl("^ENS[A-Z]*G[0-9]{6,}(\\.[0-9]+)?$", x) |
      grepl("^[A-Za-z][A-Za-z0-9]*([-.][A-Za-z0-9]+)*$", x) |
      grepl("^[0-9]+[A-Z][0-9]+Rik[0-9]*$", x)
  )
}

# Up to 200 names spread over the whole axis, not only its head: a file sorted by name can open
# with a run of one kind.
axis_sample <- function(ids, n = 200L) {
  ids <- as.character(ids)
  if (length(ids) <= n) return(ids)
  ids[unique(round(seq(1, length(ids), length.out = n)))]
}

axis_evidence <- function(ids) {
  s <- axis_sample(ids)
  if (length(s) == 0) return(c(gene = 0, spot = 0))
  c(gene = mean(looks_like_gene_id(s)), spot = mean(looks_like_spot_id(s)))
}

# "genes_x_spots", "spots_x_genes", or NA when the two axes do not tell them apart (both look
# like genes, neither does, or the evidence points both ways).
detect_counts_orientation <- function(row_ids, col_ids, cut = ORIENTATION_EVIDENCE_CUT) {
  r <- axis_evidence(row_ids)
  cc <- axis_evidence(col_ids)
  for_genes_x_spots <- (r[["gene"]] >= cut) + (cc[["spot"]] >= cut)
  for_spots_x_genes <- (cc[["gene"]] >= cut) + (r[["spot"]] >= cut)
  orientation <- if (for_genes_x_spots > 0 && for_spots_x_genes == 0) {
    "genes_x_spots"
  } else if (for_spots_x_genes > 0 && for_genes_x_spots == 0) {
    "spots_x_genes"
  } else {
    NA_character_
  }
  list(orientation = orientation, rows = r, cols = cc)
}

# First few names of an axis, for a message.
head_names <- function(ids, n = 5L) {
  ids <- as.character(ids)
  shown <- paste(utils::head(ids, n), collapse = ", ")
  if (length(ids) > n) paste0(shown, ", ... (", length(ids), " in all)") else shown
}

parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    output_dir         = NULL,
    n_topics           = 10L,
    n_top_genes        = 1000L,
    remove_below       = NULL,
    seed               = 42L,
    counts_orientation = "auto"
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
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-topics") {
      opts$n_topics <- as.integer(val)
    } else if (key == "--n-top-genes") {
      opts$n_top_genes <- as.integer(val)
    } else if (key == "--remove-below") {
      opts$remove_below <- as.numeric(val)
    } else if (key == "--seed") {
      opts$seed <- as.integer(val)
    } else if (key == "--counts-orientation") {
      opts$counts_orientation <- val
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# write.csv to <path>.partial, then rename: an interrupted write never sits at the real name.
write_csv_atomic <- function(df, path) {
  tmp <- paste0(path, ".partial")
  write.csv(df, tmp, row.names = FALSE, quote = TRUE)
  if (!file.rename(tmp, path)) {
    stop(sprintf("Could not move %s into place at %s", tmp, path))
  }
  invisible(path)
}

run_stdeconvolve <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$output_dir)) {
    stop("STdeconvolve requires --spatial-counts-csv and --output-dir")
  }

  if (!file.exists(opts$spatial_counts_csv)) {
    stop(sprintf("Input file not found: %s", opts$spatial_counts_csv))
  }

  if (!(opts$counts_orientation %in% COUNTS_ORIENTATIONS)) {
    stop(sprintf("Unsupported counts_orientation='%s'. Valid counts_orientation values: %s.",
                 opts$counts_orientation, paste0("'", COUNTS_ORIENTATIONS, "'", collapse = ", ")))
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial counts ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  counts_mat <- as.matrix(counts_df)

  # STdeconvolve expects genes x spots (per the package vignette: rows=genes,
  # cols=cells/spots; restrictCorpus and fitLDA both rely on this orientation).
  # convert_h5ad_to_csv writes genes x spots by default and spots x genes with
  # transpose_counts=False. Which one arrived is read off the identifiers of both
  # axes (detect_counts_orientation above), or taken from --counts-orientation.
  warnings <- character(0)
  if (opts$counts_orientation == "auto") {
    found <- detect_counts_orientation(rownames(counts_mat), colnames(counts_mat))
    if (is.na(found$orientation)) {
      stop(paste0(
        "Cannot tell which axis of the counts CSV holds the genes: ",
        sprintf("row names look %.0f%% gene-like / %.0f%% spot-like (%s); ",
                100 * found$rows[["gene"]], 100 * found$rows[["spot"]], head_names(rownames(counts_mat))),
        sprintf("column names look %.0f%% gene-like / %.0f%% spot-like (%s). ",
                100 * found$cols[["gene"]], 100 * found$cols[["spot"]], head_names(colnames(counts_mat))),
        "Pass counts_orientation='genes_x_spots' or 'spots_x_genes' to say which it is."
      ))
    }
    counts_orientation <- found$orientation
    orientation_source <- "detected from the row and column names"
  } else {
    counts_orientation <- opts$counts_orientation
    orientation_source <- "given"
  }
  if (counts_orientation == "spots_x_genes") {
    log_msg("Counts are spots x genes (", orientation_source, "); transposing to genes x spots")
    counts_mat <- t(counts_mat)
  } else {
    log_msg("Counts are genes x spots (", orientation_source, "); no transpose")
  }

  n_genes_raw <- nrow(counts_mat)
  n_spots <- ncol(counts_mat)
  log_msg("Loaded: ", n_genes_raw, " genes x ", n_spots, " spots")

  # --- Filter genes: remove low-count and non-informative genes ---
  # restrictCorpus expects genes x spots; rows are filtered. removeBelow=0.05
  # is too aggressive for sparse platforms (slide-seq, MERFISH); auto-relax to
  # 0.01 when median per-spot UMI is low (< 500). Override with --remove-below.
  log_msg("Filtering genes...")
  counts_sparse <- as(counts_mat, "dgCMatrix")
  rm(counts_df, counts_mat)

  # A spot with no counts at all is an empty LDA document: fitLDA stops on it ("Each row (pixel)
  # of `counts` needs to contain at least one non-zero entry"), and a whole-capture-area slide
  # carries thousands (3,396 of 4,992 on the library's gastrocnemius section). It has no topic
  # mixture to estimate, so it is left out -- counted in data.n_spots_empty, named in warnings --
  # before the gene filter, which measures detection rates and the median UMI over the spots.
  spot_umi <- Matrix::colSums(counts_sparse)
  empty_spots <- colnames(counts_sparse)[spot_umi == 0]
  n_spots_empty <- length(empty_spots)
  if (n_spots_empty == n_spots) {
    stop(sprintf("All %d spots in the counts CSV have zero counts; there is nothing to deconvolve.", n_spots))
  }
  if (n_spots_empty > 0) {
    counts_sparse <- counts_sparse[, spot_umi > 0, drop = FALSE]
    spot_umi <- spot_umi[spot_umi > 0]
    warnings <- c(warnings, paste0(
      n_spots_empty, " of ", n_spots, " spots have zero counts and were left out: LDA cannot place ",
      "an empty spot, and they have no row in stdeconvolve_theta.csv (", head_names(empty_spots), ")."
    ))
    log_msg("Left out ", n_spots_empty, " spots with zero counts")
  }

  remove_below <- opts$remove_below
  if (is.null(remove_below) || is.na(remove_below)) {
    median_umi <- stats::median(spot_umi)
    # as.numeric("abc") is NA with a warning, and that warning goes to stderr. Distinguish a
    # value we could not read from no value at all, or a typo reads back as "not given".
    unreadable <- !is.null(opts$remove_below)
    remove_below <- if (median_umi < 500) 0.01 else 0.05
    # log_msg writes to stderr, which base_mcp drops from a status-ok payload, so a caller
    # would otherwise never learn which of the two filters ran on their data.
    warnings <- c(warnings, if (unreadable) {
      paste0("--remove-below was given but is not a number, so the gene filter was chosen ",
             "from the data instead: removeBelow = ", remove_below,
             " (median per-spot UMI = ", round(median_umi, 1), ").")
    } else {
      paste0("removeBelow was chosen from the data, not given: ", remove_below,
             " (median per-spot UMI = ", round(median_umi, 1),
             "). Pass --remove-below to set it yourself.")
    })
    log_msg("Auto-set remove_below=", remove_below,
            " (median per-spot UMI = ", round(median_umi, 1), ")")
  }
  corpus <- restrictCorpus(counts_sparse,
                           removeAbove = 1.0,
                           removeBelow = remove_below,
                           nTopOD = opts$n_top_genes)
  n_genes_filtered <- nrow(corpus)
  if (n_genes_filtered == 0) {
    stop(sprintf(paste0(
      "restrictCorpus kept none of the %d genes (removeBelow = %s: no gene detected in more than that ",
      "fraction of spots was over-dispersed), so there is nothing to fit. Lower remove_below."),
      n_genes_raw, remove_below))
  }

  # restrictCorpus keeps every spot but only the over-dispersed genes, so a spot whose counts all
  # fell in the genes it dropped is an empty document too. Same treatment, counted separately.
  corpus_umi <- Matrix::colSums(corpus)
  no_corpus_spots <- colnames(corpus)[corpus_umi == 0]
  n_spots_no_corpus_counts <- length(no_corpus_spots)
  if (n_spots_no_corpus_counts == ncol(corpus)) {
    stop(sprintf(paste0(
      "None of the %d spots with counts has a count in the %d over-dispersed genes restrictCorpus kept ",
      "(removeBelow = %s, n_top_genes = %d). Lower remove_below or raise n_top_genes."),
      ncol(corpus), n_genes_filtered, remove_below, opts$n_top_genes))
  }
  if (n_spots_no_corpus_counts > 0) {
    corpus <- corpus[, corpus_umi > 0, drop = FALSE]
    warnings <- c(warnings, paste0(
      n_spots_no_corpus_counts, " spots have counts, but none in the ", n_genes_filtered,
      " over-dispersed genes kept for fitting, and were left out (", head_names(no_corpus_spots),
      "). Lowering remove_below or raising n_top_genes keeps more genes."
    ))
    log_msg("Left out ", n_spots_no_corpus_counts, " spots with no counts in the kept genes")
  }
  n_spots_kept <- ncol(corpus)
  log_msg("After filtering: ", n_genes_filtered, " genes x ", n_spots_kept, " spots retained")

  # --- Fit LDA ---
  # STdeconvolve's fitLDA expects DOCS x TERMS (spots x genes). The corpus from
  # restrictCorpus is genes x spots, so transpose. Prior bug: fitLDA without t()
  # treated genes as documents, producing theta=(n_genes_filtered x n_topics)
  # instead of theta=(n_spots x n_topics). On slide-seq aorta this gave
  # 574x15 with gene names (AAMP, AASS, ...) as the "spot" column.
  log_msg("Fitting LDA with ", opts$n_topics, " topics...")
  set.seed(opts$seed)
  corpus_docs <- t(as.matrix(corpus))  # spots (docs) x genes (terms)
  # plot = FALSE: with its default TRUE, fitLDA prints a ggplot, which under Rscript opens a pdf
  # device and leaves Rplots.pdf in whatever directory the caller launched us from.
  # verbose = FALSE: its progress lines are system("echo ...") calls, which reach this process's
  # stdout -- where the JSON payload goes -- past the sink(stderr()) that guards everything else.
  lda_model <- fitLDA(corpus_docs, Ks = opts$n_topics, seed = opts$seed, plot = FALSE, verbose = FALSE)

  # Extract the best model (fitLDA returns a list when Ks is a single value)
  model <- optimalModel(models = lda_model, opt = opts$n_topics)

  # --- Extract theta (spot x topic proportions) and beta (topic x gene) ---
  log_msg("Extracting theta and beta matrices...")
  results <- getBetaTheta(model, perc.filt = 0.05, betaScale = 1000)
  theta <- results$theta  # spots x topics
  beta  <- results$beta   # topics x genes

  # --- Save outputs ---
  theta_df <- as.data.frame(theta)
  theta_df$spot <- rownames(theta_df)
  theta_path <- file.path(opts$output_dir, "stdeconvolve_theta.csv")
  write_csv_atomic(theta_df, theta_path)

  beta_df <- as.data.frame(beta)
  beta_df$topic <- rownames(beta_df)
  beta_path <- file.path(opts$output_dir, "stdeconvolve_beta.csv")
  write_csv_atomic(beta_df, beta_path)

  log_msg("Saved theta to: ", theta_path)
  log_msg("Saved beta to: ", beta_path)

  # --- Summary ---
  n_topics_fit <- ncol(theta)
  # theta is the artifact the caller gets, so its row count is the number of spots the run
  # actually covers. n_spots above is the input's, measured before any spot was left out.
  n_spots_used <- nrow(theta)

  # Determine dominant topic per spot
  dominant <- apply(theta, 1, function(row) colnames(theta)[which.max(row)])
  dominant_counts <- as.list(table(dominant))

  list(
    status       = "ok",
    tool         = "stdeconvolve",
    task         = "deconvolution",
    data         = list(
      n_spots          = n_spots,
      n_spots_used     = n_spots_used,
      n_spots_empty    = n_spots_empty,
      n_spots_no_corpus_counts = n_spots_no_corpus_counts,
      n_genes_raw      = n_genes_raw,
      n_genes_filtered = n_genes_filtered,
      n_topics         = n_topics_fit
    ),
    output_files = list(
      theta_csv        = theta_path,
      beta_csv         = beta_path
    ),
    params       = list(
      n_topics         = opts$n_topics,
      n_top_genes      = opts$n_top_genes,
      remove_below     = remove_below,
      seed             = opts$seed,
      counts_orientation = counts_orientation,
      counts_orientation_source = orientation_source
    ),
    warnings     = I(warnings),
    summary      = list(
      n_topics_fit     = n_topics_fit,
      dominant_counts  = dominant_counts
    ),
    analysis     = paste0(
      "STdeconvolve identified ", n_topics_fit, " latent cell-type topics ",
      "across ", n_spots_used, " spots using reference-free LDA. ",
      n_genes_filtered, " over-dispersed genes were used for fitting",
      " (removeBelow = ", remove_below, "). ",
      "The counts were read as ", sub("_x_", " x ", counts_orientation), " (", orientation_source, ")",
      if (n_spots_empty + n_spots_no_corpus_counts > 0) {
        paste0("; of the ", n_spots, " input spots, ", n_spots_empty, " with zero counts and ",
               n_spots_no_corpus_counts, " with no counts in the kept genes were left out")
      } else {
        ""
      },
      "."
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  res <- tryCatch(with_r_traceback({
    sink(stderr())
    result <- run_stdeconvolve(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "stdeconvolve",
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
