#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
  library(SMART)
  library(Matrix)
  library(quanteda)
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
  message(sprintf("[smart-worker] %s", msg))
}

# A reader that opens the file while it is being written sees either the previous version or the
# whole new one, never a truncated table (rename(2) is atomic on one filesystem).
write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = FALSE, quote = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " into place at ", path)
  }
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

# The upstream signature is SMART_base(..., iterations = 2000). This worker has always run 500 --
# a value that was hard-coded beside a "for faster testing" comment and never reached the payload --
# and 500 stays the default so recorded runs remain comparable; it is now a parameter and is
# reported in params.iterations.
DEFAULT_ITERATIONS <- 500L
UPSTREAM_ITERATIONS <- 2000L

# What runs. SMART has no substitute path: SMART_base is the only implementation here.
METHOD_NAME <- "SMART_base (keyATM base model seeded by marker genes)"

# Column headings the marker CSV may use. Anything else is refused by name rather than guessed at.
MARKER_GENE_COLUMNS <- c("gene", "Gene", "genes", "Genes", "gene_name", "marker")
MARKER_TYPE_COLUMNS <- c("cell_type", "celltype", "cluster", "Cluster", "CellType", "cell_type_name", "type")

parse_flag <- function(val, key) {
  low <- tolower(val)
  if (low %in% c("true", "1", "yes")) return(TRUE)
  if (low %in% c("false", "0", "no")) return(FALSE)
  stop(sprintf("%s expects true or false, got '%s'", key, val))
}

# as.integer("abc") is NA with only a warning, and an NA topic or iteration count used to travel on
# into keyATM and die there with a message about something else. Refused here, naming the flag.
parse_int_at_least <- function(val, key, lowest) {
  parsed <- suppressWarnings(as.integer(val))
  if (is.na(parsed) || parsed < lowest) {
    stop(sprintf("%s expects an integer >= %d, got '%s'", key, lowest, val))
  }
  parsed
}

parse_args <- function(args) {
  opts <- list(
    spatial_counts_csv = NULL,
    ref_counts_csv     = NULL,
    marker_genes_csv   = NULL,
    output_dir         = NULL,
    n_topics           = 10L,
    seed               = 42L,
    iterations         = DEFAULT_ITERATIONS,
    round_counts       = FALSE
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
    } else if (key == "--marker-genes-csv") {
      opts$marker_genes_csv <- val
    } else if (key == "--output-dir") {
      opts$output_dir <- val
    } else if (key == "--n-topics") {
      # 0 has always meant "the keyword topics and no unsupervised extras"; it stays accepted.
      opts$n_topics <- parse_int_at_least(val, key, 0L)
    } else if (key == "--seed") {
      opts$seed <- suppressWarnings(as.integer(val))
      if (is.na(opts$seed)) stop(sprintf("--seed expects an integer, got '%s'", val))
    } else if (key == "--iterations") {
      opts$iterations <- parse_int_at_least(val, key, 1L)
    } else if (key == "--round-counts") {
      opts$round_counts <- parse_flag(val, key)
    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

run_smart <- function(opts) {
  # --- Validate required args ---
  if (is.null(opts$spatial_counts_csv) || is.null(opts$marker_genes_csv) ||
      is.null(opts$output_dir)) {
    stop("SMART requires --spatial-counts-csv, --marker-genes-csv, and --output-dir")
  }

  for (f in c(opts$spatial_counts_csv, opts$marker_genes_csv)) {
    if (!file.exists(f)) stop(sprintf("Input file not found: %s", f))
  }

  # Accepted-but-ignored parameters and non-fatal notes, serialised under the same keys the Python
  # workers' record_ignored() / add_warning() use, so the surfaces above read them the same way.
  ignored  <- character(0)
  warnings <- character(0)

  # SMART is seeded by the marker list and never opens a single-cell reference. The path used to be
  # existence-checked -- a wrong path failed the run over a file the method does not read -- and
  # nothing in the payload said it had no effect. It stays accepted (removing it breaks callers).
  if (!is.null(opts$ref_counts_csv) && nzchar(opts$ref_counts_csv)) {
    ignored <- c(ignored, "ref_counts_csv")
    warnings <- c(warnings, paste0(
      "ignored parameter(s) ref_counts_csv: SMART seeds its topics from marker_genes_csv and never ",
      "reads a single-cell reference; the path was not opened."))
    log_msg("ref_counts_csv is not read by SMART; recording it as ignored")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  # --- Load spatial counts ---
  log_msg("Reading spatial counts from: ", opts$spatial_counts_csv)
  sp_counts_df <- read.csv(opts$spatial_counts_csv, row.names = 1, check.names = FALSE)
  non_numeric <- names(sp_counts_df)[!vapply(sp_counts_df, is.numeric, logical(1))]
  if (length(non_numeric) > 0) {
    stop("The spatial counts CSV has ", length(non_numeric), " non-numeric column(s) (first: ",
         paste(head(non_numeric, 5), collapse = ", "), "). It must hold one numeric count per ",
         "gene and spot with the identifiers in the first column and the header.")
  }
  sp_counts_mat <- as.matrix(sp_counts_df)
  if (anyNA(sp_counts_mat)) {
    stop("The spatial counts CSV has ", sum(is.na(sp_counts_mat)), " missing value(s); keyATM ",
         "cannot model a missing count. Fill or drop them before calling SMART.")
  }

  # SMART wants spots x genes, so the gene axis has to be identified before anything reads it.
  # The evidence is the marker gene names, which are loaded below -- the orientation is settled
  # there, not here from the two dimensions. Sizing them against each other is right only where
  # the spots outnumber the genes, and a 649-gene MERFISH panel is the other way round.

  # --- Load marker genes ---
  log_msg("Reading marker genes from: ", opts$marker_genes_csv)
  marker_df <- read.csv(opts$marker_genes_csv, check.names = FALSE)

  # Support two formats:
  # 1. Two columns: 'cell_type' (or 'cluster') and 'gene' -> named list
  # 2. Single column 'gene' -> all genes assigned to one unnamed type
  ct_col <- intersect(colnames(marker_df), MARKER_TYPE_COLUMNS)[1]
  gene_col <- intersect(colnames(marker_df), MARKER_GENE_COLUMNS)[1]
  if (is.na(gene_col)) {
    # A one-column file has nothing else it could mean, so its only column is the gene list and the
    # payload says which heading was read. With several columns the old code took the first one --
    # on a cell_type,symbol file that is the cell types -- so the ambiguity is refused by name.
    if (ncol(marker_df) == 1L) {
      gene_col <- colnames(marker_df)[1]
      warnings <- c(warnings, paste0(
        "marker_genes_csv has no recognised gene column; its only column '", gene_col,
        "' was read as the gene list."))
    } else {
      stop("marker_genes_csv has no recognised gene column. Its columns are: ",
           paste(colnames(marker_df), collapse = ", "), ". Name the gene column one of: ",
           paste(MARKER_GENE_COLUMNS, collapse = ", "), " (and the cell-type column one of: ",
           paste(MARKER_TYPE_COLUMNS, collapse = ", "), ").")
    }
  }
  log_msg("Marker gene column: '", gene_col, "'",
          if (!is.na(ct_col)) paste0(", cell-type column: '", ct_col, "'") else ", no cell-type column")

  if (!is.na(ct_col)) {
    # Build named list of marker genes per cell type
    marker_list <- split(as.character(marker_df[[gene_col]]),
                         as.character(marker_df[[ct_col]]))
    marker_list <- lapply(marker_list, unique)
    log_msg("Loaded markers for ", length(marker_list), " cell types")
  } else {
    # Single list: distribute genes evenly across n_topics
    all_markers <- unique(as.character(marker_df[[gene_col]]))
    # Create one keyword topic with all markers
    marker_list <- list("markers" = all_markers)
    log_msg("Loaded ", length(all_markers), " marker genes (no cell type column; using single topic)")
  }

  # Now the gene names are known, so which axis of the counts matrix carries them can be read
  # rather than guessed. Same idiom the other R workers use: intersect, transpose and retry.
  all_marker_genes <- unique(unlist(marker_list, use.names = FALSE))
  common_markers <- intersect(all_marker_genes, colnames(sp_counts_mat))
  if (length(common_markers) == 0) {
    sp_counts_mat <- t(sp_counts_mat)
    common_markers <- intersect(all_marker_genes, colnames(sp_counts_mat))
  }

  n_spots_input <- nrow(sp_counts_mat)
  n_genes_spatial <- ncol(sp_counts_mat)
  log_msg("Spatial: ", n_spots_input, " spots x ", n_genes_spatial, " genes")

  # --- Prepare sparse matrix for SMART ---
  # SMART expects a sparse matrix (spots x genes) convertible to dfm
  sp_sparse <- as(sp_counts_mat, "dgCMatrix")
  # keyATM models non-negative integer counts. A matrix that is not one -- normalised or
  # log-transformed expression, or negative residuals -- used to be rounded and clamped here without
  # a word, so a run on the wrong layer looked like a run on counts. Values that are already whole
  # numbers (a float CSV of counts) pass untouched; anything else needs round_counts=TRUE, and the
  # payload then says how many values were changed.
  n_values_rounded <- sum(sp_sparse@x != round(sp_sparse@x))
  n_values_clamped <- sum(sp_sparse@x < 0)
  if (n_values_rounded > 0 || n_values_clamped > 0) {
    if (!opts$round_counts) {
      stop("The spatial matrix is not integer counts: ", n_values_rounded, " non-integer and ",
           n_values_clamped, " negative value(s) among ", length(sp_sparse@x), " non-zero entries ",
           "(first non-integer values: ",
           paste(head(sp_sparse@x[sp_sparse@x != round(sp_sparse@x)], 3), collapse = ", "),
           "). SMART's keyATM models raw counts. Pass the raw counts, or pass round_counts=TRUE to ",
           "round to the nearest integer and clamp negatives to 0 -- the payload will report how ",
           "many values that changed.")
    }
    sp_sparse@x <- pmax(round(sp_sparse@x), 0)
    warnings <- c(warnings, paste0(
      "round_counts=TRUE: ", n_values_rounded, " non-integer value(s) were rounded to the nearest ",
      "integer and ", n_values_clamped, " negative value(s) were clamped to 0 before keyATM saw them."))
    log_msg("round_counts: rounded ", n_values_rounded, " and clamped ", n_values_clamped, " value(s)")
  }

  # keyATM drops any document with no tokens, and when it drops one it discards the document names
  # with it -- theta comes back one row shorter with no rownames at all, so pairing its rows with
  # the input's spot names is a length error rather than a labelling. Off-tissue Visium spots read
  # as all-zero, and the rounding above (round_counts=TRUE) can empty a row that was not empty in
  # the file, so the empty spots are removed here, where they can be counted and reported.
  spot_totals <- Matrix::rowSums(sp_sparse)
  n_spots_empty <- sum(spot_totals == 0)
  if (n_spots_empty > 0) {
    log_msg("Dropping ", n_spots_empty, " spot(s) with no counts")
    sp_sparse <- sp_sparse[spot_totals > 0, , drop = FALSE]
  }
  if (nrow(sp_sparse) == 0) {
    stop("All ", n_spots_input, " spots have zero counts",
         if (n_values_rounded > 0) " once round_counts=TRUE has rounded the matrix to integers" else "",
         ", so there is nothing for SMART to model. If the input is normalised or log-transformed ",
         "expression rather than counts, pass the raw counts instead.")
  }
  n_spots <- nrow(sp_sparse)

  # --- Filter marker genes to those that can seed a topic ---
  # A keyword seeds a keyATM topic only if it occurs in the documents: keyATM builds its vocabulary
  # from the words with a count, prunes every other keyword with a console warning, and aborts with
  # "All keywords are pruned" when a whole topic loses its seeds. So a marker has to be on the panel
  # AND counted in at least one spot. The old filter checked only the first, which left a type
  # whose markers were on the panel but never detected to crash the run inside keyATM.
  available_genes <- colnames(sp_sparse)
  counted_genes <- available_genes[Matrix::colSums(sp_sparse) > 0]
  n_marker_genes_input <- length(all_marker_genes)
  on_panel_markers <- intersect(all_marker_genes, available_genes)
  n_marker_genes_off_panel <- n_marker_genes_input - length(on_panel_markers)
  n_marker_genes_without_counts <- length(setdiff(on_panel_markers, counted_genes))
  marker_list <- lapply(marker_list, function(genes) {
    intersect(genes, counted_genes)
  })
  # A cell type none of whose markers can seed a topic leaves the result: the proportions table has
  # no column for it. The old code removed the entry and said nothing; the payload now names every
  # type it lost and counts the marker genes that were off the panel or never detected.
  dropped_marker_types <- names(marker_list)[vapply(marker_list, length, integer(1)) == 0]
  marker_list <- marker_list[vapply(marker_list, length, integer(1)) > 0]
  n_marker_genes_dropped <- n_marker_genes_off_panel + n_marker_genes_without_counts
  if (length(dropped_marker_types) > 0) {
    warnings <- c(warnings, paste0(
      length(dropped_marker_types), " marker cell type(s) have no marker gene that is on the spatial ",
      "panel and counted in any spot, so they get no topic and the proportions table has no column ",
      "for them: ", paste(dropped_marker_types, collapse = ", "), "."))
    log_msg("Dropping ", length(dropped_marker_types), " marker type(s) with no usable gene: ",
            paste(dropped_marker_types, collapse = ", "))
  }
  if (n_marker_genes_dropped > 0) {
    log_msg(n_marker_genes_off_panel, " of ", n_marker_genes_input, " marker genes are not on the ",
            "spatial panel and ", n_marker_genes_without_counts, " are on it with no count in any spot")
  }

  if (length(marker_list) == 0) {
    stop("None of the ", length(all_marker_genes), " marker genes can seed a topic: ",
         length(on_panel_markers), " appear among the ", length(available_genes), " gene names SMART ",
         "reads off the columns of the spatial matrix (both orientations of that matrix were tried), ",
         "and ", n_marker_genes_without_counts, " of those have no count in any spot. First markers: ",
         paste(head(all_marker_genes, 5), collapse = ", "), ". First spatial gene names: ",
         paste(head(available_genes, 5), collapse = ", "),
         ". The two must use the same gene identifiers, and a marker must be detected on the slide.")
  }

  n_keyword_topics <- length(marker_list)
  # noMarkerCts = additional topics beyond the keyword-seeded ones
  n_extra_topics <- max(0L, opts$n_topics - n_keyword_topics)
  if (opts$n_topics < n_keyword_topics) {
    warnings <- c(warnings, paste0(
      "n_topics=", opts$n_topics, " is below the ", n_keyword_topics, " keyword-seeded topics the ",
      "marker file defines; SMART fits ", n_keyword_topics, " topics with no unsupervised extras."))
  }
  log_msg("Keyword topics: ", n_keyword_topics, ", extra topics: ", n_extra_topics)

  # --- Run SMART ---
  log_msg("Running SMART (keyATM-based) with seed ", opts$seed, " for ", opts$iterations,
          " iterations (upstream default ", UPSTREAM_ITERATIONS, ")...")
  set.seed(opts$seed)

  smart_result <- SMART_base(
    stData       = sp_sparse,
    markerGs     = marker_list,
    noMarkerCts  = n_extra_topics,
    outDir       = opts$output_dir,
    seed         = opts$seed,
    iterations   = opts$iterations
  )

  # --- Extract results ---
  theta <- smart_result$ct_proportions   # spots x topics
  phi   <- smart_result$ct_spec_gexp     # topics x genes

  # --- Save outputs ---
  # Label the rows from what SMART returned. keyATM carries the caller's document names through
  # whenever it fits every document, so this is the input's spot IDs in the input's order; if it
  # ever returns a different number of rows, say so rather than pairing them with something else.
  spot_ids <- rownames(theta)
  if (is.null(spot_ids)) spot_ids <- rownames(sp_sparse)
  if (length(spot_ids) != nrow(theta)) {
    stop("SMART returned ", nrow(theta), " proportion rows for the ", nrow(sp_sparse),
         " spots it was given, so which row belongs to which spot is no longer known.")
  }
  theta_df <- as.data.frame(theta)
  theta_df$spot <- spot_ids
  prop_path <- file.path(opts$output_dir, "smart_proportions.csv")
  write_csv_atomic(theta_df, prop_path)

  phi_df <- as.data.frame(phi)
  phi_df$topic <- rownames(phi)
  beta_path <- file.path(opts$output_dir, "smart_beta.csv")
  write_csv_atomic(phi_df, beta_path)

  log_msg("Saved proportions to: ", prop_path)
  log_msg("Saved beta to: ", beta_path)

  # SMART_base itself saves the fitted keyATM object under outDir/inst_<seed>/base_model.rds.
  output_files <- list(proportions_csv = prop_path, beta_csv = beta_path)
  model_rds <- file.path(opts$output_dir, paste0("inst_", opts$seed), "base_model.rds")
  if (file.exists(model_rds)) output_files$model_rds <- model_rds

  # --- Summary ---
  n_topics_fit <- ncol(theta)
  dominant <- apply(theta, 1, function(row) colnames(theta)[which.max(row)])
  dominant_counts <- as.list(table(dominant))
  topic_names <- colnames(theta)

  params <- list(
    n_topics           = opts$n_topics,
    seed               = opts$seed,
    iterations         = opts$iterations,
    round_counts       = opts$round_counts,
    marker_gene_column = gene_col,
    method             = METHOD_NAME,
    used_fallback      = FALSE
  )
  if (!is.na(ct_col)) params$marker_type_column <- ct_col
  if (length(ignored) > 0) params$ignored <- I(ignored)

  analysis <- paste0(
    "SMART identified ", n_topics_fit, " topics (",
    n_keyword_topics, " keyword-seeded + ", n_extra_topics,
    " unsupervised) across ", n_spots, " spatial spots with ", opts$iterations,
    " keyATM iterations."
  )
  if (length(dropped_marker_types) > 0) {
    analysis <- paste0(analysis, " ", length(dropped_marker_types), " marker cell type(s) had no marker ",
                       "gene detected on the panel and got no topic: ",
                       paste(dropped_marker_types, collapse = ", "), ".")
  }
  if (n_values_rounded > 0 || n_values_clamped > 0) {
    analysis <- paste0(analysis, " The input was not integer counts; round_counts=TRUE rounded ",
                       n_values_rounded, " and clamped ", n_values_clamped, " value(s).")
  }

  result <- list(
    status       = "ok",
    tool         = "smart_spatial",
    task         = "mapping",
    data         = list(
      n_spots                = n_spots,
      n_spots_input          = n_spots_input,
      n_spots_empty          = n_spots_empty,
      n_genes_spatial        = n_genes_spatial,
      n_keyword_topics       = n_keyword_topics,
      n_extra_topics         = n_extra_topics,
      n_marker_genes         = sum(sapply(marker_list, length)),
      n_marker_genes_input   = n_marker_genes_input,
      n_marker_genes_dropped = n_marker_genes_dropped,
      n_marker_genes_off_panel      = n_marker_genes_off_panel,
      n_marker_genes_without_counts = n_marker_genes_without_counts,
      dropped_marker_types   = I(dropped_marker_types),
      n_values_rounded       = n_values_rounded,
      n_values_clamped       = n_values_clamped
    ),
    output_files = output_files,
    params       = params,
    summary      = list(
      n_topics_fit     = n_topics_fit,
      topic_names      = topic_names,
      dominant_counts  = dominant_counts
    ),
    analysis     = analysis
  )
  if (length(warnings) > 0) result$warnings <- I(warnings)
  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)

  # parse_args refuses a bad flag with stop() (--n-topics -1, --iterations 0, an unknown flag). It runs
  # inside the tryCatch so that refusal comes back as the JSON error payload naming the flag; outside
  # it, the stop was an R top-level error with nothing on stdout, which the portal can only report as
  # "worker produced no output".
  res <- tryCatch(with_r_traceback({
    sink(stderr())
    opts <- parse_args(args)
    result <- run_smart(opts)
    sink()
    result
  }), error = function(e) {
    try(sink(), silent = TRUE)
    log_msg("ERROR: ", conditionMessage(e))
    list(
      status    = "error",
      tool      = "smart_spatial",
      task      = "mapping",
      error     = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  cat(toJSON(res, auto_unbox = TRUE, digits = NA), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}
