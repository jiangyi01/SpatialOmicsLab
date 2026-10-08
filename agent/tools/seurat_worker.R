#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
  library(methods)
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
  message(sprintf("[seurat-worker] %s", msg))
}

# Draw one plot to `path`. Returns NULL on success, or the error message as a string.
#
# The plotting calls below fail for ordinary, diagnosable reasons -- an object with no @images
# ("Could not find any spatial image information"), a gene that is not in the object ("None of
# the requested variables were found: X"). Those messages are the whole answer for the user, so
# they are returned rather than discarded, and every caller reports them.
#
# Two details this exists to get right. `expr` is lazy, so an error thrown while *building* the
# plot is caught here too, not just one thrown by print(). And the device is closed from on.exit,
# because a dev.off() written after print() is skipped by the very error it needs to survive --
# that leaks a device on every failed feature in a loop. The cairo device writes nothing until a
# page is closed, so a failed plot leaves no file at all; unlink() covers the backends that do.
save_plot_png <- function(path, expr, width = 1200, height = 1000) {
  opened <- tryCatch(
    {
      png(path, width = width, height = height)
      TRUE
    },
    error = function(e) conditionMessage(e)
  )
  if (!isTRUE(opened)) {
    return(opened)
  }
  dev_id <- dev.cur()
  drawn <- FALSE
  on.exit(
    {
      if (dev_id %in% dev.list()) dev.off(dev_id)
      if (!drawn) unlink(path)
    },
    add = TRUE
  )
  tryCatch(
    {
      print(expr)
      drawn <- TRUE
      NULL
    },
    error = function(e) conditionMessage(e)
  )
}

# A reader that opens an output while it is being written sees the previous version or the whole new
# one, never a truncated file (rename(2) is atomic on one filesystem). Same helpers as precast_worker.R.
write_csv_atomic <- function(df, path, row.names = TRUE) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = row.names, quote = TRUE)
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
  # Defaults
  opts <- list(
    mode                 = NULL,

    # single-sample QC/cluster
    counts_dir           = NULL,

    # integration
    counts_dirs          = character(0),
    sample_ids           = character(0),

    # spatial
    data_dir             = NULL,
    spatial_var          = FALSE,
    sv_assay             = "Spatial",
    sv_selection_method  = "markvariogram",
    sv_nfeatures         = 2000L,
    image_alpha          = 0.8,
    spot_size            = 1.5,
    ncol                 = 1L,

    # common
    output_dir           = NULL,
    project              = "SeuratProject",
    min_cells            = 3L,
    min_features         = 200L,
    n_hvgs               = 2000L,
    n_pcs                = 30L,
    resolution           = 0.8,
    umap                 = FALSE,
    seed                 = 0L,

    # markers / dimplot / identities
    seurat_rds           = NULL,
    ident_key            = "seurat_clusters",
    cluster              = NULL,
    group_a              = NULL,
    group_b              = NULL,
    logfc_threshold      = 0.25,
    min_pct              = 0.1,
    test_use             = "wilcox",
    reduction            = "umap",
    allow_reduction_fallback = FALSE,
    group_by             = "seurat_clusters",
    label                = FALSE,

    # the assay a mode on an existing object reads. NULL means the mode's own default:
    # spatial_variable_features uses "Spatial", find_markers uses "RNA" when the object has it
    # (see marker_assay). One flag, --assay, reaches both.
    assay                = NULL,

    # spatial-variable-features mode (on existing object)
    selection_method     = "markvariogram",
    nfeatures            = 2000L,

    # spatial feature plot
    features             = character(0)
  )

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]

    # Boolean flags without explicit values
    if (key %in% c("--umap", "--label", "--spatial-var", "--allow-reduction-fallback")) {
      if (key == "--umap")       opts$umap <- TRUE
      if (key == "--label")      opts$label <- TRUE
      if (key == "--spatial-var") opts$spatial_var <- TRUE
      if (key == "--allow-reduction-fallback") opts$allow_reduction_fallback <- TRUE
      i <- i + 1L
      next
    }

    # Options with values
    if (i == length(args)) {
      stop(sprintf("Missing value for argument %s", key))
    }
    val <- args[[i + 1L]]

    assign_opt <- function(name, value) {
      opts[[name]] <<- value
    }

    if (key == "--mode") {
      assign_opt("mode", val)

    } else if (key == "--counts-dir") {
      # For single-sample mode: counts_dir
      # For integration: append to counts_dirs
      if (is.null(opts$counts_dir)) {
        opts$counts_dir <- val
      }
      opts$counts_dirs <- c(opts$counts_dirs, val)

    } else if (key == "--sample-id") {
      opts$sample_ids <- c(opts$sample_ids, val)

    } else if (key == "--data-dir") {
      opts$data_dir <- val

    } else if (key == "--output-dir") {
      opts$output_dir <- val

    } else if (key == "--project") {
      opts$project <- val

    } else if (key == "--min-cells") {
      opts$min_cells <- as.integer(val)

    } else if (key == "--min-features") {
      opts$min_features <- as.integer(val)

    } else if (key == "--n-hvgs") {
      opts$n_hvgs <- as.integer(val)

    } else if (key == "--n-pcs") {
      opts$n_pcs <- as.integer(val)

    } else if (key == "--resolution") {
      opts$resolution <- as.numeric(val)

    } else if (key == "--seed") {
      opts$seed <- as.integer(val)

    } else if (key == "--seurat-rds") {
      opts$seurat_rds <- val

    } else if (key == "--ident-key") {
      opts$ident_key <- val

    } else if (key == "--group-a") {
      opts$group_a <- val

    } else if (key == "--group-b") {
      opts$group_b <- val

    } else if (key == "--cluster") {
      opts$cluster <- val

    } else if (key == "--logfc-threshold") {
      opts$logfc_threshold <- as.numeric(val)

    } else if (key == "--min-pct") {
      opts$min_pct <- as.numeric(val)

    } else if (key == "--test-use") {
      opts$test_use <- val

    } else if (key == "--reduction") {
      opts$reduction <- val

    } else if (key == "--group-by") {
      opts$group_by <- val

    } else if (key == "--sv-assay") {
      opts$sv_assay <- val

    } else if (key == "--sv-selection-method") {
      opts$sv_selection_method <- val

    } else if (key == "--sv-nfeatures") {
      opts$sv_nfeatures <- as.integer(val)

    } else if (key == "--image-alpha") {
      opts$image_alpha <- as.numeric(val)

    } else if (key == "--spot-size") {
      opts$spot_size <- as.numeric(val)

    } else if (key == "--ncol") {
      opts$ncol <- as.integer(val)

    } else if (key == "--assay") {
      opts$assay <- val

    } else if (key == "--selection-method") {
      opts$selection_method <- val

    } else if (key == "--nfeatures") {
      opts$nfeatures <- as.integer(val)

    } else if (key == "--feature") {
      opts$features <- c(opts$features, val)

    } else {
      stop(sprintf("Unknown argument: %s", key))
    }

    i <- i + 2L
  }

  opts
}

# CreateSeuratObject applies both QC filters -- min.features first drops every cell with fewer
# detected genes, then min.cells drops every gene detected in fewer of the cells that remain -- and
# reports neither. On sparse spatial input the default min_features = 200 removes most of the data
# (43.5% of the Xenium tonsil cells, 68.5% of the VisiumHD colon bins), and the payload used to give
# only the post-filter size. So the input size and what each filter removed are counted here and
# published, the same way run_spatial_qc_cluster reports n_genes_dropped / n_spots_dropped.
#
# Read10X returns a list when features.tsv holds more than one feature type (Gene Expression plus
# Antibody Capture, say). CreateSeuratObject accepts that list and files every type as a layer of the
# one RNA assay, so the antibody features were normalised, scaled and clustered as if they were genes.
# That is refused by name rather than guessed at.
create_qc_filtered_object <- function(mat, opts, what = "") {
  if (is.list(mat) && !inherits(mat, "Matrix")) {
    stop(sprintf(paste0(
      "%sthe counts directory holds %d feature types (%s), and this mode clusters one counts matrix: ",
      "the other types would be treated as genes. Pass a counts directory holding only the Gene Expression ",
      "features, or the .h5ad / 10x .h5 it came from: the tool then converts the Gene Expression features ",
      "and lists the others in params.feature_types_dropped."
    ), what, length(mat), paste(names(mat), collapse = ", ")))
  }
  n_cells_input <- ncol(mat)
  n_genes_input <- nrow(mat)
  if (isTRUE(opts$min_features > 0)) {
    detected <- Matrix::colSums(mat > 0)
    if (!any(detected >= opts$min_features)) {
      stop(sprintf(
        "%smin_features=%d leaves no cell; the most-detected of the %d cells has %d detected genes",
        what, opts$min_features, n_cells_input, if (length(detected)) as.integer(max(detected)) else 0L
      ))
    }
  }
  obj <- CreateSeuratObject(
    counts = mat,
    project = opts$project,
    min.cells = opts$min_cells,
    min.features = opts$min_features
  )
  if (nrow(obj) == 0L) {
    stop(sprintf("%smin_cells=%d leaves no gene", what, opts$min_cells))
  }
  n_cells_dropped <- n_cells_input - ncol(obj)
  n_genes_dropped <- n_genes_input - nrow(obj)
  log_msg(sprintf(
    "%sQC: dropped %d of %d genes (min_cells=%d) and %d of %d cells (min_features=%d)",
    what, n_genes_dropped, n_genes_input, opts$min_cells, n_cells_dropped, n_cells_input, opts$min_features
  ))
  list(
    obj             = obj,
    genes_input     = rownames(mat),
    n_cells_input   = n_cells_input,
    n_genes_input   = n_genes_input,
    n_cells_dropped = n_cells_dropped,
    n_genes_dropped = n_genes_dropped
  )
}

# Seurat's stochastic verbs reseed themselves -- FindClusters with random.seed = 0, RunPCA and
# RunUMAP with seed.use = 42 -- so the set.seed() in main() never reached them, and every `seed`
# gave the same PCA, the same clusters and the same UMAP. The seed is now handed to each verb as an
# offset from Seurat's own default, so seed = 0 reproduces exactly what was produced before (and
# what Seurat produces unseeded); params.seeds publishes the value each verb received.
seurat_seeds <- function(seed) {
  seed <- as.integer(seed)
  list(FindClusters = seed, RunPCA = 42L + seed, RunUMAP = 42L + seed)
}

run_qc_cluster <- function(opts) {
  if (is.null(opts$counts_dir) || is.null(opts$output_dir)) {
    stop("qc_cluster requires --counts-dir and --output-dir")
  }

  log_msg("Mode qc_cluster; counts_dir = ", opts$counts_dir)
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  mat <- Read10X(data.dir = opts$counts_dir)
  qc <- create_qc_filtered_object(mat, opts)
  obj <- qc$obj
  rm(mat)

  seeds <- seurat_seeds(opts$seed)
  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(
    obj,
    selection.method = "vst",
    nfeatures = opts$n_hvgs,
    verbose = FALSE
  )
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = opts$n_pcs, seed.use = seeds$RunPCA, verbose = FALSE)
  obj <- FindNeighbors(obj, dims = 1:opts$n_pcs, verbose = FALSE)
  obj <- FindClusters(obj, resolution = opts$resolution, random.seed = seeds$FindClusters, verbose = FALSE)

  if (isTRUE(opts$umap)) {
    obj <- RunUMAP(obj, dims = 1:opts$n_pcs, seed.use = seeds$RunUMAP, verbose = FALSE)
  }

  obj_path  <- file.path(opts$output_dir, "seurat_qc_cluster_obj.rds")
  meta_path <- file.path(opts$output_dir, "seurat_qc_cluster_metadata.csv")
  umap_path <- file.path(opts$output_dir, "seurat_qc_cluster_umap.csv")

  saveRDS(obj, file = obj_path)
  write.csv(obj@meta.data, meta_path, quote = TRUE)

  umap_exists <- FALSE
  if ("umap" %in% names(obj@reductions) && isTRUE(opts$umap)) {
    umap_df <- as.data.frame(Embeddings(obj, reduction = "umap"))
    umap_df$cell <- rownames(umap_df)
    write.csv(umap_df, umap_path, quote = TRUE, row.names = FALSE)
    umap_exists <- TRUE
  }

  # DimPlot snapshot
  dim_reduction <- if ("umap" %in% names(obj@reductions) && isTRUE(opts$umap)) {
    "umap"
  } else if ("tsne" %in% names(obj@reductions)) {
    "tsne"
  } else {
    "pca"
  }

  dimplot_png <- file.path(opts$output_dir, "seurat_qc_cluster_dimplot.png")
  plot_err <- save_plot_png(
    dimplot_png,
    DimPlot(obj, reduction = dim_reduction, group.by = "seurat_clusters", label = TRUE)
  )
  if (!is.null(plot_err)) log_msg("Could not draw the cluster DimPlot: ", plot_err)

  n_cells <- ncol(obj)
  n_genes <- nrow(obj)
  n_clusters <- length(unique(Idents(obj)))
  cluster_sizes <- as.list(table(Idents(obj)))

  output_files <- list(
    seurat_rds   = obj_path,
    metadata_csv = meta_path
  )
  # Only once it is on disk: a plot that threw writes no file, and naming the path anyway sends
  # every downstream reader after something that is not there.
  if (is.null(plot_err)) output_files$dimplot_png <- dimplot_png
  if (umap_exists) output_files$umap_csv <- umap_path

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "qc_cluster",
    warnings     = I(c(
      if (is.null(plot_err)) character(0) else paste("cluster DimPlot:", plot_err),
      if (qc$n_cells_dropped > 0) sprintf(
        "QC dropped %d of %d cell(s) with fewer than min_features=%d detected genes",
        qc$n_cells_dropped, qc$n_cells_input, opts$min_features
      ) else character(0)
    )),
    data         = list(
      n_cells       = n_cells,
      n_genes       = n_genes,
      n_cells_input = qc$n_cells_input,
      n_genes_input = qc$n_genes_input
    ),
    output_files = output_files,
    params       = list(
      counts_dir      = opts$counts_dir,
      output_dir      = opts$output_dir,
      project         = opts$project,
      min_cells       = opts$min_cells,
      min_features    = opts$min_features,
      n_genes_dropped = qc$n_genes_dropped,
      n_cells_dropped = qc$n_cells_dropped,
      n_hvgs          = opts$n_hvgs,
      n_pcs           = opts$n_pcs,
      resolution      = opts$resolution,
      umap            = opts$umap,
      seed            = opts$seed,
      seeds           = seeds
    ),
    summary      = list(n_clusters = n_clusters, cluster_sizes = cluster_sizes),
    analysis     = sprintf(
      paste0(
        "Seurat qc_cluster completed: %d cells x %d genes after QC (dropped %d of %d genes below ",
        "min_cells=%d and %d of %d cells below min_features=%d), found %d clusters."
      ),
      n_cells, n_genes, qc$n_genes_dropped, qc$n_genes_input, opts$min_cells,
      qc$n_cells_dropped, qc$n_cells_input, opts$min_features, n_clusters
    )
  )
}

run_integrate_qc_cluster <- function(opts) {
  if (length(opts$counts_dirs) == 0L || is.null(opts$output_dir)) {
    stop("integrate_qc_cluster requires --counts-dir (>=1) and --output-dir")
  }
  if (length(opts$counts_dirs) != length(opts$sample_ids)) {
    stop("counts_dirs and sample_ids must have the same length")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode integrate_qc_cluster; n_samples = ", length(opts$counts_dirs))

  objs <- list()
  qc_per_sample <- list()
  genes_input <- character(0)
  for (i in seq_along(opts$counts_dirs)) {
    cd <- opts$counts_dirs[[i]]
    sid <- opts$sample_ids[[i]]
    log_msg("Reading sample ", sid, " from ", cd)
    m <- Read10X(data.dir = cd)
    qc <- create_qc_filtered_object(m, opts, what = sprintf("sample %s: ", sid))
    rm(m)
    sobj <- qc$obj
    genes_input <- union(genes_input, qc$genes_input)
    qc_per_sample[[i]] <- list(
      sample_id       = sid,
      n_cells_input   = qc$n_cells_input,
      n_cells_dropped = qc$n_cells_dropped,
      n_genes_input   = qc$n_genes_input,
      n_genes_dropped = qc$n_genes_dropped
    )
    sobj$sample_id <- sid
    sobj <- NormalizeData(sobj, verbose = FALSE)
    sobj <- FindVariableFeatures(
      sobj,
      selection.method = "vst",
      nfeatures = opts$n_hvgs,
      verbose = FALSE
    )
    objs[[i]] <- sobj
  }

  features <- SelectIntegrationFeatures(object.list = objs, nfeatures = opts$n_hvgs)
  anchors <- FindIntegrationAnchors(object.list = objs, anchor.features = features)
  merged <- IntegrateData(anchorset = anchors)

  DefaultAssay(merged) <- "integrated"
  seeds <- seurat_seeds(opts$seed)
  merged <- ScaleData(merged, verbose = FALSE)
  merged <- RunPCA(merged, npcs = opts$n_pcs, seed.use = seeds$RunPCA, verbose = FALSE)
  merged <- FindNeighbors(merged, dims = 1:opts$n_pcs, verbose = FALSE)
  merged <- FindClusters(merged, resolution = opts$resolution, random.seed = seeds$FindClusters, verbose = FALSE)
  if (isTRUE(opts$umap)) {
    merged <- RunUMAP(merged, dims = 1:opts$n_pcs, seed.use = seeds$RunUMAP, verbose = FALSE)
  }

  obj_path  <- file.path(opts$output_dir, "seurat_integrate_obj.rds")
  meta_path <- file.path(opts$output_dir, "seurat_integrate_metadata.csv")
  umap_path <- file.path(opts$output_dir, "seurat_integrate_umap.csv")

  saveRDS(merged, file = obj_path)
  write.csv(merged@meta.data, meta_path, quote = TRUE)

  umap_exists <- FALSE
  if ("umap" %in% names(merged@reductions) && isTRUE(opts$umap)) {
    umap_df <- as.data.frame(Embeddings(merged, reduction = "umap"))
    umap_df$cell <- rownames(umap_df)
    write.csv(umap_df, umap_path, quote = TRUE, row.names = FALSE)
    umap_exists <- TRUE
  }

  dim_reduction <- if ("umap" %in% names(merged@reductions) && isTRUE(opts$umap)) {
    "umap"
  } else if ("tsne" %in% names(merged@reductions)) {
    "tsne"
  } else {
    "pca"
  }

  dimplot_png <- file.path(opts$output_dir, "seurat_integrate_dimplot_clusters.png")
  plot_err <- save_plot_png(
    dimplot_png,
    DimPlot(merged, reduction = dim_reduction, group.by = "seurat_clusters", label = TRUE)
  )
  if (!is.null(plot_err)) log_msg("Could not draw the integrated cluster DimPlot: ", plot_err)

  # The default assay is "integrated" by now, so nrow(merged) is the number of integration anchor
  # features (at most n_hvgs), not the number of genes. Genes are counted on the RNA assay.
  n_cells <- ncol(merged)
  n_genes <- nrow(merged[["RNA"]])
  n_integration_features <- nrow(merged[["integrated"]])
  n_clusters <- length(unique(Idents(merged)))
  cluster_sizes <- as.list(table(Idents(merged)))
  n_cells_input <- sum(vapply(qc_per_sample, function(s) as.numeric(s$n_cells_input), numeric(1)))
  n_cells_dropped <- sum(vapply(qc_per_sample, function(s) as.numeric(s$n_cells_dropped), numeric(1)))
  n_genes_input <- length(genes_input)
  # A gene dropped by min_cells in one sample but kept in another is still in the merged RNA assay;
  # this counts the genes of the input that no sample kept. The per-sample counts are in summary.
  n_genes_dropped <- n_genes_input - n_genes

  output_files <- list(
    seurat_rds   = obj_path,
    metadata_csv = meta_path
  )
  if (is.null(plot_err)) output_files$dimplot_png <- dimplot_png
  if (umap_exists) output_files$umap_csv <- umap_path

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "integrate_qc_cluster",
    warnings     = I(c(
      if (is.null(plot_err)) character(0) else paste("cluster DimPlot:", plot_err),
      if (n_cells_dropped > 0) sprintf(
        "QC dropped %d of %d cell(s) across %d samples with fewer than min_features=%d detected genes",
        as.integer(n_cells_dropped), as.integer(n_cells_input), length(opts$counts_dirs), opts$min_features
      ) else character(0)
    )),
    data         = list(
      n_cells                = n_cells,
      n_genes                = n_genes,
      n_integration_features = n_integration_features,
      n_samples              = length(opts$counts_dirs),
      n_cells_input          = n_cells_input,
      n_genes_input          = n_genes_input
    ),
    output_files = output_files,
    params       = list(
      counts_dirs     = opts$counts_dirs,
      sample_ids      = opts$sample_ids,
      output_dir      = opts$output_dir,
      project         = opts$project,
      min_cells       = opts$min_cells,
      min_features    = opts$min_features,
      n_genes_dropped = n_genes_dropped,
      n_cells_dropped = n_cells_dropped,
      n_hvgs          = opts$n_hvgs,
      n_pcs           = opts$n_pcs,
      resolution      = opts$resolution,
      umap            = opts$umap,
      seed            = opts$seed,
      seeds           = seeds
    ),
    summary      = list(
      n_clusters    = n_clusters,
      cluster_sizes = cluster_sizes,
      qc_per_sample = qc_per_sample
    ),
    analysis     = sprintf(
      paste0(
        "Seurat integrate_qc_cluster completed: %d samples, %d cells x %d genes after QC (dropped %d of %d ",
        "cells below min_features=%d and %d of %d genes below min_cells=%d in every sample), integrated on ",
        "%d anchor features, %d clusters."
      ),
      length(opts$counts_dirs), n_cells, n_genes, as.integer(n_cells_dropped), as.integer(n_cells_input),
      opts$min_features, n_genes_dropped, n_genes_input, opts$min_cells, n_integration_features, n_clusters
    )
  )
}

# Seurat keeps raw counts and normalised expression in separate layers, and every
# differential-expression verb reads the normalised one. An object assembled straight from a
# counts matrix -- which is what CreateSeuratObject(counts = ...) in h5ad_to_seurat.R produces --
# has no data layer at all, so FindAllMarkers warns that the layer is empty once per identity and
# returns zero rows. That publishes as a completed run with no markers found, which no reader can
# tell from a real negative.
data_layer_is_empty <- function(obj, assay) {
  d <- suppressWarnings(tryCatch(
    LayerData(obj, assay = assay, layer = "data"),
    error = function(e) NULL
  ))
  is.null(d) || length(dim(d)) < 2L || any(dim(d) == 0L)
}

# No mode here can normalise an existing .rds -- qc_cluster and spatial_qc_cluster both start from
# a directory -- so refusing the run would leave our own converter path with nowhere to go.
# neighborseq_worker.R answers this the same way: normalise, then test. The returned string is
# published, so a reader can tell a normalised run from a supplied one.
ensure_normalised_data <- function(obj) {
  assay <- DefaultAssay(obj)
  if (!data_layer_is_empty(obj, assay)) {
    return(list(obj = obj, normalization = "as supplied"))
  }
  log_msg(
    "Assay '", assay, "' carries counts only -- no data layer to test. ",
    "Applying LogNormalize before differential expression."
  )
  list(
    obj = NormalizeData(obj, verbose = FALSE),
    normalization = "LogNormalize applied by this worker; input carried counts only"
  )
}

# Which comparison find_markers runs, decided from group_a / group_b / cluster before anything is
# loaded. group_a and group_b used to take effect only together: group_a alone (or group_b alone)
# fell through to FindAllMarkers over every identity while params echoed the group as if it had been
# used, and cluster given beside both groups was dropped without a word. Now group_a alone is
# group_a vs the rest (FindMarkers with ident.2 = NULL, written under the one-vs-rest file name),
# group_b alone is refused, a cluster that contradicts group_a is refused, and a cluster that the
# pairwise comparison overrides is reported in params.ignored.
marker_comparison <- function(opts) {
  a <- opts$group_a
  b <- opts$group_b
  cl <- opts$cluster
  if (!is.null(b) && is.null(a)) {
    stop(sprintf(paste0(
      "group_b='%s' was given without group_a. group_b is the comparison group (ident.2) of a pairwise ",
      "test and needs group_a; to test one group against all the others, pass it as group_a or cluster."
    ), b))
  }
  if (!is.null(a) && !is.null(b)) {
    return(list(
      kind    = "pairwise",
      ident_1 = a,
      ident_2 = b,
      label   = sprintf("%s vs %s", a, b),
      ignored = if (is.null(cl)) character(0) else "cluster",
      why     = if (is.null(cl)) "" else sprintf(
        "cluster='%s' was given with group_a and group_b; the pairwise comparison %s vs %s ran", cl, a, b
      )
    ))
  }
  if (!is.null(a) && !is.null(cl) && !identical(a, cl)) {
    stop(sprintf(paste0(
      "group_a='%s' and cluster='%s' both name the group to test against all the others. Pass one of ",
      "them, or add group_b for a pairwise test."
    ), a, cl))
  }
  one <- if (!is.null(a)) a else cl
  if (!is.null(one)) {
    return(list(
      kind = "one_vs_rest", ident_1 = one, ident_2 = NULL,
      label = sprintf("%s vs rest", one), ignored = character(0), why = ""
    ))
  }
  list(
    kind = "all_vs_rest", ident_1 = NULL, ident_2 = NULL,
    label = "each identity vs rest (FindAllMarkers, positive markers only)",
    ignored = character(0), why = ""
  )
}

# The assay find_markers tests. An object from integrate_qc_cluster is saved with DefaultAssay
# "integrated": batch-corrected values over the ~n_hvgs anchor features, which Seurat advises
# against using for differential expression and on which avg_log2FC is computed from values that go
# negative. So with no assay given the RNA assay is used whenever the object has one, and the
# object's DefaultAssay otherwise (a spatial object from spatial_qc_cluster has only "Spatial").
marker_assay <- function(obj, requested) {
  available <- Assays(obj)
  if (!is.null(requested)) {
    if (!(requested %in% available)) {
      stop(sprintf("assay '%s' not found. Available: %s", requested, paste(available, collapse = ", ")))
    }
    return(list(assay = requested, choice = "requested"))
  }
  if ("RNA" %in% available) {
    return(list(assay = "RNA", choice = "default: RNA, present in the object"))
  }
  list(assay = DefaultAssay(obj), choice = "default: the object's DefaultAssay (it has no RNA assay)")
}

# Seurat 5 keeps a merged or integrated object's RNA assay as one layer per sample (counts.1,
# data.1, counts.2, ...), and FindMarkers stops on it with "data layers are not joined". Join them
# for the test; the object on disk is not touched.
join_split_layers <- function(obj, assay) {
  a <- obj[[assay]]
  if (!inherits(a, "Assay5")) {
    return(list(obj = obj, joined = FALSE))
  }
  lyr <- Layers(a)
  if (sum(grepl("^counts\\.", lyr)) < 2L && sum(grepl("^data\\.", lyr)) < 2L) {
    return(list(obj = obj, joined = FALSE))
  }
  log_msg("Assay '", assay, "' holds split layers (", paste(lyr, collapse = ", "), "); joining them for the test.")
  obj[[assay]] <- JoinLayers(a)
  list(obj = obj, joined = TRUE)
}

run_find_markers <- function(opts) {
  if (is.null(opts$seurat_rds) || is.null(opts$output_dir)) {
    stop("find_markers requires --seurat-rds and --output-dir")
  }

  cmp <- marker_comparison(opts)
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode find_markers; seurat_rds = ", opts$seurat_rds)

  obj <- readRDS(opts$seurat_rds)

  picked <- marker_assay(obj, opts$assay)
  DefaultAssay(obj) <- picked$assay
  joined <- join_split_layers(obj, picked$assay)
  obj <- joined$obj

  prepared <- ensure_normalised_data(obj)
  obj <- prepared$obj
  normalization <- prepared$normalization

  if (!(opts$ident_key %in% colnames(obj@meta.data))) {
    stop(sprintf("ident_key '%s' not found in meta.data", opts$ident_key))
  }
  Idents(obj) <- obj@meta.data[[opts$ident_key]]

  markers <- NULL
  out_path <- NULL

  if (cmp$kind == "pairwise") {
    log_msg("Running pairwise FindMarkers on assay ", picked$assay, ": ", cmp$label)
    markers <- FindMarkers(
      obj,
      ident.1 = cmp$ident_1,
      ident.2 = cmp$ident_2,
      logfc.threshold = opts$logfc_threshold,
      min.pct = opts$min_pct,
      test.use = opts$test_use
    )
    markers$gene <- rownames(markers)
    out_path <- file.path(opts$output_dir, sprintf(
      "seurat_markers_%s_vs_%s.csv", cmp$ident_1, cmp$ident_2
    ))

  } else if (cmp$kind == "one_vs_rest") {
    log_msg("Running one-vs-rest FindMarkers on assay ", picked$assay, ": ", cmp$label)
    markers <- FindMarkers(
      obj,
      ident.1 = cmp$ident_1,
      ident.2 = NULL,
      logfc.threshold = opts$logfc_threshold,
      min.pct = opts$min_pct,
      test.use = opts$test_use
    )
    markers$gene <- rownames(markers)
    out_path <- file.path(opts$output_dir, sprintf(
      "seurat_markers_cluster_%s.csv", cmp$ident_1
    ))

  } else {
    log_msg("Running FindAllMarkers for all identities on assay ", picked$assay)
    markers <- FindAllMarkers(
      obj,
      only.pos = TRUE,
      logfc.threshold = opts$logfc_threshold,
      min.pct = opts$min_pct,
      test.use = opts$test_use
    )
    out_path <- file.path(opts$output_dir, "seurat_all_markers.csv")
  }

  write.csv(markers, out_path, quote = TRUE)

  n_markers <- nrow(markers)
  n_cells <- ncol(obj)
  n_genes <- nrow(obj)

  marker_params <- list(
    seurat_rds      = opts$seurat_rds,
    output_dir      = opts$output_dir,
    ident_key       = opts$ident_key,
    group_a         = opts$group_a,
    group_b         = opts$group_b,
    cluster         = opts$cluster,
    comparison      = cmp$label,
    assay           = picked$assay,
    assay_choice    = picked$choice,
    layers_joined   = joined$joined,
    logfc_threshold = opts$logfc_threshold,
    min_pct         = opts$min_pct,
    test_use        = opts$test_use
  )
  marker_warnings <- character(0)
  if (length(cmp$ignored) > 0L) {
    marker_params$ignored <- I(cmp$ignored)
    marker_warnings <- c(marker_warnings, sprintf(
      "ignored parameter(s) %s: %s", paste(cmp$ignored, collapse = ", "), cmp$why
    ))
  }

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "find_markers",
    warnings     = I(marker_warnings),
    data         = list(n_cells = n_cells, n_genes = n_genes),
    output_files = list(marker_csv = out_path),
    params       = marker_params,
    summary      = list(
      n_markers     = n_markers,
      normalization = normalization,
      comparison    = cmp$label,
      assay         = picked$assay
    ),
    analysis     = sprintf(
      "Seurat find_markers completed: %s on assay %s, found %d markers from %d cells (expression matrix: %s).",
      cmp$label, picked$assay, n_markers, n_cells, normalization
    )
  )
}

run_dimplot <- function(opts) {
  if (is.null(opts$seurat_rds) || is.null(opts$output_dir)) {
    stop("dimplot requires --seurat-rds and --output-dir")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode dimplot; seurat_rds = ", opts$seurat_rds)

  obj <- readRDS(opts$seurat_rds)

  if (!(opts$group_by %in% colnames(obj@meta.data))) {
    stop(sprintf("group_by '%s' not found in meta.data", opts$group_by))
  }

  # A missing reduction used to be replaced by whichever of umap/tsne/pca the object had, with
  # nothing in the payload saying the requested one was absent. The substitute now runs only when
  # the caller allows it, and the payload says what was asked for and what was drawn.
  reduction_to_use <- opts$reduction
  used_fallback <- FALSE
  if (!(reduction_to_use %in% names(obj@reductions))) {
    available <- names(obj@reductions)
    if (!isTRUE(opts$allow_reduction_fallback)) {
      stop(sprintf(paste0(
        "reduction '%s' is not in the object (it has: %s). Pass one of those as reduction, or ",
        "allow_reduction_fallback=True to plot the first of umap/tsne/pca the object has."
      ), opts$reduction, if (length(available)) paste(available, collapse = ", ") else "none"))
    }
    if ("umap" %in% available) {
      reduction_to_use <- "umap"
    } else if ("tsne" %in% available) {
      reduction_to_use <- "tsne"
    } else if ("pca" %in% available) {
      reduction_to_use <- "pca"
    } else {
      stop("No suitable reduction found (umap/tsne/pca).")
    }
    used_fallback <- TRUE
  }

  png_path <- file.path(
    opts$output_dir,
    sprintf("seurat_dimplot_%s_%s.png", reduction_to_use, opts$group_by)
  )

  log_msg("Saving DimPlot to ", png_path)
  png(png_path, width = 1200, height = 1000)
  print(
    DimPlot(
      obj,
      reduction = reduction_to_use,
      group.by = opts$group_by,
      label = isTRUE(opts$label)
    )
  )
  dev.off()

  n_cells <- ncol(obj)

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "dimplot",
    warnings     = I(if (used_fallback) sprintf(
      "fallback ran: DimPlot of '%s' -- the requested reduction '%s' is not in the object",
      reduction_to_use, opts$reduction
    ) else character(0)),
    data         = list(n_cells = n_cells),
    output_files = list(dimplot_png = png_path),
    params       = list(
      seurat_rds          = opts$seurat_rds,
      output_dir          = opts$output_dir,
      reduction           = reduction_to_use,
      reduction_requested = opts$reduction,
      method              = sprintf("DimPlot (%s)", reduction_to_use),
      used_fallback       = used_fallback,
      group_by            = opts$group_by,
      label               = opts$label
    ),
    summary      = list(),
    analysis     = sprintf(
      "Seurat dimplot completed: %d cells, reduction=%s, group_by=%s.",
      n_cells, reduction_to_use, opts$group_by
    )
  )
}

# FindSpatiallyVariableFeatures returns the SEURAT OBJECT, not a results table: it writes its
# statistics into the assay's meta.features and hands the object back. as.data.frame() on a
# Seurat object raises "cannot coerce class 'Seurat' to a data.frame", so the results have to be
# read out with SVFInfo(), the documented accessor. Both spatial modes share this one helper.
spatially_variable_table <- function(obj, assay, method) {
  info <- SVFInfo(obj, assay = assay, method = method, status = TRUE)
  # SVFInfo returns `variable` and `rank` as nested one-column data.frames whose inner names
  # carry the method (moransi.spatially.variable.rank). Flatten them so the CSV is a plain table
  # and the rank is orderable -- order() on a data.frame column raises "cannot xtfrm data frames".
  for (col in c("variable", "rank")) {
    if (col %in% colnames(info) && is.data.frame(info[[col]])) {
      info[[col]] <- info[[col]][[1]]
    }
  }
  # Genes the method never scored carry rank NA, and SVFInfo does not return the frame in rank
  # order. Sort before the caller's head(nfeatures) slice, or that slice keeps an arbitrary
  # prefix of the feature list instead of the top-ranked genes.
  if ("rank" %in% colnames(info)) {
    info <- info[order(info$rank, na.last = TRUE), , drop = FALSE]
    # SVFInfo returns every feature of the assay, but the method scored only the features of the
    # scale.data layer (the n_hvgs variable genes, after spatial_qc_cluster). The unscored rest are
    # not results: with nfeatures above the scored count they used to be written to the CSV with
    # rank NA and counted as "spatially variable". Only scored rows are returned.
    info <- info[!is.na(info$rank), , drop = FALSE]
  }
  info
}

# FindSpatiallyVariableFeatures (Seurat 5.3.1) tests the features of the assay's scale.data layer
# and nothing else. With no scale.data layer it dies deep inside on "attempt to set 'colnames' on an
# object with less than two dimensions"; this says what is missing instead, and returns how many
# features the method will be given.
scaled_feature_count <- function(obj, assay) {
  scaled <- suppressWarnings(Features(obj, assay = assay, layer = "scale.data"))
  if (length(scaled) == 0L) {
    stop(sprintf(paste0(
      "assay '%s' has no scale.data layer, and FindSpatiallyVariableFeatures tests only the features ",
      "in it. Run seurat_spatial_qc_cluster (it scales the n_hvgs variable genes) or ScaleData on the ",
      "object first."
    ), assay))
  }
  length(scaled)
}

# Space Ranger writes every feature type of a run into the one filtered_feature_bc_matrix.h5: the
# library's four CytAssist protein samples hold 35 Antibody Capture features beside ~18,000 genes.
# Read10X_h5 then returns one matrix per type, and Load10X_Spatial hands that list to
# CreateSeuratObject, which files each type as a counts layer of the one Spatial assay
# ("counts.Gene Expression", "counts.Antibody Capture"). Nothing after that worked: LayerData warned
# "only the first layer is used" and returned the genes, the min_cells filter indexed the union's
# feature names with the genes' logical vector (R recycled it, so it kept and dropped the wrong
# features), and FindVariableFeatures over the disjoint layers found no variable feature -- the run
# died on "No variable features". The genes are what this mode clusters, so the Spatial assay is
# rebuilt from the Gene Expression layer (with the slide's images), the other types are returned to
# be kept unprocessed as assays of their own, and the payload says so. A matrix with no Gene
# Expression type is refused by name, as create_qc_filtered_object refuses one for qc_cluster.
split_spatial_feature_types <- function(obj, assay = "Spatial") {
  typed <- grep("^counts\\.", Layers(obj[[assay]]), value = TRUE)
  if (length(typed) < 2L) {
    return(list(obj = obj, n_per_type = NULL, others = list()))
  }
  types <- sub("^counts\\.", "", typed)
  mats <- lapply(typed, function(l) LayerData(obj, assay = assay, layer = l))
  names(mats) <- types
  n_per_type <- lapply(mats, nrow)
  if (!("Gene Expression" %in% types)) {
    stop(sprintf(paste0(
      "the counts matrix holds %d feature types (%s) and none of them is Gene Expression; ",
      "spatial_qc_cluster clusters genes. Pass a Space Ranger folder (or .h5ad) with Gene Expression features."
    ), length(types), paste(types, collapse = ", ")))
  }
  rebuilt <- CreateSeuratObject(counts = mats[["Gene Expression"]], assay = assay)
  for (img in Images(obj)) rebuilt[[img]] <- obj[[img]]
  log_msg(
    "The counts matrix holds ", length(types), " feature types (", paste(types, collapse = ", "),
    "); clustering the Gene Expression features and keeping the others as assays of their own."
  )
  list(obj = rebuilt, n_per_type = n_per_type, others = mats[setdiff(types, "Gene Expression")])
}

run_spatial_qc_cluster <- function(opts) {
  if (is.null(opts$data_dir) || is.null(opts$output_dir)) {
    stop("spatial_qc_cluster requires --data-dir and --output-dir")
  }
  log_msg("Mode spatial_qc_cluster; data_dir = ", opts$data_dir)
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)

  obj <- Load10X_Spatial(data.dir = opts$data_dir)
  feature_split <- split_spatial_feature_types(obj, "Spatial")
  obj <- feature_split$obj
  feature_split$obj <- NULL  # no second reference to hold the unfiltered object through the run
  DefaultAssay(obj) <- "Spatial"

  # Load10X_Spatial hands min.cells/min.features to Read10X_h5, which ignores them, so the QC
  # filters the portal accepts never ran here (qc_cluster applies them through CreateSeuratObject).
  # Apply them the same way and say what they removed. The genes kept are named from the counts
  # layer the detection is counted on, never from the object: the two must be the same list.
  n_genes_before <- nrow(obj)
  n_spots_before <- ncol(obj)
  obj@project.name <- opts$project
  if (isTRUE(opts$min_cells > 0)) {
    counts <- LayerData(obj, assay = "Spatial", layer = "counts")
    keep_genes <- rownames(counts)[Matrix::rowSums(counts > 0) >= opts$min_cells]
    if (length(keep_genes) == 0L) {
      stop(sprintf("min_cells=%d leaves no gene: no gene is detected in that many of the %d spots",
                   opts$min_cells, ncol(counts)))
    }
    if (length(keep_genes) < nrow(counts)) obj <- subset(obj, features = keep_genes)
    rm(counts)
  }
  if (isTRUE(opts$min_features > 0)) {
    keep_spots <- colnames(obj)[obj$nFeature_Spatial >= opts$min_features]
    if (length(keep_spots) == 0) {
      stop(sprintf("min_features=%d leaves no spot; the most-detected spot has %d features",
                   opts$min_features, max(obj$nFeature_Spatial)))
    }
    if (length(keep_spots) < ncol(obj)) obj <- subset(obj, cells = keep_spots)
  }
  n_genes_dropped <- n_genes_before - nrow(obj)
  n_spots_dropped <- n_spots_before - ncol(obj)
  log_msg(sprintf("QC: dropped %d genes (min_cells=%d) and %d spots (min_features=%d)",
                  n_genes_dropped, opts$min_cells, n_spots_dropped, opts$min_features))

  # The other feature types ride along, unprocessed, for the spots the QC kept. Assay names must be
  # syntactic ("Antibody Capture" -> "Antibody.Capture") and must not collide with "Spatial".
  other_assays <- list()
  for (ftype in names(feature_split$others)) {
    assay_name <- tail(make.unique(c(Assays(obj), make.names(ftype))), 1L)
    obj[[assay_name]] <- CreateAssay5Object(counts = feature_split$others[[ftype]][, colnames(obj), drop = FALSE])
    other_assays[[ftype]] <- assay_name
  }
  feature_split$others <- NULL
  DefaultAssay(obj) <- "Spatial"

  obj <- NormalizeData(obj, verbose = FALSE)
  obj <- FindVariableFeatures(
    obj,
    selection.method = "vst",
    nfeatures = opts$n_hvgs,
    verbose = FALSE
  )
  seeds <- seurat_seeds(opts$seed)
  obj <- ScaleData(obj, verbose = FALSE)
  obj <- RunPCA(obj, npcs = opts$n_pcs, seed.use = seeds$RunPCA, verbose = FALSE)
  obj <- FindNeighbors(obj, dims = 1:opts$n_pcs, verbose = FALSE)
  obj <- FindClusters(obj, resolution = opts$resolution, random.seed = seeds$FindClusters, verbose = FALSE)

  if (isTRUE(opts$umap)) {
    obj <- RunUMAP(obj, dims = 1:opts$n_pcs, seed.use = seeds$RunUMAP, verbose = FALSE)
  }

  sv_csv <- NULL
  n_sv_features_tested <- NULL
  n_sv_genes <- NULL
  if (isTRUE(opts$spatial_var)) {
    log_msg("Running FindSpatiallyVariableFeatures on assay ", opts$sv_assay)
    if (!(opts$sv_assay %in% Assays(obj))) {
      stop(sprintf(
        "Assay '%s' not found in object. Available: %s",
        opts$sv_assay, paste(Assays(obj), collapse = ", ")
      ))
    }
    DefaultAssay(obj) <- opts$sv_assay

    if (!"FindSpatiallyVariableFeatures" %in% getNamespaceExports("Seurat")) {
      stop("FindSpatiallyVariableFeatures not exported by current Seurat.")
    }

    scaled_feature_count(obj, opts$sv_assay)
    # nfeatures sets upstream's `variable` flag (top-n by rank); left unset it was always 2000.
    sv <- FindSpatiallyVariableFeatures(
      obj,
      selection.method = opts$sv_selection_method,
      nfeatures = opts$sv_nfeatures
    )
    sv_df <- spatially_variable_table(sv, opts$sv_assay, opts$sv_selection_method)
    sv_df$gene <- rownames(sv_df)
    n_sv_features_tested <- nrow(sv_df)

    if (!is.null(opts$sv_nfeatures) && nrow(sv_df) > opts$sv_nfeatures) {
      sv_df <- sv_df[seq_len(opts$sv_nfeatures), , drop = FALSE]
    }
    n_sv_genes <- nrow(sv_df)

    sv_csv <- file.path(opts$output_dir, "seurat_spatial_variable_features.csv")
    write_csv_atomic(sv_df, sv_csv, row.names = FALSE)
  }

  obj_path  <- file.path(opts$output_dir, "seurat_spatial_obj.rds")
  meta_path <- file.path(opts$output_dir, "seurat_spatial_metadata.csv")
  umap_path <- file.path(opts$output_dir, "seurat_spatial_umap.csv")

  save_rds_atomic(obj, obj_path)
  write_csv_atomic(obj@meta.data, meta_path)

  umap_exists <- FALSE
  if ("umap" %in% names(obj@reductions) && isTRUE(opts$umap)) {
    umap_df <- as.data.frame(Embeddings(obj, reduction = "umap"))
    umap_df$spot <- rownames(umap_df)
    write_csv_atomic(umap_df, umap_path, row.names = FALSE)
    umap_exists <- TRUE
  }

  png_path <- file.path(opts$output_dir, "seurat_spatial_dimplot_clusters.png")
  # image_alpha fades the tissue image (Seurat's image.alpha). It used to be passed as `alpha`, which
  # for a grouped SpatialDimPlot is the SPOT opacity: alpha[1] = 0.1 drew every cluster at 10% over
  # an opaque image (hunt 2026-09-30, u30-uncovered-mcp-1). Spots are drawn opaque, as Seurat does.
  plot_err <- save_plot_png(
    png_path,
    SpatialDimPlot(
      obj,
      group.by = "seurat_clusters",
      label = TRUE,
      pt.size.factor = opts$spot_size,
      alpha = 1,
      image.alpha = opts$image_alpha
    )
  )
  if (!is.null(plot_err)) log_msg("Could not draw the spatial cluster plot: ", plot_err)

  n_cells <- ncol(obj)
  # Counted on the Spatial assay: the object can also hold the other feature types' assays, and its
  # DefaultAssay is sv_assay by now.
  n_genes <- nrow(obj[["Spatial"]])
  n_clusters <- length(unique(Idents(obj)))
  cluster_sizes <- as.list(table(Idents(obj)))

  output_files <- list(
    seurat_rds   = obj_path,
    metadata_csv = meta_path
  )
  if (is.null(plot_err)) output_files$dimplot_png <- png_path
  if (umap_exists) output_files$umap_csv <- umap_path
  if (!is.null(sv_csv)) output_files$spatial_variable_csv <- sv_csv

  res <- list(
    status       = "ok",
    tool         = "seurat",
    task         = "spatial_qc_cluster",
    warnings     = I(c(
      if (is.null(plot_err)) character(0) else paste("spatial cluster plot:", plot_err),
      if (n_spots_dropped > 0) sprintf("QC dropped %d spot(s) with fewer than min_features=%d detected genes", n_spots_dropped, opts$min_features) else character(0)
    )),
    data         = list(
      n_cells       = n_cells,
      n_genes       = n_genes,
      n_cells_input = n_spots_before,
      n_genes_input = n_genes_before
    ),
    output_files = output_files,
    params       = list(
      data_dir        = opts$data_dir,
      output_dir      = opts$output_dir,
      project         = opts$project,
      min_cells       = opts$min_cells,
      min_features    = opts$min_features,
      n_genes_dropped = n_genes_dropped,
      n_spots_dropped = n_spots_dropped,
      n_hvgs          = opts$n_hvgs,
      n_pcs           = opts$n_pcs,
      resolution      = opts$resolution,
      umap            = opts$umap,
      spatial_var     = opts$spatial_var,
      sv_assay            = opts$sv_assay,
      sv_selection_method = opts$sv_selection_method,
      sv_nfeatures        = opts$sv_nfeatures,
      seed                = opts$seed,
      seeds               = seeds
    ),
    summary      = c(
      list(n_clusters = n_clusters, cluster_sizes = cluster_sizes),
      if (isTRUE(opts$spatial_var)) list(
        n_sv_features_tested = n_sv_features_tested,
        n_sv_genes           = n_sv_genes
      ) else list()
    ),
    analysis     = paste0(
      sprintf(
        "Seurat spatial_qc_cluster completed: %d spots x %d genes after QC (dropped %d genes below min_cells=%d and %d spots below min_features=%d), found %d clusters.",
        n_cells, n_genes, n_genes_dropped, opts$min_cells, n_spots_dropped, opts$min_features, n_clusters
      ),
      if (isTRUE(opts$spatial_var)) sprintf(
        " %s ranked the %d scaled features of assay %s (the only ones it tests) and the top %d are in the spatially variable CSV.",
        opts$sv_selection_method, n_sv_features_tested, opts$sv_assay, n_sv_genes
      ) else ""
    )
  )

  if (length(other_assays) > 0L) {
    others_text <- paste(vapply(names(other_assays), function(ftype) sprintf(
      "%d %s feature(s) in assay %s", as.integer(feature_split$n_per_type[[ftype]]), ftype, other_assays[[ftype]]
    ), character(1)), collapse = ", ")
    res$params$feature_types_input <- feature_split$n_per_type
    res$params$feature_type_clustered <- "Gene Expression"
    res$params$other_feature_type_assays <- other_assays
    note <- sprintf(paste0(
      "the counts matrix holds %d feature types: only its %d Gene Expression features were QC-filtered, ",
      "normalised and clustered; the others are kept unprocessed in the saved object (%s)"
    ), length(feature_split$n_per_type), as.integer(n_genes_before), others_text)
    res$warnings <- I(c(res$warnings, note))
    res$analysis <- paste0(res$analysis, " Note: ", note, ".")
  }
  res
}

run_spatial_feature_plot <- function(opts) {
  if (is.null(opts$seurat_rds) || is.null(opts$output_dir)) {
    stop("spatial_feature_plot requires --seurat-rds and --output-dir")
  }
  if (length(opts$features) == 0L) {
    stop("No features provided for spatial_feature_plot.")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode spatial_feature_plot; seurat_rds = ", opts$seurat_rds)

  obj <- readRDS(opts$seurat_rds)
  png_paths <- character(0)
  plot_errors <- character(0)

  for (gene in opts$features) {
    out_name <- paste0("seurat_spatial_feature_", gene, ".png")
    png_path <- file.path(opts$output_dir, out_name)
    log_msg("SpatialFeaturePlot for ", gene, " -> ", png_path)

    # For SpatialFeaturePlot `alpha` maps expression to spot opacity; the image is faded by
    # image.alpha. image_alpha used to set the top of that opacity range and never reached the image
    # (hunt 2026-09-30, u30-uncovered-mcp-1). c(0.1, 1) is the Seurat vignette's expression ramp.
    plot_err <- save_plot_png(
      png_path,
      SpatialFeaturePlot(
        obj,
        features = gene,
        pt.size.factor = opts$spot_size,
        alpha = c(0.1, 1),
        image.alpha = opts$image_alpha
      )
    )
    if (is.null(plot_err)) {
      png_paths <- c(png_paths, png_path)
    } else {
      log_msg("Could not plot ", gene, ": ", plot_err)
      plot_errors <- c(plot_errors, paste0(gene, ": ", plot_err))
    }
  }

  # Figures are this mode's only deliverable. If none was drawn there is no result to report, so
  # the failure goes to main() and comes back as status "error" -- the same thing run_dimplot and
  # run_spatial_dimplot do by not catching at all. Reporting "ok, 0 plotted" hid two one-line
  # diagnoses: an object with no @images, and a feature name that is not in the object.
  if (length(png_paths) == 0L) {
    stop(sprintf(
      "spatial_feature_plot drew none of the %d requested feature(s): %s",
      length(opts$features), paste(plot_errors, collapse = "; ")
    ))
  }

  n_cells <- ncol(obj)

  # ncol is accepted and has never laid anything out: each feature is drawn to its own PNG, so
  # there is no grid. A value other than the one-column default is reported as ignored rather than
  # echoed as if it had shaped the figures.
  feature_params <- list(
    seurat_rds  = opts$seurat_rds,
    output_dir  = opts$output_dir,
    features    = opts$features,
    spot_size   = opts$spot_size,
    image_alpha = opts$image_alpha
  )
  ignored_warnings <- character(0)
  if (!identical(as.integer(opts$ncol), 1L)) {
    feature_params$ignored <- I("ncol")
    ignored_warnings <- sprintf(
      "ignored parameter(s) ncol: each feature is drawn to its own PNG, so ncol=%s lays out nothing",
      as.character(opts$ncol)
    )
  }

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "spatial_feature_plot",
    # I() so a single feature stays a JSON array: jsonlite's auto_unbox renders a length-1
    # character vector as a bare string, and a caller doing list(png_paths) then gets one entry
    # per character of the path.
    warnings     = I(c(plot_errors, ignored_warnings)),
    data         = list(n_cells = n_cells),
    output_files = list(png_paths = I(png_paths)),
    params       = feature_params,
    summary      = list(
      n_features_plotted = length(png_paths),
      n_features_failed  = length(plot_errors)
    ),
    analysis     = sprintf(
      "Seurat spatial_feature_plot completed: plotted %d of %d requested features for %d spots.%s",
      length(png_paths), length(opts$features), n_cells,
      if (length(plot_errors) == 0L) "" else paste0(" Not plotted -- ", paste(plot_errors, collapse = "; "), ".")
    )
  )
}

run_spatial_dimplot <- function(opts) {
  if (is.null(opts$seurat_rds) || is.null(opts$output_dir)) {
    stop("spatial_dimplot requires --seurat-rds and --output-dir")
  }

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode spatial_dimplot; seurat_rds = ", opts$seurat_rds)

  obj <- readRDS(opts$seurat_rds)

  if (!(opts$group_by %in% colnames(obj@meta.data))) {
    stop(sprintf("group_by '%s' not found in meta.data", opts$group_by))
  }

  png_path <- file.path(
    opts$output_dir,
    paste0("seurat_spatial_dimplot_", opts$group_by, ".png")
  )
  log_msg("Saving SpatialDimPlot to ", png_path)

  png(png_path, width = 1200, height = 1000)
  print(
    # image_alpha is the tissue image's opacity, not the spots' (hunt 2026-09-30, u30-uncovered-mcp-1):
    # see run_spatial_qc_cluster's cluster figure.
    SpatialDimPlot(
      obj,
      group.by = opts$group_by,
      label = isTRUE(opts$label),
      pt.size.factor = opts$spot_size,
      alpha = 1,
      image.alpha = opts$image_alpha
    )
  )
  dev.off()

  n_cells <- ncol(obj)

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "spatial_dimplot",
    data         = list(n_cells = n_cells),
    output_files = list(dimplot_png = png_path),
    params       = list(
      seurat_rds  = opts$seurat_rds,
      output_dir  = opts$output_dir,
      group_by    = opts$group_by,
      spot_size   = opts$spot_size,
      image_alpha = opts$image_alpha,
      label       = opts$label
    ),
    summary      = list(),
    analysis     = sprintf(
      "Seurat spatial_dimplot completed: %d spots, group_by=%s.",
      n_cells, opts$group_by
    )
  )
}

run_spatial_variable_features <- function(opts) {
  if (is.null(opts$seurat_rds) || is.null(opts$output_dir)) {
    stop("spatial_variable_features requires --seurat-rds and --output-dir")
  }
  assay <- if (is.null(opts$assay)) "Spatial" else opts$assay

  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  log_msg("Mode spatial_variable_features; seurat_rds = ", opts$seurat_rds)

  obj <- readRDS(opts$seurat_rds)

  if (!(assay %in% Assays(obj))) {
    stop(sprintf(
      "Assay '%s' not found. Available: %s",
      assay, paste(Assays(obj), collapse = ", ")
    ))
  }
  DefaultAssay(obj) <- assay

  if (!"FindSpatiallyVariableFeatures" %in% getNamespaceExports("Seurat")) {
    stop("FindSpatiallyVariableFeatures not exported by current Seurat.")
  }

  # Without an image there are no spot coordinates, and upstream fails on
  # "no applicable method for 'GetTissueCoordinates' applied to an object of class NULL".
  if (length(Images(obj)) == 0L) {
    stop(paste0(
      "the object holds no spatial image, so it has no spot coordinates to test. spatial_variable_features ",
      "needs a spatial Seurat object, such as the one seurat_spatial_qc_cluster saves."
    ))
  }
  n_features_scaled <- scaled_feature_count(obj, assay)
  # nfeatures sets upstream's `variable` flag (top-n by rank); left unset it was always 2000.
  sv <- FindSpatiallyVariableFeatures(
    obj,
    selection.method = opts$selection_method,
    nfeatures = opts$nfeatures
  )
  sv_df <- spatially_variable_table(sv, assay, opts$selection_method)
  sv_df$gene <- rownames(sv_df)
  # Scored features only (the helper drops the rank-NA rest); a scaled feature with zero variance
  # is also skipped upstream, so this can be below n_features_scaled.
  n_features_tested <- nrow(sv_df)

  if (!is.null(opts$nfeatures) && nrow(sv_df) > opts$nfeatures) {
    sv_df <- sv_df[seq_len(opts$nfeatures), , drop = FALSE]
  }

  out_path <- file.path(opts$output_dir, "seurat_spatial_variable_features.csv")
  write.csv(sv_df, out_path, quote = TRUE, row.names = FALSE)

  # "Spatially variable" is upstream's own definition: the top-nfeatures features by rank, the
  # rows whose `variable` flag is TRUE -- which, with nfeatures passed through, are the rows kept.
  n_sv_genes <- nrow(sv_df)
  n_cells <- ncol(obj)
  n_genes <- nrow(obj)
  n_features_not_tested <- n_genes - n_features_tested

  list(
    status       = "ok",
    tool         = "seurat",
    task         = "spatial_variable_features",
    warnings     = I(if (n_sv_genes < opts$nfeatures) sprintf(
      paste0(
        "nfeatures=%d asked for more genes than were tested: %s scores only the %d scaled features of ",
        "assay %s, so the CSV holds %d rows"
      ),
      opts$nfeatures, opts$selection_method, n_features_tested, assay, n_sv_genes
    ) else character(0)),
    data         = list(n_cells = n_cells, n_genes = n_genes),
    output_files = list(spatial_variable_csv = out_path),
    params       = list(
      seurat_rds       = opts$seurat_rds,
      output_dir       = opts$output_dir,
      assay            = assay,
      selection_method = opts$selection_method,
      nfeatures        = opts$nfeatures,
      layer_tested     = "scale.data"
    ),
    summary      = list(
      n_sv_genes            = n_sv_genes,
      n_features_scaled     = n_features_scaled,
      n_features_tested     = n_features_tested,
      n_features_not_tested = n_features_not_tested
    ),
    analysis     = sprintf(
      paste0(
        "Seurat spatial_variable_features completed: %s ranked the %d features of assay %s's scale.data ",
        "layer (the other %d of its %d features were not tested) and the top %d are reported as spatially ",
        "variable."
      ),
      opts$selection_method, n_features_tested, assay, n_features_not_tested, n_genes, n_sv_genes
    )
  )
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  if (is.null(opts$mode)) {
    stop("Missing --mode")
  }

  set.seed(opts$seed)

  res <- tryCatch(with_r_traceback({
    if (opts$mode == "qc_cluster") {
      run_qc_cluster(opts)
    } else if (opts$mode == "integrate_qc_cluster") {
      run_integrate_qc_cluster(opts)
    } else if (opts$mode == "find_markers") {
      run_find_markers(opts)
    } else if (opts$mode == "dimplot") {
      run_dimplot(opts)
    } else if (opts$mode == "spatial_qc_cluster") {
      run_spatial_qc_cluster(opts)
    } else if (opts$mode == "spatial_feature_plot") {
      run_spatial_feature_plot(opts)
    } else if (opts$mode == "spatial_dimplot") {
      run_spatial_dimplot(opts)
    } else if (opts$mode == "spatial_variable_features") {
      run_spatial_variable_features(opts)
    } else {
      stop(paste0("Unknown mode: ", opts$mode))
    }
  }), error = function(e) {
    log_msg("ERROR: ", e$message)
    tb <- sog_traceback()
    list(
      status    = "error",
      tool      = "seurat",
      task      = opts$mode,
      error     = paste("Seurat worker failed:", e$message),
      traceback = tb
    )
  })

  # null = "null": a parameter left unset (group_b, cluster, ...) is NULL here, and jsonlite's default
  # writes a NULL list element as {} -- an empty mapping, which a reader cannot tell from a value.
  cat(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null"), "\n")
}

if (identical(environment(), globalenv())) {
  main()
}

