#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(CellTrek)
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
  message(sprintf("[celltrek-worker] %s", paste0(...)))
}

# Seurat 4 kept the counts matrix in the data slot until something normalised it, so "not yet
# normalised" could be read as counts and data being identical. Seurat 5 keeps them in separate
# layers and an object built from counts alone has no data layer at all -- LayerData returns 0 x 0
# -- so that comparison reads FALSE and says "already normalised" about an empty matrix. Ask the
# layer itself instead. Same helper as tools/seurat_worker.R; there is no shared R library on the
# workers' path, so each worker carries its own copy.
data_layer_is_empty <- function(obj, assay) {
  d <- suppressWarnings(tryCatch(
    GetAssayData(obj, assay = assay, slot = "data"),
    error = function(e) NULL
  ))
  is.null(d) || length(dim(d)) < 2L || any(dim(d) == 0L)
}

# Seurat 4's GetAssay() selects with FilterObjects(classes.keep = "Assay"), and an Assay5 -- what
# any Seurat 5 install writes, including our own tools/h5ad_to_seurat.R -- does not inherit from
# Assay. Measured in this tool's env (Seurat 4.4.0, SeuratObject 5.0.2) on a 200-spot Assay5:
# Assays(), DefaultAssay(), obj[[assay]] and GetAssayData() all read it fine, so nothing complains
# at load time, and the failure surfaces much later inside NormalizeData() as "RNA is not an assay
# present in the given object. Available assays are: " -- an empty list, about an object that
# plainly has RNA.
#
# Ask the accessor, not the class: inherits(x, "Assay") is FALSE for that same object on a Seurat 5
# install too, where everything works, so a class test would force a pointless rebuild there.
seurat_can_read_assay <- function(obj, assay) {
  isTRUE(tryCatch(
    {
      GetAssay(obj, assay = assay)
      TRUE
    },
    error = function(e) FALSE
  ))
}

# Rebuild an assay this Seurat cannot use, keeping what the rest of the run depends on: the assay's
# own name (renaming "Spatial" to "RNA" would flip the cross-assay branch below), the cell metadata,
# any normalisation already present, the H&E image traint() reads, and the spot coordinates our
# converter attaches as a reduction. Measured: the rebuilt object returns counts identical to the
# original and runs NormalizeData and FindVariableFeatures.
readable_object <- function(obj, assay, label) {
  if (seurat_can_read_assay(obj, assay)) {
    return(obj)
  }
  shape <- paste(class(obj[[assay]]), collapse = "/")
  version <- as.character(packageVersion("Seurat"))
  counts <- tryCatch(SeuratObject::LayerData(obj[[assay]], "counts"), error = function(e) NULL)
  if (is.null(counts) || length(dim(counts)) < 2L || any(dim(counts) == 0L)) {
    stop(sprintf(
      paste0(
        "The %s object's '%s' assay is a %s, which Seurat %s cannot use, and it carries no counts ",
        "layer to rebuild it from. Re-save the object from a Seurat %s session."
      ),
      label, assay, shape, version, substr(version, 1, 1)
    ))
  }
  log_msg(
    "Rebuilding the ", label, " '", assay, "' assay: it is a ", shape,
    " and Seurat ", version, " cannot read that shape"
  )
  dropped <- setdiff(Assays(obj), assay)
  if (length(dropped) > 0) {
    log_msg("Rebuild keeps only '", assay, "'; dropping assay(s): ", paste(dropped, collapse = ", "))
  }
  rebuilt <- CreateSeuratObject(counts = counts, assay = assay, meta.data = obj@meta.data)
  if (ncol(rebuilt) != ncol(obj)) {
    stop(sprintf(
      "Rebuilding the %s object dropped %d of its %d cells, so the coordinates would no longer line up.",
      label, ncol(obj) - ncol(rebuilt), ncol(obj)
    ))
  }
  normalised <- tryCatch(SeuratObject::LayerData(obj[[assay]], "data"), error = function(e) NULL)
  if (!is.null(normalised) && length(dim(normalised)) == 2L && all(dim(normalised) > 0L)) {
    rebuilt <- SetAssayData(rebuilt, assay = assay, slot = "data", new.data = normalised)
  }
  # Slot assignment, as the cross-assay rebuild below already does for @images: both carry across
  # unvalidated, and dropping either one silently changes what the analysis is run on.
  rebuilt@images <- obj@images
  rebuilt@reductions <- obj@reductions
  DefaultAssay(rebuilt) <- assay
  rebuilt
}

# CellTrek reads the spot positions off the image object, not the picture inside it. Measured in
# this env: traint() reads only st_data@images[[1]]@coordinates[, "imagerow"] and [, "imagecol"],
# and celltrek() then reads back the coord_x/coord_y metadata traint() wrote. Neither function
# touches the @image raster.
#
# That matters because our own documented entrance cannot supply one. h5ad_to_seurat.R attaches the
# positions from spatial_coords.csv as a reduction named "spatial" plus x_coord/y_coord metadata
# columns and builds no image at all, so an object carrying every number CellTrek wants used to be
# refused for missing a photograph the method never opens.
is_axis_frame <- function(coords) {
  is.data.frame(coords) && all(c("imagerow", "imagecol") %in% colnames(coords))
}

image_carries_axes <- function(obj) {
  named <- Images(obj)
  if (length(named) == 0) {
    return(FALSE)
  }
  is_axis_frame(tryCatch(obj@images[[named[1]]]@coordinates, error = function(e) NULL))
}

# Which coordinate column holds the image ROW depends on who wrote the coordinates, and the two
# writers that reach this worker disagree:
#
#   * Seurat's image accessors put the row first. GetTissueCoordinates() on a VisiumV1 returns
#     (imagerow, imagecol). Its SlideSeq accessor returns (x, y, cells) without reordering them
#     (measured, Seurat 4.4.0), but Seurat draws column 1 of that accessor vertically and column 2
#     horizontally for every image type (SingleSpatialPlot: x = column 2, y = column 1), so column 1
#     is the row by Seurat's own convention there too.
#   * Our h5ad converter writes adata.obsm["spatial"][:, :2] -- as columns it names x and y -- into
#     the 'spatial' reduction and the x_coord/y_coord metadata. AnnData's obsm["spatial"] is
#     (x, y), i.e. (pixel COLUMN, pixel ROW): scanpy's read_visium labels Space Ranger's fifth
#     column (the pixel row) "pxl_col_in_fullres" and its sixth (the pixel column)
#     "pxl_row_in_fullres", then stacks [pxl_row_in_fullres, pxl_col_in_fullres] -- which, under
#     those swapped labels, is [pixel column, pixel row]. Measured on the library's
#     Targeted_Visium_Human_SpinalCord_Neuroscience: obsm["spatial"][:, 0] equals the
#     pxl_col_in_fullres column of Space Ranger's own tissue_positions.csv, spot for spot.
#
# So for the converter's sources column 2 is the row. Reading its column 1 as imagerow transposed
# every mapped cell: CellTrek's coord_x IS imagerow (traint's coord_xy = c("imagerow",
# "imagecol")), and its 'celltrek' layout is drawn from coord_x/coord_y.
#
# Rows are matched to cells by name wherever the source carries names; a source that lines up only
# by length is used in row order and says so. Pairing a spot with a different spot's position is
# the same silent-mismatch shape as the PRECAST barcode defect.
aligned_axes <- function(obj, xy, source, row_column = 1L) {
  if (is.null(xy) || length(dim(xy)) != 2L || ncol(xy) < 2L) {
    return(NULL)
  }
  cells <- colnames(obj)
  ids <- rownames(xy)
  if (!is.null(ids) && all(cells %in% ids)) {
    xy <- xy[cells, 1:2, drop = FALSE]
  } else if (nrow(xy) == length(cells)) {
    log_msg("Spot coordinates from the ", source, " carry no matching cell names; using row order")
    xy <- xy[, 1:2, drop = FALSE]
  } else {
    return(NULL)
  }
  axes <- suppressWarnings(matrix(as.numeric(as.matrix(xy)), nrow = length(cells), ncol = 2L))
  if (any(!is.finite(axes))) {
    return(NULL)
  }
  row_axis <- axes[, row_column]
  col_axis <- axes[, 3L - row_column]
  data.frame(
    imagerow = row_axis,
    imagecol = col_axis,
    row.names = cells
  )
}

# Where each source keeps the image row: Seurat's accessor puts it first, the converter's (x, y)
# puts it second. Named, so the log and the payload can say which column went where.
axis_mapping_text <- function(row_column) {
  if (identical(as.integer(row_column), 1L)) {
    "column 1 -> imagerow, column 2 -> imagecol"
  } else {
    "column 1 (x) -> imagecol, column 2 (y) -> imagerow"
  }
}

# GetTissueCoordinates() on a SlideSeq image returns (x, y, cells). as.matrix() on that frame is a
# CHARACTER matrix whose numbers went through format() -- measured: 12345.678912 comes back as
# "12345.68". Keep the numeric columns, so the positions arrive as the numbers they are.
numeric_coordinates <- function(coords) {
  if (is.data.frame(coords)) {
    coords <- coords[, vapply(coords, is.numeric, logical(1)), drop = FALSE]
  }
  as.matrix(coords)
}

# traint() copies the positions into the spot metadata BY POSITION --
# st_data$coord_x <- st_data@images[[1]]@coordinates[, "imagerow"] -- and never looks at the row
# names. Measured in this env: a first image whose rows run in another order than the spots gives
# every spot another spot's position, and a first image that covers only some of the spots (a merged
# two-section object) is RECYCLED over the rest -- both silently. So the first image is lined up with
# the spots by name here, and an object whose first image does not cover every spot is refused: it
# holds more than the one section CellTrek maps onto.
align_own_image <- function(obj) {
  image_name <- names(obj@images)[1]
  coords <- obj@images[[1]]@coordinates
  cells <- colnames(obj)
  if (identical(rownames(coords), cells)) {
    return(list(object = obj, note = NULL))
  }
  covered <- sum(cells %in% rownames(coords))
  if (covered < length(cells)) {
    stop(sprintf(
      paste0(
        "CellTrek reads every spot's position from the object's first image, and '%s' holds positions ",
        "for %d of the object's %d spots (it carries %d image(s): %s). CellTrek maps cells onto one ",
        "section: pass a spatial object that holds only the section to map onto."
      ),
      image_name, covered, length(cells), length(obj@images), paste(names(obj@images), collapse = ", ")
    ))
  }
  obj@images[[1]]@coordinates <- coords[cells, , drop = FALSE]
  note <- sprintf(
    "rows of image '%s' matched to the spots by name (%d image rows, %d spots)",
    image_name, nrow(coords), length(cells)
  )
  log_msg("Spot coordinates: ", note)
  list(object = obj, note = note)
}

# A VisiumV1 whose raster is zero-size and whose physical measurements are NA. Measured: that
# satisfies every read and write CellTrek makes -- Images(), @coordinates[, "imagerow"/"imagecol"],
# GetTissueCoordinates(), celltrek()'s own @scale.factors$spot_dis assignment, and saveRDS --
# while inventing no pixels, no array indices and no spot geometry. The restraint is load-bearing:
# celltrek() copies @images onto the object we save as the user's celltrek_result.rds, so anything
# fabricated here would ship as if it had been measured off a slide.
coordinate_carrier <- function(obj, axes) {
  new(
    "VisiumV1",
    image = array(0, dim = c(0L, 0L, 3L)),
    scale.factors = scalefactors(
      spot = NA_real_, fiducial = NA_real_, hires = NA_real_, lowres = NA_real_
    ),
    coordinates = axes,
    spot.radius = NA_real_,
    assay = DefaultAssay(obj),
    key = "slice1_"
  )
}

with_spot_coordinates <- function(obj) {
  own_image <- function() paste0("the object's own image '", Images(obj)[1], "'")
  own_axes <- "imagerow/imagecol read by name from the image"
  if (image_carries_axes(obj)) {
    aligned <- align_own_image(obj)
    axes <- if (is.null(aligned$note)) own_axes else paste0(own_axes, "; ", aligned$note)
    return(list(object = aligned$object, source = own_image(), axes = axes))
  }
  # Assigning NULL into a list drops the entry, so a source that cannot be read simply is not
  # offered. Order is deliberate: a real image outranks what our converter derived from it. Each
  # source carries the column its image row sits in (see aligned_axes above).
  candidates <- list()
  row_columns <- list()
  if (length(Images(obj)) > 0) {
    source <- "existing image, via GetTissueCoordinates()"
    candidates[[source]] <- tryCatch(numeric_coordinates(GetTissueCoordinates(obj)), error = function(e) NULL)
    row_columns[[source]] <- 1L
  }
  if ("spatial" %in% names(obj@reductions)) {
    source <- "'spatial' reduction our h5ad converter writes"
    candidates[[source]] <- tryCatch(Embeddings(obj, reduction = "spatial"), error = function(e) NULL)
    row_columns[[source]] <- 2L
  }
  if (all(c("x_coord", "y_coord") %in% colnames(obj@meta.data))) {
    source <- "x_coord/y_coord metadata our h5ad converter writes"
    candidates[[source]] <- as.matrix(obj@meta.data[, c("x_coord", "y_coord")])
    row_columns[[source]] <- 2L
  }
  for (source in names(candidates)) {
    axes <- aligned_axes(obj, candidates[[source]], source, row_column = row_columns[[source]])
    if (is.null(axes)) {
      next
    }
    mapping <- axis_mapping_text(row_columns[[source]])
    log_msg("Spot coordinates from the ", source, ": ", mapping)
    obj@images <- list(slice1 = coordinate_carrier(obj, axes))
    return(list(object = obj, source = source, axes = mapping))
  }
  if (length(Images(obj)) > 0) {
    return(list(object = obj, source = own_image(), axes = own_axes))
  }
  list(object = obj, source = "not found", axes = "none")
}

# --- background spots ------------------------------------------------------------------------
#
# convert_h5ad_to_seurat_rds carries every obs column into meta.data, in_tissue included, and a
# CELLxGENE Visium h5ad holds every array spot: on the library's Heart Fetal12W sample 3,009 of
# 4,992 are background glass. CellTrek trains its coordinate random forest on every spot and
# interpolates its 10,000 points around every spot, so single cells were charted onto glass. The
# rule is tools/worker_utils.keep_in_tissue's, which the Python workers apply to obs['in_tissue']:
# 1 / "1" / TRUE is tissue; 0, FALSE, empty and anything else is background. No column, or one that
# is 1 everywhere, leaves the object as it is; a column that marks no spot as tissue is refused.
in_tissue_flags <- function(meta) {
  hit <- which(tolower(trimws(colnames(meta))) == "in_tissue")
  if (length(hit) == 0L) {
    return(NULL)
  }
  raw <- tolower(trimws(as.character(meta[[hit[[1L]]]])))
  raw[raw %in% "true"] <- "1"
  raw[raw %in% "false"] <- "0"
  flag <- suppressWarnings(as.numeric(raw))
  !is.na(flag) & flag == 1
}

# The spatial object without its background spots. `filter` has the shape
# worker_utils.record_in_tissue writes to params.in_tissue_filter, and is NULL, like `warning`, when
# nothing was left out.
keep_in_tissue_spots <- function(obj) {
  none <- list(object = obj, n_dropped = 0L, filter = NULL, warning = NULL)
  flags <- in_tissue_flags(obj@meta.data)
  if (is.null(flags) || ncol(obj) == 0L) {
    return(none)
  }
  n <- ncol(obj)
  if (!any(flags)) {
    seen <- utils::head(sort(unique(as.character(obj@meta.data[[
      which(tolower(trimws(colnames(obj@meta.data))) == "in_tissue")[[1L]]
    ]]))), 8L)
    stop(sprintf(
      paste0(
        "The spatial object's in_tissue metadata marks none of its %d spots as in tissue (values seen: %s); ",
        "fix the column so in-tissue spots are 1, or remove it if every spot is tissue."
      ),
      n, paste(seen, collapse = ", ")
    ))
  }
  n_dropped <- sum(!flags)
  if (n_dropped == 0L) {
    return(none)
  }
  kept <- colnames(obj)[flags]
  obj <- subset(obj, cells = kept)
  if (ncol(obj) != length(kept)) {
    stop(sprintf("Leaving out the background spots kept %d spots where %d are in tissue.", ncol(obj), length(kept)))
  }
  log_msg("Left out ", n_dropped, " of ", n, " spots with in_tissue == 0 (background outside the tissue)")
  list(
    object = obj,
    n_dropped = n_dropped,
    filter = list(n_spots_supplied = n, n_spots_off_tissue_dropped = n_dropped, n_spots_used = length(kept)),
    warning = sprintf(
      paste0(
        "%d of %d spots have in_tissue == 0 (background outside the tissue) and were left out; CellTrek mapped ",
        "cells onto the %d in-tissue spots."
      ),
      n_dropped, n, length(kept)
    )
  )
}

# --- what the embedding CellTrek builds can be asked for -------------------------------------
#
# traint() computes exactly two reductions on the joint object: RunPCA(features = the integration
# features) at Seurat's default npcs = 50, and RunUMAP(dims = 1:30) at Seurat's default
# n.components = 2. celltrek() then indexes @reductions[[reduction]]@cell.embeddings[, 1:nPCs]
# with no check, so any other name, n_components above what that reduction has, or a single
# component (the interpolation step applies over a matrix, and one column drops to a vector) fails
# with a subscript error -- after traint(), the expensive half, has already run. Ask before.
CELLTREK_REDUCTION_WIDTH <- c(pca = 50L, umap = 2L)

check_reduction_request <- function(reduction, n_components) {
  if (is.null(reduction) || length(reduction) != 1L || is.na(reduction) ||
      !(reduction %in% names(CELLTREK_REDUCTION_WIDTH))) {
    stop(sprintf(
      paste0(
        "reduction='%s' is not one CellTrek builds. traint() computes only 'pca' (%d components) ",
        "and 'umap' (%d components) on the joint embedding, and celltrek() reads the one you name."
      ),
      paste(reduction, collapse = ","), CELLTREK_REDUCTION_WIDTH[["pca"]], CELLTREK_REDUCTION_WIDTH[["umap"]]
    ))
  }
  width <- CELLTREK_REDUCTION_WIDTH[[reduction]]
  if (length(n_components) != 1L || is.na(n_components) || n_components < 2L || n_components > width) {
    stop(sprintf(
      paste0(
        "n_components=%s is outside what reduction='%s' has: traint() builds %d '%s' components, ",
        "and CellTrek needs at least 2. Pass n_components between 2 and %d%s."
      ),
      paste(n_components, collapse = ","), reduction, width, reduction, width,
      if (identical(reduction, "umap")) " (for more components, use reduction='pca')" else ""
    ))
  }
  invisible(width)
}

# The width actually built. RunPCA returns fewer than 50 components when the integration features
# or the cells are fewer, so the static bound above is necessary but not sufficient.
check_reduction_built <- function(st_sc_int, reduction, n_components) {
  emb <- tryCatch(st_sc_int@reductions[[reduction]]@cell.embeddings, error = function(e) NULL)
  built <- if (is.null(emb)) 0L else ncol(emb)
  if (n_components > built) {
    stop(sprintf(
      paste0(
        "traint() built %d '%s' component(s) on this pair of objects and n_components=%d asks for ",
        "more. Pass n_components <= %d."
      ),
      built, reduction, as.integer(n_components), built
    ))
  }
  built
}

# --- normalisation ---------------------------------------------------------------------------
#
# traint() does not normalise. FindTransferAnchors(reduction = "cca") scales the data slot it is
# handed, TransferData() transfers the single-cell data slot, and the joint PCA is run on a matrix
# that cbinds the transferred spot values to sc_data[[sc_assay]]@data. An object built from counts
# alone -- every .rds tools/h5ad_to_seurat.R writes, and every library reference -- carries raw
# counts in that slot (Seurat 4) or no data layer at all (Seurat 5). The spatial side was
# normalised here; the single-cell side never was, so log-normalised spots were co-embedded with
# raw-count cells. Both sides now ask the same question and get the same treatment.
needs_normalisation <- function(obj, assay) {
  data_layer_is_empty(obj, assay) ||
    identical(
      GetAssayData(obj, assay = assay, slot = "counts"),
      GetAssayData(obj, assay = assay, slot = "data")
    )
}

# TRUE when every stored value is a whole number. LogNormalize reads its input as counts; a matrix
# that is already log-scaled in its counts slot would be logged twice, which the payload must say.
holds_whole_numbers <- function(m) {
  values <- if (methods::is(m, "sparseMatrix")) m@x else as.vector(as.matrix(m))
  values <- values[is.finite(values)]
  length(values) == 0L || all(values == round(values))
}

# How many stored values are negative, and how many are NaN/Inf. LogNormalize reads its input as
# counts: a scaled (z-scored) matrix -- what convert_h5ad_to_seurat_rds carries across from a
# CELLxGENE export whose X is scaled, the library's Skin FaceTemple sample -- comes out of it as NaN,
# and traint() then co-embeds NaN spots. The fleet's rule (worker_utils.choose_counts_matrix) refuses
# such a matrix instead; the counts sit in adata.raw there, which an RDS does not carry.
not_count_values <- function(m) {
  values <- if (methods::is(m, "sparseMatrix")) m@x else as.vector(as.matrix(m))
  c(negative = sum(values < 0, na.rm = TRUE), nonfinite = sum(!is.finite(values)))
}

normalise_if_needed <- function(obj, assay, label) {
  if (!needs_normalisation(obj, assay)) {
    log_msg("Using the ", label, " object's own normalised '", assay, "' data layer")
    return(list(object = obj, how = "supplied data layer (used as given)", ran = FALSE, warning = NULL))
  }
  bad <- not_count_values(GetAssayData(obj, assay = assay, slot = "counts"))
  if (any(bad > 0)) {
    stop(sprintf(
      paste0(
        "The %s object's '%s' counts hold %d negative and %d NaN/Inf values (scaled data, not counts) and it has ",
        "no normalised layer, so LogNormalize would turn them into NaN. Supply raw counts: for an h5ad whose ",
        "counts sit in adata.raw (a CELLxGENE export), put adata.raw in X before convert_h5ad_to_seurat_rds, or ",
        "give the object a normalised data layer."
      ),
      label, assay, as.integer(bad[["negative"]]), as.integer(bad[["nonfinite"]])
    ))
  }
  warning_text <- NULL
  if (!holds_whole_numbers(GetAssayData(obj, assay = assay, slot = "counts"))) {
    warning_text <- paste0(
      "the ", label, " object's '", assay, "' counts hold non-integer values and it has no separate ",
      "normalised layer; LogNormalize was applied to them as if they were counts"
    )
    log_msg("WARNING: ", warning_text)
  }
  log_msg("Normalizing the ", label, " '", assay, "' assay (LogNormalize; no normalised layer present)")
  obj <- NormalizeData(obj, assay = assay, verbose = FALSE)
  list(
    object = obj, how = "LogNormalize (run by this worker; no normalised layer was supplied)",
    ran = TRUE, warning = warning_text
  )
}

# --- cell names ------------------------------------------------------------------------------
#
# traint() keys the joint object by make.names(<cell name>), and celltrek() then indexes the
# ORIGINAL single-cell object with those keys: sc_data[[sc_assay]]@data[, sc_coord$id_raw]. Any
# name make.names() changes -- every 10x barcode, "AAACCTG-1" -> "AAACCTG.1" -- is then absent and
# the run dies with "subscript out of bounds" after the random forest has run. CellTrek's tutorial
# renames both objects with make.names() before traint(); the single-cell object is the one read
# back, so it is renamed here, and the count reaches the payload.
syntactic_cell_names <- function(obj, label) {
  old <- colnames(obj)
  new <- make.names(old)
  if (anyDuplicated(new)) {
    clash <- new[duplicated(new)][1]
    stop(sprintf(
      paste0(
        "CellTrek keys cells by make.names(<cell name>), and %d of the %s object's %d cell names ",
        "collide once rewritten that way (e.g. '%s' names %s). Give those cells distinct names."
      ),
      sum(duplicated(new)), label, length(old), clash,
      paste0("'", old[new == clash], "'", collapse = " and ")
    ))
  }
  changed <- new != old
  if (any(changed)) {
    first <- which(changed)[1]
    log_msg(
      "Renamed ", sum(changed), " of ", length(old), " ", label, " cell names to the form CellTrek keys ",
      "them by (make.names), e.g. '", old[first], "' -> '", new[first], "'"
    )
    obj <- RenameCells(obj, new.names = new)
  }
  list(object = obj, n_renamed = sum(changed))
}

# traint() also stacks the spot and cell metadata under make.names(<name>) row names, so a spot
# and a cell that share a name stop it with "duplicate 'row.names' are not allowed".
check_no_shared_names <- function(st_obj, sc_obj) {
  shared <- intersect(make.names(colnames(st_obj)), make.names(colnames(sc_obj)))
  if (length(shared) > 0) {
    stop(sprintf(
      paste0(
        "%d spot name(s) in the spatial object are also cell names in the single-cell object ",
        "(e.g. '%s'), and CellTrek stacks both under one set of names. Give one of the two objects ",
        "distinct cell names."
      ),
      length(shared), shared[1]
    ))
  }
  invisible(TRUE)
}

# traint() co-embeds the two objects on the genes both measure. With none in common -- one side
# keyed by Ensembl IDs and the other by symbols, as the library's Heart sample is against a symbol
# reference -- it stops inside FindTransferAnchors with "'x' must be atomic" (measured), which names
# neither object nor the cause. Count them first; the count also reaches the payload.
check_shared_genes <- function(st_obj, st_assay, sc_obj, sc_assay) {
  st_genes <- rownames(st_obj[[st_assay]])
  sc_genes <- rownames(sc_obj[[sc_assay]])
  n_shared <- length(intersect(st_genes, sc_genes))
  if (n_shared == 0L) {
    stop(sprintf(
      paste0(
        "The spatial and single-cell objects share no gene names (%d spatial genes, e.g. %s; %d ",
        "single-cell genes, e.g. %s), and CellTrek co-embeds the two on the genes they share. One side ",
        "may be keyed by Ensembl IDs and the other by gene symbols: give both the same kind of gene name."
      ),
      length(st_genes), paste(utils::head(st_genes, 3), collapse = ", "),
      length(sc_genes), paste(utils::head(sc_genes, 3), collapse = ", ")
    ))
  }
  n_shared
}

# --- memory ----------------------------------------------------------------------------------
#
# celltrek() asks randomForestSRC for the proximity distance between EVERY pair of rows it predicts
# on: all single cells, all spots, and the points it interpolates between spots (intp_pnt). That
# N x N matrix is intrinsic to the method -- celltrek_chart() ranks each cell against every spot
# and interpolated point -- and randomForestSRC 3.3 unpacks it in R from a packed lower triangle,
# so both are alive at once: at least 1.5 * N^2 doubles. That is a floor, not the whole peak.
# When the floor alone exceeds what the machine has free, the run can only be killed, and a killed
# worker prints no JSON; say so with the numbers before traint() spends its time.
CELLTREK_INTERP_POINTS <- 10000L

# --- what leaves a cell unplaced -------------------------------------------------------------
#
# celltrek() charts cells with celltrek_chart(dist_mat, dist_cut = ntree * dist_thresh, top_spot,
# spot_n): it sets every distance above dist_cut to NA, then keeps a cell-point pair only when the
# point is among the cell's top_spot nearest (spots and interpolated points) AND the cell is among
# the point's spot_n nearest (an inner_join of the two top_n tables). The distances are
# randomForestSRC's (predict.rfsrc(distance = "all")), which its own documentation defines as a ratio
# of edge counts -- a number in [0, 1]; measured in this env (randomForestSRC 3.3.0) on the same call
# shape: 0 .. 0.997. At ntree = 1000 and dist_thresh = 0.55 the cut is 550, which no distance
# reaches, so it removes no pair and the mutual top-N pruning is the only thing that leaves a cell
# unplaced. The analysis used to say those cells "had no spot within the distance threshold".
CELLTREK_NTREE <- 1000L
CELLTREK_DIST_THRESH <- 0.55
CELLTREK_TOP_SPOT <- 5L
CELLTREK_SPOT_N <- 5L
# The largest distance randomForestSRC can return (a ratio of edge counts).
RF_DISTANCE_MAX <- 1

distance_cut_report <- function(ntree = CELLTREK_NTREE, dist_thresh = CELLTREK_DIST_THRESH) {
  cut <- as.numeric(ntree) * as.numeric(dist_thresh)
  list(
    ntree = as.integer(ntree),
    dist_thresh = as.numeric(dist_thresh),
    distance_cut = cut,
    distance_range = c(0, RF_DISTANCE_MAX),
    removes_pairs = cut < RF_DISTANCE_MAX
  )
}

# Why the cells CellTrek did not place are unplaced, in the words the analysis uses.
unplaced_reason <- function(cut_report, top_spot = CELLTREK_TOP_SPOT, spot_n = CELLTREK_SPOT_N) {
  pruning <- sprintf(
    paste0(
      "CellTrek's mutual nearest-neighbour pruning (each cell keeps its %d nearest spots or interpolated ",
      "points, each of those keeps its %d nearest cells, and a pair survives only when both keep it)"
    ),
    as.integer(top_spot), as.integer(spot_n)
  )
  if (isTRUE(cut_report$removes_pairs)) {
    return(sprintf(
      "%s and its distance cut (ntree x dist_thresh = %g)", pruning, cut_report$distance_cut
    ))
  }
  sprintf(
    paste0(
      "%s; its distance cut (ntree x dist_thresh = %d x %g = %g) lies above every random-forest distance, ",
      "which is in [0, %g], so it removed no pair"
    ),
    pruning, cut_report$ntree, cut_report$dist_thresh, cut_report$distance_cut, RF_DISTANCE_MAX
  )
}

# Said on every run where it holds: the published filter did not apply, so a reader who expects
# CellTrek's distance filter to have removed distant cells is told it did not.
distance_cut_warning <- function(cut_report) {
  if (isTRUE(cut_report$removes_pairs)) {
    return(NULL)
  }
  sprintf(
    paste0(
      "CellTrek's distance filter did not apply: celltrek() cuts cell-point distances above ntree x ",
      "dist_thresh = %d x %g = %g, and the randomForestSRC distances it cuts are ratios in [0, %g]. Cells ",
      "were placed or left unplaced by the mutual top-%d/top-%d pruning alone."
    ),
    cut_report$ntree, cut_report$dist_thresh, cut_report$distance_cut, RF_DISTANCE_MAX,
    CELLTREK_TOP_SPOT, CELLTREK_SPOT_N
  )
}

distance_matrix_bytes <- function(n_sc, n_st, n_interp = CELLTREK_INTERP_POINTS) {
  n <- as.numeric(n_sc) + as.numeric(n_st) + as.numeric(n_interp)
  1.5 * n * n * 8
}

# Free memory by the rule of tools/worker_utils.py available_memory_bytes(), which an R worker cannot
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

check_distance_matrix_fits <- function(n_sc, n_st, available = available_memory_bytes()) {
  need <- distance_matrix_bytes(n_sc, n_st)
  if (!is.finite(available)) {
    log_msg("Could not read the free memory; the random-forest distance step needs at least ",
            sprintf("%.1f GB", need / 1024^3))
    return(invisible(need))
  }
  if (need > available) {
    stop(sprintf(
      paste0(
        "CellTrek's random-forest step builds a distance matrix over every single cell, spot and ",
        "interpolated point: N = %d cells + %d spots + %d interpolated points = %d, and holding it ",
        "takes at least %.1f GB (1.5 x N^2 doubles). This machine has %.1f GB available (the smaller of ",
        "MemAvailable and the room under its cgroup memory limit, page cache counted as free). Run it ",
        "where at least %.1f GB is free."
      ),
      as.integer(n_sc), as.integer(n_st), CELLTREK_INTERP_POINTS,
      as.integer(n_sc + n_st + CELLTREK_INTERP_POINTS), need / 1024^3, available / 1024^3, need / 1024^3
    ))
  }
  invisible(need)
}

# How many threads randomForestSRC grows and predicts the forest on, read the way it reads it
# (randomForestSRC:::get.rf.cores: the rf.cores option, else the RF_CORES variable, else every
# OpenMP thread) but without setting the option as that function does. It matters for `seed`:
# measured with randomForestSRC 3.3.0, two runs at one seed on 3 threads return distance matrices
# that differ in the last bit (4e-16), which reorders spots CellTrek ranks as equally near; on one
# thread they are identical. On the library smoke that moved 99% of placements between two runs.
random_forest_threads <- function() {
  v <- getOption("rf.cores")
  if (is.null(v)) {
    v <- suppressWarnings(as.integer(Sys.getenv("RF_CORES")))
  }
  v <- suppressWarnings(as.integer(v))
  if (length(v) != 1L || is.na(v) || v < 1L) {
    return("every OpenMP thread (randomForestSRC default)")
  }
  v
}

# saveRDS / write.csv straight to the published name leave a truncated file there when the worker
# is killed mid-write; the rename is atomic on one filesystem.
save_rds_atomic <- function(object, path) {
  partial <- paste0(path, ".partial")
  saveRDS(object, partial)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " to ", path)
  }
  invisible(path)
}

write_csv_atomic <- function(df, path) {
  partial <- paste0(path, ".partial")
  write.csv(df, partial, row.names = TRUE)
  if (!file.rename(partial, path)) {
    stop("could not move ", partial, " to ", path)
  }
  invisible(path)
}

parse_args <- function(args) {
  opts <- list(
    sc_h5ad      = NULL,
    st_h5ad      = NULL,
    output_dir   = "celltrek_output",
    celltype_col = "cell_type",
    n_components = 30L,
    reduction    = "pca",
    seed         = 42L
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[i]
    if (key == "--sc-h5ad")       { opts$sc_h5ad      <- args[i+1]; i <- i + 2 }
    else if (key == "--st-h5ad")  { opts$st_h5ad      <- args[i+1]; i <- i + 2 }
    else if (key == "--output-dir")   { opts$output_dir   <- args[i+1]; i <- i + 2 }
    else if (key == "--celltype-col") { opts$celltype_col <- args[i+1]; i <- i + 2 }
    else if (key == "--n-components") { opts$n_components <- as.integer(args[i+1]); i <- i + 2 }
    else if (key == "--reduction")    { opts$reduction    <- args[i+1]; i <- i + 2 }
    else if (key == "--seed")         { opts$seed         <- as.integer(args[i+1]); i <- i + 2 }
    else { log_msg("Unknown arg: ", key); i <- i + 1 }
  }
  opts
}

run_celltrek <- function(opts) {
  log_msg("Starting CellTrek analysis")
  # Cheap, and it saves a traint() run that could only end in a subscript error.
  check_reduction_request(opts$reduction, opts$n_components)
  dir.create(opts$output_dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(opts$seed)

  log_msg("Loading scRNA-seq data: ", opts$sc_h5ad)
  log_msg("Loading spatial data: ", opts$st_h5ad)

  # CellTrek works on Seurat objects; the documented entrance for an h5ad is
  # convert_h5ad_to_seurat_rds, which writes the .rds read here.
  if (grepl("\\.rds$", opts$sc_h5ad, ignore.case = TRUE)) {
    sc_obj <- readRDS(opts$sc_h5ad)
  } else {
    stop("CellTrek requires Seurat RDS objects. Convert h5ad to RDS first.")
  }

  if (grepl("\\.rds$", opts$st_h5ad, ignore.case = TRUE)) {
    st_obj <- readRDS(opts$st_h5ad)
  } else {
    stop("CellTrek requires Seurat RDS objects. Convert h5ad to RDS first.")
  }

  n_sc <- ncol(sc_obj)
  n_st <- ncol(st_obj)
  n_st_supplied <- n_st
  log_msg("scRNA cells: ", n_sc, ", ST spots: ", n_st)

  # Detect assay names for ST and SC objects
  st_assay_orig <- if ("Spatial" %in% Assays(st_obj)) "Spatial" else DefaultAssay(st_obj)
  sc_assay <- if ("RNA" %in% Assays(sc_obj)) "RNA" else DefaultAssay(sc_obj)
  log_msg("Detected ST assay: ", st_assay_orig, ", SC assay: ", sc_assay)

  # Both objects come off disk in whatever shape their writer's Seurat chose, and this env's
  # Seurat cannot use all of them. Settle that here, where we can name the assay and the version,
  # rather than let it surface from inside NormalizeData() or traint() as an absent assay.
  st_obj <- readable_object(st_obj, st_assay_orig, "spatial")
  sc_obj <- readable_object(sc_obj, sc_assay, "single-cell")

  # Background spots out before anything is placed on (or sized from) the spots.
  tissue <- keep_in_tissue_spots(st_obj)
  st_obj <- tissue$object
  n_st <- ncol(st_obj)

  # traint() reads the spot positions out of the image object, so recover them from wherever this
  # object carries them before deciding it cannot be used.
  prepared <- with_spot_coordinates(st_obj)
  st_obj <- prepared$object
  coord_source <- prepared$source
  coord_axes <- prepared$axes
  if (length(Images(st_obj)) == 0) {
    stop(
      "ST object carries no spot coordinates. CellTrek maps cells onto positions, so the spatial ",
      "object needs a Visium slice, a 'spatial' reduction, or x_coord/y_coord metadata."
    )
  }
  if (!image_carries_axes(st_obj)) {
    stop(
      "The spatial object's first image '", Images(st_obj)[1], "' holds no imagerow/imagecol ",
      "positions, and neither a 'spatial' reduction nor x_coord/y_coord metadata could be read in their ",
      "place. CellTrek reads each spot's position from those two image columns."
    )
  }
  log_msg("ST images: ", paste(Images(st_obj), collapse = ", "))

  # Ensure the SC object has the cell type column
  if (!(opts$celltype_col %in% colnames(sc_obj@meta.data))) {
    avail <- paste(colnames(sc_obj@meta.data), collapse = ", ")
    stop(paste0("Cell type column '", opts$celltype_col, "' not found. Available: ", avail))
  }

  # celltrek() reads the single-cell object back under the names traint() gives it.
  renamed <- syntactic_cell_names(sc_obj, "single-cell")
  sc_obj <- renamed$object
  check_no_shared_names(st_obj, sc_obj)
  n_shared_genes <- check_shared_genes(st_obj, st_assay_orig, sc_obj, sc_assay)
  log_msg("Genes measured by both objects: ", n_shared_genes)

  # The one dense intermediate CellTrek cannot do without; refuse with the numbers rather than be
  # killed by it after traint().
  check_distance_matrix_fits(n_sc, n_st)

  # Normalise each side on its OWN full gene set (library sizes over every gene it measured), and
  # keep a normalised layer the caller supplied.
  st_norm <- normalise_if_needed(st_obj, st_assay_orig, "spatial")
  st_obj <- st_norm$object
  sc_norm <- normalise_if_needed(sc_obj, sc_assay, "single-cell")
  sc_obj <- sc_norm$object
  run_warnings <- c(tissue$warning, st_norm$warning, sc_norm$warning)

  # Workaround for Seurat v4/SeuratObject v5 cross-assay CCA bug:
  # When ST and SC objects use different assay names (e.g., "Spatial" vs "RNA"),
  # RunCCA inside traint() fails with "length of 'dimnames' [1] not equal to
  # array extent" because merge() and Cells() behave inconsistently across
  # assay names. Fix: convert ST to use an "RNA" assay with shared features.
  if (st_assay_orig != sc_assay) {
    log_msg("Converting ST assay '", st_assay_orig, "' to '", sc_assay,
            "' (Seurat v4/v5 cross-assay CCA workaround)")
    shared_features <- intersect(
      rownames(st_obj[[st_assay_orig]]),
      rownames(sc_obj[[sc_assay]])
    )
    log_msg("Shared features between ST and SC: ", length(shared_features))

    st_counts <- GetAssayData(st_obj, assay = st_assay_orig, slot = "counts")
    st_norm_data <- GetAssayData(st_obj, assay = st_assay_orig, slot = "data")
    st_obj_fixed <- CreateSeuratObject(
      counts = st_counts[shared_features, ],
      assay = sc_assay
    )
    # The normalised values carried across are the ones computed on the full spatial gene set
    # above; re-normalising here would compute every spot's library size over the shared genes only.
    st_obj_fixed <- SetAssayData(
      st_obj_fixed,
      assay = sc_assay, slot = "data", new.data = st_norm_data[shared_features, , drop = FALSE]
    )
    st_obj_fixed <- FindVariableFeatures(st_obj_fixed, verbose = FALSE)
    st_obj_fixed <- ScaleData(st_obj_fixed, verbose = FALSE)
    # Preserve image data (required by traint for spatial coordinates)
    st_obj_fixed@images <- st_obj@images
    st_obj <- st_obj_fixed
  } else if (isTRUE(st_norm$ran)) {
    st_obj <- FindVariableFeatures(st_obj, assay = st_assay_orig, verbose = FALSE)
  }

  st_assay <- sc_assay
  log_msg("Using assay '", st_assay, "' for both ST and SC in traint()")

  # CellTrek: Step 1 — co-embedding via traint()
  log_msg("Running CellTrek traint (co-embedding)...")
  result <- tryCatch(with_r_traceback({
    # Redirect CellTrek stdout to stderr so only JSON goes to stdout
    sink(stderr())
    st_sc_int <- CellTrek::traint(
      st_data = st_obj,
      sc_data = sc_obj,
      cell_names = opts$celltype_col,
      st_assay = st_assay,
      sc_assay = sc_assay,
      nfeatures = 2000
    )
    log_msg("CellTrek traint complete")
    n_built <- check_reduction_built(st_sc_int, opts$reduction, opts$n_components)

    # traint() ends in RunUMAP(), which calls set.seed(42) -- as RunCCA() and RunPCA() before it
    # do -- so every draw celltrek() makes after it (the interpolated points, the random forest's own
    # seed, the charting and repelling jitter) came from 42 whatever `seed` said: the parameter
    # changed nothing. Re-seed here, so `seed` governs the steps that are CellTrek's own.
    set.seed(opts$seed)
    rf_threads <- random_forest_threads()
    log_msg("Random forest threads: ", rf_threads)

    # Step 2 — cell mapping via celltrek()
    log_msg("Running CellTrek cell mapping...")
    celltrek_raw <- CellTrek::celltrek(
      st_sc_int = st_sc_int,
      sc_data = sc_obj,
      sc_assay = sc_assay,
      reduction = opts$reduction,
      nPCs = opts$n_components,
      intp_pnt = CELLTREK_INTERP_POINTS,
      ntree = CELLTREK_NTREE,
      dist_thresh = CELLTREK_DIST_THRESH,
      top_spot = CELLTREK_TOP_SPOT,
      spot_n = CELLTREK_SPOT_N,
      repel_r = 20,
      repel_iter = 20
    )
    sink()  # Restore stdout for JSON output
    log_msg("CellTrek mapping complete")

    # celltrek() returns a list with a "celltrek" element containing the Seurat object
    if (is.list(celltrek_raw) && !is(celltrek_raw, "Seurat")) {
      # Extract the Seurat object from the list
      ct_name <- intersect(names(celltrek_raw), c("celltrek", "CellTrek"))
      if (length(ct_name) > 0) {
        celltrek_result <- celltrek_raw[[ct_name[1]]]
        log_msg("Extracted Seurat object from celltrek() list element '", ct_name[1], "'")
      } else if (length(celltrek_raw) == 1 && is(celltrek_raw[[1]], "Seurat")) {
        celltrek_result <- celltrek_raw[[1]]
        log_msg("Extracted Seurat object from celltrek() list (single element)")
      } else {
        stop("celltrek() returned an unexpected list structure: ", paste(names(celltrek_raw), collapse = ", "))
      }
    } else {
      celltrek_result <- celltrek_raw
    }

    # Save results
    out_rds <- file.path(opts$output_dir, "celltrek_result.rds")
    save_rds_atomic(celltrek_result, out_rds)
    log_msg("Saved RDS: ", out_rds)

    # Extract mapped coordinates
    meta <- celltrek_result@meta.data
    coord_cols <- intersect(c("coord_x", "coord_y", "id_x", "id_y"), colnames(meta))
    type_col <- intersect(c(opts$celltype_col, "cell_type", "type", "cell_names"), colnames(meta))
    if (length(type_col) > 0 && length(coord_cols) >= 2) {
      coords <- meta[, c(type_col[1], coord_cols[1:2])]
    } else {
      coords <- meta[, intersect(c(opts$celltype_col, colnames(meta)[1:3]), colnames(meta))]
    }
    coords_csv <- file.path(opts$output_dir, "celltrek_mapped_coords.csv")
    write_csv_atomic(coords, coords_csv)
    log_msg("Saved coordinates: ", coords_csv)

    # celltrek_chart() places a cell at up to top_spot (5) positions, each row under a new id
    # ("cell", "cell.1", ...), and places none where the mutual top-N pruning leaves it no pair (see
    # distance_cut_report above: the distance cut removes none). The rows are placements; the cells
    # are the distinct id_raw behind them.
    n_placements <- nrow(coords)
    n_mapped <- if ("id_raw" %in% colnames(meta)) length(unique(as.character(meta$id_raw))) else n_placements
    n_unmapped <- n_sc - n_mapped
    cut_report <- distance_cut_report()
    run_warnings <- c(run_warnings, distance_cut_warning(cut_report))
    # Determine cell type column used for summary
    ct_col_used <- if (length(type_col) > 0) type_col[1] else NULL
    cell_types_mapped <- if (!is.null(ct_col_used) && ct_col_used %in% colnames(meta)) {
      sort(unique(as.character(meta[[ct_col_used]])))
    } else {
      character(0)
    }

    result <- list(
      status = "ok",
      tool = "celltrek",
      task = "spatial_mapping",
      # n_st_spots is the spots CellTrek mapped onto (after the in_tissue filter); n_st_spots_supplied
      # is the spatial object as it came.
      data = list(
        n_sc_cells = n_sc, n_st_spots = n_st, n_st_spots_supplied = n_st_supplied,
        n_shared_genes = n_shared_genes
      ),
      output_files = list(
        result_rds = out_rds,
        mapped_coords_csv = coords_csv
      ),
      params = list(
        celltype_col = opts$celltype_col,
        n_components = opts$n_components,
        reduction = opts$reduction,
        seed = opts$seed,
        seed_scope = paste0(
          "seed drives celltrek()'s interpolated points, random forest and charting; traint()'s ",
          "Seurat steps (CCA, PCA, UMAP) set Seurat's own fixed seed, 42. On more than one ",
          "random-forest thread the placements still differ between runs at one seed (the threaded ",
          "distances differ in the last bit, which reorders equally near spots); RF_CORES=1 in the ",
          "worker's environment makes a run repeat exactly"
        ),
        random_forest_threads = rf_threads,
        method = "CellTrek (traint co-embedding + celltrek random-forest charting)",
        used_fallback = FALSE,
        normalisation = list(spatial = st_norm$how, single_cell = sc_norm$how),
        n_components_available = n_built,
        interpolation_points = CELLTREK_INTERP_POINTS,
        top_spot = CELLTREK_TOP_SPOT,
        spot_n = CELLTREK_SPOT_N,
        # ntree x dist_thresh, the cut celltrek() applies, beside the range of the distances it cuts:
        # removes_pairs is FALSE whenever the cut lies above every distance randomForestSRC returns.
        distance_cut = cut_report
      ),
      summary = list(
        n_mapped_cells = n_mapped,
        n_placements = n_placements,
        n_sc_cells_unmapped = n_unmapped,
        n_sc_cells_renamed = renamed$n_renamed,
        cell_types = cell_types_mapped,
        # Which of the three sources supplied the positions every mapped cell sits on. A carrier we
        # built out of a reduction reads exactly like a real Visium slice unless we say which it was.
        spot_coordinates = coord_source,
        spot_coordinate_axes = coord_axes,
        # CellTrek's own names, which read like x/y and are not: coord_x is the image row.
        mapped_coordinate_columns = list(
          coord_x = "imagerow (image row, vertical)",
          coord_y = "imagecol (image column, horizontal)"
        )
      ),
      analysis = paste0(
        "CellTrek placed ", n_mapped, " of ", n_sc, " single cells at ", n_placements,
        " positions (a cell can be charted to up to ", CELLTREK_TOP_SPOT, " spots or interpolated points). ",
        n_unmapped, " cells were left unplaced by ", unplaced_reason(cut_report), ". ",
        length(cell_types_mapped), " cell types were mapped. ",
        "Spot coordinates: ", coord_source, " (", coord_axes, "). ",
        "Normalisation -- spatial: ", st_norm$how, "; single-cell: ", sc_norm$how, ". ",
        n_shared_genes, " genes are measured by both objects; traint() picks its integration features among them. ",
        "In celltrek_mapped_coords.csv coord_x is the image row and coord_y the image column.",
        if (tissue$n_dropped > 0) {
          paste0(
            " ", tissue$n_dropped, " of the ", n_st_supplied, " spots have in_tissue == 0 (background) and were ",
            "left out; cells were mapped onto the ", n_st, " in-tissue spots."
          )
        } else {
          ""
        },
        if (renamed$n_renamed > 0) {
          paste0(
            " ", renamed$n_renamed, " single-cell names were rewritten with make.names() (the form CellTrek ",
            "keys cells by), so the csv's row names use that form."
          )
        } else {
          ""
        }
      )
    )
    if (!is.null(tissue$filter)) {
      result$params$in_tissue_filter <- tissue$filter
    }
    if (length(run_warnings) > 0) {
      result$warnings <- as.list(run_warnings)
    }
    result
  }), error = function(e) {
    # sink() is process state, not block state: an error raised between sink(stderr()) above
    # and its matching sink() unwinds past the restore, so without this the error JSON below
    # is written to stderr and the portal reports the worker as having emitted nothing at all.
    # try() because the redirect may already have been popped -- popping one that is not there
    # only warns, and that warning must not displace the error being reported.
    try(sink(), silent = TRUE)
    list(
      status = "error",
      tool = "celltrek",
      task = "spatial_mapping",
      error = conditionMessage(e),
      traceback = sog_traceback()
    )
  })

  result
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- parse_args(args)

  if (is.null(opts$sc_h5ad) || is.null(opts$st_h5ad)) {
    result <- list(status = "error", tool = "celltrek",
                   task = "spatial_mapping",
                   error = "Both --sc-h5ad and --st-h5ad are required")
  } else {
    # The checks before traint() stop() too; they must reach the caller as JSON, not as an R abort.
    result <- tryCatch(with_r_traceback(run_celltrek(opts)), error = function(e) {
      # Nothing opens a redirect before these checks, but a stray one must not swallow the payload.
      while (sink.number() > 0) sink()
      list(
        status = "error",
        tool = "celltrek",
        task = "spatial_mapping",
        error = conditionMessage(e),
        traceback = sog_traceback()
      )
    })
  }

  cat(toJSON(result, auto_unbox = TRUE, digits = NA))
  cat("\n")
}

# Sourcing the file (the tests do, into their own environment) defines the functions without
# running a job.
if (identical(environment(), globalenv())) {
  main()
}
