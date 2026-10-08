"""What this toolkit can draw, what each plot needs, and what it cannot draw here and why.

One table, read by four surfaces: ``list_visualization_capabilities`` answers from it,
``recommend_visualizations`` ranks it, ``validate_plot_request`` checks against it, and a
producer's refusal quotes the same sentence it does. Five hand-written lists of "what this system
can do" is a shape this repository has already paid for once; there is one list here and the
others are derived from it.

Three rules the table enforces on itself, each with a lens:

* a capability whose tier is ``unsupported`` must carry a reason, and one that is supported must
  not -- so "we cannot do that" is never a shrug;
* every ``instead`` name must resolve to another capability that is not itself unsupported,
  because a refusal that offers nothing is a dead end;
* every ``kind`` must be a member of the post-analysis manifest's own vocabulary, so a capability
  can never mint a kind the manifest writer will reject at the end of a long render.

Availability is answered by asking the interpreter, never by a table of what was installed on
somebody's laptop. ``find_spec`` is the honest answer and it is re-asked every call, because the
environment a worker runs in is not the environment this module was written in.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

TIER_CORE = "core"
TIER_EXTENDED = "extended"
TIER_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class Capability:
    """One plot family, and everything that decides whether it can be drawn right now."""

    plot_id: str
    function: str
    kind: str
    title: str
    purpose: str
    tier: str = TIER_CORE
    requires: tuple[str, ...] = ()
    prefers: tuple[str, ...] = ()
    packages: tuple[str, ...] = ("matplotlib", "numpy")
    reads_result: str = ""
    scales_to: int = 300_000
    degrade: str = "rasterize"
    instead: tuple[str, ...] = ()
    unsupported_reason: str = ""
    inference_note: str = ""
    claim_level: str = "descriptive"
    axis_semantics: tuple[str, ...] = ()


#: Each predicate is a sentence and a test. The sentence is what a refusal says, so it is written
#: for a reader who wants to know what to do next, not for a developer reading a stack trace.
PREDICATES: dict[str, tuple[str, Callable[[dict[str, Any]], bool]]] = {
    "readable": (
        "the dataset could be opened",
        lambda p: bool(p.get("dataset", {}).get("readable")),
    ),
    "has_expression": (
        "the object has an expression matrix",
        lambda p: bool(p.get("dataset", {}).get("n_vars")),
    ),
    "has_coords": (
        "the object has spatial coordinates in obsm['spatial'] (or an x/y pair in obs)",
        lambda p: bool(p["spatial"]["has_obsm_spatial"] or p["spatial"]["obs_coord_columns"]),
    ),
    "has_3d_coords": (
        "the object carries three coordinate columns -- obsm['spatial_3d_aligned'], another "
        "three-column obsm key, or a two-column obsm['spatial'] with a section axis and a z for it: "
        "a z column (obs['slice_z'], obs['z'], obs['Bregma']) or a recorded or passed z_spacing",
        # The last two clauses are what ``layers.spatial_coords_3d`` builds a z from WITHOUT being
        # told a spacing: a z column by ``layers.z_column`` (Bregma included), or a recorded spacing.
        # Only obs 'slice_z'/'z' counted, so a refusal described the very object it refused (hunt
        # 2026-09-30, u20b-viz-rest-7); numeric section labels counted next, which made every 2D
        # concat (batch "0"/"1") a stack -- they are ids, not positions. A caller's z_spacing
        # satisfies it through ``validate``'s params.
        lambda p: bool(
            p["spatial"].get("coords_3d_keys")
            or (p["spatial"].get("n_dims", 0) >= 3)
            or (p["spatial"].get("section_candidates") and p["spatial"].get("z_unique", 0) > 1)
            or (p["spatial"].get("section_candidates") and p["spatial"].get("section_z_buildable"))
        ),
    ),
    "has_two_coordinate_frames": (
        "the object carries two distinct coordinate frames, so a before and an after both exist",
        # A "before" key AND an "after" key, by the table plot_alignment_qc reads -- not any two
        # frames: spatial + spatial_3d_raw was offered and then refused (hunt 2026-09-30,
        # u20b-viz-rest-8).
        lambda p: _has_before_and_after(p),
    ),
    "has_section_axis": (
        "the observations carry a section or slice column with two or more levels",
        lambda p: bool(p["spatial"].get("section_candidates")),
    ),
    "z_is_physical": (
        "the third coordinate is a physical position with a recorded spacing, not a section index",
        lambda p: bool(
            p["spatial"].get("z_unique", 0) > 1
            and (p["spatial"].get("spatial_3d_provenance", {}).get("z_spacing") or p["spatial"].get("z_span"))
        ),
    ),
    "coords_are_positions": (
        "the coordinates are positions rather than array row and column indices",
        # Required by the histology overlay ONLY, where a scalefactor would mis-scale an index. It
        # was required by every tissue map, so a legacy ST or DBiT lattice could draw none; a plain
        # map is drawn on the lattice with its axes labelled as indices (hunt 2026-09-30,
        # u20b-viz-rest-26, a user decision).
        lambda p: not p["spatial"]["coords_look_like_array_indices"],
    ),
    "has_histology": (
        "an embedded tissue image is present together with the scalefactor that registers it",
        lambda p: bool(p["spatial"]["overlay_ready"]),
    ),
    "has_embedding": (
        "at least one embedding is stored in obsm (X_umap, X_tsne, X_pca or similar)",
        lambda p: any(o["family"] == "embedding" for o in p.get("obsm", [])),
    ),
    "has_categorical_obs": (
        "at least one obs column is categorical with more than one level",
        lambda p: bool(p["obs_roles"]["categorical"]),
    ),
    "has_two_categorical_obs": (
        "two obs columns are categorical, one to group by and one to break each group down by",
        lambda p: len(p["obs_roles"]["categorical"]) >= 2,
    ),
    "has_numeric_obs": (
        "at least one obs column is numeric and not constant",
        lambda p: bool(p["obs_roles"]["numeric"]),
    ),
    "has_cluster_or_cell_type": (
        "an obs column names a cluster, a spatial domain or a cell type",
        lambda p: bool(p["obs_roles"]["cluster"] or p["obs_roles"]["cell_type"]),
    ),
    "has_proportions": (
        "an obsm matrix holds per-observation proportions that sum to one",
        lambda p: any(o.get("family") == "proportions" for o in p.get("obsm", [])),
    ),
    "has_de_effect_and_p": (
        "the stored differential-expression result carries fold changes and adjusted p-values",
        lambda p: _de_has(p, ("logfoldchanges", "pvals_adj")),
    ),
    "has_de_result": (
        "a differential-expression ranking is stored in uns['rank_genes_groups']",
        lambda p: bool(p["uns_analyses"]["rank_genes_groups_detail"]["present"]),
    ),
    "has_neighbors": (
        "a neighbour graph is stored (obsp['connectivities'])",
        lambda p: "connectivities" in p.get("obsp", []),
    ),
    "has_pseudotime": (
        "a pseudotime column is stored in obs",
        lambda p: any("pseudotime" in c.lower() or c.lower() == "dpt" for c in p["obs_roles"]["numeric"]),
    ),
    "has_qc_metrics": (
        "quality-control metrics are stored in obs (total_counts, n_genes_by_counts)",
        lambda p: bool(p["obs_roles"]["qc"]),
    ),
    "counts_available": (
        "a counts matrix is reachable, in X or in a layer",
        lambda p: bool(p["matrix"]["integral"] is True or "counts" in p["matrix"]["layers"]),
    ),
    "not_scaled": (
        "the matrix being drawn is not z-scored (it holds no negative values)",
        lambda p: not p["matrix"]["has_negative"] or "counts" in p["matrix"]["layers"],
    ),
    "single_library": (
        "the object holds one tissue section rather than several",
        lambda p: len(p["spatial"].get("library_ids") or []) <= 1,
    ),
}


def _has_before_and_after(profile: dict[str, Any]) -> bool:
    from spatialomicsgym.viz.profile import frame_pair

    before, after = frame_pair(profile["spatial"].get("coordinate_frames") or [])
    return bool(before and after and before != after)


def _de_has(profile: dict[str, Any], fields: tuple[str, ...]) -> bool:
    detail = profile["uns_analyses"]["rank_genes_groups_detail"]
    if not detail.get("present"):
        return False
    have = set(detail.get("fields") or ())
    return all(f in have for f in fields)


def _cap(**kw: Any) -> Capability:
    return Capability(**kw)


#: The catalogue. Ordered by the question a reader is asking, not by implementation.
CAPABILITIES: dict[str, Capability] = {
    c.plot_id: c
    for c in (
        # ---- quality control and overview -------------------------------------------------
        _cap(
            plot_id="qc.overview",
            function="generate_qc_report",
            kind="grid",
            title="Quality-control overview",
            purpose="Counts, detected genes and mitochondrial fraction, as distributions and, when there are coordinates, on the tissue.",
            requires=("readable", "has_expression"),
            prefers=("has_qc_metrics", "has_coords"),
        ),
        _cap(
            # Declared with generate_qc_report as its drawer, which only ever returns qc.overview, so
            # it was recommended and then drawn as the overview under this id (hunt 2026-09-30,
            # u20b-viz-rest-19). The on-tissue counts ARE part of the overview.
            plot_id="qc.spatial_coverage",
            function="",
            kind="spatial_map",
            title="Tissue coverage and missing data",
            purpose="Where the section has signal and where it does not.",
            tier=TIER_UNSUPPORTED,
            unsupported_reason=(
                "No function draws a separate coverage figure. The quality-control overview paints "
                "the counts and detected genes on the tissue whenever there are coordinates, which "
                "is where the signal is and where it is not."
            ),
            instead=("qc.overview",),
        ),
        # ---- single-cell exploration ------------------------------------------------------
        _cap(
            plot_id="embedding.scatter",
            function="plot_embedding",
            kind="scatter",
            title="Embedding coloured by genes or metadata",
            purpose="UMAP, t-SNE, PCA or any stored embedding, one panel per colour.",
            requires=("readable", "has_embedding"),
            prefers=("has_categorical_obs",),
            axis_semantics=("embedding",),
        ),
        _cap(
            plot_id="embedding.facet",
            function="plot_embedding",
            kind="grid",
            title="Embedding split by a grouping",
            purpose="One panel per level of a categorical column, on shared axes.",
            requires=("readable", "has_embedding", "has_categorical_obs"),
            axis_semantics=("embedding",),
        ),
        _cap(
            plot_id="markers.dotplot",
            function="plot_marker_expression",
            kind="heatmap",
            title="Marker dot plot",
            purpose="Mean expression and fraction expressing, per gene per group.",
            requires=("readable", "has_expression", "has_categorical_obs"),
            prefers=("not_scaled",),
        ),
        _cap(
            plot_id="markers.violin",
            function="plot_marker_expression",
            kind="boxplot",
            title="Expression distribution per group",
            purpose="One violin per group per gene.",
            requires=("readable", "has_expression", "has_categorical_obs"),
        ),
        _cap(
            plot_id="markers.heatmap",
            function="plot_marker_expression",
            kind="heatmap",
            title="Marker heatmap",
            purpose="Genes by observations or by group means.",
            requires=("readable", "has_expression", "has_categorical_obs"),
        ),
        _cap(
            plot_id="composition.stacked",
            function="plot_marker_expression",
            kind="bar",
            title="Cluster composition",
            purpose="What each sample or condition is made of, as proportions.",
            # The producer breaks one categorical column down by a SECOND and refuses without one,
            # so a dataset with only 'leiden' was offered this and then refused (hunt 2026-09-30,
            # u20b-viz-rest-20).
            requires=("readable", "has_two_categorical_obs"),
            claim_level="descriptive",
        ),
        # ---- spatial expression and tissue ------------------------------------------------
        _cap(
            plot_id="spatial.expression",
            function="plot_spatial_expression",
            kind="spatial_map",
            title="Gene expression on the tissue",
            purpose="One or more genes, a numeric column, or one column of a score matrix, painted on the section.",
            requires=("readable", "has_coords", "has_expression"),
            prefers=("has_histology", "counts_available"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="spatial.annotation",
            function="plot_spatial_annotation",
            kind="spatial_map",
            title="Domains and cell types on the tissue",
            purpose="A categorical annotation painted on the section, with a legend.",
            requires=("readable", "has_coords", "has_categorical_obs"),
            prefers=("has_histology", "has_cluster_or_cell_type"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="spatial.histology_overlay",
            function="plot_spatial_expression",
            kind="spatial_map",
            title="Histology underlay",
            purpose="The tissue image under the spots, at a chosen opacity, with the scalefactor applied.",
            requires=("readable", "has_coords", "coords_are_positions", "has_histology"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="spatial.sections",
            # Declared here since the toolkit shipped and implemented by nothing until 2026-09-22:
            # `plot_spatial_expression` panels per GENE, not per section, so a multi-section object
            # drew every section overlaid in one frame. plot_section_grid is the drawer.
            function="plot_section_grid",
            kind="grid",
            title="Several sections side by side",
            purpose="The same value in every section, one panel each, on one shared colour scale.",
            requires=("readable", "has_coords", "has_section_axis"),
            axis_semantics=("spatial",),
        ),
        # ---- three dimensions --------------------------------------------------------------
        _cap(
            plot_id="spatial3d.scatter",
            function="plot_spatial_3d",
            kind="scatter",
            title="The stack in three dimensions",
            purpose="Every cell in the reconstructed volume, from three fixed viewing angles.",
            requires=("readable", "has_3d_coords"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="spatial3d.depth_profile",
            function="plot_spatial_3d",
            kind="line",
            title="What sits at each depth",
            purpose=(
                "Cells per z plane and the mean value per plane, with the spacing drawn to scale. "
                "The figure that makes a broken z visible: coincident planes show as one bar, and "
                "a z built from a slice ordinal shows as perfectly even spacing where the real "
                "sections are not evenly cut."
            ),
            requires=("readable", "has_3d_coords"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="spatial3d.axis_profile",
            function="plot_spatial_3d",
            kind="line",
            title="Expression along an anatomical axis",
            purpose=(
                "A value binned along a named axis. A spatial gradient, not an ordering in time: "
                "the axis is a position, and this figure carries no pseudotime, no root and no "
                "direction of development."
            ),
            requires=("readable", "has_3d_coords"),
            claim_level="descriptive",
            inference_note=(
                "This is a position gradient. It is not a trajectory and must not be reported as "
                "one -- see trajectory.spatial_axis, which is refused for exactly that confusion."
            ),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="alignment.before_after",
            function="plot_alignment_qc",
            kind="grid",
            title="The same sections before and after alignment",
            purpose="One adjacent pair drawn from the pre-alignment frame and the aligned frame, side by side.",
            requires=("readable", "has_coords", "has_two_coordinate_frames", "has_section_axis"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="alignment.pair_overlay",
            function="plot_alignment_qc",
            kind="scatter",
            title="Two adjacent sections in one frame",
            purpose=(
                "One adjacent pair overlaid at full size, in whichever coordinate frame is named. "
                "Needs only one frame, which is what makes it the Phase-1 figure: it answers "
                "whether two adjacent sections sit on top of each other before any alignment has "
                "been run."
            ),
            requires=("readable", "has_coords", "has_section_axis"),
            axis_semantics=("spatial",),
        ),
        # ---- spatial organisation ---------------------------------------------------------
        _cap(
            plot_id="organisation.autocorrelation",
            function="plot_spatial_statistics",
            kind="bar",
            title="Spatially variable genes",
            purpose="Moran's I or Geary's C per gene, from a stored result or a table a spatial tool wrote.",
            requires=("readable", "has_coords"),
            reads_result="autocorr_csv or uns['moranI']",
        ),
        _cap(
            plot_id="organisation.neighborhood",
            function="plot_spatial_statistics",
            kind="heatmap",
            title="Neighbourhood enrichment",
            purpose="Which annotated groups sit next to which, as a z-score from a label permutation.",
            requires=("readable", "has_categorical_obs"),
            reads_result="uns['nhood_enrichment'] written by the spatial-statistics tool",
            claim_level="descriptive",
        ),
        _cap(
            plot_id="organisation.cooccurrence",
            function="plot_spatial_statistics",
            kind="line",
            title="Co-occurrence across distance",
            purpose="How the chance of finding one group near another changes with radius.",
            requires=("readable",),
            reads_result="co_occurrence_csv",
        ),
        _cap(
            plot_id="organisation.graph",
            function="plot_spatial_statistics",
            kind="spatial_map",
            title="Spatial neighbour graph",
            purpose="The edges a spatial statistic was computed over, drawn on the tissue.",
            requires=("readable", "has_coords"),
            axis_semantics=("spatial",),
        ),
        # ---- differential expression and interpretation -----------------------------------
        _cap(
            plot_id="de.volcano",
            function="plot_differential_expression",
            kind="scatter",
            title="Volcano plot",
            purpose="Effect size against significance, for one comparison.",
            requires=("readable", "has_de_effect_and_p"),
            instead=("de.ranked", "markers.dotplot", "markers.heatmap"),
            claim_level="tested",
        ),
        _cap(
            plot_id="de.ranked",
            function="plot_differential_expression",
            kind="line",
            title="Ranked markers",
            purpose="The top genes per group by whatever statistic the stored ranking holds.",
            requires=("readable", "has_de_result"),
        ),
        _cap(
            # plot_differential_expression draws a volcano or a ranked list and nothing else, so this
            # row was validated, recommended and then drawn as one of those under this id (hunt
            # 2026-09-30, u20b-viz-rest-19). The figure it describes is markers.heatmap.
            plot_id="de.heatmap",
            function="",
            kind="heatmap",
            title="Differential-expression heatmap",
            purpose="Top genes per group, as group means.",
            tier=TIER_UNSUPPORTED,
            unsupported_reason=(
                "No function draws a heatmap from the differential-expression call itself. The "
                "marker heatmap draws exactly this -- with no genes named, it takes the top genes of "
                "the stored ranking and shows their mean per group."
            ),
            instead=("markers.heatmap", "de.ranked"),
        ),
        _cap(
            plot_id="pathway.enrichment",
            function="plot_pathway_results",
            kind="scatter",
            title="Enrichment dot plot",
            purpose="Terms by effect and significance, from an enrichment table the pathway tool wrote.",
            requires=("readable",),
            reads_result="enrichment_summary.csv",
            claim_level="tested",
        ),
        _cap(
            plot_id="pathway.activity_map",
            function="plot_pathway_results",
            kind="spatial_map",
            title="Pathway activity on the tissue",
            purpose="A per-observation activity score painted on the section.",
            requires=("readable", "has_coords"),
            reads_result="pathway_activity_scores.csv or an obsm score matrix",
            axis_semantics=("spatial",),
        ),
        # ---- deconvolution ----------------------------------------------------------------
        _cap(
            plot_id="deconv.proportions",
            function="plot_deconvolution",
            kind="grid",
            title="Cell-type proportion maps",
            purpose="One panel per cell type, each spot shaded by its estimated proportion.",
            requires=("readable", "has_coords", "has_proportions"),
            reads_result="a proportions matrix in obsm, or a proportions CSV",
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="deconv.dominant",
            function="plot_deconvolution",
            kind="spatial_map",
            title="Dominant cell type per spot",
            purpose="The highest-proportion type at each spot, as a categorical map.",
            requires=("readable", "has_coords", "has_proportions"),
            reads_result="a proportions matrix in obsm, or a proportions CSV",
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="deconv.composition",
            function="plot_deconvolution",
            kind="bar",
            title="Mixture summary",
            purpose="Overall composition, and composition per domain when one is annotated.",
            requires=("readable", "has_proportions"),
            reads_result="a proportions matrix in obsm, or a proportions CSV",
        ),
        # ---- communication ----------------------------------------------------------------
        _cap(
            plot_id="communication.interactions",
            function="plot_cell_communication",
            kind="heatmap",
            title="Ligand-receptor interactions",
            purpose="Ranked interactions between sender and receiver groups, from a communication tool's own output.",
            requires=("readable",),
            reads_result="the output table of a cell-communication tool",
            inference_note=(
                "Inferred from co-expression of annotated ligand-receptor pairs. This is a "
                "hypothesis about signalling, not a measurement of it."
            ),
        ),
        # ---- trajectory --------------------------------------------------------------------
        _cap(
            plot_id="trajectory.pseudotime",
            function="plot_trajectory",
            kind="scatter",
            title="Pseudotime on an embedding",
            purpose="A stored pseudotime, coloured on the embedding it was computed in.",
            requires=("readable", "has_embedding", "has_pseudotime"),
            axis_semantics=("embedding", "pseudotime"),
        ),
        _cap(
            plot_id="trajectory.spatial_pseudotime",
            function="plot_trajectory",
            kind="spatial_map",
            title="Pseudotime on the tissue",
            purpose="A stored pseudotime painted on the section. Ordering, not elapsed time.",
            requires=("readable", "has_coords", "has_pseudotime"),
            axis_semantics=("spatial", "pseudotime"),
        ),
        _cap(
            plot_id="trajectory.gene_trend",
            function="plot_trajectory",
            kind="line",
            title="Gene expression along a trajectory",
            purpose="Smoothed expression against pseudotime, for chosen genes.",
            requires=("readable", "has_pseudotime", "has_expression"),
            axis_semantics=("pseudotime",),
        ),
        # ---- extended: needs the heavier environment ----------------------------------------
        _cap(
            plot_id="spatial.segmentation",
            function="plot_spatial_segmentation",
            kind="spatial_map",
            title="Cell boundaries on the tissue",
            purpose="Segmentation outlines over the image.",
            tier=TIER_EXTENDED,
            requires=("readable", "has_coords"),
            packages=("squidpy", "skimage"),
            instead=("spatial.annotation",),
        ),
        _cap(
            plot_id="spatial.scene",
            function="plot_spatialdata_scene",
            kind="spatial_map",
            title="SpatialData scene",
            purpose="Images, labels, shapes and points from a SpatialData store, rendered together.",
            tier=TIER_EXTENDED,
            requires=("readable",),
            packages=("spatialdata", "spatialdata_plot"),
            instead=("spatial.expression",),
        ),
        _cap(
            plot_id="spatial.molecules",
            function="plot_spatialdata_scene",
            kind="spatial_map",
            title="Transcript-level molecule map",
            purpose="Individual molecule positions, rasterised.",
            tier=TIER_EXTENDED,
            requires=("readable",),
            packages=("datashader",),
            scales_to=50_000_000,
            degrade="rasterize",
            instead=("spatial.expression",),
        ),
        # ---- unsupported, each with the reason it is unsupported ----------------------------
        _cap(
            plot_id="velocity.stream",
            function="",
            kind="scatter",
            title="RNA velocity",
            purpose="Velocity streamlines or arrows on an embedding.",
            tier=TIER_UNSUPPORTED,
            packages=("scvelo",),
            unsupported_reason=(
                "scvelo is not installed in any environment here, and velocity also needs spliced "
                "and unspliced layers that these datasets do not carry."
            ),
            instead=("trajectory.pseudotime",),
        ),
        _cap(
            plot_id="fate.probabilities",
            function="",
            kind="scatter",
            title="Fate probabilities",
            purpose="Absorption probabilities towards terminal states.",
            tier=TIER_UNSUPPORTED,
            packages=("cellrank",),
            unsupported_reason="cellrank is not installed in any environment here.",
            instead=("trajectory.pseudotime",),
        ),
        _cap(
            plot_id="de.pseudobulk_test",
            function="",
            kind="scatter",
            title="Pseudobulk differential expression",
            purpose="A sample-level negative-binomial test.",
            tier=TIER_UNSUPPORTED,
            packages=("pydeseq2",),
            unsupported_reason=(
                "pydeseq2 is not installed here. Pseudobulk aggregation and its summary plots are "
                "available; the test itself is not, and a weaker test is not substituted for it."
            ),
            instead=("de.volcano", "composition.stacked"),
        ),
        _cap(
            plot_id="organisation.hotspot",
            function="",
            kind="spatial_map",
            title="Local hotspot map (Getis-Ord, LISA)",
            purpose="Local spatial association statistics.",
            tier=TIER_UNSUPPORTED,
            packages=("esda", "libpysal"),
            unsupported_reason=(
                "esda and libpysal are not installed here, and a smoothed global Moran's I is a "
                "different statistic that must not be presented as a hotspot map."
            ),
            instead=("organisation.autocorrelation",),
        ),
        _cap(
            plot_id="interactive.any",
            function="",
            kind="scatter",
            title="Client-side interactive figure",
            purpose="Hover, lasso and live re-scaling in the browser.",
            tier=TIER_UNSUPPORTED,
            packages=("plotly",),
            # Rewritten 2026-09-22, when the sentence it used to carry stopped being true. That
            # note was the first sentence of the reason itself, which validate() hands the model
            # and the user verbatim (hunt 2026-09-30, u20b-viz-rest-31).
            unsupported_reason=(
                "A tool writes no markup and no script: the portal renders ONE shape, the "
                "3D volume, from a data-only spec it rebuilds from a key allowlist "
                "(interactive.volume). Everything else -- a hoverable heatmap, a lasso-selectable "
                "embedding, a Bokeh document -- would mean a tool handing the browser code to run, "
                "and that is what stays refused. Zoom and region selection on those are figure "
                "parameters that re-render server side, and every figure ships the table of values "
                "behind it."
            ),
            instead=("interactive.volume", "spatial.expression", "embedding.scatter"),
        ),
        _cap(
            plot_id="interactive.volume",
            function="plot_spatial_3d",
            kind="scatter",
            title="The stack, rotatable",
            purpose=(
                "The reconstructed volume as a spec the portal renders in a sandboxed frame, so a "
                "reader can turn it. A static PNG of the same points is written beside it and is "
                "what the manifest declares -- the interactive view is an addition, never the only "
                "copy, because a frame that fails to load must not take the figure with it."
            ),
            requires=("readable", "has_3d_coords"),
            axis_semantics=("spatial",),
        ),
        _cap(
            plot_id="trajectory.spatial_axis",
            function="",
            kind="spatial_map",
            title="Spatial trajectory",
            purpose="A developmental ordering read off tissue position.",
            tier=TIER_UNSUPPORTED,
            unsupported_reason=(
                "Refused as a claim rather than for a missing package: spatial proximity is not "
                "temporal order, and no figure here encodes one as the other. A distance-to-"
                "boundary expression trend is the honest version of the question."
            ),
            instead=("spatial.expression", "trajectory.gene_trend"),
        ),
    )
}


def kinds() -> frozenset[str]:
    """Every figure kind this catalogue declares."""
    return frozenset(c.kind for c in CAPABILITIES.values())


def package_availability(names: tuple[str, ...] | list[str]) -> dict[str, bool]:
    """Which of these packages this interpreter can import. Asked, never tabulated."""
    out: dict[str, bool] = {}
    for name in names:
        try:
            out[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            out[name] = False
    return out


def registered_functions() -> set[str]:
    """The tool names a portal actually exposes, read from the shipped config at call time.

    This is what stops the catalogue from advertising a plot nobody wrote. A ``Capability`` row is
    a *claim* about what could be drawn; whether a drawer exists for it is a separate fact, and the
    two drift apart in one direction only -- a row is added while its producer is still being
    built. Without this, ``recommend()`` happily returns a plot whose ``function`` is registered on
    no portal, the model calls it, and the call fails with "unknown tool" on a dataset that was
    perfectly capable of the plot. The catalogue would have been the thing that lied.

    Never raises, and an unreadable config returns an EMPTY set, which :func:`evaluate` treats as
    "cannot tell" rather than "nothing is implemented" -- refusing every plot because a YAML file
    moved would be a worse failure than the one this guards against.
    """
    try:
        import yaml

        from spatialomicsgym.mcp_config_path import find_mcp_config

        path = find_mcp_config()
        if path is None:
            return set()
        with open(path) as fh:
            config = yaml.safe_load(fh) or {}
        servers = config.get("mcp_servers")
        if not isinstance(servers, dict):
            return set()
        names: set[str] = set()
        for block in servers.values():
            if not isinstance(block, dict) or not block.get("enabled", True):
                continue
            for entry in block.get("tools") or []:
                if isinstance(entry, dict) and entry.get("spatialomicsgym_name"):
                    names.add(entry["spatialomicsgym_name"])
        return names
    except Exception:
        return set()


def evaluate(profile: dict[str, Any], *, include_unsupported: bool = True) -> list[dict[str, Any]]:
    """Every capability, against one dataset, with the reason for each refusal.

    The ``why`` a caller sees is the predicate's own sentence, so the recommendation, the
    validation and the producer's error cannot word the same failure three different ways.
    """
    rows: list[dict[str, Any]] = []
    # Empty means "could not read the config", not "nothing is registered" -- see the helper.
    registered = registered_functions()
    for cap in CAPABILITIES.values():
        if cap.tier == TIER_UNSUPPORTED and not include_unsupported:
            continue
        row: dict[str, Any] = {
            "plot_id": cap.plot_id,
            "function": cap.function,
            "implemented": (cap.function in registered) if registered else True,
            "title": cap.title,
            "purpose": cap.purpose,
            "kind": cap.kind,
            "tier": cap.tier,
            "reads_result": cap.reads_result,
            "instead": list(cap.instead),
        }
        if cap.tier == TIER_UNSUPPORTED:
            row.update({"available": False, "blocked_by": ["unsupported"], "why": cap.unsupported_reason})
            rows.append(row)
            continue

        missing_packages = [n for n, ok in package_availability(cap.packages).items() if not ok]
        if missing_packages:
            row.update(
                {
                    "available": False,
                    "blocked_by": ["packages"],
                    "missing_packages": missing_packages,
                    # There is no "extended portal": these families are declared and refused here
                    # (D-061), and the sentence sent the model looking for a tool that does not
                    # exist (hunt 2026-09-30, u20b-viz-rest-25).
                    "why": (
                        f"{', '.join(missing_packages)} cannot be imported in this environment"
                        + (
                            ", and no function in this installation draws this plot."
                            if cap.tier == TIER_EXTENDED
                            else "."
                        )
                    ),
                }
            )
            rows.append(row)
            continue

        if _drawer_exists(cap.function) is False:
            # The extended rows name functions nothing defines; with their packages importable they
            # read as available and were counted as drawable (hunt 2026-09-30, u20b-viz-rest-25).
            row.update(
                {
                    "available": False,
                    "blocked_by": ["not_implemented"],
                    "why": f"no function in this installation draws this plot; {cap.function!r} is not implemented.",
                }
            )
            rows.append(row)
            continue
        failed = [key for key in cap.requires if not _passes(key, profile)]
        row["available"] = not failed
        row["blocked_by"] = failed
        row["why"] = "" if not failed else "; ".join(PREDICATES[k][0] for k in failed if k in PREDICATES)
        # A plot that reads another tool's output is callable, but only once that output exists.
        # Reporting it as plainly "available" over-promises: nothing in the dataset says whether
        # a deconvolution has been run. The row says what it needs and the recommender ranks it
        # below plots the dataset can answer on its own.
        row["needs_result"] = bool(cap.reads_result)
        row["prefers_met"] = [k for k in cap.prefers if _passes(k, profile)]
        row["prefers_unmet"] = [k for k in cap.prefers if not _passes(k, profile)]
        n_obs = int(profile.get("dataset", {}).get("n_obs") or 0)
        if n_obs > cap.scales_to:
            row["degrade"] = cap.degrade
            row["degrade_note"] = (
                f"{n_obs:,} observations is above this plot's comfortable size; it will "
                f"{cap.degrade} and the figure will say so."
            )
        rows.append(row)
    return rows


def _passes(key: str, profile: dict[str, Any]) -> bool:
    entry = PREDICATES.get(key)
    if entry is None:
        return False
    try:
        return bool(entry[1](profile))
    except Exception:
        return False


def recommend(profile: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """A short, ordered list of what is worth drawing for this dataset.

    Ordered by how much of the dataset's own structure a plot uses: a plot whose preferred
    conditions are all met before one that merely meets its requirements, and a spatial plot
    before a single-cell one when there are coordinates. Deliberately short -- a hundred
    suggestions is not a recommendation.

    A row whose producer is not registered on any portal is never recommended, however capable the
    dataset is of it. Recommending one would hand the model a tool name that does not resolve, and
    the catalogue -- the surface whose whole job is to say what can be drawn -- would be the thing
    that lied. It stays visible in :func:`list_visualization_capabilities`, which is a catalogue and
    says so; this is the ordered short list of what to actually call.
    """
    spatial = bool(profile.get("spatial", {}).get("has_obsm_spatial"))
    rows = [
        r for r in evaluate(profile, include_unsupported=False) if r.get("available") and r.get("implemented", True)
    ]

    def score(row: dict[str, Any]) -> tuple[int, int, int, str]:
        cap = CAPABILITIES[row["plot_id"]]
        is_spatial = "spatial" in cap.axis_semantics
        return (
            1 if row.get("needs_result") else 0,  # what the dataset can answer alone, first
            0 if (is_spatial == spatial) else 1,  # then what suits this modality
            -(len(row.get("prefers_met", []))),  # then what uses most of its structure
            cap.plot_id,
        )

    rows.sort(key=score)
    # ``rows[:0]`` made the inspector say "Nothing can be drawn from this dataset" about a capable
    # one, and ``rows[:-2]`` returned nearly everything (hunt 2026-09-30, u20b-viz-rest-29).
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 8
    return rows[: max(1, limit)]


#: A caller's selector that, when it names something the dataset holds, IS the prerequisite. The
#: gate was parameter-blind: the profile's fixed column list vetoed ``section_key='sample_id'``, a
#: name-based pseudotime test vetoed ``pseudotime_key='latent_time'``, and an obsm-only proportions
#: test vetoed the ``proportions_csv`` a deconvolution tool wrote -- each before the producer that
#: would have honoured it (hunt 2026-09-30, u20b-viz-rest-4, -7, -16).
def _caller_supplies(key: str, params: dict[str, Any], profile: dict[str, Any]) -> bool:
    try:
        columns = _obs_index(profile)
        obsm = _obsm_shapes(profile)
        if key == "has_proportions":
            if str(params.get("proportions_csv") or "").strip():
                return True
            named = str(params.get("obsm_key") or "").partition(":")[0]
            return bool(named) and len(obsm.get(named) or ()) == 2 and obsm[named][1] >= 2
        if key == "has_section_axis":
            entry = columns.get(str(params.get("section_key") or ""))
            return bool(entry) and 2 <= int(entry.get("n_levels") or 0) <= 200
        if key == "has_pseudotime":
            entry = columns.get(str(params.get("pseudotime_key") or ""))
            return bool(entry) and entry.get("draws_as") == "numeric"
        if key in ("has_categorical_obs", "has_cluster_or_cell_type"):
            for name in (params.get("obs_key"), params.get("groupby")):
                entry = columns.get(str(name or ""))
                if entry and entry.get("draws_as") == "categorical" and int(entry.get("n_levels") or 0) > 1:
                    return True
            return False
        if key == "has_3d_coords":
            shape = obsm.get(str(params.get("coords_key") or ""))
            if shape and len(shape) == 2 and shape[1] >= 3:
                return True
            try:
                spacing = float(params.get("z_spacing") or 0.0)
            except (TypeError, ValueError):
                spacing = 0.0
            has_axis = bool(profile["spatial"].get("section_candidates")) or _caller_supplies(
                "has_section_axis", params, profile
            )
            return spacing > 0 and has_axis
        if key == "has_two_coordinate_frames":
            from spatialomicsgym.viz.profile import frame_pair

            default_before, default_after = frame_pair(list(obsm))
            before = str(params.get("before_key") or "") or default_before
            after = str(params.get("after_key") or "") or default_after
            return bool(before in obsm and after in obsm and before != after)
    except Exception:
        return False
    return False


def supplied_by(plot_id: str, params: dict[str, Any], profile: dict[str, Any]) -> tuple[str, ...]:
    """The prerequisites of *plot_id* the caller's own selectors supply, for a producer's gate.

    A producer validates with no parameters, so ``plot_section_grid(section_key='sample_id')`` and
    ``plot_trajectory(pseudotime_key='latent_time')`` were refused by a profile that only knows its
    fixed names (hunt 2026-09-30, u20b-viz-rest-16, -7). Only prerequisites are answered here; the
    parameter checks :func:`validate` adds for validate_plot_request are not, because a producer
    reports a bad selector in its own words.
    """
    cap = CAPABILITIES.get(plot_id)
    if cap is None or not isinstance(params, dict):
        return ()
    chosen = {k: v for k, v in params.items() if v not in (None, "", 0, 0.0)}
    if not chosen:
        return ()
    return tuple(k for k in cap.requires if _caller_supplies(k, chosen, profile))


def _obs_index(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every obs column's role and level count -- ``profile['obs_index']``, or what ``obs`` lists."""
    index = profile.get("obs_index")
    if isinstance(index, dict):
        return index
    return {
        str(c.get("name")): {
            "role": c.get("role"),
            "draws_as": c.get("draws_as", c.get("role")),
            "n_levels": c.get("n_levels", c.get("n_distinct", 0)),
        }
        for c in profile.get("obs") or []
    }


def _obsm_shapes(profile: dict[str, Any]) -> dict[str, list[int]]:
    return {str(o.get("key")): list(o.get("shape") or []) for o in profile.get("obsm") or []}


#: Parameters whose value names something in the dataset, and where to look for it.
_OBS_SELECTORS = ("obs_key", "groupby", "split_by", "section_key", "pseudotime_key")
_OBSM_SELECTORS = ("obsm_key", "coords_key", "before_key", "after_key")


def _check_params(cap: Capability, params: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    """What in *params* cannot work against this dataset, as sentences. Empty when nothing is wrong.

    validate_plot_request parsed params_json and never read it, so a bogus gene, layer or key -- or a
    parameter the function does not take -- came back ``ok: true`` and the producer refused it a
    round trip later (hunt 2026-09-30, u20b-viz-rest-15). Only what the profile can answer is
    checked; a gene is checked only when the profile was asked about it.
    """
    problems: list[str] = []
    accepted = _accepted_params(cap.function)
    if accepted:
        unknown = sorted(k for k in params if k not in accepted)
        if unknown:
            problems.append(f"{cap.function} takes no parameter named {', '.join(map(repr, unknown))}")
    columns = _obs_index(profile)
    for name in _OBS_SELECTORS:
        value = str(params.get(name) or "")
        if value and columns and value not in columns:
            problems.append(f"{name}={value!r} is not an obs column")
    obsm = _obsm_shapes(profile)
    for name in _OBSM_SELECTORS:
        value = str(params.get(name) or "").partition(":")[0]
        if value and value not in obsm:
            problems.append(f"{name}={value!r} is not an obsm key; this object has {sorted(obsm)}")
    layer = str(params.get("layer") or "")
    layers = list((profile.get("matrix") or {}).get("layers") or [])
    if layer and layer not in layers:
        problems.append(f"layer={layer!r} is not a layer; this object has {layers or 'none'}")
    present = (profile.get("var") or {}).get("requested_present") or {}
    absent = [g for g, ok in present.items() if not ok]
    if absent:
        problems.append(f"{', '.join(absent)} {'is' if len(absent) == 1 else 'are'} not a gene in this object")
    return problems


def _drawer_exists(function: str) -> bool | None:
    """Whether the producer is defined. ``None`` when the drawing module cannot be imported here."""
    if not function:
        return False
    try:
        from spatialomicsgym.viz import pipelines
    except Exception:
        return None
    return callable(getattr(pipelines, function, None))


def _accepted_params(function: str) -> set[str]:
    """The parameter names the producer takes. Empty when that cannot be read, which checks nothing."""
    if not function:
        return set()
    try:
        import inspect

        from spatialomicsgym.viz import pipelines

        return set(inspect.signature(getattr(pipelines, function)).parameters)
    except Exception:
        return set()


def _communication_table_refusal(plot_id: str, params: dict[str, Any], cap: Capability) -> dict[str, Any] | None:
    """A communication request whose ``results_path`` is no communication table, refused from its header.

    ``requires`` holds only ``readable`` -- the dataset says nothing about whether a communication tool ran -- so a
    differential-expression table passed here came back ok and was refused a round trip later. The header line
    alone, read with the standard library and told by the explorer's own detector. A file that cannot be read is
    left to the producer, which names that refusal itself.
    """
    path = str(params.get("results_path") or "") if plot_id == "communication.interactions" else ""
    if not path:
        return None
    import csv
    from pathlib import Path

    from .ccc import detect

    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            line = fh.readline()
    except OSError:
        return None
    sep = max(("\t", ",", ";"), key=line.count) if any(d in line for d in ("\t", ",", ";")) else ","
    header = next(csv.reader([line.rstrip("\r\n")], delimiter=sep), [])
    if detect.table_columns(header, Path(path).name) is not None:
        return None
    return {
        "ok": False,
        "error_kind": "prerequisite_not_met",
        "why": "the results table is not a cell-communication result: it names no sender, receiver and score, "
        "and is no sender-by-receiver matrix",
        "detail": f"{Path(path).name} has the columns {header}",
        "missing": ["has_communication_table"],
        "instead": list(cap.instead),
    }


def validate(plot_id: str, params: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    """Can this exact request be drawn? A refusal names what is missing and what to do instead.

    ``params`` are the call's own arguments. A selector that names something the dataset holds
    satisfies the prerequisite it selects for, and a selector that names nothing is refused here
    rather than a round trip later.
    """
    cap = CAPABILITIES.get(plot_id)
    if cap is None:
        # The alphabetically first eight, which were unrelated to the request and included a
        # refused one (hunt 2026-09-30, u20b-viz-rest-33). The closest drawable ids instead.
        import difflib

        drawable = sorted(
            k for k, c in CAPABILITIES.items() if c.tier != TIER_UNSUPPORTED and _drawer_exists(c.function) is not False
        )
        return {
            "ok": False,
            # Named, so the inspector's reply reads "'spatial.expresion' cannot be drawn" and not
            # "None cannot be drawn".
            "plot_id": plot_id,
            "error_kind": "unknown_plot",
            "why": (f"{plot_id!r} is not a plot this toolkit draws. list_visualization_capabilities names every one."),
            "instead": difflib.get_close_matches(str(plot_id), drawable, n=5, cutoff=0.5),
        }
    if not isinstance(params, dict):
        return {
            "ok": False,
            "error_kind": "invalid_request",
            "why": f"the parameters must be a JSON object of name to value, not a {type(params).__name__}",
            "instead": [],
        }
    if cap.tier == TIER_UNSUPPORTED:
        return {
            "ok": False,
            "error_kind": "capability_unsupported",
            "why": cap.unsupported_reason,
            "instead": list(cap.instead),
        }
    missing_packages = [n for n, ok in package_availability(cap.packages).items() if not ok]
    if missing_packages:
        return {
            "ok": False,
            "error_kind": "dependency_unavailable",
            "why": f"{', '.join(missing_packages)} cannot be imported in this environment.",
            "missing_packages": missing_packages,
            "instead": list(cap.instead),
        }
    # A row is a claim; whether a drawer exists is a separate fact. The extended rows name functions
    # nothing defines, and once their packages import this said ok and sent the model to a tool
    # name that does not resolve (hunt 2026-09-30, u20b-viz-rest-25). Asked of the code rather than
    # of the portal config, so a direct call is not refused because a YAML file moved.
    if _drawer_exists(cap.function) is False:
        return {
            "ok": False,
            "error_kind": "capability_unsupported",
            "why": f"no function in this installation draws {plot_id}; {cap.function!r} is not implemented.",
            "instead": list(cap.instead),
        }
    failed = [k for k in cap.requires if not _passes(k, profile) and not _caller_supplies(k, params, profile)]
    if failed:
        return {
            "ok": False,
            "error_kind": "prerequisite_not_met",
            "why": "; ".join(PREDICATES[k][0] for k in failed if k in PREDICATES),
            "missing": failed,
            "instead": list(cap.instead),
        }
    table = _communication_table_refusal(plot_id, params, cap)
    if table:
        return table
    problems = _check_params(cap, params, profile) if params else []
    if problems:
        return {
            "ok": False,
            "error_kind": "invalid_request",
            "why": "; ".join(problems),
            "instead": [],
        }
    return {"ok": True, "plot_id": plot_id, "function": cap.function, "kind": cap.kind}
