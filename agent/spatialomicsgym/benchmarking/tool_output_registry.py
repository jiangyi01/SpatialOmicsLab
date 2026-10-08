"""Tool output format registry — exact output specifications per MCP tool.

Each entry describes:
  - what files the tool produces
  - which file is the authoritative prediction output
  - where prediction data is stored (column name, obs key, etc.)
  - what format the prediction is in
  - what task type it belongs to

Built from direct inspection of worker source code (tools/*_worker.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OutputSpec:
    """Specification for a single output file.

    `filename_pattern` and `prediction_key` both accept either a single string
    or a list of alternatives. Lists are useful when:
      - a worker has shipped multiple historical filenames
      - the output filename is parameter-dependent
      - the prediction key (obs/obsm/CSV column) varies by tool version,
        configuration, or upstream library version
    The inspector tries every entry in order. See:
      - `output_inspector._match_files` (filename matching)
      - `output_inspector._resolve_prediction_key` (key resolution)
    """

    filename_pattern: str | list[str]  # glob, or list of glob alternatives
    format: str  # h5ad, csv, tsv, json, png, etc.
    role: str  # "prediction", "metadata", "visualization", "intermediate"
    description: str = ""
    prediction_key: str | list[str] = ""  # obs column / obsm key / CSV column, or list of alternatives
    prediction_type: str = ""  # "cluster_labels", "gene_list", "proportions_matrix", etc.
    schema: dict[str, str] = field(default_factory=dict)  # column/key descriptions
    # Label values that mean "this spot was assigned to no cluster" -- dropped like NaN by the
    # inspector, the standardizer and the post-execution gate's K-mismatch check, so they are never
    # counted or scored as a cluster (hunt 2026-09-30, u31-benchmarking-11).
    unassigned_labels: list[str] = field(default_factory=list)
    # Patterns, among ``filename_pattern``, naming a table that is already the tool's own selection --
    # its top-k or its called set -- rather than every gene it tested. A score-only SVG table of this
    # kind keeps every gene with a non-zero score: the top-N cut is for a full score table, and it had
    # halved PROST's own top 200 (hunt 2026-09-30, u31-benchmarking-4).
    selection_patterns: list[str] = field(default_factory=list)


@dataclass
class ToolOutputProfile:
    """Complete output profile for an MCP tool."""

    tool_name: str
    task_type: str  # spatial_clustering, svg_detection, deconvolution
    outputs: list[OutputSpec]
    # Filename and key now both accept str | list[str]. Same semantics as
    # OutputSpec — list entries are tried in order.
    authoritative_output: str | list[str] = ""
    prediction_key: str | list[str] = ""
    prediction_type: str = ""  # what kind of prediction
    notes: str = ""


VIZ_NOTE = (
    "A visualization run's authoritative artefact is a PICTURE, which is why this profile exists. "
    "Without one the engine types a result by CONTENT, and every file this tool writes is bait: "
    "the .figdata.npz cache holds the plotted numbers, and a proportion or marker table written "
    "beside a figure reads as a deconvolution or a gene list. The recorded consequence of a "
    "missing profile is exact -- an enrichment table read as a deconvolution, and the tool's own "
    "figure never reaching the run card. task_type 'visualization' is deliberately outside the "
    "review contract's own task-type vocabulary, the same position functional_enrichment "
    "occupies: the run is recorded as partial with one warning, and nothing re-derives or "
    "second-guesses a figure the tool already drew. There is no authoritative_output, no "
    "prediction_key and no prediction_type, because this tool predicts nothing -- it draws what "
    "another tool predicted."
)


DIAGNOSIS_NOTE = (
    "A diagnosis predicts nothing. It measures a stack that already exists and returns a class "
    "letter with the numbers behind it, so there is no authoritative_output, no prediction_key "
    "and no prediction_type -- the same position the visualization profiles occupy and for the "
    "same reason. The profile exists because without one the engine types the run by CONTENT, and "
    "adjacent_pair_metrics.csv is bait: a table of per-pair numbers written beside a JSON report "
    "reads as a clustering or a deconvolution depending on which column the heuristics reach "
    "first. task_type 'alignment' is the review contract's own word for this family, so the run "
    "is routed to the alignment runner rather than re-derived."
)

_SPATIAL3D_DIAGNOSIS = {
    "paste2_partial_align": ToolOutputProfile(
        tool_name="paste2_partial_align",
        task_type="alignment",
        outputs=[
            OutputSpec(
                "paste2_partial_aligned_slice_*.h5ad",
                "h5ad",
                "alignment",
                "one per section: obsm['spatial'] as given, plus the aligned coordinates under "
                "obsm['spatial_3d_aligned'] or obsm['spatial_aligned']",
            ),
            OutputSpec(
                "overlap_fractions.csv",
                "csv",
                "alignment",
                "the overlap fraction used for each adjacent pair, and whether it was given or estimated",
            ),
            OutputSpec("paste2_pi_*.npy", "npy", "alignment", "the partial transport coupling per pair"),
        ],
        notes=(
            "The overlap fraction is the parameter that distinguishes this from PASTE, so it is an "
            "output as much as an input: a pair aligned at 0.9 and a pair aligned at 0.3 are "
            "different claims about the tissue. No prediction_key -- an alignment predicts nothing, "
            "it moves what was already measured."
        ),
    ),
    "cast_align_slices": ToolOutputProfile(
        tool_name="cast_align_slices",
        task_type="alignment",
        outputs=[
            OutputSpec(
                "cast_aligned_slice_*.h5ad",
                "h5ad",
                "alignment",
                "one per section: obsm['spatial'] as given, plus the registered coordinates",
            ),
            OutputSpec(
                "cast_displacement.csv",
                "csv",
                "alignment",
                "median displacement of each query section onto the reference",
            ),
        ],
        notes=(
            "cast_intermediate/ holds the embedding, loss log and trained model under filenames "
            "CAST hardcodes (demo_embed_dict.pt and friends). They are scratch, not the answer, "
            "which is why they are not listed here."
        ),
    ),
    "diagnose_3d_stack": ToolOutputProfile(
        tool_name="diagnose_3d_stack",
        task_type="alignment",
        outputs=[
            OutputSpec(
                "alignment_diagnosis.json",
                "json",
                "alignment",
                "the A/B/C/unknown verdict, the criteria behind it, and the questions it could not answer",
            ),
            OutputSpec(
                "adjacent_pair_metrics.csv",
                "csv",
                "alignment",
                "one row per adjacent section pair: centroid offset, overlap, local-shift dispersion, "
                "the fitted transform, and where expression was read the agreement against its null",
            ),
        ],
        notes=DIAGNOSIS_NOTE,
    ),
}

# ═══════════════════════════════════════════════════════════════════
# SPATIAL CLUSTERING TOOLS
# ═══════════════════════════════════════════════════════════════════

TOOL_PROFILES: dict[str, ToolOutputProfile] = {
    **_SPATIAL3D_DIAGNOSIS,
    "run_scanpy_spatial_domain": ToolOutputProfile(
        tool_name="run_scanpy_spatial_domain",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "*spatial_domains.h5ad",
                "h5ad",
                "prediction",
                "Annotated AnnData with cluster labels in obs['spatial_domain']",
                prediction_key="spatial_domain",
                prediction_type="cluster_labels",
            ),
            OutputSpec("*domains_summary.csv", "csv", "metadata", "Summary table: spatial_domain, n_spots"),
            OutputSpec("*spatial_spatial_domains.png", "png", "visualization"),
            OutputSpec("*umap_spatial_domains.png", "png", "visualization"),
        ],
        # The glob is two generic words, so it scores below the floor and every domain tool in the
        # registry writes a file matching it -- that demotion is deliberate. The exact name the
        # worker writes (scanpy_spatial_worker.py, adata_out_path) carries "scanpy" and does name this tool.
        authoritative_output=["scanpy_spatial_domains.h5ad", "*spatial_domains.h5ad"],
        prediction_key="spatial_domain",
        prediction_type="cluster_labels",
    ),
    "graphst_spatial_clustering": ToolOutputProfile(
        tool_name="graphst_spatial_clustering",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "graphst_clustering_output.h5ad",
                "h5ad",
                "prediction",
                "AnnData with obs['domain'] (from GraphST clustering)",
                prediction_key="domain",
                prediction_type="cluster_labels",
            ),
            OutputSpec("graphst_domain.csv", "csv", "metadata", "CSV index=barcode, column='domain'"),
        ],
        authoritative_output="graphst_clustering_output.h5ad",
        prediction_key="domain",
        prediction_type="cluster_labels",
        notes="Key is 'domain' (not 'graphst_cluster'). domain_csv may be null if column missing.",
    ),
    "stagate_spatial_domains": ToolOutputProfile(
        tool_name="stagate_spatial_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "stagate_domains.h5ad",
                "h5ad",
                "prediction",
                "AnnData with obs['stagate_domain'] + obsm['STAGATE'] embedding",
                prediction_key="stagate_domain",
                prediction_type="cluster_labels",
            ),
            OutputSpec("stagate_domain_assignments.csv", "csv", "metadata"),
            OutputSpec("stagate_embedding.npy", "npy", "intermediate"),
        ],
        authoritative_output="stagate_domains.h5ad",
        prediction_key="stagate_domain",
        prediction_type="cluster_labels",
        notes="Uses KMeans on STAGATE embedding. Column is 'stagate_domain' (categorical).",
    ),
    "deepst_identify_domains": ToolOutputProfile(
        tool_name="deepst_identify_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "deepst_clustering.h5ad",
                "h5ad",
                "prediction",
                "AnnData with spatially refined domain labels in obs['DeepST_refine_domain'] "
                "(obs['DeepST_domain'] holds the Leiden labels before refinement)",
                prediction_key="DeepST_refine_domain",
                prediction_type="cluster_labels",
            ),
            OutputSpec("deepst_domains_per_spot.csv", "csv", "metadata"),
            OutputSpec("deepst_spatial_domains.png", "png", "visualization"),
        ],
        authoritative_output="deepst_clustering.h5ad",
        prediction_key="DeepST_refine_domain",
        prediction_type="cluster_labels",
        notes=(
            "Worker publishes obs['DeepST_refine_domain'] only (no fallback to DeepST_domain/domain/louvain; "
            "a missing refined column is an error). params.resolution_used / params.used_fallback say which "
            "Leiden resolution ran."
        ),
    ),
    "prost_pnn_domains": ToolOutputProfile(
        tool_name="prost_pnn_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "prost_domains_annotated.h5ad",
                "h5ad",
                "prediction",
                "AnnData with obs['prost_domain']",
                prediction_key="prost_domain",
                prediction_type="cluster_labels",
            ),
            OutputSpec(
                "prost_domain_labels.csv",
                "csv",
                "metadata",
                "per-spot labels: columns spot, prost_domain",
                prediction_key="prost_domain",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="prost_domains_annotated.h5ad",
        prediction_key="prost_domain",
        prediction_type="cluster_labels",
        notes="PROST writes prost_domains_annotated.h5ad (not prost_domains_output.h5ad).",
    ),
    "stlearn_spatial_clustering": ToolOutputProfile(
        tool_name="stlearn_spatial_clustering",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "stlearn_clustering.h5ad",
                "h5ad",
                "prediction",
                prediction_key="louvain",
                prediction_type="cluster_labels",
            ),
            OutputSpec("stlearn_clusters_per_spot.csv", "csv", "metadata"),
        ],
        authoritative_output="stlearn_clustering.h5ad",
        prediction_key="louvain",
        prediction_type="cluster_labels",
        notes="stlearn writes stlearn_clustering.h5ad (not stlearn_output.h5ad) with louvain clustering.",
    ),
    "cellcharter_cluster_spatial_domains": ToolOutputProfile(
        tool_name="cellcharter_cluster_spatial_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "cellcharter_annotated.h5ad",
                "h5ad",
                "prediction",
                prediction_key="cluster_cellcharter",
                prediction_type="cluster_labels",
            ),
            OutputSpec("cellcharter_clusters.csv", "csv", "metadata"),
        ],
        authoritative_output="cellcharter_annotated.h5ad",
        prediction_key="cluster_cellcharter",
        prediction_type="cluster_labels",
    ),
    "run_miso": ToolOutputProfile(
        tool_name="run_miso",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "*miso*.h5ad", "h5ad", "prediction", prediction_key="miso_cluster", prediction_type="cluster_labels"
            ),
            OutputSpec("miso_clusters.csv", "csv", "metadata"),
        ],
        authoritative_output="*miso*.h5ad",
        prediction_key="miso_cluster",
        prediction_type="cluster_labels",
    ),
    "run_sedr": ToolOutputProfile(
        tool_name="run_sedr",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "sedr_clustering.h5ad",
                "h5ad",
                "prediction",
                prediction_key="sedr_cluster",
                prediction_type="cluster_labels",
            ),
            OutputSpec("sedr_clusters_per_spot.csv", "csv", "metadata"),
            OutputSpec("sedr_embedding.csv", "csv", "intermediate"),
        ],
        authoritative_output="sedr_clustering.h5ad",
        prediction_key="sedr_cluster",
        prediction_type="cluster_labels",
        notes=(
            "SEDR writes sedr_clustering.h5ad (not sedr_output.h5ad). Key is always sedr_cluster: KMeans on the "
            "SEDR latent in both DEC modes (SEDR writes no labels of its own)."
        ),
    ),
    "mist_regions_impute": ToolOutputProfile(
        tool_name="mist_regions_impute",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "mist_region_assignments.csv",
                "csv",
                "prediction",
                "One row per spot that passed ReST's QC: spot_id, array_row, array_col, region",
                prediction_key="region",
                prediction_type="cluster_labels",
                schema={
                    "spot_id": "spot barcode",
                    "array_row": "grid row MIST read",
                    "array_col": "grid column MIST read",
                    "region": "ReST region_ind, or 'isolated' for a spot in no region",
                },
                unassigned_labels=["isolated"],
            ),
            OutputSpec("mist_region_stats.txt", "txt", "metadata", "Region sizes, isolated count, coordinate source"),
            OutputSpec(
                "mist_imputed_expression.csv",
                "csv",
                "intermediate",
                "spKNN-imputed CPM, QC-passing spots x genes (index = spot barcode)",
            ),
        ],
        authoritative_output="mist_region_assignments.csv",
        prediction_key="region",
        prediction_type="cluster_labels",
        notes=(
            "The label 'isolated' marks a spot ReST placed in no region; it is not a domain, and the "
            "prediction spec's unassigned_labels makes the inspector, the standardizer and the K-mismatch "
            "gate drop it like NaN. Rows cover only spots that passed ReST's QC (data.n_spots_used)."
        ),
    ),
    "precast_spatial_clustering": ToolOutputProfile(
        tool_name="precast_spatial_clustering",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "precast_clusters.csv",
                "csv",
                "prediction",
                prediction_key="cluster",
                prediction_type="cluster_labels",
                schema={"sample": "sample_<i>, one per input slice", "spot": "spot ID", "cluster": "integer label"},
            ),
            OutputSpec("precast_object.rds", "rds", "intermediate"),
        ],
        authoritative_output="precast_clusters.csv",
        prediction_key="cluster",
        prediction_type="cluster_labels",
        notes="PRECAST writes precast_clusters.csv with columns: sample, spot, cluster.",
    ),
    "run_bass": ToolOutputProfile(
        tool_name="run_bass",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "bass_domains.csv",
                "csv",
                "prediction",
                prediction_key="domain",
                prediction_type="cluster_labels",
                schema={"spot_id": "spot ID", "domain": "integer domain label", "cell_type": "cell type label"},
            ),
            OutputSpec("bass_result.rds", "rds", "intermediate"),
        ],
        authoritative_output="bass_domains.csv",
        prediction_key="domain",
        prediction_type="cluster_labels",
        notes=(
            "BASS writes bass_domains.csv with columns: spot_id, domain, cell_type, x, y. domain is BASS's R "
            "labelling (n_clusters), cell_type its C labelling (n_cell_types). Spots with zero total counts are "
            "not modelled and have no row (payload data.n_spots_empty)."
        ),
    ),
    "seurat_qc_cluster": ToolOutputProfile(
        tool_name="seurat_qc_cluster",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "seurat_qc_cluster_metadata.csv",
                "csv",
                "prediction",
                prediction_key="seurat_clusters",
                prediction_type="cluster_labels",
                schema={"index": "cell barcode", "seurat_clusters": "integer cluster", "RNA_snn_res.0.8": "resolution"},
            ),
            OutputSpec("seurat_qc_cluster_obj.rds", "rds", "intermediate"),
            OutputSpec("seurat_qc_cluster_umap.csv", "csv", "metadata"),
            OutputSpec("seurat_qc_cluster_dimplot.png", "png", "visualization"),
        ],
        authoritative_output="seurat_qc_cluster_metadata.csv",
        prediction_key="seurat_clusters",
        prediction_type="cluster_labels",
        notes="Seurat writes seurat_qc_cluster_metadata.csv with seurat_clusters and RNA_snn_res.* columns.",
    ),
    "seurat_spatial_qc_cluster": ToolOutputProfile(
        tool_name="seurat_spatial_qc_cluster",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "seurat_spatial_metadata.csv",
                "csv",
                "prediction",
                prediction_key="seurat_clusters",
                prediction_type="cluster_labels",
                schema={
                    "index": "spot barcode",
                    "seurat_clusters": "integer cluster",
                    "Spatial_snn_res.*": "resolution",
                },
            ),
            OutputSpec("seurat_spatial_obj.rds", "rds", "intermediate"),
            OutputSpec("seurat_spatial_umap.csv", "csv", "metadata"),
            OutputSpec("seurat_spatial_variable_features.csv", "csv", "metadata"),
            OutputSpec("seurat_spatial_dimplot_clusters.png", "png", "visualization"),
        ],
        authoritative_output="seurat_spatial_metadata.csv",
        prediction_key="seurat_clusters",
        prediction_type="cluster_labels",
        notes="Seurat's Visium pipeline writes seurat_spatial_metadata.csv with seurat_clusters and Spatial_snn_res.* columns.",
    ),
    "spaceflow_identify_domains": ToolOutputProfile(
        tool_name="spaceflow_identify_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "spaceflow_annotated.h5ad",
                "h5ad",
                "prediction",
                prediction_key="spaceflow_domain",
                prediction_type="cluster_labels",
            ),
            # domains.tsv is a prediction too, not metadata: it is SpaceFlow's own segmentation output
            # (one bare label per line, obs_names order). The worker reads it back and attaches it to the
            # h5ad as obs['spaceflow_domain'] on every successful run (a segmentation that wrote no usable
            # labels is an error, not a result). Declaring the TSV here keeps it out of
            # _is_declared_non_prediction's withhold list (output_inspector.py:889), so runs recorded before
            # the attach -- whose h5ad lacks the key -- still reach the "SpaceFlow writes domains.tsv" route
            # at output_inspector.py:1267.
            # authoritative_output stays the h5ad: it carries the barcodes. The TSV is positional --
            # readers pair it with obs_names.
            OutputSpec(
                "domains.tsv",
                "tsv",
                "prediction",
                description="Domain assignments, one label per line in obs_names order",
                prediction_type="cluster_labels",
            ),
            OutputSpec("spaceflow_embedding.csv", "csv", "intermediate"),
        ],
        authoritative_output="spaceflow_annotated.h5ad",
        prediction_key="spaceflow_domain",
        prediction_type="cluster_labels",
        notes="SpaceFlow writes spaceflow_annotated.h5ad (not spaceflow_output.h5ad).",
    ),
    # Alias: registry.yaml uses spaceflow_spatial_domains
    "spaceflow_spatial_domains": ToolOutputProfile(
        tool_name="spaceflow_spatial_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "spaceflow_annotated.h5ad",
                "h5ad",
                "prediction",
                prediction_key="spaceflow_domain",
                prediction_type="cluster_labels",
            ),
            # domains.tsv is a prediction too, not metadata: it is SpaceFlow's own segmentation output
            # (one bare label per line, obs_names order). The worker reads it back and attaches it to the
            # h5ad as obs['spaceflow_domain'] on every successful run (a segmentation that wrote no usable
            # labels is an error, not a result). Declaring the TSV here keeps it out of
            # _is_declared_non_prediction's withhold list (output_inspector.py:889), so runs recorded before
            # the attach -- whose h5ad lacks the key -- still reach the "SpaceFlow writes domains.tsv" route
            # at output_inspector.py:1267.
            # authoritative_output stays the h5ad: it carries the barcodes. The TSV is positional --
            # readers pair it with obs_names.
            OutputSpec(
                "domains.tsv",
                "tsv",
                "prediction",
                description="Domain assignments, one label per line in obs_names order",
                prediction_type="cluster_labels",
            ),
            OutputSpec("spaceflow_embedding.csv", "csv", "intermediate"),
        ],
        authoritative_output="spaceflow_annotated.h5ad",
        prediction_key="spaceflow_domain",
        prediction_type="cluster_labels",
        notes="SpaceFlow writes spaceflow_annotated.h5ad (not spaceflow_output.h5ad).",
    ),
    "spiral_spatial_domains": ToolOutputProfile(
        tool_name="spiral_spatial_domains",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "spiral_output.h5ad",
                "h5ad",
                "prediction",
                prediction_key="leiden",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="spiral_output.h5ad",
        prediction_key="leiden",
        prediction_type="cluster_labels",
        notes="SPIRAL uses leiden clustering. May also have 'louvain' column.",
    ),
    "run_spicemix": ToolOutputProfile(
        tool_name="run_spicemix",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "*spicemix*.h5ad",
                "h5ad",
                "prediction",
                prediction_key="spicemix_factor",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="*spicemix*.h5ad",
        prediction_key="spicemix_factor",
        prediction_type="cluster_labels",
    ),
    "run_spacel_splane": ToolOutputProfile(
        tool_name="run_spacel_splane",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "*spacel*.h5ad",
                "h5ad",
                "prediction",
                prediction_key="splane_cluster",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="*spacel*.h5ad",
        prediction_key="splane_cluster",
        prediction_type="cluster_labels",
    ),
    "run_stdgcn": ToolOutputProfile(
        tool_name="run_stdgcn",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*stdgcn*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec("*stdgcn*dominant*.csv", "csv", "summary"),
            OutputSpec("*stdgcn*.h5ad", "h5ad", "annotated"),
        ],
        authoritative_output="*stdgcn*proportion*.csv",
        prediction_type="proportions_matrix",
        notes="STdGCN deconvolves spots against a single-cell reference; the proportions CSV is the prediction.",
    ),
    "stage_run": ToolOutputProfile(
        tool_name="stage_run",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "adata_stage.h5ad",
                "h5ad",
                "prediction",
                description="AnnData from STAGE recovery/generation — may need post-hoc clustering",
                prediction_key="leiden",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="adata_stage.h5ad",
        prediction_key="leiden",
        prediction_type="cluster_labels",
        notes="STAGE is a spatial recovery tool. It does NOT produce cluster labels natively. "
        "The STCoscientist agent must run leiden/louvain on the STAGE embedding after recovery. "
        "Fallback keys: leiden, louvain, cluster, domain.",
    ),
    "run_iris": ToolOutputProfile(
        tool_name="run_iris",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "iris_domains.csv",
                "csv",
                "prediction",
                description="CSV with IRIS spatial domain assignments",
                prediction_key="domain",
                prediction_type="cluster_labels",
                schema={"spot": "spot name", "domain": "integer domain label"},
            ),
            OutputSpec("iris_spatial_domains.csv", "csv", "metadata"),
            OutputSpec("iris_proportions.csv", "csv", "metadata"),
            OutputSpec("iris_result.rds", "rds", "intermediate"),
        ],
        authoritative_output="iris_domains.csv",
        prediction_key="domain",
        prediction_type="cluster_labels",
        notes="IRIS is an R tool; writes iris_domains.csv (not h5ad) with columns: spot, domain. Spots IRIS's own QC leaves out (low-count spots; data.n_spots_dropped in the payload) have no row, so the file can be shorter than the input.",
    ),
    "spatialprompt_cluster": ToolOutputProfile(
        tool_name="spatialprompt_cluster",
        task_type="spatial_clustering",
        outputs=[
            OutputSpec(
                "spatialprompt_spatial_with_clusters.h5ad",
                "h5ad",
                "prediction",
                description="AnnData with SpatialPrompt cluster labels",
                prediction_key="spatialprompt_cluster",
                prediction_type="cluster_labels",
            ),
            OutputSpec(
                "spatialprompt_spot_clusters.csv",
                "csv",
                "prediction",
                description="CSV with spot cluster assignments",
                prediction_key="spatialprompt_cluster",
                prediction_type="cluster_labels",
            ),
        ],
        authoritative_output="spatialprompt_spatial_with_clusters.h5ad",
        prediction_key="spatialprompt_cluster",
        prediction_type="cluster_labels",
        notes="SpatialPrompt writes h5ad and CSV. CSV has index=spot_barcode, column=spatialprompt_cluster. "
        "Rows cover only on-tissue spots when obs['in_tissue'] is a 0/1 flag (data.n_spots_used of data.n_spots).",
    ),
    # ═══════════════════════════════════════════════════════════════════
    # SVG DETECTION TOOLS
    # ═══════════════════════════════════════════════════════════════════
    "hotspot_spatial_modules": ToolOutputProfile(
        tool_name="hotspot_spatial_modules",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "hotspot_annotated.h5ad",
                "h5ad",
                "metadata",
                description="AnnData with var['hotspot_module'] and obsm['hotspot_module_scores']",
            ),
            OutputSpec(
                "hotspot_gene_autocorrelations.csv",
                "csv",
                "prediction",
                prediction_key="Gene",
                prediction_type="gene_list_with_pvalues",
                schema={"index": "gene name", "Z": "z-score", "FDR": "adjusted p-value"},
            ),
            OutputSpec("hotspot_gene_modules.csv", "csv", "metadata"),
            OutputSpec("hotspot_module_scores_per_cell.csv", "csv", "metadata"),
        ],
        authoritative_output="hotspot_gene_autocorrelations.csv",
        prediction_key="Gene",
        prediction_type="gene_list_with_pvalues",
        notes="Module-based method. Filter by FDR < 0.05. Gene name is index, not a column.",
    ),
    "prost_index_svg": ToolOutputProfile(
        tool_name="prost_index_svg",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                [
                    "prost_top_svg_genes.csv",
                    "prost_index_all_gene_scores.csv",
                    "*svg*.csv",
                ],
                "csv",
                "prediction",
                prediction_key=["gene", "Gene", "g"],
                prediction_type="gene_list_with_scores",
                schema={"gene": "gene name", "PI": "PROST Index score"},
                # The worker's top n_top_genes by PI (prost_worker.py), not every gene it scored.
                selection_patterns=["prost_top_svg_genes.csv"],
            ),
        ],
        authoritative_output=["prost_top_svg_genes.csv", "prost_index_all_gene_scores.csv", "*svg*.csv"],
        prediction_key=["gene", "Gene", "g"],
        prediction_type="gene_list_with_scores",
    ),
    "spagft_identify_svg": ToolOutputProfile(
        tool_name="spagft_identify_svg",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                ["spagft_top_svg_genes.csv", "spagft_svg_scores.csv", "*svg*.csv"],
                "csv",
                "prediction",
                prediction_key=["gene", "Gene", "g"],
                prediction_type="gene_list_with_pvalues",
                # The worker's top n_top_genes; on the spectral substitute it carries spagft_score only.
                selection_patterns=["spagft_top_svg_genes.csv"],
            ),
        ],
        authoritative_output=["spagft_top_svg_genes.csv", "spagft_svg_scores.csv", "*svg*.csv"],
        prediction_key=["gene", "Gene", "g"],
        prediction_type="gene_list_with_pvalues",
        notes="spagft pvalue distribution is degenerate (~93% of genes pass p<0.05); always prefer the top-N file over full scores.",
    ),
    "spark_svg_detection": ToolOutputProfile(
        tool_name="spark_svg_detection",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                [
                    "spark_significant_svgs.csv",
                    "spark_top_svgs.csv",
                    "spark_results.csv",
                    "*spark*.csv",
                ],
                "csv",
                "prediction",
                prediction_key=["gene", "Gene", "g"],
                prediction_type="gene_list_with_pvalues",
            ),
        ],
        authoritative_output=[
            "spark_significant_svgs.csv",
            "spark_top_svgs.csv",
            "spark_results.csv",
            "*spark*.csv",
        ],
        prediction_key=["gene", "Gene", "g"],
        prediction_type="gene_list_with_pvalues",
    ),
    "spatialde_run_svg": ToolOutputProfile(
        tool_name="spatialde_run_svg",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "*results*.csv",
                "csv",
                "prediction",
                prediction_key="g",
                prediction_type="gene_list_with_pvalues",
                schema={"g": "gene name", "pval": "p-value", "qval": "q-value/FDR"},
            ),
        ],
        # "results" is generic, so the glob alone can never name this tool. spatialde_worker.py (res_csv, written atomically)
        # writes the specific name; the glob stays for the runs that only carry a variant of it.
        authoritative_output=["spatialde_results.csv", "*results*.csv"],
        prediction_key="g",
        prediction_type="gene_list_with_pvalues",
        notes="SpatialDE uses 'g' column for gene names",
    ),
    "svgbit_run": ToolOutputProfile(
        tool_name="svgbit_run",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "AI.csv",
                "csv",
                "prediction",
                description="AI (Autocorrelation Index, 0-1) per analysed gene; gene names are in the 'gene' column. "
                "Row count equals data.n_genes_used.",
                prediction_key="gene",
                prediction_type="gene_list_with_scores",
                schema={
                    "gene": "gene name",
                    "AI": "Autocorrelation Index score (0-1, higher = more spatially variable)",
                },
            ),
            OutputSpec(
                "hotspot_df.csv",
                "csv",
                "metadata",
                description="Local Moran's I hotspot indicator matrix (spots x genes, 0/1). The first column holds the "
                "spot barcode (headed 'gene' when the obs index is unnamed, else by the index name).",
            ),
            OutputSpec(
                "Di.csv",
                "csv",
                "metadata",
                description="Local Di per spot and gene (spots x genes); first column holds the spot barcode as in "
                "hotspot_df.csv.",
            ),
            OutputSpec(
                "svg_cluster.csv",
                "csv",
                "metadata",
                description="SVG cluster of each of the top n_svgs genes by AI (columns 'gene', '0').",
            ),
        ],
        authoritative_output="AI.csv",
        prediction_key="gene",
        prediction_type="gene_list_with_scores",
        notes="AI.csv has a 'gene' column and an 'AI' column (Autocorrelation Index, 0-1; higher = more spatially "
        "variable) and ranks every analysed gene. svg_ranked.csv is the same table sorted by AI.",
    ),
    "somde_run": ToolOutputProfile(
        tool_name="somde_run",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "*somde*.csv",
                "csv",
                "prediction",
                prediction_key="gene",
                prediction_type="gene_list_with_pvalues",
                schema={"gene": "gene name", "pval": "p-value", "qval": "adjusted p-value"},
            ),
        ],
        authoritative_output="*somde*.csv",
        prediction_key="gene",
        prediction_type="gene_list_with_pvalues",
    ),
    "spotgf_detect_svg": ToolOutputProfile(
        tool_name="spotgf_detect_svg",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "*spotgf*.csv",
                "csv",
                "prediction",
                prediction_key="gene",
                prediction_type="gene_list_with_scores",
            ),
        ],
        authoritative_output="*spotgf*.csv",
        prediction_key="gene",
        prediction_type="gene_list_with_scores",
    ),
    "spvc_svg_detection": ToolOutputProfile(
        tool_name="spvc_svg_detection",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                [
                    "spvc_top_svgs.csv",
                    "spvc_significant_svgs.csv",
                    "spvc_results.csv",
                    "*spvc*.csv",
                ],
                "csv",
                "prediction",
                prediction_key=["gene", "Gene", "g"],
                prediction_type="gene_list_with_pvalues",
            ),
        ],
        authoritative_output=[
            "spvc_top_svgs.csv",
            "spvc_significant_svgs.csv",
            "spvc_results.csv",
            "*spvc*.csv",
        ],
        prediction_key=["gene", "Gene", "g"],
        prediction_type="gene_list_with_pvalues",
    ),
    "bsp_identify_svg": ToolOutputProfile(
        tool_name="bsp_identify_svg",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "bsp_results.csv",
                "csv",
                "prediction",
                description="Full results: gene + p_values for all kept genes. Pipeline applies p<0.05 to extract significant genes.",
                prediction_key="gene",
                prediction_type="gene_list_with_pvalues",
            ),
            OutputSpec(
                "bsp_top_genes.csv",
                "csv",
                "prediction",
                description="Top-200 curated. Higher precision but lower recall vs the curated 458-gene Visium SVG GT — full-results path scores higher F1 on this benchmark.",
                prediction_key="gene",
                prediction_type="gene_list_with_pvalues",
            ),
            OutputSpec(
                "predicted_genes.json",
                "json",
                "prediction",
                description="Top-200 list in JSON form. Same recall ceiling as bsp_top_genes.csv.",
                prediction_type="gene_list",
            ),
        ],
        authoritative_output=[
            "bsp_results.csv",
            "bsp_top_genes.csv",
        ],
        prediction_key="gene",
        prediction_type="gene_list_with_pvalues",
        notes=(
            "BSP (scbsp). bsp_results.csv (full, p<0.05 → ~2101 genes on Visium) is the best "
            "F1-by-benchmark choice given the curated 458-gene GT. bsp_top_genes.csv (top-200) "
            "has higher precision but recall too low to win on F1. Worker also writes "
            "predicted_genes.json as a curated gene-list artifact."
        ),
    ),
    "run_celina": ToolOutputProfile(
        tool_name="run_celina",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "*celina*.csv",
                "csv",
                "prediction",
                description=(
                    "CELINA cell-type-specific SVG results: one row per (cell_type, gene) with Gaussian1..5, "
                    "Matern1..5, Spline and CombinedPvals"
                ),
                prediction_key="gene",
                prediction_type="gene_list_with_pvalues",
                schema={
                    "cell_type": "cell type tested",
                    "gene": "gene name",
                    "CombinedPvals": "CELINA combined p-value (uncorrected)",
                },
            ),
        ],
        authoritative_output="*celina*.csv",
        prediction_key="gene",
        prediction_type="gene_list_with_pvalues",
        notes=(
            "CELINA cell-type-specific SVG detection. Long table: cell_type, gene, 11 kernel p-values, "
            "CombinedPvals (the significance CELINA reports)."
        ),
    ),
    "spotgf_denoise": ToolOutputProfile(
        tool_name="spotgf_denoise",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                ["SpotGF_scores.txt", "*potGF*score*.txt", "*spotgf*.csv"],
                "txt",
                "prediction",
                prediction_key=["geneID", "gene", "Gene"],
                prediction_type="gene_list_with_scores",
            ),
        ],
        authoritative_output=["SpotGF_scores.txt", "*potGF*score*.txt", "*spotgf*.csv"],
        prediction_key=["geneID", "gene", "Gene"],
        prediction_type="gene_list_with_scores",
        notes=(
            "SpotGF writes a tab-separated SpotGF_scores.txt (columns geneID, SpotGF_score), "
            "not a CSV; fnmatch is case-sensitive on POSIX, hence the *potGF* alternative."
        ),
    ),
    "squidpy_spatial_autocorr": ToolOutputProfile(
        tool_name="squidpy_spatial_autocorr",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "predicted_genes.json",
                "json",
                "prediction",
                description="Curated top-N SVG list (Phase-8 worker fix). PREFERRED authoritative output.",
                prediction_type="gene_list",
            ),
            OutputSpec(
                ["squidpy_moranI.csv", "squidpy_gearyC.csv"],
                "csv",
                "prediction",
                description="Per-gene autocorrelation scores. Gene names live in the CSV index (becomes 'Unnamed: 0' on default read_csv).",
                prediction_key=["gene", "Unnamed: 0"],
                prediction_type="gene_list_with_pvalues",
                schema={
                    "index": "gene name",
                    "I": "Moran's I",
                    "C": "Geary's C",
                    "pval_norm": "p-value (normal approx)",
                },
            ),
            OutputSpec("squidpy_spatial_autocorr_*.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output=[
            "predicted_genes.json",
            "squidpy_moranI.csv",
            "squidpy_gearyC.csv",
        ],
        prediction_key=["gene", "Unnamed: 0"],
        prediction_type="gene_list_with_pvalues",
        notes=(
            "Squidpy spatial autocorrelation. Gene names are in the CSV index (first unnamed column "
            "→ becomes 'Unnamed: 0' after default read_csv). Mode-dependent filename: "
            "squidpy_moranI.csv (mode=moran) or squidpy_gearyC.csv (mode=geary). "
            "Phase 8 fix added a curated predicted_genes.json (top-N filter) which is the preferred "
            "authoritative output when present. predicted_genes.json keeps genes with positive "
            "autocorrelation (I > E[I], or C < 1) whose p in params.pvalue_column passes pvalue_threshold: "
            "pval_sim when n_perms is set (portal default 100), pval_norm when n_perms is None; the CSV keeps "
            "squidpy's columns as written ('pval_norm' is always present)."
        ),
    ),
    "svca_variance_decomposition": ToolOutputProfile(
        tool_name="svca_variance_decomposition",
        task_type="svg_detection",
        outputs=[
            OutputSpec(
                "svca_variance_decomposition.csv",
                "csv",
                "prediction",
                description="CSV with a two-way variance split per gene: intrinsic (spatial) and noise fractions",
                prediction_key="gene",
                prediction_type="gene_list_with_scores",
                schema={
                    "gene": "gene name",
                    "intrinsic": "spatial variance fraction (higher = more spatially variable)",
                    # Present for schema stability and always empty. The worker fits one spatial
                    # random effect against i.i.d. noise, so there is no third component; it used to
                    # be written as a literal 0.0, which reads as a measured absence of environmental
                    # signal rather than as a component nobody estimated.
                    "environmental": "always empty -- not estimated by this implementation",
                    "noise": "residual (non-spatial) variance fraction; intrinsic + noise = 1",
                },
            ),
            OutputSpec("svca_summary.csv", "csv", "metadata", description="Top genes by spatial variance"),
        ],
        authoritative_output="svca_variance_decomposition.csv",
        prediction_key="gene",
        prediction_type="gene_list_with_scores",
        notes="SVCA variance decomposition. Score column is 'intrinsic' (spatial variance fraction). "
        "Higher intrinsic fraction = more spatially variable. No p-value column.",
    ),
    # ═══════════════════════════════════════════════════════════════════
    # DECONVOLUTION TOOLS
    # ═══════════════════════════════════════════════════════════════════
    "run_cell2location": ToolOutputProfile(
        tool_name="run_cell2location",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                ["*proportions_normalized*.csv", "cell2location_proportions.csv"],
                "csv",
                "prediction",
                description="Optional normalized cell type proportions CSV (only written by some wrapper variants)",
                prediction_type="proportions_matrix",
            ),
            OutputSpec(
                ["sp.h5ad", "cell2location_map/sp.h5ad", "*cell2location*spatial*.h5ad"],
                "h5ad",
                "prediction",
                description="AnnData with obsm['q05_cell_abundance_w_sf'] (or means variant)",
                prediction_key=["q05_cell_abundance_w_sf", "means_cell_abundance_w_sf", "cell_abundance"],
                prediction_type="cell_type_abundance_matrix",
            ),
            OutputSpec(
                "reference_signatures/inf_aver.csv",
                "csv",
                "metadata",
                description="Inferred average expression per cell type",
            ),
        ],
        # The words were transposed here: no run has ever produced "proportions_normalized.csv",
        # and the worker writes only sp.h5ad + inf_aver.csv (cell2location_worker.py:489,562) --
        # the proportions CSVs are written downstream, as "<tool>_normalized_proportions.csv".
        # Post-analysis globs this list to find the result, so a name nothing writes made every
        # cell2location run come back "unusable". Most specific first.
        authoritative_output=[
            "*normalized_proportions*.csv",
            "cell2location_proportions.csv",
            "proportions.csv",
            "sp.h5ad",
        ],
        prediction_type="proportions_matrix",
        notes="Prefer the normalized proportions CSV. Fallback: sp.h5ad obsm['q05_cell_abundance_w_sf'].",
    ),
    "tangram_map_sc_to_spatial": ToolOutputProfile(
        tool_name="tangram_map_sc_to_spatial",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                "tangram_celltype_probabilities.csv",
                "csv",
                "prediction",
                prediction_key="index=barcode, columns=cell_types",
                prediction_type="proportions_matrix",
            ),
            OutputSpec("tangram_spatial_with_annotations.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output="tangram_celltype_probabilities.csv",
        prediction_key="columns",
        prediction_type="proportions_matrix",
        notes=(
            "tangram_celltype_probabilities.csv rows are per-spot compositions (sum to 1; clusters mode weighted by "
            "cluster_density). In tangram_spatial_with_annotations.h5ad, obsm['tangram_ct_proportions'] matches the "
            "CSV; obsm['tangram_ct_pred'] is Tangram's unweighted score matrix, not proportions."
        ),
    ),
    "tacco_annotate": ToolOutputProfile(
        tool_name="tacco_annotate",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*compositions*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec("*annotated*.h5ad", "h5ad", "metadata"),
        ],
        # tacco_worker.py:76 writes "tacco_composition.csv" -- singular. The plural glob below it
        # matched no file the tool has ever produced; the singular form covers both spellings.
        # The bare "*composition*.csv" that used to sit here is gone: "composition" is now task
        # vocabulary (detect._GENERIC_TOKENS), so the pattern could no longer score in any case, and
        # while it did it credited tacco with three recorded runs written by destvi and spacexr.
        authoritative_output=["tacco_composition.csv"],
        prediction_type="proportions_matrix",
    ),
    "graphst_deconvolution": ToolOutputProfile(
        tool_name="graphst_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec("graphst_celltype_abundance.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec("graphst_deconvolution_output.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output="graphst_celltype_abundance.csv",
        prediction_type="proportions_matrix",
    ),
    "spacexr_rctd_deconvolution": ToolOutputProfile(
        tool_name="spacexr_rctd_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                "proportions.csv",
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
                schema={"index": "barcode", "columns": "cell types", "values": "proportions"},
            ),
            # The worker writes proportions.csv as the result and then "Keep[s] the legacy
            # spacexr_weights.csv for any external consumer still expecting it" -- in full mode the
            # same numbers with a spot column instead of an index, in doublet and multi mode the
            # per-spot calls. It is not the result, so it is metadata and NOT authoritative; listing
            # it in both places let the two fields contradict each other. The historical glob also
            # matches the mode-specific sidecars spacexr_rctd_doublet_weights.csv (the doublet call,
            # at most 2 types per spot) and spacexr_rctd_multi_weights.csv (the multi decomposition),
            # which are metadata for the same reason: the benchmark scores proportions.csv.
            OutputSpec(["spacexr_weights.csv", "*rctd*weights*.csv"], "csv", "metadata"),
        ],
        authoritative_output="proportions.csv",
        prediction_type="proportions_matrix",
    ),
    "run_card": ToolOutputProfile(
        tool_name="run_card",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*card*proportions*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*card*proportions*.csv",
        prediction_type="proportions_matrix",
    ),
    "run_destvi": ToolOutputProfile(
        tool_name="run_destvi",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*destvi*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec("*destvi*.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output="*destvi*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    "run_spotlight": ToolOutputProfile(
        tool_name="run_spotlight",
        task_type="deconvolution",
        outputs=[
            OutputSpec("spotlight_proportions.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="spotlight_proportions.csv",
        prediction_type="proportions_matrix",
    ),
    "stride_deconvolution": ToolOutputProfile(
        tool_name="stride_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                ["*spot_celltype_frac*.txt", "*spot_celltype_frac*.csv"],
                "txt",
                "prediction",
                prediction_type="proportions_matrix",
            ),
            OutputSpec("*dominant_celltype*.csv", "csv", "summary"),
        ],
        authoritative_output=["*spot_celltype_frac*.txt", "*spot_celltype_frac*.csv"],
        prediction_type="proportions_matrix",
        notes=(
            "STRIDE's deconvolution result is the tab-separated <prefix>_spot_celltype_frac.txt "
            "(spots x cell types). The *_topic_spot_mat_*.txt files are the LDA topic x spot "
            "matrix -- latent topics, not cell types -- and must not be scored as proportions."
        ),
    ),
    "run_stdeconvolve": ToolOutputProfile(
        tool_name="run_stdeconvolve",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                "stdeconvolve_proportions_mapped.csv",
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
                description="Topic→celltype mapped proportions (post-processor). Prefer over raw theta when present.",
            ),
            OutputSpec("stdeconvolve_theta.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec("stdeconvolve_beta.csv", "csv", "metadata"),
        ],
        authoritative_output=["stdeconvolve_proportions_mapped.csv", "stdeconvolve_theta.csv"],
        prediction_type="proportions_matrix",
        notes=(
            "Reference-free LDA. theta=spot proportions (spots x topics, integer column names), "
            "beta=topic loadings (topics x genes). For deconv eval, run "
            "_stdec_topic_to_celltype.py to produce stdeconvolve_proportions_mapped.csv "
            "with cell-type-named columns (mapped via beta-vs-celltype-mean correlation)."
        ),
    ),
    "ucdeconvolve_base": ToolOutputProfile(
        tool_name="ucdeconvolve_base",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*ucdeconvolve*proportions*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        # ucdeconvolve_worker.py (_export_ucd_results) writes "<key>_<category>_predictions.csv" -- never a file with
        # "proportions" in the name, so the glob below it matched nothing.
        authoritative_output=["*_predictions.csv", "*ucdeconvolve*proportions*.csv"],
        prediction_type="proportions_matrix",
    ),
    "stereoscope_deconvolution": ToolOutputProfile(
        tool_name="stereoscope_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*stereoscope*W*.tsv", "tsv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*stereoscope*W*.tsv",
        prediction_type="proportions_matrix",
        notes="Stereoscope outputs TSV with W matrix (proportions).",
    ),
    "run_celldart": ToolOutputProfile(
        tool_name="run_celldart",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                ["celldart_proportions.csv", "*celldart*proportions*.csv", "cellfraction.csv"],
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
            ),
            OutputSpec("celldart_spatial.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output=["celldart_proportions.csv", "*celldart*proportions*.csv", "cellfraction.csv"],
        prediction_type="proportions_matrix",
        notes="Worker writes celldart_proportions.csv as authoritative; older / inner-loop CelldDART code paths emit cellfraction.csv.",
    ),
    "run_bulk2space": ToolOutputProfile(
        tool_name="run_bulk2space",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                [
                    "bulk2space_proportions.csv",
                    "*bulk2space*proportion*.csv",
                    "deconv_result.csv",
                    "result.csv",
                    "proportions.csv",
                ],
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
            ),
        ],
        authoritative_output=[
            "bulk2space_proportions.csv",
            "*bulk2space*proportion*.csv",
            "deconv_result.csv",
            "result.csv",
            "proportions.csv",
        ],
        prediction_type="proportions_matrix",
        notes=(
            "run_bulk2space runs NNLS on the reference's cell-type means (upstream Bulk2Space is "
            "not run) and writes bulk2space_proportions.csv atomically (.partial + rename). The "
            "generic alternatives below were what a never-reached upstream branch looked for; no "
            "bulk2space run writes them."
        ),
    ),
    "starfysh_deconvolution": ToolOutputProfile(
        tool_name="starfysh_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                ["cell_type_proportions.csv", "*starfysh*proportion*.csv"],
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
            ),
            OutputSpec("starfysh_annotated.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output=["cell_type_proportions.csv", "*starfysh*proportion*.csv"],
        prediction_type="proportions_matrix",
        notes="Worker writes cell_type_proportions.csv (no 'starfysh_' prefix). Old glob fell through to keyword fallback.",
    ),
    "bayestme_deconvolution": ToolOutputProfile(
        tool_name="bayestme_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*bayestme*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec(
                ["bayestme_deconvolved.h5ad", "*bayestme*deconvolved*.h5ad"],
                "h5ad",
                "prediction",
                prediction_key=["bayestme_cell_type_probabilities", "bayestme_cell_type_counts"],
                prediction_type="proportions_matrix",
            ),
            # Derived tables, not predictions. Labelling them matters: without this the
            # candidate scanner accepted marker_genes.csv (gene_name/rank_in_cell_type/
            # cell_type) as a (1500, 2) proportions matrix, reporting the two bookkeeping
            # columns as the cell types.
            OutputSpec("marker_genes.csv", "csv", "metadata"),
            OutputSpec("omega.csv", "csv", "metadata"),
            OutputSpec("relative_expression.csv", "csv", "metadata"),
            OutputSpec("*bayestme*marker*.h5ad", "h5ad", "metadata"),
        ],
        authoritative_output=["*bayestme*proportion*.csv", "bayestme_deconvolved.h5ad"],
        prediction_type="proportions_matrix",
        notes=(
            "BayesTME publishes its result in the h5ad (adata_h5ad), not a CSV: "
            "obsm['bayestme_cell_type_probabilities'] holds spot x cell-type proportions "
            "and obsm['bayestme_cell_type_counts'] the corresponding counts."
        ),
    ),
    "spacet_deconvolution": ToolOutputProfile(
        tool_name="spacet_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*spacet*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
            OutputSpec(
                "spacet_lineage_levels.json",
                "json",
                "metadata",
                "The table's two lineage levels, as the payload's data.major_lineages / data.sub_lineages",
                schema={"major_lineages": "rows that sum to 1 per spot", "sub_lineages": "rows that split them"},
            ),
        ],
        authoritative_output="*spacet*proportion*.csv",
        prediction_type="proportions_matrix",
        notes=(
            "propMat is cell types x spots and hierarchical: major lineages sum to 1 per spot, sub-lineages "
            "each sum to their parent. Scored on the major lineages only (named by data.major_lineages), "
            "before row normalisation; output_standardizer reads them from major_lineages= or the "
            "spacet_lineage_levels.json the worker writes beside the table, and refuses a hierarchical table "
            "whose levels it cannot name (hunt 2026-09-30, u31-benchmarking-10)."
        ),
    ),
    "redeconve_deconvolution": ToolOutputProfile(
        tool_name="redeconve_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*redeconve*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*redeconve*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    "spatialprompt_deconvolution": ToolOutputProfile(
        tool_name="spatialprompt_deconvolution",
        task_type="deconvolution",
        outputs=[
            OutputSpec(
                ["cell_type_proportions.csv", "*spatialprompt*proportion*.csv"],
                "csv",
                "prediction",
                prediction_type="proportions_matrix",
            ),
        ],
        authoritative_output=["cell_type_proportions.csv", "*spatialprompt*proportion*.csv"],
        prediction_type="proportions_matrix",
        notes="Worker writes cell_type_proportions.csv (no 'spatialprompt_' prefix). "
        "Rows cover only on-tissue spots when obs['in_tissue'] is a 0/1 flag (data.n_spots_used of data.n_spots).",
    ),
    "run_dstg": ToolOutputProfile(
        tool_name="run_dstg",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*dstg*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*dstg*proportion*.csv",
        prediction_type="proportions_matrix",
        notes="Worker writes dstg_proportions.csv (spots x reference cell types) and dstg_dominant_celltype.csv. "
        "The method is a DSTG-style GCN reimplementation; params.method in the payload says so.",
    ),
    "run_smart": ToolOutputProfile(
        tool_name="run_smart",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*smart*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*smart*proportion*.csv",
        prediction_type="proportions_matrix",
        notes=(
            "Worker writes smart_proportions.csv (spots x topics plus a 'spot' column): one column per marker "
            "cell type that seeded a keyATM topic, then Other_<k> unsupervised topics. Types in "
            "data.dropped_marker_types have no column. SMART reads no single-cell reference."
        ),
    ),
    "run_gist": ToolOutputProfile(
        tool_name="run_gist",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*gist*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*gist*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    "run_celloscope": ToolOutputProfile(
        tool_name="run_celloscope",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*celloscope*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        # celloscope_proportions.csv: spots x reference cell types, rows sum to 1, from Celloscope's point
        # estimate chain01/thetas_est.csv (dummy type removed, renormalised). "result_h.csv" is not a result:
        # impl.py appends one flattened n_spots*n_types row to it every "how often drop" iterations (the MCMC
        # trace), and naming it first here made post-analysis pick the trace over the estimate.
        authoritative_output="*celloscope*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    "run_cellpie": ToolOutputProfile(
        tool_name="run_cellpie",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*cellpie*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*cellpie*proportion*.csv",
        prediction_type="proportions_matrix",
        notes=(
            "CellPie intNMF is reference-free: cellpie_proportions.csv columns are topic_0..topic_{n-1} "
            "(row-normalised topic loadings), not cell types, and no scRNA-seq reference shapes them. "
            "params.method / params.used_fallback say whether image features, the expression-only mode, "
            "or the |PCA| stand-in ran."
        ),
    ),
    "run_scresolve": ToolOutputProfile(
        tool_name="run_scresolve",
        task_type="resolution",
        outputs=[
            OutputSpec("*scresolve*enhanced*.h5ad", "h5ad", "prediction", prediction_type="enhanced_expression"),
        ],
        authoritative_output="*scresolve*enhanced*.h5ad",
        prediction_type="enhanced_expression",
        notes=(
            "run_scresolve does not run upstream scResolve: it scores each spot for reference "
            "cell-type marker signatures (scanpy rank_genes_groups + score_genes) at the input spot "
            "resolution, so no resolution enhancement happens (ratio 1.0). Its worker declares "
            "task='resolution' and emits only scresolve_enhanced.h5ad (historical name): the in-tissue "
            "spots of the spatial input (obs['in_tissue'] == 0 background spots are left out) plus "
            "obs['score_<cell type>'] columns. It produces no cell-type proportions and "
            "must not be listed as a deconvolution tool."
        ),
    ),
    "run_spatialdecon": ToolOutputProfile(
        tool_name="run_spatialdecon",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*spatialdecon*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*spatialdecon*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    "run_spatialscope": ToolOutputProfile(
        tool_name="run_spatialscope",
        task_type="deconvolution",
        outputs=[
            OutputSpec("*spatialscope*proportion*.csv", "csv", "prediction", prediction_type="proportions_matrix"),
        ],
        authoritative_output="*spatialscope*proportion*.csv",
        prediction_type="proportions_matrix",
    ),
    # ═══════════════════════════════════════════════════════════════════
    # CELL COMMUNICATION TOOLS
    # ═══════════════════════════════════════════════════════════════════
    "ncem_cell_communication": ToolOutputProfile(
        tool_name="ncem_cell_communication",
        task_type="cell_communication",
        outputs=[
            OutputSpec(
                "ncem_communication_matrix.csv",
                "csv",
                "prediction",
                "Gene x cell-type neighbourhood effect sizes. Signed: a negative entry is a real "
                "result, not a missing one.",
                prediction_type="communication_matrix",
            ),
            OutputSpec(
                "ncem_neighborhood_composition.csv",
                "csv",
                "prediction",
                "Spot x cell-type frequencies of each spot's NEIGHBOURS -- not of the spot itself.",
                prediction_type="neighborhood_composition",
            ),
            OutputSpec("ncem_communication_strength.csv", "csv", "metadata"),
            OutputSpec("ncem_top_genes_per_type.csv", "csv", "metadata"),
        ],
        authoritative_output=["ncem_communication_matrix.csv", "ncem_neighborhood_composition.csv"],
        prediction_type="communication_matrix",
        notes=(
            "This wrapper fits an NCEM-style linear model (sklearn Ridge on neighbour cell-type "
            "composition; the ncem package is not run) of how a cell's expression depends on its "
            "neighbours; it is not a "
            "deconvolution tool and must not be listed as one. Its neighbourhood composition is "
            "200 x 3, non-negative and row-normalised -- the same shape as a proportions matrix -- "
            "so nothing in the output distinguishes the two and detection cannot be left to "
            "content. Without this entry the file was claimed by tacco_annotate's '*composition*"
            ".csv' glob and the run published as a deconvolution, crediting a tool that never ran; "
            "with the glob removed instead, the content route read the gene x cell-type effect "
            "matrix as an SVG ranking and scored genes by the column 'TypeA'."
        ),
    ),
    # ═══════════════════════════════════════════════════════════════════
    # FUNCTIONAL ENRICHMENT (pathway_enrichment portal, 2026-09-20)
    # ═══════════════════════════════════════════════════════════════════
    "run_pathway_enrichment": ToolOutputProfile(
        tool_name="run_pathway_enrichment",
        task_type="functional_enrichment",
        outputs=[
            OutputSpec(
                "enrichment_summary.csv",
                "csv",
                "prediction",
                "Top terms per collection and method: collection, method, term, fdr, effect, n_overlap, leading_edge",
                prediction_type="enrichment_table",
            ),
            OutputSpec(
                "ora_*.csv", "csv", "metadata", "ORA table for one collection (Term, FDR p-value, Odds ratio, Features)"
            ),
            OutputSpec(
                "gsea_*.csv",
                "csv",
                "metadata",
                "GSEA-prerank table for one collection (Term, NES, FDR p-value, Leading edge)",
            ),
            OutputSpec("enrichment_dotplot.png", "png", "visualization"),
            OutputSpec("input_genes_used.txt", "txt", "intermediate"),
        ],
        authoritative_output="enrichment_summary.csv",
        prediction_type="enrichment_table",
        notes=(
            "functional_enrichment is not one of the post-analysis contract's task types, and that is the "
            "point of this entry: without a profile the engine typed the term table by CONTENT -- "
            "enrichment_summary.csv read as a deconvolution and pathway_activity.h5ad as a clustering -- and "
            "the tool's own figure never reached the run card (driven 2026-09-20). With the profile the "
            "run is recorded as partial with one warning, and the card shows the dotplot the tool drew."
        ),
    ),
    "run_pathway_activity": ToolOutputProfile(
        tool_name="run_pathway_activity",
        task_type="functional_enrichment",
        outputs=[
            OutputSpec(
                "pathway_activity.h5ad",
                "h5ad",
                "prediction",
                "The input AnnData with per-spot activity in obsm['<method>_estimate'] and p-values in obsm['<method>_pvals']",
                prediction_key=["ulm_estimate", "ora_estimate"],
                prediction_type="activity_matrix",
            ),
            OutputSpec("pathway_activity_scores.csv", "csv", "metadata", "spots x pathways activity scores"),
            OutputSpec(
                "pathway_activity_by_group.csv", "csv", "metadata", "mean activity per group (when group_key was given)"
            ),
            OutputSpec(
                "pathway_activity_group_tests.csv", "csv", "metadata", "per-group t-tests (when group_key was given)"
            ),
            OutputSpec("spatial_pathway_activity.png", "png", "visualization"),
        ],
        authoritative_output="pathway_activity.h5ad",
        prediction_key=["ulm_estimate", "ora_estimate"],
        prediction_type="activity_matrix",
        notes="See run_pathway_enrichment: the same portal, the per-spot task.",
    ),
    # ═══════════════════════════════════════════════════════════════════
    # VISUALIZATION TOOLS (spatial_viz portal)
    # ═══════════════════════════════════════════════════════════════════
    "plot_spatial_expression": ToolOutputProfile(
        tool_name="plot_spatial_expression",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "genes or a numeric column painted on the tissue, optionally over histology",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_spatial_annotation": ToolOutputProfile(
        tool_name="plot_spatial_annotation",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png", "png", "visualization", "a categorical annotation painted on the tissue"
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_embedding": ToolOutputProfile(
        tool_name="plot_embedding",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a stored embedding coloured by genes or metadata",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "generate_qc_report": ToolOutputProfile(
        tool_name="generate_qc_report",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "counts, detected genes and mitochondrial fraction, as distributions and on the tissue",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_differential_expression": ToolOutputProfile(
        tool_name="plot_differential_expression",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a volcano, ranked markers or a heatmap from a STORED differential-expression result",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_deconvolution": ToolOutputProfile(
        tool_name="plot_deconvolution",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "cell-type proportion maps, the dominant type per spot, or a composition summary",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_marker_expression": ToolOutputProfile(
        tool_name="plot_marker_expression",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "marker genes across groups as a dot plot, violins or a heatmap, or group composition",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_pathway_results": ToolOutputProfile(
        tool_name="plot_pathway_results",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "enriched terms from a stored table, or per-spot pathway activity on the tissue",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_spatial_statistics": ToolOutputProfile(
        tool_name="plot_spatial_statistics",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "the spatial neighbour graph, or a statistic another tool computed",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_trajectory": ToolOutputProfile(
        tool_name="plot_trajectory",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a stored pseudotime on an embedding, on the tissue, or as gene trends",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_cell_communication": ToolOutputProfile(
        tool_name="plot_cell_communication",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "an INFERRED ligand-receptor result as a sender-receiver matrix or a ranking",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_spatial_3d": ToolOutputProfile(
        tool_name="plot_spatial_3d",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a reconstructed stack drawn in three dimensions -- the volume, its depth profile, or a value along an axis",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: which coordinate key, where the z came from, the section axis",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_section_grid": ToolOutputProfile(
        tool_name="plot_section_grid",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "one panel per section on one shared colour scale; it draws sections, it does not compare them statistically",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: which coordinate key, where the z came from, the section axis",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "plot_alignment_qc": ToolOutputProfile(
        tool_name="plot_alignment_qc",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "one adjacent pair before and after alignment -- a picture of somebody ELSE'S alignment, never a prediction of this tool's own",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: which coordinate key, where the z came from, the section axis",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "compose_figure": ToolOutputProfile(
        tool_name="compose_figure",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a contact sheet indexing figures that were each drawn separately",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "run_visualization_pipeline": ToolOutputProfile(
        tool_name="run_visualization_pipeline",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "the few figures a budgeted report decided were worth drawing",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
            OutputSpec(
                "post_analysis/tables/*.csv",
                "csv",
                "metadata",
                "the values behind a panel, written beside it -- never a prediction of this tool's own",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "export_visualization": ToolOutputProfile(
        tool_name="export_visualization",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*-export*.svg",
                "svg",
                "visualization",
                "the figure re-rendered as vector, with editable text",
            ),
            OutputSpec(
                "post_analysis/figures/*-export*.png", "png", "visualization", "the figure re-rendered as raster"
            ),
            OutputSpec(
                "post_analysis/figures/*-export*.pdf",
                "pdf",
                "visualization",
                "the figure as PDF, offered as a download because there is no viewer here",
            ),
            OutputSpec(
                "post_analysis/tables/*_values.csv",
                "csv",
                "metadata",
                "the numbers the figure was drawn from -- what turns a picture into a result somebody can check",
            ),
            OutputSpec("*_export.zip", "zip", "metadata", "the figure, its values and its record in one archive"),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
    "update_visualization": ToolOutputProfile(
        tool_name="update_visualization",
        task_type="visualization",
        outputs=[
            OutputSpec(
                "post_analysis/figures/*.png",
                "png",
                "visualization",
                "a revised version of an existing figure, drawn from its saved spec",
            ),
            OutputSpec(
                "post_analysis/figures/*.svg",
                "svg",
                "visualization",
                "the same figure as vector, when one was asked for",
            ),
            OutputSpec(
                "post_analysis/figures/*.figspec.json",
                "json",
                "metadata",
                "how the figure was drawn: the matrix slot, the transform, the colour limits, what was sampled -- what makes it reproducible and revisable",
            ),
            OutputSpec(
                "post_analysis/figures/*.figdata.npz",
                "npz",
                "intermediate",
                "the plotted values, frozen, so a presentation change re-renders without reopening the dataset",
            ),
            OutputSpec(
                "post_analysis/manifest.json",
                "json",
                "metadata",
                "the declaration without which no figure reaches the chat",
            ),
        ],
        notes=VIZ_NOTE,
    ),
}


# Names no portal exposes any more. Kept so a recorded run that names one still resolves through
# get_profile, and kept OUT of TOOL_PROFILES because detection scans that dict by filename: each of
# the first two shares its authoritative_output with the live tool (spaceflow_spatial_domains,
# spotgf_denoise), the tie made detection drop the tool name, and every SpaceFlow run resolved to no
# tool at all (hunt 2026-09-30, u29b-skills-config-7).
_RETIRED_PROFILES: dict[str, ToolOutputProfile] = {
    name: TOOL_PROFILES.pop(name)
    for name in (
        "spaceflow_identify_domains",
        "spotgf_detect_svg",
        "spiral_spatial_domains",
        "stereoscope_deconvolution",
    )
}


def get_profile(tool_name: str) -> ToolOutputProfile | None:
    """Get the output profile for a tool. Checks built-in, then retired names, then dynamic."""
    return TOOL_PROFILES.get(tool_name) or _RETIRED_PROFILES.get(tool_name) or _DYNAMIC_PROFILES.get(tool_name)


def get_all_profiles() -> dict[str, ToolOutputProfile]:
    """Get all registered tool output profiles: the built-ins, plus any registered at runtime.

    ``get_profile`` has always consulted ``_DYNAMIC_PROFILES``; this returned the built-ins alone, so a
    runtime profile would have been invisible to post-analysis's filename route even once registered
    (hunt 2026-09-30, u31-benchmarking-20). Built-ins win a name clash, as in ``get_profile``. With no
    runtime profile -- the case today -- this is ``TOOL_PROFILES`` itself, unchanged.
    """
    if not _DYNAMIC_PROFILES:
        return TOOL_PROFILES
    return {**_DYNAMIC_PROFILES, **TOOL_PROFILES}


# Dynamic profiles for user-created tools (registered at runtime)
_DYNAMIC_PROFILES: dict[str, ToolOutputProfile] = {}


def register_dynamic_profile(profile: ToolOutputProfile) -> None:
    """Register a tool output profile at runtime (for user-created tools).

    Dynamic profiles cannot override built-in profiles. Nothing in this repo calls this yet: a tool
    created at runtime (``tools_user/``) has no profile, so its category comes from name heuristics and
    its outputs are typed by content. Code that reasons about "profiles registered at runtime" is
    reasoning about this hook, not about tools that use it.
    """
    if profile.tool_name in TOOL_PROFILES or profile.tool_name in _RETIRED_PROFILES:
        return  # Built-in profiles always win
    _DYNAMIC_PROFILES[profile.tool_name] = profile


def get_prediction_key_candidates(tool_name: str) -> list[str]:
    """Get ordered list of prediction key candidates for a tool.

    Falls back to generic candidates if tool not in registry.
    """
    profile = TOOL_PROFILES.get(tool_name) or _RETIRED_PROFILES.get(tool_name)
    if profile and profile.prediction_key:
        return [profile.prediction_key]

    # Generic fallbacks by task type
    if profile:
        if profile.task_type == "spatial_clustering":
            return [
                "spatial_domain",
                "leiden",
                "louvain",
                "cluster",
                "domain",
                "mclust",
                "graphst_cluster",
                "deepst_domain",
                "prost_domain",
                "sedr_cluster",
                "miso_cluster",
                "stagate_domain",
            ]
        elif profile.task_type == "svg_detection":
            return ["Gene", "gene", "gene_name", "g", "feature"]
        elif profile.task_type == "deconvolution":
            return []  # Proportions are in the CSV matrix, not a column

    return ["cluster", "domain", "leiden", "louvain", "pred", "prediction"]


def get_tools_for_task(task_type: str) -> list[str]:
    """Get all tool names registered for a task type."""
    return [name for name, p in TOOL_PROFILES.items() if p.task_type == task_type]
