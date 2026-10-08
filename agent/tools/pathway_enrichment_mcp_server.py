#!/usr/bin/env python3
"""Pathway enrichment MCP wrapper for SpatialOmicsLab: gene-set ORA / GSEA-prerank on a ranked
gene table, and per-spot pathway activity (PROGENy / MSigDB) on a spatial h5ad, via decoupler
(gseapy optional). Two functions, one worker (``--task``), one conda env (``pathway_enrichment``)."""

from typing import Any

from base_mcp import create_mcp, get_worker_paths, run_worker_cli

TOOL_NAME = "pathway-enrichment"
WORKER_PYTHON, WORKER_SCRIPT = get_worker_paths(
    "PATHWAY_ENRICHMENT",
    "/opt/conda/envs/pathway_enrichment/bin/python",
    "/workspace/epic-fermat/agent/tools/pathway_enrichment_worker.py",
)

mcp = create_mcp(TOOL_NAME)

#: An inline gene list larger than this is refused BEFORE dispatch. Linux caps one argument at
#: 128 KB (MAX_ARG_STRLEN) and a 20k-symbol list is ~180 KB; base_mcp would then report E2BIG
#: after the fact. Said here, with the file remedy, so the model never pays for the failed launch.
INLINE_GENES_MAX_CHARS = 60_000


@mcp.tool()
def run_pathway_enrichment(
    gene_table: str = "",
    gene_column: str = "gene",
    score_column: str = "",
    genes: str = "",
    h5ad_path: str = "",
    h5ad_key: str = "rank_genes_groups",
    group: str = "",
    top_n: int = 200,
    species: str = "human",
    collections: str = "hallmark,go_bp,reactome",
    method: str = "ora",
    gsea_backend: str = "decoupler",
    background: str = "",
    min_set_size: int = 5,
    max_set_size: int = 500,
    permutations: int = 1000,
    seed: int = 0,
    license: str = "academic",
    gene_sets_dir: str = "",
    output_dir: str = "",
) -> dict[str, Any]:
    """
    Gene-set enrichment on a gene list: over-representation (Fisher, BH-adjusted) on the top
    `top_n` genes of the ranking and, when `score_column` names a numeric column, GSEA-prerank on
    the whole ranked table (decoupler by default, gseapy via gsea_backend=gseapy).

    Input is exactly ONE of `gene_table` (a CSV/TSV with a gene-symbol column -- the tables the SVG
    tools write), `h5ad_path` + `h5ad_key` (a rank_genes_groups group or a var column), or `genes`
    (a short inline list). Gene SYMBOLS only; mouse symbols are matched to the human collections
    case-insensitively and the match rate is reported. Collections are MSigDB (hallmark, go_bp,
    go_mf, go_cc, reactome, wikipathways; kegg opt-in) or a local .gmt path; fetched from
    omnipathdb.org once per machine and cached after that.

    Output: ora_<collection>.csv, gsea_<collection>.csv (when a score column was given),
    enrichment_summary.csv, enrichment_dotplot.png, input_genes_used.txt.
    """
    if len(genes or "") > INLINE_GENES_MAX_CHARS:
        return {
            "status": "error",
            "tool": TOOL_NAME,
            "error": (
                f"the inline gene list is {len(genes)} characters, which is larger than the operating system "
                "can pass to the worker as one argument."
            ),
            "diagnostic": (
                "Write the list to a file (one symbol per line, or a CSV with a gene column) and pass it as "
                "gene_table. This tool is installed correctly and does not need reprovisioning."
            ),
        }
    args = [
        "--task",
        "enrichment",
        "--gene-table",
        gene_table,
        "--gene-column",
        gene_column,
        "--score-column",
        score_column,
        "--genes",
        genes,
        "--h5ad-path",
        h5ad_path,
        "--h5ad-key",
        h5ad_key,
        "--group",
        group,
        "--top-n",
        str(int(top_n)),
        "--species",
        species,
        "--collections",
        collections,
        "--method",
        method,
        "--gsea-backend",
        gsea_backend,
        "--background",
        background,
        "--min-set-size",
        str(int(min_set_size)),
        "--max-set-size",
        str(int(max_set_size)),
        "--permutations",
        str(int(permutations)),
        "--seed",
        str(int(seed)),
        "--license",
        license,
        "--gene-sets-dir",
        gene_sets_dir,
        "--output-dir",
        output_dir,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


@mcp.tool()
def run_pathway_activity(
    data_path: str,
    output_dir: str = "",
    collections: str = "progeny",
    species: str = "human",
    method: str = "ulm",
    top_progeny: int = 500,
    n_up: int = 0,
    min_set_size: int = 5,
    layer: str = "",
    normalize: str = "auto",
    group_key: str = "",
    top_k_maps: int = 6,
    seed: int = 0,
    license: str = "academic",
    gene_sets_dir: str = "",
) -> dict[str, Any]:
    """
    Score pathway activity in every spot/cell of a spatial h5ad with decoupler: PROGENy (14
    weighted signalling pathways, method=ulm) and/or MSigDB hallmark or any collection / .gmt.
    method=ora tests each spot's top expressed genes instead. Raw counts are normalize_total +
    log1p'd (normalize=auto touches only integer counts and the payload says what happened);
    obsm['spatial'] is used for the tissue map only. `group_key` (an obs column of domains or
    cell types) adds per-group means and decoupler's rank_sources_groups t-tests.

    Output: pathway_activity.h5ad (obsm['<method>_estimate'] / ['<method>_pvals']),
    pathway_activity_scores.csv, spatial_pathway_activity.png, and with group_key
    pathway_activity_by_group.csv + pathway_activity_group_tests.csv.
    """
    args = [
        "--task",
        "activity",
        "--data-path",
        data_path,
        "--output-dir",
        output_dir,
        "--collections",
        collections,
        "--species",
        species,
        "--method",
        method,
        "--top-progeny",
        str(int(top_progeny)),
        "--n-up",
        str(int(n_up)),
        "--min-set-size",
        str(int(min_set_size)),
        "--layer",
        layer,
        "--normalize",
        normalize,
        "--group-key",
        group_key,
        "--top-k-maps",
        str(int(top_k_maps)),
        "--seed",
        str(int(seed)),
        "--license",
        license,
        "--gene-sets-dir",
        gene_sets_dir,
    ]
    return run_worker_cli(TOOL_NAME, WORKER_PYTHON, WORKER_SCRIPT, args)


if __name__ == "__main__":
    mcp.run()
