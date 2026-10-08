"""The shared vocabulary of follow-up actions: what each one is called, asks, and does.

:mod:`spatialomicsgym.postanalysis.next_step` proposes follow-up actions as snake_case tokens
(``run_marker_enrichment``). Until now those tokens were the whole story: the run report printed
them raw inside ``<code>`` tags, the web portal never showed them at all, and nothing anywhere said
HOW an action would actually be carried out. This module gives every action the three faces the
rest of the system needs:

- ``label``   -- how a human reads it (sentence case, biologist's words);
- ``prompt``  -- the ready-to-send user question that asks for it (biologist-voiced, no tool
  names, no file paths, no MCP vocabulary -- these strings become chat messages);
- ``binding`` -- how the agent should execute it: ``{"kind", "hint"}`` where ``kind`` is
  ``"mcp_tool"`` (run an analysis tool, picked via the tool recommender -- never a hardcoded
  registry string, so the empirical leaderboard keeps choosing), ``"code"`` (a concrete
  scanpy/numpy move in the REPL, for follow-ups no packaged tool covers), or ``"report"``
  (read what already exists and write for the user). The ``hint`` is one terse ASCII
  ACTION-FIRST line consumed by the follow-up prompt builder.

Everything here is data plus two tiny helpers, stdlib-only, so the webui may import it without
touching the heavy stack. The action tokens themselves are owned by ``next_step.py``; a test pins
that this table covers exactly the actions that module can emit.
"""

from __future__ import annotations

__all__ = [
    "ACTIONS",
    "NON_EXECUTABLE_ACTIONS",
    "hint_for_action",
    "humanize_action",
    "prompt_for_action",
]

#: Actions that must never be offered as a click-to-run next step or fed back into the ReAct loop:
#: they mark the END of a line of analysis. (``next_step.py`` re-exports this; the value lives here
#: so the webui can filter without importing the engine.)
NON_EXECUTABLE_ACTIONS = frozenset({"surface_unusable_result"})

ACTIONS: dict[str, dict] = {
    # ------------------------------------------------------------------ spatial_clustering
    "run_marker_enrichment": {
        "label": "Find marker genes for each domain",
        "prompt": "Which genes mark each of the spatial domains you just found?",
        "task_type": "spatial_clustering",
        "binding": {
            "kind": "code",
            "hint": "Load the clustered result, run scanpy rank_genes_groups on the domain labels, report top markers per domain.",
        },
    },
    "annotate_domains_from_markers": {
        "label": "Annotate the domains biologically",
        "prompt": "What tissue structure or cell types does each spatial domain correspond to?",
        "task_type": "spatial_clustering",
        "binding": {
            "kind": "code",
            "hint": "Score each domain's top markers against known cell-type/layer signatures (see the single-cell annotation know-how) and name every domain.",
        },
    },
    "test_domain_spatial_coherence": {
        "label": "Check the domains are spatially coherent",
        "prompt": "Are the spatial domains coherent tissue regions, or salt-and-pepper noise?",
        "task_type": "spatial_clustering",
        "binding": {
            "kind": "code",
            "hint": "From the spatial coordinates, compute per-spot neighbor label agreement (k-nearest neighbors) and flag fragmented domains.",
        },
    },
    # ------------------------------------------------------------------ svg_detection
    "run_functional_enrichment_on_svgs": {
        "label": "Run functional enrichment on the spatial genes",
        "prompt": "What biological processes are enriched among the spatially variable genes?",
        "task_type": "svg_detection",
        "binding": {
            "kind": "mcp_tool",
            "hint": (
                "Run the pathway-enrichment tool on the ranked SVG table: pass the CSV path as gene_table with its "
                "gene and score columns, the species, and collections hallmark,go_bp,reactome; report the FDR<0.05 "
                "terms per collection and their leading-edge genes."
            ),
        },
    },
    "cluster_svgs_into_spatial_modules": {
        "label": "Group the spatial genes into co-expressed modules",
        "prompt": "Do the spatially variable genes fall into co-expressed spatial modules?",
        "task_type": "svg_detection",
        "binding": {
            "kind": "code",
            "hint": "Cluster the SVG expression profiles across spots (correlation + KMeans or leiden) and plot each module's mean pattern on the tissue.",
        },
    },
    # ------------------------------------------------------------------ deconvolution
    "map_dominant_celltype_per_spot": {
        "label": "Map the dominant cell type per spot",
        "prompt": "Where does each cell type dominate across the tissue?",
        "task_type": "deconvolution",
        "binding": {
            "kind": "code",
            "hint": "Argmax the per-spot proportions matrix and scatter the spatial coordinates colored by dominant cell type.",
        },
    },
    "run_celltype_colocalization": {
        "label": "Test which cell types co-locate",
        "prompt": "Which cell types co-occur in the same spots more often than expected?",
        "task_type": "deconvolution",
        "binding": {
            "kind": "code",
            "hint": "Correlate cell-type proportion columns across spots (Spearman) and report the strongest positive and negative pairs.",
        },
    },
    "compare_against_a_second_deconvolution_tool": {
        "label": "Cross-check with a second deconvolution method",
        "prompt": "Do the estimated cell-type proportions hold up under a second deconvolution method?",
        "task_type": "deconvolution",
        "binding": {
            "kind": "mcp_tool",
            "hint": "Pick a DIFFERENT deconvolution tool via the tool recommendation, run it on the same inputs, and correlate per-spot proportions between the two runs.",
        },
    },
    # ------------------------------------------------------------------ cell_communication
    "rank_ligand_receptor_pairs_by_spatial_support": {
        "label": "Rank ligand-receptor pairs by spatial support",
        "prompt": "Which ligand-receptor interactions have the strongest spatial support?",
        "task_type": "cell_communication",
        "binding": {
            "kind": "code",
            "hint": "Sort the interaction table by the tool's own score column and check top pairs' ligand and receptor are expressed in adjacent regions.",
        },
    },
    # ------------------------------------------------------------------ alignment
    "measure_registration_error": {
        "label": "Measure the registration error",
        "prompt": "How large is the residual error after aligning the sections?",
        "task_type": "alignment",
        "binding": {
            "kind": "code",
            "hint": "Compute residual distances between matched coordinates before vs after alignment and report median/90th percentile.",
        },
    },
    "compare_the_aligned_sections_region_by_region": {
        "label": "Compare the aligned sections region by region",
        "prompt": "Now that the sections are aligned, which regions differ between them?",
        "task_type": "alignment",
        "binding": {
            "kind": "code",
            "hint": "Bin both sections on the shared coordinate frame and compare per-bin composition or expression between sections.",
        },
    },
    # ------------------------------------------------------------------ imputation
    "validate_imputed_genes_against_held_out_measurements": {
        "label": "Validate imputed genes against held-out measurements",
        "prompt": "How well do the imputed gene values match measurements the model never saw?",
        "task_type": "imputation",
        "binding": {
            "kind": "code",
            "hint": "Hold out measured genes, re-impute them, and report per-gene correlation between imputed and measured values.",
        },
    },
    "test_whether_the_imputed_genes_are_spatially_structured": {
        "label": "Test whether the imputed genes are spatially structured",
        "prompt": "Are the newly imputed genes spatially structured, or just noise?",
        "task_type": "imputation",
        "binding": {
            "kind": "mcp_tool",
            "hint": "Run a spatially-variable-gene detection tool (picked via the tool recommendation) on the imputed matrix and compare its calls against the measured genes.",
        },
    },
    # ------------------------------------------------------------------ trajectory
    "plot_gene_trends_along_the_trajectory": {
        "label": "Plot gene trends along the trajectory",
        "prompt": "How do the key genes change along the inferred trajectory?",
        "task_type": "trajectory",
        "binding": {
            "kind": "code",
            "hint": "Order spots by pseudotime and plot smoothed expression of the top trajectory-associated genes.",
        },
    },
    "map_pseudotime_onto_the_tissue": {
        "label": "Map pseudotime onto the tissue",
        "prompt": "Where does the inferred trajectory run through the tissue itself?",
        "task_type": "trajectory",
        "binding": {
            "kind": "code",
            "hint": "Scatter the spatial coordinates colored by pseudotime and describe the spatial gradient.",
        },
    },
    # ------------------------------------------------------------------ bookkeeping
    "summarize_findings_for_the_user": {
        "label": "Summarize the findings",
        "prompt": "Please summarize what this analysis found, in plain language.",
        "task_type": None,
        "binding": {
            "kind": "report",
            "hint": "Read the run's findings, figures and warnings and write a short interpretation; add no new computation.",
        },
    },
    "verify_flagged_result": {
        "label": "Verify the flagged result",
        "prompt": "Something in the last run was flagged as suspicious -- can you check whether the result is trustworthy?",
        "task_type": None,
        "binding": {
            "kind": "code",
            "hint": "Re-open the flagged output and test the reviewer's specific concern (do values vary, do proportions sum to 1, does the score column carry signal) before trusting it.",
        },
    },
    "retry_failed_analyses": {
        "label": "Retry the failed analyses",
        "prompt": "Some analyses failed -- please retry them, fixing what went wrong first.",
        "task_type": None,
        "binding": {
            "kind": "mcp_tool",
            "hint": "Read the error in the run log, fix the named argument or input, and call the same tool again -- do not switch tools on the first failure.",
        },
    },
    "surface_unusable_result": {
        "label": "Explain why the result is unusable",
        "prompt": "The last run's output looks unusable -- explain what went wrong and what to do instead.",
        "task_type": None,
        "binding": {
            "kind": "report",
            "hint": "State plainly what failed validation and which input or setting to change; quote no numbers from the unusable output.",
        },
    },
}


def humanize_action(action: str) -> str:
    """The human label for an action token; unknown tokens de-snake instead of leaking code-case."""
    entry = ACTIONS.get(action)
    if entry:
        return entry["label"]
    text = str(action).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def prompt_for_action(action: str) -> str:
    """The ready-to-send user question asking for this action (safe fallback for unknown tokens)."""
    entry = ACTIONS.get(action)
    if entry:
        return entry["prompt"]
    return f"On the analysis you just ran, please {humanize_action(action).lower()}."


def hint_for_action(action: str) -> str | None:
    """The one-line execution hint for the follow-up prompt builder; None for unknown tokens."""
    entry = ACTIONS.get(action)
    return entry["binding"]["hint"] if entry else None
