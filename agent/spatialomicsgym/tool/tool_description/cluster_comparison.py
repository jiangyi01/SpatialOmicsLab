description = [
    {
        "name": "rank_label_abundance_changes",
        "description": (
            "Rank a vocabulary's OWN terms by how much their abundance shifts between conditions, "
            "in one comparison rather than one term at a time. score_label_vocabulary says what "
            "each cluster is and compare_cluster_abundance says which clusters moved; this joins "
            "them, so a top-N answer is decided by a comparison instead of by the order terms "
            "happened to be checked in. Abundance is the within-condition fraction of the cells "
            "held by the clusters a term won. Two distinctions travel with the ranking: a term "
            "that was measured and won no cluster is ranked at zero and says where it came "
            "closest and to what it lost; a term that could not be measured at all is NOT ranked, "
            "because its zero would be an artifact of never being scored, and comes back "
            "separately with the reason (off-panel markers and invariant markers need different "
            "next steps). Also reports the share of each condition the ranking does not speak "
            "for, the orderings resting on the narrowest tie calls, and any term defined over "
            "other populations that needs composition_key to be measurable."
        ),
        "required_parameters": [
            {
                "name": "adata_or_path",
                "type": "str",
                "description": "Path to an .h5ad file (an in-memory AnnData is also accepted).",
                "default": None,
            },
            {
                "name": "cluster_key",
                "type": "str",
                "description": "Column in adata.obs holding cluster or spatial-domain labels.",
                "default": None,
            },
            {
                "name": "condition_key",
                "type": "str",
                "description": "Column in adata.obs holding the condition, genotype or timepoint.",
                "default": None,
            },
            {
                "name": "vocabulary",
                "type": "Any",
                "description": (
                    "The term vocabulary: {term: [marker, ...]}, a glossary document, or a path "
                    "to one -- whatever score_label_vocabulary accepts."
                ),
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to write the ranking to as CSV, in the same order.",
                "default": None,
            },
            {
                "name": "tie_tolerance_pct",
                "type": "float",
                "description": (
                    "Percentage points within which two conditions render as tied ('~'). If the "
                    "task states a tie rule, pass that number; the output records whether the "
                    "tolerance was the caller's or the default."
                ),
                "default": None,
            },
            {
                "name": "composition_key",
                "type": "str",
                "description": (
                    "adata.obs column of population labels, for vocabulary terms defined over "
                    "other populations rather than by markers. Without it those terms are named "
                    "in terms_needing_a_composition_key and left out of the ranking."
                ),
                "default": None,
            },
            {
                "name": "layer",
                "type": "str",
                "description": "adata.layers key to score on instead of adata.X.",
                "default": None,
            },
        ],
    },
    {
        "name": "compare_cluster_abundance",
        "description": (
            "Rank clusters by how much their abundance shifts between conditions, using "
            "within-condition fractions (of the cells in one condition, the share in this "
            "cluster) so that unequal cell counts per condition do not drive the ranking. "
            "Returns, per cluster, the fraction in each condition, the spread in percentage "
            "points, log2 of the max/min ratio, and an explicit ordering string such as "
            "'treated > control ~ vehicle' where '~' marks conditions within the tie tolerance. Sorted by "
            "spread, widest first. Every '>' and '~' in that string is a claim about the tie "
            "tolerance rather than about the data alone, so each cluster also returns "
            "ordering_margins: per adjacent pair, the two fractions, the gap in percentage "
            "points, the relation that gap produced, and how far the gap sits from the tolerance. "
            "Use those to tell a settled call from one that would reverse under a slightly "
            "different rule -- under a 5-point rule a 0.2-point gap and a 4.9-point gap render "
            "identically. The output also reports whether the tolerance was supplied by the "
            "caller or taken from the default. Use this to answer which populations changed "
            "between conditions, and in what direction, from measured composition rather than "
            "from marker scores."
        ),
        "required_parameters": [
            {
                "name": "adata_or_path",
                "type": "str",
                "description": "Path to an .h5ad file (an in-memory AnnData is also accepted).",
                "default": None,
            },
            {
                "name": "cluster_key",
                "type": "str",
                "description": "Column in adata.obs holding cluster or spatial-domain labels.",
                "default": None,
            },
            {
                "name": "condition_key",
                "type": "str",
                "description": "Column in adata.obs holding the condition, genotype or timepoint.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to write the full table to as CSV.",
                "default": None,
            },
            {
                "name": "tie_tolerance_pct",
                "type": "float",
                "description": (
                    "Two conditions whose fractions differ by no more than this many percentage "
                    "points are rendered as tied ('~') rather than ordered ('>'). If the task, a "
                    "schema or a notation legend states when two values count as tied, pass that "
                    "number: leaving this unset does not mean no rule applies, it means the "
                    "orderings were rendered under a default chosen without reference to the "
                    "task, and the output will say so."
                ),
                "default": None,
            },
            {
                "name": "permitted_orderings",
                "type": "Any",
                "description": (
                    "The closed list of ordering patterns the task will accept -- or the whole "
                    "vocabulary document that holds one, or a path to it. 'A ~ B' and 'B ~ A' are "
                    "one claim with two spellings, and a task that lists only one of them will "
                    "reject a correct finding rendered the other way. Pass the task's list and "
                    "each ordering is reported in the listed spelling whenever the listed spelling "
                    "makes the same claim; the measured string is kept beside it, and an ordering "
                    "the list cannot express is reported as measured rather than rounded to the "
                    "nearest listed pattern. Leaving it unset changes nothing."
                ),
                "default": None,
            },
        ],
    },
    {
        "name": "rank_cluster_markers",
        "description": (
            "Top differential genes per cluster via scanpy.tl.rank_genes_groups. Answers 'what is "
            "actually high in this cluster' from the data, rather than 'where are the genes I "
            "expected', so a population you were not already thinking of can still be found. Run "
            "this before proposing labels for clusters. Normalises and log-transforms automatically "
            "if the matrix looks like raw counts. Returns JSON with one line per cluster in "
            "'markers' -- its top genes, best first, as 'gene log2fc/score', as many per cluster as "
            "fit half of one observation ('genes_shown_per_cluster') -- and always writes the full "
            "per-gene table (score, log2fc, pval_adj) to the CSV at 'csv_path'."
        ),
        "required_parameters": [
            {
                "name": "adata_or_path",
                "type": "str",
                "description": "Path to an .h5ad file (an in-memory AnnData is also accepted).",
                "default": None,
            },
            {
                "name": "cluster_key",
                "type": "str",
                "description": "Column in adata.obs holding cluster or spatial-domain labels.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "n_genes",
                "type": "int",
                "description": "How many top genes to rank per cluster; all of them go to the CSV.",
                "default": 25,
            },
            {
                "name": "output_dir",
                "type": "str",
                "description": (
                    "Directory for the full per-gene CSV. Default: rank_cluster_markers/ under the run's work root "
                    "(SOG_WORK_DIR when it is set), where the MCP tools write theirs."
                ),
                "default": None,
            },
            {
                "name": "method",
                "type": "str",
                "description": "Test passed to scanpy.tl.rank_genes_groups: 'wilcoxon', 't-test' or 'logreg'.",
                "default": "wilcoxon",
            },
        ],
    },
    {
        "name": "score_label_vocabulary",
        "description": (
            "Score EVERY term in a controlled vocabulary against EVERY cluster, and return the "
            "whole table. A term's score for a cluster is the mean, over the markers the assay "
            "can actually measure, of that gene's expression z-scored across clusters. Because "
            "scoring is exhaustive, the output shows what lost as well as what won, so a term "
            "that was never considered cannot be reported as one that was ruled out. Also "
            "returns the denominator behind every score (how many listed markers are present "
            "and variable), the terms no marker in the assay can evidence at all, the "
            "best/runner-up assignment per cluster with the margin between them, and the "
            "markers claimed by more than one term. It also audits the assignment in cells rather "
            "than in clusters: the share of the sample each winning term ends up holding, the "
            "thinnest margin and weakest score it won on, the terms this assay can measure that "
            "won no cluster at all and where each came closest, and the term pairs whose usable "
            "markers nest inside one another. A term holding a large share on narrow margins is a "
            "merged catch-all, and a measurable term with zero cells is usually a clustering "
            "resolution too coarse to separate it rather than a population that is absent. Every "
            "row also carries the vocabulary's own "
            "definition of the term verbatim, because whether a term answers the question that "
            "was actually asked is decided from its definition rather than from its score, and a "
            "term defined by composition rather than by markers has no score to decide it with. "
            "Use this whenever a task supplies a vocabulary of candidate labels."
        ),
        "required_parameters": [
            {
                "name": "adata_or_path",
                "type": "str",
                "description": "Path to an .h5ad file (an in-memory AnnData is also accepted).",
                "default": None,
            },
            {
                "name": "cluster_key",
                "type": "str",
                "description": "Column in adata.obs holding cluster or spatial-domain labels.",
                "default": None,
            },
            {
                "name": "vocabulary",
                "type": "dict",
                "description": (
                    "Mapping of term to its marker genes, {term: [gene, ...]} or "
                    "{term: {'markers': [gene, ...]}}. A path to a JSON file holding one is also "
                    "accepted. Pass every candidate term, not a shortlist."
                ),
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_dir",
                "type": "str",
                "description": "Directory to write the full term x cluster table to as CSV.",
                "default": None,
            },
            {
                "name": "layer",
                "type": "str",
                "description": "adata.layers key to score on instead of adata.X.",
                "default": None,
            },
            {
                "name": "composition_key",
                "type": "str",
                "description": (
                    "adata.obs column holding cell-type or population labels. A glossary term "
                    "defined over other populations -- 'regions where Neuronal and Proliferating "
                    "cells co-localize' -- lists no genes, so marker scoring returns NaN for it "
                    "exactly as it does for a term whose markers are all off-panel, and the two "
                    "call for completely different actions. Pass this column and such a term is "
                    "scored from the share of each cluster made of the populations it names, on "
                    "the same scale as the marker terms. Terms that need it are named in "
                    "terms_needing_a_composition_key whether or not it is passed."
                ),
                "default": None,
            },
        ],
    },
]
