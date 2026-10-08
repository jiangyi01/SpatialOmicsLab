"""Comparative analysis primitives: which clusters shift between conditions, and what they are.

SpatialOmicsGym already has strong *ingestion* primitives -- ``diagnose_spatial_data`` and
``run_spatial_pipeline`` take an arbitrary vendor drop and hand back an analysis-ready h5ad. What
follows ingestion had none, so every comparative study re-derived it by hand in the REPL: cluster,
work out which clusters moved between conditions, decide what each cluster is, then write the
finding down. Hand-derived, three failure modes recur and none of them leaves a trace in the
output:

*Attrition.* A controlled vocabulary of forty terms is read once, narrowed to the handful that came
to mind, and the rest are never scored against anything. The write-up then reports the survivors as
if they had won a comparison that never happened, and a term that was never in contention is
indistinguishable from one that was ruled out. :func:`score_label_vocabulary` scores **every** term
against **every** cluster and returns a row for each, so the table shows what lost as well as what
won.

*Silent denominators.* A term whose markers are mostly absent from the panel scores low for a
reason that has nothing to do with the tissue. Every score here carries the denominator it was
computed over, and terms with no measurable marker at all are listed separately as
``unevidenceable`` rather than being reported as absent.

*Ranking by prior.* Asked which populations changed, it is easy to answer with the ones that are
interesting rather than the ones that moved. :func:`compare_cluster_abundance` ranks by measured
within-condition composition and renders the ordering as an explicit string, ties included.

Typical workflow, continuing the one in :mod:`spatialomicsgym.tool.spatial_pipeline`:

  5. rank_cluster_markers(h5ad, "leiden")            -> what each cluster is, from its OWN genes
  6. score_label_vocabulary(h5ad, "leiden", vocab)   -> every term x every cluster, with margins
  7. compare_cluster_abundance(h5ad, "leiden", "condition")
                                                     -> which clusters moved, and in what order

Nothing here knows anything about any particular tissue, panel or vocabulary: the caller supplies
the terms and their markers, and the functions supply the arithmetic and the audit trail.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatialomicsgym.paths import tool_output_root
from spatialomicsgym.utils.execution import OBSERVATION_CHARS

__all__ = [
    "rank_label_abundance_changes",
    "compare_cluster_abundance",
    "rank_cluster_markers",
    "score_label_vocabulary",
]

#: Two conditions whose within-condition fractions differ by no more than this many percentage
#: points are rendered as tied (``~``) rather than ordered (``>``). Five points is the tolerance
#: most controlled vocabularies that define a tie at all settle on; pass your own when the study
#: defines it differently.
DEFAULT_TIE_TOLERANCE_PCT = 5.0

_TARGET_SUM = 1e4

#: Gene symbols as controlled-vocabulary glossaries write them: ``COL1A2+, DCN+``. Matching this
#: convention lets a term be scored straight from the glossary that defines it, rather than from a
#: marker list the caller has to transcribe by hand -- and a transcription step is a place terms go
#: missing.
_MARKER_IN_PROSE = re.compile(r"\b([A-Z][A-Z0-9]{1,14}(?:-[A-Z0-9]{1,6})?)\+")

#: The same convention in any other casing -- mouse ``Gad1+``, ``P2ry12+``, ``S100a8+``, ``H2-Aa+``
#: and flow-style ``Ly6G+``. Only the all-caps form was read, so every term of a mouse glossary came
#: back "no markers are listed", which was false, and the run ended ``no_measurable_terms`` blaming a
#: reshaped vocabulary (hunt 2026-09-30, u22-spatial-pipeline-6). The first repair still wanted a
#: lower-case second character, which missed every symbol whose second character is a digit, and its
#: ``\b`` let a match start after a hyphen, so ``H2-Aa+`` gave a made-up ``Aa``. A match may not
#: start inside a hyphenated symbol now, and only a match holding a lower-case letter is taken
#: here: an all-caps one is ``_MARKER_IN_PROSE``'s, whose reading is unchanged. A prose word followed
#: by ``+`` (``High+``) is read too, as it was before; it lands in ``missing_from_assay`` and changes
#: no score. Ion charges (``Ca2+``, ``Na+``) fit the shape and are excluded by name, not by shape: a
#: shape rule (two letters and a digit) also drops Cd4, Il6 and Il2.
_MIXED_CASE_MARKER_IN_PROSE = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9]{1,14}(?:-[A-Za-z0-9]{1,6})?)\+")
_ION_CHARGES = frozenset(
    {"Na", "Li", "Cs", "Rb", "Ca", "Ca2", "Mg", "Mg2", "Zn", "Zn2", "Cu", "Cu2", "Fe2", "Fe3", "Mn2", "Co2", "Ni2"}
)


def _markers_in_prose(text: str) -> list[str]:
    """Marker symbols written ``SYMBOL+`` in glossary prose, in the order they appear, any casing."""
    found = [(m.start(), m.group(1)) for m in _MARKER_IN_PROSE.finditer(text)]
    found += [
        (m.start(), m.group(1))
        for m in _MIXED_CASE_MARKER_IN_PROSE.finditer(text)
        if m.group(1) not in _ION_CHARGES and any(c.islower() for c in m.group(1))
    ]
    return list(dict.fromkeys(symbol for _, symbol in sorted(found)))


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _load(adata_or_path: Any):
    """Accept a path or an in-memory AnnData, so this composes mid-analysis as well as standalone."""
    if isinstance(adata_or_path, (str, Path)):
        import anndata

        path = Path(adata_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {adata_or_path}")
        return anndata.read_h5ad(path)
    return adata_or_path


def _err(message: str, **extra: Any) -> str:
    return json.dumps({"status": "error", "message": message, **extra}, indent=2)


def _require_obs(adata, key: str, what: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(f"{what} '{key}' not found in adata.obs. Available: {sorted(adata.obs.columns)[:25]}")


def _drop_unlabelled(adata, *keys: str):
    """``(adata, n)``: the cells with a label in every one of ``keys``, and how many had none.

    Every crosstab and cluster mean here reads its labels through ``astype(str)``, which renders a
    missing label as the string ``"nan"`` -- so a cell with no condition became a condition called
    "nan" that was ranked into every ordering, and because the permitted list must name exactly the
    conditions present, it silently switched the task's vocabulary off (hunt 2026-09-30,
    u22-spatial-pipeline-5). A missing label is not a group; the repo's rule, pinned in
    ``test_an_unassigned_spot_is_not_a_cluster.py``, drops it and says how many. Untouched -- the
    same object, no copy -- when every cell is labelled.
    """
    keep = np.ones(adata.n_obs, dtype=bool)
    for key in keys:
        keep &= adata.obs[key].notna().to_numpy()
    n_dropped = int((~keep).sum())
    return (adata[keep] if n_dropped else adata), n_dropped


def _looks_like_counts(matrix) -> bool:
    """Raw counts or already normalised? Sampled, because the whole matrix may be out of core."""
    import scipy.sparse as sp

    sub = matrix[: min(1000, matrix.shape[0])]
    # A dense ndarray has a `.data` attribute too -- its raw buffer, not its values -- so ask the
    # sparse library which kind this is rather than duck-typing on the attribute name.
    data = sub.data if sp.issparse(sub) else np.asarray(sub).ravel()
    if data.size == 0:
        return False
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return False
    return bool(np.all(finite >= 0) and np.allclose(finite, np.rint(finite)))


def _light_adata(adata, cluster_key: str, layer: str | None = None):
    """A minimal AnnData carrying only the matrix, the cluster column and the gene names.

    scanpy wants an AnnData, but ``adata.copy()`` on a spatial object also duplicates every image,
    graph and embedding it is carrying. On the objects these primitives are for that is several
    gigabytes of copy to run one test, so rebuild the small part instead.
    """
    import anndata

    matrix = adata.layers[layer] if layer else adata.X
    if _looks_like_counts(matrix):
        # Normalisation below is in place, so counts have to be copied first. Already-normalised
        # input needs no copy, which on a pipeline output is the difference between a few gigabytes
        # of duplication and none.
        matrix = matrix.copy()
    obs = pd.DataFrame(
        {cluster_key: adata.obs[cluster_key].astype(str).to_numpy()},
        index=adata.obs_names.astype(str),
    )
    var = pd.DataFrame(index=adata.var_names.astype(str))
    return anndata.AnnData(X=matrix, obs=obs, var=var)


def _normalize_log1p(work) -> None:
    """Normalise and log-transform in place, but only if the matrix still looks like raw counts."""
    if not _looks_like_counts(work.X):
        return
    import scanpy as sc

    sc.pp.normalize_total(work, target_sum=_TARGET_SUM)
    sc.pp.log1p(work)


def _cluster_mean_expression(adata, cluster_key: str, layer: str | None) -> pd.DataFrame:
    """Mean expression per cluster, on log-normalised values, as clusters x genes.

    Computed with an indicator matmul rather than a per-cluster subset loop: on a half-million cell
    matrix the loop is what turns this from seconds into minutes.
    """
    import scipy.sparse as sp

    work = _light_adata(adata, cluster_key, layer)
    _normalize_log1p(work)

    labels = work.obs[cluster_key].to_numpy()
    clusters = sorted(pd.unique(labels).tolist())
    position = {name: i for i, name in enumerate(clusters)}
    rows = np.fromiter((position[x] for x in labels), dtype=np.int64, count=labels.shape[0])

    indicator = sp.csr_matrix(
        (np.ones(rows.shape[0], dtype=np.float64), (rows, np.arange(rows.shape[0]))),
        shape=(len(clusters), labels.shape[0]),
    )
    sums = indicator @ work.X
    sums = np.asarray(sums.todense() if sp.issparse(sums) else sums, dtype=np.float64)
    counts = np.asarray(indicator.sum(axis=1)).ravel()
    return pd.DataFrame(sums / np.maximum(counts, 1)[:, None], index=clusters, columns=list(work.var_names))


def _order_string(fractions: pd.Series, tie_tolerance_pct: float) -> str:
    """Render conditions high-to-low, with '~' between neighbours that are within tolerance.

    The tie relation is deliberately applied between *neighbours* only. Tie-ness is not transitive
    -- a within tolerance of b and b within tolerance of c does not put a within tolerance of c --
    so collapsing a whole run into one equivalence class would assert something the numbers do not
    say.
    """
    ordered = fractions.sort_values(ascending=False)
    names = [str(n) for n in ordered.index]
    if not names:
        return ""
    parts = [names[0]]
    for i in range(1, len(names)):
        gap_pct = abs(float(ordered.iloc[i - 1]) - float(ordered.iloc[i])) * 100.0
        parts.append("~" if gap_pct <= tie_tolerance_pct else ">")
        parts.append(names[i])
    return " ".join(parts)


def _ordering_margins(fractions: pd.Series, tie_tolerance_pct: float) -> list[dict[str, Any]]:
    """The adjacent gaps that produced the ordering string, and how close each is to flipping.

    The rendered string says ``A > B`` or ``A ~ B`` but not by how much, and those are very
    different claims: a gap of 0.2 points and a gap of 4.9 points both render ``~`` under a
    5-point rule, yet only the first would survive a slightly stricter one. A reader given only
    the string cannot tell a settled call from a coin-flip, so each relation is reported with the
    gap behind it and the distance that gap would have to move for the relation to flip.

    Only *adjacent* pairs appear, for the reason :func:`_order_string` renders only adjacent ties:
    tie-ness is not transitive, so a gap between non-neighbours is not a relation the string makes.
    """
    ordered = fractions.sort_values(ascending=False)
    names = [str(n) for n in ordered.index]
    out: list[dict[str, Any]] = []
    for i in range(1, len(names)):
        higher, lower = float(ordered.iloc[i - 1]), float(ordered.iloc[i])
        gap = abs(higher - lower) * 100.0
        out.append(
            {
                "higher": names[i - 1],
                "lower": names[i],
                "fraction_higher": round(higher, 6),
                "fraction_lower": round(lower, 6),
                "gap_pct_points": round(gap, 4),
                "relation": "~" if gap <= tie_tolerance_pct else ">",
                # The relation flips exactly when the tolerance crosses the gap, so this is the
                # whole margin of safety in one number.
                "margin_to_flip_pct_points": round(abs(gap - tie_tolerance_pct), 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Spelling an ordering the way the task's own vocabulary lists it
# ---------------------------------------------------------------------------
#: The relation symbols an ordering string is built from. ``>`` is strict, ``~`` is "within the
#: tie tolerance". Both are rendered by :func:`_order_string`.
_ORDERING_SPLIT = re.compile(r"\s*([>~])\s*")


def _parse_ordering(ordering: str) -> tuple[list[list[str]], list[str]] | None:
    """``"A > B ~ C"`` -> ``([["A"], ["B", "C"]], [">"])``: the tie runs, and the strict gaps.

    A tie run is a maximal group of names joined by ``~``. Runs are what the rewrite below is
    allowed to touch, because a run is the only part of the string whose internal order the
    measurement did not decide. ``None`` when the string is not an ordering at all.
    """
    parts = [p for p in _ORDERING_SPLIT.split(str(ordering).strip()) if p != ""]
    if not parts or len(parts) % 2 == 0:
        return None
    names, relations = parts[0::2], parts[1::2]
    if any(not n or n in {">", "~"} for n in names) or any(r not in {">", "~"} for r in relations):
        return None
    runs: list[list[str]] = [[names[0]]]
    strict: list[str] = []
    for relation, name in zip(relations, names[1:], strict=True):
        if relation == "~":
            runs[-1].append(name)
        else:
            strict.append(relation)
            runs.append([name])
    return runs, strict


def _render_ordering(runs: list[list[str]]) -> str:
    return " > ".join(" ~ ".join(run) for run in runs)


def _claim_preserving_spellings(ordering: str) -> list[str]:
    """Every spelling of ``ordering`` that asserts exactly the same relations, measured one first.

    A rendered ordering asserts its **adjacent** relations and nothing else, because tie-ness is
    not transitive -- the reason :func:`_order_string` and :func:`_ordering_margins` both work
    neighbour by neighbour. ``A ~ B ~ C`` therefore claims A~B and B~C, and says nothing about A
    and C.

    That is what bounds the rewrite. Within one tie run only the **reversal** keeps the same
    adjacent pairs: ``A ~ B ~ C`` -> ``C ~ B ~ A`` still claims B~C and A~B. Any other permutation
    invents a comparison, so ``B ~ A ~ C`` -- which asserts A~C -- is not offered even though it
    contains the same three names. Two spellings per run of length > 1, not the run's factorial.

    Strict relations are never reordered: ``>`` is precisely the relation the measurement DID
    decide.

    The measured spelling is returned first so a caller that takes ``[0]`` without looking gets
    what the data showed rather than whatever sorts first.
    """
    parsed = _parse_ordering(ordering)
    if parsed is None:
        return [str(ordering)]
    runs, _strict = parsed
    # One choice per run: as measured, or reversed. A singleton run has only itself.
    options = [[run, run[::-1]] if len(run) > 1 else [run] for run in runs]
    out: list[str] = []
    for combination in _product(options):
        spelled = _render_ordering(list(combination))
        if spelled not in out:
            out.append(spelled)
    return out


def _product(options: list[list[list[str]]]) -> list[tuple[list[str], ...]]:
    """``itertools.product`` over the per-run choices, measured-first order preserved."""
    import itertools

    return list(itertools.product(*options)) if options else []


def _permitted_orderings_from(source: Any, conditions: list[str]) -> list[str]:
    """The task's list of legal ordering patterns, out of whatever the caller was handed.

    The agent holds the vocabulary *document*, not the inner list, so accepting only the list is
    how this argument goes unpassed. A path, a document, or a bare list all work.

    Two things make a list the answer space rather than a glossary, and both are required:

    * its entries carry relation symbols -- a list of niche names has none, so it can never be
      mistaken for one;
    * its entries name **these** conditions. A list that orders somebody else's conditions is
      somebody else's answer space, and adopting it would rewrite every result into a vocabulary
      the data has no bearing on.

    Unreadable, absent or unrecognisable input answers ``[]``, which every caller treats as "no
    vocabulary supplied" -- a no-op, never a guess.
    """
    if source is None:
        return []
    if isinstance(source, str):
        try:
            source = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
    wanted = {str(c) for c in conditions}

    def _usable(candidate: Any) -> list[str]:
        if not isinstance(candidate, (list, tuple)) or not candidate:
            return []
        patterns = [str(x) for x in candidate if isinstance(x, str)]
        if len(patterns) != len(candidate):
            return []
        named: set[str] = set()
        for pattern in patterns:
            parsed = _parse_ordering(pattern)
            if parsed is None or (len(parsed[0]) == 1 and len(parsed[0][0]) == 1):
                return []  # no relation symbol: a glossary, not an answer space
            named |= {name for run in parsed[0] for name in run}
        return patterns if named == wanted else []

    if isinstance(source, (list, tuple)):
        return _usable(source)
    if isinstance(source, dict):
        for value in source.values():
            found = _usable(value)
            if found:
                return found
    return []


def _spell_as_permitted(ordering: str, permitted: Any) -> tuple[str, str | None]:
    """Re-spell ``ordering`` as the permitted list spells it, or say why it cannot be.

    Three outcomes, and the third is the one worth being careful about:

    * already listed -> returned unchanged with no note, so a legal result gains no annotation;
    * a claim-preserving spelling is listed -> that spelling, with a note saying it is the same
      claim. The rewrite is a rendering choice, so the caller keeps the measured string too;
    * nothing expressible -> the **measurement**, with a note naming the gap. Substituting the
      nearest listed pattern would trade a spelling problem for a wrong claim, which is worse than
      an answer the vocabulary cannot hold.

    The defect this exists for: SpatialBench-Long's xenium task supplies 13 patterns and requires
    one of them. At MES 0.51 / PRO 0.26 / WT 0.24 the tool renders ``MES > PRO ~ WT``; the list
    carries that same claim as ``MES > WT ~ PRO``. Two arms shipped the unlisted spelling and a
    legality check failed a result that was right.
    """
    patterns = permitted if isinstance(permitted, (list, tuple)) else []
    if not patterns:
        return str(ordering), None
    listed = [str(p) for p in patterns]
    if str(ordering) in listed:
        return str(ordering), None
    for candidate in _claim_preserving_spellings(ordering):
        if candidate in listed:
            return candidate, (
                f"reported as '{candidate}'; the measurement rendered '{ordering}', which is the "
                "same claim spelled the other way round inside a tie. Only the reversal of a tie "
                "run preserves the claim, so no other pattern was considered."
            )
    return str(ordering), (
        f"'{ordering}' is not one of the {len(listed)} supplied patterns, and neither is its only "
        "other spelling. It is reported as measured rather than rounded to the nearest listed "
        "pattern, because every one of them says something this data did not show."
    )


# ---------------------------------------------------------------------------
# Terms a glossary defines over other populations rather than by markers
# ---------------------------------------------------------------------------
#: A parenthetical is where a glossary keeps its gene symbols -- "(COL1A2+, DCN+)" -- and gene
#: symbols are capitalised exactly like population names. Stripped before entities are read, so a
#: marker never arrives as a population.
_PARENTHETICAL = re.compile(r"\([^)]*\)")

#: The phrases that make a definition *compositional* rather than merely mentioning a population.
#: Almost every definition names some population; being DEFINED OVER one is the claim, and without
#: this cue "a Neuronal marker panel from Smith et al" would read as a composition of Neuronal.
_COMPOSITION_CUE = re.compile(
    r"\b(co-?locali[sz]\w*|co-?occur\w*|dominated by|composed of|made up of|made of|"
    r"consisting of|consists of|mixture of|mix of|interface between|boundary between|"
    r"juxtaposition of|intermingl\w*)\b",
    re.I,
)

#: Capitalised words that are structure rather than population names. Without labels to resolve
#: against, these would otherwise read as components of the definition.
_NOT_A_POPULATION = frozenset(
    {
        "a",
        "an",
        "the",
        "region",
        "regions",
        "spatial",
        "area",
        "areas",
        "zone",
        "zones",
        "niche",
        "niches",
        "where",
        "and",
        "or",
        "of",
        "in",
        "with",
        "between",
        "cells",
        "cell",
        "this",
        "that",
    }
)

_CAPITALISED_RUN = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:[ -][A-Z][a-zA-Z0-9]*)*)\b")


def _composition_components(definition: str, labels: list[str] | None = None) -> list[str]:
    """The populations a compositional definition is defined *over*, in the order it names them.

    ``"spatial region where Neuronal and Proliferating cells co-localize"`` -> ``["Neuronal",
    "Proliferating"]``. Without a relational cue the answer is ``[]``, however many populations the
    prose mentions: being defined over a population and citing one are different claims.

    ``labels`` is the annotation column's own vocabulary. A glossary's spelling and an ``obs``
    column's spelling rarely match exactly, so each component is resolved against the labels that
    actually exist -- first exactly, then by a unique case-insensitive prefix in either direction.
    A component matching two labels is **dropped rather than guessed**: there is no single right
    answer, and inventing one would put a score on the wrong population.

    Passing no labels answers what the definition says, which is what the caller needs before it
    knows whether a composition column exists at all.
    """
    text = _PARENTHETICAL.sub(" ", str(definition or ""))
    if not _COMPOSITION_CUE.search(text):
        return []
    named: list[str] = []
    for match in _CAPITALISED_RUN.finditer(text):
        token = match.group(1).strip()
        if token.lower() in _NOT_A_POPULATION or token in named:
            continue
        named.append(token)
    if labels is None:
        return named

    available = [str(x) for x in labels]
    lower = {x.lower(): x for x in available}
    resolved: list[str] = []
    for token in named:
        if token in available:
            hit = token
        elif token.lower() in lower:
            hit = lower[token.lower()]
        else:
            # A unique prefix in either direction: "Neuroepithelia" resolves to "Neuroepithelial",
            # but not when "Neuroepithelia_2" is also present.
            near = [x for x in available if x.lower().startswith(token.lower()) or token.lower().startswith(x.lower())]
            if len(near) != 1:
                continue
            hit = near[0]
        if hit not in resolved:
            resolved.append(hit)
    return resolved


def _composition_key_candidates(adata, cluster_key: str) -> list[str]:
    """``adata.obs`` columns that could hold population labels, best guess, never a promise.

    A label column has a handful of levels shared by many cells. A measurement column has a level
    per cell -- ``total_counts``, ``n_genes_by_counts`` -- and offering one would send the caller
    to score a niche against a continuous QC metric.

    The clustering being assigned is excluded by name: scoring a term from the very partition it is
    being assigned to is circular, and it would win every cluster it was asked about.
    """
    out: list[str] = []
    n_obs = int(adata.n_obs) or 1
    for column in adata.obs.columns:
        name = str(column)
        if name == str(cluster_key):
            continue
        series = adata.obs[column]
        levels = int(series.nunique(dropna=True))
        if levels < 2 or levels >= n_obs:
            continue
        categorical = str(series.dtype) in {"object", "category", "bool", "string"}
        # An integer-coded label column is still a label column; a float one never is.
        coded = pd.api.types.is_integer_dtype(series) and levels <= max(2, n_obs // 10)
        if categorical or coded:
            out.append(name)
    return out


def _definition_kind(markers: list[str], definition: str) -> str:
    """How this term is defined: by ``"marker"``, by ``"composition"``, or not at all.

    Three cases that marker scoring renders identically -- all NaN -- and that call for three
    different actions. A term with off-panel markers needs a different assay; a compositional term
    needs a *label column*, which the caller may already have; a term the glossary never really
    defines needs a better vocabulary. Saying "no markers are listed for this term in the supplied
    vocabulary" about the second is simply untrue, and it was the reason the tool gave.
    """
    if markers:
        return "marker"
    return "composition" if _composition_components(definition) else "undefined"


def _write(output_dir: str | None, name: str, frame: pd.DataFrame) -> str | None:
    if not output_dir:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    frame.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# 1. which clusters moved
# ---------------------------------------------------------------------------
def compare_cluster_abundance(
    adata_or_path: Any,
    cluster_key: str,
    condition_key: str,
    output_dir: str | None = None,
    tie_tolerance_pct: float | None = None,
    permitted_orderings: Any = None,
) -> str:
    """Rank clusters by how much their abundance shifts between conditions.

    Abundance is the **within-condition** fraction: of the cells belonging to one condition, what
    share falls in this cluster. That is the quantity that survives unequal numbers of cells per
    condition, which raw cluster counts do not.

    Args:
        adata_or_path: Path to an .h5ad, or an in-memory AnnData.
        cluster_key: ``adata.obs`` column holding cluster or domain labels.
        condition_key: ``adata.obs`` column holding the condition / genotype / timepoint.
        output_dir: Optional directory; the full table is written there as CSV.
        tie_tolerance_pct: Percentage points within which two conditions are rendered tied. If
            the task states a tie rule, pass that number: the ordering is only as meaningful as
            the tolerance behind it, and leaving this unset silently accepts
            ``DEFAULT_TIE_TOLERANCE_PCT``. The output records which of the two happened.
        permitted_orderings: Optional closed list of ordering patterns the task will accept -- or
            the whole vocabulary document holding one, or a path to it. ``A ~ B`` and ``B ~ A``
            are one claim with two spellings, and a closed vocabulary carries only one of them, so
            a correct finding can otherwise come back in a form the task rejects. When supplied,
            each ordering is re-spelled to the listed form **if and only if** the listed form makes
            the same claim; the measured string stays beside it. Orderings the list cannot express
            are reported as measured and named at the top level rather than rounded to a near miss.
            Leaving this unset changes nothing: the tool behaves exactly as it did before.

    Returns:
        str: JSON with one record per cluster -- the per-condition fractions, the spread in
             percentage points, log2 of the max/min ratio, an ``abundance_ordering`` string
             such as ``"treated > control ~ vehicle"``, and ``ordering_margins`` giving the adjacent gap
             behind every ``>`` and ``~`` in that string -- sorted by spread, widest first.

    """
    tolerance_source = "default" if tie_tolerance_pct is None else "caller"
    if tie_tolerance_pct is None:
        tie_tolerance_pct = DEFAULT_TIE_TOLERANCE_PCT

    try:
        adata = _load(adata_or_path)
        _require_obs(adata, cluster_key, "cluster_key")
        _require_obs(adata, condition_key, "condition_key")
    except (FileNotFoundError, KeyError, OSError) as exc:
        return _err(str(exc))

    adata, n_unlabelled = _drop_unlabelled(adata, cluster_key, condition_key)
    counts = pd.crosstab(adata.obs[cluster_key].astype(str), adata.obs[condition_key].astype(str))
    permitted = _permitted_orderings_from(permitted_orderings, [str(c) for c in counts.columns])
    per_condition_total = counts.sum(axis=0)
    empty = [str(c) for c in per_condition_total[per_condition_total == 0].index]
    # Within-condition composition: each column sums to 1.
    fractions = counts.div(per_condition_total.replace(0, np.nan), axis=1).fillna(0.0)

    records = []
    inexpressible: list[dict[str, Any]] = []
    for cluster in fractions.index:
        row = fractions.loc[cluster]
        margins = _ordering_margins(row, tie_tolerance_pct)
        hi, lo = float(row.max()), float(row.min())
        measured_ordering = _order_string(row, tie_tolerance_pct)
        spelled_ordering, spelling_note = _spell_as_permitted(measured_ordering, permitted)
        record: dict[str, Any] = {
            "cluster": str(cluster),
            "n_cells": int(counts.loc[cluster].sum()),
            "abundance_ordering": spelled_ordering,
            "ordering_margins": margins,
            # The weakest link in the string: the whole ordering is only as firm as this.
            "narrowest_margin_to_flip_pct_points": (
                round(min(m["margin_to_flip_pct_points"] for m in margins), 4) if margins else None
            ),
            "spread_pct_points": round((hi - lo) * 100.0, 4),
            "max_condition": str(row.idxmax()),
            "min_condition": str(row.idxmin()),
            # A cluster absent from a condition has an undefined ratio, not an infinite one.
            "log2_max_over_min": (round(float(np.log2(hi / lo)), 4) if lo > 0 and hi > 0 else None),
        }
        for condition in fractions.columns:
            record[f"fraction_in_{condition}"] = round(float(row[condition]), 6)
        if spelling_note is not None:
            # Only when something was actually said. A result the vocabulary already spells the
            # way it was measured gains no annotation, so a reader can tell the two apart.
            record["ordering_spelling_note"] = spelling_note
            if spelled_ordering != measured_ordering:
                # Re-spelling is a rendering choice and must not silently overwrite the finding.
                record["abundance_ordering_as_measured"] = measured_ordering
            else:
                inexpressible.append(
                    {
                        "cluster": str(cluster),
                        "abundance_ordering_as_measured": measured_ordering,
                        "why": spelling_note,
                    }
                )
        records.append(record)

    records.sort(key=lambda r: r["spread_pct_points"], reverse=True)
    written = _write(output_dir, "cluster_abundance_by_condition.csv", pd.DataFrame(records))

    return json.dumps(
        {
            "status": "success",
            "n_clusters": int(fractions.shape[0]),
            "conditions": [str(c) for c in fractions.columns],
            "conditions_with_no_cells": empty,
            "abundance_definition": (
                "within-condition fraction: of the cells in one condition, the share in this "
                "cluster. Each condition's fractions sum to 1 across clusters."
            ),
            "tie_tolerance_pct": tie_tolerance_pct,
            "tie_tolerance_source": tolerance_source,
            "tie_tolerance_note": (
                "every '>' and '~' in an abundance_ordering is a claim about this number; "
                "ordering_margins gives the gap behind each one and how far it is from flipping"
            ),
            "orderings_resting_on_the_narrowest_calls": [
                {
                    "cluster": r["cluster"],
                    "abundance_ordering": r["abundance_ordering"],
                    "narrowest_margin_to_flip_pct_points": r["narrowest_margin_to_flip_pct_points"],
                }
                for r in sorted(
                    (r for r in records if r["narrowest_margin_to_flip_pct_points"] is not None),
                    key=lambda r: r["narrowest_margin_to_flip_pct_points"],
                )[:10]
            ],
            "permitted_orderings_in_force": len(permitted) or None,
            # Supplied and not adopted used to read exactly like not supplied (u22-spatial-pipeline-5).
            **(
                {
                    "permitted_orderings_note": (
                        "permitted_orderings was supplied, but no list in it orders exactly these conditions "
                        f"({', '.join(str(c) for c in fractions.columns)}), so every ordering is reported as measured"
                    )
                }
                if permitted_orderings is not None and not permitted
                else {}
            ),
            "orderings_not_expressible_in_the_permitted_list": inexpressible,
            "n_cells_excluded_missing_label": n_unlabelled,
            "ranked_by": "spread_pct_points, widest first",
            "clusters": records,
            "csv_path": written,
        },
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# 2. what each cluster is, from its own genes
# ---------------------------------------------------------------------------
#: The share of one observation the marker summary may fill. ``clip_observation`` cuts anything
#: longer than :data:`OBSERVATION_CHARS` out of its middle -- for a table sorted by cluster, the
#: middle clusters -- so the summary is sized to leave the other half for whatever else the same
#: cell prints.
MARKER_SUMMARY_CHARS = OBSERVATION_CHARS // 2

#: Where the full per-gene table goes when the caller names no ``output_dir``: a subdirectory of
#: the run's work root, resolved exactly as the MCP portals resolve theirs (``SOG_WORK_DIR``, then
#: a writable ``/workspace/work``, then ``./work``).
MARKER_TABLE_SUBDIR = "rank_cluster_markers"


def _short_number(field: str, value: float | None) -> str:
    """A number as one line of the summary writes it: log2fc to 2 decimals, a score to 3 significant
    digits, or as a whole number once its magnitude is 100 or more (1234.5 is 1234, never 1.23e+03).

    Not one rule for both. A log fold-change reads on a fixed scale, where 0.44 and 5.56 are both
    worth two decimals. A score's scale is the method's: a Wilcoxon z runs past 100 on a large
    sample, and a logreg coefficient can sit below 0.01, where a fixed two decimals would print 0.
    """
    if value is None:
        return "NA"
    if field == "log2fc":
        return f"{value:.2f}"
    return f"{value:.0f}" if abs(value) >= 100 else f"{value:.3g}"


def _marker_token(entry: dict[str, Any], fields: tuple[str, ...]) -> str:
    return f"{entry['gene']} " + "/".join(_short_number(field, entry.get(field)) for field in fields)


def _marker_table_name(cluster_key: str) -> str:
    """One file per clustering, so ranking a second clustering does not overwrite the first's."""
    return f"cluster_markers_{re.sub(r'[^A-Za-z0-9._-]+', '_', str(cluster_key)) or 'clusters'}.csv"


def rank_cluster_markers(
    adata_or_path: Any,
    cluster_key: str,
    n_genes: int = 25,
    output_dir: str | None = None,
    method: str = "wilcoxon",
) -> str:
    """Top differential genes per cluster -- what each cluster is, before any label is proposed.

    Naming a cluster from a remembered marker set asks "where are the genes I expected?"; naming it
    from this table asks "what is actually high here?". Only the second can return a population you
    were not already thinking of, which is the one that matters in a comparative study.

    What comes back is a summary sized to be read whole, and the table it summarises is on disk.
    Returned as one indented record per gene, 20 clusters x 25 genes came to about 86,000
    characters: the observation limit cut the middle clusters out, and a live run (E-06, xenium r3)
    re-ran the whole test on 100,000 cells to see one of them. Now every cluster is one line -- its
    top genes, best first, each with its log2 fold-change and score -- and the full per-gene table
    (score, log2fc, pval_adj, rounded as the ranking records them) is always written, to ``csv_path``.

    Args:
        adata_or_path: Path to an .h5ad, or an in-memory AnnData.
        cluster_key: ``adata.obs`` column holding cluster or domain labels.
        n_genes: How many top genes to rank per cluster. All of them go to the CSV; the summary
            shows as many as fit :data:`MARKER_SUMMARY_CHARS`, the same number for every cluster,
            and says how many that was.
        output_dir: Directory for the full table. Defaults to ``rank_cluster_markers/`` under the
            run's work root -- the directory the MCP portals write to when they are given none.
        method: Passed to ``scanpy.tl.rank_genes_groups``.

    Returns:
        str: JSON with one ``markers`` line per cluster, ``genes_shown_per_cluster``, the
             ``markers_format`` those lines are written in, and ``csv_path`` -- the full table.

    """
    try:
        adata = _load(adata_or_path)
        _require_obs(adata, cluster_key, "cluster_key")
        adata, n_unlabelled = _drop_unlabelled(adata, cluster_key)
        work = _light_adata(adata, cluster_key)
    except (FileNotFoundError, KeyError, OSError) as exc:
        return _err(str(exc))

    import scanpy as sc

    sizes = work.obs[cluster_key].value_counts()
    singletons = sorted(str(name) for name, n in sizes.items() if n < 2)
    if singletons:
        # rank_genes_groups cannot test a one-cell group; dropping it beats a raw exception.
        work = work[~work.obs[cluster_key].isin(singletons)].copy()
    work.obs[cluster_key] = work.obs[cluster_key].astype("category")
    if work.obs[cluster_key].nunique() < 2:
        return _err(
            f"need at least 2 clusters of 2+ cells to rank markers; '{cluster_key}' has "
            f"{work.obs[cluster_key].nunique()}",
            skipped_singleton_clusters=singletons,
        )

    _normalize_log1p(work)
    sc.tl.rank_genes_groups(work, groupby=cluster_key, method=method, n_genes=int(n_genes))
    result = work.uns["rank_genes_groups"]

    rows: list[dict[str, Any]] = []
    by_cluster: dict[str, list[dict[str, Any]]] = {}
    for group in result["names"].dtype.names:
        entries = []
        for rank in range(len(result["names"][group])):
            entry = {
                "cluster": str(group),
                "rank": rank + 1,
                "gene": str(result["names"][group][rank]),
                "score": round(float(result["scores"][group][rank]), 4),
            }
            for key, label in (("logfoldchanges", "log2fc"), ("pvals_adj", "pval_adj")):
                if key in result:
                    value = float(result[key][group][rank])
                    entry[label] = round(value, 6) if np.isfinite(value) else None
            entries.append(entry)
            rows.append(entry)
        by_cluster[str(group)] = entries

    csv_error = None
    try:
        directory = output_dir or tool_output_root(MARKER_TABLE_SUBDIR)
        written = os.path.abspath(_write(directory, _marker_table_name(cluster_key), pd.DataFrame(rows)))
    except OSError as exc:
        # The ranking itself succeeded, and the summary is still worth returning -- but it must not
        # point at a file that is not there.
        written, csv_error = None, f"{type(exc).__name__}: {exc}"

    # logreg returns scores alone, so the line carries exactly the numbers the method produced.
    fields = ("log2fc", "score") if "logfoldchanges" in result else ("score",)
    precision = "score to 3 significant digits, a whole number once |score| >= 100"
    if "log2fc" in fields:
        precision = f"log2fc to 2 decimals, NA where it is not finite; {precision}"
    table_fields = [field for field in ("score", "log2fc", "pval_adj") if rows and field in rows[0]]
    tokens = {cluster: [_marker_token(entry, fields) for entry in entries] for cluster, entries in by_cluster.items()}
    n_ranked = max((len(entries) for entries in by_cluster.values()), default=0)

    def render(shown: int) -> str:
        if written:
            where = (
                f"The full table -- every ranked gene with {', '.join(table_fields)} -- is the CSV at "
                "csv_path; read it for anything a line does not show "
                "rather than re-running the ranking."
            )
        else:
            where = (
                f"The full table could NOT be written ({csv_error}): nothing beyond these lines -- no "
                f"gene past the top {shown} -- was kept. Pass a writable "
                "output_dir to keep it."
            )
        payload = {
            "status": "success",
            "cluster_key": cluster_key,
            "method": method,
            "n_clusters": len(by_cluster),
            "n_genes_per_cluster": int(n_genes),
            "genes_shown_per_cluster": shown,
            "skipped_singleton_clusters": singletons,
            "n_cells_excluded_missing_label": n_unlabelled,
            "csv_path": written,
            "markers_format": (
                f"one line per cluster: its top {shown} of {n_ranked} ranked genes, best first, each "
                f"written 'gene {'/'.join(fields)}' ({precision}). {where}"
            ),
            "markers": {cluster: ", ".join(line[:shown]) for cluster, line in tokens.items()},
        }
        if csv_error:
            payload["csv_error"] = csv_error
        return json.dumps(payload, indent=2, default=str)

    # The same number of genes for every cluster, as many as fit: the length only grows with it,
    # so the largest that fits is found by bisection. One gene each is the floor -- a cluster with
    # no line at all is the failure this exists to prevent -- and the floor can be over the budget.
    # Measured 2026-09-28 on 3-cell clusters and 8-character gene names, one gene each came to
    # 7,451 characters at 200 clusters and 10,830 at 300; past one observation, clip_observation
    # cuts lines out of the middle and says so, and the CSV at csv_path still holds every cluster.
    low, high = min(1, n_ranked), n_ranked
    while low < high:
        middle = (low + high + 1) // 2
        if len(render(middle)) <= MARKER_SUMMARY_CHARS:
            low = middle
        else:
            high = middle - 1
    return render(low)


# ---------------------------------------------------------------------------
# 3. every term against every cluster
# ---------------------------------------------------------------------------
#: Overlap is quadratic in the vocabulary; the longest tail is near-zero overlap and is not
#: what the list is for. Truncation is reported rather than silent.
_MAX_OVERLAP_PAIRS = 50


# ---------------------------------------------------------------------------
# 4. which of the vocabulary's own terms moved, in one ranking
# ---------------------------------------------------------------------------
def rank_label_abundance_changes(
    adata_or_path: Any,
    cluster_key: str,
    condition_key: str,
    vocabulary: Any,
    output_dir: str | None = None,
    tie_tolerance_pct: float | None = None,
    composition_key: str | None = None,
    layer: str | None = None,
) -> str:
    """Rank a vocabulary's own terms by how much their abundance shifts between conditions.

    :func:`score_label_vocabulary` says what each cluster is; :func:`compare_cluster_abundance`
    says which clusters moved. The join between them -- which of the vocabulary's terms moved, and
    in what order -- was left to the caller, and callers do it one term at a time: notice a term,
    check it, keep it. Nothing is compared to anything, so a slot in a top-N answer goes to a
    candidate that would not have survived a comparison that never happened. This returns the
    comparison.

    Two distinctions make the ranking honest, and they are the reason this is not a sort over a
    score column:

    * a term that was **measured and won no cluster** is ranked at zero. Its zero came from the
      assignment, so it belongs in the comparison, and the row says where it came closest and to
      what it lost -- "it lost" is not actionable, "it was second in c0 behind t_a" is;
    * a term that **could not be measured at all** is not ranked. Its zero would be an artifact of
      never having been scored, and it would sort against real ones. Those terms come back
      separately, each with the reason it could not be measured, because an off-panel marker and an
      invariant one call for different next steps.

    Abundance is the within-condition fraction, as in :func:`compare_cluster_abundance`: of the
    cells in one condition, the share held by the clusters this term won.

    Args:
        adata_or_path: Path to an .h5ad, or an in-memory AnnData.
        cluster_key: ``adata.obs`` column holding cluster or domain labels.
        condition_key: ``adata.obs`` column holding the condition / genotype / timepoint.
        vocabulary: ``{term: [marker, ...]}``, a glossary document, or a path to one -- whatever
            :func:`score_label_vocabulary` accepts.
        output_dir: Optional directory; the ranking is written there as CSV, in the same order and
            without the nested columns.
        tie_tolerance_pct: Percentage points within which two conditions render as tied. The
            output records whether this was the caller's number or the default.
        composition_key: Optional ``adata.obs`` column of population labels, for terms the
            vocabulary defines over other populations rather than by markers.
        layer: Optional ``adata.layers`` key to score on instead of ``adata.X``.

    Returns:
        str: JSON with one row per measurable term ordered by ``spread_pct_points``, the terms that
             won nothing and where they came closest, the terms that could not be measured and why,
             and the share of each condition the ranking does not speak for.

    """
    tolerance_source = "default" if tie_tolerance_pct is None else "caller"
    if tie_tolerance_pct is None:
        tie_tolerance_pct = DEFAULT_TIE_TOLERANCE_PCT

    try:
        adata = _load(adata_or_path)
        _require_obs(adata, cluster_key, "cluster_key")
        _require_obs(adata, condition_key, "condition_key")
    except (FileNotFoundError, KeyError, OSError) as exc:
        return _err(str(exc))

    # Scored and counted over the same cells: the ones with both a cluster and a condition.
    adata, n_unlabelled = _drop_unlabelled(adata, cluster_key, condition_key)
    scored = json.loads(
        score_label_vocabulary(adata, cluster_key, vocabulary, composition_key=composition_key, layer=layer)
    )
    if scored.get("status") == "error":
        return json.dumps(scored, indent=2, default=str)
    if scored.get("status") == "no_measurable_terms":
        # A ranking over terms that were never measured has rows, has an order, and is all
        # artifact. Returning it with a warning invites exactly the read the warning forbids.
        return json.dumps(
            {
                "status": "no_measurable_terms",
                "ranking_not_attempted": (
                    "No term had a measurable marker, so there is nothing to rank: every ordering "
                    "would be an artifact of the sort rather than a finding. Nothing here can be "
                    "read as a result until you fix the scoring first."
                ),
                "remedy": scored.get("remedy", ""),
                "n_terms_not_ranked": len(scored.get("unevidenceable_terms", [])),
                "terms_not_ranked_because_unmeasurable": [
                    {"term": e["term"], "reason": e["reason"]} for e in scored.get("unevidenceable_terms", [])
                ],
                "csv_path": None,
            },
            indent=2,
            default=str,
        )

    assignments = {a["cluster"]: a for a in scored["assignments"]}
    bookkeeping = scored["term_bookkeeping"]
    score_table = scored["score_table"]

    counts = pd.crosstab(adata.obs[cluster_key].astype(str), adata.obs[condition_key].astype(str))
    per_condition_total = counts.sum(axis=0)
    conditions = [str(c) for c in counts.columns]

    won: dict[str, list[str]] = {}
    for cluster, entry in assignments.items():
        best = entry.get("best_term")
        if best:
            won.setdefault(str(best), []).append(str(cluster))

    # Every term the scoring could measure, whether or not it won anything. A measured term that
    # won nothing is a real zero; an unmeasurable one is not a zero at all.
    unmeasurable = {e["term"]: e for e in scored.get("unevidenceable_terms", [])}
    measurable = [t for t in bookkeeping if t not in unmeasurable]

    rows: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    for term in measurable:
        clusters_won = sorted(won.get(term, []))
        book = bookkeeping[term]
        if clusters_won:
            held = counts.loc[clusters_won].sum(axis=0)
        else:
            held = pd.Series(0, index=counts.columns)
        fractions = held.div(per_condition_total.replace(0, np.nan)).fillna(0.0)
        margins = _ordering_margins(fractions, tie_tolerance_pct)
        won_scores = [float(score_table[term][c]) for c in clusters_won if score_table[term].get(c) is not None]
        assignment_margins = [
            assignments[c]["margin"] for c in clusters_won if assignments[c].get("margin") is not None
        ]
        row: dict[str, Any] = {
            "term": term,
            "definition_kind": book.get("definition_kind", "marker"),
            "clusters_won": clusters_won,
            "won_no_cluster": not clusters_won,
            "n_cells": int(held.sum()),
            "spread_pct_points": round((float(fractions.max()) - float(fractions.min())) * 100.0, 4),
            "abundance_ordering": _order_string(fractions, tie_tolerance_pct),
            "ordering_margins": margins,
            "max_condition": str(fractions.idxmax()) if len(fractions) else None,
            "min_condition": str(fractions.idxmin()) if len(fractions) else None,
            "narrowest_margin_to_flip_pct_points": (
                min((m["margin_to_flip_pct_points"] for m in margins), default=None)
            ),
            "n_markers_usable": book.get("n_markers_usable", 0),
            "marker_coverage": book.get("marker_coverage", 0.0),
            # None, not a number invented from an empty set.
            "thinnest_assignment_margin": (min(assignment_margins) if assignment_margins else None),
            "weakest_winning_score": (round(min(won_scores), 4) if won_scores else None),
            "decided_on_shared_markers_only": bool(
                clusters_won and all(assignments[c].get("decided_on_shared_markers_only") for c in clusters_won)
            ),
        }
        for condition in conditions:
            row[f"fraction_in_{condition}"] = round(float(fractions[condition]), 6)
        rows.append(row)

        if not clusters_won:
            # "It lost" is not actionable. Where it came closest, and to what, is.
            closest, behind, lost_to = None, None, None
            for cluster in [str(c) for c in counts.index]:
                mine = score_table.get(term, {}).get(cluster)
                winner = assignments.get(cluster, {}).get("best_score")
                if mine is None or winner is None:
                    continue
                gap = float(winner) - float(mine)
                if behind is None or gap < behind:
                    closest, behind, lost_to = cluster, gap, assignments[cluster].get("best_term")
            missed.append(
                {
                    "term": term,
                    "came_closest": {
                        "closest_cluster": closest,
                        "lost_to": lost_to,
                        "behind_by": (round(behind, 4) if behind is not None else None),
                    },
                }
            )

    rows.sort(key=lambda r: r["spread_pct_points"], reverse=True)

    # The share of each condition this ranking does not speak for. Here it is usually zero -- and
    # the zero has to be stated, not implied, or a reader cannot tell it from an unasked question.
    unassigned = sorted(str(c) for c in counts.index if not assignments.get(str(c), {}).get("best_term"))
    if unassigned:
        unassigned_share = counts.loc[unassigned].sum(axis=0).div(per_condition_total.replace(0, np.nan)).fillna(0.0)
    else:
        unassigned_share = pd.Series(0.0, index=counts.columns)

    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "ordering_margins"} for r in rows])
    written = _write(output_dir, "label_abundance_ranking.csv", flat)

    return json.dumps(
        {
            "status": "success",
            "cluster_key": cluster_key,
            "condition_key": condition_key,
            "conditions": conditions,
            "abundance_definition": (
                "within-condition fraction: of the cells in one condition, the share held by the clusters this term won"
            ),
            "ranked_by": "spread_pct_points, widest first",
            "terms_by_size_of_change": [r["term"] for r in rows],
            "n_terms_ranked": len(rows),
            "n_terms_not_ranked": len(unmeasurable),
            "tie_tolerance_pct": tie_tolerance_pct,
            "tie_tolerance_source": tolerance_source,
            "orderings_resting_on_the_narrowest_calls": [
                {
                    "term": r["term"],
                    "abundance_ordering": r["abundance_ordering"],
                    "narrowest_margin_to_flip_pct_points": r["narrowest_margin_to_flip_pct_points"],
                }
                for r in sorted(
                    (r for r in rows if r["narrowest_margin_to_flip_pct_points"] is not None),
                    key=lambda r: r["narrowest_margin_to_flip_pct_points"],
                )[:10]
            ],
            "terms": rows,
            "terms_that_won_no_cluster": missed,
            "terms_not_ranked_because_unmeasurable": [
                {"term": e["term"], "reason": e["reason"], "definition_kind": e.get("definition_kind", "undefined")}
                for e in scored.get("unevidenceable_terms", [])
            ],
            "terms_needing_a_composition_key": scored.get("terms_needing_a_composition_key", []),
            "composition_defined_terms": scored.get("composition_defined_terms", []),
            "unassigned_clusters": unassigned,
            "unassigned_cell_share_by_condition": {
                str(c): round(float(unassigned_share[c]), 6) for c in counts.columns
            },
            "n_cells_excluded_missing_label": n_unlabelled,
            **(
                {"n_cells_excluded_missing_composition_label": scored.get("n_cells_excluded_missing_composition_label")}
                if "n_cells_excluded_missing_composition_label" in scored
                else {}
            ),
            "csv_path": written,
        },
        indent=2,
        default=str,
    )


def _assignment_audit(
    assignments: list[dict[str, Any]],
    bookkeeping: dict[str, dict[str, Any]],
    scores: pd.DataFrame,
    sizes: dict[str, int],
) -> dict[str, Any]:
    """What the winning terms did to the sample, counted in cells rather than in scores.

    The assignment table answers "which term fits this cluster best" one cluster at a time, and a
    cluster is the wrong unit for two questions that decide whether an annotation is usable.

    The first is how much of the sample each winning term ended up holding. A term that takes a
    quarter of the cells is a claim about the tissue, not a label, and a merged or catch-all term
    usually takes it by winning many clusters each by a *narrow* margin, or by winning a cluster its
    own score is negative in -- so the share is reported beside both tells: the thinnest margin it
    was won on, and the weakest score. Anything computed downstream over these labels, a
    count, a ranking, a co-localisation or an enrichment, inherits that share, which is why it is
    worth seeing before that step rather than after it.

    The second is which terms won nothing at all. A term whose markers this assay can measure and
    which still never wins a cluster is either genuinely absent from the sample, or was outcompeted
    at a clustering resolution too coarse to separate it from a broader neighbour -- and those two
    have different remedies (report the absence, or re-cluster and re-score). What distinguishes
    them is where the term came closest and by how little: a term sitting second in some cluster by
    a hair is not an absent population, and reporting zero cells for it is a finding about the
    clustering, not about the tissue.

    ``marker_overlap`` supports the same reading from the vocabulary side. Where one term's usable
    markers are a subset of another's, the two are competing for the same evidence and the broader
    one will win on any cluster where both are expressed, whatever the biology.

    None of these are verdicts. They are counts, returned so that they are in front of whoever is
    deciding rather than behind them.
    """
    total = sum(sizes.values())

    won: dict[str, list[dict[str, Any]]] = {}
    for row in assignments:
        term = row.get("best_term")
        if term is not None:
            won.setdefault(str(term), []).append(row)

    cells_per_term = []
    for term, rows in won.items():
        n_cells = sum(sizes.get(str(r["cluster"]), 0) for r in rows)
        margins = [r["margin"] for r in rows if r.get("margin") is not None]
        won_on = [r["best_score"] for r in rows if r.get("best_score") is not None]
        cells_per_term.append(
            {
                "term": term,
                "n_clusters_won": len(rows),
                "clusters_won": sorted(str(r["cluster"]) for r in rows),
                "n_cells": n_cells,
                "share_of_cells": round(n_cells / total, 4) if total else 0.0,
                "thinnest_margin": round(min(margins), 4) if margins else None,
                # Scores are z across clusters, so a negative one means the term is *below* its own
                # average in a cluster it nonetheless won: it won because nothing else scored, not
                # because the cluster is positive for it. That is a different problem from a narrow
                # margin and has a different remedy, so it is reported separately.
                "weakest_score_won_on": round(min(won_on), 4) if won_on else None,
                "won_a_cluster_it_is_low_in": bool(won_on and min(won_on) < 0),
            }
        )
    cells_per_term.sort(key=lambda r: (-r["share_of_cells"], r["term"]))

    # Evidenceable and yet never chosen. Terms with no usable marker are excluded on purpose: they
    # are already reported as unevidenceable_terms, where "won nothing" says nothing new.
    never_assigned = []
    for term in sorted(bookkeeping):
        if term in won or not bookkeeping[term]["n_markers_usable"]:
            continue
        row = scores.loc[term].dropna()
        if row.empty:
            continue
        cluster = str(row.idxmax())
        column = scores[cluster].dropna().sort_values(ascending=False)
        best_here = float(row.max())
        never_assigned.append(
            {
                "term": term,
                "definition": bookkeeping[term]["definition"],
                "n_markers_usable": bookkeeping[term]["n_markers_usable"],
                "marker_coverage": bookkeeping[term]["marker_coverage"],
                "usable_markers": bookkeeping[term]["usable_markers"],
                "closest_cluster": cluster,
                "score_there": round(best_here, 4),
                "rank_there": int(list(column.index).index(term)) + 1,
                "lost_to": str(column.index[0]),
                "behind_by": round(float(column.iloc[0]) - best_here, 4),
            }
        )
    never_assigned.sort(key=lambda r: (r["behind_by"], r["term"]))

    # Nesting, not similarity: share_of_smaller == 1.0 means one term's whole usable panel sits
    # inside the other's, which is the merged-versus-constituent case a Jaccard would blur.
    evidenceable = sorted(t for t in bookkeeping if bookkeeping[t]["n_markers_usable"])
    overlap = []
    for i, first in enumerate(evidenceable):
        left = set(bookkeeping[first]["usable_markers"])
        for second in evidenceable[i + 1 :]:
            right = set(bookkeeping[second]["usable_markers"])
            common = left & right
            if not common:
                continue
            overlap.append(
                {
                    "terms": [first, second],
                    "shared_markers": sorted(common),
                    "share_of_smaller": round(len(common) / min(len(left), len(right)), 4),
                }
            )
    n_overlapping = len(overlap)
    overlap.sort(key=lambda r: (-r["share_of_smaller"], -len(r["shared_markers"]), r["terms"]))

    return {
        "n_cells": total,
        "basis": (
            "each cluster takes its best-scoring term; shares are of all cells in cluster_key, and "
            "are what the labels would imply if this assignment were applied as-is"
        ),
        "cells_per_term": cells_per_term,
        "terms_never_assigned": never_assigned,
        "marker_overlap": overlap[:_MAX_OVERLAP_PAIRS],
        "n_overlapping_pairs": n_overlapping,
        "overlap_truncated": n_overlapping > _MAX_OVERLAP_PAIRS,
    }


def score_label_vocabulary(
    adata_or_path: Any,
    cluster_key: str,
    vocabulary: Any,
    output_dir: str | None = None,
    layer: str | None = None,
    composition_key: str | None = None,
) -> str:
    """Score every vocabulary term against every cluster, and report the whole table.

    A term's score for a cluster is the mean, over the markers the assay can actually measure, of
    that gene's expression z-scored across clusters. Scoring is exhaustive on purpose: the output
    has a row for every term, including the ones that lose, so a term that was never in contention
    cannot be reported as one that was ruled out.

    Four things are reported that a hand-rolled assignment usually drops:

    * ``definition`` -- the vocabulary's own words for the term, verbatim. A glossary usually says
      more than which markers a term carries: which population it names, and how its author classed
      it. Whether a term answers the question actually asked is decided from that text, not from
      its score, so the text is returned rather than parsed away.

    * ``n_markers_usable`` and ``marker_coverage`` -- the denominator each score was computed over.
      A term scoring low on one measurable marker out of eight is a different claim from a term
      scoring low on eight out of eight.
    * ``unevidenceable_terms`` -- terms with no measurable marker at all. These are not absent from
      the tissue; they are unanswerable from this panel, and saying so is the honest result.
    * ``shared_markers`` -- markers claimed by more than one term. A cluster assigned on a shared
      marker is evidence for each of those terms and decides between none of them.
    * ``assignment_audit`` -- the same assignment counted in cells instead of clusters: the share of
      the sample each winning term ends up holding and the thinnest margin it won on, the terms this
      assay can evidence that still won nothing and where they came closest, and the term pairs
      whose usable markers nest. See :func:`_assignment_audit`.

    Args:
        adata_or_path: Path to an .h5ad, or an in-memory AnnData.
        cluster_key: ``adata.obs`` column holding cluster or domain labels.
        vocabulary: ``{term: [marker, ...]}``, or a path to a JSON file holding one. A mapping of
            term to a dict containing a ``markers`` list is also accepted.
        output_dir: Optional directory; the full term x cluster table is written there as CSV.
        layer: Optional ``adata.layers`` key to score on instead of ``adata.X``.
        composition_key: Optional ``adata.obs`` column holding cell-type or population labels. A
            glossary term defined over other populations -- "where Neuronal and Proliferating cells
            co-localize" -- lists no genes, so marker scoring returns NaN for it exactly as it does
            for a term whose markers are all off-panel. Supplying this column scores such a term
            from the share of each cluster made of the populations it names, on the same z-scale as
            the marker terms so the two can be ranked together. Terms that need it are listed in
            ``terms_needing_a_composition_key`` whether or not it is passed.

    Returns:
        str: JSON with the full term x cluster score table, a best/runner-up assignment per cluster
             with the margin between them, the per-term cell shares and never-assigned terms in
             ``assignment_audit``, and the coverage bookkeeping described above.

    """
    try:
        adata = _load(adata_or_path)
        _require_obs(adata, cluster_key, "cluster_key")
        if composition_key is not None:
            # An error, not a silent skip: a caller who passed this column asked for the
            # compositional terms to be measured, and quietly scoring without them would answer a
            # different question than the one asked.
            _require_obs(adata, composition_key, "composition_key")
        panels, definitions, unscored_lists = _coerce_vocabulary(vocabulary)
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        return _err(str(exc))
    if not panels:
        return _err("vocabulary is empty: nothing to score")
    adata, n_unlabelled = _drop_unlabelled(adata, cluster_key)

    try:
        means = _cluster_mean_expression(adata, cluster_key, layer)
    except (KeyError, MemoryError, ValueError) as exc:
        return _err(f"could not compute cluster means: {exc}")

    # z across clusters, per gene. A gene with no variance between clusters cannot discriminate
    # between them, so it is dropped from the usable set rather than contributing a constant.
    spread = means.std(axis=0, ddof=0)
    informative = spread > 0
    z = pd.DataFrame(np.zeros(means.shape), index=means.index, columns=means.columns)
    subset = means.loc[:, informative]
    z.loc[:, informative] = (subset - subset.mean(axis=0)) / spread[informative]

    measurable = {str(g) for g in means.columns[informative]}
    in_assay = {str(g) for g in means.columns}

    # Markers claimed by more than one term. Restricted to the measurable ones: the list exists to
    # flag calls that rest on evidence shared between terms, and a marker the assay cannot measure
    # is not shared evidence, it is no evidence.
    claims: dict[str, int] = {}
    for markers in panels.values():
        for marker in set(markers) & measurable:
            claims[marker] = claims.get(marker, 0) + 1
    shared = sorted(marker for marker, n in claims.items() if n > 1)

    clusters = [str(c) for c in means.index]
    scores = pd.DataFrame(index=sorted(panels), columns=clusters, dtype=float)
    bookkeeping: dict[str, dict[str, Any]] = {}
    unevidenceable: list[dict[str, Any]] = []
    composition_scored: list[dict[str, Any]] = []
    needs_composition_key: list[str] = []

    # The share of each cluster held by each population label, if a label column was supplied.
    # Computed once: it is the same table every compositional term reads a different subset of.
    label_share: pd.DataFrame | None = None
    labels_present: list[str] = []
    n_without_population: int | None = None
    if composition_key is not None:
        # A cell with no population label is not a population, and not a share of one either. The
        # count goes in the report: every share in a cluster holding such cells is over fewer cells,
        # and the first repair dropped them in silence (hunt 2026-09-30, u22-spatial-pipeline-5).
        labelled, n_without_population = _drop_unlabelled(adata, composition_key)
        crosstab = pd.crosstab(labelled.obs[cluster_key].astype(str), labelled.obs[composition_key].astype(str))
        crosstab = crosstab.reindex(index=clusters, fill_value=0)
        totals = crosstab.sum(axis=1).replace(0, np.nan)
        label_share = crosstab.div(totals, axis=0).fillna(0.0)
        labels_present = [str(c) for c in label_share.columns]

    for term, markers in panels.items():
        usable = [m for m in markers if m in measurable]
        present = [m for m in markers if m in in_assay]
        bookkeeping[term] = {
            "definition": definitions.get(term, ""),
            "n_markers_listed": len(markers),
            "n_markers_in_assay": len(present),
            "n_markers_usable": len(usable),
            "marker_coverage": round(len(usable) / len(markers), 4) if markers else 0.0,
            "usable_markers": usable,
            "missing_from_assay": sorted(set(markers) - set(present)),
            "shared_markers": sorted(set(markers) & set(shared)),
        }
        definition = definitions.get(term, "")
        kind = _definition_kind(markers, definition)
        bookkeeping[term]["definition_kind"] = kind

        if usable:
            scores.loc[term] = z.loc[:, usable].mean(axis=1).to_numpy()
            continue

        # No measurable marker. Before calling it unevidenceable, ask how the glossary defined it:
        # a compositional term is not a term the vocabulary failed to define, and saying so was a
        # false statement the tool used to make.
        components = _composition_components(definition, labels_present) if label_share is not None else []
        if components:
            share = label_share.loc[:, components].sum(axis=1)
            spread_share = float(share.std(ddof=0))
            if spread_share > 0:
                # z across clusters, exactly as a marker score is, so the two sort on one scale.
                scores.loc[term] = ((share - share.mean()) / spread_share).to_numpy()
                bookkeeping[term]["n_composition_labels"] = len(components)
                bookkeeping[term]["composition_terms"] = components
                composition_scored.append({"term": term, "scored_from": components, "definition": definition})
                continue
            # A constant share discriminates nothing, exactly as an invariant gene does.
            scores.loc[term] = np.nan
            unevidenceable.append(
                {
                    "term": term,
                    "n_markers_listed": len(markers),
                    "definition_kind": kind,
                    "reason": (
                        f"defined over {', '.join(components)}, whose combined share does not vary "
                        f"between clusters in '{composition_key}'"
                    ),
                    "definition": definition,
                }
            )
            continue

        # NaN, not 0.0: "cannot be measured" must not sort alongside "measured and low".
        scores.loc[term] = np.nan
        if kind == "composition":
            named = _composition_components(definition)
            needs_composition_key.append(term)
            reason = (
                f"defined by composition over {', '.join(named)} rather than by markers. Marker "
                f"scoring cannot reach it; pass composition_key naming the adata.obs column that "
                f"holds those population labels and it becomes measurable"
                + (f" (none of them resolved against '{composition_key}')" if composition_key else "")
            )
        elif not markers:
            reason = "no markers are listed for this term in the supplied vocabulary"
        elif present:
            reason = "listed markers are present but invariant across clusters"
        else:
            reason = "no listed marker is present in this assay"
            # The symbol is absent, but the same letters in another casing may not be -- COL1A2
            # against a mouse panel's Col1a2. Matching across case is an orthology assumption this
            # tool does not make on its own; it names the candidates so the caller can (hunt
            # 2026-09-30, u22-spatial-pipeline-6).
            folded = {str(g).lower(): str(g) for g in in_assay}
            near = {m: folded[m.lower()] for m in markers if m.lower() in folded}
            if near:
                reason += (
                    " under the spelling the vocabulary uses; "
                    + ", ".join(f"{m} is present as {g}" for m, g in near.items())
                    + " (case differs -- if those are the same genes, pass the vocabulary in this assay's casing)"
                )
        unevidenceable.append(
            {
                "term": term,
                "n_markers_listed": len(markers),
                "definition_kind": kind,
                "reason": reason,
                "definition": definition,
            }
        )

    assignments = []
    for cluster in clusters:
        column = scores[cluster].dropna().sort_values(ascending=False)
        if column.empty:
            assignments.append({"cluster": cluster, "best_term": None, "reason": "no term had a measurable marker"})
            continue
        best = str(column.index[0])
        runner_up = str(column.index[1]) if len(column) > 1 else None
        assignments.append(
            {
                "cluster": cluster,
                "best_term": best,
                "best_score": round(float(column.iloc[0]), 4),
                "runner_up_term": runner_up,
                "runner_up_score": (round(float(column.iloc[1]), 4) if runner_up else None),
                "margin": (round(float(column.iloc[0] - column.iloc[1]), 4) if runner_up else None),
                "best_n_markers_usable": bookkeeping[best]["n_markers_usable"],
                "best_marker_coverage": bookkeeping[best]["marker_coverage"],
                "decided_on_shared_markers_only": bool(
                    set(bookkeeping[best]["usable_markers"]).issubset(set(shared))
                    and bookkeeping[best]["usable_markers"]
                ),
            }
        )

    sizes = {str(k): int(v) for k, v in adata.obs[cluster_key].astype(str).value_counts().items()}
    audit = _assignment_audit(assignments, bookkeeping, scores, sizes)

    # Did anything get measured at all? A NaN score reads exactly like a genuinely low one, so
    # this verdict has to travel with the table and not only with the return value.
    measured_terms = {term for term, book in bookkeeping.items() if book["n_markers_usable"] > 0}
    status = "success" if measured_terms else "no_measurable_terms"

    tidy = scores.rename_axis("term").reset_index().melt(id_vars="term", var_name="cluster", value_name="score")
    # The denominator beside every score, in the FILE. A caller who passed output_dir and then read
    # the CSV used to see bare term/cluster/score: 328 rows of NaN with nothing to distinguish "this
    # term scored low" from "this term was never measured", and one caller ranked on them.
    tidy["n_markers_usable"] = tidy["term"].map(lambda t: bookkeeping[str(t)]["n_markers_usable"])
    tidy["measured"] = tidy["term"].map(lambda t: str(t) in measured_terms)
    tidy["status"] = status
    written = _write(output_dir, "label_vocabulary_scores.csv", tidy)

    payload: dict[str, Any] = {
        "status": status,
        "cluster_key": cluster_key,
        "n_terms_scored": len(panels),
        "n_cells_excluded_missing_label": n_unlabelled,
        **(
            {"n_cells_excluded_missing_composition_label": n_without_population}
            if n_without_population is not None
            else {}
        ),
        # The denominator behind the status. "41 terms scored" and "41 terms scored, 0 of them
        # on a marker this assay can measure" are the same run and very different claims.
        "n_terms_with_a_measurable_marker": len(measured_terms),
        "vocabulary_lists_not_scored": unscored_lists,
        "composition_defined_terms": composition_scored,
        "terms_needing_a_composition_key": needs_composition_key,
        "n_clusters": len(clusters),
        "scoring": (
            "mean over a term's measurable markers of that gene's expression z-scored across "
            "clusters; every term is scored against every cluster"
        ),
        "assignments": assignments,
        "assignment_audit": audit,
        "term_bookkeeping": bookkeeping,
        "unevidenceable_terms": unevidenceable,
        "shared_markers": shared,
        "score_table": json.loads(scores.to_json(orient="index")),
        "csv_path": written,
    }
    if needs_composition_key:
        # Half a report is a reminder about a keyword argument. The other half is what to pass --
        # or, in the case that cost 19 trials, that there is nothing to pass and the column has to
        # be produced first, on an object whose obs held only QC metrics.
        candidates = _composition_key_candidates(adata, cluster_key)
        payload["composition_key_candidates_in_obs"] = candidates
        if candidates:
            where = f"This object has {', '.join(candidates)} in adata.obs; re-run with composition_key set to whichever holds population labels."
        else:
            have = ", ".join(str(c) for c in adata.obs.columns[:8]) or "nothing"
            where = (
                f"There is no such column on this object yet -- adata.obs holds {have} -- so it "
                "has to be produced first, by annotating the cells (a reference-based label "
                "transfer, or a marker-based call) into a new obs column."
            )
        payload["composition_key_remedy"] = (
            f"{len(needs_composition_key)} term(s) are defined over other populations rather than "
            f"by genes, so marker scoring left them unmeasured: {', '.join(needs_composition_key)}. "
            f"{where} Until then they are absent from the ranking rather than at the bottom of it, "
            "and this run has not ruled them out."
        )

    if status == "no_measurable_terms":
        # A diagnosis, not decoration: it is present only when there is something to fix. Every
        # term scoring on nothing is nearly always a reshaped vocabulary rather than a tissue that
        # contains none of them, so the remedy names that first.
        payload["remedy"] = (
            "No term had a single measurable marker, which almost always means the vocabulary was "
            "reshaped on the way in -- markers read from the wrong key and written as empty lists "
            "-- rather than that this assay contains none of them. Check the vocabulary against "
            "its source and confirm the marker names match this assay's var_names. Until then this "
            "table is not a measurement -- every score in it is NaN -- so do not rank on it: an "
            "order over NaN is an artifact of the sort, not a finding."
        )
    return json.dumps(payload, indent=2, default=str)


def _markers_from_value(value: Any) -> list[str]:
    """Markers out of a list, a mapping that nests them under a key, or glossary prose.

    An **empty** ``markers`` list falls through to the prose rather than answering "no markers".
    That is not leniency: an empty list is what a converter writes when it looked under the wrong
    key, and it is the exact shape that cost a benchmark trial -- a prose glossary reshaped to
    ``{"markers": []}`` for all 41 terms, scored on nothing, 328 rows of NaN written, and ranked on
    by a caller reading only the CSV. A populated list still wins over the prose beside it, so a
    vocabulary that genuinely lists its markers is never second-guessed.
    """
    if isinstance(value, dict):
        for key in ("markers", "genes", "marker_genes"):
            listed = value.get(key)
            if isinstance(listed, (list, tuple)):
                found = [str(g).strip() for g in listed if str(g).strip()]
                if found:
                    return found
                break  # declared and empty: read the prose rather than report nothing
        value = " ".join(v for v in value.values() if isinstance(v, str))
    if isinstance(value, (list, tuple)):
        return [str(g).strip() for g in value if str(g).strip()]
    if isinstance(value, str):
        return _markers_in_prose(value)
    return []


def _panels_of(mapping: dict) -> dict[str, list[str]]:
    """Every key becomes a term. A term with no markers is kept, with an empty list."""
    return {str(term): _markers_from_value(value) for term, value in mapping.items()}


def _definitions_of(mapping: dict) -> dict[str, str]:
    """The vocabulary's own words for each term, kept verbatim.

    A glossary entry is rarely just a marker list. It usually also says which population the term
    names and how the vocabulary's author classed it -- baseline or off-target, specific or
    catch-all, defined by markers or by the co-occurrence of other cell types. That text is part of
    the definition the task supplied, so it travels with the term instead of being parsed away to
    the marker symbols inside it. Nothing here interprets it; the caller reads it.
    """
    out: dict[str, str] = {}
    for term, value in mapping.items():
        text = ""
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            for key in ("description", "definition", "notes", "summary", "text"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate
                    break
        if text.strip():
            out[str(term)] = " ".join(text.split())
    return out


def _looks_like_document(mapping: dict) -> bool:
    """Is this a whole vocabulary document rather than a ``{term: markers}`` mapping?

    Decided on shape, not on content: a term mapping's values are all marker lists, prose, or
    mappings that nest a marker list under a key. Anything else -- a schema version, a nested
    glossary, a sibling term list -- means the keys at this level are structure, and the terms are
    a level down.
    """
    for value in mapping.values():
        if isinstance(value, dict):
            if not any(isinstance(value.get(key), (list, tuple)) for key in ("markers", "genes", "marker_genes")):
                return True
        elif not isinstance(value, (str, list, tuple)):
            return True
    return False


def _coerce_vocabulary(
    vocabulary: Any,
) -> tuple[dict[str, list[str]], dict[str, str], list[dict[str, Any]]]:
    """Read a vocabulary into ``{term: [marker, ...]}`` plus ``{term: definition}``.

    Terms that have no markers are kept.

    Accepts a plain ``{term: [gene, ...]}`` mapping, a path to a JSON file holding one, or a whole
    vocabulary document of the kind benchmarks ship -- a term list beside a glossary that defines
    each term in prose. For the document form, the glossary is found by looking for the nested
    mapping that yields the most markers, and any sibling list of term names that plainly names the
    same vocabulary is merged in, so a term the glossary omits still gets a row.

    Terms with no markers are deliberately **kept**. Dropping them here would reintroduce, inside
    this function, the silent narrowing the module exists to prevent: the caller would never learn
    that the term was unscoreable, only that it was absent from the results.

    Returns a third value: the document's own term lists that this scoring did **not** take. A
    document can carry several axes and exactly one of them is scored, so the ones left behind are
    returned rather than merely absent -- scoring one axis silently is how the wrong one gets
    ranked. Returned rather than stashed on the module, because the portal serves several accounts
    from one process and a shared sink would race between them.
    """
    if isinstance(vocabulary, (str, Path)):
        path = Path(vocabulary)
        if not path.exists():
            raise FileNotFoundError(f"vocabulary path does not exist: {vocabulary}")
        try:
            vocabulary = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"vocabulary file is not valid JSON: {exc}") from exc
    if not isinstance(vocabulary, dict):
        raise TypeError(f"vocabulary must be a mapping or a path to one, got {type(vocabulary).__name__}")
    if not vocabulary:
        raise ValueError("vocabulary is empty: nothing to score")

    unscored: list[dict[str, Any]] = []
    if not _looks_like_document(vocabulary):
        return _panels_of(vocabulary), _definitions_of(vocabulary), unscored

    # A document: its own keys are structure, not terms. The glossary is the nested mapping that
    # yields the most markers, breaking ties on the one that defines the most terms.
    panels: dict[str, list[str]] = {}
    definitions: dict[str, str] = {}
    best = (-1, -1)
    for value in vocabulary.values():
        if not isinstance(value, dict) or not value:
            continue
        nested = _panels_of(value)
        rank = (sum(1 for m in nested.values() if m), len(nested))
        if rank > best:
            panels, definitions, best = nested, _definitions_of(value), rank

    # A sibling list of bare term names is the authoritative vocabulary standing next to its
    # glossary, so a term the glossary omits still gets a row. Merged only when it plainly names the
    # same things -- otherwise an unrelated list, such as the answer patterns these documents also
    # carry, would arrive as a crowd of junk terms.
    name_lists = [
        value
        for value in vocabulary.values()
        if isinstance(value, list) and value and all(isinstance(n, str) for n in value)
    ]
    for names in name_lists:
        if sum(1 for n in names if n in panels) * 2 >= len(names):
            for name in names:
                panels.setdefault(name, [])

    if not panels and name_lists:
        # A document with a term list but no glossary at all: the terms are still the answer space,
        # and reporting them as unevidenceable beats reporting nothing.
        for name in max(name_lists, key=len):
            panels[name] = []

    # Which of the document's own term lists this scoring did NOT take. A document can carry
    # several axes -- two cell-type vocabularies and an enrichment-category one -- and exactly one
    # of them gets scored. Scoring one silently is how the wrong axis gets ranked, so the ones left
    # behind are named rather than simply absent.
    for key, value in vocabulary.items():
        if not (isinstance(value, list) and value and all(isinstance(n, str) for n in value)):
            continue
        missing = [n for n in value if n not in panels]
        if missing:
            unscored.append({"key": str(key), "n_not_scored": len(missing), "examples": missing[:5]})
    return panels, definitions, unscored
