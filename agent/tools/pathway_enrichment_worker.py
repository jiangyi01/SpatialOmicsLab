#!/usr/bin/env python3
"""Pathway enrichment worker: gene-set ORA / GSEA-prerank on a ranked gene table, and per-spot
pathway activity on a spatial h5ad, via decoupler (gseapy as the optional GSEA backend).

Runs inside the ``pathway_enrichment`` conda env and speaks the fleet's worker contract: one JSON
object on stdout, everything else on stderr, an error dict rather than a traceback.

    python tools/pathway_enrichment_worker.py --task enrichment --gene-table ranked.csv \\
        --gene-column gene --score-column gft_score --collections hallmark,go_bp,reactome \\
        --method both --output-dir ./work/pathway_enrichment
    python tools/pathway_enrichment_worker.py --task activity --data-path slide.h5ad \\
        --collections progeny,hallmark --method ulm --group-key spatial_domain \\
        --output-dir ./work/pathway_activity

GENE SETS AND THE NETWORK. Collections are MSigDB (hallmark, go_bp, go_mf, go_cc, reactome,
wikipathways; kegg is opt-in because of its licence) and PROGENy, fetched from OmniPath through
decoupler ONCE per machine and cached as ``.gmt`` / ``.csv`` files under ``$SOG_GENESET_CACHE``
(default ``~/.spatialomicsgym/genesets``). Every later run is offline. A collection that is neither
cached nor fetchable fails with an error dict that names the cache path, the variable and the host,
and says the tool does not need reprovisioning -- the same family as ``base_mcp``'s launch
diagnostics. A local ``.gmt`` path in ``--collections`` never touches the network at all.

SYMBOLS. Everything is matched on gene SYMBOLS against the human collections. Mouse symbols are
upper-cased (``Sparc`` -> ``SPARC``) and the match rate is reported; a rate under 60 % is a warning
in the payload, never silent. Ensembl identifiers are refused with the remedy (name the symbol
column). Duplicates are dropped and counted.

decoupler is pinned at 1.9.2 (RL-1: 2.x depends on marsilea). Its 1.x API defaults to
``use_raw=True``, which on anndata 0.12 is ``None``; every call here passes ``use_raw=False``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker_utils import (
    WorkerOutput,
    cpu_budget,
    default_output_dir,
    keep_in_tissue,
    record_in_tissue,
    sniff_tabular_sep,
)

TOOL = "pathway_enrichment"
OMNIPATH_HOST = "omnipathdb.org"
CACHE_ENV = "SOG_GENESET_CACHE"
DEFAULT_CACHE = "~/.spatialomicsgym/genesets"

#: The collection names a caller may ask for -> decoupler's MSigDB ``collection`` value.
COLLECTION_ALIASES: dict[str, str] = {
    "hallmark": "hallmark",
    "go_bp": "go_biological_process",
    "go_mf": "go_molecular_function",
    "go_cc": "go_cellular_component",
    "reactome": "reactome_pathways",
    "wikipathways": "wikipathways",
    "kegg": "kegg_pathways",
}
PROGENY = "progeny"
DEFAULT_TOP_PROGENY = 500

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+")
_PVALUE_COLUMNS = {
    "p_values",
    "pvalue",
    "pvals",
    "p_value",
    "pval",
    "fdr",
    "padj",
    "qval",
    "q_value",
    "fdr_bh",
    "p_adj",
    "pvals_adj",
}
#: A column whose NAME says it is a p-value or an FDR, beyond the fixed set above: Seurat's
#: ``p_val_adj``, edgeR's ``FDR``, DESeq2's ``padj``, limma's ``adj.P.Val``, a bare ``q``.
_PVALUE_NAME_RE = re.compile(
    r"^(?:adj[_.]?)?(?:p|q)(?:[_.]?(?:val(?:ue)?s?))?(?:[_.]?adj(?:ust(?:ed)?)?)?$|^adj\.p\.val$|^(?:fdr|padj|bh)(?:[_.].*)?$",
    re.I,
)
#: A column whose name says it is a RANK: 1 is best, so "higher = better" needs it negated.
_RANK_NAME_RE = re.compile(r"(?:^|[_.\-])ranks?(?:$|[_.\-])", re.I)
#: A GSEA run on fewer ranked genes than this is a run on a truncated list (a ``rank_genes_groups``
#: result keeps 50 markers per group by default), not a genome-wide ranking; it is allowed, and said.
TRUNCATED_RANKING_MIN = 500
MIN_PERMUTATIONS = 10
_SPECIES = {
    "human": "human",
    "homo sapiens": "human",
    "hsapiens": "human",
    "hs": "human",
    "h. sapiens": "human",
    "mouse": "mouse",
    "mus musculus": "mouse",
    "mmusculus": "mouse",
    "mm": "mouse",
    "m. musculus": "mouse",
}


def normalize_species(raw: str) -> str:
    """``human`` or ``mouse`` from the spellings a caller uses; anything else is refused, not
    silently treated as human (which is what ``"Mus musculus"`` used to get, with a 0% match)."""
    key = " ".join(str(raw or "").strip().lower().replace("_", " ").split())
    if key in _SPECIES:
        return _SPECIES[key]
    raise ToolError(
        f"species {raw!r} is not one this tool knows (human or mouse).",
        "Pass species=human or species=mouse (Homo sapiens / Mus musculus are accepted spellings). The gene "
        "sets are human; mouse symbols are upper-cased and matched to them.",
    )


def score_direction(column: str) -> str:
    """How a score column is turned into "higher = more interesting": ``neglog10p`` for a p-value
    or FDR column, ``negated_rank`` for a rank, ``as_is`` for a score. Judged by NAME, and said in
    the payload -- a Seurat ``p_val_adj`` or an ``svg_rank`` used to rank inverted, silently."""
    name = str(column or "").strip()
    low = name.lower()
    if low in _PVALUE_COLUMNS or _PVALUE_NAME_RE.match(low):
        return "neglog10p"
    if _RANK_NAME_RE.search(low):
        return "negated_rank"
    return "as_is"


_GENE_COLUMN_CANDIDATES = (
    "gene",
    "genes",
    "gene_name",
    "gene_names",
    "names",
    "symbol",
    "gene_symbol",
    "feature",
    "features",
)
LOW_MATCH_WARN = 0.60
INLINE_GENES_MAX = 2000


def log(msg: str) -> None:
    print(f"[pathway-enrichment-worker] {msg}", file=sys.stderr)


class ToolError(Exception):
    """A refusal the caller can act on: the message and a remedy, no traceback needed."""

    def __init__(self, message: str, diagnostic: str = ""):
        super().__init__(message)
        self.diagnostic = diagnostic


# --------------------------------------------------------------------------------------------- #
# symbols
# --------------------------------------------------------------------------------------------- #


def split_genes(text: str) -> list[str]:
    """An inline list separated by commas, whitespace or newlines."""
    return [g for g in re.split(r"[\s,;]+", text or "") if g]


def looks_like_ensembl(symbols: list[str]) -> bool:
    if not symbols:
        return False
    hits = sum(1 for s in symbols if _ENSEMBL_RE.match(str(s).strip()))
    return hits >= max(1, len(symbols) // 2)


def normalize_symbols(symbols, species: str) -> tuple[list[str], int, bool]:
    """(unique symbols in order, duplicates dropped, whether case was normalized).

    Mouse symbols are upper-cased so ``Sparc`` meets the human ``SPARC``; human symbols are left
    as they are (upper-casing a human list is a no-op that would hide a lower-case file).
    """
    case_normalized = species.lower().startswith("mouse")
    seen: set[str] = set()
    out: list[str] = []
    dupes = 0
    for raw in symbols:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() in ("nan", "none"):
            continue
        if case_normalized:
            s = s.upper()
        if s in seen:
            dupes += 1
            continue
        seen.add(s)
        out.append(s)
    if looks_like_ensembl(out):
        raise ToolError(
            f"the gene list holds Ensembl identifiers ({out[0]}, ...), and the collections are keyed by gene symbol.",
            "Pass the symbol column instead (gene_column=<symbol column>), or map the identifiers to symbols before "
            "calling the tool. Nothing is wrong with this tool's installation.",
        )
    return out, dupes, case_normalized


#: var columns that carry gene symbols beside Ensembl var_names: CELLxGENE exports use
#: ``feature_name``, scanpy's 10x reader ``gene_symbols``, others ``gene_name``.
_VAR_SYMBOL_COLUMNS = ("feature_name", "gene_symbols", "gene_symbol", "gene_name", "gene_names", "symbol", "symbols")


def var_symbol_column(var) -> str | None:
    """The var column of an Ensembl-indexed h5ad that holds symbols, or None."""
    for col in _VAR_SYMBOL_COLUMNS:
        if col in var.columns:
            values = [str(v) for v in var[col].head(200) if v is not None and str(v).strip()]
            if values and not looks_like_ensembl(values):
                return col
    return None


def ensembl_var_names_remedy(var, h5ad_path: str) -> str:
    """What to do with an h5ad whose var_names are Ensembl ids, naming the symbol column it really has.

    The refusal used to say "e.g. from var['gene_symbols']" -- a column the library's CELLxGENE
    samples do not have; they keep symbols in var['feature_name'] (hunt 2026-09-30, u30-uncovered-mcp-12).
    """
    col = var_symbol_column(var)
    if col:
        return (
            f"var[{col!r}] of {h5ad_path} holds the symbols: set adata.var_names = adata.var[{col!r}] (then "
            "adata.var_names_make_unique()), write the h5ad and pass that file."
        )
    return (
        f"No var column of {h5ad_path} holds symbols (var columns: {list(var.columns)[:15]}); map the Ensembl "
        "identifiers to symbols before calling."
    )


# --------------------------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------------------------- #


#: The portal's default for ``gene_column``. A caller who leaves it is asking for auto-detection;
#: a caller who names any OTHER column is asking for that column and gets a refusal when it is
#: absent -- not a silently substituted one, which is the same wrong turn as running code under an
#: interpreter the operator did not name.
DEFAULT_GENE_COLUMN = "gene"


def _pick_gene_column(columns: list[str], requested: str) -> str:
    if requested and requested in columns:
        return requested
    if requested and requested != DEFAULT_GENE_COLUMN:
        raise ToolError(
            f"gene_column {requested!r} is not in the table; the columns parsed are {columns[:20]}.",
            "Pass gene_column=<the column holding gene symbols> exactly as the header spells it, or leave it at "
            "the default to auto-detect a gene/symbol column.",
        )
    lowered = {c.lower(): c for c in columns}
    for cand in _GENE_COLUMN_CANDIDATES:
        if cand in lowered:
            return lowered[cand]
    raise ToolError(
        f"no gene column named {requested!r} in the table; the columns parsed are {columns[:20]}.",
        "Pass gene_column=<the column holding gene symbols>. If the columns look like one long name, the "
        "delimiter was mis-read: the tool sniffs comma/tab/semicolon from the first line.",
    )


def read_gene_table(path: str, gene_column: str, score_column: str):
    """(ordered symbols, scores or None, score column used, the direction applied -- see
    :func:`score_direction`; ``"as_is"`` when no score column was named).

    The shipped SVG tables (``prost_top_svg_genes.csv``, ``spagft_top_svg_genes.csv``,
    ``spatially_variable_genes.csv``) all carry a ``gene`` column and a mix of scores and
    p-values; a p-value-like column is turned into ``-log10(p)`` so "higher = more variable"
    holds for the ranking either way.
    """
    import numpy as np
    import pandas as pd

    p = Path(path)
    if not p.is_file():
        raise ToolError(f"gene_table {path!r} does not exist.", "Pass the path of a CSV/TSV whose rows are genes.")
    sep = sniff_tabular_sep(str(p))
    df = pd.read_csv(p, sep=sep)
    columns = [str(c) for c in df.columns]
    # A one-column file is a plain list -- one symbol per line, header or not -- whether or not the
    # caller left ``gene_column`` at its default. The portal always passes the default, so ``not
    # gene_column`` alone refused every plain list that arrived through it ("no gene column named
    # 'gene'").
    if len(columns) == 1 and (not gene_column or gene_column == DEFAULT_GENE_COLUMN):
        col = columns[0]
        symbols = (
            [col] + df[col].astype(str).tolist()
            if col.lower() not in _GENE_COLUMN_CANDIDATES
            else df[col].astype(str).tolist()
        )
        return symbols, None, "", "as_is"
    gcol = _pick_gene_column(columns, gene_column)
    symbols = df[gcol].astype(str).tolist()
    if not score_column:
        return symbols, None, "", "as_is"
    if score_column not in columns:
        raise ToolError(
            f"score_column {score_column!r} is not in the table; the columns parsed are {columns[:20]}.",
            "Name a numeric column of the table, or leave score_column empty to rank by file order.",
        )
    values = pd.to_numeric(df[score_column], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).all():
        raise ToolError(f"score_column {score_column!r} holds no numbers.", "Name a numeric column.")
    direction = score_direction(score_column)
    if direction == "neglog10p":
        values = -np.log10(np.clip(np.nan_to_num(values, nan=1.0), 1e-300, 1.0))
    elif direction == "negated_rank":
        # ``max + 1 - rank``, not ``-rank``: rank 1 must come first AND the statistic must stay
        # positive. An all-negative ranking made decoupler's GSEA normalisation degenerate
        # (NES = inf, FDR 1.0 for the one set that was really enriched; driven 2026-09-20).
        top = float(np.nanmax(values))
        values = (top + 1.0) - np.nan_to_num(values, nan=top)
    else:
        values = np.nan_to_num(values, nan=float(np.nanmin(values)))
    return symbols, values.tolist(), score_column, direction


def read_h5ad_genes(h5ad_path: str, key: str, group: str):
    """Genes from ``uns[key]`` (a rank_genes_groups result; ``group`` picks the cluster) or a var column."""
    import anndata as ad
    import numpy as np

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        universe = [str(v) for v in adata.var_names]
        if key in adata.uns:
            result = adata.uns[key]
            names = result.get("names") if hasattr(result, "get") else None
            if names is None:
                raise ToolError(
                    f"uns[{key!r}] has no 'names' field; it is not a rank_genes_groups result.",
                    "Point h5ad_key at a rank_genes_groups key or a var column.",
                )
            groups = list(names.dtype.names) if getattr(names, "dtype", None) is not None and names.dtype.names else []
            if not group:
                raise ToolError(
                    f"uns[{key!r}] holds markers for groups {groups[:20]}; say which one with `group`.",
                    "Pass group=<one of the groups above>.",
                )
            if group not in groups:
                raise ToolError(
                    f"group {group!r} is not in uns[{key!r}]; the groups are {groups[:20]}.",
                    "Pass one of the groups listed.",
                )
            symbols = [str(s) for s in names[group]]
            scores = None
            if "scores" in result:
                scores = np.asarray(result["scores"][group], dtype=float).tolist()
            return symbols, scores, universe
        if key in adata.var.columns:
            col = adata.var[key]
            if col.dtype == bool:
                mask = col.to_numpy()
            else:
                mask = np.nan_to_num(np.asarray(col, dtype=float)) != 0
            return [str(v) for v in adata.var_names[mask]], None, universe
        raise ToolError(
            f"h5ad_key {key!r} is neither a uns key nor a var column of {h5ad_path}. uns keys: {list(adata.uns.keys())[:15]}; var columns: {list(adata.var.columns)[:15]}.",
            "Pass h5ad_key=<a rank_genes_groups key> with `group`, or a boolean/numeric var column.",
        )
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


def read_background(path: str) -> list[str]:
    p = Path(path)
    if not p.is_file():
        raise ToolError(f"background {path!r} does not exist.", "Pass a file with one gene symbol per line.")
    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    # A table is not a background. A CSV handed here was read as one "symbol" per ROW --
    # ``gene,score,fdr`` and all -- so nothing matched and every test ran on an empty universe.
    head = lines[:50]
    delimited = sum(1 for ln in head if ("," in ln or "\t" in ln or ";" in ln))
    if head and delimited > len(head) // 2:
        raise ToolError(
            f"background {path!r} looks like a table (delimited rows), not a list of symbols.",
            "Pass a file with ONE gene symbol per line and nothing else, or pass the h5ad as h5ad_path so "
            "its var_names become the universe.",
        )
    return lines


# --------------------------------------------------------------------------------------------- #
# gene-set cache
# --------------------------------------------------------------------------------------------- #


def cache_dir(arg: str) -> Path:
    raw = arg or os.environ.get(CACHE_ENV) or DEFAULT_CACHE
    return Path(os.path.expanduser(raw))


def cache_path(directory: Path, collection: str, license_: str = "academic") -> Path:
    """Where a collection is cached. The licence is part of the name whenever it is not the
    academic default: OmniPath serves different content under ``license=commercial``, and one
    shared file would hand an academic download to a commercial run (or the reverse)."""
    tag = "" if (license_ or "academic").lower() == "academic" else f".{str(license_).lower()}"
    return directory / (
        f"{PROGENY}.human{tag}.top{DEFAULT_TOP_PROGENY}.csv"
        if collection == PROGENY
        else f"{collection}.human{tag}.gmt"
    )


def read_gmt(path: Path) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        sets[parts[0]] = [g for g in parts[2:] if g]
    if not sets:
        raise ToolError(f"{path} holds no gene sets (a .gmt line is term<TAB>source<TAB>gene...).", "Check the file.")
    return sets


def write_gmt(path: Path, sets: dict[str, list[str]], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w", encoding="utf-8") as fh:
        for term, genes in sets.items():
            fh.write("\t".join([term, source, *genes]) + "\n")
    os.replace(tmp, path)


def _fetch_error(collection: str, path: Path, exc: BaseException) -> ToolError:
    return ToolError(
        f"gene-set collection {collection!r} (human) is not cached at {path} and could not be fetched from "
        f"https://{OMNIPATH_HOST}: {type(exc).__name__}: {exc}",
        f"Pass collections=/path/to/sets.gmt, or set gene_sets_dir / {CACHE_ENV} to a directory holding "
        f"{path.name}, or run once on a host with outbound HTTPS to {OMNIPATH_HOST} to populate the cache. "
        "This tool is installed correctly and does not need reprovisioning.",
    )


def fetch_msigdb_all(directory: Path, license_: str) -> None:
    """ONE OmniPath download writes every MSigDB collection this tool knows, so a single online
    run populates the cache for all of them."""
    import decoupler as dc

    log(f"fetching MSigDB from {OMNIPATH_HOST} (once; cached under {directory})")
    msig = dc.get_resource("MSigDB", organism="human", license=license_)
    present = set(msig["collection"].astype(str).unique())
    for alias, name in COLLECTION_ALIASES.items():
        if name not in present:
            log(f"collection {name!r} is not in this MSigDB release; skipping {alias}")
            continue
        sub = msig[msig["collection"] == name]
        sets: dict[str, list[str]] = {}
        for term, genes in sub.groupby("geneset")["genesymbol"]:
            sets[str(term)] = sorted({str(g) for g in genes if str(g)})
        write_gmt(cache_path(directory, alias, license_), sets, f"msigdb:{name}")


def fetch_progeny(directory: Path, license_: str):
    import decoupler as dc

    log(f"fetching PROGENy from {OMNIPATH_HOST} (once; cached under {directory})")
    net = dc.get_progeny(organism="human", top=DEFAULT_TOP_PROGENY, license=license_)
    path = cache_path(directory, PROGENY, license_)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    net[["source", "target", "weight"]].to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_collection(name: str, directory: Path, license_: str, top_progeny: int = DEFAULT_TOP_PROGENY):
    """A long DataFrame ``source, target, weight`` for one collection, from a .gmt path, the cache,
    or (once) the network. ``fetched`` in the return says whether the network was used."""
    import pandas as pd

    fetched = False
    if name.lower().endswith(".gmt"):
        p = Path(os.path.expanduser(name))
        if not p.is_file():
            raise ToolError(f"gene-set file {name!r} does not exist.", "Pass the path of a .gmt file.")
        sets = read_gmt(p)
        label = p.stem
    elif name == PROGENY:
        path = cache_path(directory, PROGENY, license_)
        if not path.is_file():
            try:
                fetch_progeny(directory, license_)
                fetched = True
            except ToolError:
                raise
            except Exception as exc:
                raise _fetch_error(name, path, exc) from exc
        net = pd.read_csv(path)
        if top_progeny and top_progeny < DEFAULT_TOP_PROGENY:
            net["_abs"] = net["weight"].abs()
            net = (
                net.sort_values(["source", "_abs"], ascending=[True, False])
                .groupby("source")
                .head(int(top_progeny))
                .drop(columns="_abs")
            )
        return net[["source", "target", "weight"]].reset_index(drop=True), PROGENY, fetched
    else:
        if name not in COLLECTION_ALIASES:
            raise ToolError(
                f"unknown collection {name!r}; known: {', '.join([*COLLECTION_ALIASES, PROGENY])}, or a .gmt path.",
                "Pick one of the names listed.",
            )
        path = cache_path(directory, name, license_)
        if not path.is_file():
            try:
                fetch_msigdb_all(directory, license_)
                fetched = True
            except ToolError:
                raise
            except Exception as exc:
                raise _fetch_error(name, path, exc) from exc
            if not path.is_file():
                raise ToolError(
                    f"collection {name!r} is not part of the MSigDB release OmniPath serves right now.",
                    "Pick another collection or pass a local .gmt for this one.",
                )
        sets = read_gmt(path)
        label = name
    rows = [(term, gene, 1.0) for term, genes in sets.items() for gene in genes]
    return pd.DataFrame(rows, columns=["source", "target", "weight"]), label, fetched


def sets_from_net(net) -> dict[str, list[str]]:
    return {str(k): [str(g) for g in v] for k, v in net.groupby("source")["target"]}


# --------------------------------------------------------------------------------------------- #
# task: enrichment
# --------------------------------------------------------------------------------------------- #


def _bh(pvals):
    """Benjamini-Hochberg, for a backend that reports raw p-values only."""
    import numpy as np

    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _restrict(net, universe: set[str] | None, min_size: int, max_size: int):
    """Keep genes in the universe and sets whose size (in the universe) is within bounds."""
    if universe is not None:
        net = net[net["target"].isin(universe)]
    sizes = net.groupby("source")["target"].nunique()
    keep = sizes[(sizes >= min_size) & (sizes <= max_size)].index
    n_dropped = int((~sizes.index.isin(keep)).sum())
    return net[net["source"].isin(keep)].reset_index(drop=True), n_dropped


def run_ora(genes: list[str], net, universe: set[str] | None, n_background: int):
    import decoupler as dc
    import pandas as pd

    query = [g for g in genes if universe is None or g in universe]
    df = pd.DataFrame(index=pd.Index(query, name="gene"))
    res = dc.get_ora_df(df, net, source="source", target="target", n_background=n_background, verbose=False)
    res = res.sort_values(["FDR p-value", "p-value"]) if "FDR p-value" in res.columns else res
    return res.reset_index(drop=True)


def run_gsea_decoupler(scores, net, min_size: int, max_size: int, permutations: int, seed: int):
    import decoupler as dc
    import pandas as pd

    df = pd.DataFrame({"stat": scores.values}, index=scores.index)
    res = dc.get_gsea_df(
        df,
        "stat",
        net,
        source="source",
        target="target",
        times=int(permutations),
        min_n=int(min_size),
        seed=int(seed),
        verbose=False,
    )
    # decoupler 1.9.2 returns Term, ES, NES, "NOM p-value", "FDR p-value", Set size, Tag %, Rank %,
    # Leading edge; both backends are written out with ONE column vocabulary, so a reader never has
    # to know which one ran.
    res = res.rename(columns={"NOM p-value": "p-value", "Rank %": "Gene %"})
    if "FDR p-value" not in res.columns and "p-value" in res.columns:
        res["FDR p-value"] = _bh(res["p-value"].to_numpy())
    return res.sort_values(["FDR p-value", "p-value"]).reset_index(drop=True)


def run_gsea_gseapy(
    scores, sets: dict[str, list[str]], min_size: int, max_size: int, permutations: int, seed: int, threads: int
):
    import gseapy as gp

    pre = gp.prerank(
        rnk=scores,
        gene_sets=sets,
        min_size=int(min_size),
        max_size=int(max_size),
        permutation_num=int(permutations),
        seed=int(seed),
        threads=int(threads),
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    res = pre.res2d.rename(columns={"NOM p-val": "p-value", "FDR q-val": "FDR p-value", "Lead_genes": "Leading edge"})
    keep = [
        c
        for c in ("Term", "ES", "NES", "p-value", "FDR p-value", "Leading edge", "Tag %", "Gene %")
        if c in res.columns
    ]
    res = res[keep]
    for c in ("ES", "NES", "p-value", "FDR p-value"):
        if c in res.columns:
            res[c] = res[c].astype(float)
    return res.sort_values(["FDR p-value", "p-value"]).reset_index(drop=True)


def _dotplot(panels: list[tuple[str, Any]], path: Path, top: int = 15) -> Path | None:
    """-log10 FDR vs overlap/NES for the top terms per collection, one panel each. Matplotlib only."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # pragma: no cover - matplotlib is pinned in the env
        log(f"dotplot skipped: {exc}")
        return None
    panels = [(name, df) for name, df in panels if df is not None and len(df)]
    if not panels:
        return None
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 0.32 * top * len(panels) + 1.5 * len(panels)), squeeze=False)
    for ax, (name, df) in zip(axes[:, 0], panels):
        head = df.head(top).iloc[::-1]
        fdr = np.clip(head["FDR p-value"].to_numpy(dtype=float), 1e-300, 1.0)
        x = -np.log10(fdr)
        size_col = "Overlap ratio" if "Overlap ratio" in head.columns else ("NES" if "NES" in head.columns else None)
        sizes = (
            40 + 160 * np.abs(head[size_col].to_numpy(dtype=float)) / max(1e-9, float(np.abs(head[size_col]).max()))
            if size_col
            else 80
        )
        ax.scatter(x, np.arange(len(head)), s=sizes, c=x, cmap="viridis")
        ax.set_yticks(np.arange(len(head)))
        ax.set_yticklabels([str(t)[:60] for t in head["Term"]], fontsize=8)
        ax.axvline(-math.log10(0.05), color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("-log10(FDR)")
        ax.set_title(name, fontsize=10, loc="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_enrichment(args) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    out = WorkerOutput(TOOL, task="enrichment")
    args.species = normalize_species(args.species)
    seed = int(args.seed)
    np.random.seed(seed)  # both GSEA backends take the seed explicitly below; this covers anything else
    if int(args.permutations) < MIN_PERMUTATIONS:
        raise ToolError(
            f"permutations={args.permutations} cannot estimate a p-value (with 0 every term is 'significant').",
            f"Pass permutations >= {MIN_PERMUTATIONS} (1000 is the default).",
        )
    given = [
        name for name, v in (("gene_table", args.gene_table), ("genes", args.genes), ("h5ad_path", args.h5ad_path)) if v
    ]
    # h5ad_path may ALSO ride along as the universe for a gene_table; it counts as an input only
    # when no table and no inline list was given.
    inputs = [g for g in given if g != "h5ad_path"] or (["h5ad_path"] if args.h5ad_path else [])
    if len(inputs) != 1:
        raise ToolError(
            f"exactly one of gene_table, genes or h5ad_path(+h5ad_key) must supply the gene list; got {given or 'none'}.",
            "Pass gene_table=<ranked CSV/TSV> (preferred), or genes=<short inline list>, or h5ad_path + h5ad_key (+ group).",
        )
    source = inputs[0]
    scores = None
    score_column = ""
    negated = False
    universe: list[str] | None = None
    direction = "as_is"
    if source == "gene_table":
        symbols, scores, score_column, direction = read_gene_table(args.gene_table, args.gene_column, args.score_column)
        negated = direction == "neglog10p"
        if args.h5ad_path and not args.background:
            _, _, universe = read_h5ad_genes_universe(args.h5ad_path)
    elif source == "genes":
        symbols = split_genes(args.genes)
        if len(symbols) > INLINE_GENES_MAX:
            raise ToolError(
                f"the inline gene list holds {len(symbols)} symbols; the inline path is for short ad-hoc lists (<= {INLINE_GENES_MAX}).",
                "Write the list to a file (one symbol per line, or a CSV with a gene column) and pass it as gene_table.",
            )
        # h5ad_path rides along as the universe here too, as it does for a gene_table: an inline
        # list with h5ad_path used to be tested against decoupler's fixed 20,000-gene background
        # instead of the slide's measured genes (hunt 2026-09-30, u30-uncovered-mcp-11).
        if args.h5ad_path and not args.background:
            _, _, universe = read_h5ad_genes_universe(args.h5ad_path)
    else:
        symbols, scores, universe = read_h5ad_genes(args.h5ad_path, args.h5ad_key, args.group)
        score_column = f"uns[{args.h5ad_key!r}].scores" if scores is not None else ""
    if args.background:
        # The background file is the universe whenever it is given, ahead of the h5ad's var_names, so
        # the h5ad beside a gene_table or genes list is not read as one: read, its Ensembl check
        # refused a run whose universe was a symbol background (hunt 2026-09-30, rp-u30 blocker).
        universe = read_background(args.background)
        if args.h5ad_path and source != "h5ad_path":
            out.add_warning(
                f"h5ad_path {args.h5ad_path!r} was not read: background {args.background!r} is the universe and "
                "takes precedence over the h5ad's var_names. Leave background empty to use the h5ad's genes."
            )

    n_supplied = len(symbols)
    genes, n_dupes, case_normalized = normalize_symbols(symbols, args.species)
    if scores is not None:
        # Keep the score of the FIRST occurrence of a symbol, aligned to the de-duplicated list.
        first: dict[str, float] = {}
        for sym, sc in zip(symbols, scores):
            key = str(sym).strip()
            key = key.upper() if case_normalized else key
            if key and key not in first:
                first[key] = float(sc)
        ranked_scores = pd.Series([first.get(g, np.nan) for g in genes], index=genes, dtype=float).dropna()
    else:
        ranked_scores = None
    if not genes:
        raise ToolError("the gene list is empty after cleaning.", "Check the input.")

    universe_set: set[str] | None = None
    background_source = "decoupler default (n_background=20000)"
    if universe:
        if args.background and looks_like_ensembl(universe):
            # Its own message: normalize_symbols blamed "the gene list" and gene_column, which cannot
            # fix a background (hunt 2026-09-30, u30-uncovered-mcp-12).
            raise ToolError(
                f"the background file {args.background!r} holds Ensembl identifiers ({universe[0]}, ...), and the "
                "gene list and the collections are keyed by gene symbol.",
                "Pass a background of gene symbols, one per line, or leave background empty. The gene list is not "
                "the problem.",
            )
        uni, _, _ = normalize_symbols(universe, args.species)
        universe_set = set(uni)
        background_source = "background file" if args.background else "h5ad var_names"
    n_background = len(universe_set) if universe_set else 20000
    if universe_set is not None and not (set(genes) & universe_set):
        raise ToolError(
            f"none of the {len(genes)} genes is in the {background_source} of {len(universe_set)} genes; every test would "
            "run on an empty overlap.",
            "Check that the background and the list use the same symbols (species, case, symbol vs Ensembl id), or "
            "drop the background to use the default universe.",
        )

    directory = cache_dir(args.gene_sets_dir)
    collections = [c.strip() for c in (args.collections or "").split(",") if c.strip()]
    if not collections:
        raise ToolError("no collections named.", "Pass collections=hallmark,go_bp,reactome or a .gmt path.")
    method = args.method.lower()
    if method not in ("ora", "gsea", "both"):
        raise ToolError(f"method {args.method!r} is not one of ora, gsea, both.", "Pick one.")
    want_gsea = method in ("gsea", "both")
    want_ora = method in ("ora", "both")
    if want_gsea and ranked_scores is None:
        if method == "gsea":
            raise ToolError(
                "method=gsea needs a numeric ranking: name a score_column of the table (or an h5ad rank_genes_groups group).",
                "Pass score_column=<numeric column>, or use method=ora.",
            )
        out.add_warning(
            "GSEA was skipped: no score_column was given, so the table has no numeric ranking; ORA ran on the top genes."
        )
        want_gsea = False
    if want_gsea and ranked_scores is not None and len(ranked_scores) < TRUNCATED_RANKING_MIN:
        out.add_warning(
            f"GSEA ran on a ranking of only {len(ranked_scores)} genes (a truncated list, e.g. the top markers of a "
            "group), not a genome-wide ranking; its normalised enrichment scores are relative to that list. Rank all "
            "genes (score every gene, not only the top ones) for a standard GSEA."
        )
    top_n = int(args.top_n) if int(args.top_n) > 0 else len(genes)
    top_genes = genes[:top_n]

    output_dir = Path(args.output_dir or default_output_dir("pathway_enrichment"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "input_genes_used.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
    out.add_output_file("input_genes_used", output_dir / "input_genes_used.txt")

    per_collection: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    panels: list[tuple[str, Any]] = []
    fetched_any = False
    matched_any: set[str] = set()
    threads = cpu_budget(cap=4)
    for name in collections:
        net, label, fetched = load_collection(name, directory, args.license)
        fetched_any = fetched_any or fetched
        if label in per_collection:
            base, k = label, 2
            while f"{base}_{k}" in per_collection:
                k += 1
            label = f"{base}_{k}"
            out.add_warning(f"two collections share the stem {base!r}; the second is reported as {label!r}.")
        # The match rate answers "are these symbols the collection's symbols?" -- so it is read
        # off the collection as fetched, BEFORE the universe and size filters, which drop sets
        # for reasons that have nothing to do with the spelling of the genes.
        in_sets = set(net["target"])
        matched_any |= {g for g in genes if g in in_sets}
        net, n_dropped = _restrict(net, universe_set, int(args.min_set_size), int(args.max_set_size))
        entry: dict[str, Any] = {"n_sets_tested": int(net["source"].nunique()), "n_sets_outside_size_bounds": n_dropped}
        if entry["n_sets_tested"] == 0:
            out.add_warning(
                f"{label}: no gene set has between {args.min_set_size} and {args.max_set_size} genes in the universe; nothing tested."
            )
            per_collection[label] = entry
            continue
        if want_ora:
            ora = run_ora(top_genes, net, universe_set, n_background)
            path = output_dir / f"ora_{label}.csv"
            ora.to_csv(path, index=False)
            out.add_output_file(f"ora_{label}", path)
            sig = ora[ora["FDR p-value"] < 0.05] if "FDR p-value" in ora.columns else ora.iloc[0:0]
            entry["ora_n_significant_fdr05"] = int(len(sig))
            entry["ora_top_terms"] = [str(t) for t in ora["Term"].head(5)]
            panels.append((f"{label} -- ORA (top {len(top_genes)} genes)", ora))
            for _, row in ora.head(25).iterrows():
                summary_rows.append(
                    {
                        "collection": label,
                        "method": "ora",
                        "term": row["Term"],
                        "fdr": float(row["FDR p-value"]),
                        "effect": float(row.get("Odds ratio", np.nan)),
                        "n_overlap": int(str(row.get("Features", "")).count(";") + 1) if row.get("Features") else 0,
                        "leading_edge": str(row.get("Features", "")),
                    }
                )
        if want_gsea:
            try:
                if args.gsea_backend == "gseapy":
                    gsea = run_gsea_gseapy(
                        ranked_scores,
                        sets_from_net(net),
                        int(args.min_set_size),
                        int(args.max_set_size),
                        int(args.permutations),
                        seed,
                        threads,
                    )
                else:
                    gsea = run_gsea_decoupler(
                        ranked_scores,
                        net,
                        int(args.min_set_size),
                        int(args.max_set_size),
                        int(args.permutations),
                        seed,
                    )
            except ToolError:
                raise
            except Exception as exc:
                if method == "gsea":
                    raise
                # ``both``: the ORA above already ran and was written. One backend's exception must
                # not throw that away and report the whole call as failed.
                out.add_warning(
                    f"{label}: GSEA ({args.gsea_backend}) failed ({type(exc).__name__}: {exc}); ORA was kept."
                )
                entry["gsea_error"] = f"{type(exc).__name__}: {exc}"[:300]
                per_collection[label] = entry
                continue
            path = output_dir / f"gsea_{label}.csv"
            gsea.to_csv(path, index=False)
            out.add_output_file(f"gsea_{label}", path)
            sig = gsea[gsea["FDR p-value"] < 0.05]
            entry["gsea_n_significant_fdr05"] = int(len(sig))
            entry["gsea_top_terms"] = [f"{t} (NES {n:+.2f})" for t, n in zip(gsea["Term"].head(5), gsea["NES"].head(5))]
            panels.append((f"{label} -- GSEA-prerank ({args.gsea_backend})", gsea))
            for _, row in gsea.head(25).iterrows():
                le = str(row.get("Leading edge", ""))
                summary_rows.append(
                    {
                        "collection": label,
                        "method": f"gsea:{args.gsea_backend}",
                        "term": row["Term"],
                        "fdr": float(row["FDR p-value"]),
                        "effect": float(row["NES"]),
                        "n_overlap": len([g for g in re.split(r"[;,]", le) if g]),
                        "leading_edge": le,
                    }
                )
        per_collection[label] = entry

    summary = pd.DataFrame(
        summary_rows, columns=["collection", "method", "term", "fdr", "effect", "n_overlap", "leading_edge"]
    )
    summary.to_csv(output_dir / "enrichment_summary.csv", index=False)
    out.add_output_file("enrichment_summary", output_dir / "enrichment_summary.csv")
    fig = _dotplot(panels, output_dir / "enrichment_dotplot.png")
    out.add_output_file("enrichment_dotplot", fig)

    match_rate = len(matched_any) / max(1, len(genes))
    if match_rate < LOW_MATCH_WARN:
        out.add_warning(
            f"only {len(matched_any)}/{len(genes)} genes ({match_rate:.0%}) appear in any of the collections tested; "
            "check the species and that these are gene symbols."
        )
    if case_normalized:
        out.add_warning(
            f"mouse symbols were upper-cased and matched to the human collections ({len(matched_any)}/{len(genes)} matched)."
        )
    if fetched_any:
        out.add_info(
            f"gene sets were fetched from {OMNIPATH_HOST} and cached under {directory}; later runs are offline."
        )
    out.set_data(
        n_genes_supplied=n_supplied,
        n_genes_used=len(genes),
        n_genes_in_ora=len(top_genes),
        n_duplicates_dropped=n_dupes,
        n_matched_to_any_set=len(matched_any),
        n_background=n_background,
        background_source=background_source,
        input_source=source,
    )
    out.add_params(
        {
            "collections": collections,
            "method": method,
            "gsea_backend": args.gsea_backend,
            "top_n": top_n,
            "species": args.species,
            "score_column": score_column,
            "score_is_neg_log10_p": negated,
            "score_direction": direction,
            "min_set_size": int(args.min_set_size),
            "max_set_size": int(args.max_set_size),
            "permutations": int(args.permutations),
            "seed": int(args.seed),
            "license": args.license,
            "gene_sets_dir": str(directory),
            "output_dir": str(output_dir),
        }
    )
    out.set_summary(per_collection=per_collection, symbol_case_normalized=case_normalized)
    lines = []
    for label, entry in per_collection.items():
        bits = []
        if "ora_n_significant_fdr05" in entry:
            bits.append(
                f"ORA: {entry['ora_n_significant_fdr05']} terms at FDR<0.05 of {entry['n_sets_tested']} tested; top: {', '.join(entry['ora_top_terms'][:3])}"
            )
        if "gsea_n_significant_fdr05" in entry:
            bits.append(
                f"GSEA: {entry['gsea_n_significant_fdr05']} at FDR<0.05; top: {', '.join(entry['gsea_top_terms'][:3])}"
            )
        lines.append(f"{label}: " + ("; ".join(bits) if bits else "nothing tested"))
    out.set_analysis(
        f"Enrichment of {len(genes)} {args.species} genes ({source}; universe: {background_source}, n={n_background}). "
        + " | ".join(lines)
        + " MSigDB collections are licence-restricted for commercial use; see license_info.md."
    )
    return out.to_dict()


def read_h5ad_genes_universe(h5ad_path: str):
    """The var_names of an h5ad, read backed, for use as the universe."""
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        universe = [str(v) for v in adata.var_names]
        # Refused here, about the h5ad: the universe went through normalize_symbols, whose message
        # blames "the gene list" and gene_column, so a symbol gene_table that was fine was sent to be
        # changed (hunt 2026-09-30, u30-uncovered-mcp-12).
        if looks_like_ensembl(universe[:200]):
            raise ToolError(
                f"h5ad_path {h5ad_path!r} is the universe, and its var_names are Ensembl identifiers "
                f"({universe[0]}, ...); the gene list and the collections are keyed by gene symbol.",
                ensembl_var_names_remedy(adata.var, h5ad_path)
                + " Or pass background=<a file of symbols> instead of h5ad_path. The gene list is not the problem.",
            )
        return None, None, universe
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------- #
# task: activity
# --------------------------------------------------------------------------------------------- #


def _looks_like_counts(X) -> tuple[bool, float]:
    import numpy as np
    import scipy.sparse as sp

    sample = X[: min(2000, X.shape[0])]
    values = sample.data if sp.issparse(sample) else np.asarray(sample).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False, 0.0
    vmax = float(values.max())
    integer = bool(np.all(np.abs(values - np.round(values)) < 1e-6))
    return integer and vmax > 50, vmax


def _is_integer_valued(X) -> bool:
    import numpy as np
    import scipy.sparse as sp

    sample = X[: min(2000, X.shape[0])]
    values = sample.data if sp.issparse(sample) else np.asarray(sample).ravel()
    values = values[np.isfinite(values)]
    return bool(values.size) and bool(np.all(np.abs(values - np.round(values)) < 1e-6))


def run_activity(args) -> dict[str, Any]:
    import anndata as ad
    import decoupler as dc
    import numpy as np
    import pandas as pd
    import scanpy as sc

    out = WorkerOutput(TOOL, task="activity")
    args.species = normalize_species(args.species)
    seed = int(args.seed)
    np.random.seed(seed)
    if not args.data_path or not Path(args.data_path).is_file():
        raise ToolError(f"data_path {args.data_path!r} does not exist.", "Pass the path of a spatial .h5ad.")
    adata = ad.read_h5ad(args.data_path)
    # Checked before anything is scored or written: a misspelt group_key used to cost the whole run
    # and leave score files the error did not list (hunt 2026-09-30, u30-uncovered-mcp-13).
    if args.group_key and args.group_key not in adata.obs.columns:
        raise ToolError(
            f"group_key {args.group_key!r} is not an obs column; obs columns: {list(adata.obs.columns)[:20]}.",
            "Name an existing obs column or leave it empty.",
        )
    if args.layer:
        if args.layer not in adata.layers:
            raise ToolError(
                f"layer {args.layer!r} is not in the h5ad; layers: {list(adata.layers.keys())}.",
                "Name an existing layer or leave it empty for X.",
            )
        adata.X = adata.layers[args.layer]
    # Background spots (obs['in_tissue'] == 0, which CELLxGENE Visium exports carry) are left out
    # before normalising and scoring, as every other spot-level worker does: they were scored as
    # tissue, formed their own group and set the "most spatially variable" ranking on the
    # tissue-vs-glass contrast (hunt 2026-09-30, u30-uncovered-mcp-5).
    adata, n_spots_input, n_off_tissue = keep_in_tissue(adata, "spots")
    record_in_tissue(out, n_spots_input, n_off_tissue, "spots")
    adata.var_names_make_unique()
    species_mouse = args.species.lower().startswith("mouse")
    if species_mouse:
        adata.var_names = pd.Index([str(v).upper() for v in adata.var_names])
        adata.var_names_make_unique()
    if looks_like_ensembl([str(v) for v in adata.var_names[:200]]):
        raise ToolError(
            "var_names are Ensembl identifiers; the models are keyed by gene symbol.",
            ensembl_var_names_remedy(adata.var, args.data_path),
        )

    counts_like, vmax = _looks_like_counts(adata.X)
    mode = args.normalize.lower()
    normalized = False
    if mode == "always" or (mode == "auto" and counts_like):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        normalized = True
    elif mode == "auto" and not counts_like and _is_integer_valued(adata.X):
        # Integer-valued with a small maximum: a shallow count matrix looks like this, and so does
        # a matrix already scaled to small integers. ``auto`` cannot tell them apart, so it leaves
        # the data alone AND says so, instead of silently scoring raw counts.
        out.add_warning(
            f"X is integer-valued with a maximum of {vmax:g}; normalize=auto treated it as already normalised. "
            "If these are raw counts, pass normalize=always."
        )
    elif mode not in ("auto", "never"):
        raise ToolError(f"normalize {args.normalize!r} is not one of auto, always, never.", "Pick one.")

    directory = cache_dir(args.gene_sets_dir)
    collections = [c.strip() for c in (args.collections or "").split(",") if c.strip()]
    if not collections:
        raise ToolError("no collections named.", "Pass collections=progeny or hallmark, or a .gmt path.")
    method = args.method.lower()
    if method not in ("ulm", "ora"):
        raise ToolError(f"method {args.method!r} is not one of ulm, ora.", "Pick one.")
    nets = []
    fetched_any = False
    for name in collections:
        net, label, fetched = load_collection(name, directory, args.license, top_progeny=int(args.top_progeny))
        fetched_any = fetched_any or fetched
        net = net.copy()
        net["source"] = [f"{label}:{s}" if len(collections) > 1 else str(s) for s in net["source"]]
        nets.append(net)
    net = pd.concat(nets, ignore_index=True).drop_duplicates(["source", "target"])
    var_set = set(map(str, adata.var_names))
    n_in_model = int(net["target"].isin(var_set).sum())
    sizes = net[net["target"].isin(var_set)].groupby("source")["target"].nunique()
    n_pathways_total = int(net["source"].nunique())
    n_pathways_kept = int((sizes >= int(args.min_set_size)).sum())
    if n_pathways_kept == 0:
        raise ToolError(
            f"no pathway has at least {args.min_set_size} of its genes in the data ({len(var_set)} var_names; {n_in_model} model genes present).",
            "Check the species (mouse symbols are upper-cased and matched to the human models) and that var_names are symbols; a small "
            "targeted panel may simply not cover these gene sets -- lower min_set_size or choose smaller sets.",
        )
    if n_pathways_kept < n_pathways_total:
        out.add_warning(
            f"{n_pathways_total - n_pathways_kept} of {n_pathways_total} pathways have fewer than {args.min_set_size} genes in the data and were skipped."
        )

    if method == "ulm":
        dc.run_ulm(
            mat=adata,
            net=net,
            source="source",
            target="target",
            weight="weight",
            min_n=int(args.min_set_size),
            use_raw=False,
            verbose=False,
        )
        est_key, p_key = "ulm_estimate", "ulm_pvals"
    else:
        dc.run_ora(
            mat=adata,
            net=net,
            source="source",
            target="target",
            n_up=int(args.n_up) or None,
            n_bottom=0,
            min_n=int(args.min_set_size),
            seed=seed,
            use_raw=False,
            verbose=False,
        )
        est_key, p_key = "ora_estimate", "ora_pvals"
    acts = dc.get_acts(adata, obsm_key=est_key)
    scores = acts.to_df()
    output_dir = Path(args.output_dir or default_output_dir("pathway_activity"))
    output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_dir / "pathway_activity_scores.csv")
    out.add_output_file("pathway_activity_scores", output_dir / "pathway_activity_scores.csv")
    adata.write_h5ad(output_dir / "pathway_activity.h5ad")
    out.add_output_file("pathway_activity_h5ad", output_dir / "pathway_activity.h5ad")

    by_group = None
    if args.group_key:
        means = scores.groupby(adata.obs[args.group_key].astype(str).values).mean()
        means.index.name = args.group_key
        by_group = means
        try:
            ranked = dc.rank_sources_groups(
                acts, groupby=args.group_key, reference="rest", method="t-test_overestim_var"
            )
            ranked.to_csv(output_dir / "pathway_activity_group_tests.csv", index=False)
            out.add_output_file("pathway_activity_group_tests", output_dir / "pathway_activity_group_tests.csv")
        except Exception as exc:
            out.add_warning(
                f"per-group t-tests skipped ({type(exc).__name__}: {exc}); the per-group means were still written."
            )
        means.to_csv(output_dir / "pathway_activity_by_group.csv")
        out.add_output_file("pathway_activity_by_group", output_dir / "pathway_activity_by_group.csv")

    variances = scores.var(axis=0).sort_values(ascending=False)
    top_k = [str(c) for c in variances.index[: max(1, int(args.top_k_maps))]]
    figure = None
    if "spatial" in adata.obsm:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xy = np.asarray(adata.obsm["spatial"])[:, :2]
            ncol = min(3, len(top_k))
            nrow = int(math.ceil(len(top_k) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
            for ax in axes.ravel():
                ax.set_axis_off()
            for ax, name in zip(axes.ravel(), top_k):
                v = scores[name].to_numpy(dtype=float)
                lim = float(np.nanpercentile(np.abs(v), 98)) or 1.0
                sca = ax.scatter(xy[:, 0], -xy[:, 1], c=v, s=6, cmap="RdBu_r", vmin=-lim, vmax=lim)
                ax.set_title(name[:40], fontsize=9)
                ax.set_aspect("equal")
                fig.colorbar(sca, ax=ax, shrink=0.6)
            fig.suptitle(f"pathway activity ({method}), top {len(top_k)} by variance across spots", fontsize=10)
            fig.tight_layout()
            figure = output_dir / "spatial_pathway_activity.png"
            fig.savefig(figure, dpi=130)
            plt.close(fig)
        except Exception as exc:
            out.add_warning(f"tissue map skipped ({type(exc).__name__}: {exc}).")
    else:
        out.add_warning(
            "obsm['spatial'] is absent, so no tissue map was drawn; the scores and tables were still written."
        )
    out.add_output_file("spatial_pathway_activity", figure)

    if fetched_any:
        out.add_info(
            f"gene sets were fetched from {OMNIPATH_HOST} and cached under {directory}; later runs are offline."
        )
    if species_mouse:
        out.add_warning(
            f"mouse var_names were upper-cased and matched to the human models ({n_in_model} model genes present)."
        )
    out.set_data(
        n_spots=int(adata.n_obs),
        n_spots_input=int(n_spots_input),
        n_genes=int(adata.n_vars),
        n_genes_in_model=n_in_model,
        n_pathways_scored=int(scores.shape[1]),
        normalized=normalized,
        x_max_seen=vmax,
    )
    out.add_params(
        {
            "collections": collections,
            "method": method,
            "species": args.species,
            # The values in FORCE, not the ones typed: the PROGENy model holds 500 genes per
            # pathway at most, and a map count of 0 draws one.
            "top_progeny": min(int(args.top_progeny), DEFAULT_TOP_PROGENY),
            "n_up": int(args.n_up),
            "min_set_size": int(args.min_set_size),
            "layer": args.layer,
            "normalize": mode,
            "group_key": args.group_key,
            "top_k_maps": len(top_k),
            "seed": int(args.seed),
            "license": args.license,
            "gene_sets_dir": str(directory),
            "output_dir": str(output_dir),
            "obsm_keys": [est_key, p_key],
        }
    )
    top_summary = {name: float(scores[name].mean()) for name in top_k}
    out.set_summary(
        top_pathways_by_variance=top_k,
        mean_activity_top=top_summary,
        n_groups=(int(by_group.shape[0]) if by_group is not None else 0),
    )
    out.set_analysis(
        f"Scored {scores.shape[1]} pathways ({', '.join(collections)}) in {adata.n_obs} spots with decoupler {method} "
        f"({'normalized counts' if normalized else 'data used as given'}; {n_in_model} model genes present in the {adata.n_vars}-gene matrix). "
        f"Most spatially variable: {', '.join(top_k[:4])}."
        + (
            f" Per-group means over {by_group.shape[0]} groups of obs[{args.group_key!r}] were written."
            if by_group is not None
            else ""
        )
    )
    return out.to_dict()


# --------------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------------- #

TASKS = {"enrichment": run_enrichment, "activity": run_activity}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    # enrichment inputs
    p.add_argument("--gene-table", default="")
    p.add_argument("--gene-column", default="gene")
    p.add_argument("--score-column", default="")
    p.add_argument("--genes", default="")
    p.add_argument("--h5ad-path", default="")
    p.add_argument("--h5ad-key", default="rank_genes_groups")
    p.add_argument("--group", default="")
    p.add_argument("--top-n", type=int, default=200)
    p.add_argument("--gsea-backend", default="decoupler", choices=("decoupler", "gseapy"))
    p.add_argument("--background", default="")
    p.add_argument("--max-set-size", type=int, default=500)
    p.add_argument("--permutations", type=int, default=1000)
    # activity inputs
    p.add_argument("--data-path", default="")
    p.add_argument("--top-progeny", type=int, default=DEFAULT_TOP_PROGENY)
    p.add_argument("--n-up", type=int, default=0)
    p.add_argument("--layer", default="")
    p.add_argument("--normalize", default="auto")
    p.add_argument("--group-key", default="")
    p.add_argument("--top-k-maps", type=int, default=6)
    # shared
    p.add_argument("--species", default="human")
    p.add_argument("--collections", default="")
    p.add_argument("--method", default="")
    p.add_argument("--min-set-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--license", default="academic")
    p.add_argument("--gene-sets-dir", default="")
    p.add_argument("--output-dir", default="")
    args = p.parse_args(argv)
    if not args.collections:
        args.collections = "progeny" if args.task == "activity" else "hallmark,go_bp,reactome"
    if not args.method:
        args.method = "ulm" if args.task == "activity" else "ora"
    return args


def emit_error_with_diagnostic(message: str, task: str, diagnostic: str = "", exc: BaseException | None = None) -> None:
    """The standard error dict, plus ``diagnostic`` -- the sentence that says what to do."""
    payload = WorkerOutput.error(TOOL, message, task, exc=exc)
    if diagnostic:
        payload["diagnostic"] = diagnostic
    sys.stdout.flush()
    sys.stdout.write("\n" + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    args = parse_args(argv)
    orig_stdout = sys.stdout
    sys.stdout = sys.stderr
    result = None
    failure: tuple[str, str, BaseException | None] | None = None
    try:
        result = TASKS[args.task](args)
    except ToolError as exc:
        failure = (str(exc), exc.diagnostic, None)
    except Exception as exc:
        log(f"ERROR in task {args.task!r}:")
        traceback.print_exc(file=sys.stderr)
        failure = (str(exc) or type(exc).__name__, "", exc)
    finally:
        sys.stdout = orig_stdout
    if result is None:
        message, diagnostic, exc = failure or ("unknown failure", "", None)
        emit_error_with_diagnostic(message, args.task, diagnostic, exc)
        return 1
    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
