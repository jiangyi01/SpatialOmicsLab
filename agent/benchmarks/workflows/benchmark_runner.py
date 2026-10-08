"""Integrated benchmark runner - executes tools through SpatialOmicsLab and evaluates results."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

#: The agent tree this runner lives in (``<checkout>/agent``). ``benchmarks/`` and ``MCP_server/`` sit
#: here, so the config's ``benchmark_data_dir``, ``output_dir`` and ``mcp_config_path`` are read against
#: it. The agent's ``data/`` directory and ``test/test_data`` stay at the repository root -- see
#: ``benchmarks.config.benchmark_config.repository_root``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

#: Said wherever a benchmark run lacks a provider setting, because .env is where it usually lives.
_NO_DOTENV_REMEDY = (
    "Benchmark runs deliberately do not read .env: the runner sets SOG_SKIP_DOTENV=1 so that its own "
    "config, not whichever .env the working directory holds, decides how the agent is set up. Supply "
    "the provider settings in the shell that launches the run: export them (export NAME=...), or "
    "source a key file first (set -a; . /path/to/keys.env; set +a)."
)


def _keep_dotenv_out() -> None:
    """Keep the install .env out of this process, before the agent is imported.

    The agent loads the install .env on import when the process stands in the repo root -- which
    every shipped driver does -- and it loaded AFTER `spatialomicsgym.config` had snapshotted
    os.environ, so its SOG_* knobs reached the os.environ readers (llm source, workers) but not
    default_config: one .env configured the agent two ways inside one run (hunt 2026-09-30,
    uL6-parity-5). Called by the entry points that run a benchmark, not at import: importing this
    module for a helper (the tuning objectives read `_extract_svg_genes`) set it for that whole
    process and every child it spawned.
    """
    if os.environ.get("SOG_SKIP_DOTENV"):
        return
    os.environ["SOG_SKIP_DOTENV"] = "1"
    if "spatialomicsgym.agent.stcoscientist" in sys.modules:
        print(
            "WARNING: the agent was imported before this benchmark run started, so any .env it loaded "
            "is already in this process's environment.",
            file=sys.stderr,
        )


class ProviderKeyMissing(RuntimeError):
    """A benchmark run cannot reach its model's provider from this process's environment."""


def provider_key_problem(llm: str) -> str | None:
    """Why a benchmark turn with *llm* could not reach its provider from this environment, or None.

    Benchmark runs read no .env (see `_keep_dotenv_out`), and the documented setup keeps the keys
    and SOG_SOURCE there, so a run launched as before failed on every pair with the provider SDK's
    own message, which never says why the key is gone (hunt 2026-09-30, uL6-parity-5, review).
    Asked of the resolver `get_llm` uses, with the agent's own defaults; a provider with no key
    (Bedrock's AWS chain, a local or custom endpoint) is not second-guessed here.
    """
    from spatialomicsgym.chat_cli import _PROVIDER_KEY_ENV, _looks_like_placeholder
    from spatialomicsgym.config import default_config
    from spatialomicsgym.llm import effective_source

    try:
        source = effective_source(llm, base_url=default_config.base_url, config=default_config)
    except ValueError as e:
        return (
            f"Cannot tell which provider llm={llm!r} belongs to ({e}); a SOG_SOURCE kept only in .env "
            f"is not read here. {_NO_DOTENV_REMEDY}"
        )
    needed = list(_PROVIDER_KEY_ENV.get(source, ()))
    if source == "AzureOpenAI":
        needed.append("OPENAI_ENDPOINT")
    if source == "Anthropic" and _looks_like_placeholder(os.getenv("ANTHROPIC_API_KEY")):
        # get_llm reads a key kept only in the login profile; so does this check.
        from spatialomicsgym.llm import _load_anthropic_key_from_profile

        _load_anthropic_key_from_profile()
    missing = [name for name in needed if _looks_like_placeholder(os.getenv(name))]
    if not missing:
        return None
    return (
        f"No usable {source} settings for llm={llm!r}: {', '.join(missing)} "
        f"{'is' if len(missing) == 1 else 'are'} not set in this process's environment. {_NO_DOTENV_REMEDY}"
    )


# The shipped drivers launch this file by path (`python benchmarks/workflows/benchmark_runner.py`),
# which puts benchmarks/workflows/ on sys.path[0] and the repo root nowhere. `run_benchmark` then
# does `from benchmarks.data.data_registry import DataRegistry` and dies -- recorded 8 times, each
# one killing a whole run after the agent env was already up. The sibling driver
# benchmarks/run_auto_tuning_benchmark.py has carried this same line all along.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# After the sys.path insert above, for the same reason that comment gives.
from benchmarks.config.benchmark_config import repository_root
from benchmarks.turn_outcome import degrade_note, tool_was_invoked

#: What a rate limit looks like once it has been turned into a degrade note. `_react_error_note`
#: renders the exception's TYPE NAME plus its redacted message, so both the HTTP status and the
#: provider's class name are reachable; matching either is what makes the note as retryable as the
#: exception used to be.
#:
#: The status is matched as the status, not as three digits anywhere: the note carries up to 240
#: characters of the provider's message, and a context-overflow BadRequestError saying "134292
#: tokens" contains "429" -- which re-ran a whole turn twice, tool included, into the same output
#: directory (hunt 2026-09-30, u33a-bench-runner-38).
_RATE_LIMIT_MARKERS = ("rate limit", "ratelimit", "too many requests")
_RATE_LIMIT_STATUS = re.compile(r"\b(?:error code|status code|status|http)\W{0,3}429\b")


def _is_rate_limited(note: str) -> bool:
    """Whether a degrade note describes a rate limit rather than a real stop."""
    lowered = (note or "").lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS) or bool(_RATE_LIMIT_STATUS.search(lowered))


def _sleep_for_rate_limit(attempt: int, max_retries: int) -> None:
    """The existing linear backoff, lifted out so both the raise and the note path share it."""
    wait = 30 * (attempt + 1)
    print(f"  Rate limited (429), waiting {wait}s before retry {attempt + 2}/{max_retries + 1}")
    time.sleep(wait)


#: The agent's mode for a scored turn when the run's config does not name it. Fixed here, never read
#: from the process: the evaluation default followed SOG_EVALUATION_ENABLED, which run_full_benchmark
#: exports and a direct run does not, so the two drivers sent different STAGE 3 text in the scored
#: prompt (hunt 2026-09-30, u33a-bench-runner-10, review). run_full_benchmark states its own value.
_BENCHMARKING_MODE_DEFAULT = True
_EVALUATION_MODE_DEFAULT = False


def _default_benchmark_config() -> dict[str, Any]:
    return {
        "llm": "gpt-5",
        "temperature": 1.0,
        "timeout_seconds": 86400,  # 24 hours
        "max_workers": 1,
        "project_root": str(PROJECT_ROOT),
        "benchmark_data_dir": "benchmarks/benchmark_data",
        "output_dir": "benchmarks/results/benchmark_runs",
        "mcp_config_path": "MCP_server/mcp_config.yaml",
        "benchmarking_enabled": _BENCHMARKING_MODE_DEFAULT,
        "evaluation_enabled": _EVALUATION_MODE_DEFAULT,
    }


#: The keys of the OTHER config schema in benchmarks/config/ -- ``BenchmarkConfig``'s, read by the
#: strategy/runner framework -- which this runner cannot use: its ``data_dir`` and ``output_dir`` mean
#: different directories, and it has no ``llm`` or ``benchmark_data_dir``.
_FRAMEWORK_SCHEMA_KEYS = ("data_dir", "tool_filter", "metrics", "save_intermediate")


def load_benchmark_config(config_path: str | None = None) -> dict[str, Any]:
    """Load benchmark configuration: the YAML laid over the defaults.

    The YAML used to be returned verbatim, so a partial override (only ``llm``) died with KeyError
    'benchmark_data_dir' after the agent env was up, and so did the TUTORIAL's
    ``benchmarks/config/default_config.yaml`` -- a ``BenchmarkConfig`` file, the framework's schema,
    not this runner's (hunt 2026-09-30, u33a-bench-runner-26, uL6-parity-18). That one is refused by
    name rather than merged: its keys would be silently ignored and the run would use defaults.

    ``project_root`` defaults to this checkout's agent tree. A relative one is read against it; the shipped YAMLs
    no longer pin ``/workspace/epic-fermat``, which sent a worktree's or another box's run to read and
    write the main tree's results (hunt 2026-09-30, u33a-bench-runner-27).
    """
    config = _default_benchmark_config()
    if not config_path:
        return config
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: a benchmark config must be a mapping, got {type(raw).__name__}")
    framework_keys = [k for k in _FRAMEWORK_SCHEMA_KEYS if k in raw]
    if framework_keys and "benchmark_data_dir" not in raw:
        raise ValueError(
            f"{config_path} is a BenchmarkConfig file (keys {framework_keys}), the strategy framework's "
            "schema, not this runner's. benchmark_runner reads llm, temperature, timeout_seconds, "
            "project_root, benchmark_data_dir, output_dir and mcp_config_path -- see "
            "benchmarks/config/full_benchmark_config.yaml."
        )
    config.update(raw)
    root = Path(str(config.get("project_root") or PROJECT_ROOT)).expanduser()
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    config["project_root"] = str(root)
    return config


def _svg_candidate_files(output_dir: Path, suffixes: tuple[str, ...], not_before: float | None) -> list[Path]:
    """The files under *output_dir* that can be this run's SVG answer.

    Staged inputs (``csv_inputs/`` -- which the SPVC and SPARK guidance has the agent write into
    output_dir -- and the rest of the inspector's staging list), ``standardized/`` tables a previous
    pass derived, and this system's own post-analysis report are not the tool's answer: rglob'd
    unfiltered, a previous run's ``standardized/<tool>_svg_standardized.csv`` sorted first by name
    and was scored as this run's genes, and with only ``csv_inputs/`` present the first column of
    ``counts.csv`` -- every input gene -- was (hunt 2026-09-30, u33a-bench-runner-14). A file older
    than *not_before* is a previous run's (u33a-bench-runner-13).
    """
    from spatialomicsgym.benchmarking.output_inspector import _is_staged_input_path
    from spatialomicsgym.generated_report import split_report_files

    found = [p for suffix in suffixes for p in output_dir.rglob(f"*{suffix}") if p.is_file()]
    found, _ours = split_report_files(output_dir, found)
    kept = []
    for p in found:
        if _is_staged_input_path(p, root=output_dir):
            continue
        if not_before is not None and p.stat().st_mtime < not_before:
            continue
        kept.append(p)
    return kept


#: How far before a run's ``time.time()`` start a file's mtime may read and still be the run's own:
#: mtimes come from the kernel's coarse clock, which trails time.time() by up to a tick, so a file
#: written in the run's first milliseconds can read as older than its start.
_MTIME_SLACK_SECONDS = 2.0


def _files_written_since(output_dir: Path, not_before: float) -> list[Path]:
    """The tool-output files under *output_dir* written at or after *not_before*, oldest first.

    Staged inputs and this system's own post-analysis report are left out, by the same rules the
    inspector and `_svg_candidate_files` apply.
    """
    from spatialomicsgym.benchmarking.output_inspector import _is_staged_input_path
    from spatialomicsgym.generated_report import split_report_files

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    found = [p for p in output_dir.rglob("*") if p.is_file()]
    found, _ours = split_report_files(output_dir, found)
    fresh = [p for p in found if not _is_staged_input_path(p, root=output_dir) and p.stat().st_mtime >= not_before]
    return sorted(fresh, key=lambda p: p.stat().st_mtime)


def _extract_svg_genes(output_dir: Path, eval_file: Path | None, not_before: float | None = None) -> list[str]:
    """Extract predicted SVG gene names from tool output files.

    Priority order:
    1. predicted_genes.json (curated top-N gene list — preferred when present)
    2. CSV/TSV files with 'Gene' column + FDR/Pval significance filtering
    3. CSV files with gene-related filenames (autocorrelation, svg, genes)
    4. H5AD var annotations with significance flags
    5. JSON output from MCP tools (summary.top_genes or similar fields)

    *not_before* (a ``time.time()``) skips every file written before it: the run's start, so a
    previous run's table in the same output_dir cannot answer for this one.
    """
    import logging

    import pandas as pd

    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)

    # Check for empty output directory
    if not output_dir.exists() or not any(output_dir.iterdir()):
        logger.warning("SVG extraction: output directory %s is empty or does not exist", output_dir)
        return []

    # 0. Prefer the curated predicted_genes.json artifact if present.
    pg_json = output_dir / "predicted_genes.json"
    if pg_json.exists() and (not_before is None or pg_json.stat().st_mtime >= not_before):
        try:
            with open(pg_json) as fh:
                data = json.load(fh)
            if isinstance(data, list) and data:
                genes = [str(g) for g in data if g and not str(g).startswith("Blank-")]
                if genes:
                    return genes
            elif isinstance(data, dict):
                for key in ("predicted_genes", "top_genes", "svg_genes", "genes"):
                    if isinstance(data.get(key), list) and data[key]:
                        return [str(g) for g in data[key] if g and not str(g).startswith("Blank-")]
        except Exception:
            pass

    # 1. Search for gene autocorrelation/SVG result files first (prefer small, specific files)
    gene_file_keywords = ["autocorrelation", "svg", "genes", "variable", "significant", "marker"]
    csv_files = _svg_candidate_files(output_dir, (".csv", ".tsv"), not_before)

    # Sort: files with gene-related names first, then by size (smallest first — gene lists are small)
    def _file_priority(p: Path) -> tuple[int, int]:
        name_lower = p.stem.lower()
        has_keyword = any(kw in name_lower for kw in gene_file_keywords)
        return (0 if has_keyword else 1, p.stat().st_size)

    csv_files.sort(key=_file_priority)

    for csv_path in csv_files:
        try:
            sep = "\t" if csv_path.suffix == ".tsv" else ","
            df = pd.read_csv(csv_path, sep=sep)

            # Look for a column named 'Gene', 'gene', 'gene_name', etc.
            gene_col = None
            for col in df.columns:
                if col.lower() in (
                    "gene",
                    "gene_name",
                    "gene_symbol",
                    "symbol",
                    "feature",
                    "var_name",
                    "genes",
                    "g",  # SpatialDE outputs use 'g' for gene names
                ):
                    gene_col = col
                    break

            # Fallback: if first column is unnamed (likely the index — squidpy writes
            # genes as the index) AND its dtype is string/object, treat it as the gene column.
            if (
                gene_col is None
                and len(df.columns) > 0
                and (df.columns[0] == "Unnamed: 0" or df.columns[0] == "")
                and df.iloc[:, 0].dtype == object
            ):
                gene_col = df.columns[0]

            if gene_col is None:
                continue

            # Filter by significance if FDR/Pval column exists
            sig_col = None
            for col in df.columns:
                if col.lower() in (
                    "fdr",
                    "padj",
                    "p_adj",
                    "q_value",
                    "qvalue",
                    "pval",
                    "p_value",
                    "p_values",
                    "adjusted_pvalue",
                    "adjusted_p_value",
                    "pvalue",
                    "pval_norm",
                ):
                    sig_col = col
                    break

            if sig_col is not None:
                # Coerce rather than test the dtype. R's format.pval writes "<2e-16" for anything
                # under the double floor, which makes the whole column object dtype; the old
                # `dtype in (float, ...)` guard then skipped the filter entirely and fell through
                # to "no significance column -- return all genes" below. One unparseable string
                # turned a precise prediction into every gene in the table. Stripping the
                # inequality keeps "<2e-16" as the significant value it reports.
                pvals = pd.to_numeric(df[sig_col].astype(str).str.strip().str.lstrip("<>= "), errors="coerce")
                sig_genes = df[pvals < 0.05][gene_col].dropna().astype(str).tolist()
                # Filter out 'Blank-*' control probes
                sig_genes = [g for g in sig_genes if not g.startswith("Blank-")]
                if sig_genes:
                    return sig_genes
                if pvals.notna().any():
                    # The threshold ran and selected nothing. That is the tool's answer, and it is
                    # not the same as having no threshold at all: falling through to "return all
                    # genes" below scored recall 1.0 against any ground truth. Only claim this when
                    # at least one p-value parsed -- an all-NaN column means the filter never ran.
                    return []

            # No significance column — return all genes (excluding blanks)
            genes = [g for g in df[gene_col].dropna().astype(str).tolist() if not g.startswith("Blank-")]
            if genes:
                return genes
        except Exception:
            continue

    # 2. Check h5ad var for SVG annotation
    if eval_file and eval_file.exists() and (not_before is None or eval_file.stat().st_mtime >= not_before):
        try:
            from spatialomicsgym.benchmarking.output_inspector import _SCORE_COLUMN_TOP_N, _svg_column_kind
            from spatialomicsgym.utils.file_io import read_h5ad_backed

            # Released on every exit including the returns below; a .raw slot makes the loaded
            # object cyclic, so an implicit release would wait on the garbage collector and leave
            # the file locked against the next writer.
            with read_h5ad_backed(eval_file) as adata:
                for col in adata.var.columns:
                    if any(
                        kw in col.lower() for kw in ["svg", "spatially_variable", "significant", "is_svg", "hotspot"]
                    ):
                        vals = adata.var[col]
                        if vals.dtype == bool:
                            # A bool is a flag, never an ID, so the identifier check below starts
                            # after it: `in_significant_module` is membership and reads correctly.
                            return list(adata.var_names[vals])
                        elif _svg_column_kind(col) == "identifier":
                            # `hotspot_module` and its relatives: numbers that label rather than
                            # rank. Neither branch below means anything over one -- `< 0.05` keeps
                            # modules -1 and 0, `nlargest` keeps the highest-numbered module -- and
                            # `(vals < 1).all()` picks between them on whether the run found more
                            # than one module. Skipped, so the next column gets its turn.
                            continue
                        elif vals.dtype in (float, "float64", "float32"):
                            # `nlargest(...).index` is already an Index of *gene names*, so the old
                            # `adata.var_names[...]` around it raised IndexError -- swallowed by the
                            # bare `except` below, which is why this branch silently returned
                            # nothing. And `(vals < 1).all()` cannot separate a p-value from a
                            # Moran's I: both live in [0, 1]. Only the column name can.
                            kind = _svg_column_kind(col)
                            if kind == "score":
                                # dropna first: nlargest is sort_values().head(n), and sort_values
                                # puts NaN *last* rather than discarding it -- so an all-NaN score
                                # column, a tool that ran and computed nothing, came back as a full
                                # top-N, which is the first N genes in file order wearing a ranking.
                                # The same cutoff the inspector and the standardizer take, so the
                                # three copies of this rule answer the same question the same way.
                                return [str(g) for g in vals.dropna().nlargest(_SCORE_COLUMN_TOP_N).index]
                            # A p-value, or a name that says neither: threshold, which is what both
                            # of the other copies do unconditionally. The `(vals < 1).all()` gate
                            # that stood here was False the moment ONE entry was 1.0 or NaN -- which
                            # is what Benjamini-Hochberg writes for an unremarkable gene, and what a
                            # tool writes for a gene it could not test -- and it then sent the
                            # p-value column to `nlargest`, returning the LARGEST p-values: the
                            # tool's own rejects, in rank order, as its spatially-variable genes.
                            return list(adata.var_names[vals < 0.05])
        except Exception:
            pass

    # 3. Check JSON output from MCP tools (summary.top_genes or similar)
    json_files = _svg_candidate_files(output_dir, (".json",), not_before)
    # Prefer files named "predicted_genes.json" first (the conventional gene-list artifact)
    json_files.sort(key=lambda p: (0 if p.stem == "predicted_genes" else 1, p.stat().st_size))
    for json_path in json_files:
        try:
            with open(json_path) as jf:
                data = json.load(jf)
            genes = None
            # Plain list: ["GeneA", "GeneB", ...]
            if isinstance(data, list):
                genes = [str(g) for g in data if g]
            elif isinstance(data, dict):
                summary = data.get("summary", data)
                for key in (
                    "top_genes",
                    "svg_genes",
                    "spatially_variable_genes",
                    "genes",
                    "significant_genes",
                    "predicted_genes",
                ):
                    if key in summary and isinstance(summary[key], list):
                        genes = [str(g) for g in summary[key] if g]
                        break
            if genes:
                return genes
        except Exception:
            continue

    logger.warning(
        "SVG extraction: no genes found in %s (checked %d CSV/TSV files, %d JSON files, eval_file=%s)",
        output_dir,
        len(csv_files),
        len(json_files),
        eval_file,
    )
    return []


def get_ground_truth_key(dataset_entry: dict[str, Any], task_type: str) -> str:
    """Get ground truth key from dataset registry metadata, with sensible fallbacks."""
    # First: use registry-defined ground_truth_key
    metadata = dataset_entry.get("metadata", {})
    gt_key = metadata.get("ground_truth_key") if isinstance(metadata, dict) else None
    if gt_key:
        return gt_key

    # Second: check top-level ground_truth_key
    gt_key = dataset_entry.get("ground_truth_key")
    if gt_key:
        return gt_key

    # Third: task-type-based defaults
    task_defaults = {
        "spatial_clustering": "cell_type",
        "deconvolution": "cell_type",
        "svg_detection": None,
    }
    return task_defaults.get(task_type, "cell_type")


_TUNING_BLOCK_HEADER = "[Auto-Tuning]"


def _pins_parameter(prompt: str, name: str) -> bool:
    """Whether *prompt* already tells the tool what to pass for *name*.

    Word-boundary matched on the left, so ``epochs`` is not treated as pinned by ``pre_epochs=300``.
    """
    return re.search(rf"(?<!\w){re.escape(name)}\s*=", prompt) is not None


_SC_SUBSAMPLE_ENV = "SOG_SC_SUBSAMPLE_MAX_CELLS"
_SC_SUBSAMPLE_ENV_LEGACY = "BIOMNI_SC_SUBSAMPLE_MAX_CELLS"


def _sc_subsample_max() -> int:
    """How many SC-reference cells to keep, per the opt-in env var; 0 means no subsampling.

    Full-scale benchmarking is the default. Capping spotlight at 10K cells once collapsed
    ``scran::findMarkers`` output into argmax-degenerate predictions (r=-0.006 on visium), so the
    cap is opt-in -- and a value we cannot use must not quietly become one we can.

    Parsing goes through ``config._env_int``, the package's own reader, rather than a further
    inline copy of it: a malformed value warns and keeps 0. The bare ``int()`` this replaces raised
    on five of twelve plausible spellings (``"  "``, ``1e5``, ``50000.0``, ``10k``, ``none``) from
    inside ``run_benchmark_tool``'s try, so every tool in the sweep recorded ``status="fail"``
    under a message naming ``int()`` rather than the variable the user set.

    A negative cap is clamped to 0 for a quieter reason: it matched neither guard downstream --
    not ``== 0`` (the stale-artifact sweep) nor ``> 0`` (the subsampling itself) -- so it turned
    subsampling off *and* skipped the sweep that moves a stale ``sc_reference_subsampled.h5ad``
    aside, leaving a 10K-cell file for a later read to pick up.

    The legacy ``BIOMNI_`` alias keeps the precedence the ``or`` chain gave it: the ``SOG_`` name
    wins whenever it is set to something non-blank.
    """
    from spatialomicsgym.config import _env_int, _env_raw

    name = _SC_SUBSAMPLE_ENV if _env_raw(_SC_SUBSAMPLE_ENV) else _SC_SUBSAMPLE_ENV_LEGACY
    n = _env_int(name, 0)
    if n < 0:
        print(f"WARNING: {name}={n} is negative; treating as 0 (no subsampling).")
        return 0
    return n


def _tuning_prompt_block(tool_name: str, tuned_params: dict[str, Any] | None, prompt: str) -> str:
    """Render tuned hyperparameters as a prompt block, or "" if none of them can be applied.

    The tool is invoked by the agent from the prompt, so the prompt is the only channel a tuned
    value can travel down. Two classes of name are dropped, both because stating them would be
    wrong rather than merely noisy:

    * a name ``search_spaces.yaml`` does not declare tunable. A cached config is a whole baseline,
      most of which no trial ever varied -- and ``n_clusters``/``n_domains`` are derived from the
      dataset registry, not searched.
    * a name the assembled prompt already pins. ``TOOL_PROMPT_GUIDANCE`` hardcodes values for
      wallclock and correctness reasons: STAGATE is held at ``n_epochs=1000`` because 2000 exceeds
      the 1800 s tool cap, and the search space offers exactly 50-2000. A tuned value restated
      after the guidance would win by recency and re-arm the failure the guidance prevents.
    """
    if not tuned_params:
        return ""

    from spatialomicsgym.tuning.parameter_registry import get_tunable_params

    searchable = {spec.name for spec in get_tunable_params(tool_name)}
    applied: dict[str, Any] = {}
    suppressed: list[str] = []
    for name, value in tuned_params.items():
        if name not in searchable:
            suppressed.append(f"{name} (not in the search space)")
        elif _pins_parameter(prompt, name):
            suppressed.append(f"{name} (already pinned by the tool guidance)")
        else:
            applied[name] = value

    if suppressed:
        print(f"    tuning: not applying {', '.join(sorted(suppressed))}")
    if not applied:
        return ""

    lines = "\n".join(f"- {name} = {value!r}" for name, value in sorted(applied.items()))
    return (
        f"\n\n{_TUNING_BLOCK_HEADER} A hyperparameter search selected these values for this tool on "
        f"this dataset. Pass them explicitly in the tool call, and leave every other argument as "
        f"instructed above:\n{lines}\n"
    )


#: Tools whose guidance sets a topic/cell-type count from the reference, so the prompt states it.
#: STDeconvolve's guidance asked for "n_topics=<n_celltypes from registry>", a value no registry
#: entry and no prompt line carried: K was whatever each model invented (hunt 2026-09-30,
#: u33a-bench-runner-7).
_NEEDS_REFERENCE_CELLTYPE_COUNT = frozenset({"run_stdeconvolve"})

#: Markers per reference cell type in the Starfysh signature: the tool's own ``aa_n_markers``
#: default, i.e. as many as it derives per archetype when it has no signature.
_STARFYSH_MARKERS_PER_TYPE = 100

_MISSING_LABELS = frozenset({"", "nan", "none", "na", "<na>"})


def _reference_labels(adata: Any, celltype_key: str, source: Path) -> tuple[list[str], Any]:
    """The distinct non-empty labels of ``obs[celltype_key]``, and each cell's index into them (-1: none)."""
    import numpy as np

    if celltype_key not in adata.obs.columns:
        raise KeyError(f"obs[{celltype_key!r}] is not a column of {source}")
    raw = adata.obs[celltype_key].astype(object)
    text = [None if v is None or (isinstance(v, float) and v != v) else str(v).strip() for v in raw]
    cats = sorted({t for t in text if t is not None and t.lower() not in _MISSING_LABELS})
    index = {c: i for i, c in enumerate(cats)}
    codes = np.array([index.get(t, -1) if t is not None else -1 for t in text], dtype=np.int64)
    return cats, codes


def _reference_celltype_count_line(sc_ref: Path, celltype_key: str | None) -> str:
    """The prompt line stating how many cell types the reference labels, or how to find out."""
    if celltype_key:
        try:
            from spatialomicsgym.utils.file_io import read_h5ad_backed

            with read_h5ad_backed(str(sc_ref)) as adata:
                cats, _ = _reference_labels(adata, celltype_key, sc_ref)
            return f"Reference cell types: {len(cats)} distinct labels in obs['{celltype_key}'].\n"
        except Exception as e:
            reason = f"could not be read ({type(e).__name__}: {e})"[:300]
    else:
        reason = "is not named in the registry"
    return (
        f"Reference cell types: the count {reason}; count the distinct non-empty labels of the "
        "reference's cell-type column and use that integer.\n"
    )


def _starfysh_marker_signature(
    sc_ref: Path,
    celltype_key: str,
    spatial_h5ad: Path,
    out_csv: Path,
    n_markers: int = _STARFYSH_MARKERS_PER_TYPE,
    chunk_rows: int = 4096,
) -> dict[str, Any]:
    """Write a Starfysh ``signature_csv`` from the single-cell reference: top markers per cell type.

    Starfysh with no signature runs Archetypal Analysis, and its proportion columns are then
    ``arch_<i>`` factors -- which the output inspector rejects by design, so every guided run ended
    ``output_invalid``, and archetypes cannot be scored against named cell types anyway (hunt
    2026-09-30, u33a-bench-runner-6). The benchmark therefore derives the signature, the same way
    for every model: each reference cell is library-size normalised and log1p'd (Starfysh's own
    treatment of the spatial counts), and a cell type's markers are the genes whose mean in that
    type most exceeds their mean over every other labelled cell. Only genes the spatial counts
    express in at least one spot, and not MT-/RP- genes (which Starfysh's preprocessing drops),
    are eligible -- the worker refuses a signature that names no gene of the counts. Read in row
    chunks, so the reference is never densified whole. Written atomically.

    Raises when a marker table cannot be built -- the caller refuses the run rather than letting
    Starfysh fall back to archetypes.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from spatialomicsgym.utils.file_io import read_h5ad_backed

    def _rows(adata: Any, start: int) -> Any:
        block = adata.X[start : start + chunk_rows]
        return block.tocsr() if sp.issparse(block) else sp.csr_matrix(np.asarray(block))

    with read_h5ad_backed(str(spatial_h5ad)) as st:
        st_genes = pd.Index(st.var_names.astype(str))
        expressed = np.zeros(st.n_vars, dtype=bool)
        for start in range(0, st.n_obs, chunk_rows):
            expressed |= _rows(st, start).getnnz(axis=0) > 0
    eligible = {g for g in st_genes[expressed] if not g.upper().startswith(("MT-", "RP-"))}
    if not eligible:
        raise ValueError(f"{spatial_h5ad} expresses no gene a Starfysh signature could name")

    with read_h5ad_backed(str(sc_ref)) as ref:
        cats, codes = _reference_labels(ref, celltype_key, sc_ref)
        if len(cats) < 2:
            raise ValueError(f"obs[{celltype_key!r}] of {sc_ref} has {len(cats)} cell type(s); a signature needs two")
        ref_genes = pd.Index(ref.var_names.astype(str))
        usable = np.asarray(ref_genes.isin(list(eligible)))
        if not usable.any():
            raise ValueError(f"{sc_ref} shares no expressed gene with {spatial_h5ad}")
        sums = np.zeros((len(cats), ref.n_vars), dtype=np.float64)
        n_cells = np.zeros(len(cats), dtype=np.float64)
        for start in range(0, ref.n_obs, chunk_rows):
            block = _rows(ref, start)
            chunk_codes = codes[start : start + block.shape[0]]
            labelled = chunk_codes >= 0
            if not labelled.any():
                continue
            totals = np.asarray(block.sum(axis=1)).ravel().astype(np.float64)
            scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
            norm = (sp.diags(scale) @ block).tocsr()
            norm.data = np.log1p(norm.data)
            rows = np.flatnonzero(labelled)
            membership = sp.csr_matrix(
                (np.ones(rows.size), (chunk_codes[labelled], rows)), shape=(len(cats), block.shape[0])
            )
            sums += np.asarray((membership @ norm).todense())
            n_cells += np.bincount(chunk_codes[labelled], minlength=len(cats))

    inside = sums / np.maximum(n_cells, 1)[:, None]
    outside = (sums.sum(axis=0)[None, :] - sums) / np.maximum(n_cells.sum() - n_cells, 1)[:, None]
    score = inside - outside
    score[:, ~usable] = -np.inf
    columns: dict[str, list[str]] = {}
    for i, cat in enumerate(cats):
        order = np.argsort(-score[i], kind="stable")[:n_markers]
        markers = [str(ref_genes[j]) for j in order if score[i, j] > 0]
        if not markers:
            raise ValueError(f"cell type {cat!r} in obs[{celltype_key!r}] has no gene above the other types")
        columns[cat] = markers

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    partial = out_csv.with_name(out_csv.name + ".partial")
    pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in columns.items()}).to_csv(partial, index=False)
    os.replace(partial, out_csv)
    return {
        "path": str(out_csv),
        "celltype_key": celltype_key,
        "n_cell_types": len(cats),
        "n_markers_per_type": n_markers,
    }


def run_benchmark_tool(
    tool_name: str,
    dataset_entry: dict[str, Any],
    task_type: str,
    config: dict[str, Any],
    tuned_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single tool on a dataset through SpatialOmicsLab and evaluate.

    *tuned_params* is the winning configuration from a hyperparameter search, or None for an
    untuned run -- the path every recorded benchmark took, whose prompt this must leave untouched.

    Returns a result dict with execution status, metrics, and timing.
    """
    _keep_dotenv_out()
    from spatialomicsgym.benchmarking.tool_output_registry import get_profile as _get_tool_profile

    result = {
        "tool_name": tool_name,
        "dataset_name": dataset_entry["name"],
        "task_type": task_type,
        "status": "pending",
        "metrics": {},
        "elapsed_seconds": 0.0,
        "error": "",
        "is_user_tool": _get_tool_profile(tool_name) is None,
    }

    # Tools that require CSV input (R-based tools that can't read h5ad)
    CSV_INPUT_TOOLS = {
        "precast_spatial_clustering": {
            "note": "Requires counts CSV (genes x spots) and coordinates CSV (spots x 2). "
            "Call convert_h5ad_to_csv first. Pass results as --counts-csvs and --coords-csvs.",
        },
        "run_bass": {
            "note": "Requires counts CSV (genes x spots) and coordinates CSV (spot_id, x, y). "
            "Call convert_h5ad_to_csv first. Pass results as --spatial-counts-csv and --spatial-coords-csv.",
        },
        "run_iris": {
            "note": "Requires counts CSV (genes x spots) and coordinates CSV (spot_id, x, y). "
            "Call convert_h5ad_to_csv first. Pass results as --spatial-counts-csv and --spatial-coords-csv.",
        },
        # No spvc_svg_detection entry: its TOOL_PROMPT_GUIDANCE has the agent write the two CSVs
        # itself and says not to rely on convert_h5ad_to_csv, and this note said the opposite in
        # the same prompt (hunt 2026-09-30, u33a-bench-runner-9).
        "run_spotlight": {
            "note": "R-based NMF deconvolution. Requires spatial counts CSV, coordinates CSV, "
            "reference counts CSV, and reference cell types CSV. "
            "Call convert_h5ad_to_csv on both spatial and sc_reference h5ad files first.",
        },
        "run_card": {
            "note": "R-based CARD deconvolution. Requires spatial counts CSV, coordinates CSV, "
            "reference counts CSV, and reference cell types CSV. "
            "Call convert_h5ad_to_csv on both spatial and sc_reference h5ad files first.",
        },
        "run_stdeconvolve": {
            "note": "R-based STdeconvolve. Requires spatial counts CSV (genes x spots). "
            "Call convert_h5ad_to_csv first. Pass spatial counts CSV to the tool.",
        },
        "spark_svg_detection": {
            "note": "R-based SPARK. Requires counts CSV (genes x spots) and coords CSV. "
            "Call convert_h5ad_to_csv first. Pass results as --counts-csv and --coords-csv.",
        },
    }

    project_root = Path(config["project_root"])
    # data/ and test/test_data are repository-root trees; benchmarks/ and MCP_server/ are agent-side.
    repo_root = repository_root(project_root)
    data_dir = project_root / config["benchmark_data_dir"]
    output_dir = project_root / config["output_dir"] / tool_name / dataset_entry["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale outputs from previous runs (>1 hour old).
    # Includes h5ad outputs AND tool-specific prediction CSVs that the picker
    # might pick up by mistake (e.g., a leftover spacexr_weights.csv from a
    # previous dataset's Visium run reused under a slide-seq output_dir).
    import time as _time

    _stale_cutoff = _time.time() - 3600
    for stale in output_dir.glob("*.h5ad"):
        if stale.name.startswith("input") or stale.name.startswith("prepared"):
            continue
        if stale.stat().st_mtime < _stale_cutoff:
            stale.unlink()
    # Tool prediction CSVs that the file-picker uses; safe to drop because
    # the new tool run will rewrite them. Only top-level (not in csv_export/).
    _PRED_CSV_GLOBS = (
        "*_weights.csv",
        "*_proportions.csv",
        "*_theta.csv",
        "*_beta.csv",
        "spacexr_*.csv",
        "card_*.csv",
        "spotlight_*.csv",
        "stdeconvolve_*.csv",
        "destvi_*.csv",
        "tacco_*.csv",
        "tangram_*.csv",
        "starfysh_*.csv",
        "spatialprompt_*.csv",
        "celldart_*.csv",
        "spatialscope_*.csv",
        "cell2location_*.csv",
        "bulk2space_*.csv",
        "prediction.csv",
    )
    for pat in _PRED_CSV_GLOBS:
        for stale in output_dir.glob(pat):
            if stale.is_file() and stale.stat().st_mtime < _stale_cutoff:
                stale.unlink()
    # Stale SVG gene list: the priority-0 extractor trusts predicted_genes.json
    # unconditionally, so a leftover from a previous (possibly failed) run of the
    # same tool+dataset could be scored as this run's output. Drop it if stale.
    _stale_pred_genes = output_dir / "predicted_genes.json"
    if _stale_pred_genes.is_file() and _stale_pred_genes.stat().st_mtime < _stale_cutoff:
        _stale_pred_genes.unlink()

    input_path = data_dir / dataset_entry["input_path"]
    if not input_path.exists():
        # Try test_data
        input_path = repo_root / "test" / "test_data" / dataset_entry["input_path"]

    gt_path = data_dir / dataset_entry.get("ground_truth_path", "")
    if not gt_path.exists():
        gt_path = input_path  # ground truth in same file

    # Centralized SVG data prep: every SVG tool sees the same HVG-filtered AnnData.
    # No per-tool / per-prompt HVG selection downstream. Cached per-dataset.
    #
    # Exception: tools whose per-gene significance test is calibrated against the
    # full transcriptome (BSP, SpaGFT, SpVC, SpatialDE) lose sensitivity on the
    # pre-filtered HVG set. They must see the full 32k-gene catalog to match the
    # manual baseline (which ran on the unfiltered input). The evaluator records the
    # HVG-restricted F1 (f1_hvg) beside the raw F1 for the apples-to-apples comparison.
    _SVG_FULL_TRANSCRIPTOME_TOOLS = {
        "bsp_identify_svg",
        "spagft_identify_svg",
        "spvc_svg_detection",
        "spatialde_run_svg",
        "svca_variance_decomposition",
    }
    start = time.time()
    saved_modes: tuple[bool, bool] | None = None
    default_config = None

    try:
        from spatialomicsgym.config import default_config

        # One accessor for the model: this line read config.get("llm", "gpt-5") and the agent
        # constructor read config["llm"], two answers for one setting (hunt 2026-09-30, uL6-parity-18).
        llm = config.get("llm", "gpt-5")
        default_config.temperature = config.get("temperature", 1.0)
        default_config.llm = llm
        default_config.timeout_seconds = config.get("timeout_seconds", 86400)  # 24 hours

        # The agent's mode is set here, and recorded, rather than inherited from whichever driver
        # imported this function. run_full_benchmark exported SOG_BENCHMARKING_ENABLED first; the
        # multi-LLM drivers, run_all.sh and the TUTORIAL call this file directly and never did, so
        # the same scorer ran the agent with post-analysis, rescue and env-fallback in one arm and
        # with the mandatory inspection block in the other (hunt 2026-09-30, u33a-bench-runner-10,
        # uL6-parity-4). ``benchmarking_enabled: false`` in the config is still honoured -- it is how
        # an arm recorded before this is reproduced -- and either way the result says which it was.
        # Evaluation mode too comes from the config or this module's default, not the process.
        saved_modes = (default_config.benchmarking_enabled, default_config.evaluation_enabled)
        default_config.benchmarking_enabled = bool(config.get("benchmarking_enabled", _BENCHMARKING_MODE_DEFAULT))
        default_config.evaluation_enabled = bool(config.get("evaluation_enabled", _EVALUATION_MODE_DEFAULT))
        result["benchmarking_enabled"] = default_config.benchmarking_enabled
        result["evaluation_enabled"] = default_config.evaluation_enabled

        # Before anything is built: a missing key otherwise surfaced as the provider SDK's own
        # message, which cannot say that this run deliberately read no .env (hunt 2026-09-30,
        # uL6-parity-5, review).
        key_problem = provider_key_problem(llm)
        if key_problem:
            result["status"] = "fail"
            result["error"] = key_problem
            result["elapsed_seconds"] = time.time() - start
            return result

        # Inside the try, as a missing clustering input already was: a missing or unreadable SVG
        # input raised out of here and out of run_benchmark's loop, killing the sweep before its
        # summary was written (hunt 2026-09-30, u33a-bench-runner-12).
        if task_type == "svg_detection" and tool_name not in _SVG_FULL_TRANSCRIPTOME_TOOLS:
            from spatialomicsgym.benchmarking.svg_input_prep import prepare_svg_input_hvg

            try:
                input_path = prepare_svg_input_hvg(input_path)
            except Exception as e:
                result["status"] = "input_invalid"
                result["error"] = f"SVG input preparation failed for {input_path}: {type(e).__name__}: {e}"[:500]
                result["elapsed_seconds"] = time.time() - start
                return result

        # Model-written cells run in one process-global REPL namespace that nothing reset between
        # pairs, so tool N+1 on another dataset could read tool N's `adata` or `csv_dir` -- results
        # depended on sweep order (hunt 2026-09-30, u33a-bench-runner-11).
        from spatialomicsgym.tool.support_tools import reset_repl_namespace

        reset_repl_namespace()

        from spatialomicsgym.agent import STCoscientist

        agent_data_dir = repo_root / "data"
        agent_data_dir.mkdir(exist_ok=True)
        agent = STCoscientist(path=str(agent_data_dir), llm=llm)

        # The user-tool layer is wired exactly when this run asks for it. add_mcp(path) alone merged
        # it whenever SOG_TOOL_CREATION_ENABLED was on in the operator's shell, so that variable
        # decided the scored namespace (hunt 2026-09-30, uL6-parity-16). add_mcp does the merge from
        # the install's user layer -- where tool creation writes it, not a repo-anchored literal
        # (u33a-bench-runner-37) -- and falls back to the base config if the merged one will not
        # wire; merging here as well merged the layer twice.
        mcp_path = str(project_root / config["mcp_config_path"])
        include_user_tools = bool(config.get("include_user_tools", default_config.benchmark_user_tools))
        agent.add_mcp(mcp_path, merge_user=include_user_tools)
        # add_mcp catches its own failure and returns; the turn then ran with no tools and was
        # recorded "fail_prompt: Tool was not invoked by agent" -- an environment fault counted as
        # a prompting failure (hunt 2026-09-30, uL6-parity-15).
        wiring_error = getattr(agent, "_mcp_wiring_error", None)
        if wiring_error:
            result["status"] = "fail"
            result["error"] = f"MCP wiring failed before the turn ({mcp_path}): {wiring_error}"[:500]
            result["elapsed_seconds"] = time.time() - start
            return result

        # Build prompt with data readiness instructions
        prompt = (
            f"Use the {tool_name} MCP tool on the benchmark dataset.\n\n"
            f"Input data: {input_path}\n"
            f"Output directory: {output_dir}\n\n"
            f"Run the {tool_name} MCP tool directly on the inputs above. "
            f"Only if the tool errors due to data schema, inspect the h5ad and fix mismatches, then re-run.\n"
            f"After running: report the output files produced.\n"
        )

        # CSV input tool guidance
        if tool_name in CSV_INPUT_TOOLS:
            csv_info = CSV_INPUT_TOOLS[tool_name]
            prompt += f"\n\nIMPORTANT: {csv_info['note']}\nThe convert_h5ad_to_csv tool is available as an MCP tool.\n"

        sc_ref = None  # Initialize for all task types (used by pre-execution gate)
        if task_type == "deconvolution":
            # First: use sc_reference_path from registry metadata if available
            metadata = dataset_entry.get("metadata", {})
            registry_sc_path = metadata.get("sc_reference_path") if isinstance(metadata, dict) else None
            if registry_sc_path:
                sc_ref = data_dir / registry_sc_path
                if not sc_ref.exists():
                    sc_ref = None

            # Fallback: brittle string replacement for legacy datasets
            if sc_ref is None:
                sc_ref_candidate = data_dir / dataset_entry.get("input_path", "").replace(
                    "spatial_deconvolution", "sc_reference"
                ).replace("spatial_deconvolution.h5ad", "sc_reference.h5ad")
                if sc_ref_candidate.exists():
                    sc_ref = sc_ref_candidate

            # SC reference subsampling: opt-in via env var SOG_SC_SUBSAMPLE_MAX_CELLS.
            # Default behavior is full-scale benchmarking (no subsampling).
            # Past behavior subsampled spotlight to 10K cells, which collapsed scran::findMarkers
            # output and produced argmax-degenerate predictions (r=-0.006 on visium).
            # A value we cannot parse warns and reads as 0 -- see _sc_subsample_max.
            sc_subsample_max = _sc_subsample_max()

            # Defensive cleanup: when subsampling is disabled, sweep stale subsampled
            # SC ref artifacts from prior subsample-on runs. Otherwise STCoscientist (or downstream
            # convert_h5ad_to_csv) may pick up a stale 10K-cell file and produce
            # argmax-degenerate output. Rename rather than delete so audit is possible.
            if sc_subsample_max == 0 and output_dir is not None:
                import time as _time

                stale_names = ("sc_reference_subsampled.h5ad", "sc_reference_with_CellType.h5ad")
                ts = _time.strftime("%Y%m%d_%H%M%S")
                for nm in stale_names:
                    p = Path(output_dir) / nm
                    if p.exists() and p.stat().st_size > 0:
                        bak = p.with_suffix(p.suffix + f".bak_{ts}_subsample_off")
                        try:
                            p.rename(bak)
                            print(f"  [cleanup] moved stale subsampled SC ref → {bak.name}")
                        except OSError as e:
                            print(f"  WARNING: failed to move stale SC ref {p}: {e}")

            if sc_ref is not None and sc_subsample_max > 0:
                try:
                    import anndata as ad
                    import numpy as np

                    sc_adata = ad.read_h5ad(str(sc_ref))
                    if sc_adata.n_obs > sc_subsample_max:
                        ct_key = metadata.get("sc_reference_celltype_key", "cell_type")
                        rng = np.random.RandomState(42)
                        if ct_key in sc_adata.obs.columns:
                            labels = sc_adata.obs[ct_key]
                            categories = labels.unique()
                            per_cat = max(2, sc_subsample_max // len(categories))
                            indices = []
                            for cat in categories:
                                cat_idx = labels[labels == cat].index.tolist()
                                n = min(per_cat, len(cat_idx))
                                indices.extend(rng.choice(cat_idx, n, replace=False).tolist())
                            if len(indices) > sc_subsample_max:
                                indices = rng.choice(indices, sc_subsample_max, replace=False).tolist()
                            sc_sub = sc_adata[indices].copy()
                        else:
                            idx = rng.choice(sc_adata.n_obs, sc_subsample_max, replace=False)
                            sc_sub = sc_adata[idx].copy()

                        sub_path = output_dir / "sc_reference_subsampled.h5ad"
                        sc_sub.write_h5ad(str(sub_path))
                        sc_ref = sub_path
                        print(
                            f"  [opt-in subsample] sc_ref → {sc_sub.n_obs} cells (env SOG_SC_SUBSAMPLE_MAX_CELLS={sc_subsample_max})"
                        )
                except Exception as e:
                    print(f"  WARNING: sc_ref subsampling failed: {e}")
                    sub_path = output_dir / "sc_reference_subsampled.h5ad"
                    if sub_path.exists() and sub_path.stat().st_size == 0:
                        sub_path.unlink()

            sc_celltype_key = metadata.get("sc_reference_celltype_key") if isinstance(metadata, dict) else None
            if sc_ref is not None:
                prompt += f"\nSingle-cell reference: {sc_ref}\n"
                if sc_celltype_key:
                    prompt += f"Reference cell type key: obs['{sc_celltype_key}']\n"
                if tool_name in _NEEDS_REFERENCE_CELLTYPE_COUNT:
                    prompt += _reference_celltype_count_line(sc_ref, sc_celltype_key)

            if tool_name == "starfysh_deconvolution":
                # Scored only with a reference-derived signature; see _starfysh_marker_signature.
                # Refused rather than run when one cannot be built: without it Starfysh falls back
                # to archetypes, which nothing downstream can score.
                try:
                    if sc_ref is None or not sc_celltype_key:
                        raise ValueError(
                            "the dataset names no single-cell reference and cell-type key to derive "
                            "Starfysh's marker signature from (registry metadata sc_reference_path, "
                            "sc_reference_celltype_key)"
                        )
                    signature = _starfysh_marker_signature(
                        sc_ref,
                        sc_celltype_key,
                        input_path,
                        output_dir / "csv_inputs" / "starfysh_signature.csv",
                    )
                except Exception as e:
                    result["status"] = "input_invalid"
                    result["error"] = f"Starfysh marker signature could not be built: {type(e).__name__}: {e}"[:500]
                    result["elapsed_seconds"] = time.time() - start
                    return result
                result["starfysh_signature"] = signature
                prompt += (
                    f"Marker signature: {signature['path']} (the top {signature['n_markers_per_type']} marker "
                    f"genes of each of the {signature['n_cell_types']} reference cell types in "
                    f"obs['{signature['celltype_key']}'], one column per cell type)\n"
                )

        # Slide-seqV2 platform-specific guidance
        platform = dataset_entry.get("platform", "")
        if platform == "Slide-seqV2":
            prompt += "\nThis is Slide-seqV2 data with no histology images.\n"
            prompt += "The ground truth labels are in obs['cell_type'].\n"

        # Tool-specific prompt guidance.
        #
        # Scored prompt text. Changed 2026-09-30 only where it was verified wrong against the tools
        # it names: convert_h5ad_to_csv's slide kwarg is h5ad_path, not input= (FastMCP refused the
        # call), and the reference conversion passes cell_type_key, without which the MERFISH
        # reference (subclass_label) wrote no celltypes.csv (u33a-bench-runner-4); run_iris reads
        # the converter's counts.csv / coordinates.csv, not spatial_*.csv (-5); Starfysh gets the
        # reference-derived signature (-6); STDeconvolve's n_topics points at a line the prompt now
        # carries (-7); cell2location's Slide-seqV2 batch is the reference's library_uuid (38
        # libraries), not the cell-type label it was confounded with (-8). Each kwarg named in a
        # call here is checked against the portal's declaration by
        # test/test_the_benchmark_guidance_calls_the_tools_the_way_they_are_declared.py.
        TOOL_PROMPT_GUIDANCE = {
            "starfysh_deconvolution": (
                "ACTION FIRST: Call starfysh_deconvolution MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Pass: input_mode='generic_counts_coords', counts_h5ad=<input_path>, "
                "coords_csv=<extract obsm['spatial'] from input h5ad to 2-col x,y CSV first>, "
                "output_dir=<output_dir>, signature_mode='provided', signature_csv=<the Marker signature path "
                "above>, use_poe=False (no H&E image is staged in this "
                "mode and PoE needs one), n_repeats=1, epochs=100. "
                "Just call ONCE."
            ),
            "deepst_identify_domains": (
                "ACTION FIRST: Call the deepst_identify_domains MCP tool DIRECTLY before any analysis. "
                "DO NOT produce a planning <solution> checklist before invoking the tool — the tool call MUST be the first execute action in your response. "
                "DO NOT call repair_spatial_h5ad, validate_spatial_h5ad, or any pre-processing helper first — DeepST handles input validation internally. "
                "DO NOT write `from mcp_servers...import deepst_identify_domains` (that import path does not exist); invoke the tool via the standard MCP tool-call mechanism that the agent runtime exposes. "
                "Pass: st_h5ad=<input_path>, output_dir=<output_dir>, "
                "n_domains=<EXACT integer from the 'expected number of clusters/domains' line above — do NOT round, omit, or substitute>, "
                "pre_epochs=300, epochs=300 (defaults are 500/500, which exceed the 1800s tool wallclock cap on Visium and trigger OUTPUT_INVALID even though the worker keeps running and produces orphan output). "
                "Just call the MCP tool ONCE with these args and verify the output dir."
            ),
            "stagate_spatial_domains": (
                "ACTION FIRST: Call the stagate_spatial_domains MCP tool DIRECTLY before any analysis. "
                "DO NOT produce a planning <solution> checklist before invoking the tool — the tool call MUST be the first execute action in your response. "
                "DO NOT call repair_spatial_h5ad, validate_spatial_h5ad, or any pre-processing helper first — STAGATE handles input validation internally. "
                "DO NOT write `from mcp_servers.stagate import stagate_spatial_domains` (that import path does not exist); invoke the tool via the standard MCP tool-call mechanism that the agent runtime exposes. "
                "Pass: st_h5ad=<input_path>, output_dir=<output_dir>, "
                "n_clusters=<EXACT integer from the 'expected number of clusters/domains' line above>, "
                "rad_cutoff=0 (the worker derives the radius from the slide's own spot spacing -- on DLPFC the same six-neighbour graph as 150; a fixed 150 links no spots on slides spaced ~260-370 px), n_epochs=1000, device='auto', seed=0. "
                "n_epochs=1000 is the worker default and fits in the tool wallclock budget on CPU "
                "(at ~1.3 s/iter on Visium, 2000 epochs takes ~45 min and exceeds the 1800 s tool timeout, "
                "causing STCoscientist to fabricate a 'timed out' solution while the worker continues writing outputs in the background). "
                "Do NOT manually preprocess, build neighbor graphs, or write clustering code. "
                "STAGATE handles the full pipeline: spatial graph -> autoencoder -> KMeans clustering. "
                "Just call the MCP tool ONCE with these args and verify the output dir."
            ),
            "run_stdeconvolve": (
                "ACTION FIRST: Call run_stdeconvolve MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Step 1: convert_h5ad_to_csv on <input_path> → counts CSV (genes x spots). "
                "Step 2: run_stdeconvolve(spatial_counts_csv=<csv>, output_dir=<output_dir>, "
                "n_topics=<the integer on the 'Reference cell types' line above>, n_top_genes=1000, seed=42). "
                "Just call ONCE."
            ),
            "spatialprompt_deconvolution": (
                "ACTION FIRST: Call spatialprompt_deconvolution MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Pass: input_mode='h5ad', spatial_h5ad=<input_path>, sc_h5ad=<sc_reference_path>, "
                "sc_label_key=<sc_reference_celltype_key>, output_dir=<output_dir>, "
                "max_genes=2000, min_counts=1, random_seed=0. "
                "Just call ONCE."
            ),
            "spatialprompt_cluster": (
                "ACTION FIRST: Call the spatialprompt_cluster MCP tool DIRECTLY before any analysis. "
                "Pass: input_mode='h5ad', spatial_h5ad=<input_path>, output_dir=<output_dir>, "
                "n_clust=<EXACT integer from the 'expected number of clusters/domains' line above>, "
                "clust_label='spatialprompt_cluster', random_seed=0. "
                "Slide kwarg: spatial_h5ad. The aliases st_h5ad and h5ad_path are declared too and "
                "reach the same slide, so either is safe if you already have one in hand. Do not invent "
                "any other name (h5ad, input_path, adata_path): those are undeclared, so the call raises "
                "ValidationError 'Unexpected keyword argument' and the cell fails the post-execution gate. "
                "Do NOT pass cell_type_prop_csv (this is the clustering-only entry point). "
                "Do NOT manually preprocess. The wrapper handles SpatialCluster end-to-end. "
                "Just call the MCP tool ONCE with these args."
            ),
            "graphst_spatial_clustering": (
                "ACTION FIRST: Call the graphst_spatial_clustering MCP tool DIRECTLY before any analysis. "
                "DO NOT produce a planning <solution> checklist before invoking the tool — the tool call MUST be the first execute action in your response. "
                "DO NOT call repair_spatial_h5ad, validate_spatial_h5ad, or any pre-processing helper first — GraphST handles input validation internally. "
                "DO NOT write `from mcp_servers...import graphst_spatial_clustering` (that import path does not exist); invoke the tool via the standard MCP tool-call mechanism that the agent runtime exposes. "
                "Pass: st_h5ad=<input_path>, output_dir=<output_dir>, "
                "n_clusters=<EXACT integer from the 'expected number of clusters/domains' "
                "line above — do NOT round, omit, or substitute>, cluster_tool='mclust'. "
                "The tool uses mclust (R) for clustering by default. "
                "Do NOT manually preprocess the data or write code to cluster. "
                "Just call the MCP tool ONCE with these args and verify the output dir."
            ),
            "stlearn_spatial_clustering": (
                "Call the stlearn_spatial_clustering MCP tool. "
                "If the data has no histology images (e.g., MERFISH, Slide-seqV2), "
                "the tool may fail on tiling. In that case, fall back to standard "
                "scanpy-based PCA + Louvain clustering and save the result."
            ),
            "graphst_deconvolution": (
                "Call the graphst_deconvolution MCP tool DIRECTLY. "
                "Pass: st_h5ad=<spatial_input>, scrna_h5ad=<sc_reference>, "
                "celltype_key=<the reference's cell-type obs column>, output_dir=<output_dir>. "
                "Do NOT manually preprocess. Let the MCP tool handle everything."
            ),
            "spacexr_rctd_deconvolution": (
                "ACTION FIRST: Three MCP calls in order. PRE-AUTHORIZED. "
                "1) convert_h5ad_to_csv(h5ad_path=<input_path>, output_dir=<output_dir>/csv_inputs/spatial). "
                "2) convert_h5ad_to_csv(h5ad_path=<sc_reference_path>, output_dir=<output_dir>/csv_inputs/sc_ref, "
                "cell_type_key=<sc_reference_celltype_key>). "
                "3) spacexr_rctd_deconvolution(spatial_counts_csv=<output_dir>/csv_inputs/spatial/counts.csv, "
                "spatial_coords_csv=<output_dir>/csv_inputs/spatial/coordinates.csv, "
                "ref_counts_csv=<output_dir>/csv_inputs/sc_ref/counts.csv, "
                "ref_celltypes_csv=<output_dir>/csv_inputs/sc_ref/celltypes.csv, "
                "output_dir=<output_dir>, mode='full'). "
                "Just call each ONCE."
            ),
            "run_cell2location": (
                "Call run_cell2location MCP tool. "
                "Use batch_key='Seq_batch' for MERFISH data, batch_key='Sample' for Visium data, "
                "batch_key='library_uuid' for Slide-seqV2 data (the reference's sequencing libraries; never the "
                "cell-type column, which is the label being learned). "
                "X of both files must be raw integer counts: a non-integer matrix is refused unless round_counts=True "
                "(np.rint, reported in params.rounded_inputs). The MERFISH deconvolution spatial input is normalized "
                "(not integer): pass round_counts=True for it. "
                "IMPORTANT: Use max_epochs_ref=250 and max_epochs_map=30000 (full training; "
                "previous truncated values caused mean_pearson_r=null). Also set "
                "cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12 "
                "for the standard gene filtering used by manual benchmarks."
            ),
            "run_celldart": (
                "ACTION FIRST: Call run_celldart MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Pass: sc_h5ad_path=<sc_reference_path>, spatial_h5ad_path=<input_path>, "
                "output_dir=<output_dir>, cell_type_key=<sc_reference_celltype_key>, n_iterations=3000. "
                "DO NOT write `from mcp_servers... import` — call the MCP tool directly. Just call ONCE."
            ),
            "run_destvi": (
                "Call run_destvi MCP tool. "
                "IMPORTANT: pass max_epochs_st=2000, max_epochs_sc=100, n_top_genes=2000 explicitly "
                "(matches manual benchmark; smaller values like 400 epochs produce near-uniform "
                "proportions and mean_pearson_r ≈ 0.1). DO NOT truncate these values to save time. "
                "Leave batch_key unset: it is accepted for compatibility only and never reaches the model "
                "(CondSCVI is always trained without a batch covariate)."
            ),
            "run_spatialscope": (
                "Call run_spatialscope MCP tool. Use UMI_min_sigma=100 (manual-benchmark setting). "
                "The MCP default of 300 is too restrictive for Visium / Slide-seqV2 datasets."
            ),
            "run_spotlight": (
                "ACTION FIRST: Three MCP calls in order. PRE-AUTHORIZED. "
                "1) convert_h5ad_to_csv(h5ad_path=<input_path>, output_dir=<output_dir>/csv_inputs/spatial). "
                "2) convert_h5ad_to_csv(h5ad_path=<sc_reference_path>, output_dir=<output_dir>/csv_inputs/sc_ref, "
                "cell_type_key=<sc_reference_celltype_key>). "
                "3) run_spotlight(spatial_counts_csv=<output_dir>/csv_inputs/spatial/counts.csv, spatial_coords_csv=<output_dir>/csv_inputs/spatial/coordinates.csv, ref_counts_csv=<output_dir>/csv_inputs/sc_ref/counts.csv, ref_celltypes_csv=<output_dir>/csv_inputs/sc_ref/celltypes.csv, output_dir=<output_dir>, n_top=100, min_cont=0.09). "
                "DO NOT write `from mcp_servers... import` — call the MCP tool directly. output_dir must be a DIRECTORY, never a filename. Just call each ONCE."
            ),
            "squidpy_spatial_autocorr": (
                "Use squidpy_spatial_autocorr to identify the top 200 spatially variable genes "
                "in this spatial transcriptomics dataset."
            ),
            "somde_run": (
                "Call somde_run MCP tool with: input_mode='h5ad', h5ad_path=<input_path>, output_dir=<output_dir>. "
                "Default input_mode is now 'h5ad' so omitting it works too. "
                "Never pass counts_h5/spatial_dir for our benchmark inputs (those only apply to raw 10x Visium directories)."
            ),
            "svgbit_run": ("Use SVGbit to identify the top 1000 spatially variable genes in this dataset."),
            "spatialde_run_svg": (
                "ACTION FIRST: Call spatialde_run_svg MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Required kwargs: input_mode='h5ad', h5ad_path=<input_path>, output_dir=<output_dir>. "
                "Use hvg_top_n=2000 with hvg_flavor='variance' (seurat_v3 needs scikit-misc, which the SpatialDE "
                "env lacks, and now stops the run; every earlier run of this tool actually used variance ranking). "
                "Do NOT pass top_k_genes=20 (legacy default cosmetic cap) — the standardizer "
                "reads the full spatialde_results.csv and applies qval<0.05 to score F1. "
                "The worker runs the HVG-2000 prefilter + spatialde_test (SpatialDE 1.1.3) on raw counts; "
                "non-integer input stops the run unless round_counts=True. "
                "Just call ONCE. Do NOT ask permission. "
                "HARD GATE: This entire task fails if <output_dir>/spatialde_results.csv "
                "(or standardized/spatialde_run_svg_svg_standardized.csv) does not exist after "
                "your run. You MUST actually invoke spatialde_run_svg via an <execute> block "
                "(e.g. `result = spatialde_run_svg(input_mode='h5ad', h5ad_path=..., "
                "output_dir=...); print(result)`) and wait for the <observe> response BEFORE "
                "writing any <solution>. Do NOT fabricate file paths or claim 'N significant "
                "SVGs were recovered' — those numbers must come from a real <observe> result, "
                "not your memory or guess. If you write <solution> without a successful "
                "<observe>, the run will fail the post-execution gate and the score will be 0."
            ),
            "bsp_identify_svg": (
                "ACTION FIRST: Call bsp_identify_svg MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Required kwargs: st_h5ad=<input_path>, output_dir=<output_dir>, "
                "top_k_genes=10000, pvalue_cutoff=0.05. "
                "These kwargs match the manual benchmark runner — it returns ALL p<0.05 "
                "significant genes (no top-K cap), which typically yields 1000-3000 "
                "predictions on Visium and is essential for downstream HVG-filtered F1. "
                "Do NOT use the legacy top_k_genes=200 cap — that under-predicts. "
                "Just call ONCE. Do NOT ask permission. "
                "HARD GATE: This entire task fails if <output_dir>/predicted_genes.json "
                "does not exist after your run. You MUST actually invoke bsp_identify_svg "
                "via an <execute> block (e.g. `result = bsp_identify_svg(st_h5ad=..., output_dir=..., "
                "top_k_genes=10000, pvalue_cutoff=0.05); print(result)`) and wait for the "
                "<observe> response BEFORE writing any <solution>. Do NOT fabricate file paths "
                "or claim 'approximately N significant SVGs were recovered' — those numbers "
                "must come from a real <observe> result, not your memory or guess. "
                "If you write <solution> without a successful <observe>, the run will fail "
                "the post-execution gate and the score will be 0."
            ),
            "spagft_identify_svg": (
                "ACTION FIRST: Call spagft_identify_svg MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Required kwargs: st_h5ad=<input_path>, output_dir=<output_dir>, "
                "n_top_genes=10000. "
                "The worker uses SpaGFT.detect_svg with significance filter "
                "(cutoff_gft_score / fdr<0.05 / pvalue<0.05) FIRST and then caps at "
                "n_top_genes. Passing n_top_genes=10000 ensures the cap never bites and "
                "the agent returns the same significant-gene set the manual benchmark uses. "
                "Just call ONCE. Do NOT ask permission. "
                "HARD GATE: This entire task fails if <output_dir>/predicted_genes.json "
                "does not exist after your run. You MUST actually invoke spagft_identify_svg "
                "via an <execute> block (e.g. `result = spagft_identify_svg(st_h5ad=..., "
                "output_dir=..., n_top_genes=10000); print(result)`) and wait for the <observe> "
                "response BEFORE writing any <solution>. Do NOT fabricate file paths or "
                "claim 'planned next step' or 'I would run' — actually run it. "
                "If you write <solution> without a successful <observe>, the score is 0."
            ),
            "prost_index_svg": (
                "Use prost_index_svg to identify the top 200 spatially variable genes "
                "in this spatial transcriptomics dataset."
            ),
            "svca_variance_decomposition": (
                "Use SVCA to identify spatially variable genes across all genes in the "
                "dataset (do not pre-restrict to a small subset)."
            ),
            "spvc_svg_detection": (
                "ACTION FIRST: Call spvc_svg_detection MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Step 1: convert h5ad to CSV in pure Python (DO NOT rely on convert_h5ad_to_csv "
                "being in scope — write the CSVs yourself). Example: \n"
                "```\n"
                "import anndata as ad, pandas as pd, os, numpy as np\n"
                "csv_dir = '<output_dir>/csv_inputs'; os.makedirs(csv_dir, exist_ok=True)\n"
                "a = ad.read_h5ad('<input_path>')\n"
                "X = a.X.toarray() if hasattr(a.X, 'toarray') else np.asarray(a.X)  # spots x genes\n"
                "counts = pd.DataFrame(X.T, index=a.var_names, columns=a.obs_names)  # genes x spots\n"
                "counts.to_csv(f'{csv_dir}/counts.csv')\n"
                "coords = pd.DataFrame(a.obsm['spatial'][:, :2], columns=['x','y'], index=a.obs_names)\n"
                "coords.to_csv(f'{csv_dir}/coords.csv')\n"
                "```\n"
                "Step 2: invoke `result = spvc_svg_detection(counts_csv=f'{csv_dir}/counts.csv', "
                "coords_csv=f'{csv_dir}/coords.csv', output_dir='<output_dir>', n_top=100, max_genes=1000); print(result)` "
                "in an <execute> block and wait for <observe>. "
                "These kwargs match the manual benchmark runner (top 10% of HVG-1000 by spatial p-value). "
                "Do NOT reduce max_genes below 1000 — that under-selects the candidate pool. "
                "Just call ONCE. Do NOT ask permission. "
                "HARD GATE: This entire task fails if <output_dir>/predicted_genes.json does not "
                "exist after your run. Do NOT write <solution> before spvc_svg_detection has "
                "returned a successful <observe>. Do NOT fabricate file paths or claim 'I plan "
                "to run...' — actually run it. If the run is incomplete, the score is 0."
            ),
            "spark_svg_detection": (
                "ACTION FIRST: Call spark_svg_detection MCP tool DIRECTLY. PRE-AUTHORIZED. "
                "Step 1: convert_h5ad_to_csv on <input_path> → counts CSV (genes x spots) + coords CSV (spots x 2). "
                "Step 2: spark_svg_detection(counts_csv=<counts>, coords_csv=<coords>, "
                "output_dir=<output_dir>, n_top=200). "
                "The kwarg name is n_top (NOT n_top_genes, NOT top_n). "
                "Just call ONCE. Do NOT ask permission."
            ),
            "prost_pnn_domains": (
                "CALL prost_pnn_domains MCP tool DIRECTLY on the original h5ad. "
                "Do NOT manually preprocess, repair, or run diagnose_spatial_data first — the tool handles its own validation. "
                "Required kwargs: st_h5ad=<input_path>, output_dir=<output_dir>, n_domains=<n_clusters integer above>. "
                "The kwarg name is n_domains (NOT n_clusters). "
                "ACTION FIRST: invoke the tool in your very first execute step. Verification comes AFTER, not before."
            ),
            "spaceflow_spatial_domains": ("Use SpaceFlow to identify the expected number of spatial domains."),
            "seurat_qc_cluster": (
                "ACTION FIRST. You are PRE-AUTHORIZED to invoke seurat_qc_cluster — DO NOT ask for confirmation. "
                "Invoke it in your VERY FIRST execute step before any diagnosis, dataset inspection, or planning. "
                "ONLY accepted kwargs: data_path=<input_path>, output_dir=<output_dir>, resolution=0.5. "
                "DO NOT pass n_clusters (the wrapper does NOT accept it — Seurat targets cluster count via resolution). "
                "DO NOT pass counts_dir (the wrapper does NOT accept it — pass data_path which auto-detects format). "
                "Wrapper signature is exactly: data_path, output_dir, project='SeuratQC', min_cells=3, "
                "min_features=200, n_hvgs=2000, n_pcs=30, resolution=0.8, umap=True, seed=0. "
                "Any unlisted kwarg (n_clusters, counts_dir, etc.) raises ValidationError 'Unexpected keyword argument' and the cell fails. "
                "The wrapper AUTOMATICALLY converts h5ad → 10x MTX internally; do NOT create tenx_counts/ or run conversion yourself. "
                "DO NOT write 'I'm ready to do X next' or 'I need your confirmation' — just invoke the tool. "
                "DO NOT produce a final answer until the seurat_qc_cluster tool result block appears in your transcript. "
                "An empty cell with no tool invocation is a FAILURE; the cell only succeeds when seurat writes its h5ad/CSV output."
            ),
            "run_bass": ("Use BASS to identify the expected number of spatial domains."),
            "precast_spatial_clustering": (
                "CALL precast_spatial_clustering MCP tool. R-based — first call convert_h5ad_to_csv to create "
                "counts CSV (genes x spots) and coordinates CSV (spots x 2). "
                "Then invoke precast with --counts-csvs and --coords-csvs (plural — accepts a list, even for one sample), "
                "plus K=<n_clusters integer above>. "
                "ACTION-COMPLETE: do not stop after CSV conversion — continue and call precast in the same loop."
            ),
            "stlearn_spatial_clustering_force_action": (
                "CALL stlearn_spatial_clustering MCP tool DIRECTLY in your first execute step. "
                "Do NOT spend loops on planning, diagnosis, or repair before calling the tool. "
                "If it fails on tiling (no histology), THEN fall back to scanpy PCA + Louvain "
                "and save the result as predicted_labels.csv with columns [spot_id, cluster]."
            ),
            "run_iris": (
                "ACTION FIRST. R-based tool. You MUST invoke two MCP tools, in order, BEFORE producing any final answer. "
                "Step 1: invoke convert_h5ad_to_csv (h5ad_path=<input_path>, output_dir=<output_dir>/csv_inputs) — wait for the tool result. "
                "Step 2: invoke run_iris (spatial_counts_csv=<output_dir>/csv_inputs/counts.csv, "
                "spatial_coords_csv=<output_dir>/csv_inputs/coordinates.csv, output_dir=<output_dir>, "
                "n_clusters=<EXACT integer from the 'expected number of clusters/domains' line above>, seed=42). "
                "Do NOT pass ref_counts_csv or ref_celltypes_csv — running in IRISfree mode (reference-free), "
                "which is the only mode compatible with raw-clustering benchmarks like visium_dlpfc_domain. "
                "An empty cell with only a CSV conversion is a FAILURE; the cell only succeeds when run_iris writes iris_domains.csv."
            ),
            "run_spicemix": (
                "ACTION FIRST. Call the run_spicemix MCP tool DIRECTLY in your first execute step. "
                "Required kwargs: spatial_h5ad_path=<input_path>, output_dir=<output_dir>, "
                "K=<EXACT integer from the 'expected number of clusters/domains' line above>, n_epochs=100, device='cpu'. "
                "Do NOT manually preprocess; the wrapper handles input validation. "
                "The worker writes obs['spicemix_factor'] (categorical, K labels via argmax of factor loadings) — "
                "this is the cluster prediction column the evaluator reads. "
                "An empty cell with only a prepared h5ad is a FAILURE; the cell only succeeds when run_spicemix writes its h5ad output with spicemix_factor."
            ),
            "tacco_annotate": (
                "ACTION FIRST. Call the tacco_annotate MCP tool DIRECTLY in your first execute step. "
                "Required kwargs: sc_h5ad=<Single-cell reference path from the line above>, "
                "spatial_h5ad=<input_path>, output_dir=<output_dir>, "
                "annotation_key='<sc_reference_celltype_key from registry — e.g. \"CellType\" for visium_deconvolution>'. "
                "Optional: method='OT', multi_center=3, lamb=0.001 (defaults are correct). "
                "The worker writes tacco_composition.csv (spots × cell_types proportions) and tacco_dominant_label_per_spot.csv. "
                "Do NOT manually preprocess. Do NOT call repair_spatial_h5ad first. "
                "An empty cell with only a data inspection is a FAILURE; the cell only succeeds when tacco writes tacco_composition.csv."
            ),
            "tangram_map_sc_to_spatial": (
                "ACTION FIRST. Call the tangram_map_sc_to_spatial MCP tool DIRECTLY in your first execute step. "
                "Required kwargs: sc_h5ad=<Single-cell reference path from the line above>, "
                "spatial_h5ad=<input_path>, output_dir=<output_dir>, "
                "annotation_key='<sc_reference_celltype_key — e.g. \"CellType\" for visium_deconvolution>', "
                "mode='clusters' (paper-grade default for proportion estimation; 'cells' is per-cell mapping and unnecessary here), "
                "n_markers_per_class=100, num_epochs=1000, density_prior='rna_count_based', device='auto'. "
                "Do NOT manually preprocess; Tangram handles training-gene selection and mapping internally. "
                "An empty cell with only a prepared h5ad is a FAILURE; the cell only succeeds when tangram writes its proportion CSV."
            ),
            "run_card": (
                "ACTION FIRST: Three MCP calls in order. PRE-AUTHORIZED. "
                "1) convert_h5ad_to_csv(h5ad_path=<input_path>, output_dir=<output_dir>/csv_inputs/spatial). "
                "2) convert_h5ad_to_csv(h5ad_path=<sc_reference_path>, output_dir=<output_dir>/csv_inputs/sc_ref, "
                "cell_type_key=<sc_reference_celltype_key>). "
                "3) run_card(spatial_counts_csv=<output_dir>/csv_inputs/spatial/counts.csv, spatial_coords_csv=<output_dir>/csv_inputs/spatial/coordinates.csv, ref_counts_csv=<output_dir>/csv_inputs/sc_ref/counts.csv, ref_celltypes_csv=<output_dir>/csv_inputs/sc_ref/celltypes.csv, output_dir=<output_dir>, min_count_gene=100, min_count_spot=5). "
                "DO NOT write `from mcp_servers... import` or any Python import of the tool — call the MCP tool directly. Just call each ONCE."
            ),
            "run_scanpy_spatial_domain": ("Use Scanpy to identify the expected number of spatial domains."),
        }
        # Override the existing stlearn entry with a stronger ACTION-FIRST version
        # for multi-LLM benchmarking (5.4 family wastes loops planning otherwise).
        TOOL_PROMPT_GUIDANCE["stlearn_spatial_clustering"] = (
            "CALL stlearn_spatial_clustering MCP tool DIRECTLY in your first execute step. "
            "Do NOT spend loops on planning, diagnosis, or repair before calling the tool. "
            "Required kwargs: st_h5ad=<input_path>, output_dir=<output_dir>. "
            "stLearn takes no n_clusters; tune resolution to target the expected count above. "
            "If it stops because the data has no histology image (e.g. MERFISH/Slide-seqV2), "
            "call it again with allow_pca_fallback=True, which clusters expression PCA alone "
            "and says so in params.method."
        )
        # Add n_clusters hint from registry metadata FIRST so tool guidance
        # below can refer to it as "the line above". STCoscientist was previously asking
        # the user for this value when the registry entry omitted n_clusters
        # (e.g., visium_dlpfc_domain pre-2026-05-01) because the prompt
        # silently dropped the line. Now we always emit either the integer
        # or an explicit "must derive from GT" instruction.
        n_clusters = (dataset_entry.get("metadata") or {}).get("n_clusters")
        if task_type == "spatial_clustering":
            if n_clusters is not None:
                prompt += f"\nThe expected number of clusters/domains is {n_clusters} (use this EXACT integer).\n"
            else:
                gt_key_hint = (dataset_entry.get("metadata") or {}).get("ground_truth_key") or "<gt key>"
                prompt += (
                    f"\nThe expected number of clusters/domains is NOT pre-specified in the registry. "
                    f"Derive it as: len(unique values in obs['{gt_key_hint}']) "
                    f"after dropping NaN/None/empty entries, then use that integer EXACTLY.\n"
                )

        guidance = TOOL_PROMPT_GUIDANCE.get(tool_name)
        if guidance:
            prompt += f"\n\n{guidance}\n"

        # ── SVG inputs: HVG-pre-filtered EXCEPT for full-transcriptome tools ──
        # The SVG pre-filter is applied during data prep for most tools, but
        # significance-calibrated tools (BSP, SpaGFT, SpVC, SpatialDE) need the
        # full 32k-gene catalog to match the manual baseline (see
        # _SVG_FULL_TRANSCRIPTOME_TOOLS above).
        #
        # Scored prompt text. The full-transcriptome note said "the benchmark evaluator applies a
        # post-hoc HVG filter"; it never did -- f1 is scored on the list as given, and f1_hvg is
        # recorded beside it (hunt 2026-09-30, u33b-bench-scoring-11).
        if task_type == "svg_detection":
            if tool_name in _SVG_FULL_TRANSCRIPTOME_TOOLS:
                prompt += (
                    "\n\nNote: the input AnnData contains the full ~32k-gene catalog (no HVG pre-filter "
                    "for this tool — its significance test is calibrated against the full transcriptome). "
                    "Call the tool directly on `Input data` — do NOT pre-filter or re-normalize. "
                    "The benchmark evaluator records F1 on your gene list as given and, beside it, F1 restricted "
                    "to the dataset's top 1500 Seurat HVGs (f1_hvg).\n"
                )
            else:
                prompt += (
                    "\n\nNote: the input AnnData has already been pre-filtered to top ~1500 Seurat HVGs "
                    "during data prep. Call the tool directly on `Input data` — do NOT re-filter or re-normalize.\n"
                )

        # ── Pre-execution gate (mandatory input checking) ──
        try:
            from spatialomicsgym.benchmarking.workflow_gates import pre_execution_gate

            pre_gate = pre_execution_gate(
                tool_name=tool_name,
                task_type=task_type,
                input_path=str(input_path),
                output_dir=str(output_dir),
                sc_ref_path=str(sc_ref) if sc_ref is not None else None,
                platform=dataset_entry.get("platform", ""),
            )
            result["pre_execution_gate"] = pre_gate.to_dict()

            if not pre_gate.passed:
                result["status"] = "input_invalid"
                result["error"] = "Pre-execution gate blocked: " + "; ".join(pre_gate.reasons)
                result["elapsed_seconds"] = time.time() - start
                return result

            # Include gate warnings in prompt so agent is aware
            if pre_gate.warnings:
                prompt += "\n\nWARNINGS from input validation:\n"
                for w in pre_gate.warnings:
                    prompt += f"- {w}\n"
        except Exception as e:
            # Non-blocking: if gate fails to run, proceed with tool execution
            result.setdefault("warnings", []).append(f"Pre-execution gate error: {e}")

        # ── Tuned hyperparameters, last, so the filter sees the whole assembled prompt ──
        tuning_block = _tuning_prompt_block(tool_name, tuned_params, prompt)
        if tuning_block:
            prompt += tuning_block
            result["tuned_params_applied"] = {
                name: value for name, value in (tuned_params or {}).items() if f"- {name} = " in tuning_block
            }

        # Retry logic for rate-limit (429) errors.
        #
        # The `except` alone never fired. `_iter_react_stream` catches every provider exception,
        # records it as a degrade note and returns the partial transcript
        # (`agent/stcoscientist.py:441-447`), so a 429 arrives here as a NORMAL return with an
        # empty answer -- `break  # Success`, no retry, and a rate-limited run scored as a run
        # that produced nothing. Both shapes are handled now: the note is read on the success
        # path and drives the same backoff.
        max_retries = 2
        degraded = ""
        for attempt in range(max_retries + 1):
            try:
                log, response = agent.go(prompt)
                degraded = degrade_note(agent)
                if _is_rate_limited(degraded) and attempt < max_retries:
                    _sleep_for_rate_limit(attempt, max_retries)
                    continue
                break  # a real answer, or a degrade this loop cannot fix
            except Exception as e:
                if _is_rate_limited(f"{type(e).__name__}: {e}") and attempt < max_retries:
                    _sleep_for_rate_limit(attempt, max_retries)
                    continue
                raise
        result["elapsed_seconds"] = time.time() - start
        if degraded:
            result["degraded"] = degraded

        # Whether the tool was actually CALLED.
        #
        # This used to be `tool_name in " ".join(log)`, and the log's first entry is the
        # pretty-printed prompt -- into which the retriever writes the recommended tool's name up
        # to six times. It was true before the model emitted a token. The widening fallback below
        # was worse still: `"MCP tool"` and `"Running worker"` match the prompt's own boilerplate,
        # so it could not return False. `turn_outcome.tool_was_invoked` reads the cells that ran
        # and the provider tool-call blocks; the markers are kept, but searched over the same
        # agent-side text rather than the whole transcript.
        #
        # The second marker used to be f"call_tool.*{tool_name}": a regex, handed to a substring
        # test, so it could never match anything (hunt 2026-09-30, u33a-bench-runner-30).
        tool_invoked = tool_was_invoked(agent, log, tool_name) or tool_was_invoked(agent, log, f"mcp_{tool_name}")

        if not tool_invoked:
            # A turn that never finished did not decline to call the tool -- it never got there.
            # Its status says so too: "fail_prompt" is tabulated as agent_skip, which counted 429
            # give-ups and spent step budgets as prompting failures when models were compared
            # (hunt 2026-09-30, u33a-bench-runner-17).
            result["status"] = "degraded" if degraded else "fail_prompt"
            result["error"] = f"The turn did not finish: {degraded}" if degraded else "Tool was not invoked by agent"
            return result

        tool_elapsed = time.time() - start

        # ── Post-execution gate (mandatory output checking) ──
        try:
            from benchmarks.evaluation.evaluator import classify_evaluation, evaluate_tool_output
            from spatialomicsgym.benchmarking.workflow_gates import post_execution_gate, pre_evaluation_gate

            gt_key = get_ground_truth_key(dataset_entry, task_type)
            result["ground_truth_key"] = gt_key
            print(f"  gt_key={gt_key} (from registry metadata)", flush=True)

            # STEP 1: Post-execution gate
            post_gate = post_execution_gate(
                tool_name=tool_name,
                task_type=task_type,
                output_dir=str(output_dir),
                ground_truth_path=str(gt_path),
                expected_n_clusters=(dataset_entry.get("metadata") or {}).get("n_clusters"),
            )
            result["post_execution_gate"] = post_gate.to_dict()

            if not post_gate.passed:
                result["status"] = "output_invalid"
                result["error"] = "Post-execution gate blocked: " + "; ".join(post_gate.reasons)
                # Include misplaced file hints if available
                if post_gate.details.get("misplaced_candidates"):
                    result["misplaced_output_candidates"] = post_gate.details["misplaced_candidates"]
                return result

            # Extract prediction info from the gate's inspection results
            inspection_details = post_gate.details.get("inspection", {})
            eval_file = (
                Path(post_gate.details.get("prediction_file", "")) if post_gate.details.get("prediction_file") else None
            )
            predicted_key_from_gate = post_gate.details.get("prediction_key", "")

            # The gate reads the whole output tree, and output_dir is reused run to run: the sweep
            # above only removes top-level files over an hour old. So a re-run whose tool errored
            # and wrote nothing passed on the previous run's prediction and was recorded as this
            # run's "evaluated" metrics (hunt 2026-09-30, u33a-bench-runner-13). A prediction that
            # predates this run is not this run's. The gate does not filter its candidates by run
            # start, so it can also pass over a file this run DID write: that is said, and the run
            # is blocked from scoring rather than called invalid (u33a-bench-runner-13, review).
            run_began = start - _MTIME_SLACK_SECONDS
            if eval_file is not None and eval_file.exists() and eval_file.stat().st_mtime < run_began:
                picked = (
                    f"Post-execution gate picked {eval_file}, written "
                    f"{datetime.fromtimestamp(eval_file.stat().st_mtime).isoformat(timespec='seconds')}, "
                    "before this run started: it is a previous run's output"
                )
                fresh = _files_written_since(output_dir, run_began)
                if fresh:
                    shown = ", ".join(str(p.relative_to(output_dir)) for p in fresh[:5])
                    result["status"] = "eval_blocked"
                    result["error"] = (
                        f"{picked}. This run did write {len(fresh)} file(s) there ({shown}"
                        f"{', ...' if len(fresh) > 5 else ''}), but the gate does not filter by run start "
                        "and chose the older file, so this run's own output was not inspected; move the "
                        "previous run's files out of the output directory and score again"
                    )
                else:
                    result["status"] = "output_invalid"
                    result["error"] = f"{picked}, and this run wrote no output file there (staged inputs aside)"
                return result

            # STEP 2: Build evaluation kwargs
            eval_kwargs: dict[str, Any] = {}

            if task_type == "svg_detection":
                svg_genes = inspection_details.get("svg_genes") or []
                if not svg_genes:
                    svg_genes = _extract_svg_genes(output_dir, eval_file, not_before=run_began)
                eval_kwargs["predicted_genes"] = svg_genes
                eval_kwargs["spatial_h5ad"] = str(input_path)

            # Same path the guarded read above resolved -- the gate publishes it under both keys
            # from one inspection. Guarded too: prediction_file is None when unset, not "".
            if (
                task_type == "deconvolution"
                and inspection_details.get("prediction_format") == "csv"
                and inspection_details.get("prediction_file")
            ):
                eval_file = Path(inspection_details["prediction_file"])

            if task_type == "spatial_clustering" and predicted_key_from_gate:
                eval_kwargs["predicted_key"] = predicted_key_from_gate

            # STEP 3: Pre-evaluation gate
            if eval_file and eval_file.exists():
                pre_eval_gate = pre_evaluation_gate(
                    task_type=task_type,
                    prediction_file=str(eval_file),
                    ground_truth_path=str(gt_path),
                    ground_truth_key=gt_key,
                    predicted_key=predicted_key_from_gate,
                    predicted_genes=eval_kwargs.get("predicted_genes"),
                )
                result["pre_evaluation_gate"] = pre_eval_gate.to_dict()

                if not pre_eval_gate.passed:
                    result["status"] = "eval_blocked"
                    result["error"] = "Pre-evaluation gate blocked: " + "; ".join(pre_eval_gate.reasons)
                    return result

            # STEP 3.5: Standardize output
            try:
                from spatialomicsgym.benchmarking.output_standardizer import standardize_output

                std_result = standardize_output(
                    task_type=task_type,
                    prediction_file=str(eval_file) if eval_file else "",
                    prediction_key=predicted_key_from_gate,
                    output_dir=str(output_dir),
                    tool_name=tool_name,
                    ground_truth_path=str(gt_path),
                    svg_genes=eval_kwargs.get("predicted_genes"),
                    spatial_h5ad=str(input_path),
                )
                result["standardization"] = std_result

                if std_result["success"]:
                    result["standardized_file"] = std_result["standardized_file"]
            except Exception as e:
                result.setdefault("warnings", []).append(f"Standardization skipped: {e}")

            # STEP 4: Evaluate (only if all gates passed)
            if (
                eval_file
                and eval_file.exists()
                and task_type in ("spatial_clustering", "svg_detection", "deconvolution")
            ):
                # Pass standardized file if available
                std_file = result.get("standardized_file")
                metrics = evaluate_tool_output(
                    task_type,
                    str(eval_file),
                    str(gt_path),
                    ground_truth_key=gt_key,
                    standardized_file=std_file,
                    dataset_name=dataset_entry["name"],
                    **eval_kwargs,
                )
                result["metrics"] = metrics
                # The evaluator's shared rule, not an inline copy of it: the copy counted a bare
                # diagnostic count (n_predicted_genes, with nothing measured) as a metric, so such a
                # run was recorded "evaluated" here and in metrics.json (hunt 2026-09-30,
                # u33b-bench-scoring-10).
                result["status"], eval_error = classify_evaluation(metrics)
                if eval_error is not None:
                    result["error"] = eval_error

                metrics_to_persist = dict(metrics)
                metrics_to_persist["status"] = result["status"]
                if result.get("error"):
                    metrics_to_persist["error"] = result["error"]
                if std_file:
                    metrics_to_persist["_standardized_file"] = std_file
                metrics_path = Path(output_dir) / "metrics.json"
                # ``json.dumps`` writes a non-finite float as the bare token ``NaN``/``Infinity``,
                # which is not JSON (RFC 8259) -- and ``default=`` never fires for it, because a
                # NaN *is* serialisable, just not to anything a strict parser accepts. Nine files
                # under the recorded results tree are unreadable to ``jq``/``JSON.parse``/Arrow for
                # exactly this reason. Guarding here rather than in each metric producer is
                # deliberate: the JSD producer was fixed at source and the class came straight back
                # through ``mean_entropy``.
                from spatialomicsgym.utils.file_io import json_safe

                metrics_path.write_text(json.dumps(json_safe(metrics_to_persist), indent=2, default=str))
            else:
                result["status"] = "tool_ran"
        except Exception as e:
            result["status"] = "tool_ran"
            result["error"] = f"Evaluation error: {e}"

        result["timing"] = {
            "tool_execution": round(tool_elapsed, 1),
            "gates_and_eval": round(time.time() - start - tool_elapsed, 1),
            "total": round(time.time() - start, 1),
        }

    except Exception as e:
        result["elapsed_seconds"] = time.time() - start
        result["status"] = "fail"
        result["error"] = str(e)[:500]
    finally:
        # The mode is this run's, not the process's: a caller that imports this function (a tuning
        # sweep, a test) gets its own setting back, as run_formal_stcoscientist_benchmark does.
        if saved_modes is not None and default_config is not None:
            default_config.benchmarking_enabled, default_config.evaluation_enabled = saved_modes

    return result


#: Every status run_benchmark_tool can return, for the summary's breakdown.
_FAILURE_STATUSES = (
    "input_invalid",
    "fail_prompt",
    "degraded",
    "output_invalid",
    "eval_blocked",
    "eval_failed",
    "tool_ran",
    "fail",
)


def _write_text_atomically(path: Path, text: str) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    os.replace(partial, path)


def _write_summary(output_dir: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    """Write the sweep summary as JSON and markdown, under this run's id and as the latest copy.

    Called after every pair, so a sweep that dies part-way keeps what it finished (hunt 2026-09-30,
    u33a-bench-runner-12). Written as ``benchmark_results_<run_id>.*`` as well as the fixed names:
    the fixed names were all there was, so a retry or gap-fill that re-invoked the runner for one
    tool erased the full sweep's summary -- and the statuses of runs that never reached
    metrics.json lived nowhere else (u33a-bench-runner-28).
    """
    from spatialomicsgym.utils.file_io import json_safe

    total = len(results)
    passed = sum(1 for r in results if r["status"] in ("pass", "evaluated"))
    summary["total_runs"] = total
    summary["passed"] = passed
    # Every run that did not pass. Counting only statuses spelled "fail*" left output_invalid,
    # eval_failed, eval_blocked, input_invalid and tool_ran in neither column, so a sweep of ten
    # gate-blocked runs read "Passed: 0 | Failed: 0" (hunt 2026-09-30, u33a-bench-runner-16,
    # uL6-parity-17).
    summary["failed"] = total - passed
    summary["evaluated"] = sum(1 for r in results if r["status"] == "evaluated")
    summary["failure_breakdown"] = {
        status: sum(1 for r in results if r["status"] == status) for status in _FAILURE_STATUSES
    }
    summary["results"] = results

    # summary["results"] embeds the same per-run metric dicts the guarded write above produces, so
    # this needs the same guard: without it a run whose metric could not be computed writes a bare
    # ``NaN`` here and the whole summary stops being JSON to every reader outside Python.
    payload = json.dumps(json_safe(summary), indent=2, default=str)

    md_lines = [
        f"# Benchmark Results: {summary['run_id']}",
        f"**Date:** {summary['timestamp']}",
        f"**Total:** {summary['total_runs']} | **Passed:** {summary['passed']}"
        f" | **Failed:** {summary['failed']} | **Evaluated:** {summary['evaluated']}",
        f"**Benchmarking mode:** {summary.get('benchmarking_enabled')} | "
        f"**Evaluation mode:** {summary.get('evaluation_enabled')}",
        "",
        "**Not passed, by status:** "
        + (", ".join(f"{k}={v}" for k, v in summary["failure_breakdown"].items() if v) or "none"),
        "",
        "| Tool | Dataset | Task | Status | Metrics | Time |",
        "|------|---------|------|--------|---------|------|",
    ]
    for r in results:
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in r.get("metrics", {}).items() if isinstance(v, (int, float)))
        md_lines.append(
            f"| {r['tool_name']} | {r['dataset_name']} | {r['task_type']}"
            f" | {r['status']} | {metrics_str or '-'} | {r['elapsed_seconds']:.0f}s |"
        )
    markdown = "\n".join(md_lines)

    for stem in (f"benchmark_results_{summary['run_id']}", "benchmark_results"):
        _write_text_atomically(output_dir / f"{stem}.json", payload)
        _write_text_atomically(output_dir / f"{stem}.md", markdown)


def run_benchmark(
    task_types: list[str] | None = None,
    tools: list[str] | None = None,
    datasets: list[str] | None = None,
    max_workers: int = 1,
    config_path: str | None = None,
    tuning_mode: str | None = None,
    tuning_dir: str | None = None,
    include_user_tools: bool = False,
) -> dict[str, Any]:
    """Run benchmark for specified task types and tools.

    Args:
        task_types: Filter by task type (spatial_clustering, svg_detection, deconvolution).
        tools: Filter by specific tool names.
        datasets: Filter by exact dataset name (e.g. ["visium_svg","merfish_svg"]).
        max_workers: Must be 1. Pairs run one after another in this process; see the check below.
        config_path: Benchmark config YAML path.
        tuning_mode: If set, apply hyperparameter tuning before each tool run.
            One of: 'benchmark_tuning.light', 'benchmark_tuning.full',
            'adaptive_tuning', 'default_fallback', or None (use tool defaults).
        tuning_dir: Directory for tuning results. Defaults to benchmarks/results/tuning.

    Raises :class:`ProviderKeyMissing` before any pair runs when the model's provider settings are
    not in this process's environment -- every pair would fail the same way.
    """
    _keep_dotenv_out()
    # Accepted, recorded and ignored, the flag let a sweep sized for --workers 4 take four times
    # the wall-clock it was planned for. It is refused instead: pairs share one process, and the
    # agent's REPL namespace and stdout are process-global, so parallel pairs would read each
    # other's variables (hunt 2026-09-30, u33a-bench-runner-29, uL6-parity-17).
    if max_workers != 1:
        raise ValueError(
            f"max_workers={max_workers}: benchmark_runner runs tool-dataset pairs one at a time in one "
            "process (the agent's REPL namespace and stdout are process-global), so only 1 is supported. "
            "Run several invocations with disjoint --tools/--datasets to parallelise."
        )
    config = load_benchmark_config(config_path)
    config["max_workers"] = max_workers
    config["include_user_tools"] = include_user_tools
    missing = [
        k for k in ("project_root", "benchmark_data_dir", "output_dir", "mcp_config_path", "llm") if not config.get(k)
    ]
    if missing:
        raise ValueError(f"benchmark config {config_path or '(defaults)'} has no value for {missing}")
    # Once, before the sweep, rather than as the same failure on every pair (hunt 2026-09-30,
    # uL6-parity-5, review).
    key_problem = provider_key_problem(config["llm"])
    if key_problem:
        raise ProviderKeyMissing(key_problem)

    # Load dataset registry
    from benchmarks.data.data_registry import DataRegistry

    registry_path = Path(config["project_root"]) / config["benchmark_data_dir"] / "registry.yaml"
    registry = DataRegistry.from_yaml(registry_path)

    # Inject user tools into compatible_tools when enabled
    if include_user_tools:
        from spatialomicsgym.mcp_user_config import install_log_path

        # Where tool creation keeps it, not a repo-anchored literal (hunt 2026-09-30, u33a-bench-runner-37).
        install_log = Path(install_log_path())
        injected = registry.inject_user_tools(install_log)
        if injected:
            n = sum(len(v) for v in injected.values())
            print(f"  User tools: {n} tool(s) injected into {len(injected)} dataset(s)")

    # Filter datasets and tools
    all_datasets = registry.list_all()
    if task_types:
        all_datasets = [d for d in all_datasets if d.task_type in task_types]
    if datasets:
        all_datasets = [d for d in all_datasets if d.name in set(datasets)]

    run_id = f"bench-{uuid.uuid4().hex[:8]}"
    results = []

    print(f"=== Benchmark Run {run_id} ===")
    print(f"Datasets: {len(all_datasets)}")

    # Set up tuning runner if tuning_mode is specified
    tuning_runner = None
    if tuning_mode:
        try:
            from spatialomicsgym.tuning.core import TuningMode
            from spatialomicsgym.tuning.integration import TunedBenchmarkRunner

            mode = TuningMode(tuning_mode)
            tuning_runner = TunedBenchmarkRunner(mode=mode, tuning_dir=tuning_dir, use_cached=True)
            print(f"  Tuning enabled: mode={tuning_mode}")
        except Exception as e:
            print(f"  Warning: Could not initialize tuning ({e}), using defaults")

    output_dir = Path(config["project_root"]) / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        # Recorded so a tuned run and an untuned one are distinguishable on disk. None on the
        # default path, which is every benchmark number in benchmarks/results/.
        "tuning_mode": tuning_mode,
        "tuning_dir": tuning_dir,
        # The agent mode run_benchmark_tool sets for every pair; each result carries it as well.
        "benchmarking_enabled": bool(config.get("benchmarking_enabled", _BENCHMARKING_MODE_DEFAULT)),
        "evaluation_enabled": bool(config.get("evaluation_enabled", _EVALUATION_MODE_DEFAULT)),
        "llm": config["llm"],
    }

    data_root = Path(config["project_root"]) / config["benchmark_data_dir"]
    for dataset in all_datasets:
        compatible = dataset.compatible_tools
        if tools:
            compatible = [t for t in compatible if t in tools]

        for tool_name in compatible:
            # Apply tuning if available
            tuning_info = ""
            tuned_params: dict[str, Any] | None = None
            if tuning_runner:
                try:
                    from spatialomicsgym.tuning.integration import registry_fixed_params

                    # What the search needs to score this dataset: the reference (a deconvolution
                    # search had none to score against), the curated SVG list (an svg search "cannot
                    # score any trial" without one), the registry's ground-truth column and the
                    # values the dataset fixes (hunt 2026-09-30, u33a-bench-runner-31,
                    # u32-tuning-4/5/8). An SVG "truth" that is the input h5ad itself is no curated list.
                    metadata = dataset.metadata or {}
                    sc_rel = metadata.get("sc_reference_path")
                    svg_truth = (
                        str(data_root / dataset.ground_truth_path)
                        if dataset.task_type == "svg_detection"
                        and dataset.ground_truth_path
                        and dataset.ground_truth_path != dataset.input_path
                        else None
                    )
                    tuned_params, tuning_info = tuning_runner.get_params_for_tool(
                        tool_name=tool_name,
                        task_type=dataset.task_type,
                        dataset_path=str(data_root / dataset.input_path),
                        sc_reference_path=str(data_root / sc_rel) if sc_rel else None,
                        svg_ground_truth_path=svg_truth,
                        ground_truth_key=get_ground_truth_key({"metadata": metadata}, dataset.task_type),
                        fixed_params=registry_fixed_params(tool_name, dataset.task_type, metadata),
                    )
                    tuning_info = f" [tuning:{tuning_info}]"
                except Exception as e:
                    tuned_params = None
                    tuning_info = f" [tuning:error:{e}]"

            print(f"  Running {tool_name} on {dataset.name}...{tuning_info}")
            pair_start = time.time()
            # One pair's crash is that pair's result, not the sweep's end (u33a-bench-runner-12).
            try:
                result = run_benchmark_tool(
                    tool_name=tool_name,
                    dataset_entry={
                        "name": dataset.name,
                        "input_path": dataset.input_path,
                        "ground_truth_path": dataset.ground_truth_path or dataset.input_path,
                        "metadata": dataset.metadata,
                        "platform": dataset.platform,
                        "ground_truth_key": get_ground_truth_key({"metadata": dataset.metadata}, dataset.task_type),
                    },
                    task_type=dataset.task_type,
                    config=config,
                    tuned_params=tuned_params,
                )
            except Exception as e:
                result = {
                    "tool_name": tool_name,
                    "dataset_name": dataset.name,
                    "task_type": dataset.task_type,
                    "status": "fail",
                    "metrics": {},
                    "elapsed_seconds": time.time() - pair_start,
                    "error": f"{type(e).__name__}: {e}"[:500],
                }
            if tuning_info:
                result["tuning_info"] = tuning_info.strip()
            results.append(result)
            status = result["status"]
            metrics_str = ", ".join(
                f"{k}={v:.4f}" for k, v in result.get("metrics", {}).items() if isinstance(v, (int, float))
            )
            print(
                f"    [{status.upper()}] {metrics_str or result.get('error', '')[:100]}"
                f" ({result['elapsed_seconds']:.1f}s)"
            )
            _write_summary(output_dir, summary, results)

    _write_summary(output_dir, summary, results)
    print(f"\nResults saved to {output_dir}/ (benchmark_results_{run_id}.json)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run SpatialOmicsLab benchmark")
    parser.add_argument("--task-types", nargs="+", default=None, help="Task types to benchmark")
    parser.add_argument("--tools", nargs="+", default=None, help="Specific tools to benchmark")
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific dataset names to benchmark")
    parser.add_argument(
        "--workers", type=int, default=1, help="Must be 1: pairs run one at a time in this process (see run_benchmark)"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument(
        "--tuning-mode",
        type=str,
        default=None,
        choices=[
            "benchmark_tuning.light",
            "benchmark_tuning.full",
            "adaptive_tuning",
            "default_fallback",
        ],
        help="Hyperparameter tuning mode",
    )
    parser.add_argument("--tuning-dir", type=str, default=None, help="Tuning results directory")
    parser.add_argument(
        "--include-user-tools",
        action="store_true",
        default=False,
        help="Include user-created MCP tools (matched by task_type from tools_user/install_log.json)",
    )
    args = parser.parse_args()
    try:
        run_benchmark(
            task_types=args.task_types,
            tools=args.tools,
            datasets=args.datasets,
            max_workers=args.workers,
            config_path=args.config,
            tuning_mode=args.tuning_mode,
            tuning_dir=args.tuning_dir,
            include_user_tools=args.include_user_tools,
        )
    except ProviderKeyMissing as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
