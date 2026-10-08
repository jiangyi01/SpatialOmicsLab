#!/usr/bin/env python3
"""Seed the agent's procedural memory with the 21-tool creation recipes (Fig. 2C).

Populates short-term memory (keyed by GitHub URL) with the winning install path,
dependency hints, primary function, and gotchas discovered during the runtime
tool-creation experiment, so that a *simple* prompt — "create <tool> from <github>" —
triggers memory-assisted creation: STCoscientist's Phase 1.0 calls
MemoryManager.read_short_term(url) / format_hint_for_prompt(url) and applies the recipe.

The store is HMAC-signed, so seeding goes through the MemoryManager API (not raw JSON).
voyager is recorded as a known hang (R/Bioconductor infinite compile on CPU hosts).

Recipes are the real per-tool outcomes from dev_logs/creation_deletion_benchmarking/
(iteration logs + debug narratives + state files). 20/21 tools were successfully
created; only voyager could not be registered on this platform.

Usage:
    SOG_MEMORY_ENABLED=true python -m tools_user.seed_tool_creation_memory
    # idempotent: deletes any existing short-term entry per URL before seeding one attempt.
"""
import os

os.environ.setdefault("SOG_MEMORY_ENABLED", "true")  # must be set before config import

from tools_user.memory_manager import (
    OUTCOME_FULL_PASS,
    OUTCOME_HUNG,
    OUTCOME_SOFT_DEGRADED,
    OUTCOME_VENDOR_FALLBACK,
    MemoryManager,
)

FULL, VENDOR, SOFT, HUNG = (
    OUTCOME_FULL_PASS, OUTCOME_VENDOR_FALLBACK, OUTCOME_SOFT_DEGRADED, OUTCOME_HUNG,
)

# tool_id, github, language, outcome, install_path, primary_fn, deps(dict), gotchas(list)
RECIPES = [
    ("banksy", "https://github.com/prabhakarlab/Banksy_py", "python", FULL,
     "pip_pypi", "", {"pybanksy": ""},
     ["install name is 'pybanksy' but the IMPORT module is 'banksy' (use `import banksy`, NOT pybanksy) -- verified_module=banksy  [corrected by live run 2026-06]",
      "banksy exposes no top-level banksy_clustering(); introspect the package for its real API instead of assuming a function name",
      "keep generated worker code ASCII-only -- non-ASCII chars from know-how leak and break exec"]),
    ("seagal", "https://github.com/linhuawang/SEAGAL", "python", FULL,
     "pip_pypi", "seagal_lindex", {"seagal": ""},
     ["pip install seagal; L-index gene colocalization; clean single-function API"]),
    ("spage", "https://github.com/tabdelaal/SpaGE", "python", VENDOR,
     "git_vendor", "SpaGE", {},
     ["no PyPI package — vendor the GitHub repo (module spage_vendor)",
      "reference-based imputation: requires a scRNA-seq reference alongside the spatial input"]),
    ("spavae", "https://github.com/ttgump/spaVAE", "python", VENDOR,
     "git_vendor", "spavae_run", {},
     ["vendor the repo (vendor_spavae/repo)",
      "registered MCP function name must match the worker (spavae_* naming); a mismatch causes 'Unknown tool'"]),
    ("spadecon", "https://github.com/kpcoleman/SpaDecon", "python", FULL,
     "pip_conda", "spadecon_deconvolution", {},
     ["clean conda+pip install; semi-supervised deconvolution; needs scRNA-seq reference"]),
    ("gaston", "https://github.com/raphael-group/GASTON", "python", FULL,
     "conda_yml_or_pip", "gaston_run", {"gaston-spatial": ""},
     ["install via conda env-from-yml or pip gaston-spatial (PyTorch, CPU OK)",
      "ensure the worker WRITES output files — a status:ok run with 0 output files counts as a failure"]),
    ("spatopic", "https://github.com/compbioNJU/SpaTopic", "r", FULL,
     "remotes_github", "spatopic_run", {},
     ["R: remotes::install_github('compbioNJU/SpaTopic')",
      "WorkerOutput helper must implement add_warning()",
      "avoid R integer-literal '6L' leaking into generated Python/MCP code — ASCII/literal hygiene"]),
    ("semla", "https://github.com/ludvigla/semla", "r", FULL,
     "remotes_github", "semla_run", {},
     ["R env needs geospatial/image system libs: gdal, geos, proj, udunits2, imagemagick + r-sf, r-terra, r-magick",
      "for R-via-rpy2 tools, install_log 'module' must be the R package name (semla), NOT 'rpy2'"]),
    ("hdwgcna", "https://github.com/smorabit/hdWGCNA", "r", FULL,
     "conda_binary_then_github", "", {},
     ["INSTALL DEPS AS CONDA BINARIES FIRST to avoid source-compile failure: conda install -n <env> -c conda-forge -c bioconda r-seurat r-wgcna r-harmony r-igraph r-uwot bioconductor-genomicranges bioconductor-summarizedexperiment; THEN remotes::install_github('smorabit/hdWGCNA', upgrade='never')",
      "do NOT let BiocManager source-compile the heavy chain — prefer conda-forge/bioconda prebuilt binaries"]),
    ("spatialinfercnv", "https://github.com/aerickso/SpatialInferCNV", "r", FULL,
     "remotes_github", "spatialinfercnv_run", {},
     ["needs HDF5 system library + the infercnv dependency"]),
    ("nichecompass", "https://github.com/Lotfollahi-lab/nichecompass", "python", FULL,
     "pip_pypi", "nichecompass_run", {"nichecompass": ""},
     ["PyG-based: match torch_sparse/torch_scatter/pyg-lib wheels to the installed torch version + CPU/GLIBC via the PyG wheel index URL; else fall back to the pyg conda channel",
      "CPU OK for small slides (slow)"]),
    ("liana", "https://github.com/saezlab/liana-py", "python", SOFT,
     "pip_pypi", "liana_run", {"liana": ""},
     ["pip install liana (LIANA+)",
      "real-data invocation needs a groupby / cell-label argument — without it the worker returns status:error",
      "register ONCE as liana_run; do not create duplicate liana / liana_py tool ids"]),
    ("stereopy", "https://github.com/STOmics/Stereopy", "python", FULL,
     "pip_pypi_altname", "stereopy_run", {},
     ["primary PyPI name 'stereopy' has a wheel-build bug; install the alternate name 'stereo' instead (verified_module = stereo)"]),
    ("sopa", "https://github.com/gustaveroussy/sopa", "python", SOFT,
     "pip_pypi", "sopa_cluster", {"sopa": ""},
     ["input is SpatialData/zarr, NOT h5ad — needs an h5ad->SpatialData bridge for real data (real_data_testable: partial)",
      "Cellpose CPU mode works (slower)"]),
    ("giotto", "https://github.com/drieslab/Giotto", "r", FULL,
     "conda_binary_then_github", "", {},
     ["library(Giotto) fails to load without the FULL geospatial+image stack as conda BINARIES. Install ALL of these first: conda install -n <env> -c conda-forge gdal geos proj udunits2 cairo harfbuzz fribidi pango fontconfig imagemagick r-sf r-terra r-sp r-matrix r-rcpp r-rcpparmadillo r-reticulate r-igraph bioconductor-singlecellexperiment bioconductor-summarizedexperiment bioconductor-s4vectors",
      "THEN remotes::install_github('drieslab/Giotto', upgrade='never') and verify library(Giotto); do NOT source-compile gdal/geos/proj"]),
    ("novae", "https://github.com/MICS-Lab/novae", "python", FULL,
     "pip_pypi", "novae_run", {"novae": ""},
     ["foundation model; zero-shot inference runs on CPU; pretrained weights pulled from HuggingFace"]),
    ("voyager", "https://github.com/pachterlab/voyager", "r", FULL,
     "conda_binary", "", {},
     ["INSTALL THE BIOCONDA BINARY to avoid the infinite source compile: conda install -n <env> -c bioconda -c conda-forge bioconductor-voyager bioconductor-spatialfeatureexperiment r-sf r-sp; verify library(Voyager) — do NOT BiocManager::install from source (it loops 8+ h on CPU hosts)",
      "the SpatialFeatureExperiment S4 class comes from the bioconductor-spatialfeatureexperiment binary"]),
    ("ficture", "https://github.com/seqscope/ficture", "python", SOFT,
     "pip_pypi", "ficture_run", {"ficture": ""},
     ["CLI-only (no Python API) — wrap CLI calls via set_meta(cli_name, subcommand, extra_args)",
      "input is TSV not h5ad — needs an h5ad->TSV bridge for real data (real_data_testable: partial)"]),
    ("sctm", "https://github.com/JinmiaoChenLab/scTM", "python", FULL,
     "pip_pypi", "stamp_run", {"sctm": ""},
     ["STAMP ships inside the scTM package: pip install sctm; import module 'sctm'"]),
    ("spanorm", "https://github.com/bhuvad/SpaNorm", "r", SOFT,
     "biocmanager", "spanorm_run", {},
     ["Bioconductor install (BiocManager::install)",
      "R env needs font/graphics system libs: freetype, harfbuzz, fribidi, pango, cairo, fontconfig + network libs openssl, curl, libssh2, zlib, bzip2, xz, zstd"]),
    ("splisosm", "https://github.com/JiayuSuPKU/SPLISOSM", "python", FULL,
     "pip_pypi", "splisosm_run", {},
     ["new isoform_detection task type (extend TOOL_CATEGORY_MAP); HSIC kernel tests; fully CPU"]),
]


def main():
    mm = MemoryManager.get()
    seeded = passes = soft = hung = 0
    for tid, url, lang, outcome, install_path, primary_fn, deps, gotchas in RECIPES:
        mm.delete_short_term(url)  # idempotent: one clean seeded attempt per URL
        record = {
            "outcome": outcome,
            "tool_id": tid,
            "language": lang,
            "strategy": {"install_path": install_path, "key_deps_pinned": deps},
            "api_shape": {"primary_fn": primary_fn} if primary_fn else {},
            "gotchas": gotchas,
            "error_classes_seen": [],
            "seed_source": "fig2c_tool_creation_experiment",
        }
        mm.append_attempt(url, record)
        mm.update_long_term(outcome=outcome, language=lang, install_path=install_path,
                            domain="spatial_omics")
        if outcome == HUNG:
            mm.mark_known_hang(url, gotchas[0] if gotchas else "known hang")
            hung += 1
        elif outcome in (FULL, VENDOR):
            passes += 1
        else:
            soft += 1
        seeded += 1
    print(f"Seeded {seeded} tool recipes into {mm.root} "
          f"({passes} pass-recipe, {soft} soft/partial, {hung} known-hang).")
    print("Memory-assisted creation is now primed; run with SOG_MEMORY_ENABLED=true.")


if __name__ == "__main__":
    main()
