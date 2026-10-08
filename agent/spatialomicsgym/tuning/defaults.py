"""Official default registry with provenance tracking.

Loads official default hyperparameters from configs/official_defaults.yaml
and provides lookup by tool name. Distinguishes between official repo defaults,
SpatialOmicsLab config defaults, documentation defaults, and inferred defaults.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from spatialomicsgym.mcp_config_path import find_mcp_config
from spatialomicsgym.tuning.core import DefaultSource, ParameterValue

_CONFIGS_DIR = Path(__file__).parent / "configs"
_OFFICIAL_DEFAULTS_PATH = _CONFIGS_DIR / "official_defaults.yaml"
_SOG_DEFAULTS_CACHE: dict[str, dict[str, ParameterValue]] | None = None

# Name components that mean "this parameter says *where* something is", and filename extensions that
# mean the same when they end the name. Matched by underscore-separated component rather than by
# substring, because substring matching is wrong in both directions here: "path" appears inside
# ``pathway_sum`` (a bool), while ``counts_h5`` and ``signature_csv`` are locations containing
# neither "path" nor "dir".
_LOCATION_WORDS = frozenset({"path", "paths", "dir", "dirs", "directory", "file", "files", "folder", "prefix"})

#: A description that says the value is a place on disk, for a parameter whose NAME does not:
#: `gene_table` ("Path to a CSV/TSV whose rows are genes..."), `background` ("Path to a file with
#: one background gene symbol per line"), `collections` ("... or the path to a .gmt"). The same
#: oracle `test/test_an_output_location_is_never_tuned_as_a_hyperparameter.py` reads the config
#: with; mirrored here so the fence and the lens cannot disagree about what a location is.
_DESCRIPTION_SAYS_LOCATION = re.compile(r"\bpath to\b|\bprefix for\b.*\boutput\b|\boutput prefix\b", re.IGNORECASE)


def describes_a_location(description: object) -> bool:
    """Whether a parameter's own description says its value is a path or an output prefix."""
    return bool(isinstance(description, str) and _DESCRIPTION_SAYS_LOCATION.search(description))


# Trailing components that mean the same when they end the name: mostly filename extensions, plus
# "image", which is how ``hires_image`` names a .png on disk. Matched on the last component only,
# which is what keeps ``image_alpha`` (a blend factor) and ``image_mpp`` (a resolution) tunable --
# they are about an image without being one.
_LOCATION_SUFFIXES = frozenset(
    {
        "h5",
        "h5ad",
        "csv",
        "tsv",
        "txt",
        "json",
        "yaml",
        "yml",
        "rds",
        "gz",
        "loom",
        "zarr",
        "mtx",
        "npz",
        "npy",
        "image",
    }
)

# Component matching cannot see a location word that was run together with its role, so ``outprefix``
# escaped a vocabulary that already contains "prefix". This is the closed set of those compounds and
# deliberately not an ``endswith`` test: "profile" ends in "file" and "nadir" ends in "dir", and
# neither names a place.
_ROLE_PREFIXES = ("out", "output", "in", "input", "save", "res", "result", "tmp", "temp", "work")

# The words above name a whole place. These name a *piece* of one, which gets interpolated into it:
# spaotsc's ``out_tag`` is "Short tag appended to /workspace/work/spaotsc_<tag>" and moscot's derives
# its ``output_dir``, so pinning either makes two runs of one tool write to one directory -- but no
# set above holds a word for a fragment. Measured against the shipped config before widening: of the
# 415 distinct parameter names it declares, exactly three follow a role prefix with a word the sets
# above miss -- "tag", "mode" and "key" -- and only "tag" can be adopted. "mode" is itself one of the
# 63 knobs the search spaces declare, and "key" names an obs/obsm column on 20 parameters; classing
# either as a location would delete a real hyperparameter. Gated on the role prefix below, so
# ``cluster_tag`` keeps its meaning.
_FRAGMENT_WORDS = frozenset({"tag"})

_RUN_TOGETHER_LOCATIONS = frozenset(
    role + word for role in _ROLE_PREFIXES for word in _LOCATION_WORDS | _FRAGMENT_WORDS
)


def is_io_param(name: str) -> bool:
    """Whether *name* denotes an input/output location rather than a hyperparameter.

    A location is a property of the *invocation* -- which slide, which output directory -- not of
    the configuration being tuned, so it has no business in a tuning baseline.
    ``integration.get_tunable_tools`` asks a similar-looking question when it counts what a tool can
    tune, but it is not the same question and not this code: it substring-matches its own, narrower
    vocabulary and never calls this function, so the two disagree in both directions. Do not read
    them as one rule.

    Four ways a name can say "a place on disk", because three locations in the shipped config got
    past a fence that only knew the first:

    * a location word as its own component;
    * a trailing component that stands in for one;
    * for a name with no components at all, one of the closed set of run-together role compounds;
    * a role prefix followed by a word for a fragment of a location, which is what ``out_tag`` is.

    Widening it costs nothing a trial could have varied: none of the 63 parameter names in
    ``configs/search_spaces.yaml`` is classed as a location.
    """
    parts = name.lower().split("_")
    if _LOCATION_WORDS.intersection(parts) or parts[-1] in _LOCATION_SUFFIXES:
        return True
    if len(parts) == 1:
        return parts[0] in _RUN_TOGETHER_LOCATIONS
    return parts[0] in _ROLE_PREFIXES and bool(_FRAGMENT_WORDS.intersection(parts[1:]))


def _load_official_defaults() -> dict[str, dict[str, ParameterValue]]:
    """Load official defaults from YAML config."""
    if not _OFFICIAL_DEFAULTS_PATH.exists():
        return {}

    with open(_OFFICIAL_DEFAULTS_PATH) as f:
        raw = yaml.safe_load(f) or {}

    registry: dict[str, dict[str, ParameterValue]] = {}
    for tool_name, params in raw.get("tools", {}).items():
        registry[tool_name] = {}
        for param_name, info in (params or {}).items():
            if not isinstance(info, dict):
                continue
            source_str = info.get("source", "inferred")
            try:
                source = DefaultSource(source_str)
            except ValueError:
                source = DefaultSource.INFERRED
            registry[tool_name][param_name] = ParameterValue(
                name=param_name,
                value=info.get("value"),
                source=source,
                source_file=info.get("source_file", ""),
                confidence=info.get("confidence", "medium"),
                version=info.get("version"),
                notes=info.get("notes"),
                adopted=info.get("adopted", True),
            )
    return registry


def get_official_defaults(tool_name: str) -> dict[str, ParameterValue]:
    """Get official default hyperparameters for a tool.

    Returns dict mapping parameter name to ParameterValue with provenance.

    I/O locations are excluded, for the same reason as in
    ``load_spatialomicsgym_defaults_from_mcp_config``: every caller overlays this on top of the
    mcp_config defaults to build a tuning baseline, and filtering only that first source would let
    this one put a location back -- ``run_spacel_scube.output_dir``,
    ``spacel_scube.spatial_h5ad_paths`` and ``stride_deconvolution.outprefix`` are the three the
    registry declares. The last of those was invisible to the fence until it learned to read a
    run-together name, so this enumeration counted two for as long as one of them was escaping.

    Rows flagged ``adopted: false`` are excluded on the same grounds. The registry records what
    upstream's code uses, and a large minority of rows were written during a divergence audit -- they
    document a value we looked at and deliberately did *not* take. Overlaying one reinstates exactly
    what its own notes say we rejected, so it is withheld and the mcp_config default stands.

    A row whose ``value:`` is ``null`` is withheld for a third reason: it has no default to serve.
    Such a row records that upstream leaves the parameter unset, which is a fact about upstream and
    not a value -- and overlaying it does not say "unset", it *blanks* whatever mcp_config declared.
    ``build_tool_command`` then drops the flag, so the worker never sees it. 26 rows are null-valued
    and 20 of them pass the two filters above; the two that were reachable through a registered
    tool's declared parameters, ``gpsa_align_slices.n_latent_gps`` and
    ``seurat_spatial_feature_plot.ncol``, each blanked a real portal default of 3 and 1. That was
    invisible for as long as mcp_config declared ``null`` there too, so the overlay was writing None
    over None; correcting those two config rows to mirror their portal is what exposed it.
    """
    global _SOG_DEFAULTS_CACHE
    if _SOG_DEFAULTS_CACHE is None:
        _SOG_DEFAULTS_CACHE = _load_official_defaults()
    return {
        name: pv
        for name, pv in _SOG_DEFAULTS_CACHE.get(tool_name, {}).items()
        if pv.adopted and pv.value is not None and not is_io_param(name)
    }


def load_spatialomicsgym_defaults_from_mcp_config(
    mcp_config_path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract current SpatialOmicsLab defaults from mcp_config.yaml.

    Returns dict mapping tool_name -> {param_name: default_value}.
    """
    if mcp_config_path is None:
        # Not a package-relative path: MCP_server/ is not in the wheel, so on a non-editable
        # install that path does not exist and every SpatialOmicsLab default silently vanished.
        resolved = find_mcp_config()
        if resolved is None:
            return {}
        mcp_config_path = str(resolved)

    path = Path(mcp_config_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    defaults: dict[str, dict[str, Any]] = {}
    for _server_key, server_cfg in config.get("mcp_servers", {}).items():
        for tool in server_cfg.get("tools", []):
            tool_name = tool.get("spatialomicsgym_name", "")
            if not tool_name:
                continue
            params: dict[str, Any] = {}
            for param_name, param_cfg in tool.get("parameters", {}).items():
                # I/O locations are excluded: this dict is the tuning baseline, and every search
                # candidate is built as ``baseline.copy()`` updated with the searched values, so a
                # ``data_path`` here rides through the winning trial into ``best_config.yaml`` and
                # then over the caller's own dataset in ``integration.inject_tuned_params``. None of
                # the 63 parameter names in ``configs/search_spaces.yaml`` is classed as a location,
                # so no knob a trial could have varied is removed.
                if (
                    isinstance(param_cfg, dict)
                    and "default" in param_cfg
                    and not is_io_param(param_name)
                    and not describes_a_location(param_cfg.get("description"))
                ):
                    params[param_name] = param_cfg["default"]
            if params:
                defaults[tool_name] = params
    return defaults
