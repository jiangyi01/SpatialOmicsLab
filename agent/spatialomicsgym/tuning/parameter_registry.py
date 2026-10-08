"""Tool parameter registry - defines tunable parameters per tool.

Loads from configs/search_spaces.yaml and mcp_config.yaml to build
a complete inventory of what can be tuned for each tool.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from spatialomicsgym.tuning.core import ParameterSpec, ParameterType

_CONFIGS_DIR = Path(__file__).parent / "configs"
_SEARCH_SPACES_PATH = _CONFIGS_DIR / "search_spaces.yaml"
_REGISTRY_CACHE: dict[str, list[ParameterSpec]] | None = None


def _parse_param_type(type_str: str) -> ParameterType:
    """Convert YAML type string to ParameterType enum."""
    mapping = {
        "int": ParameterType.INT,
        "float": ParameterType.FLOAT,
        "str": ParameterType.STR,
        "bool": ParameterType.BOOL,
        "categorical": ParameterType.CATEGORICAL,
    }
    return mapping.get(type_str.lower(), ParameterType.STR)


def _load_search_spaces() -> dict[str, list[ParameterSpec]]:
    """Load parameter definitions from search_spaces.yaml."""
    if not _SEARCH_SPACES_PATH.exists():
        return {}

    with open(_SEARCH_SPACES_PATH) as f:
        raw = yaml.safe_load(f) or {}

    registry: dict[str, list[ParameterSpec]] = {}
    for tool_name, params in raw.get("tools", {}).items():
        specs = []
        for param_info in params.get("parameters", []):
            spec = ParameterSpec(
                name=param_info["name"],
                param_type=_parse_param_type(param_info.get("type", "float")),
                default=param_info.get("default"),
                description=param_info.get("description", ""),
                min_value=param_info.get("min"),
                max_value=param_info.get("max"),
                choices=param_info.get("choices"),
                step=param_info.get("step"),
                tunable=param_info.get("tunable", True),
                risk_level=param_info.get("risk_level", "safe"),
                dependencies=param_info.get("dependencies"),
                output_cap=bool(param_info.get("output_cap", False)),
                inert_when_fixed=list(param_info.get("inert_when_fixed") or []) or None,
            )
            specs.append(spec)
        registry[tool_name] = specs
    return registry


def get_tunable_params(tool_name: str) -> list[ParameterSpec]:
    """Get tunable parameter specs for a tool."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _load_search_spaces()
    return [p for p in _REGISTRY_CACHE.get(tool_name, []) if p.tunable]


def get_all_params(tool_name: str) -> list[ParameterSpec]:
    """Get all parameter specs for a tool (including non-tunable)."""
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = _load_search_spaces()
    return _REGISTRY_CACHE.get(tool_name, [])


def get_light_params(tool_name: str) -> list[ParameterSpec]:
    """Get only the most impactful parameters for light tuning.

    Returns the top 2-3 parameters marked as high-impact.
    """
    params = get_tunable_params(tool_name)
    # Parameters with risk_level="safe" and marked tunable are eligible
    # Sort by impact: numeric params with wider ranges first
    safe_params = [p for p in params if p.risk_level == "safe"]
    return safe_params[:3]
