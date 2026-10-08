"""Registry of available benchmark datasets.

Manages dataset metadata, discovery, and validation for benchmark runs.
Datasets are organized by task type (spatial_clustering, svg_detection, deconvolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetEntry:
    """Metadata for a single benchmark dataset.

    Attributes:
        name: Unique dataset identifier (e.g., 'visium_lymph_node').
        task_type: Benchmark task category (spatial_clustering, svg_detection, deconvolution).
        input_path: Path to input file (relative to data_dir).
        ground_truth_path: Path to ground truth file, if available.
        description: Human-readable description.
        platform: Data platform (Visium, Xenium, MERFISH, etc.).
        species: Species (human, mouse, etc.).
        n_spots: Number of spots/cells, if known.
        n_genes: Number of genes, if known.
        compatible_tools: List of tool names known to work with this dataset.
        metadata: Additional metadata.
    """

    name: str
    task_type: str
    input_path: str
    ground_truth_path: str | None = None
    description: str = ""
    platform: str = ""
    species: str = ""
    n_spots: int | None = None
    n_genes: int | None = None
    compatible_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolve_input(self, data_dir: Path) -> Path:
        """Resolve input_path relative to data_dir."""
        return data_dir / self.input_path

    def resolve_ground_truth(self, data_dir: Path) -> Path | None:
        """Resolve ground_truth_path relative to data_dir."""
        if self.ground_truth_path:
            return data_dir / self.ground_truth_path
        return None

    def has_ground_truth(self) -> bool:
        return self.ground_truth_path is not None


class DataRegistry:
    """Registry of benchmark datasets with lookup by task type, platform, and tool compatibility.

    Datasets can be loaded from a YAML registry file or registered programmatically.
    """

    def __init__(self) -> None:
        self._entries: dict[str, DatasetEntry] = {}

    def register(self, entry: DatasetEntry) -> None:
        """Register a dataset entry."""
        self._entries[entry.name] = entry

    def get(self, name: str) -> DatasetEntry | None:
        """Get a dataset by name."""
        return self._entries.get(name)

    def list_all(self) -> list[DatasetEntry]:
        """List all registered datasets."""
        return list(self._entries.values())

    def filter_by_task(self, task_type: str) -> list[DatasetEntry]:
        """Get datasets for a specific task type."""
        return [e for e in self._entries.values() if e.task_type == task_type]

    def filter_by_platform(self, platform: str) -> list[DatasetEntry]:
        """Get datasets for a specific platform."""
        return [e for e in self._entries.values() if e.platform.lower() == platform.lower()]

    def filter_by_tool(self, tool_name: str) -> list[DatasetEntry]:
        """Get datasets compatible with a specific tool."""
        return [e for e in self._entries.values() if tool_name in e.compatible_tools]

    def inject_user_tools(self, install_log_path: str | Path) -> dict[str, list[str]]:
        """Inject user-created tools into compatible_tools lists by task_type matching.

        Reads install_log.json and for each active user tool, appends it to
        datasets whose task_type matches the tool's task_type.

        Only modifies in-memory objects — never writes to registry.yaml.

        Returns:
            Dict mapping dataset_name -> list of user tool names that were injected.
            Empty dict if no user tools found or on any error.
        """
        import json

        injected: dict[str, list[str]] = {}
        try:
            log_path = Path(install_log_path)
            if not log_path.exists():
                return injected
            entries = json.loads(log_path.read_text())
            if not isinstance(entries, list):
                return injected
            active_tools = [
                e for e in entries if isinstance(e, dict) and e.get("status") == "active" and e.get("function_name")
            ]
            if not active_tools:
                return injected
            for dataset in self._entries.values():
                for tool_entry in active_tools:
                    if tool_entry.get("task_type") == dataset.task_type:
                        fn_name = tool_entry["function_name"]
                        if fn_name not in dataset.compatible_tools:
                            dataset.compatible_tools.append(fn_name)
                            injected.setdefault(dataset.name, []).append(fn_name)
        except Exception:
            pass  # Non-fatal: return whatever was injected so far
        return injected

    def get_task_types(self) -> list[str]:
        """List unique task types across all datasets."""
        return sorted({e.task_type for e in self._entries.values()})

    @classmethod
    def from_yaml(cls, path: str | Path) -> DataRegistry:
        """Load registry from a YAML file.

        Expected YAML format (paths relative to the registry's directory; see
        benchmarks/data/README.md for what the evaluator reads as ground truth):
            datasets:
              - name: visium_dlpfc_domain
                task_type: spatial_clustering
                input_path: Visium_for_spatial_domain/Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad
                ground_truth_path: Visium_for_spatial_domain/Spatial_data/Standard_h5ad/spatial_transcriptomics.h5ad
                platform: Visium
                species: human
                metadata:
                  ground_truth_key: layer_guess_reordered_short   # an obs column of that h5ad

        Clustering and deconvolution ground truth is an obs column of an h5ad, never a
        ``ground_truth.csv`` of labels or proportions -- the layout this docstring used to show,
        which the evaluator cannot read (hunt 2026-09-30, u33b-bench-scoring-19). SVG ground
        truth is a one-column gene list (.csv/.tsv/.txt).
        """
        registry = cls()
        with open(path) as f:
            raw = yaml.safe_load(f)
        for item in raw.get("datasets", []):
            entry = DatasetEntry(
                name=item["name"],
                task_type=item["task_type"],
                input_path=item["input_path"],
                ground_truth_path=item.get("ground_truth_path"),
                description=item.get("description", ""),
                platform=item.get("platform", ""),
                species=item.get("species", ""),
                n_spots=item.get("n_spots"),
                n_genes=item.get("n_genes"),
                compatible_tools=item.get("compatible_tools", []),
                metadata=item.get("metadata", {}),
            )
            registry.register(entry)
        return registry

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry for reporting."""
        return {
            "total_datasets": len(self._entries),
            "task_types": self.get_task_types(),
            "datasets": [
                {
                    "name": e.name,
                    "task_type": e.task_type,
                    "input_path": e.input_path,
                    "has_ground_truth": e.has_ground_truth(),
                    "platform": e.platform,
                }
                for e in self._entries.values()
            ],
        }
