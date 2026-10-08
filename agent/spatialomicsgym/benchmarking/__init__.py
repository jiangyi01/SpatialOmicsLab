"""SpatialOmicsLab Benchmarking Skill — output-aware evaluation pipeline.

Hot-pluggable, OFF by default. When enabled:
  1. Output inspection is MANDATORY before evaluation
  2. Tool-specific output rules identify the correct prediction files
  3. Evaluation only runs when explicitly enabled
"""

from spatialomicsgym.benchmarking.output_inspector import OutputInspectionResult, inspect_tool_output
from spatialomicsgym.benchmarking.output_standardizer import standardize_output

__all__ = ["inspect_tool_output", "OutputInspectionResult", "standardize_output"]
