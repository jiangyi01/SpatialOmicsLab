"""Capture and persist all artifacts from a SpatialOmicsLab-driven MCP smoke test run.

This module provides :class:`SmokeTestRecord` for structured test-run data and
:class:`ResultRecorder` for writing those records to disk in a reproducible
directory layout.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class SmokeTestRecord:
    """Immutable snapshot of a single smoke-test execution."""

    tool_name: str
    dataset_name: str
    mode: str  # "direct" or "full"

    prompt: str = ""
    spatialomicsgym_call_record: dict[str, Any] = field(default_factory=dict)
    mcp_invocation_record: dict[str, Any] = field(default_factory=dict)
    stdout_log: str = ""
    stderr_log: str = ""
    parsed_output: dict[str, Any] = field(default_factory=dict)

    # One of: PASS, FAIL_PROMPT, FAIL_MCP, FAIL_TOOL, FAIL_OUTPUT, TIMEOUT, DEGRADED (the turn
    # stopped early -- a 429 give-up, a spent step budget -- see benchmarks/turn_outcome.py)
    status: str = "pending"

    error_message: str = ""
    error_traceback: str = ""
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of the record."""
        return asdict(self)


class ResultRecorder:
    """Persist :class:`SmokeTestRecord` artifacts to a structured directory tree.

    Directory layout::

        {output_dir}/{tool_name}/{timestamp_YYYYMMDD_HHMMSS}/
            prompt.txt
            spatialomicsgym_call_record.json
            mcp_invocation_record.json
            stdout.log
            stderr.log
            parsed_output.json
            result.json
            error_report.txt          # only when status != "PASS"

    Parameters
    ----------
    output_dir : str | Path
        Root directory under which per-tool result trees are created.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def save(self, record: SmokeTestRecord) -> Path:
        """Write all artifacts for *record* and return the run directory.

        Parameters
        ----------
        record : SmokeTestRecord
            Completed test record to persist.

        Returns
        -------
        Path
            The directory containing the saved artifacts.
        """
        # Build a timestamp-based subdirectory from the record's ISO timestamp.
        try:
            dt = datetime.fromisoformat(record.timestamp)
        except (ValueError, TypeError):
            dt = datetime.now()
        ts_dir = dt.strftime("%Y%m%d_%H%M%S")

        run_dir = self.output_dir / record.tool_name / ts_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- plain-text artifacts ---
        (run_dir / "prompt.txt").write_text(record.prompt, encoding="utf-8")
        (run_dir / "stdout.log").write_text(record.stdout_log, encoding="utf-8")
        (run_dir / "stderr.log").write_text(record.stderr_log, encoding="utf-8")

        # --- JSON artifacts ---
        _write_json(run_dir / "spatialomicsgym_call_record.json", record.spatialomicsgym_call_record)
        _write_json(run_dir / "mcp_invocation_record.json", record.mcp_invocation_record)
        _write_json(run_dir / "parsed_output.json", record.parsed_output)
        _write_json(run_dir / "result.json", record.to_dict())

        # --- conditional error report ---
        if record.status != "PASS":
            error_text = textwrap.dedent(f"""\
                Status : {record.status}
                Tool   : {record.tool_name}
                Dataset: {record.dataset_name}
                Mode   : {record.mode}

                Error Message
                -------------
                {record.error_message}

                Traceback
                ---------
                {record.error_traceback}
            """)
            (run_dir / "error_report.txt").write_text(error_text, encoding="utf-8")

        return run_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Serialise *data* as pretty-printed JSON to *path*."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
