"""The handle a task runner is given: what to read, where to write, and what to record.

Keeping figure/table registration on one object is what makes contract non-negotiable #1 -- "every
path in the manifest is relative to the results dir" -- structurally true rather than a rule each
task runner has to remember. A runner cannot register a figure it did not save through here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from . import plots
from .manifest import check_artifact_filename

if TYPE_CHECKING:
    from .manifest import Manifest


@dataclass
class AnalysisContext:
    manifest: Manifest
    files: list[Path]
    results_dir: Path
    tool_name: str | None = None
    dpi: int = plots.DEFAULT_DPI
    #: The rest of the directory: files ``collect_files`` dropped for their suffix or for being
    #: empty. Not analysis input -- no runner may read it -- but the inventory surfaces have to,
    #: because ``files`` alone is the candidate list and they describe it as the directory.
    #: Keeping it here rather than passing it twice is what stops the scan's count and the shallow
    #: runner's bar chart from answering "how many files did the tool write" differently.
    not_analysed: list[Path] = field(default_factory=list)
    _analysed: list[str] = field(default_factory=list)

    # ---- recording ------------------------------------------------------------------

    def analysed(self, path: Path | str) -> None:
        """Record a file as an input to this analysis; it lands in ``source_outputs``."""
        text = str(Path(path).resolve())
        if text not in self._analysed:
            self._analysed.append(text)
            self.manifest.source_outputs.append(text)

    def figure(self, fig, filename: str, *, title: str, caption: str, kind: str) -> None:
        """Save and register one figure. A figure that produced no file is never registered."""
        check_artifact_filename(filename)
        relative = plots.save_figure(fig, self.results_dir, filename, dpi=self.dpi)
        written = self.results_dir / relative
        if not written.is_file() or written.stat().st_size == 0:
            self.manifest.warn(f"{filename}: matplotlib wrote no bytes; the figure was dropped")
            self.manifest.degrade()
            return
        self.manifest.add_figure(relative, title, caption, kind)

    def table(self, frame, filename: str, *, title: str, index: bool = True) -> None:
        """Write a DataFrame under ``tables/`` and register it with its row count.

        The name is checked first. ``add_table`` refuses one that escapes the results dir, but it
        only ran *after* ``to_csv`` had already created the file wherever the name pointed.
        """
        check_artifact_filename(filename)
        relative = f"tables/{filename}"
        tables = self.results_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        frame.to_csv(tables / filename, index=index)
        self.manifest.add_table(relative, title, int(len(frame)))

    def warn(self, message: str) -> None:
        self.manifest.warn(message)

    def find(self, key: str, value, label: str) -> None:
        self.manifest.add_finding(key, value, label)
